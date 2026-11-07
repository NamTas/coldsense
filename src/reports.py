"""
Step 7: the deliverables an operations / quality team actually uses.

  excursion_events.csv   every period a unit was outside its safe range (start, end, duration, peak, severity)
  unit_compliance.csv    one row per unit: % time in range, mean kinetic temperature (MKT), grade, door openings ...
  fleet_insights.json    plain-English findings about the whole fleet (+ small CSV tables behind them)
  coldsense_fleet_report.html   a self-contained report you can e-mail or open in a browser

MKT (mean kinetic temperature) is the standard pharma cold-chain number that summarises a temperature
history in one value, giving extra weight to warm periods.
"""
import html
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import config

MKT_DH_OVER_R = 10000.0   # kelvin (activation energy 83.144 kJ/mol divided by gas constant)


def mean_kinetic_temperature(temps_c) -> float:
    t = np.asarray(temps_c, dtype=float)
    t = t[~np.isnan(t)]
    if len(t) == 0:
        return float("nan")
    tk = t + 273.15
    mkt_k = MKT_DH_OVER_R / -np.log(np.mean(np.exp(-MKT_DH_OVER_R / tk)))
    return float(mkt_k - 273.15)


def _out_of_range(p: pd.Series, cfg) -> pd.Series:
    out = p > cfg.hi
    if cfg.lo is not None:
        out = out | (p < cfg.lo)
    return (out & p.notna())


# --------------------------------------------------------------------------
def find_excursions(clean: pd.DataFrame, primary: str, cfg) -> pd.DataFrame:
    """List every excursion event (>= MIN_EXCURSION_MIN minutes outside the safe range)."""
    min_steps = max(1, config.MIN_EXCURSION_MIN // config.SAMPLE_MIN)
    rows = []
    for fid, g in clean.groupby("fridge_id", sort=True):
        g = g.sort_values("timestamp").reset_index(drop=True)
        p = g[primary]
        out = _out_of_range(p, cfg)
        run_id = (out != out.shift()).cumsum()
        for _, seg in g.groupby(run_id):
            idx = seg.index
            if not bool(out.loc[idx[0]]) or len(idx) < min_steps:
                continue
            ps = p.loc[idx]
            over_hi = (ps - cfg.hi).clip(lower=0)
            under_lo = (cfg.lo - ps).clip(lower=0) if cfg.lo is not None else pd.Series(0.0, index=idx)
            dev = pd.concat([over_hi, under_lo], axis=1).max(axis=1)
            too_warm = over_hi.max() >= under_lo.max()
            minutes = len(idx) * config.SAMPLE_MIN
            rows.append({
                "fridge_id": fid,
                "fridge_type": g["fridge_type"].iloc[0],
                "start": g.loc[idx[0], "timestamp"],
                "end": g.loc[idx[-1], "timestamp"] + pd.Timedelta(minutes=config.SAMPLE_MIN),
                "start_step": int(g.loc[idx[0], "step"]),
                "end_step": int(g.loc[idx[-1], "step"]) + 1,
                "duration_min": minutes,
                "direction": "too warm" if too_warm else "too cold",
                "peak_temp_c": round(float(ps.max() if too_warm else ps.min()), 1),
                "max_beyond_limit_c": round(float(dev.max()), 1),
                "degree_minutes": round(float(dev.sum() * config.SAMPLE_MIN), 0),
                "severity": "brief" if minutes < 30 else ("moderate" if minutes < 120 else "prolonged"),
            })
    cols = ["fridge_id", "fridge_type", "start", "end", "start_step", "end_step", "duration_min", "direction",
            "peak_temp_c", "max_beyond_limit_c", "degree_minutes", "severity"]
    return pd.DataFrame(rows, columns=cols)


def grade_for(pct_in_range: float, prolonged: int) -> str:
    if pct_in_range >= 98 and prolonged == 0:
        return "A"
    if pct_in_range >= 90:
        return "B"
    if pct_in_range >= 70:
        return "C"
    return "D"


def unit_compliance(clean: pd.DataFrame, primary: str, events: pd.DataFrame, scored: pd.DataFrame, cfg) -> pd.DataFrame:
    rows = []
    for fid, g in clean.groupby("fridge_id", sort=True):
        g = g.sort_values("timestamp")
        p = g[primary]
        valid = int(p.notna().sum())
        out_steps = int(_out_of_range(p, cfg).sum())
        ev = events[events["fridge_id"] == fid]
        hours = len(g) * config.SAMPLE_MIN / 60
        door_opens = np.nan
        if "temp_door" in g and primary != "temp_door":
            door_opens = float((g["temp_door"].diff() > config.DOOR_OPEN_JUMP_C).sum() / max(hours / 24, 1e-9))
        pct_in = 100 * (1 - out_steps / valid) if valid else np.nan
        prolonged = int((ev["severity"] == "prolonged").sum())
        sc = scored[scored["fridge_id"] == fid]
        rows.append({
            "fridge_id": fid,
            "fridge_type": g["fridge_type"].iloc[0],
            "hours_recorded": round(hours, 1),
            "data_completeness_pct": round(100 * valid / max(len(g), 1), 1),
            "pct_time_in_range": round(pct_in, 1),
            "minutes_out_of_range": out_steps * config.SAMPLE_MIN,
            "mean_temp_c": round(float(p.mean()), 2),
            "mkt_c": round(mean_kinetic_temperature(p.values), 2),
            "max_temp_c": round(float(p.max()), 1),
            "min_temp_c": round(float(p.min()), 1),
            "n_excursions": int(len(ev)),
            "longest_excursion_min": int(ev["duration_min"].max()) if len(ev) else 0,
            "door_openings_per_day": round(door_opens, 1) if not np.isnan(door_opens) else np.nan,
            "ambient_mean_c": round(float(g["temp_ambient"].mean()), 1) if "temp_ambient" in g else np.nan,
            "sensor_alert_pct": round(100 * float(sc["anomaly_flag"].fillna(False).mean()), 2) if len(sc) else np.nan,
            "grade": grade_for(pct_in, prolonged),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def _spearman(a: pd.Series, b: pd.Series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return a.corr(b, method="spearman")


def fleet_insights(clean, comp, events, primary, cfg):
    """Plain-English findings + the tables behind them."""
    out = {"sentences": [], "tables": {}}
    n = len(comp)
    with_exc = int((comp["n_excursions"] > 0).sum())
    pct_units = 100 * with_exc / max(n, 1)
    lo_txt = f"{cfg.lo:g}-" if cfg.lo is not None else "<="
    out["sentences"].append(
        f"{with_exc} of {n} units ({pct_units:.0f}%) had at least one excursion outside {lo_txt}{cfg.hi:g} C "
        f"during the recording; the median unit spent {comp['pct_time_in_range'].median():.0f}% of the time in range."
    )
    if len(events):
        long_share = 100 * (events["severity"] == "prolonged").mean()
        out["sentences"].append(
            f"{len(events)} excursion events were found; {long_share:.0f}% lasted two hours or more "
            f"(median {events['duration_min'].median():.0f} min, longest {events['duration_min'].max():.0f} min)."
        )

    # which sensor position runs warmest
    pos_rows = []
    for c in [c for c in clean.columns if c.startswith("temp_")]:
        s = clean[c].dropna()
        if len(s):
            pos_rows.append({"position": c.replace("temp_", ""), "mean_c": round(s.mean(), 2),
                             "p95_c": round(s.quantile(0.95), 2), "pct_above_upper_limit": round(100 * (s > cfg.hi).mean(), 1)})
    pos = pd.DataFrame(pos_rows)
    out["tables"]["position_summary"] = pos
    air = pos[pos["position"].isin(["top", "middle", "veg", "door", "air", "x"])]
    if len(air) >= 2:
        warm, cold = air.sort_values("mean_c").iloc[-1], air.sort_values("mean_c").iloc[0]
        out["sentences"].append(
            f"The {warm['position']} position runs warmest (mean {warm['mean_c']:.1f} C), "
            f"{warm['mean_c'] - cold['mean_c']:.1f} C above the {cold['position']} position."
        )

    # time of day
    tmp = clean.copy()
    tmp["out"] = _out_of_range(tmp[primary], cfg)
    tmp["hour"] = tmp["timestamp"].dt.hour
    hourly = tmp.groupby("hour")["out"].mean().mul(100).round(2).reset_index(name="pct_out_of_range")
    out["tables"]["hourly_profile"] = hourly
    if hourly["pct_out_of_range"].max() > 0:
        roll = hourly["pct_out_of_range"].rolling(3, center=True, min_periods=1).mean()
        peak = int(hourly.loc[roll.idxmax(), "hour"])
        out["sentences"].append(
            f"Out-of-range time peaks around {peak:02d}:00 ({hourly['pct_out_of_range'].max():.1f}% of unit-time at the "
            f"worst hour vs {hourly['pct_out_of_range'].mean():.1f}% on average)."
        )

    # unit type
    types = comp.groupby("fridge_type").agg(units=("fridge_id", "count"),
                                            pct_in_range=("pct_time_in_range", "mean")).round(1).reset_index()
    out["tables"]["type_summary"] = types
    big = types[types["units"] >= 3]
    if len(big) >= 2:
        worst = big.sort_values("pct_in_range").iloc[0]
        best = big.sort_values("pct_in_range").iloc[-1]
        out["sentences"].append(
            f"Unit type '{worst['fridge_type']}' spent the least time in range ({worst['pct_in_range']:.0f}% on average) "
            f"vs '{best['fridge_type']}' ({best['pct_in_range']:.0f}%)."
        )

    # warm rooms
    if n >= 8 and comp["ambient_mean_c"].notna().sum() >= 8:
        rho = _spearman(comp["ambient_mean_c"], 100 - comp["pct_time_in_range"])
        if pd.notna(rho):
            if abs(rho) >= 0.3:
                way = "more" if rho > 0 else "less"
                out["sentences"].append(f"Units in warmer rooms spent {way} time out of range (Spearman rho = {rho:.2f}).")
            else:
                out["sentences"].append(f"No clear link between room temperature and time out of range (Spearman rho = {rho:.2f}).")

    # door openings
    if comp["door_openings_per_day"].notna().sum() >= 8:
        rho = _spearman(comp["door_openings_per_day"], 100 - comp["pct_time_in_range"])
        if pd.notna(rho) and abs(rho) >= 0.3:
            way = "more" if rho > 0 else "less"
            out["sentences"].append(f"Units with more door openings spent {way} time out of range (Spearman rho = {rho:.2f}).")

    worst5 = comp.sort_values("pct_time_in_range").head(5)["fridge_id"].tolist()
    out["worst_units"] = worst5
    return out


# --------------------------------------------------------------------------
# standalone HTML report
# --------------------------------------------------------------------------
def _bars_svg(labels, values, width=560, color="#0F766E", unit=""):
    if len(values) == 0:
        return ""
    vmax = max(max(values), 1e-9)
    row_h = 24
    h = row_h * len(values) + 8
    parts = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img">']
    for i, (lab, v) in enumerate(zip(labels, values)):
        y = 4 + i * row_h
        w = max(1, (width - 230) * v / vmax)
        parts.append(f'<text x="0" y="{y + 15}" font-size="12" fill="#3b4148">{html.escape(str(lab))[:26]}</text>')
        parts.append(f'<rect x="150" y="{y + 3}" width="{w:.1f}" height="14" rx="3" fill="{color}"/>')
        parts.append(f'<text x="{150 + w + 6:.1f}" y="{y + 15}" font-size="12" fill="#3b4148">{v:g}{unit}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_html_report(path: Path, summary: dict, comp: pd.DataFrame, events: pd.DataFrame, insights: dict, cfg):
    esc = html.escape
    lo_txt = f"{cfg.lo:g} to " if cfg.lo is not None else "up to "
    demo = '<div class="demo">DEMO: synthetic data used to test the software - not real measurements.</div>' if summary.get("demo_data") else ""
    kpis = [
        ("Units analysed", f"{len(comp)}"),
        ("Units with an excursion", f"{int((comp['n_excursions'] > 0).sum())}"),
        ("Excursion events", f"{len(events)}"),
        ("Median time in range", f"{comp['pct_time_in_range'].median():.0f}%"),
    ]
    kpi_html = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{esc(k)}</div></div>' for k, v in kpis)
    sent_html = "".join(f"<li>{esc(s)}</li>" for s in insights["sentences"])

    worst = comp.sort_values(["pct_time_in_range", "n_excursions"], ascending=[True, False]).head(10)
    rows = "".join(
        f"<tr><td>{esc(str(r.fridge_id))}</td><td>{esc(str(r.fridge_type))}</td><td class='g g{r.grade}'>{r.grade}</td>"
        f"<td>{r.pct_time_in_range:.1f}%</td><td>{r.n_excursions}</td><td>{r.longest_excursion_min} min</td>"
        f"<td>{r.mean_temp_c:.1f}</td><td>{r.mkt_c:.1f}</td></tr>"
        for r in worst.itertuples()
    )
    hourly = insights["tables"]["hourly_profile"]
    hour_svg = _bars_svg([f"{int(h):02d}:00" for h in hourly["hour"]], hourly["pct_out_of_range"].tolist(), unit="%", color="#C2410C")
    pos = insights["tables"]["position_summary"]
    pos_svg = _bars_svg(pos["position"].tolist(), pos["mean_c"].tolist(), unit=" C")
    sev = events["severity"].value_counts().reindex(["brief", "moderate", "prolonged"]).fillna(0).astype(int) if len(events) else pd.Series(dtype=int)
    sev_svg = _bars_svg(sev.index.tolist(), sev.values.tolist(), color="#B7791F") if len(sev) else "<p>No excursion events.</p>"

    best = next((m for m in summary.get("risk_models", []) if m["model"] == summary.get("final_risk_model")), None)
    model_line = ""
    if best:
        model_line = (f"<p>Early-warning model ({esc(summary['final_risk_model'])}): PR-AUC {best['pr_auc']:.2f} on units never seen in training "
                      f"(excursion rate in test data {100 * best['prevalence']:.1f}%).</p>")
    ev = summary.get("anomaly_eval", {})
    anom_line = ""
    if ev:
        anom_line = (f"<p>Sensor-fault detector: caught {100 * ev['event_recall_overall']:.0f}% of {ev['events_injected']} injected test faults, "
                     f"false-alarm rate {100 * ev['false_alarm_rate_on_normal_rows']:.1f}% (simulated faults - no real fault labels exist).</p>")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>ColdSense fleet report</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body{{margin:0;background:#F6F3EE;color:#1B1F23;font:15px/1.55 'IBM Plex Sans',system-ui,Segoe UI,sans-serif}}
.wrap{{max-width:920px;margin:0 auto;padding:32px 20px 60px}}
h1{{font-size:28px;margin:0 0 4px}} h2{{font-size:18px;margin:34px 0 10px;border-bottom:2px solid #1B1F23;padding-bottom:4px}}
.sub{{color:#5b6470}} .demo{{background:#FEF3C7;border:1px solid #B7791F;padding:8px 12px;margin:14px 0;border-radius:6px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:18px 0}}
.kpi{{background:#fff;border:1px solid #E2DCD1;border-radius:10px;padding:14px}} .kpi .v{{font-size:28px;font-weight:700}} .kpi .l{{color:#5b6470;font-size:13px}}
table{{border-collapse:collapse;width:100%;background:#fff}} th,td{{text-align:left;padding:7px 9px;border-bottom:1px solid #E2DCD1;font-size:13.5px}} th{{background:#EDE7DD}}
.g{{font-weight:700;text-align:center}} .gA{{color:#0F766E}} .gB{{color:#4D7C0F}} .gC{{color:#B7791F}} .gD{{color:#C2410C}}
li{{margin:6px 0}} .card{{background:#fff;border:1px solid #E2DCD1;border-radius:10px;padding:14px 16px}}
.foot{{color:#5b6470;font-size:12.5px;margin-top:28px}}
</style></head><body><div class="wrap">
<h1>ColdSense fleet report</h1>
<div class="sub">Safe range: {lo_txt}{cfg.hi:g} C &middot; profile: {esc(cfg.profile)} &middot; excursion = at least {config.MIN_EXCURSION_MIN} min outside the range</div>
{demo}
<div class="kpis">{kpi_html}</div>
<h2>Key findings</h2><div class="card"><ul>{sent_html}</ul></div>
<h2>Units that need attention</h2>
<table><tr><th>Unit</th><th>Type</th><th>Grade</th><th>Time in range</th><th>Excursions</th><th>Longest</th><th>Mean C</th><th>MKT C</th></tr>{rows}</table>
<p class="sub">Grade: A = 98%+ in range and no excursion over 2 h, B = 90%+, C = 70%+, D = below 70%. MKT = mean kinetic temperature.</p>
<h2>When do excursions happen?</h2><div class="card">{hour_svg}</div>
<h2>Which sensor positions run warm?</h2><div class="card">{pos_svg}</div>
<h2>How long do excursions last?</h2><div class="card">{sev_svg}</div>
<h2>Method and limits</h2><div class="card">{model_line}{anom_line}
<p>Data: measurements from domestic refrigerators, not pharmaceutical shipments. Each unit was recorded for a few days. Excursions are
defined by the safety limit above; there are no labelled real alerts in the data.</p></div>
<div class="foot">Generated by ColdSense.</div></div></body></html>"""
    Path(path).write_text(doc, encoding="utf-8")


def save_all(out_dir: Path, clean, scored, primary, summary, cfg):
    """Create every report file in out_dir/reports and return the main tables."""
    rep = Path(out_dir) / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    events = find_excursions(clean, primary, cfg)
    comp = unit_compliance(clean, primary, events, scored, cfg)
    insights = fleet_insights(clean, comp, events, primary, cfg)
    events.to_csv(rep / "excursion_events.csv", index=False)
    comp.to_csv(rep / "unit_compliance.csv", index=False)
    for name, df in insights["tables"].items():
        df.to_csv(rep / f"{name}.csv", index=False)
    (rep / "fleet_insights.json").write_text(
        json.dumps({"sentences": insights["sentences"], "worst_units": insights["worst_units"]}, indent=2), encoding="utf-8")
    build_html_report(rep / "coldsense_fleet_report.html", summary, comp, events, insights, cfg)
    return events, comp, insights

"""
ColdSense control room.   Run from the project folder:   streamlit run app/streamlit_app.py

Tabs
  Unit wall      every unit as a live tile (status, temperature, 6-hour sparkline, why)
  Work list      what to inspect first + recommended action; download list / handover note / fleet report
  Unit report    compliance card, excursion ledger and chart for one unit
  Fleet insights findings from the data + "what if the limit were different?" explorer
  Trust          how well the models work, data quality, limitations
"""
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src import priority

st.set_page_config(page_title="ColdSense control room", page_icon="🧊", layout="wide")

INK, PAPER, MUTED, TEAL = "#1B1F23", "#F6F3EE", "#5B6470", "#0F766E"
STATUS = {  # key: (label, colour)
    "ALERT": ("Out of range", "#C2410C"),
    "WATCH": ("Watch", "#B7791F"),
    "SENSOR": ("Check sensor", "#6D28D9"),
    "OK": ("Stable", TEAL),
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame { font-family: 'IBM Plex Sans', system-ui, 'Segoe UI', sans-serif; }
.block-container { padding-top: 0.6rem; max-width: 1280px; }
#MainMenu, footer { visibility: hidden; }
.cs-head { background:#16212B; color:#F6F3EE; border-radius:14px; padding:16px 22px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:14px; }
.cs-brand { display:flex; align-items:center; gap:12px; }
.cs-name { font-size:22px; font-weight:700; letter-spacing:.2px; line-height:1.1; }
.cs-tag { font-size:12px; color:#9FB3C0; letter-spacing:.8px; text-transform:uppercase; }
.cs-meta { display:flex; gap:22px; flex-wrap:wrap; font-size:13px; color:#9FB3C0; }
.cs-meta b { color:#F6F3EE; font-family:'IBM Plex Mono', monospace; font-weight:600; font-size:14px; margin-left:4px; }
.demo-banner { background:#FEF3C7; border:1px solid #B7791F; color:#5b3a00; padding:8px 14px; border-radius:8px; font-size:13px; margin-bottom:12px; }
.state-bar { display:flex; height:30px; border-radius:8px; overflow:hidden; border:1px solid #E2DCD1; margin:4px 0 6px 0; }
.state-bar div { display:flex; align-items:center; justify-content:center; color:#fff; font-size:12px; font-weight:600; min-width:0; }
.state-legend { display:flex; gap:18px; flex-wrap:wrap; font-size:13px; color:#5B6470; margin-bottom:10px; }
.state-legend i { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; }
.wall { display:grid; grid-template-columns:repeat(auto-fill,minmax(215px,1fr)); gap:12px; margin-top:6px; }
.tile { background:#fff; border:1px solid #E2DCD1; border-left:6px solid var(--c); border-radius:10px; padding:10px 12px 9px 12px; }
.tile .top { display:flex; justify-content:space-between; align-items:center; gap:6px; }
.tile .uid { font-weight:600; font-size:13px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.tile .pill { background:var(--c); color:#fff; font-size:10.5px; font-weight:600; padding:1px 8px; border-radius:20px; white-space:nowrap; }
.tile .row { display:flex; justify-content:space-between; align-items:baseline; margin-top:2px; }
.tile .temp { font-family:'IBM Plex Mono', monospace; font-size:28px; font-weight:600; line-height:1.15; }
.tile .temp small { font-size:13px; color:#5B6470; margin-left:2px; }
.tile .risk { font-size:11.5px; color:#5B6470; }
.tile .why { font-size:11.5px; color:#3b4148; min-height:30px; line-height:1.3; margin-top:2px; }
.callout { background:#fff; border:1px solid #E2DCD1; border-left:5px solid #0F766E; border-radius:8px; padding:9px 14px; margin:7px 0; font-size:14.5px; }
.ucard { display:flex; gap:16px; align-items:stretch; flex-wrap:wrap; margin-bottom:6px; }
.grade { width:86px; border-radius:12px; display:flex; flex-direction:column; align-items:center; justify-content:center; color:#fff; font-family:'IBM Plex Mono', monospace; padding:8px 0; }
.grade .g { font-size:42px; font-weight:700; line-height:1; } .grade .l { font-size:10px; letter-spacing:.8px; margin-top:4px; }
.stats { flex:1; display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px; }
.stat { background:#fff; border:1px solid #E2DCD1; border-radius:10px; padding:9px 12px; }
.stat .v { font-family:'IBM Plex Mono', monospace; font-size:21px; font-weight:600; } .stat .k { font-size:12px; color:#5B6470; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_run(folder: str):
    d = Path(folder)
    scored = pd.read_csv(d / "scored.csv.gz", parse_dates=["timestamp"])
    scored["anomaly_flag"] = scored["anomaly_flag"].fillna(False).astype(bool)
    rep = d / "reports"
    out = {
        "scored": scored,
        "summary": json.loads((d / "run_summary.json").read_text(encoding="utf-8")),
        "dq": pd.read_csv(d / "data_quality_report.csv"),
        "model_cmp": pd.read_csv(d / "model_comparison.csv"),
        "events": pd.read_csv(rep / "excursion_events.csv", parse_dates=["start", "end"]),
        "unit_comp": pd.read_csv(rep / "unit_compliance.csv"),
        "insights": json.loads((rep / "fleet_insights.json").read_text(encoding="utf-8")),
        "hourly": pd.read_csv(rep / "hourly_profile.csv"),
        "position": pd.read_csv(rep / "position_summary.csv"),
        "report_html": (rep / "coldsense_fleet_report.html").read_bytes(),
    }
    imp = d / "feature_importance.csv"
    out["imp"] = pd.read_csv(imp) if imp.exists() else None
    out["series"] = {fid: g.sort_values("step")["p"].to_numpy() for fid, g in scored.groupby("fridge_id")}
    return out


runs = [p for p in (ROOT / "outputs", ROOT / "outputs_demo") if (p / "reports" / "coldsense_fleet_report.html").exists()]
if not runs:
    st.error("No results found yet. Run  `python scripts/02_run_pipeline.py`  first (see README, Step 6).")
    st.stop()

labels = {p.name: p for p in runs}
with st.sidebar:
    st.markdown("### Controls")
    choice = st.selectbox("Results folder", list(labels), help="'outputs_demo' = synthetic demo data")
D = load_run(str(labels[choice]))
scored, summary = D["scored"], D["summary"]
cfg = config.get_config(summary["profile"])

with st.sidebar:
    only_test = st.checkbox("Only units the model never saw in training", value=True,
                            help="Honest view: predictions for these units are genuine out-of-sample predictions.")
    view = scored[scored["split"] == "test"] if only_test else scored
    if view.empty:
        view = scored
    max_step = int(view.groupby("fridge_id")["step"].max().min())
    max_hours = round(max_step * config.SAMPLE_MIN / 60, 1)
    hours = st.slider("Replay time (hours since the loggers started)", 0.0, max_hours,
                      value=round(max_hours * 0.6, 1), step=0.5)
    step_now = int(hours * 60 / config.SAMPLE_MIN)
    show_n = st.selectbox("Units shown on the wall", ["Top 12", "Top 24", "All"], index=0)
    st.caption("Priority = 55% risk of leaving the range in 30 min + 25% time out of range in the last hour + 20% sensor alert.")

lo_txt = f"{cfg.lo:g} to " if cfg.lo is not None else "up to "
st.markdown(f"""
<div class="cs-head">
  <div class="cs-brand">
    <svg width="34" height="34" viewBox="0 0 34 34"><rect width="34" height="34" rx="9" fill="#0F766E"/>
      <polyline points="6,11 13,12 18,21 28,24" fill="none" stroke="#F6F3EE" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="28" cy="24" r="3.2" fill="#F6F3EE"/></svg>
    <div><div class="cs-name">ColdSense</div><div class="cs-tag">cold-storage control room</div></div>
  </div>
  <div class="cs-meta"><span>Safe range<b>{lo_txt}{cfg.hi:g} °C</b></span><span>Replay<b>{hours:g} h</b></span>
  <span>Warning horizon<b>{summary['horizon_min']} min</b></span><span>Profile<b>{cfg.profile}</b></span></div>
</div>
""", unsafe_allow_html=True)
if summary.get("demo_data"):
    st.markdown('<div class="demo-banner"><b>DEMO MODE</b> - synthetic data used only to test the software. These are not real measurements.</div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def style_chart(c):
    return (c.configure_view(strokeWidth=0).configure_axis(gridColor="#E7E1D8", domainColor="#B9B1A4", labelColor=MUTED,
                                                           titleColor=MUTED, labelFont="IBM Plex Sans", titleFont="IBM Plex Sans")
            .configure_title(color=INK, font="IBM Plex Sans", fontSize=13, anchor="start"))


def sparkline(values, hi, lo, color, w=190, h=40):
    v = np.asarray(values, dtype=float)
    if len(v) < 2 or np.isnan(v).all():
        return ""
    bounds = [np.nanmin(v), np.nanmax(v), hi] + ([lo] if lo is not None else [])
    vmin, vmax = min(bounds), max(bounds)
    span = max(vmax - vmin, 1e-6)

    def y(t):
        return h - 3 - (t - vmin) / span * (h - 6)

    xs = np.linspace(2, w - 2, len(v))
    pts = " ".join(f"{x:.1f},{y(t):.1f}" for x, t in zip(xs, v) if not np.isnan(t))
    lines = f'<line x1="0" x2="{w}" y1="{y(hi):.1f}" y2="{y(hi):.1f}" stroke="#C2410C" stroke-width="1" stroke-dasharray="4 3" opacity=".7"/>'
    if lo is not None:
        lines += f'<line x1="0" x2="{w}" y1="{y(lo):.1f}" y2="{y(lo):.1f}" stroke="#C2410C" stroke-width="1" stroke-dasharray="4 3" opacity=".7"/>'
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none">{lines}'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.8" stroke-linejoin="round"/></svg>')


def build_queue(snapshot: pd.DataFrame) -> pd.DataFrame:
    q = priority.priority_queue(snapshot, cfg, top_n=len(snapshot))
    recs = q.to_dict("records")
    q["status"] = [priority.status_of(r) for r in recs]
    q["action"] = [priority.action_for_row(r) for r in recs]
    return q


def handover_text(q: pd.DataFrame, counts: dict, n: int, top: int = 5) -> str:
    lines = [
        "SHIFT HANDOVER - ColdSense",
        f"Recording time: {hours:g} h after the loggers started   |   Safe range: {lo_txt}{cfg.hi:g} C",
        f"Fleet: {n} units | out of range: {counts['ALERT']} | watch: {counts['WATCH']} | "
        f"sensor checks: {counts['SENSOR']} | stable: {counts['OK']}",
        "",
        "Inspect first:",
    ]
    attention = q[q["status"] != "OK"]
    for r in attention.head(top).itertuples():
        lines.append(f"{r.rank}. {r.fridge_id} ({STATUS[r.status][0]}) now {r.p:.1f} C, risk {100 * r.risk:.0f}% - {r.reasons}")
        lines.append(f"   Action: {r.action}")
    if attention.empty:
        lines.append("Nothing needs attention right now.")
    if summary.get("demo_data"):
        lines += ["", "(DEMO data - not real measurements)"]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# current snapshot
# --------------------------------------------------------------------------
snap = view[view["step"] == step_now].dropna(subset=["p"]).copy()
queue = build_queue(snap) if len(snap) else pd.DataFrame()
counts = {k: int((queue["status"] == k).sum()) if len(queue) else 0 for k in STATUS}
n_units = len(snap)

tab_wall, tab_work, tab_unit, tab_fleet, tab_trust = st.tabs(
    ["Unit wall", "Work list", "Unit report", "Fleet insights", "Trust & method"])

# ============================ UNIT WALL ====================================
with tab_wall:
    if queue.empty:
        st.info("No readings at this moment - move the replay slider.")
    else:
        segs = "".join(f'<div style="width:{100 * counts[k] / n_units:.2f}%;background:{STATUS[k][1]}">{counts[k] if counts[k] else ""}</div>'
                       for k in STATUS if counts[k])
        legend = "".join(f'<span><i style="background:{STATUS[k][1]}"></i>{STATUS[k][0]} <b>{counts[k]}</b></span>' for k in STATUS)
        st.markdown(f'<div class="state-bar">{segs}</div><div class="state-legend">{legend}<span>of <b>{n_units}</b> units</span></div>',
                    unsafe_allow_html=True)

        fleet = view.groupby("step")["over_now"].sum().reset_index(name="units_out")
        fleet["hours"] = fleet["step"] * config.SAMPLE_MIN / 60
        area = alt.Chart(fleet).mark_area(color="#C2410C", opacity=0.35, line={"color": "#C2410C"}).encode(
            x=alt.X("hours:Q", title="Hours since start"), y=alt.Y("units_out:Q", title="Units out of range"))
        now = alt.Chart(pd.DataFrame({"x": [hours]})).mark_rule(color=INK, strokeWidth=2).encode(x="x:Q")
        st.altair_chart(style_chart((area + now).properties(height=95, title="How many units were outside the safe range over time")),
                        width="stretch")

        limit = {"Top 12": 12, "Top 24": 24, "All": len(queue)}[show_n]
        tiles = []
        for r in queue.head(limit).itertuples():
            label, color = STATUS[r.status]
            series = D["series"].get(r.fridge_id, np.array([]))
            recent = series[max(0, step_now - 72):step_now + 1]
            spark = sparkline(recent, cfg.hi, cfg.lo, color)
            sensor_note = " · sensor?" if r.anomaly_flag and r.status != "SENSOR" else ""
            why = r.reasons if r.reasons != "Normal" else "Within range, no warning signs"
            tiles.append(
                f'<div class="tile" style="--c:{color}"><div class="top"><span class="uid" title="{r.fridge_id}">{r.fridge_id}</span>'
                f'<span class="pill">{label}</span></div>'
                f'<div class="row"><span class="temp">{r.p:.1f}<small>°C</small></span>'
                f'<span class="risk">risk {100 * r.risk:.0f}%{sensor_note}</span></div>'
                f'{spark}<div class="why">#{r.rank} · {why}</div></div>')
        st.markdown(f'<div class="wall">{"".join(tiles)}</div>', unsafe_allow_html=True)
        st.caption("Tiles are sorted by priority. The line shows the last 6 hours; the dashed line is the safe limit.")

# ============================ WORK LIST ====================================
with tab_work:
    if queue.empty:
        st.info("No readings at this moment - move the replay slider.")
    else:
        todo = queue[queue["status"] != "OK"].copy()
        st.subheader("Inspect first")
        if todo.empty:
            st.success("Nothing needs attention at this moment.")
        else:
            show = pd.DataFrame({
                "Rank": todo["rank"], "Unit": todo["fridge_id"], "Status": todo["status"].map(lambda s: STATUS[s][0]),
                "Temp °C": todo["p"].round(1), "Risk %": (todo["risk"] * 100).round(0),
                "Priority": todo["priority"], "Why": todo["reasons"], "Recommended action": todo["action"]})
            st.dataframe(show.head(25), hide_index=True, width="stretch", column_config={
                "Risk %": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100, format="%d%%"),
                "Priority": st.column_config.ProgressColumn("Priority", min_value=0, max_value=100, format="%.0f"),
            })
            c1, c2, c3 = st.columns(3)
            c1.download_button("Download work list (CSV)", show.to_csv(index=False).encode("utf-8"),
                               file_name=f"coldsense_worklist_{hours:g}h.csv", mime="text/csv")
            note = handover_text(queue, counts, n_units)
            c2.download_button("Download shift handover note (TXT)", note.encode("utf-8"),
                               file_name=f"coldsense_handover_{hours:g}h.txt", mime="text/plain")
            c3.download_button("Download fleet report (HTML)", D["report_html"],
                               file_name="coldsense_fleet_report.html", mime="text/html")
            with st.expander("Preview the shift handover note"):
                st.code(note, language=None)

# ============================ UNIT REPORT ==================================
with tab_unit:
    options = sorted(view["fridge_id"].unique())
    default = options.index(queue.iloc[0]["fridge_id"]) if len(queue) and queue.iloc[0]["fridge_id"] in options else 0
    fridge = st.selectbox("Choose a unit", options, index=default)
    u = D["unit_comp"][D["unit_comp"]["fridge_id"] == fridge].iloc[0]
    gcol = {"A": TEAL, "B": "#4D7C0F", "C": "#B7791F", "D": "#C2410C"}[u["grade"]]

    def fmt(v, suffix="", nd=1):
        return "n/a" if pd.isna(v) else f"{v:.{nd}f}{suffix}"

    stats = [
        ("Time in range", fmt(u["pct_time_in_range"], "%")), ("Excursions", f"{int(u['n_excursions'])}"),
        ("Longest excursion", f"{int(u['longest_excursion_min'])} min"), ("Mean temp", fmt(u["mean_temp_c"], " °C")),
        ("MKT", fmt(u["mkt_c"], " °C")), ("Max temp", fmt(u["max_temp_c"], " °C")),
        ("Door openings / day", fmt(u["door_openings_per_day"], "", 0)), ("Data completeness", fmt(u["data_completeness_pct"], "%")),
    ]
    stat_html = "".join(f'<div class="stat"><div class="v">{v}</div><div class="k">{k}</div></div>' for k, v in stats)
    st.markdown(f'<div class="ucard"><div class="grade" style="background:{gcol}"><div class="g">{u["grade"]}</div><div class="l">COMPLIANCE</div></div>'
                f'<div class="stats">{stat_html}</div></div>', unsafe_allow_html=True)
    st.caption(f"{fridge} · type: {u['fridge_type']} · {u['hours_recorded']:.0f} h recorded · "
               "Grade A = 98%+ in range with no 2 h excursion, B = 90%+, C = 70%+, D = below. "
               "MKT = mean kinetic temperature, the standard pharma way to summarise a temperature history.")

    d = scored[scored["fridge_id"] == fridge].copy()
    d["hours"] = d["step"] * config.SAMPLE_MIN / 60
    ev = D["events"][D["events"]["fridge_id"] == fridge].copy()
    ev["x0"] = ev["start_step"] * config.SAMPLE_MIN / 60
    ev["x1"] = ev["end_step"] * config.SAMPLE_MIN / 60
    ymin, ymax = float(d["p"].min()) - 0.5, float(d["p"].max()) + 0.5
    ymin, ymax = min(ymin, cfg.hi - 1), max(ymax, cfg.hi + 1)
    ev["y0"], ev["y1"] = ymin, ymax
    x = alt.X("hours:Q", title="Hours since start")
    dom = alt.Scale(domain=[ymin, ymax])
    layers = []
    if len(ev):
        layers.append(alt.Chart(ev).mark_rect(color="#C2410C", opacity=0.14).encode(
            x="x0:Q", x2="x1:Q", y=alt.Y("y0:Q", scale=dom), y2="y1:Q"))
    layers.append(alt.Chart(d).mark_line(color=INK, strokeWidth=1.6).encode(
        x=x, y=alt.Y("p:Q", title="Temperature (°C)", scale=dom)))
    for lim in [cfg.hi] + ([cfg.lo] if cfg.lo is not None else []):
        layers.append(alt.Chart(pd.DataFrame({"y": [lim]})).mark_rule(color="#C2410C", strokeDash=[6, 4]).encode(y=alt.Y("y:Q", scale=dom)))
    flagged = d[d["anomaly_flag"]]
    if len(flagged):
        layers.append(alt.Chart(flagged).mark_point(color="#6D28D9", size=55, filled=True).encode(x="hours:Q", y=alt.Y("p:Q", scale=dom)))
    layers.append(alt.Chart(pd.DataFrame({"x": [hours]})).mark_rule(color=TEAL, strokeWidth=2).encode(x="x:Q"))
    st.altair_chart(style_chart(alt.layer(*layers).properties(
        height=300, title="Temperature · shaded = excursion · dashed = safe limit · purple = sensor looks abnormal · teal line = replay time")),
        width="stretch")
    risk = alt.Chart(d).mark_area(color="#B7791F", opacity=0.45).encode(
        x=x, y=alt.Y("risk:Q", title="Risk", scale=alt.Scale(domain=[0, 1])))
    st.altair_chart(style_chart(risk.properties(height=110, title="Predicted risk of leaving the safe range within 30 minutes")),
                    width="stretch")

    st.markdown("##### Excursion ledger")
    if ev.empty:
        st.success("No excursion events for this unit.")
    else:
        led = ev[["start", "end", "duration_min", "direction", "peak_temp_c", "max_beyond_limit_c", "severity"]].copy()
        led.columns = ["Start", "End", "Minutes", "Direction", "Peak °C", "Beyond limit °C", "Severity"]
        st.dataframe(led, hide_index=True, width="stretch")
        st.download_button("Download this unit's excursion log (CSV)", led.to_csv(index=False).encode("utf-8"),
                           file_name=f"{fridge}_excursions.csv", mime="text/csv")

# ============================ FLEET INSIGHTS ===============================
with tab_fleet:
    st.subheader("What the data says")
    for s in D["insights"]["sentences"]:
        st.markdown(f'<div class="callout">{s}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    hourly = D["hourly"]
    c1.altair_chart(style_chart(alt.Chart(hourly).mark_bar(color="#C2410C").encode(
        x=alt.X("hour:O", title="Hour of day"), y=alt.Y("pct_out_of_range:Q", title="% of unit-time out of range"),
        tooltip=["hour", "pct_out_of_range"]).properties(height=240, title="When do excursions happen?")), width="stretch")
    pos = D["position"]
    c2.altair_chart(style_chart(alt.Chart(pos).mark_bar(color=TEAL).encode(
        y=alt.Y("position:N", sort="-x", title=None), x=alt.X("mean_c:Q", title="Mean temperature (°C)"),
        tooltip=["position", "mean_c", "p95_c", "pct_above_upper_limit"]).properties(height=240, title="Which sensor positions run warm?")),
        width="stretch")

    st.subheader("What if the limit were different?")
    st.caption("Move the upper limit and see how many units would spend a meaningful share of time out of range. "
               "This reuses the recorded temperatures of the units selected in the sidebar.")
    hi_try = st.slider("Upper safe limit (°C)", 0.0, 15.0, float(cfg.hi), step=0.5)
    share = view.assign(out=view["p"] > hi_try).groupby("fridge_id")["out"].mean().mul(100).round(1).sort_values(ascending=False)
    k1, k2, k3 = st.columns(3)
    k1.metric("Units over 10% of the time out of range", int((share > 10).sum()))
    k2.metric("Units never out of range", int((share == 0).sum()))
    k3.metric("Median % of time out of range", f"{share.median():.1f}%")
    top = share.head(15).reset_index()
    top.columns = ["fridge_id", "pct_out"]
    st.altair_chart(style_chart(alt.Chart(top).mark_bar(color="#C2410C").encode(
        y=alt.Y("fridge_id:N", sort="-x", title=None), x=alt.X("pct_out:Q", title="% of time above the limit"),
        tooltip=["fridge_id", "pct_out"]).properties(height=360, title=f"Units with the most time above {hi_try:g} °C")),
        width="stretch")

    st.subheader("All units")
    grades = D["unit_comp"].sort_values("pct_time_in_range")[
        ["fridge_id", "fridge_type", "grade", "pct_time_in_range", "n_excursions", "longest_excursion_min", "mean_temp_c", "mkt_c"]]
    grades.columns = ["Unit", "Type", "Grade", "% in range", "Excursions", "Longest (min)", "Mean °C", "MKT °C"]
    st.dataframe(grades, hide_index=True, width="stretch", column_config={
        "% in range": st.column_config.ProgressColumn("% in range", min_value=0, max_value=100, format="%.0f%%")})

# ============================ TRUST ========================================
with tab_trust:
    st.subheader("Early-warning models (tested on units never seen in training)")
    show = D["model_cmp"][["model", "pr_auc", "roc_auc", "precision", "recall", "f1", "prevalence"]].round(3)
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption(f"Final model, chosen on validation units: **{summary['final_risk_model']}**. PR-AUC is the headline score because "
               "excursions are the minority. 'prevalence' is the share of test moments with a real excursion - a model that guesses "
               "randomly scores about that. The rule baseline ('closer to the limit = riskier') shows what machine learning adds.")
    if D["imp"] is not None:
        st.altair_chart(style_chart(alt.Chart(D["imp"].head(10)).mark_bar(color=TEAL).encode(
            y=alt.Y("feature:N", sort="-x", title=None), x=alt.X("importance:Q", title="Importance")).properties(
            height=260, title="What the model looks at")), width="stretch")

    st.subheader("Sensor-fault detector (tested with injected faults)")
    ev_ = summary["anomaly_eval"]
    a, b, c = st.columns(3)
    a.metric("Fake faults injected", ev_["events_injected"])
    b.metric("Caught", f"{ev_['event_recall_overall']:.0%}")
    c.metric("False alarms on normal data", f"{ev_['false_alarm_rate_on_normal_rows']:.1%}")
    st.write("Caught by fault type:", {k: f"{v:.0%}" for k, v in ev_["event_recall_by_type"].items()})

    st.subheader("Data quality")
    dq = summary["data_quality"]
    st.write(f"{dq['fridges']} units · {dq['total_raw_rows']:,} raw rows · {dq['duplicate_timestamps']} duplicate timestamps · "
             f"{dq['out_of_range_values']} impossible values removed · {dq['fridges_with_gaps_over_15min']} units with gaps over 15 min · "
             f"{dq['fridges_with_frozen_sensor']} units with a frozen-sensor period")
    with st.expander("Per-unit data quality table"):
        st.dataframe(D["dq"], hide_index=True, width="stretch")

    st.subheader("Limitations")
    st.markdown(
        "- The measurements come from **domestic refrigerators**, not pharmaceutical storage or shipments. This is a prototype of the method.\n"
        "- Each unit was recorded for only a few days.\n"
        "- The data contain **no labels** for real alerts or sensor faults: excursions follow the configured safety limit and the "
        "fault test uses **injected** faults.\n"
        "- Limits are configurable (food 5 °C, vaccine 2-8 °C).")

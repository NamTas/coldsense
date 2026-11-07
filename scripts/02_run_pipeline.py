"""
Run the whole ColdSense pipeline: load -> validate -> features -> risk model -> anomaly model -> dashboard files.

    python scripts/02_run_pipeline.py                  # real data in data/raw
    python scripts/02_run_pipeline.py --profile vaccine
    python scripts/02_run_pipeline.py --demo           # synthetic demo data (testing only)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src import anomaly, features, loader, models, reports, validate


def step(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=config.DEFAULT_PROFILE, choices=list(config.PROFILES))
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--demo", action="store_true", help="use synthetic demo data (testing only)")
    args = ap.parse_args()

    data_dir, out_dir = args.data_dir, args.out_dir
    if args.demo:
        data_dir = data_dir or ROOT / "data" / "demo_raw"
        out_dir = out_dir or ROOT / "outputs_demo"
        if not list(Path(data_dir).glob("fridge_*.txt")):
            sys.path.insert(0, str(ROOT / "scripts"))
            import make_demo_data
            make_demo_data.main(data_dir)
        print("*** DEMO MODE: synthetic data. Do not quote these results anywhere. ***")
    cfg = config.get_config(args.profile, data_dir, out_dir)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"profile": cfg.profile, "limit_hi": cfg.hi, "limit_lo": cfg.lo,
               "horizon_min": config.HORIZON_MIN, "demo_data": bool(args.demo)}

    # 1 ------------------------------------------------------------------
    step(f"1/7  Loading files from {cfg.data_dir}")
    raw, infos = loader.load_all(cfg.data_dir, ROOT)
    print(f"  loaded {raw['fridge_id'].nunique()} fridges, {len(raw):,} raw rows")
    summary["files_loaded"] = int(raw["fridge_id"].nunique())
    summary["unmapped_columns"] = sorted({c for i in infos for c in i.get("unmapped", [])})
    if summary["unmapped_columns"]:
        print(f"  note: columns not understood (ignored): {summary['unmapped_columns']}")

    # 2 ------------------------------------------------------------------
    step("2/7  Validating and cleaning sensor data")
    clean, report = validate.validate_and_clean(raw)
    report.to_csv(cfg.out_dir / "data_quality_report.csv", index=False)
    dq = validate.summarize_report(report)
    summary["data_quality"] = dq
    print("  " + json.dumps(dq))
    clean.to_csv(cfg.out_dir / "clean_data.csv.gz", index=False)

    # 3 ------------------------------------------------------------------
    step("3/7  Building features and labels")
    primary = features.choose_primary(clean.columns)
    summary["primary_sensor"] = primary
    print(f"  primary sensor: {primary}   limits: {cfg.lo if cfg.lo is not None else '-'} .. {cfg.hi} C")
    table = features.build_feature_table(clean, cfg, primary)
    feats = features.pick_risk_features(table)
    elig = table[table["eligible"] == 1]
    rate = float(elig["y"].mean()) if len(elig) else 0.0
    share_ok = float((table["p"] <= cfg.hi).mean())
    print(f"  usable rows: {len(elig):,}   share of time below upper limit: {share_ok:.0%}   excursion rate: {rate:.1%}")
    if len(elig) < 500 or rate < 0.01 or rate > 0.60:
        print("  ! The class balance looks unusual for this limit. If the model step fails, try  --profile household")
    summary["features"] = feats

    # 4 ------------------------------------------------------------------
    step("4/7  Training the excursion-risk models (split by fridge)")
    res, best_name, model, thr, groups = models.train_and_compare(table, feats, cfg, cfg.out_dir)
    summary["risk_models"] = res.round(4).to_dict(orient="records")
    summary["final_risk_model"] = best_name
    summary["split_counts"] = {k: len(v) for k, v in groups.items()}
    summary["test_fridges"] = sorted(groups["test"])
    summary["split_fridges"] = {k: sorted(v) for k, v in groups.items()}

    # 5 ------------------------------------------------------------------
    step("5/7  Sensor anomaly detection (Isolation Forest + injected faults)")
    atable = anomaly.anomaly_feature_table(clean, primary)
    afeats = anomaly.usable_features(atable)
    atable["split"] = models.assign_split(atable, groups)
    iso = anomaly.fit_detector(atable[atable["split"] == "train"], afeats)
    val_scores = anomaly.anomaly_scores(iso, atable[atable["split"] == "val"], afeats)
    a_thr = float(np.quantile(val_scores, config.ANOMALY_QUANTILE))
    faulty, events = anomaly.inject_faults(clean, primary, sorted(groups["test"]))
    faulty_feat = anomaly.anomaly_feature_table(faulty, primary)
    ev = anomaly.evaluate_detector(iso, afeats, a_thr, faulty_feat, events)
    summary["anomaly_eval"] = ev
    summary["anomaly_features"] = afeats
    print(f"  injected {ev['events_injected']} fake faults into held-out fridges")
    print(f"  caught {ev['event_recall_overall']:.0%} overall   by type: {ev['event_recall_by_type']}")
    print(f"  false-alarm rate on normal readings: {ev['false_alarm_rate_on_normal_rows']:.1%}")

    # 6 ------------------------------------------------------------------
    step("6/7  Scoring every reading and saving dashboard files")
    table["risk"] = models.score_all(table, model, feats)
    atable["anomaly_score"] = anomaly.anomaly_scores(iso, atable, afeats)
    merged = table.merge(atable[["fridge_id", "step", "anomaly_score"]], on=["fridge_id", "step"], how="left")
    merged["anomaly_flag"] = merged["anomaly_score"] > a_thr
    keep = ["fridge_id", "fridge_type", "timestamp", "step", "split", "p", "amb", "margin_hi", "p_d6",
            "mins_over_60", "over_now", "door_jump12", "risk", "anomaly_score", "anomaly_flag", "y", "eligible"]
    keep = [c for c in keep if c in merged.columns]
    out = merged[keep].copy()
    num_cols = out.select_dtypes("number").columns
    out[num_cols] = out[num_cols].round(4)
    out.to_csv(cfg.out_dir / "scored.csv.gz", index=False)
    summary["anomaly_threshold"] = a_thr

    # 7 ------------------------------------------------------------------
    step("7/7  Building excursion log, unit compliance table and fleet report")
    events, comp, insights = reports.save_all(cfg.out_dir, clean, merged, primary, summary, cfg)
    summary["n_excursion_events"] = int(len(events))
    summary["insights"] = insights["sentences"]
    for sentence in insights["sentences"]:
        print("  - " + sentence)
    (cfg.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\nDone. Files written to {cfg.out_dir}")
    print(f"Fleet report (open in a browser): {cfg.out_dir / 'reports' / 'coldsense_fleet_report.html'}")
    print("Next:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()

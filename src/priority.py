"""
Step 6: "Which unit should the operations team check first?"

priority_score (0-100) =
      55%  risk          (chance of an excursion in the next 30 min; 1.0 if already outside the limit)
    + 25%  persistence   (share of the last hour spent outside the limit)
    + 20%  sensor alert  (anomaly detector flagged this sensor)

Simple, explainable weights (editable in config.py). Every alert also gets plain-English reasons.
"""
import numpy as np
import pandas as pd

import config


def add_priority(df: pd.DataFrame) -> pd.DataFrame:
    w = config.PRIORITY_WEIGHTS
    persistence = (df["mins_over_60"].fillna(0) / 60.0).clip(0, 1)
    anomaly = df["anomaly_flag"].fillna(False).astype(float)
    risk = df["risk"].fillna(0)
    df = df.copy()
    df["priority"] = (100 * (w["risk"] * risk + w["persistence"] * persistence + w["anomaly"] * anomaly)).round(1)
    return df


def reasons_for_row(r, cfg) -> str:
    out = []
    unit = "C"
    if r.get("over_now", 0) == 1:
        mins = int(r.get("mins_over_60", 0))
        out.append(f"Outside safe range now (about {mins} of the last 60 min)")
    else:
        if r.get("p_d6", 0) is not None and r.get("p_d6", 0) > 0.5:
            out.append(f"Warming fast: +{r['p_d6']:.1f}{unit} in 30 min")
        m = r.get("margin_hi")
        if m is not None and not np.isnan(m) and m < 1.0:
            out.append(f"Only {max(m, 0):.1f}{unit} below the {cfg.hi:g}{unit} limit")
    dj = r.get("door_jump12")
    if dj is not None and not np.isnan(dj) and dj > 3:
        out.append("Door sensor spiking (door opened often)")
    amb = r.get("amb")
    if amb is not None and not np.isnan(amb) and amb > 32:
        out.append(f"Hot room ({amb:.0f}{unit})")
    if bool(r.get("anomaly_flag", False)):
        out.append("Sensor reading looks abnormal - check the sensor")
    return "; ".join(out) if out else "Normal"


def priority_queue(snapshot: pd.DataFrame, cfg, top_n: int = 10) -> pd.DataFrame:
    """snapshot = one row per fridge (the 'current' moment). Returns the top_n to investigate."""
    snap = add_priority(snapshot)
    snap = snap.sort_values("priority", ascending=False).head(top_n).copy()
    snap["reasons"] = [reasons_for_row(r, cfg) for r in snap.to_dict("records")]
    snap.insert(0, "rank", range(1, len(snap) + 1))
    return snap


# --------------------------------------------------------------------------
# status labels and recommended actions (used by the dashboard and the work list)
# --------------------------------------------------------------------------
def status_of(r) -> str:
    """ALERT = outside the safe range now | WATCH = likely to leave it soon | SENSOR = sensor looks faulty | OK."""
    if r.get("over_now", 0) == 1:
        return "ALERT"
    risk = r.get("risk", 0)
    if risk is not None and not np.isnan(risk) and risk >= 0.5:
        return "WATCH"
    if bool(r.get("anomaly_flag", False)):
        return "SENSOR"
    return "OK"


def action_for_row(r) -> str:
    st = status_of(r)
    sensor = bool(r.get("anomaly_flag", False))
    if st == "ALERT":
        mins = r.get("mins_over_60", 0) or 0
        base = "Check door seal, power and thermostat now"
        if mins >= 45:
            base += "; if it stays out of range, move contents to a working unit"
        return base + ("; also verify the sensor with a reference thermometer" if sensor else "")
    if st == "WATCH":
        return "Check the door is closed and the load is not blocking airflow; re-check in 30 minutes"
    if st == "SENSOR":
        return "Verify the sensor against a reference thermometer; replace it if readings stay odd"
    return "No action needed"

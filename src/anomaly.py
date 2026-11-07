"""
Step 5: sensor anomaly detection (Isolation Forest).

Problem: the dataset has NO labels saying "this sensor was faulty".
Honest solution: we INJECT fake faults into a copy of the held-out fridges, then check
how many of them the model catches. Real sensor faults are rare, so this is a
simulation-based test - say that clearly in your README / interview.

Fault types we inject:
    spike    one reading jumps by +12 C
    stuck    sensor repeats the same value for 3 hours
    drift    reading slowly climbs by +5 C over 6 hours
    dropout  sensor reads 0.0 for 1 hour
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config
from src import features as F

ANOMALY_FEATURES = ["p_d1", "p_std6", "p_std12", "flat_run", "sib_diff", "dev_mean12"]


def anomaly_feature_table(clean: pd.DataFrame, primary: str) -> pd.DataFrame:
    """Signal features used by the anomaly model (past values only)."""
    parts = []
    for fid, g in clean.groupby("fridge_id", sort=True):
        g = g.sort_values("timestamp").set_index("timestamp")
        f = F.add_signal_features(g, primary)
        f["dev_mean12"] = (f["p"] - f["p_mean12"]).abs()
        f["p_d1"] = f["p_d1"].abs()
        f.insert(0, "fridge_id", fid)
        f["step"] = g["step"].values
        parts.append(f.reset_index())
    return pd.concat(parts, ignore_index=True)


def usable_features(table: pd.DataFrame):
    return [c for c in ANOMALY_FEATURES if c in table.columns and table[c].notna().any()]


def fit_detector(train_table: pd.DataFrame, feats):
    X = train_table[feats].fillna(0.0)
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=config.SEED, n_jobs=2)
    iso.fit(X)
    return iso


def anomaly_scores(iso, table: pd.DataFrame, feats) -> np.ndarray:
    """Higher = more unusual."""
    return -iso.score_samples(table[feats].fillna(0.0))


# --------------------------------------------------------------------------
# fault injection (for testing the detector)
# --------------------------------------------------------------------------
def inject_faults(clean: pd.DataFrame, primary: str, fridge_ids, seed=config.SEED):
    """Return (faulty_copy_of_clean_for_those_fridges, events_df)."""
    rng = np.random.RandomState(seed)
    parts, events = [], []
    for fid in fridge_ids:
        g = clean[clean["fridge_id"] == fid].sort_values("timestamp").copy().reset_index(drop=True)
        n = len(g)
        if n < 200:
            parts.append(g)
            continue
        kinds = ["spike", "stuck", "drift", "dropout"]
        rng.shuffle(kinds)
        seg = n // len(kinds)
        for i, kind in enumerate(kinds):
            length = {"spike": 1, "stuck": 36, "drift": 72, "dropout": 12}[kind]
            lo = i * seg + 20
            hi = (i + 1) * seg - length - 20
            if hi <= lo:
                continue
            start = int(rng.randint(lo, hi))
            end = start + length  # exclusive
            col = g[primary].copy()
            if kind == "spike":
                col.iloc[start] = col.iloc[start] + 12.0
            elif kind == "stuck":
                col.iloc[start:end] = col.iloc[start]
            elif kind == "drift":
                col.iloc[start:end] = col.iloc[start:end].values + np.linspace(0, 5, length)
            elif kind == "dropout":
                col.iloc[start:end] = 0.0
            g[primary] = col
            events.append({"fridge_id": fid, "kind": kind, "start": start, "end": end})
        parts.append(g)
    faulty = pd.concat(parts, ignore_index=True)
    faulty["step"] = faulty.groupby("fridge_id").cumcount()
    return faulty, pd.DataFrame(events)


def evaluate_detector(iso, feats, threshold, faulty_feat: pd.DataFrame, events: pd.DataFrame):
    """Event-level recall per fault type + false-alarm rate on normal rows."""
    ft = faulty_feat.copy()
    ft["score"] = anomaly_scores(iso, ft, feats)
    ft["flag"] = ft["score"] > threshold
    ft["in_event"] = False
    detected = []
    for _, ev in events.iterrows():
        m = ft["fridge_id"] == ev["fridge_id"]
        steps = ft["step"]
        win = m & (steps >= ev["start"]) & (steps < ev["end"] + 6)
        near = m & (steps >= ev["start"] - 2) & (steps < ev["end"] + 24)
        ft.loc[near, "in_event"] = True
        detected.append(bool(ft.loc[win, "flag"].any()))
    events = events.copy()
    events["detected"] = detected
    per_type = events.groupby("kind")["detected"].mean().round(3).to_dict()
    normal = ft[~ft["in_event"]]
    return {
        "events_injected": int(len(events)),
        "event_recall_overall": float(events["detected"].mean()) if len(events) else None,
        "event_recall_by_type": per_type,
        "false_alarm_rate_on_normal_rows": float(normal["flag"].mean()) if len(normal) else None,
        "threshold": float(threshold),
    }

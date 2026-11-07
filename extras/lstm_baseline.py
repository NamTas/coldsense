"""
OPTIONAL bonus: can a small LSTM forecast the fridge temperature 30 minutes ahead better than simple baselines?

Run AFTER the main pipeline (it reuses outputs/clean_data.csv.gz):
    pip install tensorflow          (only needed for this file)
    python extras/lstm_baseline.py                 # real data
    python extras/lstm_baseline.py --out-dir outputs_demo

We compare 3 forecasters on fridges never seen in training:
    persistence  : "in 30 minutes it will be the same as now"
    ridge        : linear model on the last 2 hours
    lstm         : small TensorFlow LSTM on the last 2 hours
Honest result to report: if the LSTM does not clearly beat ridge/persistence, say so - that is a real finding.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from src import features, models

LOOKBACK = 24                 # 24 readings = 2 hours
AHEAD = config.HORIZON_MIN // config.SAMPLE_MIN


def make_windows(series: np.ndarray):
    X, y = [], []
    for i in range(LOOKBACK, len(series) - AHEAD):
        w = series[i - LOOKBACK:i]
        target = series[i + AHEAD - 1]
        if np.isnan(w).any() or np.isnan(target):
            continue
        last = w[-1]
        X.append(w - last)            # shape of the recent curve
        y.append(target - last)       # predict the CHANGE, not the absolute value
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "outputs"))
    ap.add_argument("--epochs", type=int, default=15)
    args = ap.parse_args()
    clean = pd.read_csv(Path(args.out_dir) / "clean_data.csv.gz", parse_dates=["timestamp"])
    primary = features.choose_primary(clean.columns)
    # use exactly the same train / val / test fridges as the main pipeline
    summary = json.loads((Path(args.out_dir) / "run_summary.json").read_text(encoding="utf-8"))
    groups = {k: set(v) for k, v in summary["split_fridges"].items()}

    data = {"train": ([], []), "val": ([], []), "test": ([], [])}
    for fid, g in clean.groupby("fridge_id"):
        split = next(k for k, v in groups.items() if fid in v)
        X, y = make_windows(g.sort_values("timestamp")[primary].values)
        if len(X):
            data[split][0].append(X)
            data[split][1].append(y)
    Xtr, ytr = np.concatenate(data["train"][0]), np.concatenate(data["train"][1])
    Xva, yva = np.concatenate(data["val"][0]), np.concatenate(data["val"][1])
    Xte, yte = np.concatenate(data["test"][0]), np.concatenate(data["test"][1])
    print(f"windows: train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

    def report(name, pred):
        mae = float(np.mean(np.abs(pred - yte)))
        rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
        print(f"  {name:<12} MAE={mae:.3f} C   RMSE={rmse:.3f} C")
        return {"model": name, "mae": mae, "rmse": rmse}

    rows = [report("persistence", np.zeros_like(yte))]

    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    rows.append(report("ridge", ridge.predict(Xte)))

    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow is not installed. Run: pip install tensorflow   (then run this file again)")
        return
    tf.random.set_seed(config.SEED)
    net = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(LOOKBACK, 1)),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    net.compile(optimizer="adam", loss="mae")
    net.fit(Xtr[..., None], ytr, validation_data=(Xva[..., None], yva), epochs=args.epochs,
            batch_size=128, verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
    rows.append(report("lstm", net.predict(Xte[..., None], verbose=0).ravel()))
    out = Path(args.out_dir) / "forecast_comparison.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

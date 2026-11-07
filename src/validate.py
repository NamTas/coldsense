"""
Step 2: sensor data validation + cleaning.

For every fridge we:
  * drop rows with unreadable timestamps and duplicate timestamps,
  * turn physically impossible values (e.g. -50 C or humidity 140%) into "missing",
  * put the readings on a regular 5-minute grid,
  * fill only SHORT gaps (<= 15 min); longer gaps stay missing (we do not invent data),
  * write a data-quality report.
"""
import numpy as np
import pandas as pd

import config


def _sensor_cols(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]


def clean_one_fridge(g: pd.DataFrame, fridge_id: str, fridge_type: str):
    """Return (clean_df indexed by time, stats dict) for a single fridge."""
    stats = {"fridge_id": fridge_id, "fridge_type": fridge_type, "rows_raw": len(g)}
    g = g.copy()
    stats["bad_timestamps"] = int(g["timestamp"].isna().sum())
    g = g.dropna(subset=["timestamp"]).sort_values("timestamp")
    dups = int(g["timestamp"].duplicated().sum())
    stats["duplicate_timestamps"] = dups
    g = g.drop_duplicates(subset="timestamp").set_index("timestamp")

    temp_cols = _sensor_cols(g, "temp_")
    hum_cols = _sensor_cols(g, "hum_")

    # impossible values -> missing
    bad = 0
    lo, hi = config.TEMP_VALID
    for c in temp_cols:
        mask = (g[c] < lo) | (g[c] > hi)
        bad += int(mask.sum())
        g.loc[mask, c] = np.nan
    lo, hi = config.HUM_VALID
    for c in hum_cols:
        mask = (g[c] < lo) | (g[c] > hi)
        bad += int(mask.sum())
        g.loc[mask, c] = np.nan
    stats["out_of_range_values"] = bad

    if len(g) < 2:
        return None, {**stats, "problem": "fewer than 2 valid rows"}

    steps = g.index.to_series().diff().dropna()
    median_step = steps.median()
    stats["median_step_min"] = round(median_step.total_seconds() / 60, 2)
    stats["gaps_over_15min"] = int((steps > pd.Timedelta(minutes=15)).sum())
    stats["start"] = g.index.min()
    stats["end"] = g.index.max()
    stats["duration_h"] = round((g.index.max() - g.index.min()).total_seconds() / 3600, 1)

    # regular grid + short-gap interpolation
    num_cols = temp_cols + hum_cols
    g = g[num_cols].resample(f"{config.SAMPLE_MIN}min").mean()
    missing_before = int(g[num_cols].isna().sum().sum())
    g[num_cols] = g[num_cols].interpolate(
        method="time", limit=config.MAX_INTERPOLATE_STEPS, limit_area="inside"
    )
    missing_after = int(g[num_cols].isna().sum().sum())
    stats["missing_cells_before_fill"] = missing_before
    stats["missing_cells_after_fill"] = missing_after
    cells = max(1, len(g) * max(1, len(num_cols)))
    stats["missing_pct"] = round(100 * missing_after / cells, 2)

    # frozen sensor check: same reading for 3+ hours
    frozen = 0
    for c in temp_cols:
        same = g[c].diff().eq(0)
        run = same.groupby((~same).cumsum()).cumsum()
        frozen += int((run >= 36).sum())
    stats["frozen_readings"] = frozen
    stats["n_grid_rows"] = len(g)
    stats["n_temp_sensors"] = len(temp_cols)
    stats["n_hum_sensors"] = len(hum_cols)
    return g, stats


def validate_and_clean(raw: pd.DataFrame):
    """Run the cleaning for all fridges. Returns (clean_long_df, report_df)."""
    cleaned, rows = [], []
    for fid, g in raw.groupby("fridge_id", sort=True):
        ftype = g["fridge_type"].iloc[0]
        g = g.drop(columns=["fridge_id", "fridge_type"])
        clean, stats = clean_one_fridge(g, fid, ftype)
        rows.append(stats)
        if clean is None:
            continue
        clean = clean.reset_index()
        clean.insert(0, "fridge_type", ftype)
        clean.insert(0, "fridge_id", fid)
        clean["step"] = np.arange(len(clean))
        cleaned.append(clean)
    if not cleaned:
        raise SystemExit("No fridge had usable data after validation.")
    return pd.concat(cleaned, ignore_index=True), pd.DataFrame(rows)


def summarize_report(report: pd.DataFrame) -> dict:
    """Short numbers for the README / dashboard."""
    return {
        "fridges": int(len(report)),
        "total_raw_rows": int(report["rows_raw"].sum()),
        "duplicate_timestamps": int(report["duplicate_timestamps"].sum()),
        "out_of_range_values": int(report["out_of_range_values"].sum()),
        "fridges_with_gaps_over_15min": int((report["gaps_over_15min"] > 0).sum()),
        "fridges_with_frozen_sensor": int((report["frozen_readings"] > 0).sum()),
        "median_missing_pct": float(report["missing_pct"].median()),
    }

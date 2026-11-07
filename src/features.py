"""
Step 3: features + labels.

Rules that keep the model honest:
  * every feature only looks at the PAST (rolling windows end at "now"),
  * the label looks at the FUTURE (next 30 minutes),
  * we only train on moments when the fridge is currently inside its safe range,
    so the model must predict a NEW excursion, not just copy "temperature > limit".
"""
import numpy as np
import pandas as pd

import config


def choose_primary(columns) -> str:
    """Pick the main fridge-air temperature column."""
    if config.PRIMARY_SENSOR and config.PRIMARY_SENSOR in columns:
        return config.PRIMARY_SENSOR
    for c in config.PRIMARY_PRIORITY:
        if c in columns:
            return c
    temps = [c for c in columns if c.startswith("temp_") and c not in ("temp_ambient",)]
    if not temps:
        raise SystemExit("No temperature column found. Check column mapping (README, step 4).")
    return temps[0]


def _air_siblings(columns, primary):
    skip = {primary, "temp_ambient", "temp_product", "temp_freezer"}
    return [c for c in columns if c.startswith("temp_") and c not in skip]


def flat_run_length(s: pd.Series, cap: int = 48) -> pd.Series:
    """How many readings in a row are exactly identical (frozen-sensor signal)."""
    same = s.diff().eq(0)
    run = same.groupby((~same).cumsum()).cumsum()
    return run.clip(upper=cap)


def add_signal_features(g: pd.DataFrame, primary: str) -> pd.DataFrame:
    """Features for ONE fridge (g is sorted by time). Uses only past values."""
    sw, lw = config.SHORT_WINDOW, config.LONG_WINDOW
    f = pd.DataFrame(index=g.index)
    p = g[primary]
    f["p"] = p
    f["p_d1"] = p.diff(1)
    f["p_d3"] = p.diff(3)
    f["p_d6"] = p.diff(sw)
    f["p_mean6"] = p.rolling(sw, min_periods=sw).mean()
    f["p_std6"] = p.rolling(sw, min_periods=sw).std()
    f["p_mean12"] = p.rolling(lw, min_periods=lw).mean()
    f["p_std12"] = p.rolling(lw, min_periods=lw).std()
    f["p_max12"] = p.rolling(lw, min_periods=lw).max()
    f["p_min12"] = p.rolling(lw, min_periods=lw).min()
    f["flat_run"] = flat_run_length(p)

    if "temp_ambient" in g:
        f["amb"] = g["temp_ambient"]
        f["amb_d6"] = g["temp_ambient"].diff(sw)
    if "temp_door" in g and primary != "temp_door":
        f["door"] = g["temp_door"]
        f["door_jump12"] = g["temp_door"].diff().abs().rolling(lw, min_periods=lw).max()
    hum = [c for c in g.columns if c.startswith("hum_") and "ambient" not in c]
    if hum:
        h = g[hum].mean(axis=1)
        f["hum"] = h
        f["hum_d6"] = h.diff(sw)
    sib = _air_siblings(g.columns, primary)
    if sib:
        f["sib_diff"] = p - g[sib].mean(axis=1)

    ts = pd.Series(g.index, index=g.index)
    hour = ts.dt.hour + ts.dt.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return f


def add_limit_features(f: pd.DataFrame, cfg) -> pd.DataFrame:
    """Features that depend on the safety limits."""
    lw = config.LONG_WINDOW
    p = f["p"]
    f["margin_hi"] = cfg.hi - p
    over = p > cfg.hi
    if cfg.lo is not None:
        f["margin_lo"] = p - cfg.lo
        over = over | (p < cfg.lo)
    f["over_now"] = over.astype(float).where(p.notna())
    f["mins_over_60"] = over.astype(float).rolling(lw, min_periods=1).sum() * config.SAMPLE_MIN
    return f


def add_labels(f: pd.DataFrame, cfg) -> pd.DataFrame:
    """y = will an excursion happen in the next HORIZON minutes?  (future-looking)"""
    h = cfg.horizon_steps
    p = f["p"]
    # max/min over the NEXT h readings (t+1 ... t+h)
    fut_max = p[::-1].rolling(h, min_periods=h).max()[::-1].shift(-1)
    fut_min = p[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1)
    y = fut_max > cfg.hi
    if cfg.lo is not None:
        y = y | (fut_min < cfg.lo)
    f["y"] = y.astype(float).where(fut_max.notna())
    # eligible = fridge is OK right now and we know the full future window
    ok_now = (p <= cfg.hi) & (p >= cfg.lo if cfg.lo is not None else True)
    f["eligible"] = (ok_now & f["y"].notna() & p.notna()).astype(int)
    return f


def build_feature_table(clean: pd.DataFrame, cfg, primary: str) -> pd.DataFrame:
    """Features + labels for all fridges (long table)."""
    parts = []
    for fid, g in clean.groupby("fridge_id", sort=True):
        g = g.sort_values("timestamp").set_index("timestamp")
        f = add_signal_features(g, primary)
        f = add_limit_features(f, cfg)
        f = add_labels(f, cfg)
        f.insert(0, "fridge_type", g["fridge_type"].iloc[0])
        f.insert(0, "fridge_id", fid)
        f["step"] = g["step"].values
        parts.append(f.reset_index())
    return pd.concat(parts, ignore_index=True)


# columns used by the risk model (the ones that exist are picked automatically)
RISK_FEATURES = [
    "p", "p_d1", "p_d3", "p_d6", "p_mean6", "p_std6", "p_mean12", "p_std12", "p_max12", "p_min12",
    "margin_hi", "margin_lo", "mins_over_60", "amb", "amb_d6", "door", "door_jump12",
    "hum", "hum_d6", "sib_diff", "hour_sin", "hour_cos",
]


def pick_risk_features(table: pd.DataFrame):
    return [c for c in RISK_FEATURES if c in table.columns and table[c].notna().any()]

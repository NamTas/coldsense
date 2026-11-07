"""
ColdSense settings. Everything you may want to change is in this one file.

Tip for interviews: "I kept all thresholds in one config file so the same code
can run in a food-safety mode (5 C) or a vaccine mode (2-8 C)."
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- Safety-limit profiles (degrees Celsius) -------------------------------
# hi = upper safe limit, lo = lower safe limit (None = no lower limit)
PROFILES = {
    "food": {"hi": 5.0, "lo": None},         # typical food-safety fridge limit
    "household": {"hi": 8.0, "lo": None},    # relaxed limit, useful if 5 C flags almost everything
    "vaccine": {"hi": 8.0, "lo": 2.0},       # WHO-style 2-8 C vaccine range ("same engine, different limits")
}
DEFAULT_PROFILE = "food"

# --- Time settings ----------------------------------------------------------
SAMPLE_MIN = 5            # logger interval in minutes (data is resampled to this grid)
HORIZON_MIN = 30          # we predict: "will an excursion start within the next 30 minutes?"
SHORT_WINDOW = 6          # 6 steps  = 30 minutes
LONG_WINDOW = 12          # 12 steps = 60 minutes
MAX_INTERPOLATE_STEPS = 3 # fill gaps up to 15 minutes, leave longer gaps empty
MIN_EXCURSION_MIN = 10    # an "excursion event" = out of range for at least 10 minutes in a row
DOOR_OPEN_JUMP_C = 2.0    # door-shelf sensor jumping by more than this in 5 min counts as a door opening

# --- Which sensor is "the" fridge temperature? -----------------------------
# None = auto-pick from this priority list (first one that exists).
PRIMARY_SENSOR = None     # e.g. "temp_middle"
PRIMARY_PRIORITY = ["temp_middle", "temp_top", "temp_air", "temp_veg", "temp_door", "temp_product"]

# --- Plausible physical ranges (values outside become missing) --------------
TEMP_VALID = (-35.0, 60.0)
HUM_VALID = (0.0, 100.0)

# --- Modelling --------------------------------------------------------------
SEED = 42
SPLIT = (0.60, 0.15, 0.25)   # train / validation / test, split BY FRIDGE (no leakage)
ANOMALY_QUANTILE = 0.99      # top 1% most unusual healthy readings are flagged
PRIORITY_WEIGHTS = {"risk": 0.55, "persistence": 0.25, "anomaly": 0.20}


@dataclass
class Config:
    profile: str
    hi: float
    lo: object  # float or None
    data_dir: Path
    out_dir: Path

    @property
    def horizon_steps(self) -> int:
        return max(1, HORIZON_MIN // SAMPLE_MIN)


def get_config(profile=None, data_dir=None, out_dir=None) -> Config:
    profile = profile or DEFAULT_PROFILE
    if profile not in PROFILES:
        raise SystemExit(f"Unknown profile '{profile}'. Choose one of: {list(PROFILES)}")
    p = PROFILES[profile]
    return Config(
        profile=profile,
        hi=p["hi"],
        lo=p["lo"],
        data_dir=Path(data_dir) if data_dir else ROOT / "data" / "raw",
        out_dir=Path(out_dir) if out_dir else ROOT / "outputs",
    )

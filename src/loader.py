"""
Step 1: read the raw logger files (tab-delimited .txt) into one tidy table.

I could not open the real dataset while writing this, so the loader does NOT
hard-code column names. It:
  * guesses the delimiter and finds the header row,
  * finds the date/time column(s),
  * guesses what each numeric column is (temperature or humidity + sensor position)
    from its name, e.g. "Temp_Top" -> temp_top, "RH door" -> hum_door.

If a guess is wrong, run  python scripts/01_inspect_data.py  and fix it in
column_map.json (no code change needed).
"""
import json
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ENCODINGS = ["utf-8-sig", "utf-8", "cp874", "latin-1"]  # cp874 = Thai Windows encoding

# (position name, keywords found in the column name). Order matters.
POSITION_KEYWORDS = [
    ("product", ["product", "meat", "prod", "probe"]),
    ("ambient", ["ambient", "amb", "room", "outside", "external", "kitchen"]),
    ("freezer", ["freezer", "coldest", "frz"]),
    ("door", ["door"]),
    ("veg", ["veg", "drawer", "crisper", "bottom"]),
    ("middle", ["middle", "mid"]),
    ("top", ["top", "upper"]),
    ("air", ["air"]),
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _norm(name: str) -> str:
    """'Temp Top (°C)' -> 'temp_top_°c'"""
    return re.sub(r"[^a-z0-9%°]+", "_", str(name).lower()).strip("_")


def _kind(name: str):
    n = _norm(name)
    if re.search(r"hum|(^|_)(rh|h)(_|\d|$)|%rh", n):
        return "hum"
    if re.search(r"temp|°|celsius|deg|(^|_)t(_|\d|$)", n):
        return "temp"
    return None


def _position(name: str):
    """Which part of the fridge is this sensor in? Works on words, so 'humidity' never matches 'mid'."""
    n = _norm(name)
    n = re.sub(r"humidity|temperature|temp|hum|celsius|[°%]", " ", n)  # remove the 'kind' words first
    tokens = [t for t in re.split(r"[_\s]+", n) if t]
    for pos, words in POSITION_KEYWORDS:           # 1) whole-word match (token starts with keyword)
        if any(tok.startswith(w) for tok in tokens for w in words):
            return pos
    joined = "".join(tokens)                      # 2) fallback for names like 'TempTop' -> 'top'
    for pos, words in POSITION_KEYWORDS:
        if any(w in joined for w in words):
            return pos
    return None


def _to_number(s: pd.Series) -> pd.Series:
    """Convert text to numbers; also handles decimal commas like '4,5'."""
    if s.dtype == object or str(s.dtype).startswith("str"):
        s = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _guess_sep(lines):
    best_sep, best_cols = "\t", 1
    for sep in ["\t", ";", ",", "|"]:
        counts = [len(l.split(sep)) for l in lines[:60]]
        modal = Counter(counts).most_common(1)[0][0]
        if modal > best_cols:
            best_sep, best_cols = sep, modal
    return best_sep


def parse_datetime(s: pd.Series) -> pd.Series:
    """Parse timestamps. Tries month-first and day-first and keeps the one that gives a
    regular 5-minute-style series (the wrong choice makes huge jumps between days).
    Also converts Buddhist-era years (2568 -> 2025), which Thai loggers may use."""
    s = s.astype(str).str.strip()
    first = s.iloc[0] if len(s) else ""
    if re.search(r"\b25\d{2}\b", first):
        s = s.str.replace(r"\b(25\d{2})\b", lambda m: str(int(m.group(1)) - 543), regex=True)
    iso = bool(re.match(r"^\d{4}-\d{2}-\d{2}", first))
    best, best_key = None, None
    for dayfirst in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
        nat_share = parsed.isna().mean()
        diffs = parsed.diff().dropna()
        if len(diffs):
            regular = (diffs == diffs.median()).mean()
        else:
            regular = 0
        prefer = 0 if (dayfirst == (not iso)) else 1   # tie-break: ISO dates month-first, others day-first
        key = (round(nat_share, 3), -round(regular, 3), prefer)
        if best_key is None or key < best_key:
            best, best_key = parsed, key
    return best


# --------------------------------------------------------------------------
# reading one file
# --------------------------------------------------------------------------
def read_raw_table(path: Path) -> pd.DataFrame:
    text = None
    for enc in ENCODINGS:
        try:
            text = Path(path).read_text(encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f"Cannot decode {path}")
    lines = [l for l in text.splitlines() if l.strip()]
    sep = _guess_sep(lines)
    counts = [len(l.split(sep)) for l in lines]
    modal = Counter(counts).most_common(1)[0][0]
    rows = [[c.strip() for c in l.split(sep)] for l, c in zip(lines, counts) if c == modal]

    def numeric_share(row):
        return np.mean([not np.isnan(pd.to_numeric(c.replace(",", "."), errors="coerce")) for c in row])

    has_header = len(rows) > 1 and numeric_share(rows[0]) < 0.5 * max(numeric_share(rows[1]), 1e-9)
    if has_header or (len(rows) > 1 and numeric_share(rows[0]) == 0 and numeric_share(rows[1]) > 0):
        df = pd.DataFrame(rows[1:], columns=[c or f"col{i}" for i, c in enumerate(rows[0])])
    else:
        df = pd.DataFrame(rows, columns=[f"col{i}" for i in range(modal)])
    return df


def _find_timestamp(df: pd.DataFrame, override: dict):
    """Return (timestamp_series, list_of_columns_used)."""
    if override.get("datetime") in df.columns:
        c = override["datetime"]
        return parse_datetime(df[c]), [c]
    if override.get("date") in df.columns and override.get("time") in df.columns:
        d, t = override["date"], override["time"]
        return parse_datetime(df[d] + " " + df[t]), [d, t]

    cols = list(df.columns)
    nm = {c: _norm(c) for c in cols}
    dt = [c for c in cols if re.search(r"date_?time|timestamp", nm[c])]
    if dt:
        return parse_datetime(df[dt[0]]), [dt[0]]
    date_cols = [c for c in cols if "date" in nm[c]]
    time_cols = [c for c in cols if re.search(r"(^|_)time($|_)", nm[c])]
    if date_cols and time_cols:
        return parse_datetime(df[date_cols[0]] + " " + df[time_cols[0]]), [date_cols[0], time_cols[0]]
    if date_cols:
        return parse_datetime(df[date_cols[0]]), [date_cols[0]]
    if time_cols:
        return parse_datetime(df[time_cols[0]]), [time_cols[0]]
    for c in cols[:3]:
        parsed = parse_datetime(df[c])
        if parsed.notna().mean() > 0.9:
            return parsed, [c]
    raise ValueError("Could not find a date/time column. Use column_map.json (see README).")


def standardize_file(path: Path, override: dict):
    """Read one logger file -> (DataFrame[timestamp, temp_*, hum_*], info dict)."""
    raw = read_raw_table(path)
    ts, used = _find_timestamp(raw, override)
    out = pd.DataFrame({"timestamp": ts})
    sensor_override = override.get("sensors", {})
    mapping, unmapped, taken = {}, [], set()
    for c in raw.columns:
        if c in used:
            continue
        num = _to_number(raw[c])
        if num.notna().mean() < 0.5:
            continue  # not a numeric column (e.g. a text note)
        if c in sensor_override:
            key = sensor_override[c]
        else:
            kind, pos = _kind(c), _position(c)
            if kind is None:
                unmapped.append(c)
                continue
            key = f"{kind}_{pos or 'x'}"
        base, i = key, 2
        while key in taken:
            key, i = f"{base}{i}", i + 1
        taken.add(key)
        mapping[c] = key
        out[key] = num.values
    info = {"file": Path(path).name, "rows": len(out), "mapping": mapping, "unmapped": unmapped,
            "time_columns": used, "bad_timestamps": int(out["timestamp"].isna().sum())}
    return out, info


def parse_fridge_name(stem: str):
    """'fridge_012_twodoor' -> id 'fridge_012', type 'twodoor' (best effort)."""
    tokens = [t for t in re.split(r"[_\-\s]+", stem) if t]
    words = [t for t in tokens if not t.isdigit() and t.lower() not in ("fridge", "refrigerator", "ref", "no")]
    return stem, ("_".join(words) if words else "unknown")


# --------------------------------------------------------------------------
# reading a whole folder
# --------------------------------------------------------------------------
def load_override(root: Path) -> dict:
    p = Path(root) / "column_map.json"
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return {k: v for k, v in data.items() if not k.startswith("_")}
        except json.JSONDecodeError as e:
            raise SystemExit(f"column_map.json is not valid JSON: {e}")
    return {}


def list_data_files(data_dir: Path):
    files = []
    for pat in ("*.txt", "*.tsv", "*.csv"):
        files += list(Path(data_dir).rglob(pat))   # also finds files inside sub-folders
    return sorted(f for f in files if "readme" not in f.name.lower())


def load_all(data_dir: Path, root: Path):
    """Load every logger file. Returns (big_dataframe, list_of_info)."""
    override = load_override(root)
    files = list_data_files(data_dir)
    if not files:
        raise SystemExit(
            f"No data files found in {data_dir}\n"
            "Put the dataset .txt files there (see README step 3), or try the demo: "
            "python scripts/02_run_pipeline.py --demo"
        )
    frames, infos = [], []
    for f in files:
        try:
            df, info = standardize_file(f, override)
        except Exception as e:  # keep going, report the problem file
            infos.append({"file": f.name, "error": str(e)})
            print(f"  ! skipped {f.name}: {e}")
            continue
        fid, ftype = parse_fridge_name(f.stem)
        df.insert(0, "fridge_type", ftype)
        df.insert(0, "fridge_id", fid)
        frames.append(df)
        infos.append(info)
    if not frames:
        raise SystemExit("All files failed to load. Run scripts/01_inspect_data.py for details.")
    return pd.concat(frames, ignore_index=True), infos

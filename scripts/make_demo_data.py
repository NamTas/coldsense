"""
Make SYNTHETIC demo fridge files so you can test the whole project before you download the real dataset.

!! This is fake data. Never put demo results in your README, CV or interview.
   Real results = run the pipeline on the real dataset in data/raw/.

Usage:  python scripts/make_demo_data.py            (writes to data/demo_raw/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STEP = 5  # minutes


def simulate_fridge(rng, n_steps=1440, failing=False, outage=False):
    setpoint = rng.uniform(3.0, 7.5)
    hyst = rng.uniform(0.5, 0.9)
    start = pd.Timestamp("2025-11-03 08:00:00") + pd.Timedelta(days=int(rng.randint(0, 20)))
    idx = pd.date_range(start, periods=n_steps, freq=f"{STEP}min")
    hours = (idx.hour + idx.minute / 60).values
    amb = 30 + 3 * np.sin((hours - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.3, n_steps)

    t = setpoint
    comp_on = False
    air = np.zeros(n_steps)
    door_extra = np.zeros(n_steps)
    outage_start = int(rng.randint(400, 1000))
    for i in range(n_steps):
        sp = setpoint
        if failing and i > n_steps * 0.55:
            sp = setpoint + 0.012 * (i - n_steps * 0.55)  # thermostat slowly failing
        # door openings
        if rng.rand() < 0.012:
            door_extra[i] = rng.uniform(2.0, 5.0)
        heat = 0.006 * (amb[i] - t) + 0.05 * door_extra[i]
        if outage and outage_start <= i < outage_start + 36:
            comp_on = False  # power cut for 3 hours
        elif t > sp + hyst:
            comp_on = True
        elif t < sp - hyst:
            comp_on = False
        cool = 0.35 if comp_on else 0.0
        t = t + heat * 5 / 5 - cool + rng.normal(0, 0.03)
        air[i] = t
    prod = pd.Series(air).ewm(alpha=0.06).mean().values
    cols = {
        "temp_top": air + 0.6 + rng.normal(0, 0.05, n_steps),
        "temp_middle": air + rng.normal(0, 0.05, n_steps),
        "temp_veg": air - 0.5 + rng.normal(0, 0.05, n_steps),
        "temp_door": air + 1.5 + door_extra + rng.normal(0, 0.1, n_steps),
        "temp_freezer": air - 12 + rng.normal(0, 0.2, n_steps),
        "temp_ambient": amb,
        "temp_product": prod,
        "hum_top": 65 + rng.normal(0, 2, n_steps) + door_extra,
        "hum_middle": 70 + rng.normal(0, 2, n_steps),
        "hum_veg": 80 + rng.normal(0, 2, n_steps),
    }
    df = pd.DataFrame(cols, index=idx).round(1)
    return df


def write_format_a(path, df):
    out = df.copy()
    out.insert(0, "Date_Time", out.index.strftime("%Y-%m-%d %H:%M:%S"))
    out.columns = ["Date_Time", "Temp_Top (°C)", "Temp_Middle (°C)", "Temp_Veg (°C)", "Temp_Door (°C)",
                   "Temp_Freezer (°C)", "Temp_Ambient (°C)", "Temp_Product (°C)",
                   "RH_Top (%)", "RH_Middle (%)", "RH_Veg (%)"]
    out.to_csv(path, sep="\t", index=False)


def write_format_b(path, df, decimal_comma=False, buddhist_year=False):
    """Different layout: metadata lines, separate Date and Time columns, dd/mm/yyyy."""
    out = df.copy()
    years = out.index.year + (543 if buddhist_year else 0)
    date = [f"{d.day:02d}/{d.month:02d}/{y}" for d, y in zip(out.index, years)]
    out.insert(0, "Time", out.index.strftime("%H:%M:%S"))
    out.insert(0, "Date", date)
    out.columns = ["Date", "Time", "Top shelf temperature", "Middle shelf temperature",
                   "Vegetable drawer temperature", "Door shelf temperature", "Freezer temperature",
                   "Ambient temperature", "Product temperature",
                   "Top shelf humidity", "Middle shelf humidity", "Vegetable drawer humidity"]
    text = out.to_csv(sep="\t", index=False, decimal="," if decimal_comma else ".")
    header = "Logger export\nSerial: TP-0001\n"
    Path(path).write_text(header + text, encoding="utf-8")


def main(out_dir=None, n_fridges=30):
    out_dir = Path(out_dir) if out_dir else ROOT / "data" / "demo_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.txt"):
        old.unlink()
    rng = np.random.RandomState(7)
    types = ["onedoor", "twodoor", "sidebyside"]
    for k in range(1, n_fridges + 1):
        failing = k % 7 == 0
        outage = k % 5 == 0
        df = simulate_fridge(rng, failing=failing, outage=outage)
        name = f"fridge_{k:03d}_{types[k % 3]}.txt"
        if k <= n_fridges // 2:
            write_format_a(out_dir / name, df)
        else:
            write_format_b(out_dir / name, df, decimal_comma=(k % 4 == 0), buddhist_year=(k % 6 == 0))
    (out_dir / "README.txt").write_text("Synthetic demo data. Not real measurements.\n")
    print(f"Wrote {n_fridges} synthetic fridge files to {out_dir}")


if __name__ == "__main__":
    main()

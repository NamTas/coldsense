"""Small safety-net tests. Run:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config
from src import features, loader, validate


def test_sensor_name_guessing():
    assert loader._kind("Top shelf humidity") == "hum"
    assert loader._position("Top shelf humidity") == "top"      # 'humidity' must not look like 'mid'
    assert loader._kind("Temp_Middle (°C)") == "temp"
    assert loader._position("Temp_Middle (°C)") == "middle"
    assert loader._position("Door shelf temperature") == "door"


def test_datetime_parsing_dayfirst_and_buddhist_year():
    s = pd.Series(["12/11/2568 23:55:00", "13/11/2568 00:00:00", "13/11/2568 00:05:00"])
    t = loader.parse_datetime(s)
    assert t.iloc[0] == pd.Timestamp("2025-11-12 23:55:00")
    assert t.iloc[1] == pd.Timestamp("2025-11-13 00:00:00")


def _fake_clean(n=300):
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    temp = np.concatenate([np.full(200, 4.0), np.linspace(4, 9, 100)])  # fridge warms up at the end
    return pd.DataFrame({"fridge_id": "f1", "fridge_type": "t", "timestamp": idx,
                         "temp_middle": temp, "step": np.arange(n)})


def test_labels_look_forward_and_only_ok_rows_are_eligible():
    cfg = config.get_config("food")           # limit 5 C
    clean = _fake_clean()
    table = features.build_feature_table(clean, cfg, "temp_middle")
    # a row 3 steps before the temperature crosses 5 C must be labelled 1
    cross = int(np.argmax(clean["temp_middle"].values > 5))
    row = table[table["step"] == cross - 3].iloc[0]
    assert row["y"] == 1 and row["eligible"] == 1
    # rows already above the limit are never used for training
    assert table[table["p"] > 5]["eligible"].sum() == 0
    # a row early in the flat part has no excursion ahead
    assert table[table["step"] == 50].iloc[0]["y"] == 0


def test_features_do_not_peek_into_the_future():
    cfg = config.get_config("food")
    clean = _fake_clean()
    a = features.build_feature_table(clean, cfg, "temp_middle")
    changed = clean.copy()
    changed.loc[changed["step"] >= 250, "temp_middle"] += 20   # change only the far future
    b = features.build_feature_table(changed, cfg, "temp_middle")
    cols = [c for c in ["p", "p_d6", "p_mean12", "p_std12"] if c in a.columns]
    pd.testing.assert_frame_equal(a[a["step"] < 250][cols].reset_index(drop=True),
                                  b[b["step"] < 250][cols].reset_index(drop=True))


def test_validation_removes_impossible_values():
    idx = pd.date_range("2025-01-01", periods=50, freq="5min")
    g = pd.DataFrame({"timestamp": idx, "temp_middle": 4.0, "hum_middle": 60.0})
    g.loc[5, "temp_middle"] = -99      # impossible
    g.loc[6, "hum_middle"] = 250       # impossible
    clean, stats = validate.clean_one_fridge(g, "f", "t")
    assert stats["out_of_range_values"] == 2
    assert clean["temp_middle"].notna().all()   # single bad points are interpolated away


def test_mean_kinetic_temperature_and_excursion_events():
    from src import reports
    # constant temperature -> MKT equals that temperature
    assert abs(reports.mean_kinetic_temperature([5.0] * 50) - 5.0) < 1e-6
    # warm periods weigh more than cold ones: MKT is above the plain mean
    temps = [2.0] * 50 + [12.0] * 50
    assert reports.mean_kinetic_temperature(temps) > np.mean(temps)

    cfg = config.get_config("food")           # limit 5 C
    n = 100
    idx = pd.date_range("2025-01-01", periods=n, freq="5min")
    temp = np.full(n, 4.0)
    temp[20:30] = 7.0                         # 10 readings = 50 min above the limit -> one event
    temp[60] = 9.0                            # a single reading (5 min) is too short to count
    clean = pd.DataFrame({"fridge_id": "f1", "fridge_type": "t", "timestamp": idx, "temp_middle": temp, "step": np.arange(n)})
    ev = reports.find_excursions(clean, "temp_middle", cfg)
    assert len(ev) == 1
    assert ev.iloc[0]["duration_min"] == 50 and ev.iloc[0]["severity"] == "moderate"
    assert ev.iloc[0]["direction"] == "too warm"


def test_status_and_action():
    from src import priority
    assert priority.status_of({"over_now": 1, "risk": 1.0}) == "ALERT"
    assert priority.status_of({"over_now": 0, "risk": 0.7}) == "WATCH"
    assert priority.status_of({"over_now": 0, "risk": 0.1, "anomaly_flag": True}) == "SENSOR"
    assert priority.status_of({"over_now": 0, "risk": 0.1, "anomaly_flag": False}) == "OK"
    assert "No action" in priority.action_for_row({"over_now": 0, "risk": 0.1})

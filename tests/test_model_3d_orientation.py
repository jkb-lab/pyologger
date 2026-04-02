import io
import pathlib
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pyologger.dash.minimal_interactive.model_3d import build_orientation_data_json


def _read_split_json(json_text):
    return pd.read_json(io.StringIO(json_text), orient="split")


def test_build_orientation_data_json_accepts_case_insensitive_grouped_prh():
    signal_data = {
        "PRH": pd.DataFrame(
            {
                "DateTime": pd.date_range("2024-01-01T00:00:00Z", periods=4, freq="1s"),
                "Pitch": [1.0, 2.0, 3.0, 4.0],
                "Roll": [0.5, 0.6, 0.7, 0.8],
                "Heading": [10.0, 20.0, 30.0, 40.0],
            }
        )
    }
    data_pkl = SimpleNamespace(signal_data=signal_data)

    result = build_orientation_data_json(data_pkl)

    assert result["ok"] is True
    frame = _read_split_json(result["data_json"])
    assert {"pitch", "roll", "heading"}.issubset(set(frame.columns))
    assert len(frame) == 4


def test_build_orientation_data_json_falls_back_to_split_signals():
    base_dt = pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="1s")
    signal_data = {
        "pitch": pd.DataFrame({"datetime": base_dt, "Pitch": [1.0, 1.5, 2.0]}),
        "roll": pd.DataFrame({"datetime": base_dt, "roll": [0.1, 0.2, 0.3]}),
        "heading": pd.DataFrame({"datetime": base_dt, "yaw": [90.0, 95.0, 100.0]}),
    }
    data_pkl = SimpleNamespace(signal_data=signal_data)

    result = build_orientation_data_json(data_pkl)

    assert result["ok"] is True
    frame = _read_split_json(result["data_json"])
    assert {"pitch", "roll", "heading"}.issubset(set(frame.columns))
    assert len(frame) == 3


def test_build_orientation_data_json_uses_zero_heading_when_missing():
    base_dt = pd.date_range("2024-01-01T00:00:00Z", periods=3, freq="1s")
    signal_data = {
        "pitch": pd.DataFrame({"datetime": base_dt, "pitch": [1.0, 1.5, 2.0]}),
        "roll": pd.DataFrame({"datetime": base_dt, "roll": [0.1, 0.2, 0.3]}),
    }
    data_pkl = SimpleNamespace(signal_data=signal_data)

    result = build_orientation_data_json(data_pkl)

    assert result["ok"] is True
    frame = _read_split_json(result["data_json"])
    assert "heading" in frame.columns
    assert frame["heading"].fillna(0.0).eq(0.0).all()

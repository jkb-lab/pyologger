from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODULE_PATH = REPO_ROOT / "pyologger" / "load_data" / "datareader.py"
spec = importlib.util.spec_from_file_location("datareader_test", MODULE_PATH)
datareader_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = datareader_mod
assert spec.loader is not None
spec.loader.exec_module(datareader_mod)
DataReader = datareader_mod.DataReader


def _reader_without_init() -> DataReader:
    reader = DataReader.__new__(DataReader)
    reader.event_data = pd.DataFrame(columns=["datetime", "key", "value", "note", "type", "duration"])
    reader.event_manager = {}
    reader.deployment_info = {"Time Zone": "UTC"}
    return reader


def test_append_label_state_events_adds_unscorable_dive_type_when_dive_num_missing():
    reader = _reader_without_init()
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2022-01-01 00:00:00", periods=8, freq="1min", tz="UTC"),
            "dive_type_project": [3, 3, 6, 6, 9, 9, 3, 3],
            "dive_num_all-incl-inacc-depth": [1, 1, 2, 2, 3, 3, 4, 4],
            "dive_num": [1, 1, 2, 2, pd.NA, pd.NA, 4, 4],
        }
    )
    metadata = {
        "dive_type_project": {
            "original_name": "pred_divetype",
            "parent_signal": "label",
            "unit": "code",
            "standardized_unit": "str",
        },
        "dive_num_all-incl-inacc-depth": {
            "original_name": "divenumber_inc_inaccuratedepth",
            "parent_signal": "dives",
            "unit": "count",
            "standardized_unit": "count",
        },
        "dive_num": {
            "original_name": "divenumber",
            "parent_signal": "dives",
            "unit": "count",
            "standardized_unit": "count",
        },
    }

    created = reader._append_label_state_events_from_dataframe(df, metadata, logger_id="PD-1")

    assert created > 0
    assert not reader.event_data.empty
    assert "dive-type_unscorable" in set(reader.event_data["key"].astype(str))

    unscorable = reader.event_data.loc[reader.event_data["key"].astype(str) == "dive-type_unscorable"].copy()
    assert len(unscorable) == 1
    row = unscorable.iloc[0]
    assert str(row["type"]) == "state"
    assert float(row["duration"]) == 120.0
    assert str(row["note"]) == "UNSCORABLE"

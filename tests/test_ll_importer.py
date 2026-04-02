from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODULE_PATH = REPO_ROOT / "pyologger" / "io_operations" / "ll_importer.py"
spec = importlib.util.spec_from_file_location("ll_importer_test", MODULE_PATH)
ll_importer_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ll_importer_mod
assert spec.loader is not None
spec.loader.exec_module(ll_importer_mod)
LLImporter = ll_importer_mod.LLImporter


def _ll_importer_without_init() -> LLImporter:
    return LLImporter.__new__(LLImporter)


def test_ll_importer_converts_stroke_rate_count_per_5s_to_spm():
    importer = _ll_importer_without_init()
    df = pd.DataFrame({"stroke_rate": [2.0, 3.5, np.nan]})
    channel_metadata = {
        "stroke_rate": {
            "parent_signal": "stroke_rate",
            "unit": "count/5s",
            "standardized_unit": "count/5s",
        }
    }

    out = importer._convert_stroke_rate_count_per_5s_to_spm(df, channel_metadata)

    assert out["stroke_rate"].tolist()[:2] == [24.0, 42.0]
    assert np.isnan(out["stroke_rate"].tolist()[2])
    assert channel_metadata["stroke_rate"]["unit"] == "spm"
    assert channel_metadata["stroke_rate"]["standardized_unit"] == "spm"


def test_ll_importer_keeps_stroke_rate_when_not_count_per_5s():
    importer = _ll_importer_without_init()
    df = pd.DataFrame({"stroke_rate": [24.0, 42.0]})
    channel_metadata = {
        "stroke_rate": {
            "parent_signal": "stroke_rate",
            "unit": "spm",
            "standardized_unit": "spm",
        }
    }

    out = importer._convert_stroke_rate_count_per_5s_to_spm(df, channel_metadata)

    assert out["stroke_rate"].tolist() == [24.0, 42.0]
    assert channel_metadata["stroke_rate"]["unit"] == "spm"
    assert channel_metadata["stroke_rate"]["standardized_unit"] == "spm"

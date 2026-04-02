from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MODULE_PATH = REPO_ROOT / "pyologger" / "io_operations" / "pd_importer.py"
spec = importlib.util.spec_from_file_location("pd_importer_test", MODULE_PATH)
pd_importer_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pd_importer_mod
assert spec.loader is not None
spec.loader.exec_module(pd_importer_mod)
PDImporter = pd_importer_mod.PDImporter


def _pd_importer_without_init() -> PDImporter:
    # apply_post_rename_transforms does not depend on BaseImporter init state.
    return PDImporter.__new__(PDImporter)


def test_pd_importer_converts_stroke_rate_hz_to_strokes_per_minute():
    importer = _pd_importer_without_init()
    df = pd.DataFrame({"stroke_rate": [0.5, 1.0, np.nan]})
    channel_metadata = {
        "stroke_rate": {
            "parent_signal": "stroke_rate",
            "unit": "Hz",
            "standardized_unit": "Hz",
        }
    }

    out = importer.apply_post_rename_transforms(df, channel_metadata)

    assert out["stroke_rate"].tolist()[:2] == [30.0, 60.0]
    assert np.isnan(out["stroke_rate"].tolist()[2])
    assert channel_metadata["stroke_rate"]["unit"] == "spm"
    assert channel_metadata["stroke_rate"]["standardized_unit"] == "spm"


def test_pd_importer_keeps_stroke_rate_when_not_hz():
    importer = _pd_importer_without_init()
    df = pd.DataFrame({"stroke_rate": [30.0, 45.0]})
    channel_metadata = {
        "stroke_rate": {
            "parent_signal": "stroke_rate",
            "unit": "spm",
            "standardized_unit": "spm",
        }
    }

    out = importer.apply_post_rename_transforms(df, channel_metadata)

    assert out["stroke_rate"].tolist() == [30.0, 45.0]
    assert channel_metadata["stroke_rate"]["unit"] == "spm"
    assert channel_metadata["stroke_rate"]["standardized_unit"] == "spm"

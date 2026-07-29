"""
Tests for accelerometer unit standardization.

Mirrors how the pressure pipeline standardizes depth to metres: the montage
declares `standardized_unit`, and the importer must actually convert the values
rather than only relabel them. Previously `standardized_unit` was a claim only,
so stored values could be off by ~9.81x from what the metadata said.

**The project standard is `g`**, matching the convention that ODBA/VeDBA are
reported in g. Conversion therefore runs in both directions: tags recording in
g pass through unchanged, while natively-m/s^2 tags (CATS) are divided by
standard gravity. The g -> m/s^2 direction is retained and tested because the
montage may still request it per channel.
"""

import numpy as np
import pandas as pd
import pytest

from pyologger.io_operations.base_importer import BaseImporter

G = BaseImporter.STANDARD_GRAVITY_MS2


class _Importer(BaseImporter):
    """Bare subclass: BaseImporter.__init__ needs a DataReader we don't want here."""

    def __init__(self):
        pass


def _frame_and_metadata(unit="g", target="g", parent="accelerometer"):
    df = pd.DataFrame({
        "ax": [1.0, -1.0, 0.5, np.nan],
        "ay": [0.0, 0.5, -0.25, 0.0],
        "az": [-1.0, 0.0, 0.0, 1.0],
    })
    metadata = {
        axis: {"unit": unit, "standardized_unit": target, "parent_signal": parent}
        for axis in ("ax", "ay", "az")
    }
    return df, metadata


def test_standard_gravity_constant():
    assert G == pytest.approx(9.80665)


@pytest.mark.parametrize("unit", ["g", "G", " g ", "gs", "g-force"])
def test_g_unit_variants_detected(unit):
    assert BaseImporter._is_g_unit(unit)


@pytest.mark.parametrize("unit", ["m/s^2", "m/s2", "m/s²", "M/S^2", "m / s ^ 2"])
def test_ms2_unit_variants_detected(unit):
    assert BaseImporter._is_ms2_unit(unit)


@pytest.mark.parametrize("unit", ["lux", "count", "m", "degreeC", None, ""])
def test_non_acceleration_units_not_detected(unit):
    assert not BaseImporter._is_g_unit(unit)
    assert not BaseImporter._is_ms2_unit(unit)


def test_g_values_are_scaled_when_target_is_ms2():
    df, metadata = _frame_and_metadata(target="m/s^2")
    out = _Importer().convert_acceleration_to_standard_unit(df, metadata)
    assert out["ax"].iloc[0] == pytest.approx(G)
    assert out["ax"].iloc[1] == pytest.approx(-G)
    assert out["ax"].iloc[2] == pytest.approx(0.5 * G)
    assert out["az"].iloc[3] == pytest.approx(G)


def test_ms2_values_are_scaled_down_to_g():
    """CATS tags record native m/s^2; the g standard requires dividing."""
    df = pd.DataFrame({"ax": [G, -G, G / 2.0]})
    metadata = {"ax": {"unit": "m/s^2", "standardized_unit": "g",
                       "parent_signal": "accelerometer"}}
    out = _Importer().convert_acceleration_to_standard_unit(df, metadata)
    assert out["ax"].tolist() == pytest.approx([1.0, -1.0, 0.5])
    assert metadata["ax"]["unit"] == "g"


def test_metadata_unit_updated_to_achieved_unit():
    df, metadata = _frame_and_metadata(target="m/s^2")
    _Importer().convert_acceleration_to_standard_unit(df, metadata)
    for axis in ("ax", "ay", "az"):
        assert metadata[axis]["unit"] == "m/s^2"
        assert metadata[axis]["standardized_unit"] == "m/s^2"


@pytest.mark.parametrize("target", ["g", "m/s^2"])
def test_conversion_is_idempotent(target):
    """A second pass must not scale by gravity again, in either direction."""
    df, metadata = _frame_and_metadata(unit="m/s^2" if target == "g" else "g", target=target)
    importer = _Importer()
    once = importer.convert_acceleration_to_standard_unit(df, metadata)
    twice = importer.convert_acceleration_to_standard_unit(once, metadata)
    pd.testing.assert_frame_equal(once, twice)


def test_channels_already_in_ms2_are_untouched():
    df, metadata = _frame_and_metadata(unit="m/s^2", target="m/s^2")
    out = _Importer().convert_acceleration_to_standard_unit(df.copy(), metadata)
    pd.testing.assert_frame_equal(out, df)


def test_non_accelerometer_signals_are_untouched():
    df = pd.DataFrame({"light": [100.0, 200.0], "depth": [5.0, 10.0]})
    metadata = {
        "light": {"unit": "count", "standardized_unit": "lux", "parent_signal": "light"},
        "depth": {"unit": "m", "standardized_unit": "m", "parent_signal": "depth"},
    }
    out = _Importer().convert_acceleration_to_standard_unit(df.copy(), metadata)
    pd.testing.assert_frame_equal(out, df)


def test_no_conversion_when_target_is_not_ms2():
    """If the montage wants g, values in g must stay in g."""
    df, metadata = _frame_and_metadata(unit="g", target="g")
    out = _Importer().convert_acceleration_to_standard_unit(df.copy(), metadata)
    pd.testing.assert_frame_equal(out, df)
    assert metadata["ax"]["unit"] == "g"


def test_nan_values_survive_conversion():
    df, metadata = _frame_and_metadata(target="m/s^2")
    out = _Importer().convert_acceleration_to_standard_unit(df, metadata)
    assert bool(out["ax"].isna().iloc[3])


def test_derived_accel_parent_signals_are_converted():
    for parent in ("corrected_acc", "dynamic_accel", "calibrated_acc"):
        df, metadata = _frame_and_metadata(parent=parent, target="m/s^2")
        out = _Importer().convert_acceleration_to_standard_unit(df, metadata)
        assert out["ax"].iloc[0] == pytest.approx(G), parent


def test_missing_columns_are_skipped_not_raised():
    df = pd.DataFrame({"ax": [1.0]})
    metadata = {
        "ax": {"unit": "g", "standardized_unit": "m/s^2", "parent_signal": "accelerometer"},
        "ay": {"unit": "g", "standardized_unit": "m/s^2", "parent_signal": "accelerometer"},
    }
    out = _Importer().convert_acceleration_to_standard_unit(df, metadata)
    assert out["ax"].iloc[0] == pytest.approx(G)


def test_empty_metadata_returns_frame_unchanged():
    df = pd.DataFrame({"ax": [1.0]})
    out = _Importer().convert_acceleration_to_standard_unit(df.copy(), {})
    pd.testing.assert_frame_equal(out, df)


def test_g_is_the_project_standard_in_montage():
    """Every accelerometer channel in montage_log.json must target g."""
    import json
    from pathlib import Path

    montage = json.loads(
        (Path(__file__).resolve().parents[1] / "montage_log.json").read_text()
    )
    offenders = []
    for manufacturer, montages in montage.items():
        for name, channels in montages.items():
            if not isinstance(channels, dict):
                continue
            for channel, meta in channels.items():
                if not isinstance(meta, dict):
                    continue
                if meta.get("parent_signal") in (
                    "accelerometer", "corrected_acc", "dynamic_accel", "calibrated_acc"
                ) and meta.get("standardized_unit") != "g":
                    offenders.append(f"{manufacturer}/{name}/{channel}")
    assert offenders == []


def test_gravity_magnitude_becomes_about_9_81():
    """
    End-to-end sanity check: a tag at rest reads ~1 g, which must become
    ~9.81 m/s^2 -- the property that makes ODBA physically meaningful.
    """
    n = 100
    df = pd.DataFrame({
        "ax": np.full(n, 0.9), "ay": np.full(n, 0.1), "az": np.full(n, -0.4),
    })
    metadata = {
        axis: {"unit": "g", "standardized_unit": "m/s^2", "parent_signal": "accelerometer"}
        for axis in ("ax", "ay", "az")
    }
    before = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2).mean()
    out = _Importer().convert_acceleration_to_standard_unit(df, metadata)
    after = np.sqrt(out["ax"] ** 2 + out["ay"] ** 2 + out["az"] ** 2).mean()
    assert before == pytest.approx(0.9899, abs=1e-3)
    assert after == pytest.approx(before * G, rel=1e-9)


def test_csv_importer_hook_applies_conversion():
    """The CSVImporter transform hook must call the conversion."""
    from pyologger.io_operations.csv_importer import CSVImporter

    class _CSV(CSVImporter):
        def __init__(self):
            pass

    df, metadata = _frame_and_metadata(target="m/s^2")
    out = _CSV().apply_post_rename_transforms(df, metadata)
    assert out["ax"].iloc[0] == pytest.approx(G)


def test_subclass_overrides_still_chain_to_conversion():
    """
    PDImporter and StarOddiImporter override the hook; they must call super()
    so acceleration standardization is not silently skipped.
    """
    import inspect

    from pyologger.io_operations.pd_importer import PDImporter
    from pyologger.io_operations.star_oddi_importer import StarOddiImporter

    for cls in (PDImporter, StarOddiImporter):
        source = inspect.getsource(cls.apply_post_rename_transforms)
        assert "super().apply_post_rename_transforms" in source, cls.__name__

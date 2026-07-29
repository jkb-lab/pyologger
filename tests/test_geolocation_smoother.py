import numpy as np
import pandas as pd

from pyologger.utils.geolocation_smoother import (
    EQUINOX_LAT_INFLATION,
    days_from_equinox,
    latitude_sigma,
    regularize_track,
    smooth_geolocation_track,
)


def _track(n=48, start="2021-08-03 17:38:00", step_hours=3, noise=0.0, seed=0):
    times = pd.date_range(start, periods=n, freq=f"{step_hours}h")
    rng = np.random.default_rng(seed)
    lat = 26.9 + np.linspace(0, 0.5, n) + rng.normal(0, noise, n)
    lon = -79.9 - np.linspace(0, 0.5, n) + rng.normal(0, noise, n)
    return pd.DataFrame({"datetime": times, "latitude": lat, "longitude": lon})


def test_days_from_equinox_zero_on_equinox():
    s = pd.Series(pd.to_datetime(["2021-03-20", "2021-09-22"]))
    assert list(days_from_equinox(s)) == [0.0, 0.0]


def test_days_from_equinox_searches_adjacent_years():
    # 2021-01-02 is 77 days before the March 2021 equinox and 102 days after
    # the September 2020 one, so the nearest is 77 -- which requires scanning
    # neighbouring years, not just the timestamp's own.
    value = days_from_equinox(pd.Series(pd.to_datetime(["2021-01-02"]))).iloc[0]
    assert value == 77.0

    # A late-December date must find the *following* year's March equinox.
    dec = days_from_equinox(pd.Series(pd.to_datetime(["2020-12-28"]))).iloc[0]
    assert dec == 82.0


def test_latitude_sigma_inflated_near_equinox():
    s = pd.Series(pd.to_datetime(["2021-03-21", "2021-06-21"]))
    sigma = latitude_sigma(s, base_sigma_deg=2.0)
    assert sigma.iloc[0] == 2.0 * EQUINOX_LAT_INFLATION
    assert sigma.iloc[1] == 2.0


def test_smoothing_reduces_point_to_point_variability():
    # Noise scale matches real GPE3 exports (~0.04-0.05 deg between fixes);
    # larger values imply swim speeds the outlier filter correctly rejects.
    df = _track(noise=0.04, seed=3)
    out = smooth_geolocation_track(df, window_hours=24.0)
    assert out["latitude_smoothed"].notna().all()
    assert out["longitude_smoothed"].notna().all()
    # Smoothing must reduce point-to-point variability for both axes.
    assert out["latitude_smoothed"].diff().std() < out["latitude"].diff().std()
    assert out["longitude_smoothed"].diff().std() < out["longitude"].diff().std()


def test_smoothing_preserves_original_columns():
    df = _track(noise=0.1)
    out = smooth_geolocation_track(df)
    # Raw positions must survive so smoothing can be audited.
    pd.testing.assert_series_equal(out["latitude"], df["latitude"], check_names=False)
    pd.testing.assert_series_equal(out["longitude"], df["longitude"], check_names=False)


def test_speed_outlier_flagged_and_excluded():
    df = _track(n=12, noise=0.0)
    # Teleport one fix ~1000 km away; at 3 h spacing that is >>10 km/h.
    df.loc[6, "latitude"] = df.loc[6, "latitude"] + 9.0
    out = smooth_geolocation_track(df, max_speed_kmh=10.0)
    assert bool(out.loc[6, "outlier"]) is True
    assert out.loc[6, ["latitude_smoothed", "longitude_smoothed"]].isna().all()
    # The wild fix must not drag its neighbours far from the true path.
    assert abs(out.loc[7, "latitude_smoothed"] - df.loc[7, "latitude"]) < 1.0


def test_single_outlier_does_not_cascade():
    df = _track(n=10, noise=0.0)
    df.loc[4, "longitude"] = df.loc[4, "longitude"] - 9.0
    out = smooth_geolocation_track(df, max_speed_kmh=10.0)
    # Only the wild fix is flagged; comparison is against last accepted fix.
    assert out["outlier"].sum() == 1


def test_empty_input_returns_empty_frame():
    out = smooth_geolocation_track(pd.DataFrame())
    assert out.empty
    for col in ("latitude_smoothed", "longitude_smoothed", "outlier"):
        assert col in out.columns


def test_too_few_clean_fixes_passes_positions_through():
    df = _track(n=1)
    out = smooth_geolocation_track(df)
    assert out.loc[0, "latitude_smoothed"] == df.loc[0, "latitude"]


def test_regularize_track_uses_fixed_step():
    df = _track(n=24, step_hours=2, noise=0.05)
    out = regularize_track(smooth_geolocation_track(df), step="3h")
    gaps = out["datetime"].diff().dropna().unique()
    assert len(gaps) == 1
    assert gaps[0] == pd.Timedelta("3h")
    assert out["latitude"].notna().all()
    assert out["longitude"].notna().all()


def test_regularize_track_handles_unsorted_input():
    df = _track(n=16, noise=0.05).sample(frac=1.0, random_state=1)
    out = regularize_track(smooth_geolocation_track(df), step="3h")
    assert out["datetime"].is_monotonic_increasing


def test_irregular_spacing_is_weighted_in_time_domain():
    # Two clusters far apart in time: smoothing must not average across the gap.
    times = list(pd.date_range("2021-08-01", periods=5, freq="3h"))
    times += list(pd.date_range("2021-09-01", periods=5, freq="3h"))
    df = pd.DataFrame({
        "datetime": times,
        "latitude": [26.0] * 5 + [30.0] * 5,
        "longitude": [-80.0] * 5 + [-80.0] * 5,
    })
    out = smooth_geolocation_track(df, window_hours=24.0, max_speed_kmh=None)
    assert out.loc[0, "latitude_smoothed"] < 27.0
    assert out.loc[9, "latitude_smoothed"] > 29.0

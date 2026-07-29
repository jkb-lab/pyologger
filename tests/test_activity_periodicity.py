import numpy as np
import pandas as pd
import pytest

from pyologger.analyze_data.activity_periodicity import (
    compute_activity_from_acc,
    correlate_temperature_activity,
    dominant_periods,
    estimate_sampling_rate_hz,
    resample_regular,
    wavelet_power_spectrum,
)

# MiniPAT sampling: 3 s interval == 1/3 Hz.
FS_HZ = 1.0 / 3.0


def _acc_frame(hours=48, period_hours=24.0, amplitude=0.5, seed=0):
    n = int(hours * 3600 * FS_HZ)
    dt = pd.date_range("2021-08-03 17:34:21", periods=n, freq="3s", tz="UTC")
    t_hours = np.arange(n) / (3600.0 * FS_HZ)
    rng = np.random.default_rng(seed)
    # Slow postural drift plus a diel-modulated jitter component.
    jitter = rng.normal(0, 0.05, n) * (1 + amplitude * np.sin(2 * np.pi * t_hours / period_hours))
    return pd.DataFrame({
        "datetime": dt,
        "ax": 0.9 + 0.1 * np.sin(2 * np.pi * t_hours / 12.0) + jitter,
        "ay": 0.1 + jitter * 0.5,
        "az": -0.2 + jitter * 0.5,
    })


def test_estimate_sampling_rate_matches_3_second_interval():
    df = _acc_frame(hours=4)
    assert estimate_sampling_rate_hz(df["datetime"]) == pytest.approx(FS_HZ, rel=1e-6)


def test_activity_window_is_multi_sample_at_low_fs():
    """
    Regression: the pipeline's 2 s window rounds to 1 sample at 1/3 Hz, making
    dynamic acceleration identically zero. An hours-based window must not.
    """
    out = compute_activity_from_acc(_acc_frame(hours=12), window_hours=1.0)
    assert out.attrs["static_window_samples"] > 100
    assert out.attrs["static_window_samples"] % 2 == 1  # centred, so odd


def test_activity_is_not_degenerately_zero():
    out = compute_activity_from_acc(_acc_frame(hours=12), window_hours=1.0)
    activity = out["activity"].dropna()
    assert activity.std() > 0
    assert (activity > 0).mean() > 0.9


def test_activity_rejects_too_short_window():
    # 2 s at 1/3 Hz is under 3 samples and must be refused, not silently used.
    with pytest.raises(ValueError, match="choose a longer window"):
        compute_activity_from_acc(_acc_frame(hours=4), window_hours=2.0 / 3600.0)


def test_activity_requires_expected_columns():
    df = _acc_frame(hours=2).drop(columns=["az"])
    with pytest.raises(ValueError, match="missing columns"):
        compute_activity_from_acc(df)


def test_activity_all_nan_rows_stay_nan():
    df = _acc_frame(hours=6)
    df.loc[100:200, ["ax", "ay", "az"]] = np.nan
    out = compute_activity_from_acc(df, window_hours=1.0)
    assert out["activity"].isna().any()


def test_odba_and_vedba_methods_differ():
    df = _acc_frame(hours=8)
    vedba = compute_activity_from_acc(df, window_hours=1.0, method="vedba")["activity"]
    odba = compute_activity_from_acc(df, window_hours=1.0, method="odba")["activity"]
    # 1-norm >= 2-norm for the same vector.
    assert odba.dropna().mean() >= vedba.dropna().mean()


def test_unknown_activity_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        compute_activity_from_acc(_acc_frame(hours=2), window_hours=1.0, method="bogus")


def test_resample_regular_produces_uniform_grid():
    out = resample_regular(_acc_frame(hours=10), "ax", rule="1h")
    gaps = pd.Series(out["datetime"]).diff().dropna().unique()
    assert len(gaps) == 1
    assert gaps[0] == pd.Timedelta("1h")


def test_wavelet_recovers_known_24h_period():
    """A pure 24 h oscillation must peak near 24 h in period space."""
    n = int(60 * 24 * 40)  # 40 days at 1-minute spacing
    dt = pd.date_range("2021-08-03", periods=n, freq="1min", tz="UTC")
    t_hours = np.arange(n) / 60.0
    df = pd.DataFrame({"datetime": dt, "activity": np.sin(2 * np.pi * t_hours / 24.0)})

    spectrum = wavelet_power_spectrum(
        df, "activity", rule="30min", min_period_hours=4, max_period_hours=24 * 10
    )
    peak_period = spectrum["periods_hours"][int(np.argmax(spectrum["global_power"]))]
    assert peak_period == pytest.approx(24.0, rel=0.15)


def test_wavelet_labels_known_cycles():
    n = int(60 * 24 * 40)
    dt = pd.date_range("2021-08-03", periods=n, freq="1min", tz="UTC")
    t_hours = np.arange(n) / 60.0
    df = pd.DataFrame({"datetime": dt, "activity": np.sin(2 * np.pi * t_hours / 24.0)})
    spectrum = wavelet_power_spectrum(
        df, "activity", rule="30min", min_period_hours=4, max_period_hours=24 * 10
    )
    table = dominant_periods(spectrum, top_n=3)
    assert "24 h (diel)" in set(table["nearest_known_cycle"])


def test_wavelet_reports_cone_of_influence():
    spectrum = wavelet_power_spectrum(
        _acc_frame(hours=24 * 20).assign(activity=lambda d: d["ax"]),
        "activity", rule="1h",
    )
    assert spectrum["cone_of_influence_hours"] > 0
    assert "within_cone_of_influence" in dominant_periods(spectrum).columns


def test_wavelet_rejects_too_few_samples():
    df = pd.DataFrame({
        "datetime": pd.date_range("2021-08-03", periods=5, freq="1h", tz="UTC"),
        "activity": np.arange(5, dtype=float),
    })
    with pytest.raises(ValueError, match="finite samples"):
        wavelet_power_spectrum(df, "activity", rule="1h")


def test_correlation_detects_known_positive_relationship():
    df = _acc_frame(hours=24 * 15)
    activity = compute_activity_from_acc(df, window_hours=1.0)
    # Build temperature as a noisy linear function of activity.
    hourly = resample_regular(activity, "activity", rule="1h")
    rng = np.random.default_rng(1)
    temp = pd.DataFrame({
        "datetime": hourly["datetime"],
        "temp_ext": 25.0 + 40.0 * hourly["activity"] + rng.normal(0, 0.05, len(hourly)),
    })
    res = correlate_temperature_activity(temp, activity, rule="1h")
    assert res["pearson_r"] > 0.5
    assert res["pearson_p"] < 0.01


def test_correlation_lag_curve_is_symmetric_range():
    df = _acc_frame(hours=24 * 10)
    activity = compute_activity_from_acc(df, window_hours=1.0)
    temp = resample_regular(activity, "activity", rule="1h").rename(
        columns={"activity": "temp_ext"}
    )
    res = correlate_temperature_activity(temp, activity, rule="1h", max_lag_hours=12)
    lags = res["lag_curve"]["lag_hours"].to_numpy()
    assert lags.min() == pytest.approx(-12.0)
    assert lags.max() == pytest.approx(12.0)
    # Identical series must correlate best at zero lag.
    assert res["best_lag_hours"] == pytest.approx(0.0)


def test_correlation_requires_enough_overlap():
    a = pd.DataFrame({
        "datetime": pd.date_range("2021-08-03", periods=3, freq="1h", tz="UTC"),
        "activity": [1.0, 2.0, 3.0],
    })
    t = pd.DataFrame({
        "datetime": pd.date_range("2022-01-01", periods=3, freq="1h", tz="UTC"),
        "temp_ext": [10.0, 11.0, 12.0],
    })
    with pytest.raises(ValueError, match="aligned bins"):
        correlate_temperature_activity(t, a, rule="1h")


# --------------------------------------------------------------------------
# Wildlife Computers "Series" aggregation
# --------------------------------------------------------------------------

from pyologger.analyze_data.activity_periodicity import (  # noqa: E402
    WC_SERIES_INTERVAL_SECONDS,
    compute_series_aggregates,
)


def test_wc_series_interval_is_450_seconds():
    """7.5 min, derived from the vendor export's own timestamp spacing."""
    assert WC_SERIES_INTERVAL_SECONDS == 450.0


def test_series_activity_is_mean_vector_magnitude_including_gravity():
    """
    Vendor Activity = mean(||a||) over the bin, gravity included. A constant
    1 g on one axis must therefore yield exactly 1.0, not 0.
    """
    n = 150
    df = pd.DataFrame({
        "datetime": pd.date_range("2021-08-03 17:30:00", periods=n, freq="3s", tz="UTC"),
        "ax": np.ones(n),
        "ay": np.zeros(n),
        "az": np.zeros(n),
    })
    out = compute_series_aggregates(df, interval="450s")
    assert len(out) == 1
    assert out["activity"].iloc[0] == pytest.approx(1.0)
    assert out["activity_range"].iloc[0] == pytest.approx(0.0)


def test_series_activity_range_is_peak_to_peak():
    n = 150
    mag = np.full(n, 1.0)
    mag[10] = 1.5
    mag[20] = 0.5
    df = pd.DataFrame({
        "datetime": pd.date_range("2021-08-03 17:30:00", periods=n, freq="3s", tz="UTC"),
        "ax": mag, "ay": np.zeros(n), "az": np.zeros(n),
    })
    out = compute_series_aggregates(df, interval="450s")
    assert out["activity_range"].iloc[0] == pytest.approx(1.0)


def test_series_bins_are_left_closed_and_labelled_by_start():
    n = 300  # exactly two 450 s bins at 3 s
    df = pd.DataFrame({
        "datetime": pd.date_range("2021-08-03 17:30:00", periods=n, freq="3s", tz="UTC"),
        "ax": np.ones(n), "ay": np.zeros(n), "az": np.zeros(n),
    })
    out = compute_series_aggregates(df, interval="450s")
    assert len(out) == 2
    assert out["n_samples"].tolist() == [150, 150]
    assert pd.Timestamp(out["datetime"].iloc[0]) == pd.Timestamp("2021-08-03 17:30:00", tz="UTC")
    assert pd.Timestamp(out["datetime"].iloc[1]) == pd.Timestamp("2021-08-03 17:37:30", tz="UTC")


def test_series_interval_is_configurable():
    n = 1200  # 1 hour at 3 s
    df = pd.DataFrame({
        "datetime": pd.date_range("2021-08-03 17:00:00", periods=n, freq="3s", tz="UTC"),
        "ax": np.ones(n), "ay": np.zeros(n), "az": np.zeros(n),
    })
    assert len(compute_series_aggregates(df, interval="450s")) == 8
    assert len(compute_series_aggregates(df, interval="30min")) == 2
    assert len(compute_series_aggregates(df, interval="1h")) == 1
    # Numeric seconds are accepted as well as offset strings.
    assert len(compute_series_aggregates(df, interval=1800)) == 2


def test_series_accepts_depth_and_temperature():
    n = 150
    df = pd.DataFrame({
        "datetime": pd.date_range("2021-08-03 17:30:00", periods=n, freq="3s", tz="UTC"),
        "ax": np.ones(n), "ay": np.zeros(n), "az": np.zeros(n),
    })
    depth = pd.DataFrame({"datetime": df["datetime"], "depth": np.linspace(0, 100, n)})
    temp = pd.DataFrame({"datetime": df["datetime"], "temp_ext": np.linspace(20, 30, n)})
    out = compute_series_aggregates(df, interval="450s", depth_df=depth, temperature_df=temp)
    assert out["depth"].iloc[0] == pytest.approx(50.0)
    assert out["depth_range"].iloc[0] == pytest.approx(100.0)
    assert out["temperature"].iloc[0] == pytest.approx(25.0)
    assert out["temperature_range"].iloc[0] == pytest.approx(10.0)


def test_series_centre_labelling_shifts_by_half_bin():
    n = 150
    df = pd.DataFrame({
        "datetime": pd.date_range("2021-08-03 17:30:00", periods=n, freq="3s", tz="UTC"),
        "ax": np.ones(n), "ay": np.zeros(n), "az": np.zeros(n),
    })
    out = compute_series_aggregates(df, interval="450s", label=False)
    assert pd.Timestamp(out["datetime"].iloc[0]) == pd.Timestamp("2021-08-03 17:33:45", tz="UTC")


def test_series_requires_accelerometer_columns():
    df = pd.DataFrame({"datetime": pd.date_range("2021-08-03", periods=5, freq="3s"), "ax": 1.0})
    with pytest.raises(ValueError, match="missing columns"):
        compute_series_aggregates(df)


def test_series_rejects_missing_named_depth_column():
    n = 150
    df = pd.DataFrame({
        "datetime": pd.date_range("2021-08-03 17:30:00", periods=n, freq="3s", tz="UTC"),
        "ax": np.ones(n), "ay": np.zeros(n), "az": np.zeros(n),
    })
    with pytest.raises(ValueError, match="Expected column"):
        compute_series_aggregates(df, depth_df=pd.DataFrame({"datetime": df["datetime"]}))


def test_series_activity_differs_from_gravity_removed_activity():
    """
    The two metrics are not interchangeable: vendor Activity sits near 1 g,
    the recomputed dynamic index near 0.
    """
    df = _acc_frame(hours=6)
    vendor = compute_series_aggregates(df, interval="450s")["activity"].mean()
    dynamic = compute_activity_from_acc(df, window_hours=1.0)["activity"].mean()
    assert vendor > 0.5
    assert dynamic < 0.5

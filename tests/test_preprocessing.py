"""Unit tests for the preprocessing module."""
import io

import numpy as np
import pandas as pd
import pytest
from src.preprocessing import load_light_curve, parse_light_curve_frame


def test_load_light_curve_detrends_known_dip():
    """Verify that light curve detrending flattens baseline flux around 1.0."""
    time = np.linspace(0, 10, 1000)
    # Flux with a slow parabolic trend and a transit dip
    trend = 0.05 * (time - 5) ** 2
    flux = 1.0 + trend
    flux[480:520] -= 0.02

    lc = load_light_curve(time, flux, window_length=101)

    assert lc.flux.value.mean() == pytest.approx(1.0, abs=0.01)
    assert len(lc.time) == len(time)


def test_load_light_curve_handles_nans_and_sorting():
    """Verify that NaNs are dropped and out-of-order timestamps are sorted."""
    time = np.array([2.0, 1.0, 3.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    flux = np.array([1.0, 1.0, np.nan, 0.98, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    lc = load_light_curve(time, flux)

    assert np.all(np.diff(lc.time.value) > 0)  # strictly increasing
    assert not np.any(np.isnan(lc.flux.value))


def test_load_light_curve_length_mismatch():
    """Verify ValueError when time and flux dimensions do not match."""
    with pytest.raises(ValueError, match="Length mismatch"):
        load_light_curve(np.array([1.0, 2.0]), np.array([1.0]))


def test_load_light_curve_all_nan_rejected():
    """A flux column of only NaNs is rejected with a clear message."""
    t = np.arange(0, 30, 0.02)
    with pytest.raises(ValueError, match="fewer than 10 valid"):
        load_light_curve(t, np.full_like(t, np.nan))


# ---- parse_light_curve_frame (CSV validation used by the app) ----

def test_parse_frame_case_insensitive_columns():
    df = pd.DataFrame({" Time ": [0.0, 1.0, 2.0], "FLUX": [1.0, 0.99, 1.0]})
    time, flux = parse_light_curve_frame(df)
    assert time.tolist() == [0.0, 1.0, 2.0]
    assert flux.tolist() == [1.0, 0.99, 1.0]


def test_parse_frame_missing_columns():
    df = pd.DataFrame({"t": [0.0, 1.0], "brightness": [1.0, 1.0]})
    with pytest.raises(ValueError, match="'time' and 'flux'"):
        parse_light_curve_frame(df)


def test_parse_frame_header_only():
    df = pd.read_csv(io.StringIO("time,flux\n"))
    with pytest.raises(ValueError, match="no data rows"):
        parse_light_curve_frame(df)


def test_parse_frame_non_numeric_values():
    df = pd.DataFrame({"time": ["a", "b", "c"], "flux": ["x", "y", "z"]})
    with pytest.raises(ValueError, match="numeric"):
        parse_light_curve_frame(df)


def test_parse_frame_coerces_bad_cells_to_nan():
    df = pd.DataFrame({"time": ["0", "1", "oops"], "flux": [1.0, "bad", 1.0]})
    time, flux = parse_light_curve_frame(df)
    assert np.isnan(time[2]) and np.isnan(flux[1])

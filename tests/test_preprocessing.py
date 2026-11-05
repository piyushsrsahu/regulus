"""Unit tests for the preprocessing module."""
import numpy as np
import pytest
from src.preprocessing import load_light_curve


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

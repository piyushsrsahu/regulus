"""Unit tests for feature extraction routines."""
import numpy as np
import pytest
from src.preprocessing import load_light_curve
from src.features import extract_features


def test_extract_features_finds_known_transit():
    """Verify that BLS periodogram correctly recovers known periodic transit parameters."""
    t = np.arange(0, 20, 0.02)
    flux = np.ones_like(t)

    # Injected transit: period = 4.0 days, depth = 0.02, duration = 0.2 days
    period = 4.0
    t0 = 1.0
    duration = 0.2
    depth = 0.02
    phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
    flux[np.abs(phase) < (duration / 2.0)] -= depth

    lc = load_light_curve(t, flux)
    features = extract_features(lc, min_period=1.0, max_period=8.0)

    assert "period" in features
    assert "depth" in features
    assert "duration" in features
    assert "bls_power" in features

    assert features["period"] == pytest.approx(4.0, abs=0.1)
    assert features["depth"] == pytest.approx(0.02, abs=0.005)
    assert features["bls_power"] > 0.0


def test_extract_features_recovers_period_on_long_baseline():
    """A 5-day transit in a 200-day series must not be reported as a harmonic (e.g. 25 days)."""
    rng = np.random.default_rng(7)
    t = np.arange(0, 200, 0.02)
    flux = 1.0 + rng.normal(0, 1e-3, len(t))
    phase = ((t - 1.0 + 2.5) % 5.0) - 2.5
    flux[np.abs(phase) < 0.1] -= 0.01

    features = extract_features(load_light_curve(t, flux))

    assert features["period"] == pytest.approx(5.0, abs=0.02)
    assert features["depth"] == pytest.approx(0.01, abs=0.003)


def test_extract_features_zero_variance_flux():
    """A perfectly flat light curve yields zero depth/power instead of crashing."""
    t = np.arange(0, 30, 0.02)
    features = extract_features(load_light_curve(t, np.ones_like(t)))

    assert features["depth"] == pytest.approx(0.0, abs=1e-9)
    assert features["bls_power"] == pytest.approx(0.0, abs=1e-9)


def test_extract_features_short_series():
    """A 12-point light curve (the minimum regime) still produces a feature dict."""
    rng = np.random.default_rng(3)
    t = np.arange(12, dtype=float)
    features = extract_features(load_light_curve(t, 1.0 + rng.normal(0, 1e-3, 12)))

    assert set(features) >= {"period", "depth", "duration", "bls_power", "transit_time"}

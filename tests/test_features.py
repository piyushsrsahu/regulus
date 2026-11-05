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

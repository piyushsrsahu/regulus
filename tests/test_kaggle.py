"""Tests for the Kaggle Kepler preprocessing path and the scale-free features it relies on."""
import numpy as np
import pandas as pd
import pytest

from src import PROJECT_ROOT
from src.features import extract_features
from src.preprocessing import load_light_curve
from src.model import predict
from train_model import KEPLER_CADENCE_DAYS, kaggle_row_to_relative_flux

KAGGLE_TEST = PROJECT_ROOT / "data" / "raw" / "exoTest.csv"


def _kaggle_like_row(seed=0, period=4.0, duration=0.25, snr=25.0, sd=80.0):
    """De-meaned flux in arbitrary units, like the Kaggle CSVs, with spikes and one transit train."""
    rng = np.random.default_rng(seed)
    t = np.arange(3197) * KEPLER_CADENCE_DAYS
    f = rng.normal(0, sd, len(t)) + 3 * sd * np.sin(2 * np.pi * t / 30.0)
    f[[100, 900, 2500]] += 40 * sd
    phase = ((t - 1.0 + period / 2) % period) - period / 2
    m = np.abs(phase) < duration / 2
    f[m] -= snr * sd / np.sqrt(m.sum())
    return t, f - f.mean()


def test_relative_flux_has_positive_unit_baseline():
    _, raw = _kaggle_like_row()
    assert raw.min() < 0  # the original bug: lightkurve divides by a trend near zero
    rel = kaggle_row_to_relative_flux(raw)
    finite = rel[np.isfinite(rel)]
    assert np.all(finite > 0)
    assert np.median(finite) == pytest.approx(1.0, abs=1e-3)
    assert np.isnan(rel[[100, 900, 2500]]).all()  # upward spikes clipped


def test_kaggle_like_transit_gives_sane_features():
    t, raw = _kaggle_like_row(period=4.0)
    feats = extract_features(load_light_curve(t, kaggle_row_to_relative_flux(raw)))
    assert feats["period"] == pytest.approx(4.0, abs=0.05)
    assert 0 < feats["depth"] < 0.05  # was ~1098 before the fix
    assert feats["depth_snr"] > 10
    assert feats["bls_sde"] > 5


def test_scale_free_features_ignore_flux_units():
    """Multiplying the de-meaned flux by 50 (different units) must not change the model inputs."""
    t, raw = _kaggle_like_row(seed=3)
    a = extract_features(load_light_curve(t, kaggle_row_to_relative_flux(raw)))
    b = extract_features(load_light_curve(t, kaggle_row_to_relative_flux(raw * 50)))
    for key in ("period", "duration", "depth_snr", "bls_sde", "box_vs_sine"):
        assert a[key] == pytest.approx(b[key], rel=0.02)


@pytest.mark.skipif(not KAGGLE_TEST.exists(), reason="Kaggle exoTest.csv not downloaded into data/raw/")
def test_real_exotest_rows_run_end_to_end():
    df = pd.read_csv(KAGGLE_TEST, nrows=10)  # the first 5 rows are the confirmed planet hosts
    flux = df[[c for c in df.columns if c.startswith("FLUX")]].to_numpy(dtype=float)
    t = np.arange(flux.shape[1]) * KEPLER_CADENCE_DAYS
    for row in flux:
        result = predict(extract_features(load_light_curve(t, kaggle_row_to_relative_flux(row))))
        assert result["prediction"] in ("exoplanet", "no_exoplanet")
        assert 0.0 <= result["confidence"] <= 1.0
        assert np.isfinite(result["depth_snr"])

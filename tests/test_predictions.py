"""Functional tests evaluating model classification against the bundled benchmark samples.

The bundled samples are synthetic light curves at Kepler-like (~30 min) cadence:
one with an injected periodic transit, a quiet star, and a pulsating variable star.
"""
import pandas as pd
import pytest

from src import PROJECT_ROOT
from src.preprocessing import load_light_curve
from src.features import extract_features
from src.model import predict

SAMPLES = PROJECT_ROOT / "data" / "sample_light_curves"


def _classify(name: str) -> dict:
    df = pd.read_csv(SAMPLES / name)
    lc = load_light_curve(df["time"].values, df["flux"].values)
    return predict(extract_features(lc))


def test_known_positive_sample_classified_correctly():
    """The injected-transit sample is classified as an exoplanet candidate."""
    result = _classify("sample_kepler_transit.csv")
    assert result["prediction"] == "exoplanet"
    assert result["confidence"] >= 0.5
    assert result["period"] == pytest.approx(3.5, abs=0.05)


def test_known_negative_sample_classified_correctly():
    """The quiet-star sample is classified as no exoplanet."""
    result = _classify("sample_quiet_star.csv")
    assert result["prediction"] == "no_exoplanet"
    assert result["confidence"] >= 0.5


def test_variable_star_sample_classified_negative():
    """Sinusoidal stellar pulsation must not be mistaken for a transit."""
    result = _classify("sample_variable_star.csv")
    assert result["prediction"] == "no_exoplanet"
    assert result["confidence"] >= 0.5

"""Functional tests evaluating model classification against benchmark dataset samples."""
import pandas as pd
from src.preprocessing import load_light_curve
from src.features import extract_features
from src.model import predict


def test_known_positive_sample_classified_correctly():
    """Verify that a confirmed exoplanet transit light curve is classified as positive."""
    df = pd.read_csv("data/sample_light_curves/sample_kepler_transit.csv")
    lc = load_light_curve(df["time"].values, df["flux"].values)
    features = extract_features(lc)
    result = predict(features)

    assert result["prediction"] == "exoplanet"
    assert result["confidence"] >= 0.5


def test_known_negative_sample_classified_correctly():
    """Verify that a non-transit stellar light curve is classified as negative."""
    df = pd.read_csv("data/sample_light_curves/sample_quiet_star.csv")
    lc = load_light_curve(df["time"].values, df["flux"].values)
    features = extract_features(lc)
    result = predict(features)

    assert result["prediction"] == "no_exoplanet"
    assert result["confidence"] >= 0.5

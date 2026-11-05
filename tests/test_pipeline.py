"""Integration tests for the end-to-end processing, prediction, and database pipeline."""
import numpy as np
from src.preprocessing import load_light_curve
from src.features import extract_features
from src.model import predict
from src.db import init_db, save_analysis, load_history


def test_full_pipeline_produces_well_formed_result():
    """Verify that end-to-end data flow produces a valid prediction structure and persists to DB."""
    init_db()

    time = np.linspace(0, 10, 1000)
    flux = np.ones_like(time)
    flux[480:520] -= 0.02

    lc = load_light_curve(time, flux)
    features = extract_features(lc)
    result = predict(features)

    assert result["prediction"] in ("exoplanet", "no_exoplanet")
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["depth"] > 0.0
    assert result["period"] > 0.0

    # Test database logging and retrieval
    save_analysis("test_run_sample.csv", result)
    history = load_history()

    assert len(history) > 0
    latest = history[0]
    assert latest["filename"] == "test_run_sample.csv"
    assert latest["prediction"] == result["prediction"]

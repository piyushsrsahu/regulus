"""
Inference module for evaluating candidate light curve features.
"""
import os
import joblib
import pandas as pd

MODEL_PATH = "models/random_forest.joblib"
DEFAULT_FEATURES = ["period", "depth", "duration", "bls_power"]

_model_bundle = None


def _get_model_bundle():
    global _model_bundle
    if _model_bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. Please run `python train_model.py` first."
            )
        data = joblib.load(MODEL_PATH)
        if isinstance(data, dict) and "model" in data:
            _model_bundle = data
        else:
            _model_bundle = {"model": data, "feature_names": DEFAULT_FEATURES}
    return _model_bundle


def predict(features: dict) -> dict:
    """
    Generate an exoplanet transit prediction for a set of extracted features.

    Args:
        features: Dictionary containing extracted transit parameters (period, depth, duration, bls_power).

    Returns:
        dict: Prediction label ("exoplanet" | "no_exoplanet"), confidence score, and input features.
    """
    bundle = _get_model_bundle()
    model = bundle["model"]
    feature_names = bundle.get("feature_names", DEFAULT_FEATURES)

    # Construct DataFrame with matching feature names to avoid sklearn feature name warnings
    row_df = pd.DataFrame([{name: float(features.get(name, 0.0)) for name in feature_names}])

    # Predict probabilities and class label
    probs = model.predict_proba(row_df)[0]
    pred_idx = int(model.predict(row_df)[0])

    confidence = float(probs[pred_idx])
    label = "exoplanet" if pred_idx == 1 else "no_exoplanet"

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "period": float(features.get("period", 0.0)),
        "depth": float(features.get("depth", 0.0)),
        "duration": float(features.get("duration", 0.0)),
        "bls_power": float(features.get("bls_power", 0.0)),
        "transit_time": float(features.get("transit_time", 0.0)),
    }

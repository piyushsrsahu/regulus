"""
Model training script for the Kepler exoplanet transit classifier.
Trains a balanced Random Forest classifier on extracted BLS features and exports the model artifact.

Usage:
    python train_model.py
"""
import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing import load_light_curve
from src.features import extract_features

FEATURE_COLUMNS = ["period", "depth", "duration", "bls_power"]
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest.joblib")


def generate_synthetic_training_dataset(n_samples: int = 120) -> pd.DataFrame:
    """
    Generate synthetic photometric time-series samples spanning both transit candidates
    and non-transit stellar baselines with realistic noise and astrophysical variation.
    """
    rows = []
    rng = np.random.default_rng(42)

    for i in range(n_samples):
        t = np.arange(0, 30, 0.03)  # 30-day baseline, ~1,000 points
        noise_level = rng.uniform(0.0008, 0.003)
        drift_amplitude = rng.uniform(0.001, 0.01)
        drift_period = rng.uniform(5.0, 30.0)

        # Baseline flux with low-frequency drift and Gaussian noise
        flux = 1.0 + drift_amplitude * np.sin(2 * np.pi * t / drift_period) + rng.normal(0, noise_level, size=len(t))

        # 35% positives (transit present), 65% negatives (no transit / stellar variability)
        has_transit = (rng.random() < 0.35)

        if has_transit:
            period = rng.uniform(1.2, 8.5)
            t0 = rng.uniform(0.1, period)
            depth = rng.uniform(0.005, 0.035)  # 0.5% to 3.5% transit depth
            duration = rng.uniform(0.12, 0.35)

            phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
            in_transit = np.abs(phase) < (duration / 2.0)
            flux[in_transit] -= depth
            label = 1
        else:
            # Maybe add stellar pulsation / flares / quiet baseline
            if rng.random() < 0.4:
                # Variable / pulsating star
                pulsation_p = rng.uniform(0.5, 4.0)
                flux += rng.uniform(0.002, 0.008) * np.sin(2 * np.pi * t / pulsation_p)
            label = 0

        # Extract features through standard pipeline
        try:
            lc = load_light_curve(t, flux)
            feats = extract_features(lc, min_period=0.5, max_period=10.0)
            feats["label"] = label
            rows.append(feats)
        except Exception:
            continue

    return pd.DataFrame(rows)


def load_dataset() -> pd.DataFrame:
    """
    Attempt to load real Kepler dataset from data/raw/ if available,
    otherwise fallback to synthetic dataset generation.
    """
    raw_csvs = glob.glob("data/raw/*.csv")
    if raw_csvs:
        print(f"Found raw dataset in data/raw/: {raw_csvs[0]}. Processing...")
        # If Kaggle exoTrain format (FLUX.1...FLUX.N, LABEL)
        df_raw = pd.read_csv(raw_csvs[0])
        if "LABEL" in df_raw.columns:
            rows = []
            for _, row in df_raw.iterrows():
                label = 1 if row["LABEL"] == 2 else 0
                flux_cols = [c for c in df_raw.columns if c.startswith("FLUX.")]
                flux = row[flux_cols].values.astype(float)
                time = np.arange(len(flux)) * 0.0208  # Kepler 30-min cadence in days
                try:
                    lc = load_light_curve(time, flux)
                    feats = extract_features(lc)
                    feats["label"] = label
                    rows.append(feats)
                except Exception:
                    continue
            return pd.DataFrame(rows)

    print("No raw dataset found in data/raw/. Generating synthetic photometric dataset for training...")
    return generate_synthetic_training_dataset(n_samples=120)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Building training dataset...")
    df = load_dataset()
    print(f"Dataset ready: {len(df)} samples ({df['label'].sum()} transits, {(df['label'] == 0).sum()} non-transits).")

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    print(f"Training RandomForestClassifier on {len(X_train)} samples...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Test Set Evaluation ---")
    print(classification_report(y_test, y_pred, target_names=["No Exoplanet", "Exoplanet"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump({"model": model, "feature_names": FEATURE_COLUMNS}, MODEL_PATH)
    print(f"\nModel artifact saved successfully to {MODEL_PATH}")


if __name__ == "__main__":
    main()

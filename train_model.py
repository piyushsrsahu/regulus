"""
Model training script for the Kepler exoplanet transit classifier.
Trains a balanced Random Forest classifier on extracted BLS features and exports the model artifact.

Data sources, in order of preference:
  1. Kaggle "Kepler Labelled Time Series Data": data/raw/exoTrain.csv (+ data/raw/exoTest.csv).
     Train on exoTrain, evaluate on the official exoTest split.
  2. Synthetic fallback (when data/raw/ has no exoTrain.csv): generated light curves with
     injected transits, quiet stars and pulsating stars.

Usage:
    python train_model.py              # Kaggle if present, else synthetic
    python train_model.py --synthetic  # force the synthetic dataset
    python train_model.py --augment 0  # Kaggle without injected-transit augmentation
"""
import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import PROJECT_ROOT
from src.features import extract_features, robust_std
from src.preprocessing import load_light_curve

# Scale-free features: comparable between de-meaned Kaggle flux and baseline-1 CSVs.
FEATURE_COLUMNS = ["period", "duration", "depth_snr", "bls_sde", "box_vs_sine"]
FEATURE_VERSION = 3          # bump whenever extract_features changes (invalidates the cache)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
KAGGLE_TRAIN = RAW_DIR / "exoTrain.csv"
KAGGLE_TEST = RAW_DIR / "exoTest.csv"

KEPLER_CADENCE_DAYS = 0.02043   # Kepler long cadence: 29.4 minutes
DEFAULT_AUGMENT = 400           # injected-transit copies of exoTrain negatives (training only)
SEED = 42


# ------------------------------------------------------------------
# Kaggle preprocessing
# ------------------------------------------------------------------
def kaggle_row_to_relative_flux(flux_row: np.ndarray) -> np.ndarray:
    """
    Kaggle flux values are de-meaned (the stellar baseline is gone, values sit near 0 and
    are often negative), which makes lightkurve's divide-by-trend detrending blow up.

    Clip upward outliers (cosmic rays / flares; transits only go down), then map the
    series onto a positive baseline of 1. The absolute scale is arbitrary, which is why
    the model uses scale-free features (depth_snr, bls_sde) rather than raw depth/power.
    """
    f = np.asarray(flux_row, dtype=float).copy()
    med = np.nanmedian(f)
    sd = robust_std(f)
    if not np.isfinite(sd) or sd == 0:
        sd = float(np.nanstd(f)) or 1.0
    f[f > med + 3.0 * sd] = np.nan
    centered = f - med
    scale = max(1000.0 * sd, 2.0 * float(np.nanmax(np.abs(centered))))  # keeps 1 + x > 0
    return 1.0 + centered / scale


def inject_transit(rel_flux: np.ndarray, time: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a box transit with a random period/duration and a target SNR of 7-40."""
    f = rel_flux.copy()
    t_span = time[-1] - time[0]
    period = rng.uniform(0.8, min(20.0, t_span / 2.2))
    duration = rng.uniform(0.08, 0.40)
    t0 = rng.uniform(time[0], time[0] + period)
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) < 0.5 * duration
    n_in = max(int(in_transit.sum()), 1)
    depth = rng.uniform(7.0, 40.0) * robust_std(f) / np.sqrt(n_in)
    f[in_transit] -= depth
    return f


def _row_features(rel_flux: np.ndarray, time: np.ndarray, label: int, row_id: int, injected: bool):
    try:
        feats = extract_features(load_light_curve(time, rel_flux))
    except Exception:
        return None
    feats.update(label=label, row_id=row_id, injected=injected)
    return feats


def _read_kaggle(path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    flux_cols = [c for c in df.columns if str(c).startswith("FLUX")]
    labels = (df["LABEL"].to_numpy() == 2).astype(int)
    return df[flux_cols].to_numpy(dtype=float), labels


def kaggle_features(path, n_jobs: int = -1) -> pd.DataFrame:
    """One feature row per star in a Kaggle CSV, cached under data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cache = PROCESSED_DIR / f"{path.stem}_features_v{FEATURE_VERSION}.csv"
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        print(f"  using cached features: {cache.relative_to(PROJECT_ROOT)}")
        return pd.read_csv(cache)

    flux, labels = _read_kaggle(path)
    time = np.arange(flux.shape[1]) * KEPLER_CADENCE_DAYS
    print(f"  extracting BLS features for {len(labels)} light curves from {path.name} "
          f"({labels.sum()} planet hosts); this takes a few minutes...")
    rows = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(_row_features)(kaggle_row_to_relative_flux(flux[i]), time, int(labels[i]), i, False)
        for i in range(len(labels))
    )
    df = pd.DataFrame([r for r in rows if r is not None])
    dropped = len(labels) - len(df)
    if dropped:
        print(f"  skipped {dropped} light curves that failed preprocessing")
    df.to_csv(cache, index=False)
    return df


def augmentation_features(path, negative_row_ids: np.ndarray, n: int, n_jobs: int = -1) -> pd.DataFrame:
    """Inject synthetic transits into copies of randomly chosen negative stars (training only)."""
    if n <= 0:
        return pd.DataFrame()
    flux, _ = _read_kaggle(path)
    time = np.arange(flux.shape[1]) * KEPLER_CADENCE_DAYS
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(negative_row_ids, size=min(n, len(negative_row_ids)), replace=False)
    seeds = rng.integers(0, 2**32 - 1, size=len(chosen))
    print(f"  injecting transits into {len(chosen)} copies of negative training stars...")
    rows = Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(_row_features)(
            inject_transit(kaggle_row_to_relative_flux(flux[i]), time, np.random.default_rng(s)),
            time, 1, int(i), True)
        for i, s in zip(chosen, seeds)
    )
    return pd.DataFrame([r for r in rows if r is not None])


def build_kaggle_dataset(augment: int, n_jobs: int):
    print(f"Kaggle Kepler dataset found in {RAW_DIR.relative_to(PROJECT_ROOT)}/")
    train = kaggle_features(KAGGLE_TRAIN, n_jobs)
    if KAGGLE_TEST.exists():
        test = kaggle_features(KAGGLE_TEST, n_jobs)
        split = "official Kaggle split (train: exoTrain, test: exoTest)"
    else:
        train, test = train_test_split(train, test_size=0.25, random_state=SEED, stratify=train["label"])
        split = "stratified 75/25 split of exoTrain (exoTest.csv not found)"
    aug = augmentation_features(KAGGLE_TRAIN, train.loc[train["label"] == 0, "row_id"].to_numpy(), augment, n_jobs)
    dataset = f"Kaggle Kepler Labelled Time Series (exoTrain + {len(aug)} injected-transit copies)"
    return train, aug, test, dataset, split


# ------------------------------------------------------------------
# Synthetic fallback
# ------------------------------------------------------------------
def generate_synthetic_training_dataset(n_samples: int = 300) -> pd.DataFrame:
    """
    Generate synthetic photometric time-series samples spanning both transit candidates
    and non-transit stellar baselines with realistic noise and astrophysical variation.
    """
    rows = []
    rng = np.random.default_rng(SEED)

    for _ in range(n_samples):
        t = np.arange(0, 30, 0.03)  # 30-day baseline, ~1,000 points
        noise_level = rng.uniform(0.0008, 0.003)
        drift_amplitude = rng.uniform(0.001, 0.01)
        drift_period = rng.uniform(5.0, 30.0)

        # Baseline flux with low-frequency drift and Gaussian noise
        flux = 1.0 + drift_amplitude * np.sin(2 * np.pi * t / drift_period) + rng.normal(0, noise_level, size=len(t))

        # 35% positives (transit present), 65% negatives (no transit / stellar variability)
        if rng.random() < 0.35:
            period = rng.uniform(1.2, 8.5)
            t0 = rng.uniform(0.1, period)
            depth = rng.uniform(0.005, 0.035)  # 0.5% to 3.5% transit depth
            duration = rng.uniform(0.12, 0.35)
            phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
            flux[np.abs(phase) < (duration / 2.0)] -= depth
            label = 1
        else:
            if rng.random() < 0.4:  # variable / pulsating star
                flux += rng.uniform(0.002, 0.008) * np.sin(2 * np.pi * t / rng.uniform(0.5, 4.0))
            label = 0

        try:
            feats = extract_features(load_light_curve(t, flux))  # same search range as the app
        except Exception:
            continue
        feats["label"] = label
        rows.append(feats)

    return pd.DataFrame(rows)


def build_synthetic_dataset(n_samples: int = 300):
    print("No Kaggle exoTrain.csv in data/raw/. Generating a synthetic photometric dataset...")
    df = generate_synthetic_training_dataset(n_samples)
    train, test = train_test_split(df, test_size=0.25, random_state=SEED, stratify=df["label"])
    return train, pd.DataFrame(), test, f"Synthetic light curves ({len(df)} generated)", "stratified 75/25 split"


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def describe(name: str, df: pd.DataFrame) -> str:
    return f"{name}: {len(df)} light curves ({int(df['label'].sum())} transits, {int((df['label'] == 0).sum())} non-transits)"


def make_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )


def exoplanet_scores(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Precision/recall/F1 at the 0.5 threshold plus threshold-free average precision (PR-AUC)."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, proba >= 0.5, labels=[1], zero_division=0)
    return {
        "precision": round(float(p[0]), 4),
        "recall": round(float(r[0]), 4),
        "f1": round(float(f1[0]), 4),
        "average_precision": round(float(average_precision_score(y_true, proba)), 4),
    }


def cross_validate(natural: pd.DataFrame, aug: pd.DataFrame, k: int = 5) -> dict:
    """
    Stratified k-fold CV over the real training curves. Injected-transit copies are only
    used in a fold's training part, and never when they were made from a held-out star.
    With very few real positives this is a steadier estimate than one small test split.
    """
    ys, ps = [], []
    for tr_idx, te_idx in StratifiedKFold(k, shuffle=True, random_state=0).split(natural, natural["label"]):
        fold_train, fold_test = natural.iloc[tr_idx], natural.iloc[te_idx]
        if not aug.empty:
            fold_train = pd.concat([fold_train, aug[~aug["row_id"].isin(fold_test["row_id"])]])
        model = make_model().fit(fold_train[FEATURE_COLUMNS], fold_train["label"])
        ps.append(model.predict_proba(fold_test[FEATURE_COLUMNS])[:, 1])
        ys.append(fold_test["label"].to_numpy())
    scores = exoplanet_scores(np.concatenate(ys), np.concatenate(ps))
    return {"folds": k, "n_positive": int(natural["label"].sum()), **scores}


def main():
    parser = argparse.ArgumentParser(description="Train the Regulus transit classifier.")
    parser.add_argument("--synthetic", action="store_true", help="ignore data/raw and use the synthetic dataset")
    parser.add_argument("--augment", type=int, default=DEFAULT_AUGMENT,
                        help=f"injected-transit training copies for Kaggle mode (default {DEFAULT_AUGMENT})")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel workers for feature extraction")
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if KAGGLE_TRAIN.exists() and not args.synthetic:
        natural, aug, test, dataset, split = build_kaggle_dataset(args.augment, args.n_jobs)
    else:
        natural, aug, test, dataset, split = build_synthetic_dataset()
    train = pd.concat([natural, aug], ignore_index=True)

    print(describe("Train", train))
    print(describe("Test ", test))

    model = make_model().fit(train[FEATURE_COLUMNS], train["label"])

    y_true = test["label"].to_numpy()
    proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    test_scores = exoplanet_scores(y_true, proba)

    print("\nCross-validating on the training curves...")
    cv = cross_validate(natural, aug)

    print(f"\n--- Test Set Evaluation ({split}) ---")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["No Exoplanet", "Exoplanet"], zero_division=0))
    print("Confusion Matrix [rows = true, cols = predicted; order: no_exoplanet, exoplanet]:")
    print(cm)
    print(f"Exoplanet average precision (PR-AUC): {test_scores['average_precision']:.3f}")
    print(f"\n--- {cv['folds']}-fold cross-validation on training curves ({cv['n_positive']} real transits) ---")
    print(f"precision {cv['precision']:.2f}  recall {cv['recall']:.2f}  F1 {cv['f1']:.2f}  "
          f"average precision {cv['average_precision']:.2f}")
    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:10s} {imp:.3f}")

    metrics = {
        "dataset": dataset,
        "split": split,
        "n_train": int(len(train)), "n_train_positive": int(train["label"].sum()),
        "n_test": int(len(test)), "n_test_positive": int(test["label"].sum()),
        "exoplanet_precision": test_scores["precision"],
        "exoplanet_recall": test_scores["recall"],
        "exoplanet_f1": test_scores["f1"],
        "exoplanet_average_precision": test_scores["average_precision"],
        "accuracy": round(float((y_true == y_pred).mean()), 4),
        "confusion_matrix": cm.tolist(),
        "cross_validation": cv,
        "feature_importances": {n: round(float(v), 4) for n, v in zip(FEATURE_COLUMNS, model.feature_importances_)},
    }
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    joblib.dump({
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "feature_version": FEATURE_VERSION,
        "dataset": dataset,
        "metrics": metrics,
        "trained_at": trained_at,
        "sklearn_version": sklearn.__version__,
    }, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({**metrics, "trained_at": trained_at}, indent=2) + "\n")
    print(f"\nModel artifact saved to {MODEL_PATH.relative_to(PROJECT_ROOT)}; "
          f"metrics written to {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

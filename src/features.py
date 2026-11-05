"""
Feature extraction module for light curve transit candidate analysis.
"""
import numpy as np
import lightkurve as lk


def extract_features(lc: lk.LightCurve, min_period: float = 0.5, max_period: float = None, n_periods: int = 1000) -> dict:
    """
    Compute Box Least Squares (BLS) periodogram features from a cleaned light curve.

    Args:
        lc: Detrended lightkurve.LightCurve instance.
        min_period: Minimum orbital period to test in days (default: 0.5 days).
        max_period: Maximum orbital period to test in days. If None, uses half baseline.
        n_periods: Number of test frequencies/periods in grid.

    Returns:
        dict: Extracted transit properties including period, depth, duration, bls_power, and transit_time.
    """
    if len(lc.time) < 10:
        raise ValueError("Light curve must contain at least 10 points for periodogram analysis.")

    # Calculate baseline time span
    t_span = float(lc.time.value.max() - lc.time.value.min())
    if t_span <= 0:
        raise ValueError("Light curve time span must be strictly positive.")

    # Determine max period search boundary (cannot exceed half the total observation baseline)
    if max_period is None or max_period > (t_span / 2.0):
        max_period = max(min_period + 0.1, t_span / 2.0)

    if min_period >= max_period:
        min_period = max(0.1, max_period / 2.0)

    periods = np.linspace(min_period, max_period, n_periods)

    # Compute Box Least Squares (BLS) periodogram
    periodogram = lc.to_periodogram(method="bls", period=periods)

    period = float(periodogram.period_at_max_power.value)
    depth = float(periodogram.depth_at_max_power.value)
    duration = float(periodogram.duration_at_max_power.value)
    bls_power = float(periodogram.max_power.value)
    transit_time = float(periodogram.transit_time_at_max_power.value)

    return {
        "period": period,
        "depth": max(0.0, depth),
        "duration": max(0.0, duration),
        "bls_power": max(0.0, bls_power),
        "transit_time": transit_time,
    }

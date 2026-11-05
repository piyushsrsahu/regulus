"""
Feature extraction module for light curve transit candidate analysis.
"""
import numpy as np
import lightkurve as lk

# Period-grid density. A trial grid that is uniform in *frequency* keeps the
# transit-phase drift across the whole baseline below the shortest searched
# duration, so long baselines do not "step over" the true period
# (spacing df = FREQUENCY_FACTOR * GRID_MIN_DURATION / T**2, as in astropy's autoperiod).
GRID_MIN_DURATION = 0.1   # days
FREQUENCY_FACTOR = 2.0
MAX_GRID_POINTS = 50_000  # keeps a single analysis well under the 5 s NFR

# lightkurve's default BLS duration grid starts at 0.05 d; it uses that value
# in its own grid-size sanity check, so we mirror it when reporting our grid.
_LK_MIN_DURATION = 0.05


def build_period_grid(t_span: float, min_period: float, max_period: float, n_min: int = 1000) -> np.ndarray:
    """Return trial periods (days), evenly spaced in frequency between min_period and max_period."""
    f_lo, f_hi = 1.0 / max_period, 1.0 / min_period
    df = FREQUENCY_FACTOR * GRID_MIN_DURATION / t_span ** 2
    n = int(np.ceil((f_hi - f_lo) / df))
    n = int(np.clip(n, n_min, MAX_GRID_POINTS))
    return 1.0 / np.linspace(f_lo, f_hi, n)


def extract_features(lc: lk.LightCurve, min_period: float = 0.5, max_period: float = None,
                     n_periods: int = 1000) -> dict:
    """
    Compute Box Least Squares (BLS) periodogram features from a cleaned light curve.

    Args:
        lc: Detrended lightkurve.LightCurve instance.
        min_period: Minimum orbital period to test in days (default: 0.5 days).
        max_period: Maximum orbital period to test in days. If None, uses half baseline.
        n_periods: Minimum number of trial periods; the grid grows with the baseline.

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

    periods = build_period_grid(t_span, min_period, max_period, n_min=n_periods)

    # lightkurve estimates a grid size from `frequency_factor` even when explicit
    # periods are given (and raises above 1e7 points). Pass the factor that
    # matches the grid we actually use so that check reflects reality.
    lk_factor = (1.0 / min_period - 1.0 / max_period) * t_span ** 2 / (_LK_MIN_DURATION * len(periods))

    # Compute Box Least Squares (BLS) periodogram
    periodogram = lc.to_periodogram(method="bls", period=periods, frequency_factor=lk_factor)

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

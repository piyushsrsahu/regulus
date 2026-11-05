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


SUN_RADIUS_IN_EARTH = 109.0     # R_sun / R_earth
JUPITER_RADIUS_IN_EARTH = 11.21  # R_jup / R_earth

# Upper radius bound (Earth radii) for each size class.
PLANET_CLASSES = [
    (2.0, "Terrestrial / Super-Earth"),
    (4.0, "Sub-Neptune / Mini-Neptune"),
    (8.0, "Neptune-like"),
    (float("inf"), "Gas Giant / Jovian"),
]


def estimate_planet_size(depth: float, stellar_radius_suns: float = 1.0) -> dict:
    """
    Convert fractional transit depth to planet radius via depth = (Rp / R*)**2.

    Assumes a solar-type host unless stellar_radius_suns is given.
    Returns radius in Earth and Jupiter radii plus a coarse size class.
    """
    depth = max(0.0, float(depth))
    r_earth = SUN_RADIUS_IN_EARTH * stellar_radius_suns * np.sqrt(depth)
    planet_class = next(name for limit, name in PLANET_CLASSES if r_earth < limit)
    return {
        "radius_earth": float(r_earth),
        "radius_jupiter": float(r_earth / JUPITER_RADIUS_IN_EARTH),
        "planet_class": planet_class,
    }


def robust_std(x: np.ndarray) -> float:
    """1.4826 x median absolute deviation: a standard deviation estimate that ignores outliers."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def box_vs_sine(time: np.ndarray, flux: np.ndarray, period: float, in_transit: np.ndarray) -> float:
    """
    Transit shape test: log10 of the variance explained by the BLS box model divided by the
    variance explained by a sinusoid at the same period.

    A flat-bottomed transit is fit far better by a box (positive values), while pulsating or
    rotating stars are fit better by a sine wave (negative values). Scale-free.
    """
    y = np.asarray(flux, dtype=float)
    ss0 = float(np.sum((y - y.mean()) ** 2))
    if ss0 == 0 or period <= 0:
        return 0.0
    if in_transit.any() and (~in_transit).any():
        sse_box = float(np.sum((y[in_transit] - y[in_transit].mean()) ** 2)
                        + np.sum((y[~in_transit] - y[~in_transit].mean()) ** 2))
    else:
        sse_box = ss0
    w = 2 * np.pi * np.asarray(time, dtype=float) / period
    design = np.column_stack([np.sin(w), np.cos(w), np.ones_like(w)])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    sse_sine = float(np.sum((y - design @ coef) ** 2))
    eps = 1e-12 * ss0
    return float(np.log10((ss0 - sse_box + eps) / (ss0 - sse_sine + eps)))


def build_period_grid(t_span: float, min_period: float, max_period: float, n_min: int = 1000) -> np.ndarray:
    """Return trial periods (days), evenly spaced in frequency between min_period and max_period."""
    f_lo, f_hi = 1.0 / max_period, 1.0 / min_period
    df = FREQUENCY_FACTOR * GRID_MIN_DURATION / t_span ** 2
    n = int(np.ceil((f_hi - f_lo) / df))
    n = int(np.clip(n, n_min, MAX_GRID_POINTS))
    return 1.0 / np.linspace(f_lo, f_hi, n)


def extract_features(lc: lk.LightCurve, min_period: float = 0.5, max_period: float = None,
                     n_periods: int = 1000, return_periodogram: bool = False):
    """
    Compute Box Least Squares (BLS) periodogram features from a cleaned light curve.

    Args:
        lc: Detrended lightkurve.LightCurve instance.
        min_period: Minimum orbital period to test in days (default: 0.5 days).
        max_period: Maximum orbital period to test in days. If None, uses half baseline.
        n_periods: Minimum number of trial periods; the grid grows with the baseline.
        return_periodogram: If True, also return (periods, power) arrays so callers
            can plot the exact spectrum the features came from.

    Returns:
        dict: Extracted transit properties including period, depth, duration, bls_power, and transit_time.
        If return_periodogram is True: (dict, (periods, power)).
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

    power = np.asarray(periodogram.power.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    time = np.asarray(lc.time.value, dtype=float)

    # Scale-free statistics (unchanged by any affine rescaling of the flux), so
    # curves with a lost baseline (e.g. the de-meaned Kaggle set) and ordinary
    # baseline-1 CSVs land in the same feature space.
    phase = ((time - transit_time + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) < 0.5 * duration
    n_in = int(in_transit.sum())
    noise = robust_std(flux[~in_transit]) if np.any(~in_transit) else robust_std(flux)
    depth_snr = max(0.0, depth) / noise * np.sqrt(max(n_in, 1)) if noise > 0 else 0.0
    power_scatter = robust_std(power)
    bls_sde = (power.max() - np.median(power)) / power_scatter if power_scatter > 0 else 0.0

    features = {
        "box_vs_sine": box_vs_sine(time, flux, period, in_transit),
        "period": period,
        "depth": max(0.0, depth),
        "duration": max(0.0, duration),
        "bls_power": max(0.0, bls_power),
        "transit_time": transit_time,
        "noise": float(noise),
        "n_in_transit": n_in,
        "depth_snr": float(depth_snr),
        "bls_sde": float(bls_sde),
    }
    if return_periodogram:
        return features, (np.asarray(periodogram.period.value), np.asarray(periodogram.power.value))
    return features

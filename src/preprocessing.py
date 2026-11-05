"""
Preprocessing routines for raw photometric time-series data.
"""
import numpy as np
import pandas as pd
import lightkurve as lk


def parse_light_curve_frame(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Pull numeric time and flux arrays out of an uploaded CSV table.

    Column names are matched case-insensitively ("Time", " FLUX " etc.).
    Non-numeric cells become NaN and are later dropped by load_light_curve.

    Raises:
        ValueError: if the columns are missing, the table is empty, or no
            row has both a numeric time and a numeric flux value.
    """
    col_map = {str(c).lower().strip(): c for c in df.columns}
    time_col = col_map.get("time")
    flux_col = col_map.get("flux")
    if time_col is None or flux_col is None:
        found = ", ".join(map(str, df.columns)) or "(none)"
        raise ValueError(f"CSV must contain columns labeled 'time' and 'flux'. Found: {found}")

    if len(df) == 0:
        raise ValueError("CSV has a header row but no data rows.")

    time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    flux = pd.to_numeric(df[flux_col], errors="coerce").to_numpy(dtype=float)

    if not np.any(np.isfinite(time) & np.isfinite(flux)):
        raise ValueError("No rows contain numeric values in both the 'time' and 'flux' columns.")

    return time, flux


def load_light_curve(time: np.ndarray, flux: np.ndarray, window_length: int = 101) -> lk.LightCurve:
    """
    Wrap time and flux arrays into a LightCurve object and apply baseline detrending.

    Args:
        time: Array of observation timestamps (e.g. days / BJD).
        flux: Array of observed relative flux values.
        window_length: Filter window size for detrending (must be odd).

    Returns:
        Flattened lightkurve.LightCurve instance with baseline centered at ~1.0.
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if len(time) != len(flux):
        raise ValueError(f"Length mismatch: time has {len(time)} points but flux has {len(flux)} points.")

    if len(time) < 10:
        raise ValueError("Light curve must contain at least 10 data points.")

    # Remove non-finite values (NaNs, infs)
    valid_mask = np.isfinite(time) & np.isfinite(flux)
    time = time[valid_mask]
    flux = flux[valid_mask]

    if len(time) < 10:
        raise ValueError("Light curve has fewer than 10 valid finite data points after NaN filtering.")

    # Ensure strictly sorted by time
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    flux = flux[sort_idx]

    # Create raw LightCurve object
    lc = lk.LightCurve(time=time, flux=flux)

    # Adapt window_length if light curve has fewer points
    n_points = len(time)
    if window_length >= n_points:
        window_length = n_points - 1 if (n_points % 2 == 0) else n_points - 2
    if window_length % 2 == 0:
        window_length += 1
    if window_length < 3:
        window_length = 3

    # Flatten and detrend long-term stellar variability
    try:
        flat_lc = lc.flatten(window_length=window_length, polyorder=2)
    except Exception:
        # Fallback if polynomial fitting fails on very short series
        flat_lc = lc.flatten()

    return flat_lc

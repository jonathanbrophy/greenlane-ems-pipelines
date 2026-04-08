"""SoC resampling and session filtering utilities.

All functions operate on plain numpy arrays — no Spark dependency.
Only sessions with OCPP-reported SoC data are used (no SoC reconstruction).
"""

from __future__ import annotations

import numpy as np


def resample_to_soc_grid(
    soc: np.ndarray,
    power: np.ndarray,
    grid_points: int = 101,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a session's power curve onto a uniform SoC grid.

    Args:
        soc: Observed SoC values (must be monotonically non-decreasing).
        power: Observed power values (kW) at each SoC point.
        grid_points: Number of evenly-spaced SoC points (default 101 = 1% intervals from 0-100%).

    Returns:
        (soc_grid, power_grid) tuple of resampled arrays.
    """
    soc_grid = np.linspace(soc.min(), soc.max(), grid_points)
    power_grid = np.interp(soc_grid, soc, power)
    return soc_grid, power_grid


def filter_session(
    soc: np.ndarray,
    power: np.ndarray,
    min_points: int = 5,
    max_gap_pct: float = 0.20,
) -> bool:
    """Check whether a session has sufficient data quality for aggregation.

    Args:
        soc: SoC values for the session.
        power: Power values for the session.
        min_points: Minimum number of data points required.
        max_gap_pct: Maximum allowed gap in SoC coverage as a fraction of the
            total SoC range. E.g. 0.20 means no single gap can exceed 20% of
            (soc_max - soc_min).

    Returns:
        True if the session passes quality checks.
    """
    if len(soc) < min_points:
        return False

    if len(soc) != len(power):
        return False

    soc_range = soc.max() - soc.min()
    if soc_range <= 0:
        return False

    sorted_soc = np.sort(soc)
    gaps = np.diff(sorted_soc)
    if len(gaps) > 0 and gaps.max() / soc_range > max_gap_pct:
        return False

    return True


def normalize_power(
    power_kw: np.ndarray,
    evse_max_power_kw: float,
) -> np.ndarray:
    """Normalize power as a fraction of EVSE max rated power.

    Makes curves from different EVSE power tiers comparable by removing
    the EVSE ceiling effect at low SoC.

    Args:
        power_kw: Absolute power values in kW.
        evse_max_power_kw: EVSE rated max power in kW.

    Returns:
        Power as a fraction of EVSE max [0, 1].
    """
    return power_kw / evse_max_power_kw


def classify_evse_power_tier(max_power_kw: float) -> str:
    """Classify an EVSE by its max power into a tier label.

    Args:
        max_power_kw: EVSE rated max power in kW.

    Returns:
        One of 'L2', 'DCFC_50', 'DCFC_150+'.
    """
    if max_power_kw <= 22:
        return "L2"
    elif max_power_kw <= 100:
        return "DCFC_50"
    else:
        return "DCFC_150+"

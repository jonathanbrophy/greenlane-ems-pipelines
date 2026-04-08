"""Piecewise linear fitting utilities for charge curve approximation.

Fits PWL models to P50 (median) charge curves. The resulting breakpoints
and slopes map directly to MPC LP constraints.
"""

from __future__ import annotations

import numpy as np
import pwlf


def fit_pwl(
    soc_grid: np.ndarray,
    power_values: np.ndarray,
    n_segments: int = 5,
) -> pwlf.PiecewiseLinFit:
    """Fit a piecewise linear model to a charge curve.

    Args:
        soc_grid: Uniform SoC grid points (e.g. 0.0 to 1.0).
        power_values: Power values (kW) at each SoC grid point.
        n_segments: Number of linear segments (4-8 typical).

    Returns:
        Fitted pwlf.PiecewiseLinFit model.
    """
    model = pwlf.PiecewiseLinFit(soc_grid, power_values)
    model.fit(n_segments)
    return model


def pwl_to_breakpoints(
    model: pwlf.PiecewiseLinFit,
) -> list[tuple[float, float]]:
    """Extract breakpoints as (soc, power_kw) pairs from a fitted PWL model.

    These breakpoints define the charge curve approximation. Each pair of
    consecutive breakpoints defines a linear segment that becomes a constraint
    in the MPC LP.

    Args:
        model: A fitted pwlf.PiecewiseLinFit model.

    Returns:
        List of (soc, power_kw) tuples at each breakpoint.
    """
    soc_breaks = model.fit_breaks
    power_breaks = model.predict(soc_breaks)
    return [(float(s), float(p)) for s, p in zip(soc_breaks, power_breaks)]


def pwl_to_slopes(model: pwlf.PiecewiseLinFit) -> list[float]:
    """Extract slopes (dP/dSoC) for each segment of a fitted PWL model.

    These slopes represent the rate of power change per unit SoC and can
    be loaded directly into the MPC LP as dP/dSoC bounds.

    Args:
        model: A fitted pwlf.PiecewiseLinFit model.

    Returns:
        List of slopes, one per segment.
    """
    return [float(s) for s in model.calc_slopes()]


def compute_fit_rmse(
    model: pwlf.PiecewiseLinFit,
    soc_grid: np.ndarray,
    power_values: np.ndarray,
) -> float:
    """Compute RMSE between the PWL fit and the original curve.

    Args:
        model: A fitted pwlf.PiecewiseLinFit model.
        soc_grid: The SoC grid used for fitting.
        power_values: The original power values.

    Returns:
        Root mean squared error in kW.
    """
    predicted = model.predict(soc_grid)
    return float(np.sqrt(np.mean((predicted - power_values) ** 2)))

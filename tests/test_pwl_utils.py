import numpy as np
import pytest

from ems_pipelines.pwl_utils import (
    compute_fit_rmse,
    fit_pwl,
    pwl_to_breakpoints,
    pwl_to_slopes,
)


def _make_typical_charge_curve() -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic charge curve: flat at 150kW until 80% SoC, then taper to 10kW."""
    soc = np.linspace(0.0, 1.0, 101)
    power = np.where(soc <= 0.8, 150.0, 150.0 - (soc - 0.8) / 0.2 * 140.0)
    return soc, power


class TestFitPwl:
    def test_basic_fit_runs(self):
        soc, power = _make_typical_charge_curve()
        model = fit_pwl(soc, power, n_segments=3)
        assert model is not None
        assert hasattr(model, "fit_breaks")

    def test_breakpoints_count(self):
        soc, power = _make_typical_charge_curve()
        model = fit_pwl(soc, power, n_segments=3)
        breakpoints = pwl_to_breakpoints(model)
        # n_segments + 1 breakpoints (start + end + interior)
        assert len(breakpoints) == 4

    def test_breakpoints_span_soc_range(self):
        soc, power = _make_typical_charge_curve()
        model = fit_pwl(soc, power, n_segments=3)
        breakpoints = pwl_to_breakpoints(model)
        assert breakpoints[0][0] == pytest.approx(0.0, abs=0.01)
        assert breakpoints[-1][0] == pytest.approx(1.0, abs=0.01)

    def test_slopes_count(self):
        soc, power = _make_typical_charge_curve()
        model = fit_pwl(soc, power, n_segments=3)
        slopes = pwl_to_slopes(model)
        assert len(slopes) == 3


class TestComputeFitRmse:
    def test_perfect_fit_has_low_rmse(self):
        # A piecewise linear curve with 2 segments should fit nearly perfectly
        # with 2 segments
        soc = np.linspace(0.0, 1.0, 101)
        power = np.where(soc <= 0.5, 100.0, 100.0 - (soc - 0.5) * 200.0)
        model = fit_pwl(soc, power, n_segments=2)
        rmse = compute_fit_rmse(model, soc, power)
        assert rmse < 5.0  # should be very close to 0

    def test_rmse_decreases_with_more_segments(self):
        soc, power = _make_typical_charge_curve()
        model_3 = fit_pwl(soc, power, n_segments=3)
        model_5 = fit_pwl(soc, power, n_segments=5)
        rmse_3 = compute_fit_rmse(model_3, soc, power)
        rmse_5 = compute_fit_rmse(model_5, soc, power)
        # More segments should give at least as good a fit
        assert rmse_5 <= rmse_3 + 1.0  # allow small tolerance for optimizer variance

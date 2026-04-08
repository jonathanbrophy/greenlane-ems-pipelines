import numpy as np
import pytest

from ems_pipelines.soc_utils import (
    classify_evse_power_tier,
    filter_session,
    normalize_power,
    reconstruct_soc,
    resample_to_soc_grid,
)


# --- reconstruct_soc ---


class TestReconstructSoc:
    def test_basic_integration(self):
        energy = np.array([5.0, 5.0, 5.0, 5.0])
        soc = reconstruct_soc(energy, battery_capacity_kwh=100.0, soc_start=0.0)
        assert soc[-1] == pytest.approx(0.20)

    def test_with_start_soc(self):
        energy = np.array([10.0, 10.0])
        soc = reconstruct_soc(energy, battery_capacity_kwh=100.0, soc_start=0.5)
        assert soc[-1] == pytest.approx(0.70)

    def test_monotonically_increasing(self):
        energy = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        soc = reconstruct_soc(energy, battery_capacity_kwh=60.0, soc_start=0.1)
        assert np.all(np.diff(soc) > 0)

    def test_zero_energy(self):
        energy = np.array([0.0, 0.0, 0.0])
        soc = reconstruct_soc(energy, battery_capacity_kwh=100.0, soc_start=0.5)
        np.testing.assert_array_almost_equal(soc, [0.5, 0.5, 0.5])

    def test_zero_capacity_returns_inf(self):
        result = reconstruct_soc(np.array([5.0]), battery_capacity_kwh=0.0)
        assert np.isinf(result[0])


# --- resample_to_soc_grid ---


class TestResampleToSocGrid:
    def test_preserves_endpoints(self):
        soc = np.array([0.2, 0.5, 0.8])
        power = np.array([150.0, 150.0, 80.0])
        soc_grid, power_grid = resample_to_soc_grid(soc, power, grid_points=50)
        assert soc_grid[0] == pytest.approx(0.2)
        assert soc_grid[-1] == pytest.approx(0.8)

    def test_output_length_matches_grid_points(self):
        soc = np.array([0.0, 0.5, 1.0])
        power = np.array([100.0, 100.0, 50.0])
        soc_grid, power_grid = resample_to_soc_grid(soc, power, grid_points=101)
        assert len(soc_grid) == 101
        assert len(power_grid) == 101

    def test_constant_power_stays_constant(self):
        soc = np.array([0.0, 0.5, 1.0])
        power = np.array([120.0, 120.0, 120.0])
        _, power_grid = resample_to_soc_grid(soc, power, grid_points=50)
        np.testing.assert_array_almost_equal(power_grid, 120.0)

    def test_linear_interpolation(self):
        soc = np.array([0.0, 1.0])
        power = np.array([0.0, 100.0])
        soc_grid, power_grid = resample_to_soc_grid(soc, power, grid_points=11)
        expected = np.linspace(0.0, 100.0, 11)
        np.testing.assert_array_almost_equal(power_grid, expected)


# --- filter_session ---


class TestFilterSession:
    def test_good_session_passes(self):
        soc = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        power = np.array([150, 150, 145, 140, 130, 120, 100, 80, 60])
        assert filter_session(soc, power) is True

    def test_too_few_points_fails(self):
        soc = np.array([0.1, 0.5])
        power = np.array([150, 100])
        assert filter_session(soc, power, min_points=5) is False

    def test_large_gap_fails(self):
        # Gap from 0.1 to 0.9 = 0.8, range = 0.8, gap/range = 1.0 > 0.20
        soc = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
        power = np.array([150, 150, 150, 150, 150, 60])
        assert filter_session(soc, power, max_gap_pct=0.20) is False

    def test_zero_range_fails(self):
        soc = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        power = np.array([100, 100, 100, 100, 100])
        assert filter_session(soc, power) is False

    def test_mismatched_lengths_fails(self):
        soc = np.array([0.1, 0.5, 0.9])
        power = np.array([150, 100])
        assert filter_session(soc, power) is False


# --- normalize_power ---


class TestNormalizePower:
    def test_at_max_power(self):
        power = np.array([150.0])
        result = normalize_power(power, evse_max_power_kw=150.0)
        assert result[0] == pytest.approx(1.0)

    def test_at_half_power(self):
        power = np.array([75.0])
        result = normalize_power(power, evse_max_power_kw=150.0)
        assert result[0] == pytest.approx(0.5)


# --- classify_evse_power_tier ---


class TestClassifyEvsePowerTier:
    def test_l2(self):
        assert classify_evse_power_tier(7.0) == "L2"
        assert classify_evse_power_tier(22.0) == "L2"

    def test_dcfc_50(self):
        assert classify_evse_power_tier(50.0) == "DCFC_50"
        assert classify_evse_power_tier(100.0) == "DCFC_50"

    def test_dcfc_150_plus(self):
        assert classify_evse_power_tier(150.0) == "DCFC_150+"
        assert classify_evse_power_tier(350.0) == "DCFC_150+"

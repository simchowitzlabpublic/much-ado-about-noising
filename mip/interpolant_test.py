"""
Tests for the Interpolant class
"""

import pytest
import torch
from mip.interpolant import Interpolant


class TestInterpolant:
    """Test suite for the Interpolant class"""

    def test_linear_interpolant_creation(self):
        """Test creating a linear interpolant"""
        linear_interp = Interpolant("linear")
        assert linear_interp is not None

    def test_trig_interpolant_creation(self):
        """Test creating a trigonometric interpolant"""
        trig_interp = Interpolant("trig")
        assert trig_interp is not None

    def test_invalid_interpolant_type(self):
        """Test that invalid interpolant type raises NotImplementedError"""
        with pytest.raises(
            NotImplementedError, match="Interpolant type 'invalid_type' not implemented"
        ):
            Interpolant("invalid_type")

    def test_linear_interpolation_single_point(self):
        """Test linear interpolation at a single time point"""
        linear_interp = Interpolant("linear")
        x0 = torch.tensor([1.0, 2.0, 3.0])
        x1 = torch.tensor([4.0, 5.0, 6.0])
        t_single = 0.5

        result = linear_interp.calc_It(t_single, x0, x1)
        expected = torch.tensor([2.5, 3.5, 4.5])

        assert torch.allclose(result, expected)

    def test_linear_interpolation_endpoints(self):
        """Test linear interpolation at t=0 and t=1"""
        linear_interp = Interpolant("linear")
        x0 = torch.tensor([1.0, 2.0, 3.0])
        x1 = torch.tensor([4.0, 5.0, 6.0])

        # At t=0, should return x0
        result_t0 = linear_interp.calc_It(0.0, x0, x1)
        assert torch.allclose(result_t0, x0)

        # At t=1, should return x1
        result_t1 = linear_interp.calc_It(1.0, x0, x1)
        assert torch.allclose(result_t1, x1)

    def test_trig_interpolation_single_point(self):
        """Test trigonometric interpolation at a single time point"""
        trig_interp = Interpolant("trig")
        x0 = torch.tensor([1.0, 2.0, 3.0])
        x1 = torch.tensor([4.0, 5.0, 6.0])
        t_single = 0.5

        result = trig_interp.calc_It(t_single, x0, x1)
        # At t=0.5, alpha = cos(pi/4), beta = sin(pi/4)
        alpha_val = torch.cos(torch.tensor(0.5) * torch.pi / 2)
        beta_val = torch.sin(torch.tensor(0.5) * torch.pi / 2)
        expected = alpha_val * x0 + beta_val * x1

        assert torch.allclose(result, expected)

    def test_trig_interpolation_endpoints(self):
        """Test trigonometric interpolation at t=0 and t=1"""
        trig_interp = Interpolant("trig")
        x0 = torch.tensor([1.0, 2.0, 3.0])
        x1 = torch.tensor([4.0, 5.0, 6.0])

        # At t=0, should return x0
        result_t0 = trig_interp.calc_It(0.0, x0, x1)
        assert torch.allclose(result_t0, x0)

        # At t=1, should return x1
        result_t1 = trig_interp.calc_It(1.0, x0, x1)
        assert torch.allclose(result_t1, x1)

    def test_linear_derivative_single_point(self):
        """Test linear interpolant derivative at a single time point"""
        linear_interp = Interpolant("linear")
        x0 = torch.tensor([1.0, 2.0, 3.0])
        x1 = torch.tensor([4.0, 5.0, 6.0])
        t_single = 0.5

        result = linear_interp.calc_It_dot(t_single, x0, x1)
        # For linear: alpha_dot = -1, beta_dot = 1
        expected = -1.0 * x0 + 1.0 * x1

        assert torch.allclose(result, expected)

    def test_trig_derivative_single_point(self):
        """Test trigonometric interpolant derivative at a single time point"""
        trig_interp = Interpolant("trig")
        x0 = torch.tensor([1.0, 2.0, 3.0])
        x1 = torch.tensor([4.0, 5.0, 6.0])
        t_single = 0.5

        result = trig_interp.calc_It_dot(t_single, x0, x1)
        # alpha_dot = -0.5 * pi * sin(t * pi / 2)
        # beta_dot = 0.5 * pi * cos(t * pi / 2)
        alpha_dot_val = -0.5 * torch.pi * torch.sin(torch.tensor(0.5) * torch.pi / 2)
        beta_dot_val = 0.5 * torch.pi * torch.cos(torch.tensor(0.5) * torch.pi / 2)
        expected = alpha_dot_val * x0 + beta_dot_val * x1

        assert torch.allclose(result, expected)

    def test_batch_interpolation(self):
        """Test interpolation with batch of data points"""
        linear_interp = Interpolant("linear")
        # Use batched x0 and x1 instead of batched t
        x0 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        x1 = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
        t = 0.5

        result = linear_interp.calc_It(t, x0, x1)
        # At t=0.5, should be midpoint
        expected = 0.5 * x0 + 0.5 * x1
        assert torch.allclose(result, expected)

    def test_linear_derivative_constant(self):
        """Test that linear derivative is constant for all t"""
        linear_interp = Interpolant("linear")
        x0 = torch.tensor([1.0, 2.0, 3.0])
        x1 = torch.tensor([4.0, 5.0, 6.0])

        result_t0 = linear_interp.calc_It_dot(0.0, x0, x1)
        result_t05 = linear_interp.calc_It_dot(0.5, x0, x1)
        result_t1 = linear_interp.calc_It_dot(1.0, x0, x1)

        assert torch.allclose(result_t0, result_t05)
        assert torch.allclose(result_t05, result_t1)

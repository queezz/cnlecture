"""Tests for cnlecture.visualizations.euler — pure math functions only."""

import math

import numpy as np
import pytest

from cnlecture.visualizations.euler import euler_series_points, euler_series_terms


class TestEulerSeriesTerms:
    def test_shape(self):
        terms = euler_series_terms(1.0, n_terms=10)
        assert terms.shape == (10,)

    def test_dtype_is_complex(self):
        terms = euler_series_terms(1.0, n_terms=5)
        assert np.iscomplexobj(terms)

    def test_zeroth_term_is_one(self):
        for phi in [0.0, 1.0, 2.5]:
            terms = euler_series_terms(phi, n_terms=5)
            assert terms[0] == pytest.approx(1.0 + 0j)

    def test_first_term_is_i_phi(self):
        phi = 1.7
        terms = euler_series_terms(phi, n_terms=5)
        assert terms[1] == pytest.approx(1j * phi)

    def test_second_term(self):
        phi = 1.3
        terms = euler_series_terms(phi, n_terms=5)
        expected = (1j * phi) ** 2 / math.factorial(2)
        assert terms[2] == pytest.approx(expected)

    def test_phi_zero_all_terms_zero_except_first(self):
        terms = euler_series_terms(0.0, n_terms=8)
        assert terms[0] == pytest.approx(1.0 + 0j)
        assert np.allclose(terms[1:], 0.0)

    def test_n_terms_one(self):
        terms = euler_series_terms(2.0, n_terms=1)
        assert terms.shape == (1,)
        assert terms[0] == pytest.approx(1.0 + 0j)


class TestEulerSeriesPoints:
    def test_shape(self):
        pts = euler_series_points(1.0, n_terms=15)
        assert pts.shape == (15,)

    def test_first_point_is_one(self):
        pts = euler_series_points(1.0, n_terms=5)
        assert pts[0] == pytest.approx(1.0 + 0j)

    def test_phi_zero_all_points_one(self):
        """exp(i·0) = 1, so every partial sum should be 1."""
        pts = euler_series_points(0.0, n_terms=10)
        assert np.allclose(pts, 1.0 + 0j)

    @pytest.mark.parametrize("phi", [0.5, 1.0, 1.5708, 2.0, 3.14159])
    def test_convergence_to_exp(self, phi):
        """With 40 terms the partial sum should be very close to exp(i·phi)."""
        pts = euler_series_points(phi, n_terms=40)
        expected = np.exp(1j * phi)
        assert abs(pts[-1] - expected) < 1e-10

    def test_partial_sums_are_cumulative(self):
        phi = 1.2
        n = 10
        terms = euler_series_terms(phi, n)
        pts = euler_series_points(phi, n)
        for k in range(n):
            assert pts[k] == pytest.approx(np.sum(terms[: k + 1]))

    def test_phi_zero_endpoint_is_one(self):
        pts = euler_series_points(0.0, n_terms=5)
        assert pts[-1] == pytest.approx(1.0 + 0j)

    def test_endpoint_approaches_exact_with_more_terms(self):
        phi = 1.0
        exact = np.exp(1j * phi)
        # Use term counts where convergence is still in progress (not at machine eps)
        errors = [abs(euler_series_points(phi, n)[-1] - exact) for n in [3, 5, 8, 12]]
        # Each error should be strictly smaller than the previous
        assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))

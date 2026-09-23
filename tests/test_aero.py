import numpy as np
import pytest

from aero import naca, panel, theory


def test_parse_code_reads_the_digits():
    assert naca.parse_code("2412") == (0.02, 0.4, 0.12)


def test_parse_code_rejects_bad_input():
    with pytest.raises(ValueError):
        naca.parse_code("241")
    with pytest.raises(ValueError):
        naca.parse_code("2012")  # camber with no camber position


def test_coordinates_close_at_the_trailing_edge_and_run_clockwise():
    x, y = naca.coordinates("2412", 100)
    assert len(x) == 101
    assert (x[0], y[0]) == pytest.approx((x[-1], y[-1]))
    # Clockwise order gives a negative signed area (shoelace formula).
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    assert area < 0


def test_symmetric_airfoil_makes_no_lift_at_zero_angle():
    x, y = naca.coordinates("0012", 160)
    sol = panel.solve(x, y, 0.0)
    assert sol.cl == pytest.approx(0, abs=1e-6)
    assert sol.cm_quarter == pytest.approx(0, abs=1e-6)


def test_lift_from_pressure_matches_lift_from_circulation():
    x, y = naca.coordinates("2412", 200)
    sol = panel.solve(x, y, 5.0)
    assert sol.cl == pytest.approx(sol.cl_circulation, rel=0.01)


def test_pressure_drag_is_close_to_zero():
    # Inviscid flow round a closed body makes no drag (d'Alembert's paradox).
    x, y = naca.coordinates("2412", 200)
    assert abs(panel.solve(x, y, 6.0).cd_pressure) < 1e-3


def test_thin_airfoil_matches_theory():
    # A 2 percent thick airfoil is close to the thin airfoil limit of 2 pi.
    x, y = naca.coordinates("0002", 300)
    cl, _ = panel.polar(x, y, [0.0, 4.0])
    slope = (cl[1] - cl[0]) / np.radians(4.0)
    assert slope == pytest.approx(2 * np.pi, rel=0.02)


def test_cambered_zero_lift_angle_and_moment_match_theory():
    x, y = naca.coordinates("2412", 200)
    cl, cm = panel.polar(x, y, [-2.0, 2.0])
    zero = -2.0 - cl[0] * 4.0 / (cl[1] - cl[0])
    assert zero == pytest.approx(theory.zero_lift_angle_deg("2412"), abs=0.3)
    assert cm.mean() == pytest.approx(theory.moment_quarter_chord("2412"), abs=0.01)


def test_thin_airfoil_theory_zero_lift_angle_for_naca_2412():
    # Standard textbook value is about -2.08 degrees.
    assert theory.zero_lift_angle_deg("2412") == pytest.approx(-2.08, abs=0.02)


def test_lift_converges_with_more_panels():
    x1, y1 = naca.coordinates("2412", 160)
    x2, y2 = naca.coordinates("2412", 320)
    cl1 = panel.solve(x1, y1, 4.0).cl
    cl2 = panel.solve(x2, y2, 4.0).cl
    assert cl1 == pytest.approx(cl2, rel=0.005)

"""Thin airfoil theory and a rough drag estimate.

Thin airfoil theory gives a closed form answer for lift and moment from the
camber line alone. The panel method should agree with it closely for thin
airfoils at small angles, which makes it a useful check.
"""

import numpy as np

from .naca import camber_line, parse_code


def _fourier(code, n_points=20001):
    """Fourier coefficients A0, A1, A2 of the camber line slope."""
    m, p, _ = parse_code(code)
    theta = np.linspace(0, np.pi, n_points)
    x = 0.5 * (1 - np.cos(theta))
    _, dyc = camber_line(x, m, p)
    a0_camber = -np.trapezoid(dyc, theta) / np.pi
    a1 = 2 / np.pi * np.trapezoid(dyc * np.cos(theta), theta)
    a2 = 2 / np.pi * np.trapezoid(dyc * np.cos(2 * theta), theta)
    return a0_camber, a1, a2


def zero_lift_angle_deg(code):
    """Angle of attack at which the airfoil makes no lift."""
    a0_camber, a1, _ = _fourier(code)
    # Cl = 2 pi (A0 + A1 / 2) with A0 = alpha + a0_camber
    return np.degrees(-(a0_camber + a1 / 2))


def lift_coefficient(code, alpha_deg):
    """Cl = 2 pi (alpha - zero lift angle), with angles in radians."""
    return 2 * np.pi * np.radians(np.asarray(alpha_deg) - zero_lift_angle_deg(code))


def moment_quarter_chord(code):
    """Pitching moment about the quarter chord. Constant with alpha."""
    _, a1, a2 = _fourier(code)
    return np.pi / 4 * (a2 - a1)


def skin_friction_drag(code, reynolds, max_thickness_at=0.3):
    """Rough profile drag at zero lift from a flat plate estimate.

    Assumes fully turbulent flow over both surfaces (Prandtl's 0.074 Re^-0.2)
    and scales it up with Raymer's form factor for thickness. Real airfoils
    keep some laminar flow near the nose, so this reads high. It is a sanity
    check, not a prediction, and it does not come from the panel method.
    """
    _, _, t = parse_code(code)
    cf = 0.074 * reynolds ** -0.2
    form_factor = 1 + 0.6 / max_thickness_at * t + 100 * t**4
    return 2 * cf * form_factor

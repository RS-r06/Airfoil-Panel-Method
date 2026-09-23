"""NACA 4-digit airfoil shapes.

Coordinates are in chord lengths: the leading edge sits at x = 0 and the
trailing edge at x = 1.
"""

import numpy as np


def parse_code(code):
    """Split a 4-digit NACA code into max camber, camber position and thickness.

    "2412" gives m = 0.02 (2 percent camber), p = 0.4 (at 40 percent chord)
    and t = 0.12 (12 percent thick).
    """
    code = str(code)
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"expected a 4-digit NACA code, got {code!r}")
    m = int(code[0]) / 100
    p = int(code[1]) / 10
    t = int(code[2:]) / 100
    if (m == 0) != (p == 0):
        raise ValueError(f"NACA {code}: camber and camber position must both be zero or both non-zero")
    return m, p, t


def camber_line(x, m, p):
    """Height of the camber line and its slope at each x."""
    x = np.asarray(x, dtype=float)
    if m == 0:
        return np.zeros_like(x), np.zeros_like(x)
    front = x < p
    yc = np.where(front,
                  m / p**2 * (2 * p * x - x**2),
                  m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2))
    dyc = np.where(front,
                   2 * m / p**2 * (p - x),
                   2 * m / (1 - p)**2 * (p - x))
    return yc, dyc


def thickness(x, t):
    """Half thickness at each x. Uses -0.1036 so the trailing edge closes."""
    x = np.asarray(x, dtype=float)
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
                    + 0.2843 * x**3 - 0.1036 * x**4)


def coordinates(code, n_panels=160):
    """Surface points for the panel method.

    Returns x, y arrays of n_panels + 1 points that run clockwise: from the
    trailing edge along the lower surface to the leading edge, then back along
    the upper surface. The first and last points are both the trailing edge.
    Points bunch up near both edges (cosine spacing), where the flow changes
    fastest.
    """
    if n_panels % 2:
        raise ValueError("n_panels must be even")
    m, p, t = parse_code(code)
    beta = np.linspace(0, np.pi, n_panels // 2 + 1)
    xc = 0.5 * (1 - np.cos(beta))
    yc, dyc = camber_line(xc, m, p)
    yt = thickness(xc, t)
    theta = np.arctan(dyc)

    xu, yu = xc - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = xc + yt * np.sin(theta), yc - yt * np.cos(theta)

    # Lower surface from trailing edge to leading edge, then the upper surface
    # without repeating the leading edge point.
    x = np.concatenate([xl[::-1], xu[1:]])
    y = np.concatenate([yl[::-1], yu[1:]])
    return x, y

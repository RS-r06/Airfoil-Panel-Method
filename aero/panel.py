"""Hess-Smith panel method for the flow around a 2D airfoil.

The surface is cut into straight panels. Each panel carries its own constant
source strength, and every panel shares one constant vortex strength. The
sources stop flow passing through the surface. The vortex adds circulation,
and its strength is set by the Kutta condition: the flow must leave the sharp
trailing edge smoothly.

The flow is inviscid and incompressible, so the method predicts lift, pressure
and pitching moment, but not drag or stall.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Solution:
    alpha_deg: float
    xc: np.ndarray      # control point x (panel midpoints)
    yc: np.ndarray      # control point y
    cp: np.ndarray      # pressure coefficient at each control point
    cl: float           # lift coefficient from the pressure
    cl_circulation: float  # lift coefficient from the circulation
    cd_pressure: float  # pressure drag, should be close to zero
    cm_quarter: float   # pitching moment about the quarter chord, nose up positive


def _panel_velocity(px, py, x1, y1, x2, y2):
    """Velocity at points (px, py) from every panel.

    Returns (source_u, source_v, vortex_u, vortex_v), each of shape
    (points, panels): the velocity from a unit-strength source or vortex
    spread evenly along each panel.
    """
    length = np.hypot(x2 - x1, y2 - y1)
    tx, ty = (x2 - x1) / length, (y2 - y1) / length
    nx, ny = -ty, tx  # outward normal for clockwise point order

    # Every point in each panel's own frame: panel along the x axis from
    # 0 to length, with the outside of the airfoil at positive y.
    dx = px[:, None] - x1[None, :]
    dy = py[:, None] - y1[None, :]
    xl = dx * tx + dy * ty
    yl = dx * nx + dy * ny

    # A control point on its own panel sits exactly on it. Take the limit
    # from outside, where atan2 then gives an angle of pi.
    on_panel = np.abs(yl) < 1e-12
    yl = np.where(on_panel, 0.0, yl)

    r1 = np.hypot(xl, yl)
    r2 = np.hypot(xl - length, yl)
    angle = np.arctan2(yl, xl - length) - np.arctan2(yl, xl)
    log_ratio = np.log(r1 / r2)

    # Local velocities, then turned back into the global frame.
    su, sv = log_ratio / (2 * np.pi), angle / (2 * np.pi)
    vu, vv = angle / (2 * np.pi), -log_ratio / (2 * np.pi)
    source_u, source_v = su * tx + sv * nx, su * ty + sv * ny
    vortex_u, vortex_v = vu * tx + vv * nx, vu * ty + vv * ny
    return source_u, source_v, vortex_u, vortex_v


def solve(x, y, alpha_deg):
    """Solve the flow around the airfoil with surface points x, y.

    x and y must run clockwise and start and end at the trailing edge, as
    naca.coordinates returns them. The chord is taken as 1 and the free
    stream speed as 1.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    alpha = np.radians(alpha_deg)
    x1, y1, x2, y2 = x[:-1], y[:-1], x[1:], y[1:]
    n = len(x1)

    length = np.hypot(x2 - x1, y2 - y1)
    tx, ty = (x2 - x1) / length, (y2 - y1) / length
    nx, ny = -ty, tx
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2

    su, sv, vu, vv = _panel_velocity(xc, yc, x1, y1, x2, y2)
    source_n = su * nx[:, None] + sv * ny[:, None]
    source_t = su * tx[:, None] + sv * ty[:, None]
    vortex_n = (vu * nx[:, None] + vv * ny[:, None]).sum(axis=1)
    vortex_t = (vu * tx[:, None] + vv * ty[:, None]).sum(axis=1)

    free_n = np.cos(alpha) * nx + np.sin(alpha) * ny
    free_t = np.cos(alpha) * tx + np.sin(alpha) * ty

    # n equations for no flow through each panel, plus the Kutta condition:
    # equal speed leaving the trailing edge on the upper and lower surfaces.
    a = np.zeros((n + 1, n + 1))
    b = np.zeros(n + 1)
    a[:n, :n] = source_n
    a[:n, n] = vortex_n
    b[:n] = -free_n
    a[n, :n] = source_t[0] + source_t[-1]
    a[n, n] = vortex_t[0] + vortex_t[-1]
    b[n] = -(free_t[0] + free_t[-1])

    strengths = np.linalg.solve(a, b)
    q, gamma = strengths[:n], strengths[n]

    vt = source_t @ q + gamma * vortex_t + free_t
    cp = 1 - vt**2

    # Pressure pushes inward on each panel. Resolve the total force across
    # and along the free stream.
    fx = -(cp * length * nx).sum()
    fy = -(cp * length * ny).sum()
    cl = fy * np.cos(alpha) - fx * np.sin(alpha)
    cd = fx * np.cos(alpha) + fy * np.sin(alpha)
    moment = ((xc - 0.25) * (-cp * length * ny) - yc * (-cp * length * nx)).sum()

    # Kutta-Joukowski: lift = circulation * 2 / (speed * chord). The vortex
    # sheet runs clockwise round the surface, so its circulation is
    # gamma times the perimeter.
    cl_circ = 2 * gamma * length.sum()

    return Solution(alpha_deg, xc, yc, cp, cl, cl_circ, cd, -moment)


def polar(x, y, alphas_deg):
    """Lift and moment coefficients over a range of angles of attack."""
    sols = [solve(x, y, a) for a in alphas_deg]
    return (np.array([s.cl for s in sols]),
            np.array([s.cm_quarter for s in sols]))

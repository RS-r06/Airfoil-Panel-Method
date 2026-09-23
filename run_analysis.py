"""Run the panel method on a few NACA airfoils and save charts and a table.

    python run_analysis.py
    python run_analysis.py --airfoils 0012 2412 4415 --panels 200 --show
"""

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

from aero import naca, panel, theory

OUT = Path("results")
ALPHAS = np.arange(-4, 10.5, 1.0)


def lift_slope_and_zero_angle(alphas, cl):
    """Straight line fit of Cl against alpha: slope per radian, zero lift angle."""
    slope_deg, intercept = np.polyfit(alphas, cl, 1)
    return np.degrees(slope_deg), -intercept / slope_deg


def plot_shapes(codes, n_panels, plt):
    fig, ax = plt.subplots(figsize=(9, 3))
    for code in codes:
        x, y = naca.coordinates(code, n_panels)
        ax.plot(x, y, label=f"NACA {code}")
    ax.set_aspect("equal")
    ax.set_xlabel("x / chord")
    ax.set_ylabel("y / chord")
    ax.set_title("Airfoil shapes")
    ax.legend()
    return fig


def plot_pressure(code, alpha, n_panels, plt):
    x, y = naca.coordinates(code, n_panels)
    sol = panel.solve(x, y, alpha)
    lower = slice(0, n_panels // 2)
    upper = slice(n_panels // 2, None)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sol.xc[upper], sol.cp[upper], label="Upper surface")
    ax.plot(sol.xc[lower], sol.cp[lower], label="Lower surface")
    ax.invert_yaxis()  # suction (negative Cp) drawn upward, as is usual
    ax.set_xlabel("x / chord")
    ax.set_ylabel("Pressure coefficient Cp")
    ax.set_title(f"NACA {code} at {alpha:g} degrees: Cl = {sol.cl:.3f}")
    ax.legend()
    return fig


def plot_lift(codes, results, plt):
    fig, ax = plt.subplots(figsize=(8, 5))
    for code in codes:
        line, = ax.plot(ALPHAS, results[code]["cl"], "o-", ms=4, label=f"NACA {code} panel method")
        ax.plot(ALPHAS, theory.lift_coefficient(code, ALPHAS), "--",
                color=line.get_color(), label=f"NACA {code} thin airfoil theory")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("Angle of attack (degrees)")
    ax.set_ylabel("Lift coefficient Cl")
    ax.set_title("Lift against angle of attack (inviscid, so no stall)")
    ax.legend(fontsize=8)
    return fig


def convergence(code, alpha, counts):
    cls = []
    for n in counts:
        x, y = naca.coordinates(code, n)
        cls.append(panel.solve(x, y, alpha).cl)
    return np.array(cls)


def plot_convergence(code, alpha, counts, cls, plt):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(counts, cls, "o-")
    ax.set_xscale("log")
    ax.set_xlabel("Number of panels")
    ax.set_ylabel("Lift coefficient Cl")
    ax.set_title(f"NACA {code} at {alpha:g} degrees: Cl settles as panels are added")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--airfoils", nargs="+", default=["0012", "2412", "4412"])
    parser.add_argument("--panels", type=int, default=200, help="even number of panels (default 200)")
    parser.add_argument("--reynolds", type=float, default=3e6, help="Reynolds number for the drag estimate")
    parser.add_argument("--show", action="store_true", help="open each chart as well as saving it")
    args = parser.parse_args()

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    codes = args.airfoils

    results = {}
    for code in codes:
        x, y = naca.coordinates(code, args.panels)
        cl, cm = panel.polar(x, y, ALPHAS)
        slope, zero = lift_slope_and_zero_angle(ALPHAS, cl)
        results[code] = {"cl": cl, "cm": cm, "slope": slope, "zero": zero}

    rows = []
    for code in codes:
        r = results[code]
        rows.append({
            "airfoil": f"NACA {code}",
            "lift_slope_per_rad_panel": round(r["slope"], 3),
            "lift_slope_per_rad_thin": round(2 * np.pi, 3),
            "zero_lift_angle_deg_panel": round(r["zero"], 2) + 0.0,
            "zero_lift_angle_deg_thin": round(theory.zero_lift_angle_deg(code), 2) + 0.0,
            "cm_quarter_at_0deg_panel": round(r["cm"][list(ALPHAS).index(0.0)], 4) + 0.0,
            "cm_quarter_thin": round(theory.moment_quarter_chord(code), 4) + 0.0,
            "cd_estimate_zero_lift": round(theory.skin_friction_drag(code, args.reynolds), 4),
        })
    with open(OUT / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Panel method with {args.panels} panels against thin airfoil theory\n")
    print(f"{'Airfoil':<11}{'dCl/da panel':>13}{'thin':>7}{'a0 panel':>10}{'thin':>7}"
          f"{'Cm panel':>10}{'thin':>8}{'Cd est':>8}")
    for r in rows:
        print(f"{r['airfoil']:<11}{r['lift_slope_per_rad_panel']:>13.3f}{r['lift_slope_per_rad_thin']:>7.3f}"
              f"{r['zero_lift_angle_deg_panel']:>10.2f}{r['zero_lift_angle_deg_thin']:>7.2f}"
              f"{r['cm_quarter_at_0deg_panel']:>10.4f}{r['cm_quarter_thin']:>8.4f}"
              f"{r['cd_estimate_zero_lift']:>8.4f}")
    print(f"\nLift slope per radian, zero lift angle in degrees, Cm about the quarter chord at 0 degrees.")
    print(f"Cd est is the flat plate estimate at Re = {args.reynolds:.0e}, not a panel method result.")

    counts = [20, 40, 80, 160, 320, 640]
    conv_code = codes[1] if len(codes) > 1 else codes[0]
    conv = convergence(conv_code, 4.0, counts)
    print(f"\nNACA {conv_code} at 4 degrees, Cl by panel count:")
    for n, c in zip(counts, conv):
        print(f"  {n:>4} panels  Cl = {c:.4f}")

    figures = {
        "shapes.png": plot_shapes(codes, args.panels, plt),
        "pressure.png": plot_pressure(conv_code, 4.0, args.panels, plt),
        "lift.png": plot_lift(codes, results, plt),
        "convergence.png": plot_convergence(conv_code, 4.0, counts, conv, plt),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(OUT / name, dpi=120)
        print(f"saved {OUT / name}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

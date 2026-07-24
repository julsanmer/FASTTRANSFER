"""Plot clamped B-spline basis functions for publication.

The figure shows the compact support and partition-of-unity behavior of a
one-dimensional B-spline basis. It is independent of the trajectory optimizer
and is meant as a small explanatory graphic for manuscripts.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-ctrl", type=int, default=8, help="Number of B-spline basis/control functions.")
    parser.add_argument("--degree", type=int, default=3, help="B-spline polynomial degree.")
    parser.add_argument("--samples", type=int, default=1200, help="Number of sample points.")
    parser.add_argument("--output-dir", default="output/bspline_basis", help="Directory for generated figures.")
    parser.add_argument("--stem", default=None, help="Output filename stem. Defaults to bspline_basis_nctrl*_deg*.")
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], choices=["pdf", "png", "eps", "svg"])
    parser.add_argument("--dpi", type=int, default=300, help="DPI for raster output.")
    parser.add_argument("--use-tex", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def clamped_uniform_knots(n_ctrl: int, degree: int) -> np.ndarray:
    if n_ctrl <= degree:
        raise ValueError("n_ctrl must be larger than degree")
    n_internal = n_ctrl - degree - 1
    internal = np.linspace(0.0, 1.0, n_internal + 2)[1:-1] if n_internal > 0 else np.array([])
    return np.concatenate([np.zeros(degree + 1), internal, np.ones(degree + 1)])


def basis_matrix(knots: np.ndarray, degree: int, samples: np.ndarray, n_ctrl: int) -> np.ndarray:
    values = np.zeros((n_ctrl, samples.size), dtype=float)
    for idx in range(n_ctrl):
        coeffs = np.zeros(n_ctrl, dtype=float)
        coeffs[idx] = 1.0
        values[idx] = BSpline(knots, coeffs, degree, extrapolate=False)(samples)
    values[~np.isfinite(values)] = 0.0
    return values


def configure_matplotlib(use_tex: bool) -> None:
    import matplotlib

    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "text.usetex": use_tex,
        }
    )
    if use_tex:
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


def plot_basis(path: Path, tau: np.ndarray, basis: np.ndarray, knots: np.ndarray, degree: int, dpi: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 4.25))
    colors = plt.get_cmap("tab10").colors
    for idx, values in enumerate(basis):
        ax.plot(tau, values, linewidth=2.0, color=colors[idx % len(colors)], label=rf"$B_{{{idx},{degree}}}(\tau)$")

    unique_knots = np.unique(knots)
    internal_knots = unique_knots[(unique_knots > 0.0) & (unique_knots < 1.0)]
    for knot in internal_knots:
        ax.axvline(knot, color="0.78", linewidth=1.0, zorder=0)

    ax.plot(
        tau,
        np.sum(basis, axis=0),
        color="0.12",
        linewidth=1.5,
        linestyle="--",
        label=rf"$\sum B_{{i,{degree}}}(\tau)$",
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel(r"$\tau$ [-]")
    ax.set_ylabel(r"Basis value [-]")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    configure_matplotlib(bool(args.use_tex))
    knots = clamped_uniform_knots(int(args.n_ctrl), int(args.degree))
    tau = np.linspace(0.0, 1.0, int(args.samples))
    basis = basis_matrix(knots, int(args.degree), tau, int(args.n_ctrl))
    stem = args.stem or f"bspline_basis_nctrl{args.n_ctrl}_deg{args.degree}"

    for fmt in args.formats:
        path = output_dir / f"{stem}.{fmt}"
        plot_basis(path, tau, basis, knots, int(args.degree), int(args.dpi))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

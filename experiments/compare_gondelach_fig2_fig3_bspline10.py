"""Compare Gondelach time-driven cases and cylindrical B-spline.

The selected paper case is evaluated on a shared departure/TOF grid and N
branch range.  The B-spline case defaults to 10 control points in cylindrical
coordinates and uses formal Ipopt convergence only.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reproduce_gondelach_fig2 import (  # noqa: E402
    AU_KM,
    DAY_S,
    GONDELACH_FORMULATION_VERSION,
    build_coefficient_rows as build_fig2_coefficient_rows,
    compute_grid as compute_fig2_grid,
    evaluate_time_driven_reference_metrics,
    evaluate_time_driven_metrics,
    configure_ephemeris,
    ephemeris_metadata,
    inclusive_grid,
    planet_state,
    write_coefficients_npz as write_fig2_coefficients_npz,
)
from experiments.reproduce_gondelach_fig3 import (  # noqa: E402
    FIG3_BASIS,
    compute_grid as compute_fig3_grid,
    write_coefficients_npz as write_fig3_coefficients_npz,
)
from experiments.reference_metrics import REFERENCE_METRIC_VERSION  # noqa: E402
from optimizer.canonical_units import UA_AU_PER_YR2  # noqa: E402


FIG2_BASIS = "CPowPow2-CPowPow2-CosN5P3CosN5P3SinN5"
AU_PER_YR2_TO_M_PER_S2 = (AU_KM * 1000.0) / ((365.25 * DAY_S) ** 2)
MJD2000_EPOCH = datetime(2000, 1, 1, 12, 0, 0)
CASE_CONFIGS = {
    "mars": {
        "target": "mars",
        "display": "Mars",
        "dep_min": 7304.0,
        "dep_max": 10225.0,
        "tof_min": 500.0,
        "tof_max": 2000.0,
        "n_min": 0,
        "n_max": 5,
        "lower_basis": FIG2_BASIS,
        "higher_basis": FIG3_BASIS,
        "lower_label": "Gondelach Fig. 2",
        "higher_label": "Gondelach Fig. 3",
        "output_name": "mars",
    },
    "1989ml": {
        "target": "1989ml",
        "display": "1989ML",
        "dep_min": 7304.0,
        "dep_max": 10225.0,
        "tof_min": 100.0,
        "tof_max": 1000.0,
        "n_min": 0,
        "n_max": 2,
        "lower_basis": "CPowPow2-CPowPow2-CosN5PCosN5PSinN5",
        "higher_basis": "CPowPow2PSinPCos-CPowPow2PSinPCos-Cos05PCos05PSin05P6Cos05P6Sin05",
        "lower_label": "Gondelach Fig. 9",
        "higher_label": "Gondelach Fig. 10",
        "output_name": "1989ml",
    },
    "tempel1": {
        "target": "tempel1",
        "display": "Tempel 1",
        "dep_min": 0.0,
        "dep_max": 5845.0,
        "tof_min": 400.0,
        "tof_max": 1500.0,
        "n_min": 0,
        "n_max": 5,
        "lower_basis": "CPowPow2-CPowPow2-CosR5PCos05Sin05",
        "higher_basis": "CPowSin05PSinPCos-CPowSin05PSinPCos-Cos05Pow2Pow3P3Cos05P3Sin05",
        "lower_label": "Gondelach Tempel 1 low-order",
        "higher_label": "Gondelach Tempel 1 high-order",
        "output_name": "tempel1",
    },
    "mercury": {
        "target": "mercury",
        "display": "Mercury",
        "dep_min": 3285.0,
        "dep_max": 5475.0,
        "tof_min": 100.0,
        "tof_max": 1400.0,
        "n_min": 0,
        "n_max": 5,
        "lower_basis": "CPowPow2-CPowPow2-CosR5P6CosR5P6SinR5",
        "higher_basis": "CPowPow2PSinPCos-CPowPow2PSinPCos-Cos05Pow5Pow6P6Cos05P6Sin05",
        "lower_label": "Gondelach Mercury low-order",
        "higher_label": "Gondelach Mercury high-order",
        "output_name": "mercury",
    },
}


def namespace(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def apply_step_grid(args: argparse.Namespace) -> None:
    """Validate spacings and trim maxima to the last included grid values."""
    for axis in ("dep", "tof"):
        step = getattr(args, f"{axis}_step")
        step = float(step)
        if step <= 0.0:
            raise ValueError(f"--{axis}-step must be positive")
        vmin = float(getattr(args, f"{axis}_min"))
        vmax = float(getattr(args, f"{axis}_max"))
        values = np.arange(vmin, vmax + 0.5 * step, step, dtype=float)
        values = values[values <= vmax + 1e-9]
        if values.size == 0:
            raise ValueError(f"--{axis}-step produced an empty grid")
        setattr(args, f"{axis}_max", float(values[-1]))


def step_suffix(args: argparse.Namespace) -> str:
    parts = []
    if args.dep_step is not None:
        parts.append(f"dep{float(args.dep_step):g}d")
    if args.tof_step is not None:
        parts.append(f"tof{float(args.tof_step):g}d")
    return "" if not parts else "_" + "_".join(parts)


def finite_limits(*arrays: np.ndarray) -> tuple[float, float]:
    finite_parts = [np.asarray(arr, dtype=float)[np.isfinite(arr)] for arr in arrays]
    finite = np.concatenate([part for part in finite_parts if part.size]) if any(part.size for part in finite_parts) else np.array([])
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if abs(vmax - vmin) < 1e-12:
        pad = max(1.0, abs(vmin) * 0.1)
        return vmin - pad, vmax + pad
    return vmin, vmax


def canonical_accel_to_m_s2(value: float) -> float:
    return float(value) * UA_AU_PER_YR2 * AU_PER_YR2_TO_M_PER_S2


def pareto_front(points: list[dict]) -> list[dict]:
    usable = [
        point
        for point in points
        if np.isfinite(float(point.get("delta_v_km_s", np.nan)))
        and np.isfinite(float(point.get("fmax_m_s2", np.nan)))
    ]
    usable.sort(key=lambda point: (float(point["delta_v_km_s"]), float(point["fmax_m_s2"])))
    front = []
    best_fmax = float("inf")
    for point in usable:
        fmax = float(point["fmax_m_s2"])
        if fmax < best_fmax:
            front.append(point)
            best_fmax = fmax
    return front


def grid_extent(dep_grid: np.ndarray, tof_grid: np.ndarray) -> list[float]:
    dep_pad = 0.5 if len(dep_grid) == 1 else 0.0
    tof_pad = 0.5 if len(tof_grid) == 1 else 0.0
    return [
        float(dep_grid[0]) - dep_pad,
        float(dep_grid[-1]) + dep_pad,
        float(tof_grid[0]) - tof_pad,
        float(tof_grid[-1]) + tof_pad,
    ]


def grid_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("grid values must be a non-empty 1D array")
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
    mid = 0.5 * (values[:-1] + values[1:])
    first = values[0] - (mid[0] - values[0])
    last = values[-1] + (values[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def mjd2000_to_datetime(value: float) -> datetime:
    return MJD2000_EPOCH + timedelta(days=float(value))


def mjd2000_to_date_num(values: np.ndarray):
    import matplotlib.dates as mdates

    values = np.asarray(values, dtype=float)
    flat = [mjd2000_to_datetime(value) for value in values.ravel()]
    return np.asarray(mdates.date2num(flat), dtype=float).reshape(values.shape)


def configure_calendar_axes(ax, dep_grid: np.ndarray, tof_grid: np.ndarray) -> None:
    import matplotlib
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    dep_min = float(dep_grid[0])
    dep_max = float(dep_grid[-1])
    tof_day_min = float(tof_grid[0])
    tof_day_max = float(tof_grid[-1])
    dep_max_date = mjd2000_to_datetime(dep_max)
    dep_right = datetime(dep_max_date.year + 1, 1, 1)

    ax.set_xlim(float(mjd2000_to_date_num(np.array([dep_min]))[0]), mdates.date2num(dep_right))
    ax.set_ylim(tof_day_min, tof_day_max)
    # Both the calendar-date coordinates and TOF now use days.
    ax.set_aspect(1.0, adjustable="box")
    ax.xaxis.set_major_locator(mdates.YearLocator(base=2, month=1, day=1))
    use_tex = bool(matplotlib.rcParams.get("text.usetex", False))

    def date_label(value, _position):
        label = mdates.num2date(value).strftime("%Y-%m-%d")
        return label.replace("-", "{-}") if use_tex else label

    ax.xaxis.set_major_formatter(FuncFormatter(date_label))
    if np.isclose(tof_day_min, 100.0) and np.isclose(tof_day_max, 1000.0):
        ax.set_yticks([200.0, 400.0, 600.0, 800.0, 1000.0])
    elif np.isclose(tof_day_min, 400.0) and np.isclose(tof_day_max, 1500.0):
        ax.set_yticks([500.0, 1000.0, 1500.0])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.tick_params(axis="x", labelrotation=38)
    ax.grid(
        True,
        which="major",
        axis="both",
        color="black",
        linestyle="--",
        linewidth=0.75,
        alpha=0.3,
    )
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def configure_plot_fonts() -> None:
    import matplotlib

    use_tex = os.environ.get("FASTTRANSFER_USE_LATEX", "0") == "1"
    params = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.titlesize": 18,
        "figure.labelsize": 17,
        "legend.fontsize": 12,
        "text.usetex": use_tex,
    }
    if use_tex:
        params["text.latex.preamble"] = r"\usepackage{amsmath}"
    matplotlib.rcParams.update(params)


def plot_panel(
    axes,
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
    grids: list[np.ndarray],
    titles: list[str],
    path: Path,
    label: str,
    cmap,
    vmin: float,
    vmax: float,
    integer_ticks: bool = False,
    calendar_axes: bool = False,
    bin_edges: list[float] | None = None,
    bin_labels: list[str] | None = None,
    bin_tick_style: str = "buckets",
    colorbar_extend: str = "neither",
    smooth_regions: bool = False,
    panel_layout: str = "horizontal",
    departure_label_x_offset: float = 0.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    configure_plot_fonts()
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    image = None
    norm = None
    if bin_edges is not None:
        bin_edges = [float(edge) for edge in bin_edges]
        if len(bin_edges) < 2:
            raise ValueError("bin_edges must contain at least two values")
        extra_colors = 0 if colorbar_extend == "neither" else colorbar_extend.count("min") + colorbar_extend.count("max")
        if colorbar_extend == "both":
            extra_colors = 2
        color_count = len(bin_edges) - 1 + extra_colors
        if isinstance(cmap, str):
            cmap_obj = plt.get_cmap(cmap, color_count)
        elif getattr(cmap, "N", None) == color_count:
            cmap_obj = cmap
        else:
            cmap_obj = cmap.resampled(color_count)
        norm = BoundaryNorm(bin_edges, cmap_obj.N, clip=False, extend=colorbar_extend)
    else:
        cmap_obj = cmap
    if calendar_axes:
        dep_edges = grid_edges(dep_grid)
        tof_edges = grid_edges(tof_grid)
        x_edges = mjd2000_to_date_num(dep_edges)
        y_edges = tof_edges
    else:
        extent = grid_extent(dep_grid, tof_grid)
    for ax, grid, title in zip(axes, grids, titles):
        if calendar_axes and smooth_regions:
            x_centers = mjd2000_to_date_num(dep_grid)
            y_centers = tof_grid
            smooth_x = x_centers
            smooth_y = y_centers
            smooth_grid = grid
            if len(dep_grid) > 3 and len(tof_grid) > 3:
                try:
                    from scipy.interpolate import RegularGridInterpolator

                    smooth_x = np.linspace(float(x_centers[0]), float(x_centers[-1]), max(len(x_centers) * 4, len(x_centers)))
                    smooth_y = np.linspace(float(y_centers[0]), float(y_centers[-1]), max(len(y_centers) * 4, len(y_centers)))
                    interp = RegularGridInterpolator(
                        (y_centers, x_centers),
                        grid,
                        method="linear",
                        bounds_error=False,
                        fill_value=np.nan,
                    )
                    yy, xx = np.meshgrid(smooth_y, smooth_x, indexing="ij")
                    smooth_grid = interp(np.column_stack([yy.ravel(), xx.ravel()])).reshape(len(smooth_y), len(smooth_x))
                except Exception:
                    smooth_x = x_centers
                    smooth_y = y_centers
                    smooth_grid = grid
            image = ax.pcolormesh(
                grid_edges(smooth_x),
                grid_edges(smooth_y),
                smooth_grid,
                shading="auto",
                cmap=cmap_obj,
                norm=norm,
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
                rasterized=True,
                linewidth=0,
                edgecolors="none",
                antialiased=False,
            )
            configure_calendar_axes(ax, dep_grid, tof_grid)
        elif calendar_axes:
            image = ax.pcolormesh(
                x_edges,
                y_edges,
                grid,
                shading="auto",
                cmap=cmap_obj,
                norm=norm,
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
                rasterized=True,
                linewidth=0,
                edgecolors="none",
                antialiased=False,
            )
            configure_calendar_axes(ax, dep_grid, tof_grid)
        else:
            image = ax.imshow(
                grid,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap_obj,
                norm=norm,
                vmin=None if norm is not None else vmin,
                vmax=None if norm is not None else vmax,
                interpolation="nearest",
            )
            ax.set_xlabel("Departure date [MJD2000]")
            ax.set_ylabel("TOF [days]")
        ax.set_title(title)
    fig = axes[-1].figure
    if calendar_axes:
        if len(axes) > 1:
            if panel_layout == "vertical":
                fig.subplots_adjust(
                    left=0.15,
                    right=0.82,
                    bottom=0.14,
                    top=0.93,
                    hspace=0.0,
                )
                axes[0].set_anchor("S")
                axes[-1].set_anchor("N")
            else:
                fig.subplots_adjust(
                    left=0.10,
                    right=0.84,
                    bottom=0.28,
                    top=0.84,
                    wspace=0.20,
                )
            panel_center = 0.5 * (axes[0].get_position().x0 + axes[-1].get_position().x1)
            if fig._suptitle is not None:
                fig._suptitle.set_x(panel_center)
            if panel_layout == "vertical":
                axes[-1].set_xlabel("Departure date", labelpad=10)
            else:
                fig.supxlabel(
                    "Departure date",
                    x=panel_center + float(departure_label_x_offset),
                    y=0.12,
                )
            fig.supylabel(
                "Transfer time [days]",
                x=0.025,
                y=0.52 if panel_layout == "vertical" else 0.58,
            )
        else:
            fig.subplots_adjust(left=0.16, right=0.82, bottom=0.26, top=0.86)
            panel_center = 0.5 * (axes[0].get_position().x0 + axes[0].get_position().x1)
            if fig._suptitle is not None:
                fig._suptitle.set_x(panel_center)
            fig.supxlabel(
                "Departure date",
                x=panel_center + float(departure_label_x_offset),
                y=0.12,
            )
            fig.supylabel("Transfer time [days]", x=0.035, y=0.58)
    cbar_kwargs = {"shrink": 0.88, "pad": 0.03, "fraction": 0.035}
    if bin_edges is not None:
        cbar_kwargs["boundaries"] = bin_edges
        cbar_kwargs["extend"] = colorbar_extend
        if bin_tick_style == "boundaries":
            cbar_kwargs["ticks"] = bin_edges
        else:
            cbar_kwargs["ticks"] = [0.5 * (bin_edges[idx] + bin_edges[idx + 1]) for idx in range(len(bin_edges) - 1)]
    cbar = fig.colorbar(image, ax=axes, **cbar_kwargs)
    cbar.set_label(label, fontsize=15, labelpad=18 if bin_edges is not None else 6)
    cbar.ax.tick_params(labelsize=13)
    if bin_edges is not None and bin_labels is not None:
        cbar.ax.set_yticklabels(bin_labels)
    if integer_ticks:
        cbar.set_ticks(range(int(vmin), int(vmax) + 1))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(
    output_dir: Path,
    title_target: str,
    lower_label: str,
    higher_label: str,
    bspline_label: str,
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
    fig2_dv: np.ndarray,
    fig2_n: np.ndarray,
    fig3_dv: np.ndarray,
    fig3_n: np.ndarray,
    bspline_dv: np.ndarray,
    bspline_n: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    configure_plot_fonts()
    import matplotlib.pyplot as plt

    vmin, vmax = finite_limits(fig2_dv, fig3_dv, bspline_dv)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    fig.suptitle(f"{title_target} porkchop comparison, same grid and N branch range")
    plot_panel(
        axes,
        dep_grid,
        tof_grid,
        [fig2_dv, fig3_dv, bspline_dv],
        [lower_label, higher_label, bspline_label],
        output_dir / "comparison_delta_v.png",
        "Delta V [km/s]",
        "viridis_r",
        vmin,
        vmax,
    )

    vmin_3_bs, vmax_3_bs = finite_limits(fig3_dv, bspline_dv)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
    fig.suptitle(title_target)
    plot_panel(
        axes,
        dep_grid,
        tof_grid,
        [fig3_dv, bspline_dv],
        [higher_label, bspline_label],
        output_dir / "comparison_fig3_vs_bspline_delta_v.png",
        "Delta V [km/s]",
        "viridis_r",
        vmin_3_bs,
        vmax_3_bs,
        calendar_axes=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    fig.suptitle("Best selected branch N")
    plot_panel(
        axes,
        dep_grid,
        tof_grid,
        [fig2_n, fig3_n, bspline_n],
        [lower_label, higher_label, bspline_label],
        output_dir / "comparison_best_N.png",
        "Best N",
        "tab10",
        -0.5,
        5.5,
        integer_ticks=True,
    )

    diff_32 = fig3_dv - fig2_dv
    diff_b3 = bspline_dv - fig3_dv
    diff_b2 = bspline_dv - fig2_dv
    max_abs = max(abs(finite_limits(diff_32, diff_b3, diff_b2)[0]), abs(finite_limits(diff_32, diff_b3, diff_b2)[1]))
    if not np.isfinite(max_abs) or max_abs < 1e-12:
        max_abs = 1.0
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), sharex=True, sharey=True)
    fig.suptitle("Delta-V differences on the shared grid")
    plot_panel(
        axes,
        dep_grid,
        tof_grid,
        [diff_32, diff_b3, diff_b2],
        [f"{higher_label} - {lower_label}", f"{bspline_label} - {higher_label}", f"{bspline_label} - {lower_label}"],
        output_dir / "comparison_delta_v_differences.png",
        "Delta V difference [km/s]",
        "coolwarm",
        -max_abs,
        max_abs,
    )

    diff_b3_finite = diff_b3[np.isfinite(diff_b3)]
    if diff_b3_finite.size:
        diff_limit = float(np.nanmax(np.abs(diff_b3_finite)))
        if diff_limit < 1e-12:
            diff_limit = 1.0
    else:
        diff_limit = 1.0
    fig, axes = plt.subplots(1, 1, figsize=(6.4, 4.8))
    plot_panel(
        [axes],
        dep_grid,
        tof_grid,
        [diff_b3],
        [title_target],
        output_dir / "comparison_fig3_vs_bspline_delta_v_difference.png",
        "Delta V difference [km/s]",
        "coolwarm",
        -diff_limit,
        diff_limit,
        calendar_axes=True,
    )


def plot_pareto(
    path: Path,
    points_by_method: dict[str, list[dict]],
    x_max: float | None = None,
    method_labels: dict[str, str] | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "gondelach_fig2": "tab:blue",
        "gondelach_fig3": "tab:orange",
        "bspline10": "tab:green",
    }
    labels = {
        "gondelach_fig2": "Gondelach Fig. 2",
        "gondelach_fig3": "Gondelach Fig. 3",
        "bspline10": "B-spline cylindrical, 10 ctrl",
    }
    if method_labels:
        labels.update(method_labels)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for method, points in points_by_method.items():
        finite = [
            point
            for point in points
            if np.isfinite(float(point.get("delta_v_km_s", np.nan)))
            and np.isfinite(float(point.get("fmax_m_s2", np.nan)))
        ]
        if not finite:
            continue
        x = np.asarray([float(point["delta_v_km_s"]) for point in finite], dtype=float)
        y = np.asarray([float(point["fmax_m_s2"]) for point in finite], dtype=float)
        ax.scatter(x, y, s=18, alpha=0.28, color=colors.get(method), label=f"{labels.get(method, method)} attempts")

        front = pareto_front(finite)
        if front:
            fx = np.asarray([float(point["delta_v_km_s"]) for point in front], dtype=float)
            fy = np.asarray([float(point["fmax_m_s2"]) for point in front], dtype=float)
            order = np.argsort(fx)
            ax.plot(
                fx[order],
                fy[order],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=colors.get(method),
                label=f"{labels.get(method, method)} Pareto",
            )

    ax.set_xlabel("Delta V [km/s]")
    ax.set_ylabel("max thrust acceleration [m/s^2]")
    ax.set_title("Pareto comparison: Delta V vs max thrust acceleration")
    if x_max is not None and np.isfinite(float(x_max)) and float(x_max) > 0.0:
        ax.set_xlim(left=0.0, right=float(x_max))
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_best_csv(
    path: Path,
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
    fig2_dv: np.ndarray,
    fig2_n: np.ndarray,
    fig3_dv: np.ndarray,
    fig3_n: np.ndarray,
    bspline_dv: np.ndarray,
    bspline_n: np.ndarray,
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "departure_mjd2000",
                "tof_days",
                "fig2_delta_v_km_s",
                "fig2_best_N",
                "fig3_delta_v_km_s",
                "fig3_best_N",
                "bspline10_delta_v_km_s",
                "bspline10_best_N",
                "fig3_minus_fig2_km_s",
                "bspline10_minus_fig3_km_s",
                "bspline10_minus_fig2_km_s",
            ]
        )
        for i_tof, tof in enumerate(tof_grid):
            for i_dep, dep in enumerate(dep_grid):
                f2 = fig2_dv[i_tof, i_dep]
                f3 = fig3_dv[i_tof, i_dep]
                bs = bspline_dv[i_tof, i_dep]
                writer.writerow(
                    [
                        dep,
                        tof,
                        f2,
                        fig2_n[i_tof, i_dep],
                        f3,
                        fig3_n[i_tof, i_dep],
                        bs,
                        bspline_n[i_tof, i_dep],
                        f3 - f2,
                        bs - f3,
                        bs - f2,
                    ]
                )


def write_summary_csv(path: Path, grids: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "finite_points", "min_delta_v_km_s", "median_delta_v_km_s", "max_delta_v_km_s"])
        for name, grid in grids.items():
            finite = np.asarray(grid, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size:
                writer.writerow([name, finite.size, np.nanmin(finite), np.nanmedian(finite), np.nanmax(finite)])
            else:
                writer.writerow([name, 0, np.nan, np.nan, np.nan])


def fair_timing_metrics(
    attempt_rows: list[dict],
    parallel_wall_time_s: float,
    requested_workers: int = 1,
    used_workers: int | None = None,
    grid_points: int | None = None,
    branch_attempts: int | None = None,
) -> dict:
    times = []
    for row in attempt_rows:
        try:
            value = float(row.get("wall_time_s", np.nan))
        except (TypeError, ValueError):
            value = float("nan")
        if np.isfinite(value) and value >= 0.0:
            times.append(value)

    arr = np.asarray(times, dtype=float)
    wall = float(parallel_wall_time_s)
    if used_workers is None:
        used_workers = requested_workers
    requested_workers = int(requested_workers)
    used_workers = int(used_workers)
    if arr.size:
        sum_attempt = float(np.sum(arr))
        mean_attempt = float(np.mean(arr))
        median_attempt = float(np.median(arr))
        p95_attempt = float(np.percentile(arr, 95.0))
        max_attempt = float(np.max(arr))
    else:
        sum_attempt = mean_attempt = median_attempt = p95_attempt = max_attempt = float("nan")

    effective_parallelism = sum_attempt / wall if np.isfinite(sum_attempt) and wall > 0.0 else float("nan")
    parallel_efficiency = (
        effective_parallelism / used_workers
        if np.isfinite(effective_parallelism) and used_workers > 0
        else float("nan")
    )
    serial_per_grid = (
        sum_attempt / max(int(grid_points), 1)
        if grid_points is not None and np.isfinite(sum_attempt)
        else float("nan")
    )
    serial_per_branch = (
        sum_attempt / max(int(branch_attempts), 1)
        if branch_attempts is not None and np.isfinite(sum_attempt)
        else float("nan")
    )
    worker_normalized = wall * used_workers if np.isfinite(wall) and used_workers > 0 else float("nan")

    return {
        "requested_workers": requested_workers,
        "used_workers": used_workers,
        "parallel_wall_time_s": wall,
        "worker_normalized_wall_time_s": worker_normalized,
        "sum_attempt_wall_time_s": sum_attempt,
        "mean_attempt_wall_time_s": mean_attempt,
        "median_attempt_wall_time_s": median_attempt,
        "p95_attempt_wall_time_s": p95_attempt,
        "max_attempt_wall_time_s": max_attempt,
        "serial_equivalent_seconds_per_grid_point": serial_per_grid,
        "serial_equivalent_seconds_per_branch_attempt": serial_per_branch,
        "effective_parallelism": effective_parallelism,
        "parallel_efficiency": parallel_efficiency,
    }


def write_timing_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "method",
        "wall_time_s",
        "requested_workers",
        "used_workers",
        "parallel_wall_time_s",
        "worker_normalized_wall_time_s",
        "sum_attempt_wall_time_s",
        "mean_attempt_wall_time_s",
        "median_attempt_wall_time_s",
        "p95_attempt_wall_time_s",
        "max_attempt_wall_time_s",
        "serial_equivalent_seconds_per_grid_point",
        "serial_equivalent_seconds_per_branch_attempt",
        "effective_parallelism",
        "parallel_efficiency",
        "grid_points",
        "branch_attempts",
        "seconds_per_grid_point",
        "seconds_per_branch_attempt",
        "finite_points",
        "usable_attempts",
        "formal_success_attempts",
        "optimizer_function_evaluations",
        "notes",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_pareto_csv(path: Path, points_by_method: dict[str, list[dict]]) -> None:
    fieldnames = [
        "method",
        "is_pareto",
        "departure_mjd2000",
        "tof_days",
        "N",
        "delta_v_km_s",
        "delta_v_optimizer_km_s",
        "delta_v_reference_km_s",
        "delta_v_reference_error_km_s",
        "fmax_m_s2",
        "u_max_reference_m_s2",
        "u_max_reference_error_m_s2",
        "reference_quadrature_order",
        "reference_evaluations",
        "reference_converged",
        "reference_metric_version",
        "source_success",
        "message",
    ]
    front_ids = {
        method: {
            (point.get("departure_mjd2000"), point.get("tof_days"), point.get("N"), point.get("delta_v_km_s"), point.get("fmax_m_s2"))
            for point in pareto_front(points)
        }
        for method, points in points_by_method.items()
    }
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for method, points in points_by_method.items():
            for point in points:
                key = (
                    point.get("departure_mjd2000"),
                    point.get("tof_days"),
                    point.get("N"),
                    point.get("delta_v_km_s"),
                    point.get("fmax_m_s2"),
                )
                writer.writerow(
                    {
                        "method": method,
                        "is_pareto": key in front_ids[method],
                        "departure_mjd2000": point.get("departure_mjd2000", ""),
                        "tof_days": point.get("tof_days", ""),
                        "N": point.get("N", ""),
                        "delta_v_km_s": point.get("delta_v_km_s", ""),
                        "delta_v_optimizer_km_s": point.get("delta_v_optimizer_km_s", ""),
                        "delta_v_reference_km_s": point.get("delta_v_reference_km_s", ""),
                        "delta_v_reference_error_km_s": point.get("delta_v_reference_error_km_s", ""),
                        "fmax_m_s2": point.get("fmax_m_s2", ""),
                        "u_max_reference_m_s2": point.get("u_max_reference_m_s2", ""),
                        "u_max_reference_error_m_s2": point.get("u_max_reference_error_m_s2", ""),
                        "reference_quadrature_order": point.get("reference_quadrature_order", ""),
                        "reference_evaluations": point.get("reference_evaluations", ""),
                        "reference_converged": point.get("reference_converged", ""),
                        "reference_metric_version": point.get("reference_metric_version", ""),
                        "source_success": point.get("source_success", ""),
                        "message": point.get("message", ""),
                    }
                )


def read_pareto_csv(path: Path) -> dict[str, list[dict]]:
    points: dict[str, list[dict]] = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            method = str(row.get("method", ""))
            if not method:
                continue
            try:
                point = {
                    "departure_mjd2000": float(row["departure_mjd2000"]),
                    "tof_days": float(row["tof_days"]),
                    "N": int(float(row["N"])),
                    "delta_v_km_s": float(row["delta_v_km_s"]),
                    "fmax_m_s2": float(row["fmax_m_s2"]),
                    "source_success": str(row.get("source_success", "")).strip().lower()
                    in {"1", "true", "yes", "y"},
                    "message": row.get("message", ""),
                }
                for key in (
                    "delta_v_optimizer_km_s",
                    "delta_v_reference_km_s",
                    "delta_v_reference_error_km_s",
                    "u_max_reference_m_s2",
                    "u_max_reference_error_m_s2",
                ):
                    point[key] = float(row[key]) if row.get(key, "") != "" else float("nan")
                point["reference_quadrature_order"] = int(float(row.get("reference_quadrature_order", -1) or -1))
                point["reference_evaluations"] = int(float(row.get("reference_evaluations", 0) or 0))
                point["reference_converged"] = str(row.get("reference_converged", "")).strip().lower() in {"1", "true", "yes", "y"}
                point["reference_metric_version"] = row.get("reference_metric_version", "")
            except (KeyError, TypeError, ValueError):
                continue
            points.setdefault(method, []).append(point)
    return points


def read_timing_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def plot_timing(path: Path, rows: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["method"]) for row in rows]
    totals = np.asarray([float(row["wall_time_s"]) for row in rows], dtype=float)
    per_branch = np.asarray([float(row["seconds_per_branch_attempt"]) for row in rows], dtype=float)
    palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue", "tab:orange", "tab:green"])
    colors = [palette[idx % len(palette)] for idx in range(len(rows))]

    fig_width = max(11.5, 2.0 * len(rows) + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, 4.8))
    axes[0].bar(labels, totals, color=colors)
    axes[0].set_ylabel("Wall time [s]")
    axes[0].set_title("Total runtime")
    axes[0].tick_params(axis="x", rotation=28)
    for label in axes[0].get_xticklabels():
        label.set_horizontalalignment("right")

    axes[1].bar(labels, per_branch, color=colors)
    axes[1].set_ylabel("Wall time / branch attempt [s]")
    axes[1].set_title("Normalized by attempted N branch")
    axes[1].tick_params(axis="x", rotation=28)
    for label in axes[1].get_xticklabels():
        label.set_horizontalalignment("right")

    fig.suptitle("Computational time comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_bspline_attempts(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "departure_mjd2000",
        "tof_days",
        "N",
        "delta_v_km_s",
        "delta_v_optimizer_km_s",
        "delta_v_reference_km_s",
        "delta_v_reference_error_km_s",
        "success",
        "usable",
        "message",
        "endpoint_error",
        "winding_target_rev",
        "winding_sum_rev",
        "winding_error_rev",
        "winding_fine_rev",
        "u_max_reference_m_s2",
        "u_max_reference_error_m_s2",
        "reference_quadrature_order",
        "reference_evaluations",
        "reference_converged",
        "reference_metric_version",
        "wall_time_s",
        "boundary_control_points_fixed",
        "boundary_control_points_eliminated",
        "n_free_control_points",
        "linear_solver",
        "coinhsl_library",
        "ipopt_iterations",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["tof_days"], item["departure_mjd2000"], item["N"])):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_fig3_attempts(path: Path, rows: list[dict]) -> None:
    max_free = max((len(row.get("free_coefficients", [])) for row in rows), default=0)
    fieldnames = [
        "departure_mjd2000",
        "tof_days",
        "N",
        "delta_v_km_s",
        "delta_v_optimizer_km_s",
        "delta_v_reference_km_s",
        "delta_v_reference_error_km_s",
        "u_max_reference_m_s2",
        "u_max_reference_error_m_s2",
        "reference_quadrature_order",
        "reference_evaluations",
        "reference_converged",
        "reference_metric_version",
        "start_delta_v_km_s",
        "optimizer",
        "optimizer_success",
        "usable",
        "nfev",
        "nit",
        "num_restarts",
        "wall_time_s",
        "message",
        *[f"free_{idx}" for idx in range(max_free)],
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key, "") for key in fieldnames}
            for idx, value in enumerate(row.get("free_coefficients", [])):
                out[f"free_{idx}"] = float(value)
            writer.writerow(out)


def collect_fig2_pareto_points(args: argparse.Namespace, dep_grid: np.ndarray, tof_grid: np.ndarray) -> list[dict]:
    points = []
    for tof in tof_grid:
        for dep in dep_grid:
            for n_rev in range(args.n_min, args.n_max + 1):
                try:
                    metrics = evaluate_time_driven_metrics(
                        float(dep),
                        float(tof),
                        int(n_rev),
                        int(args.fig2_n_quad),
                        args.lower_basis,
                        target=args.target,
                    )
                    points.append(
                        {
                            "departure_mjd2000": float(dep),
                            "tof_days": float(tof),
                            "N": int(n_rev),
                            "delta_v_km_s": float(metrics["delta_v_km_s"]),
                            "fmax_m_s2": float(metrics["fmax_m_s2"]),
                            "source_success": True,
                            "message": "",
                        }
                    )
                except Exception as exc:
                    points.append(
                        {
                            "departure_mjd2000": float(dep),
                            "tof_days": float(tof),
                            "N": int(n_rev),
                            "delta_v_km_s": float("nan"),
                            "fmax_m_s2": float("nan"),
                            "source_success": False,
                            "message": str(exc).splitlines()[-1],
                        }
                    )
    return points


def apply_gondelach_reference_metrics(args: argparse.Namespace, rows: list[dict]) -> None:
    for row in rows:
        row["delta_v_optimizer_km_s"] = float(
            row.get("delta_v_optimizer_km_s", row.get("delta_v_km_s", np.nan))
        )
        if not bool(row.get("usable", False)):
            continue
        try:
            reference = evaluate_time_driven_reference_metrics(
                float(row["departure_mjd2000"]),
                float(row["tof_days"]),
                int(row["N"]),
                int(args.fig3_n_quad),
                str(args.higher_basis),
                np.asarray(row.get("free_coefficients", []), dtype=float),
                target=str(args.target),
            )
            row.update(reference)
            row["delta_v_km_s"] = float(reference["delta_v_reference_km_s"])
            row["fmax_m_s2"] = float(reference["u_max_reference_m_s2"])
            row["usable"] = bool(
                np.isfinite(row["delta_v_km_s"])
                and np.isfinite(row["fmax_m_s2"])
            )
        except Exception as exc:
            row["usable"] = False
            row["delta_v_km_s"] = float("nan")
            row["fmax_m_s2"] = float("nan")
            row["message"] = f"{row.get('message', '')}; reference: {str(exc).splitlines()[-1]}".strip("; ")


def metric_grids_from_rows(
    rows: list[dict],
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
    metric_key: str = "delta_v_km_s",
) -> tuple[np.ndarray, np.ndarray]:
    delta_v = np.full((len(tof_grid), len(dep_grid)), np.nan)
    best_n = np.full((len(tof_grid), len(dep_grid)), -1, dtype=int)
    by_grid: dict[tuple[float, float], list[dict]] = {}
    for row in rows:
        if not bool(row.get("usable", False)):
            continue
        value = float(row.get(metric_key, np.nan))
        if not np.isfinite(value):
            continue
        by_grid.setdefault((float(row["departure_mjd2000"]), float(row["tof_days"])), []).append(row)
    for i_tof, tof in enumerate(tof_grid):
        for i_dep, dep in enumerate(dep_grid):
            candidates = by_grid.get((float(dep), float(tof)), [])
            if candidates:
                best = min(candidates, key=lambda item: float(item[metric_key]))
                delta_v[i_tof, i_dep] = float(best[metric_key])
                best_n[i_tof, i_dep] = int(best["N"])
    return delta_v, best_n


def write_spice_boundary_states(
    path: Path,
    target: str,
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
) -> None:
    earth_position = np.empty((len(dep_grid), 3), dtype=float)
    earth_velocity = np.empty((len(dep_grid), 3), dtype=float)
    target_position = np.empty((len(tof_grid), len(dep_grid), 3), dtype=float)
    target_velocity = np.empty_like(target_position)
    for i_dep, departure in enumerate(dep_grid):
        earth_position[i_dep], earth_velocity[i_dep] = planet_state("earth", float(departure))
        for i_tof, tof in enumerate(tof_grid):
            target_position[i_tof, i_dep], target_velocity[i_tof, i_dep] = planet_state(
                target, float(departure + tof)
            )
    np.savez(
        path,
        departure_mjd2000=np.asarray(dep_grid, dtype=float),
        tof_days=np.asarray(tof_grid, dtype=float),
        earth_position_au=earth_position,
        earth_velocity_au_day=earth_velocity,
        target_position_au=target_position,
        target_velocity_au_day=target_velocity,
        target=np.asarray(target),
        **{key: np.asarray(value) for key, value in ephemeris_metadata().items()},
    )


def collect_fig3_pareto_points(args: argparse.Namespace, rows: list[dict]) -> list[dict]:
    points = []
    for row in rows:
        if not bool(row.get("usable", False)):
            continue
        points.append(
            {
                "departure_mjd2000": float(row["departure_mjd2000"]),
                "tof_days": float(row["tof_days"]),
                "N": int(row["N"]),
                "delta_v_km_s": float(row.get("delta_v_km_s", np.nan)),
                "delta_v_optimizer_km_s": float(row.get("delta_v_optimizer_km_s", np.nan)),
                "delta_v_reference_km_s": float(row.get("delta_v_reference_km_s", np.nan)),
                "delta_v_reference_error_km_s": float(row.get("delta_v_reference_error_km_s", np.nan)),
                "fmax_m_s2": float(row.get("u_max_reference_m_s2", np.nan)),
                "u_max_reference_m_s2": float(row.get("u_max_reference_m_s2", np.nan)),
                "u_max_reference_error_m_s2": float(row.get("u_max_reference_error_m_s2", np.nan)),
                "reference_quadrature_order": int(row.get("reference_quadrature_order", -1)),
                "reference_evaluations": int(row.get("reference_evaluations", 0)),
                "reference_converged": bool(row.get("reference_converged", False)),
                "reference_metric_version": str(row.get("reference_metric_version", "")),
                "source_success": bool(row.get("optimizer_success", False)),
                "message": row.get("message", ""),
            }
        )
    return points


def collect_bspline_pareto_points(rows: list[dict]) -> list[dict]:
    points = []
    for row in rows:
        if not bool(row.get("usable", False)):
            continue
        points.append(
            {
                "departure_mjd2000": float(row["departure_mjd2000"]),
                "tof_days": float(row["tof_days"]),
                "N": int(row["N"]),
                "delta_v_km_s": float(row.get("delta_v_km_s", np.nan)),
                "delta_v_optimizer_km_s": float(row.get("delta_v_optimizer_km_s", np.nan)),
                "delta_v_reference_km_s": float(row.get("delta_v_reference_km_s", np.nan)),
                "delta_v_reference_error_km_s": float(row.get("delta_v_reference_error_km_s", np.nan)),
                "fmax_m_s2": float(row.get("u_max_reference_m_s2", np.nan)),
                "u_max_reference_m_s2": float(row.get("u_max_reference_m_s2", np.nan)),
                "u_max_reference_error_m_s2": float(row.get("u_max_reference_error_m_s2", np.nan)),
                "reference_quadrature_order": int(row.get("reference_quadrature_order", -1)),
                "reference_evaluations": int(row.get("reference_evaluations", 0)),
                "reference_converged": bool(row.get("reference_converged", False)),
                "reference_metric_version": str(row.get("reference_metric_version", "")),
                "source_success": bool(row.get("success", False)),
                "message": row.get("message", ""),
            }
        )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_CONFIGS), default="mars")
    parser.add_argument("--dep-min", type=float, default=None)
    parser.add_argument("--dep-max", type=float, default=None)
    parser.add_argument("--tof-min", type=float, default=None)
    parser.add_argument("--tof-max", type=float, default=None)
    parser.add_argument("--dep-step", type=float, default=20.0, help="Departure-date spacing in days.")
    parser.add_argument("--tof-step", type=float, default=20.0, help="Time-of-flight spacing in days.")
    parser.add_argument("--n-min", type=int, default=None)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--fig2-n-quad", type=int, default=401)
    parser.add_argument("--fig3-n-quad", type=int, default=51)
    parser.add_argument("--fig3-maxfev", type=int, default=1000)
    parser.add_argument("--fig3-xatol", type=float, default=1e-6)
    parser.add_argument("--fig3-fatol", type=float, default=1e-6)
    parser.add_argument("--fig3-simplex-scale", type=float, default=1e-2)
    parser.add_argument("--fig3-optimizer", choices=["scipy", "oneill"], default="scipy")
    parser.add_argument("--fig3-oneill-reqmin", type=float, default=1e-8)
    parser.add_argument("--fig3-oneill-konvge", type=int, default=10)
    parser.add_argument("--fig3-oneill-factorial-epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--higher-basis",
        default=None,
        help="Override the case's high-order hodographic basis (for controlled experiments).",
    )
    parser.add_argument("--bspline-n-ctrl", type=int, default=10)
    parser.add_argument("--bspline-degree", type=int, default=5)
    parser.add_argument("--bspline-n-fine", type=int, default=400)
    parser.add_argument("--bspline-max-iter", type=int, default=700)
    parser.add_argument("--bspline-workers", type=int, default=1)
    parser.add_argument(
        "--bspline-linear-solver",
        choices=["mumps", "ma27", "ma57", "ma77", "ma86", "ma97"],
        default="mumps",
    )
    parser.add_argument(
        "--coinhsl-library",
        default=None,
        help="Path to libcoinhsl.dylib; required when an HSL linear solver is selected.",
    )
    parser.add_argument("--ephemeris", choices=["kepler", "spice"], default="kepler")
    parser.add_argument("--spice-meta-kernel", default=None)
    parser.add_argument(
        "--spice-target-name",
        default=None,
        help="Optional SPICE target name or NAIF ID; Earth is always queried as EARTH.",
    )
    parser.add_argument(
        "--bspline-only",
        action="store_true",
        help=(
            "Recompute only the canonical B-spline baseline, reusing the saved Gondelach "
            "grids and Pareto data in --output-dir."
        ),
    )
    parser.add_argument(
        "--gondelach-only",
        action="store_true",
        help=(
            "Recompute and overwrite only the Gondelach results, reusing the saved "
            "canonical B-spline baseline in --output-dir without running IPOPT."
        ),
    )
    parser.add_argument("--accept-debug-feasible", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.bspline_only and args.gondelach_only:
        parser.error("--bspline-only and --gondelach-only are mutually exclusive")
    if not args.gondelach_only and args.bspline_linear_solver != "mumps":
        if not args.coinhsl_library:
            parser.error("--coinhsl-library is required with an HSL B-spline linear solver")
        if not Path(args.coinhsl_library).expanduser().is_file():
            parser.error(f"CoinHSL library not found: {args.coinhsl_library}")

    case = CASE_CONFIGS[args.case]
    args.target = str(case["target"])
    args.dep_min = float(case["dep_min"] if args.dep_min is None else args.dep_min)
    args.dep_max = float(case["dep_max"] if args.dep_max is None else args.dep_max)
    args.tof_min = float(case["tof_min"] if args.tof_min is None else args.tof_min)
    args.tof_max = float(case["tof_max"] if args.tof_max is None else args.tof_max)
    args.n_min = int(case["n_min"] if args.n_min is None else args.n_min)
    args.n_max = int(case["n_max"] if args.n_max is None else args.n_max)
    args.lower_basis = str(case["lower_basis"])
    args.higher_basis = str(case["higher_basis"] if args.higher_basis is None else args.higher_basis)
    configure_ephemeris(args.ephemeris, args.spice_meta_kernel, args.spice_target_name)
    apply_step_grid(args)
    if args.output_dir is None:
        args.output_dir = f"output/compare_gondelach_{case['output_name']}_bspline10{step_suffix(args)}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    script_t0 = perf_counter()

    if args.progress:
        dep_count = len(inclusive_grid(args.dep_min, args.dep_max, args.dep_step))
        tof_count = len(inclusive_grid(args.tof_min, args.tof_max, args.tof_step))
        print(
            "Grid: "
            f"{dep_count} departure dates from {args.dep_min:g} to {args.dep_max:g}, "
            f"{tof_count} TOFs from {args.tof_min:g} to {args.tof_max:g}, "
            f"N={args.n_min}..{args.n_max}"
        )
    fig2_args = namespace(
        dep_min=args.dep_min,
        dep_max=args.dep_max,
        tof_min=args.tof_min,
        tof_max=args.tof_max,
        dep_step=args.dep_step,
        tof_step=args.tof_step,
        n_quad=args.fig2_n_quad,
        n_min=args.n_min,
        n_max=args.n_max,
        basis=args.lower_basis,
        target=args.target,
        progress=False,
        ephemeris=args.ephemeris,
        spice_meta_kernel=args.spice_meta_kernel,
        spice_target_name=args.spice_target_name,
    )
    fig3_args = namespace(
        dep_min=args.dep_min,
        dep_max=args.dep_max,
        tof_min=args.tof_min,
        tof_max=args.tof_max,
        dep_step=args.dep_step,
        tof_step=args.tof_step,
        n_quad=args.fig3_n_quad,
        n_min=args.n_min,
        n_max=args.n_max,
        basis=args.higher_basis,
        target=args.target,
        maxfev=args.fig3_maxfev,
        xatol=args.fig3_xatol,
        fatol=args.fig3_fatol,
        simplex_scale=args.fig3_simplex_scale,
        optimizer=args.fig3_optimizer,
        oneill_reqmin=args.fig3_oneill_reqmin,
        oneill_konvge=args.fig3_oneill_konvge,
        oneill_factorial_epsilon=args.fig3_oneill_factorial_epsilon,
        progress=args.progress,
        progress_every=args.progress_every,
        ephemeris=args.ephemeris,
        spice_meta_kernel=args.spice_meta_kernel,
        spice_target_name=args.spice_target_name,
    )

    if args.bspline_only:
        grids_path = output_dir / "comparison_grids.npz"
        if not grids_path.exists():
            raise FileNotFoundError(
                f"--bspline-only requires the cached Gondelach grid: {grids_path}"
            )
        with np.load(grids_path) as cached:
            cached_formulation = (
                str(np.asarray(cached["gondelach_formulation_version"]).item())
                if "gondelach_formulation_version" in cached.files
                else ""
            )
            if cached_formulation != GONDELACH_FORMULATION_VERSION:
                raise ValueError(
                    "Cached Gondelach results use an obsolete formulation. "
                    "Rerun without --bspline-only to regenerate them."
                )
            dep_grid = np.asarray(cached["departure_mjd2000"], dtype=float)
            tof_grid = np.asarray(cached["tof_days"], dtype=float)
            fig2_dv = np.asarray(cached["fig2_delta_v_km_s"], dtype=float)
            fig2_n = np.asarray(cached["fig2_best_N"], dtype=int)
            fig3_dv = np.asarray(cached["fig3_delta_v_km_s"], dtype=float)
            fig3_n = np.asarray(cached["fig3_best_N"], dtype=int)
            fig2_optimizer_dv = np.asarray(
                cached["fig2_optimizer_delta_v_km_s"]
                if "fig2_optimizer_delta_v_km_s" in cached.files
                else fig2_dv,
                dtype=float,
            )
            fig3_optimizer_dv = np.asarray(
                cached["fig3_optimizer_delta_v_km_s"]
                if "fig3_optimizer_delta_v_km_s" in cached.files
                else fig3_dv,
                dtype=float,
            )
        expected_dep = inclusive_grid(args.dep_min, args.dep_max, args.dep_step)
        expected_tof = inclusive_grid(args.tof_min, args.tof_max, args.tof_step)
        if not np.allclose(dep_grid, expected_dep) or not np.allclose(tof_grid, expected_tof):
            raise ValueError(
                "The requested departure/TOF grid does not match the cached Gondelach grid."
            )
        fig2_time = 0.0
        fig3_time = 0.0
        fig3_rows: list[dict] = []
        if args.progress:
            print(f"Reusing cached Gondelach grids from {grids_path}")
    else:
        if args.progress:
            print("Computing Gondelach Fig. 2 grid...")
        t0 = perf_counter()
        dep_grid, tof_grid, fig2_dv, fig2_n = compute_fig2_grid(fig2_args)
        fig2_time = perf_counter() - t0
        fig2_optimizer_dv = np.asarray(fig2_dv, dtype=float).copy()

        if args.progress:
            print("Computing Gondelach Fig. 3 grid...")
        t0 = perf_counter()
        _, _, fig3_rows, fig3_dv, fig3_n = compute_fig3_grid(fig3_args)
        fig3_time = perf_counter() - t0
        fig3_optimizer_dv = np.asarray(fig3_dv, dtype=float).copy()

    from experiments.reproduce_gondelach_fig2_bspline_cylindrical import (
        run_tasks as run_bspline_tasks,
        write_control_points_npz,
    )

    bspline_args = namespace(
        dep_min=args.dep_min,
        dep_max=args.dep_max,
        tof_min=args.tof_min,
        tof_max=args.tof_max,
        dep_step=args.dep_step,
        tof_step=args.tof_step,
        n_min=args.n_min,
        n_max=args.n_max,
        target=args.target,
        n_ctrl=args.bspline_n_ctrl,
        degree=args.bspline_degree,
        n_fine=args.bspline_n_fine,
        seed_profile="quintic",
        quadrature_order=6,
        r_bound=20.0,
        dv_eps=1e-6,
        smoothness_weight=0.0,
        endpoint_control_weight=0.0,
        max_iter=args.bspline_max_iter,
        linear_solver=args.bspline_linear_solver,
        coinhsl_library=args.coinhsl_library,
        print_level=0,
        kepler_substeps=80,
        accept_debug_feasible=args.accept_debug_feasible,
        endpoint_tol=1e-6,
        winding_tol_rev=1e-3,
        workers=args.bspline_workers,
        progress=args.progress,
        progress_every=args.progress_every,
        ephemeris=args.ephemeris,
        spice_meta_kernel=args.spice_meta_kernel,
        spice_target_name=args.spice_target_name,
    )
    cached_bspline_pareto_rows: list[dict] | None = None
    if args.gondelach_only:
        grids_path = output_dir / "comparison_grids.npz"
        pareto_path = output_dir / "comparison_pareto_points.csv"
        if not grids_path.exists() or not pareto_path.exists():
            raise FileNotFoundError(
                "--gondelach-only requires existing comparison_grids.npz and "
                "comparison_pareto_points.csv files containing the B-spline baseline."
            )
        with np.load(grids_path) as cached:
            cached_dep = np.asarray(cached["departure_mjd2000"], dtype=float)
            cached_tof = np.asarray(cached["tof_days"], dtype=float)
            if not np.allclose(dep_grid, cached_dep) or not np.allclose(tof_grid, cached_tof):
                raise ValueError(
                    "The requested departure/TOF grid does not match the cached B-spline grid."
                )
            bspline_dv = np.asarray(cached["bspline10_delta_v_km_s"], dtype=float)
            bspline_n = np.asarray(cached["bspline10_best_N"], dtype=int)
            bspline_optimizer_dv = np.asarray(
                cached["bspline10_optimizer_delta_v_km_s"]
                if "bspline10_optimizer_delta_v_km_s" in cached.files
                else bspline_dv,
                dtype=float,
            )
        cached_pareto = read_pareto_csv(pareto_path)
        if not cached_pareto.get("bspline10"):
            raise ValueError("Cached Pareto data do not contain the B-spline baseline")
        cached_bspline_pareto_rows = cached_pareto["bspline10"]
        bspline_rows: list[dict] = []
        bspline_time = 0.0
        if args.progress:
            print(f"Reusing cached B-spline baseline from {grids_path}")
    else:
        if args.progress:
            print("Computing cylindrical B-spline grid with 10 control points...")
        t0 = perf_counter()
        _, _, bspline_rows, bspline_dv, bspline_n = run_bspline_tasks(bspline_args)
        bspline_time = perf_counter() - t0
        bspline_optimizer_dv, _ = metric_grids_from_rows(
            bspline_rows,
            dep_grid,
            tof_grid,
            metric_key="delta_v_optimizer_km_s",
        )

    if args.bspline_only:
        pareto_path = output_dir / "comparison_pareto_points.csv"
        if not pareto_path.exists():
            raise FileNotFoundError(
                f"--bspline-only requires the cached Gondelach Pareto data: {pareto_path}"
            )
        cached_pareto = read_pareto_csv(pareto_path)
        try:
            fig2_coeff_rows = cached_pareto["gondelach_fig2"]
            fig3_pareto_rows = cached_pareto["gondelach_fig3"]
        except KeyError as exc:
            raise ValueError(f"Missing cached Pareto method {exc.args[0]!r} in {pareto_path}") from exc
        fig2_coeff_build_time = 0.0
        fig3_pareto_time = 0.0
        if args.progress:
            print(f"Reusing cached Gondelach Pareto data from {pareto_path}")
    else:
        if args.progress:
            print("Collecting low-order Pareto metrics and coefficients...")
        t0 = perf_counter()
        fig2_coeff_rows = build_fig2_coefficient_rows(dep_grid, tof_grid, fig2_args)
        fig2_coeff_build_time = perf_counter() - t0
        fig2_dv, fig2_n = metric_grids_from_rows(fig2_coeff_rows, dep_grid, tof_grid)

        if args.progress:
            print("Applying shared reference metrics to high-order solutions...")
        t0 = perf_counter()
        apply_gondelach_reference_metrics(args, fig3_rows)
        fig3_dv, fig3_n = metric_grids_from_rows(fig3_rows, dep_grid, tof_grid)
        fig3_pareto_rows = collect_fig3_pareto_points(args, fig3_rows)
        fig3_pareto_time = perf_counter() - t0

    if args.gondelach_only:
        bspline_pareto_rows = cached_bspline_pareto_rows or []
        bspline_pareto_time = 0.0
    else:
        t0 = perf_counter()
        bspline_pareto_rows = collect_bspline_pareto_points(bspline_rows)
        bspline_pareto_time = perf_counter() - t0
    pareto_points = {
        "gondelach_fig2": fig2_coeff_rows,
        "gondelach_fig3": fig3_pareto_rows,
        "bspline10": bspline_pareto_rows,
    }

    grid_points = int(len(dep_grid) * len(tof_grid))
    branch_attempts = int(grid_points * (args.n_max - args.n_min + 1))
    timing_rows = [
        {
            "method": "gondelach_fig2",
            "wall_time_s": fig2_time,
            **fair_timing_metrics([], fig2_time, requested_workers=1, used_workers=1, grid_points=grid_points, branch_attempts=branch_attempts),
            "grid_points": grid_points,
            "branch_attempts": branch_attempts,
            "seconds_per_grid_point": fig2_time / max(grid_points, 1),
            "seconds_per_branch_attempt": fig2_time / max(branch_attempts, 1),
            "finite_points": int(np.isfinite(fig2_dv).sum()),
            "usable_attempts": "",
            "formal_success_attempts": "",
            "optimizer_function_evaluations": "",
            "notes": "direct lower-order evaluator",
        },
        {
            "method": "gondelach_fig3",
            "wall_time_s": fig3_time,
            **fair_timing_metrics(fig3_rows, fig3_time, requested_workers=1, used_workers=1, grid_points=grid_points, branch_attempts=branch_attempts),
            "grid_points": grid_points,
            "branch_attempts": branch_attempts,
            "seconds_per_grid_point": fig3_time / max(grid_points, 1),
            "seconds_per_branch_attempt": fig3_time / max(branch_attempts, 1),
            "finite_points": int(np.isfinite(fig3_dv).sum()),
            "usable_attempts": int(sum(bool(row.get("usable", False)) for row in fig3_rows)),
            "formal_success_attempts": int(sum(bool(row.get("optimizer_success", False)) for row in fig3_rows)),
            "optimizer_function_evaluations": int(sum(int(row.get("nfev", 0)) for row in fig3_rows)),
            "notes": f"{args.fig3_optimizer} Nelder-Mead optimized higher-order evaluator",
        },
        {
            "method": f"bspline_cylindrical_nctrl{args.bspline_n_ctrl}",
            "wall_time_s": bspline_time,
            **fair_timing_metrics(
                bspline_rows,
                bspline_time,
                requested_workers=int(args.bspline_workers),
                used_workers=int(args.bspline_workers),
                grid_points=grid_points,
                branch_attempts=branch_attempts,
            ),
            "grid_points": grid_points,
            "branch_attempts": branch_attempts,
            "seconds_per_grid_point": bspline_time / max(grid_points, 1),
            "seconds_per_branch_attempt": bspline_time / max(branch_attempts, 1),
            "finite_points": int(np.isfinite(bspline_dv).sum()),
            "usable_attempts": int(sum(bool(row.get("usable", False)) for row in bspline_rows)),
            "formal_success_attempts": int(sum(bool(row.get("success", False)) for row in bspline_rows)),
            "optimizer_function_evaluations": "",
            "notes": (
                f"wall time includes ProcessPool overhead; workers={args.bspline_workers}; "
                f"linear_solver={args.bspline_linear_solver}"
            ),
        },
    ]
    if args.bspline_only:
        preserved_timing = [
            row
            for row in read_timing_csv(output_dir / "comparison_timing.csv")
            if str(row.get("method", "")).startswith("gondelach_")
        ]
        timing_rows = preserved_timing + [timing_rows[-1]]
    elif args.gondelach_only:
        preserved_bspline_timing = [
            row
            for row in read_timing_csv(output_dir / "comparison_timing.csv")
            if str(row.get("method", "")).startswith("bspline_cylindrical_")
        ]
        timing_rows = timing_rows[:2] + preserved_bspline_timing

    write_best_csv(
        output_dir / "comparison_best.csv",
        dep_grid,
        tof_grid,
        fig2_dv,
        fig2_n,
        fig3_dv,
        fig3_n,
        bspline_dv,
        bspline_n,
    )
    write_summary_csv(
        output_dir / "comparison_summary.csv",
        {
            f"{args.case}_gondelach_low_order": fig2_dv,
            f"{args.case}_gondelach_high_order": fig3_dv,
            f"bspline_cylindrical_nctrl{args.bspline_n_ctrl}": bspline_dv,
        },
    )
    write_pareto_csv(output_dir / "comparison_pareto_points.csv", pareto_points)
    if args.bspline_only:
        fig2_coeff_write_time = 0.0
        fig3_coeff_time = 0.0
    else:
        write_fig3_attempts(output_dir / "fig3_attempts.csv", fig3_rows)
        t0 = perf_counter()
        write_fig2_coefficients_npz(
            output_dir / "gondelach_low_order_coefficients.npz",
            fig2_coeff_rows,
            dep_grid,
            tof_grid,
            fig2_args,
        )
        fig2_coeff_write_time = perf_counter() - t0

        t0 = perf_counter()
        write_fig3_coefficients_npz(
            output_dir / "gondelach_high_order_coefficients.npz",
            fig3_rows,
            dep_grid,
            tof_grid,
            fig3_args,
        )
        fig3_coeff_time = perf_counter() - t0

    if not args.gondelach_only:
        write_bspline_attempts(output_dir / "bspline10_attempts.csv", bspline_rows)
        write_control_points_npz(
            output_dir / "bspline10_control_points.npz",
            bspline_rows,
            dep_grid,
            tof_grid,
            bspline_args,
        )
    ephemeris_info = ephemeris_metadata()
    np.savez(
        output_dir / "comparison_grids.npz",
        departure_mjd2000=dep_grid,
        tof_days=tof_grid,
        fig2_delta_v_km_s=fig2_dv,
        fig2_best_N=fig2_n,
        fig2_optimizer_delta_v_km_s=fig2_optimizer_dv,
        fig3_delta_v_km_s=fig3_dv,
        fig3_best_N=fig3_n,
        fig3_optimizer_delta_v_km_s=fig3_optimizer_dv,
        fig3_basis=np.asarray(args.higher_basis),
        fig3_optimizer=np.asarray(args.fig3_optimizer),
        fig3_maxfev=np.asarray(args.fig3_maxfev, dtype=int),
        fig3_simplex_scale=np.asarray(args.fig3_simplex_scale, dtype=float),
        fig3_oneill_reqmin=np.asarray(args.fig3_oneill_reqmin, dtype=float),
        fig3_oneill_konvge=np.asarray(args.fig3_oneill_konvge, dtype=int),
        fig3_oneill_factorial_epsilon=np.asarray(
            args.fig3_oneill_factorial_epsilon, dtype=float
        ),
        bspline10_delta_v_km_s=bspline_dv,
        bspline10_best_N=bspline_n,
        bspline10_optimizer_delta_v_km_s=bspline_optimizer_dv,
        comparison_metric=np.asarray("reference"),
        reference_metric_version=np.asarray(REFERENCE_METRIC_VERSION),
        gondelach_formulation_version=np.asarray(GONDELACH_FORMULATION_VERSION),
        **{key: np.asarray(value) for key, value in ephemeris_info.items()},
    )
    if args.ephemeris == "spice":
        write_spice_boundary_states(
            output_dir / "spice_boundary_states.npz",
            args.target,
            dep_grid,
            tof_grid,
        )
    postprocess_pareto_time = fig3_pareto_time + bspline_pareto_time
    fig2_coeff_time = fig2_coeff_build_time + fig2_coeff_write_time
    timing_rows.extend(
        [
            {
                "method": "gondelach_fig2_coefficients",
                "wall_time_s": fig2_coeff_time,
                "grid_points": grid_points,
                "branch_attempts": branch_attempts,
                "seconds_per_grid_point": fig2_coeff_time / max(grid_points, 1),
                "seconds_per_branch_attempt": fig2_coeff_time / max(branch_attempts, 1),
                "finite_points": "",
                "usable_attempts": int(sum(bool(row.get("usable", False)) for row in fig2_coeff_rows)),
                "formal_success_attempts": int(sum(bool(row.get("source_success", False)) for row in fig2_coeff_rows)),
                "optimizer_function_evaluations": "",
                "notes": "low-order Pareto metrics plus coefficient archive",
            },
            {
                "method": "gondelach_fig3_coefficients",
                "wall_time_s": fig3_coeff_time,
                "grid_points": grid_points,
                "branch_attempts": branch_attempts,
                "seconds_per_grid_point": fig3_coeff_time / max(grid_points, 1),
                "seconds_per_branch_attempt": fig3_coeff_time / max(branch_attempts, 1),
                "finite_points": "",
                "usable_attempts": int(sum(bool(row.get("usable", False)) for row in fig3_rows)),
                "formal_success_attempts": int(sum(bool(row.get("optimizer_success", False)) for row in fig3_rows)),
                "optimizer_function_evaluations": "",
                "notes": "high-order coefficient archive reconstruction",
            },
            {
                "method": "pareto_metric_postprocess",
                "wall_time_s": postprocess_pareto_time,
                "grid_points": grid_points,
                "branch_attempts": branch_attempts,
                "seconds_per_grid_point": postprocess_pareto_time / max(grid_points, 1),
                "seconds_per_branch_attempt": postprocess_pareto_time / max(branch_attempts, 1),
                "finite_points": "",
                "usable_attempts": int(
                    sum(np.isfinite(float(row.get("delta_v_km_s", np.nan))) for row in fig3_pareto_rows)
                )
                + int(sum(np.isfinite(float(row.get("delta_v_km_s", np.nan))) for row in bspline_pareto_rows)),
                "formal_success_attempts": "",
                "optimizer_function_evaluations": "",
                "notes": "high-order and B-spline Pareto metric collection",
            },
        ]
    )
    if args.bspline_only:
        bspline_timing_row = next(
            row for row in timing_rows if str(row.get("method", "")).startswith("bspline_cylindrical_")
        )
        pareto_timing_row = timing_rows[-1]
        pareto_timing_row["notes"] = "B-spline Pareto metrics; Gondelach Pareto loaded from cache"
        timing_rows = preserved_timing + [bspline_timing_row, pareto_timing_row]
    elif args.gondelach_only:
        timing_rows[-1]["notes"] = "Gondelach Pareto metrics; B-spline Pareto loaded from cache"
    overall_time = perf_counter() - script_t0
    timing_rows.append(
        {
            "method": "overall_compute",
            "wall_time_s": overall_time,
            "grid_points": grid_points,
            "branch_attempts": branch_attempts,
            "seconds_per_grid_point": overall_time / max(grid_points, 1),
            "seconds_per_branch_attempt": overall_time / max(branch_attempts, 1),
            "finite_points": "",
            "usable_attempts": "",
            "formal_success_attempts": "",
            "optimizer_function_evaluations": "",
            "notes": "through CSV/NPZ archive writing; plots not included",
        }
    )
    write_timing_csv(output_dir / "comparison_timing.csv", timing_rows)
    plot_comparison(
        output_dir,
        str(case["display"]),
        str(case["lower_label"]),
        str(case["higher_label"]),
        f"B-spline cylindrical, {args.bspline_n_ctrl} ctrl",
        dep_grid,
        tof_grid,
        fig2_dv,
        fig2_n,
        fig3_dv,
        fig3_n,
        bspline_dv,
        bspline_n,
    )
    plot_timing(output_dir / "comparison_timing.png", timing_rows)
    plot_pareto(
        output_dir / "comparison_pareto.png",
        pareto_points,
        None,
        {
            "gondelach_fig2": str(case["lower_label"]),
            "gondelach_fig3": str(case["higher_label"]),
            "bspline10": f"B-spline cylindrical, {args.bspline_n_ctrl} ctrl",
        },
    )

    print(f"wrote {output_dir / 'comparison_delta_v.png'}")
    print(f"wrote {output_dir / 'comparison_fig3_vs_bspline_delta_v.png'}")
    print(f"wrote {output_dir / 'comparison_best_N.png'}")
    print(f"wrote {output_dir / 'comparison_delta_v_differences.png'}")
    print(f"wrote {output_dir / 'comparison_fig3_vs_bspline_delta_v_difference.png'}")
    print(f"wrote {output_dir / 'comparison_best.csv'}")
    print(f"wrote {output_dir / 'comparison_summary.csv'}")
    print(f"wrote {output_dir / 'comparison_timing.csv'}")
    print(f"wrote {output_dir / 'comparison_timing.png'}")
    print(f"wrote {output_dir / 'comparison_pareto_points.csv'}")
    print(f"wrote {output_dir / 'comparison_pareto.png'}")
    if args.bspline_only:
        print(f"kept {output_dir / 'fig3_attempts.csv'}")
        print(f"kept {output_dir / 'gondelach_low_order_coefficients.npz'}")
        print(f"kept {output_dir / 'gondelach_high_order_coefficients.npz'}")
    else:
        print(f"wrote {output_dir / 'fig3_attempts.csv'}")
        print(f"wrote {output_dir / 'gondelach_low_order_coefficients.npz'}")
        print(f"wrote {output_dir / 'gondelach_high_order_coefficients.npz'}")
    if args.gondelach_only:
        print(f"kept {output_dir / 'bspline10_attempts.csv'}")
        print(f"kept {output_dir / 'bspline10_control_points.npz'}")
    else:
        print(f"wrote {output_dir / 'bspline10_attempts.csv'}")
        print(f"wrote {output_dir / 'bspline10_control_points.npz'}")
    if args.ephemeris == "spice":
        print(f"wrote {output_dir / 'spice_boundary_states.npz'}")


if __name__ == "__main__":
    main()

"""Reproduce Gondelach & Noomen 2015, Fig. 3, with optimized DoF.

Figure 3 is the higher-order time-driven Mars porkchop.  It uses the same
Earth-to-Mars flight window as Fig. 2 and selects the lowest Delta V over
N=0..5, but the velocity basis has six free coefficients.  Those free
coefficients are optimized here with Nelder-Mead, following the paper's
minimum-Delta-V setup.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reproduce_gondelach_fig2 import (  # noqa: E402
    GONDELACH_FORMULATION_VERSION,
    basis_integral_matrix,
    basis_matrix,
    cartesian_to_cylindrical,
    count_free_coefficients,
    evaluate_time_driven,
    inclusive_grid,
    parse_basis_group,
    planet_state,
    solve_component_coefficients,
    solve_transverse_coefficients,
    split_free_coefficients,
    wrap_0_2pi,
    ephemeris_metadata,
)
from optimizer.oneill_nelder_mead import minimize_oneill  # noqa: E402


FIG3_BASIS = (
    "CPowPow2PSin05PCos05-"
    "CPowPow2PSin05PCos05-"
    "CosN5P3CosN5P3SinN5P4CosN5P4SinN5"
)
FIG3_LEVELS = [5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0]


def finite_or_penalty(value: float, free: np.ndarray) -> float:
    if np.isfinite(value):
        return float(value)
    return 1.0e6 + 1.0e3 * float(np.linalg.norm(free))


def initial_simplex(x0: np.ndarray, scale: float) -> np.ndarray:
    simplex = np.tile(x0, (len(x0) + 1, 1))
    for idx in range(len(x0)):
        simplex[idx + 1, idx] += float(scale)
    return simplex


def optimize_grid_point(
    dep: float,
    tof: float,
    n_rev: int,
    target: str,
    basis: str,
    n_quad: int,
    x0: np.ndarray,
    maxfev: int,
    xatol: float,
    fatol: float,
    simplex_scale: float,
    optimizer: str = "scipy",
    oneill_reqmin: float = 1.0e-8,
    oneill_konvge: int = 10,
    oneill_factorial_epsilon: float = 1.0e-3,
) -> dict:
    branch_t0 = perf_counter()
    x0 = np.asarray(x0, dtype=float)

    def objective(free: np.ndarray) -> float:
        try:
            dv = evaluate_time_driven(
                dep,
                tof,
                n_rev,
                n_quad,
                basis,
                free,
                target=target,
            )
        except Exception:
            dv = float("nan")
        return finite_or_penalty(dv, np.asarray(free, dtype=float))

    start_dv = objective(x0)
    if optimizer == "scipy":
        from scipy.optimize import minimize

        result = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={
                "maxfev": int(maxfev),
                "xatol": float(xatol),
                "fatol": float(fatol),
                "initial_simplex": initial_simplex(x0, simplex_scale),
                "disp": False,
            },
        )
        result_restarts = 0
    elif optimizer == "oneill":
        result = minimize_oneill(
            objective,
            x0,
            step=float(simplex_scale),
            reqmin=float(oneill_reqmin),
            konvge=int(oneill_konvge),
            maxfev=int(maxfev),
            factorial_epsilon=float(oneill_factorial_epsilon),
        )
        result_restarts = int(result.num_restarts)
    else:
        raise ValueError(f"Unknown Figure 3 optimizer: {optimizer!r}")

    candidates = [(start_dv, x0), (float(result.fun), np.asarray(result.x, dtype=float))]
    best_value, best_free = min(candidates, key=lambda item: item[0])
    usable = bool(np.isfinite(best_value) and best_value < 1.0e5)
    return {
        "departure_mjd2000": float(dep),
        "tof_days": float(tof),
        "N": int(n_rev),
        "delta_v_km_s": float(best_value) if usable else float("nan"),
        "optimizer_success": bool(result.success),
        "optimizer": str(optimizer),
        "usable": usable,
        "nfev": int(result.nfev),
        "nit": int(result.nit),
        "num_restarts": result_restarts,
        "message": str(result.message),
        "free_coefficients": best_free,
        "start_delta_v_km_s": float(start_dv) if start_dv < 1.0e5 else float("nan"),
        "wall_time_s": perf_counter() - branch_t0,
    }


def compute_grid(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray]:
    groups = [parse_basis_group(part) for part in args.basis.split("-")]
    if len(groups) != 3:
        raise ValueError("Basis must have exactly three groups: radial-transverse-axial")
    n_free = count_free_coefficients(*groups)
    if n_free <= 0:
        raise ValueError("Figure 3 reproduction expects a higher-order basis with free coefficients")

    dep_grid = inclusive_grid(args.dep_min, args.dep_max, args.dep_step)
    tof_grid = inclusive_grid(args.tof_min, args.tof_max, args.tof_step)
    dv_grid = np.full((len(tof_grid), len(dep_grid)), np.nan)
    best_n_grid = np.full((len(tof_grid), len(dep_grid)), -1, dtype=int)
    rows: list[dict] = []

    previous_row: dict[tuple[int, int], np.ndarray] = {}
    total = len(dep_grid) * len(tof_grid) * (args.n_max - args.n_min + 1)
    count = 0

    for i_tof, tof in enumerate(tof_grid):
        current_row: dict[tuple[int, int], np.ndarray] = {}
        previous_dep: dict[int, np.ndarray] = {}
        for i_dep, dep in enumerate(dep_grid):
            branch_rows = []
            for n_rev in range(args.n_min, args.n_max + 1):
                x0 = previous_dep.get(n_rev)
                if x0 is None:
                    x0 = previous_row.get((i_dep, n_rev))
                if x0 is None:
                    x0 = np.zeros(n_free, dtype=float)
                row = optimize_grid_point(
                    float(dep),
                    float(tof),
                    int(n_rev),
                    str(getattr(args, "target", "mars")),
                    args.basis,
                    int(args.n_quad),
                    x0,
                    int(args.maxfev),
                    float(args.xatol),
                    float(args.fatol),
                    float(args.simplex_scale),
                    str(getattr(args, "optimizer", "scipy")),
                    float(getattr(args, "oneill_reqmin", 1.0e-8)),
                    int(getattr(args, "oneill_konvge", 10)),
                    float(getattr(args, "oneill_factorial_epsilon", 1.0e-3)),
                )
                rows.append(row)
                branch_rows.append(row)
                if row["usable"]:
                    previous_dep[n_rev] = np.asarray(row["free_coefficients"], dtype=float)
                    current_row[(i_dep, n_rev)] = np.asarray(row["free_coefficients"], dtype=float)
                count += 1
                if args.progress and (count == total or count % max(1, args.progress_every) == 0):
                    print(f"completed {count}/{total}")

            usable = [row for row in branch_rows if row["usable"] and np.isfinite(row["delta_v_km_s"])]
            if usable:
                best = min(usable, key=lambda row: float(row["delta_v_km_s"]))
                dv_grid[i_tof, i_dep] = float(best["delta_v_km_s"])
                best_n_grid[i_tof, i_dep] = int(best["N"])
        previous_row = current_row

    return dep_grid, tof_grid, rows, dv_grid, best_n_grid


def write_attempts_csv(path: Path, rows: list[dict]) -> None:
    max_free = max((len(row["free_coefficients"]) for row in rows), default=0)
    fieldnames = [
        "departure_mjd2000",
        "tof_days",
        "N",
        "delta_v_km_s",
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
            for idx, value in enumerate(row["free_coefficients"]):
                out[f"free_{idx}"] = float(value)
            writer.writerow(out)


def reconstruct_coefficients_from_free(
    dep_mjd2000: float,
    tof_days: float,
    n_rev: int,
    n_quad: int,
    basis: str,
    free_coefficients: np.ndarray,
    target: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(n_quad) % 2 == 0:
        n_quad += 1
    tau = np.linspace(0.0, 1.0, int(n_quad))
    radial_terms, transverse_terms, axial_terms = [parse_basis_group(part) for part in basis.split("-")]
    radial_free, transverse_free, axial_free = split_free_coefficients(
        free_coefficients,
        radial_terms,
        transverse_terms,
        axial_terms,
    )

    pos0, vel0 = planet_state("earth", dep_mjd2000)
    posf, velf = planet_state(target, dep_mjd2000 + tof_days)
    q0, qdot0 = cartesian_to_cylindrical(pos0, vel0)
    qf, qdotf = cartesian_to_cylindrical(posf, velf)

    theta_target = wrap_0_2pi(float(qf[1] - q0[1])) + 2.0 * math.pi * int(n_rev)
    coeff_r = solve_component_coefficients(
        radial_terms,
        tau,
        n_rev,
        tof_days,
        qdot0[0],
        qdotf[0],
        qf[0] - q0[0],
        radial_free,
    )
    radial_integrals = basis_integral_matrix(radial_terms, tau, n_rev)
    rho = q0[0] + tof_days * (radial_integrals @ coeff_r)
    if np.any(~np.isfinite(rho)) or np.min(rho) <= 1e-6:
        raise ValueError("Invalid radial profile while reconstructing high-order coefficients")

    coeff_theta = solve_transverse_coefficients(
        transverse_terms,
        tau,
        n_rev,
        tof_days,
        qdot0[1],
        qdotf[1],
        theta_target,
        rho,
        transverse_free,
    )
    coeff_z = solve_component_coefficients(
        axial_terms,
        tau,
        n_rev,
        tof_days,
        qdot0[2],
        qdotf[2],
        qf[2] - q0[2],
        axial_free,
    )
    return coeff_r, coeff_theta, coeff_z


def write_coefficients_npz(
    path: Path,
    rows: list[dict],
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
    args: argparse.Namespace,
) -> None:
    ordered = sorted(rows, key=lambda item: (item["tof_days"], item["departure_mjd2000"], item["N"]))
    radial_terms, transverse_terms, axial_terms = [parse_basis_group(part) for part in str(args.basis).split("-")]
    free_count = count_free_coefficients(radial_terms, transverse_terms, axial_terms)
    coeff_shapes = (len(radial_terms), len(transverse_terms), len(axial_terms))

    free_coefficients = np.full((len(ordered), free_count), np.nan, dtype=float)
    radial_coefficients = np.full((len(ordered), coeff_shapes[0]), np.nan, dtype=float)
    transverse_coefficients = np.full((len(ordered), coeff_shapes[1]), np.nan, dtype=float)
    axial_coefficients = np.full((len(ordered), coeff_shapes[2]), np.nan, dtype=float)
    coefficient_reconstruction_success = np.zeros(len(ordered), dtype=bool)
    coefficient_reconstruction_message = [""] * len(ordered)

    for idx, row in enumerate(ordered):
        free = np.asarray(row.get("free_coefficients", []), dtype=float).reshape(-1)
        if free.shape[0] == free_count:
            free_coefficients[idx] = free
        try:
            coeff_r, coeff_theta, coeff_z = reconstruct_coefficients_from_free(
                float(row["departure_mjd2000"]),
                float(row["tof_days"]),
                int(row["N"]),
                int(args.n_quad),
                str(args.basis),
                free,
                str(getattr(args, "target", "mars")),
            )
            radial_coefficients[idx] = coeff_r
            transverse_coefficients[idx] = coeff_theta
            axial_coefficients[idx] = coeff_z
            coefficient_reconstruction_success[idx] = True
        except Exception as exc:
            coefficient_reconstruction_message[idx] = str(exc).splitlines()[-1]

    np.savez(
        path,
        departure_mjd2000=np.asarray([row["departure_mjd2000"] for row in ordered], dtype=float),
        tof_days=np.asarray([row["tof_days"] for row in ordered], dtype=float),
        N=np.asarray([row["N"] for row in ordered], dtype=int),
        delta_v_km_s=np.asarray([row.get("delta_v_km_s", np.nan) for row in ordered], dtype=float),
        delta_v_optimizer_km_s=np.asarray(
            [row.get("delta_v_optimizer_km_s", row.get("delta_v_km_s", np.nan)) for row in ordered],
            dtype=float,
        ),
        delta_v_reference_km_s=np.asarray(
            [row.get("delta_v_reference_km_s", np.nan) for row in ordered], dtype=float
        ),
        delta_v_reference_error_km_s=np.asarray(
            [row.get("delta_v_reference_error_km_s", np.nan) for row in ordered], dtype=float
        ),
        u_max_reference_m_s2=np.asarray(
            [row.get("u_max_reference_m_s2", np.nan) for row in ordered], dtype=float
        ),
        u_max_reference_error_m_s2=np.asarray(
            [row.get("u_max_reference_error_m_s2", np.nan) for row in ordered], dtype=float
        ),
        reference_quadrature_order=np.asarray(
            [row.get("reference_quadrature_order", -1) for row in ordered], dtype=int
        ),
        reference_evaluations=np.asarray(
            [row.get("reference_evaluations", 0) for row in ordered], dtype=int
        ),
        reference_converged=np.asarray(
            [bool(row.get("reference_converged", False)) for row in ordered], dtype=bool
        ),
        reference_metric_version=np.asarray(
            [str(row.get("reference_metric_version", "")) for row in ordered]
        ),
        start_delta_v_km_s=np.asarray([row.get("start_delta_v_km_s", np.nan) for row in ordered], dtype=float),
        optimizer_success=np.asarray([bool(row.get("optimizer_success", False)) for row in ordered], dtype=bool),
        usable=np.asarray([bool(row.get("usable", False)) for row in ordered], dtype=bool),
        nfev=np.asarray([row.get("nfev", 0) for row in ordered], dtype=int),
        nit=np.asarray([row.get("nit", 0) for row in ordered], dtype=int),
        num_restarts=np.asarray([row.get("num_restarts", 0) for row in ordered], dtype=int),
        optimizer=np.asarray(str(getattr(args, "optimizer", "scipy"))),
        wall_time_s=np.asarray([row.get("wall_time_s", np.nan) for row in ordered], dtype=float),
        message=np.asarray([str(row.get("message", "")) for row in ordered]),
        free_coefficients=free_coefficients,
        radial_coefficients=radial_coefficients,
        transverse_coefficients=transverse_coefficients,
        axial_coefficients=axial_coefficients,
        coefficient_reconstruction_success=coefficient_reconstruction_success,
        coefficient_reconstruction_message=np.asarray(coefficient_reconstruction_message),
        grid_departure_mjd2000=np.asarray(dep_grid, dtype=float),
        grid_tof_days=np.asarray(tof_grid, dtype=float),
        basis=np.asarray(str(args.basis)),
        radial_basis=np.asarray(str(args.basis).split("-")[0]),
        transverse_basis=np.asarray(str(args.basis).split("-")[1]),
        axial_basis=np.asarray(str(args.basis).split("-")[2]),
        target=np.asarray(str(getattr(args, "target", "mars"))),
        n_quad=np.asarray(int(args.n_quad), dtype=int),
        n_min=np.asarray(int(args.n_min), dtype=int),
        n_max=np.asarray(int(args.n_max), dtype=int),
        maxfev=np.asarray(int(args.maxfev), dtype=int),
        xatol=np.asarray(float(args.xatol), dtype=float),
        fatol=np.asarray(float(args.fatol), dtype=float),
        simplex_scale=np.asarray(float(args.simplex_scale), dtype=float),
        oneill_reqmin=np.asarray(float(getattr(args, "oneill_reqmin", 1.0e-8)), dtype=float),
        oneill_konvge=np.asarray(int(getattr(args, "oneill_konvge", 10)), dtype=int),
        oneill_factorial_epsilon=np.asarray(
            float(getattr(args, "oneill_factorial_epsilon", 1.0e-3)), dtype=float
        ),
        free_coefficient_count=np.asarray(free_count, dtype=int),
        radial_coefficient_count=np.asarray(coeff_shapes[0], dtype=int),
        transverse_coefficient_count=np.asarray(coeff_shapes[1], dtype=int),
        axial_coefficient_count=np.asarray(coeff_shapes[2], dtype=int),
        coefficient_components=np.asarray(["v_r", "v_theta", "v_z"]),
        gondelach_formulation_version=np.asarray(GONDELACH_FORMULATION_VERSION),
        **{key: np.asarray(value) for key, value in ephemeris_metadata().items()},
    )


def write_best_csv(path: Path, dep_grid: np.ndarray, tof_grid: np.ndarray, dv_grid: np.ndarray, best_n_grid: np.ndarray) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["departure_mjd2000", "tof_days", "delta_v_km_s", "best_N"])
        for i_tof, tof in enumerate(tof_grid):
            for i_dep, dep in enumerate(dep_grid):
                writer.writerow([dep, tof, dv_grid[i_tof, i_dep], best_n_grid[i_tof, i_dep]])


def write_timing_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "method",
        "wall_time_s",
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


def plot_fig3(path: Path, dep_grid: np.ndarray, tof_grid: np.ndarray, dv_grid: np.ndarray, best_n_grid: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dep_pad = 0.5 if len(dep_grid) == 1 else 0.0
    tof_pad = 0.5 if len(tof_grid) == 1 else 0.0
    extent = [
        float(dep_grid[0]) - dep_pad,
        float(dep_grid[-1]) + dep_pad,
        float(tof_grid[0]) - tof_pad,
        float(tof_grid[-1]) + tof_pad,
    ]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if dv_grid.shape[0] >= 2 and dv_grid.shape[1] >= 2:
        cf = ax.contourf(dep_grid, tof_grid, dv_grid, levels=FIG3_LEVELS, cmap="viridis_r", extend="both")
        cs = ax.contour(dep_grid, tof_grid, dv_grid, levels=FIG3_LEVELS, colors="black", linewidths=0.85)
        ax.clabel(cs, inline=True, fmt="%g", fontsize=9)
    else:
        cf = ax.imshow(
            dv_grid,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis_r",
            vmin=FIG3_LEVELS[0],
            vmax=FIG3_LEVELS[-1],
        )
    ax.set_xlabel("Departure date [MJD2000]")
    ax.set_ylabel("Time of flight [days]")
    ax.set_title("Gondelach Fig. 3 reproduction: Mars, higher-order time-driven, N=0-5")
    cbar = fig.colorbar(cf, ax=ax, ticks=FIG3_LEVELS)
    cbar.set_label("Delta V [km/s]")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    n_path = path.with_name(path.stem + "_best_N.png")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    image = ax.imshow(
        best_n_grid,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="tab10",
        vmin=-0.5,
        vmax=5.5,
    )
    ax.set_xlabel("Departure date [MJD2000]")
    ax.set_ylabel("Time of flight [days]")
    ax.set_title("Best optimized branch selected at each grid point, N=0-5")
    cbar = fig.colorbar(image, ax=ax, ticks=range(0, 6))
    cbar.set_label("best N")
    fig.tight_layout()
    fig.savefig(n_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dep-min", type=float, default=7304.0)
    parser.add_argument("--dep-max", type=float, default=10225.0)
    parser.add_argument("--tof-min", type=float, default=500.0)
    parser.add_argument("--tof-max", type=float, default=2000.0)
    parser.add_argument("--dep-step", type=float, default=20.0)
    parser.add_argument("--tof-step", type=float, default=20.0)
    parser.add_argument("--n-quad", type=int, default=51)
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=5)
    parser.add_argument("--basis", default=FIG3_BASIS)
    parser.add_argument("--target", choices=["mars", "1989ml", "tempel1", "mercury"], default="mars")
    parser.add_argument("--maxfev", type=int, default=5000)
    parser.add_argument("--xatol", type=float, default=1e-6)
    parser.add_argument("--fatol", type=float, default=1e-6)
    parser.add_argument("--simplex-scale", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=["scipy", "oneill"], default="scipy")
    parser.add_argument("--oneill-reqmin", type=float, default=1e-8)
    parser.add_argument("--oneill-konvge", type=int, default=10)
    parser.add_argument("--oneill-factorial-epsilon", type=float, default=1e-3)
    parser.add_argument("--output-dir", default="output/gondelach_fig3")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    t0 = perf_counter()
    dep_grid, tof_grid, rows, dv_grid, best_n_grid = compute_grid(args)
    grid_time = perf_counter() - t0

    npz_path = output_dir / "gondelach_fig3_reproduction.npz"
    coeff_path = output_dir / "gondelach_fig3_reproduction_coefficients.npz"
    csv_path = output_dir / "gondelach_fig3_reproduction.csv"
    attempts_path = output_dir / "gondelach_fig3_reproduction_attempts.csv"
    timing_path = output_dir / "gondelach_fig3_reproduction_timing.csv"
    png_path = output_dir / "gondelach_fig3_reproduction.png"
    np.savez(
        npz_path,
        departure_mjd2000=dep_grid,
        tof_days=tof_grid,
        delta_v_km_s=dv_grid,
        best_N=best_n_grid,
        basis=args.basis,
        optimizer=args.optimizer,
        maxfev=args.maxfev,
        simplex_scale=args.simplex_scale,
        oneill_reqmin=args.oneill_reqmin,
        oneill_konvge=args.oneill_konvge,
        oneill_factorial_epsilon=args.oneill_factorial_epsilon,
    )
    t0 = perf_counter()
    write_coefficients_npz(coeff_path, rows, dep_grid, tof_grid, args)
    coefficient_time = perf_counter() - t0
    write_best_csv(csv_path, dep_grid, tof_grid, dv_grid, best_n_grid)
    write_attempts_csv(attempts_path, rows)
    grid_points = int(len(dep_grid) * len(tof_grid))
    branch_attempts = int(grid_points * (args.n_max - args.n_min + 1))
    write_timing_csv(
        timing_path,
        [
            {
                "method": "gondelach_fig3_grid",
                "wall_time_s": grid_time,
                "grid_points": grid_points,
                "branch_attempts": branch_attempts,
                "seconds_per_grid_point": grid_time / max(grid_points, 1),
                "seconds_per_branch_attempt": grid_time / max(branch_attempts, 1),
                "finite_points": int(np.isfinite(dv_grid).sum()),
                "usable_attempts": int(sum(bool(row.get("usable", False)) for row in rows)),
                "formal_success_attempts": int(sum(bool(row.get("optimizer_success", False)) for row in rows)),
                "optimizer_function_evaluations": int(sum(int(row.get("nfev", 0)) for row in rows)),
                "notes": f"{args.optimizer} Nelder-Mead optimized higher-order evaluator",
            },
            {
                "method": "gondelach_fig3_coefficients",
                "wall_time_s": coefficient_time,
                "grid_points": grid_points,
                "branch_attempts": branch_attempts,
                "seconds_per_grid_point": coefficient_time / max(grid_points, 1),
                "seconds_per_branch_attempt": coefficient_time / max(branch_attempts, 1),
                "finite_points": "",
                "usable_attempts": int(sum(bool(row.get("usable", False)) for row in rows)),
                "formal_success_attempts": int(sum(bool(row.get("optimizer_success", False)) for row in rows)),
                "optimizer_function_evaluations": "",
                "notes": "high-order coefficient archive reconstruction",
            },
            {
                "method": "overall_compute",
                "wall_time_s": grid_time + coefficient_time,
                "grid_points": grid_points,
                "branch_attempts": branch_attempts,
                "seconds_per_grid_point": (grid_time + coefficient_time) / max(grid_points, 1),
                "seconds_per_branch_attempt": (grid_time + coefficient_time) / max(branch_attempts, 1),
                "finite_points": int(np.isfinite(dv_grid).sum()),
                "usable_attempts": int(sum(bool(row.get("usable", False)) for row in rows)),
                "formal_success_attempts": int(sum(bool(row.get("optimizer_success", False)) for row in rows)),
                "optimizer_function_evaluations": int(sum(int(row.get("nfev", 0)) for row in rows)),
                "notes": "optimizer plus coefficient archive reconstruction",
            },
        ],
    )
    plot_fig3(png_path, dep_grid, tof_grid, dv_grid, best_n_grid)

    finite = dv_grid[np.isfinite(dv_grid)]
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {attempts_path}")
    print(f"wrote {coeff_path}")
    print(f"wrote {timing_path}")
    print(f"wrote {npz_path}")
    if finite.size:
        best_idx = np.unravel_index(int(np.nanargmin(dv_grid)), dv_grid.shape)
        print(
            "best grid point: "
            f"dep={dep_grid[best_idx[1]]:.1f} MJD2000, "
            f"TOF={tof_grid[best_idx[0]]:.1f} days, "
            f"DeltaV={dv_grid[best_idx]:.3f} km/s, "
            f"N={best_n_grid[best_idx]}"
        )
    else:
        print("No finite optimized solutions found.")


if __name__ == "__main__":
    main()

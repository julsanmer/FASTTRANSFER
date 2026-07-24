"""B-spline cylindrical comparison for Gondelach & Noomen 2015, Fig. 2.

This script uses the same Earth-to-Mars MJD2000/TOF ranges and N=0..5
transfer-angle branches as ``reproduce_gondelach_fig2.py``, but solves each
grid point with the local cylindrical-coordinate B-spline optimizer.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reproduce_gondelach_fig2 import (  # noqa: E402
    AU_KM,
    DAY_S,
    planet_state,
    configure_ephemeris,
    wrap_0_2pi,
)
from optimizer.canonical_units import (  # noqa: E402
    MU_CANONICAL,
    UT_YR,
    UV_AU_PER_YR,
    velocity_to_canonical,
)
from optimizer.helpers_Bspline import build_bspline_matrices  # noqa: E402
from optimizer.helpers_Bspline import derivative_matrix  # noqa: E402
from optimizer.optimization_Bspline_freetf import (  # noqa: E402
    cartesian_to_cylindrical_state,
    solve_free_tf_cylindrical_bspline,
)
from utils.utils import kepler_coast_np, rv2mee  # noqa: E402


YEAR_DAYS = 365.25
AU_PER_YR_TO_KM_PER_S = AU_KM / (YEAR_DAYS * DAY_S)
CONTROL_POINT_COMPONENTS = np.array(["rho", "theta", "z"])
CONTROL_POINT_COORDINATES = "cylindrical"


def inclusive_grid(lower: float, upper: float, step: float) -> np.ndarray:
    step = float(step)
    if step <= 0.0:
        raise ValueError("Grid spacing must be positive")
    values = np.arange(float(lower), float(upper) + 0.5 * step, step, dtype=float)
    values = values[values <= float(upper) + 1.0e-9]
    if values.size == 0:
        raise ValueError("Grid spacing produced an empty grid")
    return values


def rv_au_day_to_mee_canonical(pos_au: np.ndarray, vel_au_day: np.ndarray) -> np.ndarray:
    vel_au_yr = np.asarray(vel_au_day, dtype=float) * YEAR_DAYS
    return rv2mee(np.asarray(pos_au, dtype=float), velocity_to_canonical(vel_au_yr), MU_CANONICAL)


def tof_days_to_canonical(tof_days: float) -> float:
    return float(tof_days) / (YEAR_DAYS * UT_YR)


def canonical_dv_to_km_s(dv_canonical: float) -> float:
    return float(dv_canonical) * UV_AU_PER_YR * AU_PER_YR_TO_KM_PER_S


def endpoint_norm(endpoint: dict) -> float:
    vals = [float(endpoint.get(key, np.inf)) for key in ["r0", "v0", "rf", "vf"]]
    return float(max(vals))


def is_usable(row: dict, options: dict) -> bool:
    if not np.isfinite(float(row.get("delta_v_km_s", np.nan))):
        return False
    if bool(row.get("success", False)):
        return True
    if not bool(options.get("accept_debug_feasible", False)):
        return False
    endpoint_error = float(row.get("endpoint_error", np.inf))
    winding_error = abs(float(row.get("winding_error_rev", np.inf)))
    return bool(
        np.isfinite(endpoint_error)
        and endpoint_error <= float(options["endpoint_tol"])
        and np.isfinite(winding_error)
        and winding_error <= float(options["winding_tol_rev"])
    )


def endpoint_projection(
    control_points: np.ndarray,
    tf: float,
    pos0: np.ndarray,
    vel0: np.ndarray,
    posf: np.ndarray,
    velf: np.ndarray,
    winding_target_rev: float,
    options: dict,
) -> tuple[np.ndarray, float, float]:
    """Minimally adjust a cylindrical control polygon to the current endpoints."""
    mats = build_bspline_matrices(
        int(options["n_ctrl"]),
        degree=int(options["degree"]),
        n_fine=int(options["n_fine"]),
    )
    control_points = np.asarray(control_points, dtype=float)
    expected_shape = (mats.b0_fine.shape[1], 3)
    if control_points.shape != expected_shape:
        raise ValueError(f"initial_control_points must have shape {expected_shape}, got {control_points.shape}")

    q0_state = cartesian_to_cylindrical_state(pos0, vel0)
    thetaf_wrapped = float(math.atan2(float(posf[1]), float(posf[0])))
    theta_ref = float(q0_state[1]) + 2.0 * math.pi * float(winding_target_rev)
    theta_offset = 2.0 * math.pi * float(np.rint((theta_ref - thetaf_wrapped) / (2.0 * math.pi)))
    thetaf_unwrapped = thetaf_wrapped + theta_offset
    qf_state = cartesian_to_cylindrical_state(posf, velf, theta_reference=thetaf_unwrapped)

    a_eq = np.vstack(
        [
            mats.b0_start,
            mats.b1_start / float(tf),
            mats.b0_end,
            mats.b1_end / float(tf),
        ]
    )
    b_eq = np.vstack(
        [
            q0_state[:3].reshape(1, 3),
            q0_state[3:].reshape(1, 3),
            qf_state[:3].reshape(1, 3),
            qf_state[3:].reshape(1, 3),
        ]
    )
    before = float(np.linalg.norm(a_eq @ control_points - b_eq, ord=np.inf))
    multiplier = np.linalg.lstsq(a_eq @ a_eq.T, a_eq @ control_points - b_eq, rcond=None)[0]
    projected = control_points - a_eq.T @ multiplier
    after = float(np.linalg.norm(a_eq @ projected - b_eq, ord=np.inf))
    return np.asarray(projected, dtype=float), before, after


def seed_control_points_from_row(row: dict) -> np.ndarray | None:
    if not bool(row.get("usable", False)):
        return None
    control_points = row.get("control_points")
    if control_points is None:
        return None
    arr = np.asarray(control_points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.all(np.isfinite(arr)):
        return None
    return arr.copy()


def solve_one(dep_mjd2000: float, tof_days: float, n_rev: int, options: dict) -> dict:
    branch_t0 = perf_counter()
    configure_ephemeris(
        str(options.get("ephemeris", "kepler")),
        options.get("spice_meta_kernel"),
        options.get("spice_target_name"),
    )
    pos0, vel0 = planet_state("earth", dep_mjd2000)
    posf, velf = planet_state(str(options.get("target", "mars")), dep_mjd2000 + tof_days)

    mee0 = rv_au_day_to_mee_canonical(pos0, vel0)
    meef = rv_au_day_to_mee_canonical(posf, velf)
    tf = tof_days_to_canonical(tof_days)

    # The B-spline solver coasts the target epoch forward by tf internally.
    # Back-propagating the arrival MEE makes that endpoint match the paper grid.
    mee_target_epoch = kepler_coast_np(meef, -tf, MU_CANONICAL, n_iter=int(options["kepler_substeps"]))

    theta0 = math.atan2(float(pos0[1]), float(pos0[0]))
    thetaf = math.atan2(float(posf[1]), float(posf[0]))
    winding_target_rev = (wrap_0_2pi(thetaf - theta0) / (2.0 * math.pi)) + int(n_rev)
    seed_source = str(options.get("initial_control_points_source", "analytic"))
    seed_projection_success = False
    seed_projection_message = ""
    seed_projection_before = float("nan")
    seed_projection_after = float("nan")

    try:
        initial_control_points = options.get("initial_control_points")
        if initial_control_points is not None:
            initial_control_points = np.asarray(initial_control_points, dtype=float)
            if bool(options.get("project_initial_control_points", False)):
                try:
                    initial_control_points, seed_projection_before, seed_projection_after = endpoint_projection(
                        initial_control_points,
                        tf,
                        pos0,
                        vel0,
                        posf,
                        velf,
                        winding_target_rev,
                        options,
                    )
                    seed_projection_success = True
                except Exception as exc:
                    seed_projection_message = str(exc).splitlines()[-1]

        result = solve_free_tf_cylindrical_bspline(
            tf,
            mee0=mee0,
            mee_target_epoch=mee_target_epoch,
            mee_target_final=meef,
            mu=MU_CANONICAL,
            n_ctrl=int(options["n_ctrl"]),
            degree=int(options["degree"]),
            n_fine=int(options["n_fine"]),
            seed_profile=str(options["seed_profile"]),
            quadrature_order=int(options["quadrature_order"]),
            R_bound=float(options["r_bound"]),
            dv_eps=float(options["dv_eps"]),
            smoothness_weight=float(options["smoothness_weight"]),
            endpoint_control_weight=float(options["endpoint_control_weight"]),
            initial_control_points=initial_control_points,
            fixed_tf=True,
            winding_target_rev=winding_target_rev,
            max_iter=int(options["max_iter"]),
            print_level=int(options["print_level"]),
            linear_solver=str(options.get("linear_solver", "mumps")),
            coinhsl_library=str(options.get("coinhsl_library", "") or "") or None,
        )
        dv_optimizer_km_s = canonical_dv_to_km_s(float(result["delta_v_optimizer_canonical"]))
        dv_km_s = float(result["delta_v_reference_km_s"])
        energy = float(result["energy_pure_fine"])
        return {
            "departure_mjd2000": float(dep_mjd2000),
            "tof_days": float(tof_days),
            "N": int(n_rev),
            "delta_v_km_s": dv_km_s,
            "delta_v_optimizer_km_s": dv_optimizer_km_s,
            "delta_v_reference_km_s": dv_km_s,
            "delta_v_reference_error_km_s": float(result["delta_v_reference_error_km_s"]),
            "u_max_reference_m_s2": float(result["u_max_reference_m_s2"]),
            "u_max_reference_error_m_s2": float(result["u_max_reference_error_m_s2"]),
            "reference_quadrature_order": int(result["reference_quadrature_order"]),
            "reference_evaluations": int(result["reference_evaluations"]),
            "reference_converged": bool(result["reference_converged"]),
            "reference_metric_version": str(result["reference_metric_version"]),
            "energy_canonical": energy,
            "linear_solver": str(options.get("linear_solver", "mumps")),
            "coinhsl_library": str(options.get("coinhsl_library", "") or ""),
            "ipopt_iterations": int(result.get("ipopt_iterations", -1)),
            "success": bool(result["success"]),
            "usable": is_usable(
                {
                    "delta_v_km_s": dv_km_s,
                    "success": bool(result["success"]),
                    "endpoint_error": endpoint_norm(result.get("endpoint_errors", {})),
                    "winding_error_rev": float(result.get("winding_error_rev", np.nan)),
                },
                options,
            ),
            "message": str(result["message"]),
            "endpoint_error": endpoint_norm(result.get("endpoint_errors", {})),
            "winding_target_rev": float(result.get("winding_target_rev", np.nan)),
            "winding_sum_rev": float(result.get("winding_sum_rev", np.nan)),
            "winding_error_rev": float(result.get("winding_error_rev", np.nan)),
            "winding_fine_rev": float(result.get("winding_fine_rev", np.nan)),
            "max_u_fine": float(result.get("max_u_fine", np.nan)),
            "wall_time_s": perf_counter() - branch_t0,
            "t_transfer_canonical": float(result.get("t_transfer", tf)),
            "t_transfer_initial_canonical": float(result.get("t_transfer_initial", np.nan)),
            "boundary_control_points_fixed": bool(
                result.get("boundary_control_points_fixed", result.get("boundary_control_points_eliminated", False))
            ),
            "boundary_control_points_eliminated": bool(result.get("boundary_control_points_eliminated", False)),
            "n_free_control_points": int(result.get("n_free_control_points", -1)),
            "seed_source": seed_source,
            "seed_projection_success": seed_projection_success,
            "seed_projection_before": seed_projection_before,
            "seed_projection_after": seed_projection_after,
            "seed_projection_message": seed_projection_message,
            "control_point_coordinates": str(result.get("control_point_coordinates", CONTROL_POINT_COORDINATES)),
            "control_points": np.asarray(result["control_points"], dtype=float),
            "control_points_initial": np.asarray(result["control_points_initial"], dtype=float),
            "mee0": np.asarray(result.get("mee0", mee0), dtype=float),
            "mee_target_epoch": np.asarray(result.get("mee_target_epoch", mee_target_epoch), dtype=float),
            "mu": float(result.get("mu", MU_CANONICAL)),
        }
    except Exception as exc:
        return {
            "departure_mjd2000": float(dep_mjd2000),
            "tof_days": float(tof_days),
            "N": int(n_rev),
            "delta_v_km_s": float("nan"),
            "delta_v_optimizer_km_s": float("nan"),
            "delta_v_reference_km_s": float("nan"),
            "delta_v_reference_error_km_s": float("nan"),
            "u_max_reference_m_s2": float("nan"),
            "u_max_reference_error_m_s2": float("nan"),
            "reference_quadrature_order": -1,
            "reference_evaluations": 0,
            "reference_converged": False,
            "reference_metric_version": "",
            "energy_canonical": float("nan"),
            "success": False,
            "usable": False,
            "message": str(exc).splitlines()[-1],
            "endpoint_error": float("nan"),
            "winding_target_rev": winding_target_rev,
            "winding_sum_rev": float("nan"),
            "winding_error_rev": float("nan"),
            "winding_fine_rev": float("nan"),
            "max_u_fine": float("nan"),
            "wall_time_s": perf_counter() - branch_t0,
            "t_transfer_canonical": tf,
            "t_transfer_initial_canonical": float("nan"),
            "boundary_control_points_fixed": True,
            "boundary_control_points_eliminated": True,
            "n_free_control_points": -1,
            "linear_solver": str(options.get("linear_solver", "mumps")),
            "coinhsl_library": str(options.get("coinhsl_library", "") or ""),
            "ipopt_iterations": -1,
            "seed_source": seed_source,
            "seed_projection_success": seed_projection_success,
            "seed_projection_before": seed_projection_before,
            "seed_projection_after": seed_projection_after,
            "seed_projection_message": seed_projection_message,
            "control_point_coordinates": CONTROL_POINT_COORDINATES,
            "mu": MU_CANONICAL,
        }


def task_payloads(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, int, dict]]]:
    dep_grid = inclusive_grid(args.dep_min, args.dep_max, args.dep_step)
    tof_grid = inclusive_grid(args.tof_min, args.tof_max, args.tof_step)
    options = {
        "n_ctrl": args.n_ctrl,
        "degree": args.degree,
        "n_fine": args.n_fine,
        "seed_profile": args.seed_profile,
        "quadrature_order": args.quadrature_order,
        "r_bound": args.r_bound,
        "dv_eps": args.dv_eps,
        "smoothness_weight": args.smoothness_weight,
        "endpoint_control_weight": args.endpoint_control_weight,
        "max_iter": args.max_iter,
        "linear_solver": str(getattr(args, "linear_solver", "mumps")),
        "coinhsl_library": str(getattr(args, "coinhsl_library", "") or ""),
        "print_level": args.print_level,
        "kepler_substeps": args.kepler_substeps,
        "target": args.target,
        "accept_debug_feasible": args.accept_debug_feasible,
        "endpoint_tol": args.endpoint_tol,
        "winding_tol_rev": args.winding_tol_rev,
        "project_initial_control_points": bool(
            getattr(
                args,
                "project_initial_control_points",
                getattr(args, "grid_continuation_project", False),
            )
        ),
        "ephemeris": str(getattr(args, "ephemeris", "kepler")),
        "spice_meta_kernel": str(getattr(args, "spice_meta_kernel", "") or ""),
        "spice_target_name": str(getattr(args, "spice_target_name", "") or ""),
    }
    initial_control_points_by_key = getattr(args, "initial_control_points_by_key", None)
    tasks = []
    for tof in tof_grid:
        for dep in dep_grid:
            for n_rev in range(args.n_min, args.n_max + 1):
                task_options = dict(options)
                if initial_control_points_by_key is not None:
                    key = (float(dep), float(tof), int(n_rev))
                    initial_control_points = initial_control_points_by_key.get(key)
                    if initial_control_points is not None:
                        task_options["initial_control_points"] = initial_control_points
                        task_options["initial_control_points_source"] = "archive"
                tasks.append((float(dep), float(tof), int(n_rev), task_options))
    return dep_grid, tof_grid, tasks


def run_tasks(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray, np.ndarray]:
    dep_grid, tof_grid, tasks = task_payloads(args)
    rows: list[dict] = []
    total = len(tasks)

    if bool(getattr(args, "grid_continuation", False)):
        if args.progress and int(getattr(args, "workers", 1)) > 1:
            print("grid continuation requested; running B-spline tasks serially")
        n_branch = int(args.n_max - args.n_min + 1)
        previous_row: dict[tuple[int, int], np.ndarray] = {}
        idx = 0
        for i_tof, _tof in enumerate(tof_grid):
            current_row: dict[tuple[int, int], np.ndarray] = {}
            previous_dep: dict[int, np.ndarray] = {}
            for i_dep, _dep in enumerate(dep_grid):
                for n_rev in range(args.n_min, args.n_max + 1):
                    task_index = (i_tof * len(dep_grid) + i_dep) * n_branch + (int(n_rev) - int(args.n_min))
                    dep, tof, branch, options = tasks[task_index]
                    task_options = dict(options)
                    grid_seed = previous_dep.get(int(branch))
                    seed_source = "grid_dep"
                    if grid_seed is None:
                        grid_seed = previous_row.get((i_dep, int(branch)))
                        seed_source = "grid_tof"
                    if grid_seed is not None:
                        task_options["initial_control_points"] = grid_seed
                        task_options["initial_control_points_source"] = seed_source
                    elif "initial_control_points" not in task_options:
                        task_options["initial_control_points_source"] = "analytic"

                    row = solve_one(dep, tof, branch, task_options)
                    rows.append(row)
                    seed = seed_control_points_from_row(row)
                    if seed is not None:
                        previous_dep[int(branch)] = seed
                        current_row[(i_dep, int(branch))] = seed
                    idx += 1
                    if args.progress and (idx == total or idx % max(1, args.progress_every) == 0):
                        print(f"completed {idx}/{total}")
            previous_row = current_row
    elif args.workers <= 1:
        for idx, (dep, tof, n_rev, options) in enumerate(tasks, start=1):
            rows.append(solve_one(dep, tof, n_rev, options))
            if args.progress and (idx == total or idx % max(1, args.progress_every) == 0):
                print(f"completed {idx}/{total}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(solve_one, dep, tof, n_rev, options) for dep, tof, n_rev, options in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if args.progress and (idx == total or idx % max(1, args.progress_every) == 0):
                    print(f"completed {idx}/{total}")

    dv_grid = np.full((len(tof_grid), len(dep_grid)), np.nan)
    best_n_grid = np.full((len(tof_grid), len(dep_grid)), -1, dtype=int)
    for i_tof, tof in enumerate(tof_grid):
        for i_dep, dep in enumerate(dep_grid):
            candidates = [
                row
                for row in rows
                if abs(row["tof_days"] - float(tof)) < 1e-9
                and abs(row["departure_mjd2000"] - float(dep)) < 1e-9
                and row["usable"]
                and np.isfinite(row["delta_v_km_s"])
            ]
            if candidates:
                best = min(candidates, key=lambda row: float(row["delta_v_km_s"]))
                dv_grid[i_tof, i_dep] = float(best["delta_v_km_s"])
                best_n_grid[i_tof, i_dep] = int(best["N"])
    return dep_grid, tof_grid, rows, dv_grid, best_n_grid


def write_attempts_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "departure_mjd2000",
        "tof_days",
        "N",
        "delta_v_km_s",
        "delta_v_optimizer_km_s",
        "delta_v_reference_km_s",
        "delta_v_reference_error_km_s",
        "energy_canonical",
        "success",
        "usable",
        "message",
        "endpoint_error",
        "winding_target_rev",
        "winding_sum_rev",
        "winding_error_rev",
        "winding_fine_rev",
        "max_u_fine",
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
        "seed_source",
        "seed_projection_success",
        "seed_projection_before",
        "seed_projection_after",
        "seed_projection_message",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["tof_days"], item["departure_mjd2000"], item["N"])):
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def sorted_attempt_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda item: (item["tof_days"], item["departure_mjd2000"], item["N"]))


def infer_control_point_shape(rows: list[dict], args: argparse.Namespace) -> tuple[int, int]:
    for row in rows:
        for key in ("control_points", "control_points_initial"):
            value = row.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return int(arr.shape[0]), 3
    return int(args.n_ctrl), 3


def control_point_array(row: dict, key: str, shape: tuple[int, int]) -> np.ndarray:
    value = row.get(key)
    if value is None:
        return np.full(shape, np.nan, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape:
        out = np.full(shape, np.nan, dtype=float)
        rows = min(shape[0], arr.shape[0]) if arr.ndim == 2 else 0
        cols = min(shape[1], arr.shape[1]) if arr.ndim == 2 else 0
        if rows and cols:
            out[:rows, :cols] = arr[:rows, :cols]
        return out
    return arr


def vector_array(row: dict, key: str, length: int) -> np.ndarray:
    value = row.get(key)
    if value is None:
        return np.full(length, np.nan, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    out = np.full(length, np.nan, dtype=float)
    out[: min(length, arr.size)] = arr[:length]
    return out


def lift_control_points_to_mats(
    source_control_points: np.ndarray,
    source_knots: np.ndarray,
    source_degree: int,
    target_mats,
    n_fit: int = 600,
    derivative_weight: float = 0.0,
) -> np.ndarray:
    """Refit a saved cylindrical spline onto another B-spline basis."""
    source_control_points = np.asarray(source_control_points, dtype=float)
    lift_matrix = control_point_lift_matrix(
        source_knots,
        source_degree,
        target_mats,
        n_fit=int(n_fit),
        derivative_weight=float(derivative_weight),
    )
    return np.asarray(lift_matrix @ source_control_points, dtype=float)


def control_point_lift_matrix(
    source_knots: np.ndarray,
    source_degree: int,
    target_mats,
    n_fit: int = 600,
    derivative_weight: float = 0.0,
) -> np.ndarray:
    """Build the linear map from one cylindrical B-spline basis to another."""
    source_knots = np.asarray(source_knots, dtype=float)
    source_degree = int(source_degree)
    if source_knots.ndim != 1:
        raise ValueError("source_knots must be a 1D array")
    n_source = len(source_knots) - source_degree - 1
    if n_source <= 0:
        raise ValueError("source_knots/source_degree imply no source control points")

    tau_fit = np.linspace(0.0, 1.0, int(n_fit))
    b_source = BSpline.design_matrix(tau_fit, source_knots, source_degree).toarray()
    b_fit = BSpline.design_matrix(tau_fit, target_mats.knots, target_mats.degree).toarray()
    ata = b_fit.T @ b_fit
    rhs_top_transform = b_fit.T @ b_source

    if float(derivative_weight) > 0.0:
        b1_source = (
            BSpline.design_matrix(tau_fit, source_knots[1:-1], source_degree - 1).toarray()
            @ derivative_matrix(source_knots, source_degree)
        )
        b1_fit = (
            BSpline.design_matrix(tau_fit, target_mats.knots[1:-1], target_mats.degree - 1).toarray()
            @ derivative_matrix(target_mats.knots, target_mats.degree)
        )
        weight = float(derivative_weight)
        ata += weight * (b1_fit.T @ b1_fit)
        rhs_top_transform += weight * (b1_fit.T @ b1_source)

    source_b0_start = BSpline.design_matrix([0.0], source_knots, source_degree).toarray()
    source_b1_start = (
        BSpline.design_matrix([0.0], source_knots[1:-1], source_degree - 1).toarray()
        @ derivative_matrix(source_knots, source_degree)
    )
    source_b0_end = BSpline.design_matrix([1.0], source_knots, source_degree).toarray()
    source_b1_end = (
        BSpline.design_matrix([1.0], source_knots[1:-1], source_degree - 1).toarray()
        @ derivative_matrix(source_knots, source_degree)
    )

    a_eq = np.vstack(
        [
            target_mats.b0_start,
            target_mats.b1_start,
            target_mats.b0_end,
            target_mats.b1_end,
        ]
    )
    source_eq = np.vstack(
        [
            source_b0_start,
            source_b1_start,
            source_b0_end,
            source_b1_end,
        ]
    )
    lhs = np.block(
        [
            [2.0 * ata, a_eq.T],
            [a_eq, np.zeros((a_eq.shape[0], a_eq.shape[0]))],
        ]
    )
    rhs_transform = np.vstack([2.0 * rhs_top_transform, source_eq])
    sol = np.linalg.lstsq(lhs, rhs_transform, rcond=None)[0]
    return np.asarray(sol[: b_fit.shape[1], :], dtype=float)


def load_continuation_control_points(
    path: Path,
    target_args: argparse.Namespace,
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
    n_fit: int = 600,
    derivative_weight: float = 0.0,
    require_usable: bool = True,
) -> dict[tuple[float, float, int], np.ndarray]:
    data = np.load(path, allow_pickle=False)
    source_control_points = np.asarray(data["control_points"], dtype=float)
    if source_control_points.ndim != 3 or source_control_points.shape[2] != 3:
        raise ValueError(f"{path} has invalid control_points shape {source_control_points.shape}")

    coordinate_value = data["control_point_coordinates"] if "control_point_coordinates" in data.files else CONTROL_POINT_COORDINATES
    if str(np.asarray(coordinate_value).item()) != CONTROL_POINT_COORDINATES:
        raise ValueError(f"{path} is not a cylindrical control-point archive")

    source_degree = int(np.asarray(data["degree"]).item())
    source_knots = np.asarray(data["knots"], dtype=float)
    departures = np.asarray(data["departure_mjd2000"], dtype=float)
    tofs = np.asarray(data["tof_days"], dtype=float)
    branches = np.asarray(data["N"], dtype=int)
    usable = np.asarray(data["usable"], dtype=bool) if "usable" in data.files else np.ones(len(departures), dtype=bool)

    target_mats = build_bspline_matrices(
        int(target_args.n_ctrl),
        degree=int(target_args.degree),
        n_fine=int(target_args.n_fine),
    )
    lift_matrix = control_point_lift_matrix(
        source_knots,
        source_degree,
        target_mats,
        n_fit=int(n_fit),
        derivative_weight=float(derivative_weight),
    )
    dep_values = {float(value) for value in np.asarray(dep_grid, dtype=float)}
    tof_values = {float(value) for value in np.asarray(tof_grid, dtype=float)}

    seeds: dict[tuple[float, float, int], np.ndarray] = {}
    for idx in range(len(departures)):
        if require_usable and not bool(usable[idx]):
            continue
        dep = float(departures[idx])
        tof = float(tofs[idx])
        if dep not in dep_values or tof not in tof_values:
            continue
        key = (dep, tof, int(branches[idx]))
        seeds[key] = np.asarray(lift_matrix @ source_control_points[idx], dtype=float)
    return seeds


def write_control_points_npz(path: Path, rows: list[dict], dep_grid: np.ndarray, tof_grid: np.ndarray, args: argparse.Namespace) -> None:
    ordered = sorted_attempt_rows(rows)
    cp_shape = infer_control_point_shape(ordered, args)
    if ordered:
        control_points = np.stack([control_point_array(row, "control_points", cp_shape) for row in ordered])
        control_points_initial = np.stack([control_point_array(row, "control_points_initial", cp_shape) for row in ordered])
        mee0 = np.stack([vector_array(row, "mee0", 6) for row in ordered])
        mee_target_epoch = np.stack([vector_array(row, "mee_target_epoch", 6) for row in ordered])
    else:
        control_points = np.empty((0, *cp_shape), dtype=float)
        control_points_initial = np.empty((0, *cp_shape), dtype=float)
        mee0 = np.empty((0, 6), dtype=float)
        mee_target_epoch = np.empty((0, 6), dtype=float)

    try:
        mats = build_bspline_matrices(
            int(args.n_ctrl),
            degree=int(args.degree),
            n_fine=int(args.n_fine),
        )
        knots = np.asarray(mats.knots, dtype=float)
        tau_start = float(mats.tau_start)
        tau_end = float(mats.tau_end)
    except Exception:
        knots = np.array([], dtype=float)
        tau_start = float("nan")
        tau_end = float("nan")

    np.savez(
        path,
        control_points=control_points,
        control_points_initial=control_points_initial,
        departure_mjd2000=np.asarray([row["departure_mjd2000"] for row in ordered], dtype=float),
        tof_days=np.asarray([row["tof_days"] for row in ordered], dtype=float),
        N=np.asarray([row["N"] for row in ordered], dtype=int),
        success=np.asarray([bool(row.get("success", False)) for row in ordered], dtype=bool),
        usable=np.asarray([bool(row.get("usable", False)) for row in ordered], dtype=bool),
        delta_v_km_s=np.asarray([row.get("delta_v_km_s", np.nan) for row in ordered], dtype=float),
        delta_v_optimizer_km_s=np.asarray(
            [row.get("delta_v_optimizer_km_s", np.nan) for row in ordered], dtype=float
        ),
        delta_v_reference_km_s=np.asarray(
            [row.get("delta_v_reference_km_s", np.nan) for row in ordered], dtype=float
        ),
        delta_v_reference_error_km_s=np.asarray(
            [row.get("delta_v_reference_error_km_s", np.nan) for row in ordered], dtype=float
        ),
        energy_canonical=np.asarray([row.get("energy_canonical", np.nan) for row in ordered], dtype=float),
        endpoint_error=np.asarray([row.get("endpoint_error", np.nan) for row in ordered], dtype=float),
        winding_target_rev=np.asarray([row.get("winding_target_rev", np.nan) for row in ordered], dtype=float),
        winding_sum_rev=np.asarray([row.get("winding_sum_rev", np.nan) for row in ordered], dtype=float),
        winding_error_rev=np.asarray([row.get("winding_error_rev", np.nan) for row in ordered], dtype=float),
        winding_fine_rev=np.asarray([row.get("winding_fine_rev", np.nan) for row in ordered], dtype=float),
        max_u_fine=np.asarray([row.get("max_u_fine", np.nan) for row in ordered], dtype=float),
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
        wall_time_s=np.asarray([row.get("wall_time_s", np.nan) for row in ordered], dtype=float),
        boundary_control_points_fixed=np.asarray(
            [bool(row.get("boundary_control_points_fixed", row.get("boundary_control_points_eliminated", False))) for row in ordered],
            dtype=bool,
        ),
        boundary_control_points_eliminated=np.asarray(
            [bool(row.get("boundary_control_points_eliminated", False)) for row in ordered],
            dtype=bool,
        ),
        n_free_control_points=np.asarray([row.get("n_free_control_points", -1) for row in ordered], dtype=int),
        seed_source=np.asarray([str(row.get("seed_source", "")) for row in ordered]),
        seed_projection_success=np.asarray(
            [bool(row.get("seed_projection_success", False)) for row in ordered],
            dtype=bool,
        ),
        seed_projection_before=np.asarray(
            [row.get("seed_projection_before", np.nan) for row in ordered],
            dtype=float,
        ),
        seed_projection_after=np.asarray(
            [row.get("seed_projection_after", np.nan) for row in ordered],
            dtype=float,
        ),
        seed_projection_message=np.asarray([str(row.get("seed_projection_message", "")) for row in ordered]),
        t_transfer_canonical=np.asarray([row.get("t_transfer_canonical", np.nan) for row in ordered], dtype=float),
        t_transfer_initial_canonical=np.asarray(
            [row.get("t_transfer_initial_canonical", np.nan) for row in ordered],
            dtype=float,
        ),
        mee0=mee0,
        mee_target_epoch=mee_target_epoch,
        mu=np.asarray([row.get("mu", MU_CANONICAL) for row in ordered], dtype=float),
        grid_departure_mjd2000=np.asarray(dep_grid, dtype=float),
        grid_tof_days=np.asarray(tof_grid, dtype=float),
        n_ctrl_requested=np.asarray(int(args.n_ctrl), dtype=int),
        n_ctrl_actual=np.asarray(cp_shape[0], dtype=int),
        degree=np.asarray(int(args.degree), dtype=int),
        n_fine=np.asarray(int(args.n_fine), dtype=int),
        target=np.asarray(str(args.target)),
        seed_profile=np.asarray(str(args.seed_profile)),
        objective=np.asarray("dv"),
        objective_quadrature=np.asarray("gauss"),
        run_id=np.asarray(str(getattr(args, "run_id", ""))),
        config_json=np.asarray(str(getattr(args, "config_json", ""))),
        linear_solver=np.asarray(str(getattr(args, "linear_solver", "mumps"))),
        coinhsl_library=np.asarray(str(getattr(args, "coinhsl_library", "") or "")),
        ipopt_iterations=np.asarray([row.get("ipopt_iterations", -1) for row in ordered], dtype=int),
        ephemeris_source=np.asarray(str(getattr(args, "ephemeris", "kepler"))),
        spice_meta_kernel=np.asarray(str(getattr(args, "spice_meta_kernel", "") or "")),
        spice_target_name=np.asarray(str(getattr(args, "spice_target_name", "") or "")),
        quadrature_order=np.asarray(int(getattr(args, "quadrature_order", 6)), dtype=int),
        r_bound=np.asarray(float(args.r_bound), dtype=float),
        dv_eps=np.asarray(float(args.dv_eps), dtype=float),
        smoothness_weight=np.asarray(float(args.smoothness_weight), dtype=float),
        endpoint_control_weight=np.asarray(float(args.endpoint_control_weight), dtype=float),
        kepler_substeps=np.asarray(int(args.kepler_substeps), dtype=int),
        continuation_seed_source=np.asarray(str(getattr(args, "continuation_seed_source", ""))),
        continuation_seed_count=np.asarray(int(getattr(args, "continuation_seed_count", 0)), dtype=int),
        grid_continuation=np.asarray(bool(getattr(args, "grid_continuation", False)), dtype=bool),
        grid_continuation_project=np.asarray(bool(getattr(args, "grid_continuation_project", False)), dtype=bool),
        knots=knots,
        tau_start=np.asarray(tau_start, dtype=float),
        tau_end=np.asarray(tau_end, dtype=float),
        control_point_coordinates=np.asarray(CONTROL_POINT_COORDINATES),
        control_point_components=CONTROL_POINT_COMPONENTS,
    )


def write_best_csv(path: Path, dep_grid: np.ndarray, tof_grid: np.ndarray, dv_grid: np.ndarray, best_n_grid: np.ndarray) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["departure_mjd2000", "tof_days", "delta_v_km_s", "best_N"])
        for i_tof, tof in enumerate(tof_grid):
            for i_dep, dep in enumerate(dep_grid):
                writer.writerow([dep, tof, dv_grid[i_tof, i_dep], best_n_grid[i_tof, i_dep]])


def dv_levels(dv_grid: np.ndarray, scale: str) -> np.ndarray:
    if scale == "gondelach":
        return np.array([6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0], dtype=float)

    finite = np.asarray(dv_grid[np.isfinite(dv_grid)], dtype=float)
    if finite.size == 0:
        return np.linspace(0.0, 1.0, 9)
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if abs(vmax - vmin) < 1e-12:
        pad = max(1.0, abs(vmin) * 0.1)
        vmin -= pad
        vmax += pad
    return np.linspace(vmin, vmax, 10)


def plot_bspline_with_matplotlib(
    path: Path,
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
    dv_grid: np.ndarray,
    best_n_grid: np.ndarray,
    scale: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = dv_levels(dv_grid, scale)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cf = ax.contourf(dep_grid, tof_grid, dv_grid, levels=levels, cmap="viridis_r", extend="both")
    cs = ax.contour(dep_grid, tof_grid, dv_grid, levels=levels, colors="black", linewidths=0.75)
    ax.clabel(cs, inline=True, fmt="%.2g", fontsize=8)
    ax.set_xlabel("Departure date [MJD2000]")
    ax.set_ylabel("Time of flight [days]")
    scale_label = "auto Delta-V scale" if scale == "auto" else "Gondelach 6-40 km/s scale"
    ax.set_title(f"Cylindrical B-spline Mars porkchop, N=0-5 ({scale_label})")
    cbar = fig.colorbar(cf, ax=ax, ticks=levels)
    cbar.set_label("Delta V [km/s]")
    ax.set_xlim(float(dep_grid[0]), float(dep_grid[-1]))
    ax.set_ylim(float(tof_grid[0]), float(tof_grid[-1]))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    n_path = path.with_name(path.stem + "_best_N.png")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    image = ax.imshow(
        best_n_grid,
        origin="lower",
        aspect="auto",
        extent=[float(dep_grid[0]), float(dep_grid[-1]), float(tof_grid[0]), float(tof_grid[-1])],
        cmap="tab10",
        vmin=-0.5,
        vmax=5.5,
    )
    ax.set_xlabel("Departure date [MJD2000]")
    ax.set_ylabel("Time of flight [days]")
    ax.set_title("Best formally converged B-spline branch, N=0-5")
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
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=5)
    parser.add_argument("--target", choices=["mars", "1989ml", "tempel1", "mercury"], default="mars")
    parser.add_argument("--n-ctrl", type=int, default=40)
    parser.add_argument("--degree", type=int, default=5)
    parser.add_argument("--n-fine", type=int, default=600)
    parser.add_argument("--seed-profile", choices=["linear", "cubic", "quintic"], default="quintic")
    parser.add_argument("--quadrature-order", type=int, default=6)
    parser.add_argument("--r-bound", type=float, default=20.0)
    parser.add_argument("--dv-eps", type=float, default=1e-6)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--endpoint-control-weight", type=float, default=0.0)
    parser.add_argument("--max-iter", type=int, default=700)
    parser.add_argument("--print-level", type=int, default=0)
    parser.add_argument("--kepler-substeps", type=int, default=80)
    parser.add_argument(
        "--accept-debug-feasible",
        action="store_true",
        help="Use Ipopt debug trajectories when endpoint and winding residuals pass tolerance.",
    )
    parser.add_argument("--endpoint-tol", type=float, default=1e-6)
    parser.add_argument("--winding-tol-rev", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--grid-continuation",
        action="store_true",
        help="Serially warm-start each branch from the previous departure/TOF solution, Gondelach-style.",
    )
    parser.add_argument(
        "--grid-continuation-project",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project continuation control points onto the current endpoint constraints before solving.",
    )
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument(
        "--dv-plot-scale",
        choices=["auto", "gondelach"],
        default="auto",
        help="Use automatic B-spline Delta-V scale, or force Gondelach's 6-40 km/s scale.",
    )
    parser.add_argument("--output-dir", default="output/gondelach_fig2_bspline_cylindrical")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    dep_grid, tof_grid, rows, dv_grid, best_n_grid = run_tasks(args)
    stem = "gondelach_fig2_bspline_cylindrical"
    attempts_path = output_dir / f"{stem}_attempts.csv"
    csv_path = output_dir / f"{stem}.csv"
    npz_path = output_dir / f"{stem}.npz"
    control_points_path = output_dir / f"{stem}_control_points.npz"
    png_path = output_dir / f"{stem}.png"

    write_attempts_csv(attempts_path, rows)
    write_control_points_npz(control_points_path, rows, dep_grid, tof_grid, args)
    write_best_csv(csv_path, dep_grid, tof_grid, dv_grid, best_n_grid)
    np.savez(
        npz_path,
        departure_mjd2000=dep_grid,
        tof_days=tof_grid,
        delta_v_km_s=dv_grid,
        best_N=best_n_grid,
    )
    plot_bspline_with_matplotlib(png_path, dep_grid, tof_grid, dv_grid, best_n_grid, args.dv_plot_scale)

    finite = dv_grid[np.isfinite(dv_grid)]
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {attempts_path}")
    print(f"wrote {control_points_path}")
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
        print("No finite successful B-spline solutions found.")


if __name__ == "__main__":
    main()

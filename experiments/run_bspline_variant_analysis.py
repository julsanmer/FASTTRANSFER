"""Run a B-spline variant against cached Gondelach comparison outputs.

This entry point is for trying a different number of B-spline control points
or polynomial degree without recomputing the Gondelach grids and without
overwriting the kept ``bspline10`` baseline files. It reads the existing
``comparison_grids.npz`` in the target output folder and writes variant-named
artifacts into that same folder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_gondelach_fig2_fig3_bspline10 import (  # noqa: E402
    CASE_CONFIGS,
    collect_bspline_pareto_points,
    configure_plot_fonts,
    fair_timing_metrics,
    finite_limits,
    plot_panel,
    plot_pareto,
    step_suffix,
    write_pareto_csv,
    write_summary_csv,
    write_timing_csv,
)
from experiments.reproduce_gondelach_fig2 import GONDELACH_FORMULATION_VERSION  # noqa: E402


BASELINE_N_CTRL = 10
BASELINE_DEGREE = 5
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_CONFIGS), default="mars")
    parser.add_argument("--bspline-n-ctrl", type=int, required=True)
    parser.add_argument("--bspline-degree", type=int, required=True)
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
    parser.add_argument("--ephemeris", choices=["auto", "kepler", "spice"], default="auto")
    parser.add_argument("--spice-meta-kernel", default=None)
    parser.add_argument("--spice-target-name", default=None)
    parser.add_argument("--bspline-quadrature-order", type=int, default=6)
    parser.add_argument(
        "--bspline-grid-continuation",
        action="store_true",
        help="Run the B-spline grid serially, warm-starting from neighboring solved grid points.",
    )
    parser.add_argument(
        "--bspline-grid-continuation-project",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Project grid-continuation seeds onto the current endpoint constraints before IPOPT.",
    )
    parser.add_argument(
        "--continuation-seed",
        choices=["auto", "none"],
        default="auto",
        help="Warm-start variants from a saved n_ctrl=10, degree=5 control-point archive when available.",
    )
    parser.add_argument(
        "--continuation-control-points",
        default=None,
        help="Explicit .npz control-point archive to use as continuation seed.",
    )
    parser.add_argument("--continuation-fit-points", type=int, default=600)
    parser.add_argument("--continuation-derivative-weight", type=float, default=0.0)
    parser.add_argument("--accept-debug-feasible", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--dep-step", type=float, default=20.0)
    parser.add_argument("--tof-step", type=float, default=20.0)
    parser.add_argument("--n-min", type=int, default=None)
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite this variant's own files. Baseline Gondelach/bspline10 files are still untouched.",
    )
    parser.add_argument(
        "--use-tex",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render Matplotlib text through an external LaTeX installation.",
    )
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    case = CASE_CONFIGS[args.case]
    suffix_args = argparse.Namespace(dep_step=args.dep_step, tof_step=args.tof_step)
    return Path(f"output/compare_gondelach_{case['output_name']}_bspline10{step_suffix(suffix_args)}")


def variant_tag(n_ctrl: int, degree: int, run_id: str = "") -> str:
    base = f"bspline_nctrl{int(n_ctrl)}_deg{int(degree)}"
    run_id = str(run_id or "").strip()
    return f"{base}_{run_id}" if run_id else base


def numerical_config(args: argparse.Namespace) -> dict:
    return {
        "schema": "bspline-variant-config-v1",
        "case": str(args.case),
        "n_control_points": int(args.bspline_n_ctrl),
        "degree": int(args.bspline_degree),
        "n_fine": int(args.bspline_n_fine),
        "quadrature_order": int(args.bspline_quadrature_order),
        "max_iterations": int(args.bspline_max_iter),
        "linear_solver": str(args.bspline_linear_solver),
        "ephemeris": str(args.ephemeris),
        "spice_meta_kernel": str(args.spice_meta_kernel or ""),
        "spice_target_name": str(args.spice_target_name or ""),
        "departure_step_days": float(args.dep_step),
        "tof_step_days": float(args.tof_step),
        "n_min": None if args.n_min is None else int(args.n_min),
        "n_max": None if args.n_max is None else int(args.n_max),
        "grid_continuation": bool(args.bspline_grid_continuation),
        "grid_continuation_project": bool(args.bspline_grid_continuation_project),
        "continuation_mode": str(args.continuation_seed),
        "continuation_control_points": str(args.continuation_control_points or ""),
        "continuation_fit_points": int(args.continuation_fit_points),
        "continuation_derivative_weight": float(args.continuation_derivative_weight),
        "accept_debug_feasible": bool(args.accept_debug_feasible),
    }


def config_run_id(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:8]


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: object, default: float = float("nan")) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_bspline_attempts(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "departure_mjd2000": parse_float(row.get("departure_mjd2000")),
                    "tof_days": parse_float(row.get("tof_days")),
                    "N": int(parse_float(row.get("N"), -1)),
                    "delta_v_km_s": parse_float(row.get("delta_v_km_s")),
                    "delta_v_optimizer_km_s": parse_float(row.get("delta_v_optimizer_km_s")),
                    "delta_v_reference_km_s": parse_float(row.get("delta_v_reference_km_s")),
                    "delta_v_reference_error_km_s": parse_float(row.get("delta_v_reference_error_km_s")),
                    "energy_canonical": parse_float(row.get("energy_canonical")),
                    "success": parse_bool(row.get("success")),
                    "usable": parse_bool(row.get("usable")),
                    "message": row.get("message", ""),
                    "endpoint_error": parse_float(row.get("endpoint_error")),
                    "winding_target_rev": parse_float(row.get("winding_target_rev")),
                    "winding_sum_rev": parse_float(row.get("winding_sum_rev")),
                    "winding_error_rev": parse_float(row.get("winding_error_rev")),
                    "winding_fine_rev": parse_float(row.get("winding_fine_rev")),
                    "max_u_fine": parse_float(row.get("max_u_fine")),
                    "u_max_reference_m_s2": parse_float(row.get("u_max_reference_m_s2")),
                    "u_max_reference_error_m_s2": parse_float(row.get("u_max_reference_error_m_s2")),
                    "reference_quadrature_order": int(parse_float(row.get("reference_quadrature_order"), -1)),
                    "reference_evaluations": int(parse_float(row.get("reference_evaluations"), 0)),
                    "reference_converged": parse_bool(row.get("reference_converged")),
                    "reference_metric_version": row.get("reference_metric_version", ""),
                    "wall_time_s": parse_float(row.get("wall_time_s")),
                    "boundary_control_points_fixed": parse_bool(
                        row.get("boundary_control_points_fixed", row.get("boundary_control_points_eliminated"))
                    ),
                    "boundary_control_points_eliminated": parse_bool(row.get("boundary_control_points_eliminated")),
                    "n_free_control_points": int(parse_float(row.get("n_free_control_points"), -1)),
                    "seed_source": row.get("seed_source", ""),
                    "seed_projection_success": parse_bool(row.get("seed_projection_success")),
                    "seed_projection_before": parse_float(row.get("seed_projection_before")),
                    "seed_projection_after": parse_float(row.get("seed_projection_after")),
                    "seed_projection_message": row.get("seed_projection_message", ""),
                }
            )
    return rows


def attempts_cache_has_fixed_control_metadata(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="") as file:
        reader = csv.reader(file)
        header = next(reader, [])
    return "boundary_control_points_fixed" in header or "boundary_control_points_eliminated" in header


def attempts_cache_linear_solver(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open(newline="") as file:
        row = next(csv.DictReader(file), {})
    return str(row.get("linear_solver", "") or "mumps").lower()


def load_timing_row(path: Path, method: str) -> dict:
    if not path.exists():
        return {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if str(row.get("method", "")) == str(method):
                return dict(row)
    return {}


def timing_csv_lacks_fair_columns(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open(newline="") as file:
        reader = csv.reader(file)
        header = next(reader, [])
    return "sum_attempt_wall_time_s" not in header


def write_bspline_attempts(path: Path, rows: list[dict]) -> None:
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


def bspline_grids_from_rows(
    rows: list[dict],
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dv_grid = np.full((len(tof_grid), len(dep_grid)), np.nan)
    best_n_grid = np.full((len(tof_grid), len(dep_grid)), -1, dtype=int)
    for i_tof, tof in enumerate(tof_grid):
        for i_dep, dep in enumerate(dep_grid):
            candidates = [
                row
                for row in rows
                if abs(float(row["tof_days"]) - float(tof)) < 1e-9
                and abs(float(row["departure_mjd2000"]) - float(dep)) < 1e-9
                and parse_bool(row.get("usable"))
                and np.isfinite(parse_float(row.get("delta_v_km_s")))
            ]
            if candidates:
                best = min(candidates, key=lambda row: float(row["delta_v_km_s"]))
                dv_grid[i_tof, i_dep] = float(best["delta_v_km_s"])
                best_n_grid[i_tof, i_dep] = int(best["N"])
    return dv_grid, best_n_grid


def should_write(path: Path, force: bool) -> bool:
    if path.exists() and not force:
        print(f"kept existing {path}")
        return False
    return True


def write_variant_best_csv(
    path: Path,
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
    fig2_dv: np.ndarray,
    fig2_n: np.ndarray,
    fig3_dv: np.ndarray,
    fig3_n: np.ndarray,
    variant_dv: np.ndarray,
    variant_n: np.ndarray,
    variant_name: str,
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
                f"{variant_name}_delta_v_km_s",
                f"{variant_name}_best_N",
                "fig3_minus_fig2_km_s",
                f"{variant_name}_minus_fig3_km_s",
                f"{variant_name}_minus_fig2_km_s",
            ]
        )
        for i_tof, tof in enumerate(tof_grid):
            for i_dep, dep in enumerate(dep_grid):
                f2 = fig2_dv[i_tof, i_dep]
                f3 = fig3_dv[i_tof, i_dep]
                bs = variant_dv[i_tof, i_dep]
                writer.writerow(
                    [
                        dep,
                        tof,
                        f2,
                        fig2_n[i_tof, i_dep],
                        f3,
                        fig3_n[i_tof, i_dep],
                        bs,
                        variant_n[i_tof, i_dep],
                        f3 - f2,
                        bs - f3,
                        bs - f2,
                    ]
                )


def write_baseline_variant_comparison_csv(
    path: Path,
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
    baseline_dv: np.ndarray,
    baseline_n: np.ndarray,
    variant_dv: np.ndarray,
    variant_n: np.ndarray,
    variant_name: str,
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "departure_mjd2000",
                "tof_days",
                "bspline10_delta_v_km_s",
                "bspline10_best_N",
                f"{variant_name}_delta_v_km_s",
                f"{variant_name}_best_N",
                f"{variant_name}_minus_bspline10_km_s",
            ]
        )
        for i_tof, tof in enumerate(tof_grid):
            for i_dep, dep in enumerate(dep_grid):
                base = baseline_dv[i_tof, i_dep]
                variant = variant_dv[i_tof, i_dep]
                writer.writerow(
                    [
                        dep,
                        tof,
                        base,
                        baseline_n[i_tof, i_dep],
                        variant,
                        variant_n[i_tof, i_dep],
                        variant - base,
                    ]
                )


def read_existing_pareto_points(path: Path, methods: set[str]) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    points_by_method: dict[str, list[dict]] = {method: [] for method in methods}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            method = str(row.get("method", ""))
            if method not in methods:
                continue
            try:
                points_by_method[method].append(
                    {
                        "departure_mjd2000": float(row["departure_mjd2000"]),
                        "tof_days": float(row["tof_days"]),
                        "N": int(float(row["N"])),
                        "delta_v_km_s": float(row["delta_v_km_s"]),
                        "fmax_m_s2": float(row["fmax_m_s2"]),
                        "source_success": parse_bool(row.get("source_success")),
                        "message": row.get("message", ""),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return {method: points for method, points in points_by_method.items() if points}


def build_bspline_args(args: argparse.Namespace, dep_grid: np.ndarray, tof_grid: np.ndarray) -> argparse.Namespace:
    case = CASE_CONFIGS[args.case]
    return argparse.Namespace(
        dep_min=float(dep_grid[0]),
        dep_max=float(dep_grid[-1]),
        tof_min=float(tof_grid[0]),
        tof_max=float(tof_grid[-1]),
        dep_step=float(args.dep_step),
        tof_step=float(args.tof_step),
        n_min=int(case["n_min"] if args.n_min is None else args.n_min),
        n_max=int(case["n_max"] if args.n_max is None else args.n_max),
        target=str(case["target"]),
        n_ctrl=int(args.bspline_n_ctrl),
        degree=int(args.bspline_degree),
        n_fine=int(args.bspline_n_fine),
        seed_profile="quintic",
        quadrature_order=int(args.bspline_quadrature_order),
        r_bound=20.0,
        dv_eps=1e-6,
        smoothness_weight=0.0,
        endpoint_control_weight=0.0,
        max_iter=int(args.bspline_max_iter),
        linear_solver=str(args.bspline_linear_solver),
        coinhsl_library=str(args.coinhsl_library or ""),
        print_level=0,
        kepler_substeps=80,
        accept_debug_feasible=bool(args.accept_debug_feasible),
        endpoint_tol=1e-6,
        winding_tol_rev=1e-3,
        workers=int(args.bspline_workers),
        grid_continuation=bool(args.bspline_grid_continuation),
        grid_continuation_project=bool(args.bspline_grid_continuation_project),
        project_initial_control_points=bool(args.bspline_grid_continuation and args.bspline_grid_continuation_project),
        progress=bool(args.progress),
        progress_every=int(args.progress_every),
        ephemeris=str(args.ephemeris),
        spice_meta_kernel=str(args.spice_meta_kernel or ""),
        spice_target_name=str(args.spice_target_name or ""),
    )


def continuation_source_path(output_dir: Path, args: argparse.Namespace):
    if args.continuation_control_points:
        return Path(args.continuation_control_points)
    candidates = [
        output_dir / f"{variant_tag(BASELINE_N_CTRL, BASELINE_DEGREE)}_gaussq6_control_points.npz",
        output_dir / f"{variant_tag(BASELINE_N_CTRL, BASELINE_DEGREE)}_control_points.npz",
        output_dir / "bspline10_control_points.npz",
    ]
    for path in candidates:
        if path.exists() and control_archive_uses_gauss(path):
            return path
    return None


def control_archive_uses_gauss(path: Path) -> bool:
    try:
        with np.load(path) as data:
            if "objective_quadrature" not in data.files:
                return False
            return str(np.asarray(data["objective_quadrature"]).item()).lower() == "gauss"
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    if args.bspline_linear_solver != "mumps":
        if not args.coinhsl_library:
            raise ValueError("--coinhsl-library is required with an HSL B-spline linear solver")
        if not Path(args.coinhsl_library).expanduser().is_file():
            raise FileNotFoundError(f"CoinHSL library not found: {args.coinhsl_library}")
    case = CASE_CONFIGS[args.case]
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    grids_path = output_dir / "comparison_grids.npz"
    if not grids_path.exists():
        raise FileNotFoundError(f"Missing cached Gondelach grid file: {grids_path}")

    os.environ["FASTTRANSFER_USE_LATEX"] = "1" if args.use_tex else "0"
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    cached = np.load(grids_path)
    cached_formulation = (
        str(np.asarray(cached["gondelach_formulation_version"]).item())
        if "gondelach_formulation_version" in cached.files
        else ""
    )
    if cached_formulation != GONDELACH_FORMULATION_VERSION:
        raise ValueError(
            "Cached Gondelach results use an obsolete formulation. "
            "Regenerate the combined baseline before running B-spline variants."
        )
    saved_ephemeris = str(np.asarray(cached["ephemeris_source"]).item()) if "ephemeris_source" in cached.files else "kepler"
    if args.ephemeris == "auto":
        args.ephemeris = saved_ephemeris
    saved_meta_kernel = str(np.asarray(cached["spice_meta_kernel"]).item()) if "spice_meta_kernel" in cached.files else ""
    saved_target_name = str(np.asarray(cached["spice_target_name"]).item()) if "spice_target_name" in cached.files else ""
    args.spice_meta_kernel = args.spice_meta_kernel or saved_meta_kernel or None
    args.spice_target_name = args.spice_target_name or saved_target_name or None
    if args.ephemeris != saved_ephemeris:
        raise ValueError(
            f"Variant ephemeris {args.ephemeris!r} does not match cached baseline {saved_ephemeris!r}."
        )
    dep_grid = np.asarray(cached["departure_mjd2000"], dtype=float)
    tof_grid = np.asarray(cached["tof_days"], dtype=float)
    fig2_dv = np.asarray(cached["fig2_delta_v_km_s"], dtype=float)
    fig2_n = np.asarray(cached["fig2_best_N"], dtype=int)
    fig3_dv = np.asarray(cached["fig3_delta_v_km_s"], dtype=float)
    fig3_n = np.asarray(cached["fig3_best_N"], dtype=int)
    baseline_bspline_dv = (
        np.asarray(cached["bspline10_delta_v_km_s"], dtype=float)
        if "bspline10_delta_v_km_s" in cached.files
        else None
    )
    baseline_bspline_n = (
        np.asarray(cached["bspline10_best_N"], dtype=int)
        if "bspline10_best_N" in cached.files
        else None
    )
    variant_optimizer_dv = None

    config = numerical_config(args)
    run_id = config_run_id(config)
    tag = variant_tag(args.bspline_n_ctrl, args.bspline_degree, run_id)
    label = f"B-spline, {args.bspline_n_ctrl} ctrl, degree {args.bspline_degree}"
    if args.bspline_grid_continuation:
        label += ", grid continuation"
    attempts_path = output_dir / f"{tag}_attempts.csv"
    control_points_path = output_dir / f"{tag}_control_points.npz"
    variant_grids_path = output_dir / f"comparison_grids_{tag}.npz"
    best_csv_path = output_dir / f"comparison_best_{tag}.csv"
    summary_csv_path = output_dir / f"comparison_summary_{tag}.csv"
    timing_csv_path = output_dir / f"comparison_timing_{tag}.csv"
    pareto_csv_path = output_dir / f"comparison_pareto_points_{tag}.csv"
    comparison_png_path = output_dir / f"comparison_fig3_vs_{tag}_delta_v.png"
    difference_png_path = output_dir / f"comparison_fig3_vs_{tag}_delta_v_difference.png"
    baseline_compare_csv_path = output_dir / f"comparison_bspline10_vs_{tag}.csv"
    baseline_difference_png_path = output_dir / f"comparison_bspline10_vs_{tag}_delta_v_difference.png"
    best_n_png_path = output_dir / f"comparison_best_N_{tag}.png"
    pareto_png_path = output_dir / f"comparison_pareto_{tag}.png"

    cache_has_fixed_metadata = attempts_cache_has_fixed_control_metadata(attempts_path)
    cache_linear_solver = attempts_cache_linear_solver(attempts_path)
    loaded_from_cache = (
        attempts_path.exists()
        and not args.force
        and cache_has_fixed_metadata
        and cache_linear_solver == args.bspline_linear_solver
    )
    if attempts_path.exists() and not args.force and not cache_has_fixed_metadata:
        print(f"existing {attempts_path} predates fixed boundary control points; recomputing canonical output")
    elif attempts_path.exists() and not args.force and cache_linear_solver != args.bspline_linear_solver:
        print(
            f"existing {attempts_path} uses linear_solver={cache_linear_solver}; "
            f"recomputing with {args.bspline_linear_solver}"
        )
    if loaded_from_cache:
        rows = load_bspline_attempts(attempts_path)
        variant_dv, variant_n = bspline_grids_from_rows(rows, dep_grid, tof_grid)
        cached_timing = load_timing_row(timing_csv_path, tag)
        bspline_time = parse_float(cached_timing.get("wall_time_s"), 0.0)
        timing_note = "loaded from cached variant attempts"
        if cached_timing.get("notes"):
            timing_note = str(cached_timing["notes"])
        print(f"loaded {attempts_path}")
        if control_points_path.exists():
            print(f"kept existing {control_points_path}")
        else:
            print(f"no control point archive found; rerun with --force to generate {control_points_path}")
    else:
        from experiments.reproduce_gondelach_fig2_bspline_cylindrical import (
            load_continuation_control_points,
            run_tasks,
            write_control_points_npz,
        )

        bspline_args = build_bspline_args(args, dep_grid, tof_grid)
        bspline_args.run_id = run_id
        bspline_args.config_json = json.dumps(config, sort_keys=True)
        use_continuation = (
            args.continuation_seed == "auto"
            and not (args.bspline_n_ctrl == BASELINE_N_CTRL and args.bspline_degree == BASELINE_DEGREE)
        )
        if use_continuation:
            source_path = continuation_source_path(output_dir, args)
            if source_path is None:
                print("no continuation control-point archive found; using analytic quintic seed")
            elif not source_path.exists():
                raise FileNotFoundError(f"Missing continuation control-point archive: {source_path}")
            else:
                seeds = load_continuation_control_points(
                    source_path,
                    bspline_args,
                    dep_grid,
                    tof_grid,
                    n_fit=int(args.continuation_fit_points),
                    derivative_weight=float(args.continuation_derivative_weight),
                    require_usable=True,
                )
                bspline_args.initial_control_points_by_key = seeds
                bspline_args.continuation_seed_source = str(source_path)
                bspline_args.continuation_seed_count = len(seeds)
                print(f"loaded {len(seeds)} continuation seeds from {source_path}")
        else:
            bspline_args.continuation_seed_source = ""
            bspline_args.continuation_seed_count = 0
        t0 = perf_counter()
        computed_dep, computed_tof, rows, variant_dv, variant_n = run_tasks(bspline_args)
        bspline_time = perf_counter() - t0
        if args.bspline_grid_continuation:
            timing_note = (
                f"variant solve with grid continuation; serial run; requested workers={args.bspline_workers}; "
                f"linear_solver={args.bspline_linear_solver}"
            )
        else:
            timing_note = (
                f"variant solve; workers={args.bspline_workers}; "
                f"linear_solver={args.bspline_linear_solver}"
            )
        if not np.allclose(computed_dep, dep_grid) or not np.allclose(computed_tof, tof_grid):
            raise RuntimeError("Computed B-spline grid does not match cached Gondelach grid.")
        if should_write(attempts_path, force=True):
            write_bspline_attempts(attempts_path, rows)
            print(f"wrote {attempts_path}")
        if should_write(control_points_path, force=True):
            write_control_points_npz(control_points_path, rows, dep_grid, tof_grid, bspline_args)
            print(f"wrote {control_points_path}")

    variant_optimizer_dv, _ = bspline_grids_from_rows(
        [
            {**row, "delta_v_km_s": row.get("delta_v_optimizer_km_s", np.nan)}
            for row in rows
        ],
        dep_grid,
        tof_grid,
    )

    grid_points = int(len(dep_grid) * len(tof_grid))
    n_min = int(case["n_min"] if args.n_min is None else args.n_min)
    n_max = int(case["n_max"] if args.n_max is None else args.n_max)
    branch_attempts = int(grid_points * (n_max - n_min + 1))
    used_workers = 1 if bool(args.bspline_grid_continuation) else max(1, int(args.bspline_workers))
    timing_rows = [
        {
            "method": tag,
            "wall_time_s": bspline_time,
            **fair_timing_metrics(
                rows,
                bspline_time,
                requested_workers=int(args.bspline_workers),
                used_workers=used_workers,
                grid_points=grid_points,
                branch_attempts=branch_attempts,
            ),
            "grid_points": grid_points,
            "branch_attempts": branch_attempts,
            "seconds_per_grid_point": bspline_time / max(grid_points, 1),
            "seconds_per_branch_attempt": bspline_time / max(branch_attempts, 1),
            "finite_points": int(np.isfinite(variant_dv).sum()),
            "usable_attempts": int(sum(parse_bool(row.get("usable")) for row in rows)),
            "formal_success_attempts": int(sum(parse_bool(row.get("success")) for row in rows)),
            "optimizer_function_evaluations": "",
            "notes": timing_note,
        }
    ]

    if should_write(variant_grids_path, args.force):
        np.savez(
            variant_grids_path,
            departure_mjd2000=dep_grid,
            tof_days=tof_grid,
            fig2_delta_v_km_s=fig2_dv,
            fig2_best_N=fig2_n,
            fig3_delta_v_km_s=fig3_dv,
            fig3_best_N=fig3_n,
            variant_delta_v_km_s=variant_dv,
            variant_best_N=variant_n,
            variant_optimizer_delta_v_km_s=variant_optimizer_dv,
            comparison_metric=np.asarray("reference"),
            reference_metric_version=np.asarray("composite_gauss_legendre_v1"),
            gondelach_formulation_version=np.asarray(GONDELACH_FORMULATION_VERSION),
            ephemeris_source=np.asarray(str(args.ephemeris)),
            spice_meta_kernel=np.asarray(str(args.spice_meta_kernel or "")),
            spice_target_name=np.asarray(str(args.spice_target_name or "")),
            bspline_n_ctrl=int(args.bspline_n_ctrl),
            bspline_degree=int(args.bspline_degree),
            bspline_variant_tag=np.asarray(tag),
            bspline_run_id=np.asarray(run_id),
            bspline_config_json=np.asarray(json.dumps(config, sort_keys=True)),
            bspline_grid_continuation=np.asarray(bool(args.bspline_grid_continuation), dtype=bool),
        )
        print(f"wrote {variant_grids_path}")

    if should_write(best_csv_path, args.force):
        write_variant_best_csv(best_csv_path, dep_grid, tof_grid, fig2_dv, fig2_n, fig3_dv, fig3_n, variant_dv, variant_n, tag)
        print(f"wrote {best_csv_path}")

    if (
        baseline_bspline_dv is not None
        and baseline_bspline_n is not None
        and should_write(baseline_compare_csv_path, args.force)
    ):
        write_baseline_variant_comparison_csv(
            baseline_compare_csv_path,
            dep_grid,
            tof_grid,
            baseline_bspline_dv,
            baseline_bspline_n,
            variant_dv,
            variant_n,
            tag,
        )
        print(f"wrote {baseline_compare_csv_path}")

    if should_write(summary_csv_path, args.force):
        write_summary_csv(
            summary_csv_path,
            {
                f"{args.case}_gondelach_low_order": fig2_dv,
                f"{args.case}_gondelach_high_order": fig3_dv,
                tag: variant_dv,
            },
        )
        print(f"wrote {summary_csv_path}")

    if should_write(timing_csv_path, args.force) or timing_csv_lacks_fair_columns(timing_csv_path):
        write_timing_csv(timing_csv_path, timing_rows)
        print(f"wrote {timing_csv_path}")

    pareto_points = read_existing_pareto_points(output_dir / "comparison_pareto_points.csv", {"gondelach_fig2", "gondelach_fig3"})
    pareto_points["bspline_variant"] = collect_bspline_pareto_points(rows)
    if should_write(pareto_csv_path, args.force):
        write_pareto_csv(pareto_csv_path, pareto_points)
        print(f"wrote {pareto_csv_path}")

    if should_write(comparison_png_path, args.force):
        import matplotlib

        matplotlib.use("Agg")
        configure_plot_fonts()
        import matplotlib.pyplot as plt

        vmin, vmax = finite_limits(fig3_dv, variant_dv)
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
        fig.suptitle(str(case["display"]))
        plot_panel(
            axes,
            dep_grid,
            tof_grid,
            [fig3_dv, variant_dv],
            [str(case["higher_label"]), label],
            comparison_png_path,
            "Delta V [km/s]",
            "viridis_r",
            vmin,
            vmax,
            calendar_axes=True,
        )
        print(f"wrote {comparison_png_path}")

    if should_write(difference_png_path, args.force):
        import matplotlib

        matplotlib.use("Agg")
        configure_plot_fonts()
        import matplotlib.pyplot as plt

        diff = variant_dv - fig3_dv
        finite_diff = diff[np.isfinite(diff)]
        diff_limit = float(np.nanmax(np.abs(finite_diff))) if finite_diff.size else 1.0
        if diff_limit < 1e-12:
            diff_limit = 1.0
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        plot_panel(
            [ax],
            dep_grid,
            tof_grid,
            [diff],
            [str(case["display"])],
            difference_png_path,
            "Delta V difference [km/s]",
            "coolwarm",
            -diff_limit,
            diff_limit,
            calendar_axes=True,
        )
        print(f"wrote {difference_png_path}")

    if baseline_bspline_dv is not None and should_write(baseline_difference_png_path, args.force):
        import matplotlib

        matplotlib.use("Agg")
        configure_plot_fonts()
        import matplotlib.pyplot as plt

        diff = variant_dv - baseline_bspline_dv
        finite_diff = diff[np.isfinite(diff)]
        diff_limit = float(np.nanmax(np.abs(finite_diff))) if finite_diff.size else 1.0
        if diff_limit < 1e-12:
            diff_limit = 1.0
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
        plot_panel(
            [ax],
            dep_grid,
            tof_grid,
            [diff],
            [f"{label} minus current 10-control baseline"],
            baseline_difference_png_path,
            "Delta V difference [km/s]",
            "coolwarm",
            -diff_limit,
            diff_limit,
            calendar_axes=True,
        )
        print(f"wrote {baseline_difference_png_path}")

    if should_write(best_n_png_path, args.force):
        import matplotlib

        matplotlib.use("Agg")
        configure_plot_fonts()
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
        fig.suptitle("Best selected branch N")
        plot_panel(
            axes,
            dep_grid,
            tof_grid,
            [fig3_n, variant_n],
            [str(case["higher_label"]), label],
            best_n_png_path,
            "Best N",
            "tab10",
            -0.5,
            max(5.5, float(n_max) + 0.5),
            integer_ticks=True,
            calendar_axes=True,
        )
        print(f"wrote {best_n_png_path}")

    if should_write(pareto_png_path, args.force):
        configure_plot_fonts()
        plot_pareto(
            pareto_png_path,
            pareto_points,
            None,
            {
                "gondelach_fig2": str(case["lower_label"]),
                "gondelach_fig3": str(case["higher_label"]),
                "bspline_variant": label,
            },
        )
        print(f"wrote {pareto_png_path}")


if __name__ == "__main__":
    main()

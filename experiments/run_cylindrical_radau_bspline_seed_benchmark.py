"""Benchmark cylindrical Radau direct collocation from saved B-spline seeds.

This script reads an existing Gondelach/B-spline comparison folder, rebuilds
the saved cylindrical B-spline profiles, and uses them as initial guesses for a
new cylindrical-coordinate Radau direct transcription.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_gondelach_fig2_fig3_bspline10 import CASE_CONFIGS, step_suffix  # noqa: E402
from experiments.reproduce_gondelach_fig2 import AU_KM, DAY_S, planet_state, wrap_0_2pi  # noqa: E402
from experiments.reproduce_gondelach_fig2_bspline_cylindrical import (  # noqa: E402
    CONTROL_POINT_COORDINATES,
    canonical_dv_to_km_s,
    rv_au_day_to_mee_canonical,
    tof_days_to_canonical,
)
from optimizer.canonical_units import MU_CANONICAL, UA_AU_PER_YR2  # noqa: E402
from optimizer.helpers_Bspline import derivative_matrix  # noqa: E402
from optimizer.optimization_cylindrical_Radaucollocation import (  # noqa: E402
    cylindrical_control_from_profile,
    solve_cylindrical_radau_collocation,
)
from utils.utils import kepler_coast_np  # noqa: E402


YEAR_DAYS = 365.25
AU_PER_YR2_TO_M_PER_S2 = (AU_KM * 1000.0) / ((YEAR_DAYS * DAY_S) ** 2)
CANONICAL_ACCEL_TO_M_S2 = UA_AU_PER_YR2 * AU_PER_YR2_TO_M_PER_S2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_CONFIGS), default="mars")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dep-step", type=float, default=20.0)
    parser.add_argument("--tof-step", type=float, default=20.0)
    parser.add_argument("--dep-min", type=float, default=None)
    parser.add_argument("--dep-max", type=float, default=None)
    parser.add_argument("--tof-min", type=float, default=None)
    parser.add_argument("--tof-max", type=float, default=None)
    parser.add_argument(
        "--bspline-variant",
        default="10:5",
        help="B-spline seed variant as n_ctrl:degree[:run_id]. Default uses the kept bspline10 baseline.",
    )
    parser.add_argument(
        "--selection",
        choices=["best-grid", "all-branches"],
        default="best-grid",
        help="Use the best usable B-spline branch per grid point, or every usable branch.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20,
        help="Maximum selected points to run. Use 0 for the full selected set.",
    )
    parser.add_argument("--n-intervals", type=int, default=30)
    parser.add_argument("--radau-degree", type=int, default=3)
    parser.add_argument("--objective", choices=["energy", "dv"], default="dv")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--print-level", type=int, default=0)
    parser.add_argument("--u-max-m-s2", type=float, default=None)
    parser.add_argument("--control-bound-m-s2", type=float, default=None)
    parser.add_argument("--endpoint-tol", type=float, default=1e-6)
    parser.add_argument(
        "--accept-debug-feasible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat finite IPOPT debug iterates as successful when endpoint residuals pass tolerance.",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def default_output_dir(args: argparse.Namespace) -> Path:
    case = CASE_CONFIGS[args.case]
    suffix_args = argparse.Namespace(dep_step=args.dep_step, tof_step=args.tof_step)
    return Path(f"output/compare_gondelach_{case['output_name']}_bspline10{step_suffix(suffix_args)}")


def parse_variant_key(spec: str) -> str:
    text = str(spec).strip()
    if text.startswith("bspline_nctrl"):
        if not re.fullmatch(r"bspline_nctrl\d+_deg\d+(?:_[0-9a-fA-F]{8})?", text):
            raise ValueError(f"Invalid configuration-hashed B-spline key: {spec!r}")
        return text
    parts = text.split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Expected B-spline variant as n_ctrl:degree[:run_id], got {spec!r}")
    n_ctrl = int(parts[0])
    degree = int(parts[1])
    run_id = parts[2].lower() if len(parts) == 3 else ""
    if run_id and not re.fullmatch(r"[0-9a-f]{8}", run_id):
        raise ValueError(f"B-spline run ID must contain eight hexadecimal characters: {spec!r}")
    base = f"bspline_nctrl{n_ctrl}_deg{degree}"
    return f"{base}_{run_id}" if run_id else base


def variant_paths(output_dir: Path, variant_key: str) -> tuple[Path, Path]:
    if variant_key == "bspline_nctrl10_deg5":
        attempts = output_dir / "bspline10_attempts.csv"
        controls = output_dir / "bspline10_control_points.npz"
        if attempts.exists() and controls.exists():
            return attempts, controls
    attempts = output_dir / f"{variant_key}_attempts.csv"
    controls = output_dir / f"{variant_key}_control_points.npz"
    if attempts.exists() and controls.exists():
        return attempts, controls
    matches = sorted(output_dir.glob(f"{variant_key}_[0-9a-f]" + "[0-9a-f]" * 7 + "_attempts.csv"))
    if len(matches) == 1:
        hashed_attempts = matches[0]
        hashed_key = hashed_attempts.name[: -len("_attempts.csv")]
        hashed_controls = output_dir / f"{hashed_key}_control_points.npz"
        if hashed_controls.exists():
            return hashed_attempts, hashed_controls
    if len(matches) > 1:
        raise ValueError(f"Multiple saved runs match {variant_key}; include the run ID")
    return attempts, controls


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: object, default: float = float("nan")) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_attempt_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            row["departure_mjd2000"] = parse_float(row.get("departure_mjd2000"))
            row["tof_days"] = parse_float(row.get("tof_days"))
            row["N"] = int(parse_float(row.get("N"), -1))
            row["delta_v_km_s"] = parse_float(row.get("delta_v_km_s"))
            row["max_u_fine"] = parse_float(row.get("max_u_fine"))
            row["usable"] = parse_bool(row.get("usable"))
            rows.append(row)
    return rows


def selected_seed_rows(rows: list[dict], selection: str, max_points: int) -> list[dict]:
    usable = [
        row
        for row in rows
        if bool(row.get("usable", False)) and np.isfinite(float(row.get("delta_v_km_s", np.nan)))
    ]
    if selection == "all-branches":
        selected = sorted(usable, key=lambda item: (item["tof_days"], item["departure_mjd2000"], item["N"]))
    else:
        best_by_grid: dict[tuple[float, float], dict] = {}
        for row in usable:
            key = (float(row["departure_mjd2000"]), float(row["tof_days"]))
            if key not in best_by_grid or float(row["delta_v_km_s"]) < float(best_by_grid[key]["delta_v_km_s"]):
                best_by_grid[key] = row
        selected = sorted(best_by_grid.values(), key=lambda item: (item["tof_days"], item["departure_mjd2000"]))
    if int(max_points) > 0:
        selected = selected[: int(max_points)]
    return selected


def filter_rows(rows: list[dict], args: argparse.Namespace) -> list[dict]:
    out = []
    for row in rows:
        dep = float(row["departure_mjd2000"])
        tof = float(row["tof_days"])
        if args.dep_min is not None and dep < float(args.dep_min) - 1e-9:
            continue
        if args.dep_max is not None and dep > float(args.dep_max) + 1e-9:
            continue
        if args.tof_min is not None and tof < float(args.tof_min) - 1e-9:
            continue
        if args.tof_max is not None and tof > float(args.tof_max) + 1e-9:
            continue
        out.append(row)
    return out


def control_archive_index(data, row: dict) -> int:
    dep = float(row["departure_mjd2000"])
    tof = float(row["tof_days"])
    n_rev = int(row["N"])
    matches = np.flatnonzero(
        np.isclose(np.asarray(data["departure_mjd2000"], dtype=float), dep, atol=1e-9, rtol=0.0)
        & np.isclose(np.asarray(data["tof_days"], dtype=float), tof, atol=1e-9, rtol=0.0)
        & (np.asarray(data["N"], dtype=int) == n_rev)
    )
    if not matches.size:
        raise ValueError(f"No saved controls for dep={dep:g}, tof={tof:g}, N={n_rev}")
    return int(matches[0])


def bspline_profile_seed(archive_path: Path, row: dict, n_fine: int = 1000) -> dict:
    with np.load(archive_path) as data:
        coordinates = str(np.asarray(data["control_point_coordinates"]).item()) if "control_point_coordinates" in data.files else CONTROL_POINT_COORDINATES
        if coordinates != CONTROL_POINT_COORDINATES:
            raise ValueError(f"Expected cylindrical controls in {archive_path}, got {coordinates!r}")
        idx = control_archive_index(data, row)
        control_points = np.asarray(data["control_points"][idx], dtype=float)
        degree = int(np.asarray(data["degree"]).item())
        knots = np.asarray(data["knots"], dtype=float)
        tf = float(np.asarray(data["t_transfer_canonical"], dtype=float)[idx])
        mu = float(np.asarray(data["mu"], dtype=float)[idx]) if "mu" in data.files else MU_CANONICAL

    tau = np.linspace(0.0, 1.0, int(n_fine))
    d1 = derivative_matrix(knots, degree)
    d2 = derivative_matrix(knots[1:-1], degree - 1)
    b0 = BSpline.design_matrix(tau, knots, degree).toarray()
    b1 = BSpline.design_matrix(tau, knots[1:-1], degree - 1).toarray() @ d1
    b2 = BSpline.design_matrix(tau, knots[2:-2], degree - 2).toarray() @ d2 @ d1
    q = b0 @ control_points
    qdot = (b1 @ control_points) / tf
    qddot = (b2 @ control_points) / (tf**2)
    u_cyl = cylindrical_control_from_profile(q, qdot, qddot, mu)
    return {
        "profile_fine": {
            "tau": tau,
            "q_cyl": q,
            "qdot_cyl": qdot,
            "qddot_cyl": qddot,
        },
        "t_transfer": tf,
        "mu": mu,
        "max_u": float(np.nanmax(np.linalg.norm(u_cyl, axis=1))),
    }


def accel_m_s2_to_canonical(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) / CANONICAL_ACCEL_TO_M_S2


def mars_grid_endpoint_data(row: dict, target: str) -> tuple[np.ndarray, np.ndarray, float, float]:
    dep = float(row["departure_mjd2000"])
    tof = float(row["tof_days"])
    n_rev = int(row["N"])
    pos0, vel0 = planet_state("earth", dep)
    posf, velf = planet_state(target, dep + tof)
    mee0 = rv_au_day_to_mee_canonical(pos0, vel0)
    meef = rv_au_day_to_mee_canonical(posf, velf)
    tf = tof_days_to_canonical(tof)
    mee_target_epoch = kepler_coast_np(meef, -tf, MU_CANONICAL, n_iter=80)
    theta0 = float(np.arctan2(pos0[1], pos0[0]))
    thetaf = float(np.arctan2(posf[1], posf[0]))
    winding_target_rev = (wrap_0_2pi(thetaf - theta0) / (2.0 * np.pi)) + n_rev
    return mee0, mee_target_epoch, tf, winding_target_rev


def write_results_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "departure_mjd2000",
        "tof_days",
        "N",
        "bspline_delta_v_km_s",
        "bspline_max_u_m_s2",
        "direct_delta_v_km_s",
        "direct_max_u_m_s2",
        "direct_success",
        "direct_formal_success",
        "direct_usable",
        "direct_endpoint_error_norm",
        "direct_wall_time_s",
        "direct_message",
        "seed_source",
        "n_intervals",
        "radau_degree",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def filter_suffix(args: argparse.Namespace) -> str:
    parts = []
    for name in ("dep_min", "dep_max", "tof_min", "tof_max"):
        value = getattr(args, name)
        if value is not None:
            parts.append(f"{name.replace('_', '')}{float(value):g}")
    return "" if not parts else "_" + "_".join(parts)


def main() -> None:
    args = parse_args()
    case = CASE_CONFIGS[args.case]
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    variant_key = parse_variant_key(args.bspline_variant)
    attempts_path, controls_path = variant_paths(output_dir, variant_key)
    if not attempts_path.exists():
        raise FileNotFoundError(f"Missing B-spline attempts file: {attempts_path}")
    if not controls_path.exists():
        raise FileNotFoundError(f"Missing B-spline control archive: {controls_path}")

    attempts = filter_rows(read_attempt_rows(attempts_path), args)
    selected = selected_seed_rows(attempts, args.selection, args.max_points)
    result_rows = []
    direct_dv_by_key: dict[tuple[float, float], float] = {}
    direct_n_by_key: dict[tuple[float, float], int] = {}
    u_max = accel_m_s2_to_canonical(args.u_max_m_s2)
    control_bound = accel_m_s2_to_canonical(args.control_bound_m_s2)

    for idx, row in enumerate(selected, start=1):
        if args.progress:
            print(f"direct cylindrical Radau {idx}/{len(selected)}: dep={row['departure_mjd2000']:g}, tof={row['tof_days']:g}, N={row['N']}")
        wall_t0 = perf_counter()
        try:
            seed = bspline_profile_seed(controls_path, row)
            mee0, mee_target_epoch, tf, winding_target_rev = mars_grid_endpoint_data(row, str(case["target"]))
            result = solve_cylindrical_radau_collocation(
                tf,
                mee0=mee0,
                mee_target_epoch=mee_target_epoch,
                mu=MU_CANONICAL,
                winding_target_rev=winding_target_rev,
                n_intervals=int(args.n_intervals),
                degree=int(args.radau_degree),
                objective=str(args.objective),
                u_max=u_max,
                control_bound=control_bound,
                dv_eps=1e-6,
                bspline_seed=seed,
                fixed_tf=True,
                max_iter=int(args.max_iter),
                print_level=int(args.print_level),
            )
            wall_time = perf_counter() - wall_t0
            direct_dv_km_s = canonical_dv_to_km_s(float(result["dv_quad"]))
            direct_max_u_m_s2 = float(result["max_u"]) * CANONICAL_ACCEL_TO_M_S2
            endpoint_norm = float(result["endpoint_error_norm"])
            direct_formal_success = bool(result["success"])
            message = str(result["message"])
            seed_source = str(result["seed_source"])
            bspline_max_u_m_s2 = float(seed.get("max_u", np.nan)) * CANONICAL_ACCEL_TO_M_S2
        except Exception as exc:
            wall_time = perf_counter() - wall_t0
            direct_dv_km_s = float("nan")
            direct_max_u_m_s2 = float("nan")
            endpoint_norm = float("nan")
            direct_formal_success = False
            message = str(exc).splitlines()[-1]
            seed_source = "failed_before_solve"
            bspline_max_u_m_s2 = float(row.get("max_u_fine", np.nan)) * CANONICAL_ACCEL_TO_M_S2

        feasible_debug = bool(np.isfinite(direct_dv_km_s) and endpoint_norm <= float(args.endpoint_tol))
        direct_success = bool(direct_formal_success or (args.accept_debug_feasible and feasible_debug))
        usable = bool(direct_success and feasible_debug)
        result_row = {
            "departure_mjd2000": float(row["departure_mjd2000"]),
            "tof_days": float(row["tof_days"]),
            "N": int(row["N"]),
            "bspline_delta_v_km_s": float(row["delta_v_km_s"]),
            "bspline_max_u_m_s2": bspline_max_u_m_s2,
            "direct_delta_v_km_s": direct_dv_km_s,
            "direct_max_u_m_s2": direct_max_u_m_s2,
            "direct_success": direct_success,
            "direct_formal_success": direct_formal_success,
            "direct_usable": usable,
            "direct_endpoint_error_norm": endpoint_norm,
            "direct_wall_time_s": wall_time,
            "direct_message": message,
            "seed_source": seed_source,
            "n_intervals": int(args.n_intervals),
            "radau_degree": int(args.radau_degree),
        }
        result_rows.append(result_row)

        key = (float(row["departure_mjd2000"]), float(row["tof_days"]))
        if usable and (key not in direct_dv_by_key or direct_dv_km_s < direct_dv_by_key[key]):
            direct_dv_by_key[key] = direct_dv_km_s
            direct_n_by_key[key] = int(row["N"])

    tag = f"direct_cylradau_from_{variant_key}_nint{int(args.n_intervals)}_deg{int(args.radau_degree)}"
    tag += filter_suffix(args)
    if int(args.max_points) > 0:
        tag += f"_first{int(args.max_points)}"
    csv_path = output_dir / f"{tag}.csv"
    write_results_csv(csv_path, result_rows)
    print(f"wrote {csv_path}")

    grids_path = output_dir / "comparison_grids.npz"
    if grids_path.exists():
        cached = np.load(grids_path)
        dep_grid = np.asarray(cached["departure_mjd2000"], dtype=float)
        tof_grid = np.asarray(cached["tof_days"], dtype=float)
        dv_grid = np.full((len(tof_grid), len(dep_grid)), np.nan)
        best_n = np.full((len(tof_grid), len(dep_grid)), -1, dtype=int)
        for i_tof, tof in enumerate(tof_grid):
            for i_dep, dep in enumerate(dep_grid):
                key = (float(dep), float(tof))
                if key in direct_dv_by_key:
                    dv_grid[i_tof, i_dep] = direct_dv_by_key[key]
                    best_n[i_tof, i_dep] = direct_n_by_key[key]
        npz_path = output_dir / f"{tag}_grids.npz"
        np.savez(
            npz_path,
            departure_mjd2000=dep_grid,
            tof_days=tof_grid,
            direct_delta_v_km_s=dv_grid,
            direct_best_N=best_n,
            source_variant=variant_key,
            n_intervals=int(args.n_intervals),
            radau_degree=int(args.radau_degree),
        )
        print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()

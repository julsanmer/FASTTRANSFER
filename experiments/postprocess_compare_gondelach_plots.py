"""Redraw saved Gondelach vs B-spline comparison plots from existing grids.

This is a plotting-only post-processing entry point. It reads
``comparison_grids.npz`` from an existing output folder and redraws the
high-order Gondelach vs cylindrical B-spline porkchop plots with:

- x-axis: calendar departure date
- y-axis: transfer time in years
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_gondelach_fig2_fig3_bspline10 import (  # noqa: E402
    CASE_CONFIGS,
    configure_plot_fonts,
    finite_limits,
    mjd2000_to_datetime,
    pareto_front,
    plot_panel,
    step_suffix,
)
from experiments.reproduce_gondelach_fig2 import (  # noqa: E402
    AU_KM,
    AU_PER_DAY2_TO_M_PER_S2,
    DAY_S,
    GONDELACH_FORMULATION_VERSION,
    MU_SUN_AU_DAY,
    basis_integral_matrix,
    basis_matrix,
    cartesian_to_cylindrical,
    configure_ephemeris,
    cumulative_simpson,
    parse_basis_group,
    planet_state,
    evaluate_time_driven_reference_metrics,
    solve_component_coefficients,
    solve_transverse_coefficients,
    split_free_coefficients,
    wrap_0_2pi,
)
from experiments.reference_metrics import (  # noqa: E402
    REFERENCE_METRIC_VERSION,
    evaluate_reference_metrics,
)
from optimizer.canonical_units import MU_CANONICAL, UA_AU_PER_YR2, UT_YR, UV_AU_PER_YR  # noqa: E402
from utils.utils import mee2rv, rv2mee  # noqa: E402


YEAR_DAYS = 365.25
AU_PER_YR2_TO_M_PER_S2 = (AU_KM * 1000.0) / ((YEAR_DAYS * DAY_S) ** 2)
CANONICAL_ACCEL_TO_M_S2 = UA_AU_PER_YR2 * AU_PER_YR2_TO_M_PER_S2
CANONICAL_DV_TO_KM_S = UV_AU_PER_YR * AU_KM / (YEAR_DAYS * DAY_S)
LEGEND_FONTSIZE = 14


@dataclass(frozen=True)
class BsplineVariantSpec:
    n_ctrl: int
    degree: int
    run_id: str = ""

    @property
    def key(self) -> str:
        base = f"bspline_nctrl{self.n_ctrl}_deg{self.degree}"
        return f"{base}_{self.run_id}" if self.run_id else base


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_CONFIGS), default="mars")
    parser.add_argument("--dep-step", type=float, default=20.0)
    parser.add_argument("--tof-step", type=float, default=20.0)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Existing comparison output directory. Defaults to the kept 20-day folder for --case.",
    )
    parser.add_argument(
        "--plots-output-dir",
        default=None,
        help="Directory for generated plots and profile CSVs. Defaults to --output-dir.",
    )
    parser.add_argument(
        "--departure-window-days",
        type=float,
        default=None,
        help=(
            "Restrict all plots and best-profile selection to departures in the half-open "
            "interval [first saved departure, first departure + DAYS)."
        ),
    )
    parser.add_argument(
        "--bspline-label",
        default="Quintic B-spline ($n_c=10$)",
        help="Panel label for the B-spline method.",
    )
    parser.add_argument(
        "--bspline-variant",
        default=None,
        help=(
            "B-spline result variant to use for the main porkchop and best-profile plots. "
            "Accepts n_ctrl:degree, n_ctrl:degree:run_id, or a full configuration-hashed key."
        ),
    )
    parser.add_argument(
        "--figure-format",
        choices=["png", "pdf", "eps"],
        default="png",
        help="Output format for the publication figures.",
    )
    parser.add_argument(
        "--use-tex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render all Matplotlib text through an external LaTeX installation.",
    )
    parser.add_argument("--ephemeris", choices=["auto", "kepler", "spice"], default="auto")
    parser.add_argument("--spice-meta-kernel", default=None)
    parser.add_argument("--spice-target-name", default=None)
    parser.add_argument(
        "--color-scale",
        choices=["paper", "low-dv"],
        default="low-dv",
        help=(
            "Delta-V color buckets. paper uses the Gondelach paper levels "
            "6,7,8,10,15,20,40 km/s; low-dv also includes a 5-6 km/s bucket."
        ),
    )
    parser.add_argument(
        "--color-map",
        choices=["gondelach-blue", "viridis-r"],
        default="gondelach-blue",
        help="Delta-V color palette for the comparison plot.",
    )
    parser.add_argument(
        "--delta-v-min",
        type=float,
        default=None,
        help="Minimum Delta-V color scale value [km/s]. Some cases use tailored bucket scales when omitted.",
    )
    parser.add_argument(
        "--delta-v-max",
        type=float,
        default=None,
        help="Maximum Delta-V color scale value [km/s]. Some cases use tailored bucket scales when omitted.",
    )
    parser.add_argument(
        "--smooth-regions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use filled contours for smoother bucket regions instead of cell-by-cell rendering.",
    )
    parser.add_argument(
        "--relative-error-limit",
        type=float,
        default=None,
        help="Symmetric percent limit for the relative-error plot. Defaults to the finite data maximum.",
    )
    parser.add_argument("--pareto-x-min", type=float, default=None, help="Minimum x-axis value for the zoomed Pareto plot.")
    parser.add_argument("--pareto-x-max", type=float, default=None, help="Maximum x-axis value for the zoomed Pareto plot.")
    parser.add_argument(
        "--pareto-y-min",
        type=float,
        default=None,
        help="Minimum y-axis value for the zoomed Pareto plot.",
    )
    parser.add_argument(
        "--pareto-y-max",
        type=float,
        default=None,
        help="Maximum y-axis value for the zoomed Pareto plot.",
    )
    parser.add_argument(
        "--pareto-swap-axes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Plot u_max on the x-axis and Delta-V on the y-axis. Mercury defaults to this layout.",
    )
    parser.add_argument(
        "--pareto-bspline-variants",
        nargs="+",
        default=None,
        help=(
            "B-spline variants to include in the publication Pareto plot. "
            "Accepts n_ctrl:degree, n_ctrl:degree:run_id, or full configuration-hashed keys. "
            "The kept baseline 10:5 is read from comparison_pareto_points.csv; other variants "
            "are read from comparison_pareto_points_bspline_nctrl*_deg*.csv when present."
        ),
    )
    parser.add_argument(
        "--plot-best-profiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot the best-Delta-V transfer/control profiles, using cached profile CSVs when present.",
    )
    parser.add_argument(
        "--compare-bspline-best-profiles",
        nargs=2,
        metavar=("VARIANT_A", "VARIANT_B"),
        default=None,
        help=(
            "Plot the best trajectory/control histories for two saved B-spline variants, "
            "for example 10:5:gaussq6 40:5:gaussq6. Profiles are reconstructed from saved controls."
        ),
    )
    parser.add_argument(
        "--stacked-bspline-best-profiles",
        nargs=2,
        metavar=("VARIANT_A", "VARIANT_B"),
        default=None,
        help=(
            "Plot two saved B-spline best trajectories and control histories in the same stacked "
            "layout as the standard best-control figure, for example 40:3 40:5. Profiles are "
            "reconstructed from saved controls."
        ),
    )
    parser.add_argument(
        "--write-global-statistics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write case-level global statistics CSVs from the saved grids using "
            "the selected --pareto-bspline-variants."
        ),
    )
    parser.add_argument("--best-profile-n-quad", type=int, default=1001)
    args = parser.parse_args()
    if args.pareto_bspline_variants is None:
        args.pareto_bspline_variants = (
            [str(args.bspline_variant)] if args.bspline_variant else ["10:5", "10:3", "40:5", "40:3"]
        )
    return args


def default_output_dir(args: argparse.Namespace) -> Path:
    case = CASE_CONFIGS[args.case]
    suffix_args = argparse.Namespace(dep_step=args.dep_step, tof_step=args.tof_step)
    return Path(f"output/compare_gondelach_{case['output_name']}_bspline10{step_suffix(suffix_args)}")


def transfer_title(case_name: str) -> str:
    case = CASE_CONFIGS[case_name]
    if case_name == "mars":
        return "Earth-Mars Transfer"
    return f"Earth-{case['display']} Transfer"


def output_figure_suffix(case_name: str) -> str:
    return str(CASE_CONFIGS[case_name]["display"]).replace(" ", "")


def output_figure_path(output_dir: Path, case_name: str, stem: str, extension: str = "png") -> Path:
    return output_dir / f"{stem}_{output_figure_suffix(case_name)}.{extension}"


def parse_bspline_variant_spec(spec: str) -> BsplineVariantSpec:
    text = str(spec).strip()
    key_match = re.fullmatch(r"bspline_nctrl(\d+)_deg(\d+)(?:_(.+))?", text)
    if key_match:
        n_ctrl = int(key_match.group(1))
        degree = int(key_match.group(2))
        run_id = str(key_match.group(3) or "")
    else:
        parts = text.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError(
                f"Expected B-spline variant spec as n_ctrl:degree[:run_id], got {spec!r}"
            )
        try:
            n_ctrl = int(parts[0])
            degree = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"Expected B-spline variant spec as n_ctrl:degree[:run_id], got {spec!r}"
            ) from exc
        run_id = parts[2].strip() if len(parts) == 3 else ""
    if n_ctrl <= 0 or degree <= 0:
        raise ValueError(f"B-spline variant values must be positive, got {spec!r}")
    if run_id and not re.fullmatch(r"[0-9a-fA-F]{8}", run_id):
        raise ValueError(f"B-spline run ID must contain exactly eight hexadecimal characters: {spec!r}")
    return BsplineVariantSpec(n_ctrl=n_ctrl, degree=degree, run_id=run_id.lower())


def bspline_variant_label(variant: BsplineVariantSpec) -> str:
    degree_names = {3: "Cubic", 5: "Quintic"}
    degree_label = degree_names.get(int(variant.degree), f"Degree {int(variant.degree)}")
    label = f"{degree_label} B-spline ($n_c={int(variant.n_ctrl)}$)"
    return label


def resolve_variant_artifact(
    output_dir: Path,
    exact_name: str,
    hashed_pattern: str,
) -> Path:
    exact = output_dir / exact_name
    if exact.exists():
        return exact
    matches = sorted(output_dir.glob(hashed_pattern))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(
            "Multiple configuration-hashed B-spline runs match this variant. "
            f"Select one by its eight-character run ID: {names}"
        )
    return exact


def is_nominal_baseline_variant(variant: BsplineVariantSpec | None) -> bool:
    return variant is None or (
        variant.n_ctrl == 10
        and variant.degree == 5
        and not variant.run_id
    )


def read_pareto_points(path: Path, method_map: dict[str, str]) -> dict[str, list[dict]]:
    points_by_method: dict[str, list[dict]] = {}
    if not path.exists():
        return points_by_method
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            source_method = str(row.get("method", ""))
            if source_method not in method_map:
                continue
            method = method_map[source_method]
            try:
                point = {
                    "method": method,
                    "delta_v_km_s": float(row["delta_v_km_s"]),
                    "fmax_m_s2": float(row["fmax_m_s2"]),
                    "N": int(float(row["N"])),
                    "departure_mjd2000": float(row["departure_mjd2000"]),
                    "tof_days": float(row["tof_days"]),
                    "reference_metric_version": row.get("reference_metric_version", ""),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(point["delta_v_km_s"]) and np.isfinite(point["fmax_m_s2"]):
                points_by_method.setdefault(method, []).append(point)
    return points_by_method


def merge_pareto_points(base: dict[str, list[dict]], extra: dict[str, list[dict]]) -> None:
    for method, points in extra.items():
        base.setdefault(method, []).extend(points)


def read_publication_pareto_points(
    output_dir: Path,
    variant_specs: list[str],
) -> tuple[dict[str, list[dict]], list[str], dict[str, str]]:
    variants = sorted(
        (parse_bspline_variant_spec(spec) for spec in variant_specs),
        key=lambda variant: (variant.n_ctrl, variant.degree, variant.run_id),
    )
    method_order = ["gondelach_fig2", "gondelach_fig3"]
    method_labels = {
        "gondelach_fig2": "Low-order hodographic",
        "gondelach_fig3": "High-order hodographic",
    }
    points_by_method = read_pareto_points(
        output_dir / "comparison_pareto_points.csv",
        {"gondelach_fig2": "gondelach_fig2", "gondelach_fig3": "gondelach_fig3"},
    )

    for variant in variants:
        key = variant.key
        method_order.append(key)
        method_labels[key] = bspline_variant_label(variant)
        if variant.n_ctrl == 10 and variant.degree == 5 and not variant.run_id:
            variant_path = resolve_variant_artifact(
                output_dir,
                f"comparison_pareto_points_{key}.csv",
                f"comparison_pareto_points_{key}_[0-9a-f]" + "[0-9a-f]" * 7 + ".csv",
            )
            if not points_by_method.get("gondelach_fig3") and variant_path.exists():
                merge_pareto_points(
                    points_by_method,
                    read_pareto_points(variant_path, {"gondelach_fig3": "gondelach_fig3"}),
                )
            if not points_by_method.get("gondelach_fig2") and variant_path.exists():
                merge_pareto_points(
                    points_by_method,
                    read_pareto_points(variant_path, {"gondelach_fig2": "gondelach_fig2"}),
                )
            baseline_points = read_pareto_points(
                output_dir / "comparison_pareto_points.csv",
                {"bspline10": key},
            )
            if baseline_points.get(key):
                merge_pareto_points(points_by_method, baseline_points)
            elif variant_path.exists():
                merge_pareto_points(
                    points_by_method,
                    read_pareto_points(variant_path, {"bspline_variant": key}),
                )
            continue

        variant_path = resolve_variant_artifact(
            output_dir,
            f"comparison_pareto_points_{key}.csv",
            f"comparison_pareto_points_{key}_[0-9a-f]" + "[0-9a-f]" * 7 + ".csv",
        )
        if not points_by_method.get("gondelach_fig3") and variant_path.exists():
            merge_pareto_points(
                points_by_method,
                read_pareto_points(variant_path, {"gondelach_fig3": "gondelach_fig3"}),
            )
        if not points_by_method.get("gondelach_fig2") and variant_path.exists():
            merge_pareto_points(
                points_by_method,
                read_pareto_points(variant_path, {"gondelach_fig2": "gondelach_fig2"}),
            )
        merge_pareto_points(points_by_method, read_pareto_points(variant_path, {"bspline_variant": key}))

    method_order = [method for method in method_order if points_by_method.get(method)]
    return points_by_method, method_order, method_labels


def filter_departure_rows(
    rows: list[dict[str, str]],
    departure_min: float | None,
    departure_max_exclusive: float | None,
) -> list[dict[str, str]]:
    if departure_min is None or departure_max_exclusive is None:
        return rows
    return [
        row
        for row in rows
        if departure_min <= parse_float(row.get("departure_mjd2000", "nan")) < departure_max_exclusive
    ]


def filter_pareto_departures(
    points_by_method: dict[str, list[dict]],
    departure_min: float | None,
    departure_max_exclusive: float | None,
) -> dict[str, list[dict]]:
    if departure_min is None or departure_max_exclusive is None:
        return points_by_method
    return {
        method: [
            point
            for point in points
            if departure_min <= float(point["departure_mjd2000"]) < departure_max_exclusive
        ]
        for method, points in points_by_method.items()
    }


def variant_file_suffix(variant: BsplineVariantSpec | None) -> str:
    return "" if is_nominal_baseline_variant(variant) else f"_{variant.key}"


def comparison_grids_path(output_dir: Path, variant: BsplineVariantSpec | None) -> Path:
    if is_nominal_baseline_variant(variant):
        return output_dir / "comparison_grids.npz"
    return resolve_variant_artifact(
        output_dir,
        f"comparison_grids_{variant.key}.npz",
        f"comparison_grids_{variant.key}_[0-9a-f]" + "[0-9a-f]" * 7 + ".npz",
    )


def bspline_attempts_path(output_dir: Path, variant: BsplineVariantSpec | None) -> Path:
    if is_nominal_baseline_variant(variant):
        return output_dir / "bspline10_attempts.csv"
    return resolve_variant_artifact(
        output_dir,
        f"{variant.key}_attempts.csv",
        f"{variant.key}_[0-9a-f]" + "[0-9a-f]" * 7 + "_attempts.csv",
    )


def bspline_control_points_path(output_dir: Path, variant: BsplineVariantSpec | None) -> Path:
    if is_nominal_baseline_variant(variant):
        baseline_path = output_dir / "bspline10_control_points.npz"
        if baseline_path.exists():
            return baseline_path
        return output_dir / "bspline_nctrl10_deg5_control_points.npz"
    return resolve_variant_artifact(
        output_dir,
        f"{variant.key}_control_points.npz",
        f"{variant.key}_[0-9a-f]" + "[0-9a-f]" * 7 + "_control_points.npz",
    )


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: str | float | int | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def best_delta_v_row(rows: list[dict[str, str]], usable_field: str) -> dict[str, str]:
    candidates = [
        row
        for row in rows
        if parse_bool(row.get(usable_field))
        and np.isfinite(parse_float(row.get("delta_v_km_s")))
    ]
    if not candidates:
        raise ValueError(f"No usable finite Delta-V rows found in {usable_field}")
    return min(candidates, key=lambda row: parse_float(row.get("delta_v_km_s")))


def free_coefficients_from_row(row: dict[str, str]) -> np.ndarray:
    coeffs = []
    idx = 0
    while f"free_{idx}" in row:
        value = row.get(f"free_{idx}", "")
        if value == "":
            break
        coeffs.append(float(value))
        idx += 1
    return np.asarray(coeffs, dtype=float)


def cylindrical_control_components_m_s2(
    rho: np.ndarray,
    z: np.ndarray,
    rho_dot: np.ndarray,
    theta_dot: np.ndarray,
    rho_ddot: np.ndarray,
    theta_ddot: np.ndarray,
    z_ddot: np.ndarray,
    mu: float,
    scale: float,
) -> np.ndarray:
    radius = np.sqrt(rho * rho + z * z)
    u_r = rho_ddot - rho * theta_dot * theta_dot + float(mu) * rho / (radius**3)
    u_theta = 2.0 * rho_dot * theta_dot + rho * theta_ddot
    u_z = z_ddot + float(mu) * z / (radius**3)
    return np.column_stack([u_r, u_theta, u_z]) * float(scale)


def evaluate_time_driven_profile(
    dep_mjd2000: float,
    tof_days: float,
    n_rev: int,
    n_quad: int,
    basis: str,
    free_coefficients: np.ndarray,
    target: str,
) -> dict:
    if int(n_quad) % 2 == 0:
        n_quad += 1
    tau = np.linspace(0.0, 1.0, int(n_quad))
    radial_terms, transverse_terms, axial_terms = [parse_basis_group(part) for part in basis.split("-")]

    pos0, vel0 = planet_state("earth", dep_mjd2000)
    posf, velf = planet_state(target, dep_mjd2000 + tof_days)
    q0, qdot0 = cartesian_to_cylindrical(pos0, vel0)
    qf, qdotf = cartesian_to_cylindrical(posf, velf)
    radial_free, transverse_free, axial_free = split_free_coefficients(
        free_coefficients,
        radial_terms,
        transverse_terms,
        axial_terms,
    )

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
    radial_values, radial_derivs = basis_matrix(radial_terms, tau, n_rev)
    radial_integrals = basis_integral_matrix(radial_terms, tau, n_rev)
    v_r = radial_values @ coeff_r
    rho = q0[0] + tof_days * (radial_integrals @ coeff_r)
    if np.any(~np.isfinite(rho)) or np.min(rho) <= 1e-6:
        raise ValueError("Invalid radial profile in Gondelach reconstruction")

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

    theta_values, theta_derivs = basis_matrix(transverse_terms, tau, n_rev)
    axial_values, axial_derivs = basis_matrix(axial_terms, tau, n_rev)
    axial_integrals = basis_integral_matrix(axial_terms, tau, n_rev)
    v_theta = theta_values @ coeff_theta
    v_z = axial_values @ coeff_z
    theta = q0[1] + tof_days * cumulative_simpson(v_theta / rho, tau)
    z = q0[2] + tof_days * (axial_integrals @ coeff_z)

    v_r_dot = (radial_derivs @ coeff_r) / tof_days
    v_theta_dot = (theta_derivs @ coeff_theta) / tof_days
    v_z_dot = (axial_derivs @ coeff_z) / tof_days
    theta_dot = v_theta / rho
    rho_dot = v_r
    z_dot = v_z
    theta_ddot = (v_theta_dot * rho - v_theta * rho_dot) / (rho * rho)
    rho_ddot = v_r_dot
    z_ddot = v_z_dot

    u_components = cylindrical_control_components_m_s2(
        rho,
        z,
        rho_dot,
        theta_dot,
        rho_ddot,
        theta_ddot,
        z_ddot,
        MU_SUN_AU_DAY,
        AU_PER_DAY2_TO_M_PER_S2,
    )
    u_norm = np.linalg.norm(u_components, axis=1)
    pos = np.column_stack([rho * np.cos(theta), rho * np.sin(theta), z])
    t_days = tau * tof_days
    reference = evaluate_time_driven_reference_metrics(
        dep_mjd2000,
        tof_days,
        n_rev,
        n_quad,
        basis,
        free_coefficients,
        target=target,
        coefficients=(coeff_r, coeff_theta, coeff_z),
    )
    return {
        "method": "gondelach_higher_order",
        "departure_mjd2000": float(dep_mjd2000),
        "tof_days": float(tof_days),
        "N": int(n_rev),
        "t_days": t_days,
        "pos_au": pos,
        "u_components_m_s2": u_components,
        "u_norm_m_s2": u_norm,
        "delta_v_km_s": float(reference["delta_v_reference_km_s"]),
        "fmax_m_s2": float(reference["u_max_reference_m_s2"]),
        **reference,
        "success": True,
        "message": "reconstructed from fig3_attempts.csv",
    }


def npz_scalar(data, key: str, default):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return value.item()
    if value.size == 1:
        return value.reshape(-1)[0].item()
    return default


def bspline_basis_matrix(tau: np.ndarray, knots: np.ndarray, degree: int) -> np.ndarray:
    tau = np.asarray(tau, dtype=float).reshape(-1)
    knots = np.asarray(knots, dtype=float).reshape(-1)
    degree = int(degree)
    if degree < 0:
        raise ValueError("B-spline degree must be non-negative")
    if len(knots) < degree + 2:
        raise ValueError("Knot vector is too short for requested degree")

    basis = np.zeros((tau.size, len(knots) - 1), dtype=float)
    for idx in range(len(knots) - 1):
        left = knots[idx]
        right = knots[idx + 1]
        basis[:, idx] = ((tau >= left) & (tau < right)).astype(float)
    basis[tau == knots[-1], -1] = 1.0

    for order in range(1, degree + 1):
        n_basis = len(knots) - order - 1
        next_basis = np.zeros((tau.size, n_basis), dtype=float)
        for idx in range(n_basis):
            left_denom = knots[idx + order] - knots[idx]
            right_denom = knots[idx + order + 1] - knots[idx + 1]
            if abs(left_denom) > 1e-14:
                next_basis[:, idx] += ((tau - knots[idx]) / left_denom) * basis[:, idx]
            if abs(right_denom) > 1e-14:
                next_basis[:, idx] += ((knots[idx + order + 1] - tau) / right_denom) * basis[:, idx + 1]
        basis = next_basis
    end_mask = np.isclose(tau, knots[-1], atol=1e-14, rtol=0.0)
    if np.any(end_mask):
        basis[end_mask, :] = 0.0
        basis[end_mask, -1] = 1.0
    return basis


def bspline_derivative_matrix(knots: np.ndarray, degree: int) -> np.ndarray:
    knots = np.asarray(knots, dtype=float).reshape(-1)
    degree = int(degree)
    n_ctrl = len(knots) - degree - 1
    matrix = np.zeros((n_ctrl - 1, n_ctrl), dtype=float)
    for idx in range(n_ctrl - 1):
        denom = knots[idx + degree + 1] - knots[idx + 1]
        if abs(denom) > 1e-14:
            matrix[idx, idx] = -degree / denom
            matrix[idx, idx + 1] = degree / denom
    return matrix


def bspline_profile_matrices_from_knots(
    tau: np.ndarray,
    knots: np.ndarray,
    degree: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b0 = bspline_basis_matrix(tau, knots, degree)
    if degree <= 0:
        return b0, np.zeros_like(b0), np.zeros_like(b0)

    d1 = bspline_derivative_matrix(knots, degree)
    b1 = bspline_basis_matrix(tau, knots[1:-1], degree - 1) @ d1
    if degree <= 1:
        return b0, b1, np.zeros_like(b0)

    d2 = bspline_derivative_matrix(knots[1:-1], degree - 1)
    b2 = bspline_basis_matrix(tau, knots[2:-2], degree - 2) @ d2 @ d1
    return b0, b1, b2


def control_archive_index(data, row: dict[str, str]) -> int:
    dep = parse_float(row["departure_mjd2000"])
    tof = parse_float(row["tof_days"])
    n_rev = int(parse_float(row["N"]))
    matches = np.flatnonzero(
        np.isclose(np.asarray(data["departure_mjd2000"], dtype=float), dep, atol=1e-9, rtol=0.0)
        & np.isclose(np.asarray(data["tof_days"], dtype=float), tof, atol=1e-9, rtol=0.0)
        & (np.asarray(data["N"], dtype=int) == n_rev)
    )
    if not matches.size:
        raise ValueError(f"No saved control points for dep={dep:g}, tof={tof:g}, N={n_rev}")
    return int(matches[0])


def reconstruct_bspline_profile_from_controls(
    archive_path: Path,
    row: dict[str, str],
) -> dict:
    with np.load(archive_path) as data:
        coordinates = str(np.asarray(data["control_point_coordinates"]).item()) if "control_point_coordinates" in data.files else "cylindrical"
        if coordinates != "cylindrical":
            raise ValueError(f"Saved control points are {coordinates!r}, expected 'cylindrical'")
        if "knots" not in data.files:
            raise ValueError(f"Saved control archive does not include a knot vector: {archive_path}")

        idx = control_archive_index(data, row)
        control_points = np.asarray(data["control_points"][idx], dtype=float)
        if control_points.ndim != 2 or control_points.shape[1] != 3 or not np.all(np.isfinite(control_points)):
            raise ValueError(f"Invalid saved control point array at index {idx}: shape={control_points.shape}")

        if "degree" not in data.files:
            raise ValueError(f"Saved control archive does not include the B-spline degree: {archive_path}")
        degree = int(npz_scalar(data, "degree", -1))
        n_fine = 1000
        tf = float(np.asarray(data["t_transfer_canonical"], dtype=float)[idx])
        mu = float(np.asarray(data["mu"], dtype=float)[idx]) if "mu" in data.files else MU_CANONICAL
        success = bool(np.asarray(data["success"], dtype=bool)[idx]) if "success" in data.files else parse_bool(row.get("success"))
        knots = np.asarray(data["knots"], dtype=float)

    tau_fine = np.linspace(0.0, 1.0, n_fine)
    b0, b1, b2 = bspline_profile_matrices_from_knots(tau_fine, knots, degree)
    if b0.shape[1] != control_points.shape[0]:
        raise ValueError(
            f"Saved knot vector/control point mismatch: basis has {b0.shape[1]} columns, "
            f"control points have {control_points.shape[0]} rows"
        )
    q = b0 @ control_points
    qdot = (b1 @ control_points) / tf
    qddot = (b2 @ control_points) / (tf**2)
    rho = q[:, 0]
    theta = q[:, 1]
    z = q[:, 2]
    pos = np.column_stack([rho * np.cos(theta), rho * np.sin(theta), z])
    u_components = cylindrical_control_components_m_s2(
        rho,
        z,
        qdot[:, 0],
        qdot[:, 1],
        qddot[:, 0],
        qddot[:, 1],
        qddot[:, 2],
        mu,
        CANONICAL_ACCEL_TO_M_S2,
    )
    u_norm = np.linalg.norm(u_components, axis=1)

    def acceleration_norm(tau: np.ndarray) -> np.ndarray:
        rb0, rb1, rb2 = bspline_profile_matrices_from_knots(tau, knots, degree)
        rq = rb0 @ control_points
        rqdot = (rb1 @ control_points) / tf
        rqddot = (rb2 @ control_points) / (tf**2)
        components = cylindrical_control_components_m_s2(
            rq[:, 0], rq[:, 2], rqdot[:, 0], rqdot[:, 1],
            rqddot[:, 0], rqddot[:, 1], rqddot[:, 2], mu, 1.0,
        )
        return np.linalg.norm(components, axis=1)

    reference = evaluate_reference_metrics(
        acceleration_norm,
        delta_v_scale_km_s=tf * CANONICAL_DV_TO_KM_S,
        acceleration_scale_m_s2=CANONICAL_ACCEL_TO_M_S2,
    ).as_dict()

    return {
        "method": "bspline_cylindrical",
        "departure_mjd2000": parse_float(row["departure_mjd2000"]),
        "tof_days": parse_float(row["tof_days"]),
        "N": int(parse_float(row["N"])),
        "t_days": tau_fine * tf * UT_YR * YEAR_DAYS,
        "pos_au": pos,
        "u_components_m_s2": u_components,
        "u_norm_m_s2": u_norm,
        "delta_v_km_s": float(reference["delta_v_reference_km_s"]),
        "fmax_m_s2": float(reference["u_max_reference_m_s2"]),
        **reference,
        "success": success,
        "message": f"reconstructed from saved control points: {archive_path.name}",
    }


def ephemeris_arc(body: str, t_start_mjd2000: float, t_end_mjd2000: float, count: int = 500) -> np.ndarray:
    epochs = np.linspace(float(t_start_mjd2000), float(t_end_mjd2000), int(count))
    return np.vstack([planet_state(body, epoch)[0] for epoch in epochs])


def target_orbit_arcs(
    body: str,
    departure_mjd2000: float,
    tof_days: float,
    count: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the transfer-time SPICE arc and departure osculating orbit."""
    departure = float(departure_mjd2000)
    arrival = departure + float(tof_days)
    solid_arc = ephemeris_arc(body, departure, arrival, count=count)

    position, velocity = planet_state(body, departure)
    radius = float(np.linalg.norm(position))
    specific_energy = 0.5 * float(np.dot(velocity, velocity)) - MU_SUN_AU_DAY / radius
    if not np.isfinite(specific_energy) or specific_energy >= 0.0:
        return solid_arc, np.empty((0, 3), dtype=float)

    departure_mee = rv2mee(position, velocity, MU_SUN_AU_DAY)
    longitudes = np.linspace(departure_mee[5], departure_mee[5] + 2.0 * math.pi, int(count))
    osculating_orbit = np.vstack(
        [
            mee2rv(
                np.concatenate((departure_mee[:5], np.array([longitude]))),
                MU_SUN_AU_DAY,
            )[0]
            for longitude in longitudes
        ]
    )
    osculating_orbit[-1] = osculating_orbit[0]
    return solid_arc, osculating_orbit


def format_departure_date(mjd2000: float) -> str:
    return mjd2000_to_datetime(float(mjd2000)).strftime("%Y-%m-%d")


def write_profile_csv(path: Path, profile: dict) -> None:
    pos = np.asarray(profile["pos_au"], dtype=float)
    u = np.asarray(profile["u_components_m_s2"], dtype=float)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["t_days", "x_au", "y_au", "z_au", "u_r_m_s2", "u_theta_m_s2", "u_z_m_s2", "u_norm_m_s2"])
        for idx in range(len(profile["t_days"])):
            writer.writerow(
                [
                    float(profile["t_days"][idx]),
                    float(pos[idx, 0]),
                    float(pos[idx, 1]),
                    float(pos[idx, 2]),
                    float(u[idx, 0]),
                    float(u[idx, 1]),
                    float(u[idx, 2]),
                    float(profile["u_norm_m_s2"][idx]),
                ]
            )


def load_profile_csv(path: Path, metadata: dict[str, str]) -> dict:
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"Cached profile CSV is empty: {path}")
    t_days = np.asarray([parse_float(row["t_days"]) for row in rows], dtype=float)
    pos = np.asarray(
        [
            [parse_float(row["x_au"]), parse_float(row["y_au"]), parse_float(row["z_au"])]
            for row in rows
        ],
        dtype=float,
    )
    u_components = np.asarray(
        [
            [parse_float(row["u_r_m_s2"]), parse_float(row["u_theta_m_s2"]), parse_float(row["u_z_m_s2"])]
            for row in rows
        ],
        dtype=float,
    )
    u_norm = np.asarray([parse_float(row["u_norm_m_s2"]) for row in rows], dtype=float)
    return {
        "method": str(metadata["method"]),
        "departure_mjd2000": parse_float(metadata["departure_mjd2000"]),
        "tof_days": parse_float(metadata["tof_days"]),
        "N": int(parse_float(metadata["N"])),
        "t_days": t_days,
        "pos_au": pos,
        "u_components_m_s2": u_components,
        "u_norm_m_s2": u_norm,
        "delta_v_km_s": parse_float(metadata["delta_v_km_s"]),
        "fmax_m_s2": parse_float(metadata["fmax_m_s2"]),
        "success": parse_bool(metadata.get("success")),
        "message": str(metadata.get("message", "loaded from cached profile CSV")),
        "reference_metric_version": str(metadata.get("reference_metric_version", "")),
        "reference_converged": parse_bool(metadata.get("reference_converged")),
        "delta_v_reference_error_km_s": parse_float(metadata.get("delta_v_reference_error_km_s")),
        "u_max_reference_error_m_s2": parse_float(metadata.get("u_max_reference_error_m_s2")),
    }


def write_best_profile_summary(path: Path, profiles: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "departure_mjd2000", "tof_days", "N", "delta_v_km_s", "fmax_m_s2", "delta_v_reference_error_km_s", "u_max_reference_error_m_s2", "reference_converged", "reference_metric_version", "gondelach_formulation_version", "success", "message"])
        for profile in profiles:
            writer.writerow(
                [
                    profile["method"],
                    profile["departure_mjd2000"],
                    profile["tof_days"],
                    profile["N"],
                    profile["delta_v_km_s"],
                    profile["fmax_m_s2"],
                    profile.get("delta_v_reference_error_km_s", ""),
                    profile.get("u_max_reference_error_m_s2", ""),
                    profile.get("reference_converged", ""),
                    profile.get("reference_metric_version", ""),
                    GONDELACH_FORMULATION_VERSION,
                    profile.get("success", ""),
                    profile.get("message", ""),
                ]
            )


def trajectory_z_limits(case_name: str) -> tuple[float, float] | None:
    return (-0.2, 0.2) if case_name == "mars" else None


def trajectory_xy_padding(case_name: str) -> float:
    return 0.15 if case_name == "1989ml" else 0.05


def calculate_3d_axis_limits(
    arrays: list[np.ndarray],
    z_limits: tuple[float, float] | None = None,
    xy_padding: float = 0.05,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    stacked = np.vstack([np.asarray(values, dtype=float) for values in arrays if len(values)])
    stacked = stacked[np.all(np.isfinite(stacked), axis=1)]
    if not len(stacked):
        raise ValueError("Cannot calculate 3D axis limits without finite positions")

    lower = np.min(stacked, axis=0)
    upper = np.max(stacked, axis=0)
    limits: list[tuple[float, float]] = []
    for axis in range(3):
        if axis == 2 and z_limits is not None:
            limits.append((float(z_limits[0]), float(z_limits[1])))
            continue
        minimum_span = 1e-3 if axis == 2 else 1e-6
        span = max(float(upper[axis] - lower[axis]), minimum_span)
        padding = (float(xy_padding) if axis < 2 else 0.05) * span
        limits.append((float(lower[axis] - padding), float(upper[axis] + padding)))
    return limits[0], limits[1], limits[2]


def set_3d_axis_limits(
    ax,
    limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    fixed_z_ticks: bool = False,
) -> None:
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    if fixed_z_ticks:
        ax.set_zticks(np.linspace(limits[2][0], limits[2][1], 5))


def format_3d_axis_ticks(ax) -> None:
    from matplotlib.ticker import FormatStrFormatter

    formatter = FormatStrFormatter("%.1f")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.zaxis.set_major_formatter(formatter)


def plot_3d_transfer_panel(
    ax,
    profile: dict,
    earth_arc: np.ndarray,
    target_arc: np.ndarray,
    target_osculating_orbit: np.ndarray,
    target_label: str,
    color: str,
    label: str,
    show_legend_labels: bool = False,
    z_limits: tuple[float, float] | None = None,
    axis_limits: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    pos = np.asarray(profile["pos_au"], dtype=float)
    ax.plot(
        earth_arc[:, 0],
        earth_arc[:, 1],
        earth_arc[:, 2],
        color="0.68",
        linestyle="-",
        linewidth=1.0,
        label="Earth orbit arc" if show_legend_labels else "_nolegend_",
    )
    if len(target_osculating_orbit):
        ax.plot(
            target_osculating_orbit[:, 0],
            target_osculating_orbit[:, 1],
            target_osculating_orbit[:, 2],
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            label="_nolegend_",
        )
    ax.plot(
        target_arc[:, 0],
        target_arc[:, 1],
        target_arc[:, 2],
        color="0.35",
        linestyle="-",
        linewidth=1.2,
        label=f"{target_label} orbit arc" if show_legend_labels else "_nolegend_",
    )
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, linewidth=2.0, label=label if show_legend_labels else "_nolegend_")
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color=color, marker="o", s=28, depthshade=False)
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], color=color, marker="s", s=28, depthshade=False)
    ax.scatter(
        [0.0],
        [0.0],
        [0.0],
        color="#f4d03f",
        edgecolor="black",
        s=55,
        depthshade=False,
        label="Sun" if show_legend_labels else "_nolegend_",
    )
    if axis_limits is None:
        axis_limits = calculate_3d_axis_limits(
            [earth_arc, target_arc, target_osculating_orbit, pos, np.zeros((1, 3))],
            z_limits=z_limits,
        )
    set_3d_axis_limits(ax, axis_limits, fixed_z_ticks=z_limits is not None)
    ax.set_xlabel(r"$x\ [\mathrm{AU}]$")
    ax.set_ylabel(r"$y\ [\mathrm{AU}]$")
    ax.set_zlabel(r"$z\ [\mathrm{AU}]$")
    format_3d_axis_ticks(ax)
    ax.view_init(elev=24.0, azim=-58.0)
    ax.grid(True, alpha=0.25)


def plot_best_transfer_profiles(
    path: Path,
    gondelach: dict,
    bspline: dict,
    case_name: str,
    bspline_label: str = "Quintic B-spline ($n_c=10$)",
    gondelach_label: str = "High-order hodographic",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    configure_plot_fonts()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    case = CASE_CONFIGS[case_name]
    epoch_start = min(float(gondelach["departure_mjd2000"]), float(bspline["departure_mjd2000"]))
    epoch_end = max(
        float(gondelach["departure_mjd2000"]) + float(gondelach["tof_days"]),
        float(bspline["departure_mjd2000"]) + float(bspline["tof_days"]),
    )
    earth_arc = ephemeris_arc("earth", epoch_start, epoch_end)
    gondelach_target_arc, gondelach_target_osculating_orbit = target_orbit_arcs(
        str(case["target"]),
        float(gondelach["departure_mjd2000"]),
        float(gondelach["tof_days"]),
    )
    bspline_target_arc, bspline_target_osculating_orbit = target_orbit_arcs(
        str(case["target"]),
        float(bspline["departure_mjd2000"]),
        float(bspline["tof_days"]),
    )
    z_limits = trajectory_z_limits(case_name)
    shared_3d_limits = calculate_3d_axis_limits(
        [
            earth_arc,
            gondelach_target_arc,
            gondelach_target_osculating_orbit,
            bspline_target_arc,
            bspline_target_osculating_orbit,
            np.asarray(gondelach["pos_au"], dtype=float),
            np.asarray(bspline["pos_au"], dtype=float),
            np.zeros((1, 3)),
        ],
        z_limits=z_limits,
        xy_padding=trajectory_xy_padding(case_name),
    )

    fig = plt.figure(figsize=(11.5, 8.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.12, 0.88))
    ax_hodo_3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_bspline_3d = fig.add_subplot(grid[1, 0], projection="3d")
    axes = np.array(
        [
            [ax_hodo_3d, fig.add_subplot(grid[0, 1])],
            [ax_bspline_3d, fig.add_subplot(grid[1, 1])],
        ],
        dtype=object,
    )
    plot_3d_transfer_panel(
        ax_hodo_3d,
        gondelach,
        earth_arc,
        gondelach_target_arc,
        gondelach_target_osculating_orbit,
        str(case["display"]),
        "b",
        "Trajectory",
        show_legend_labels=True,
        z_limits=z_limits,
        axis_limits=shared_3d_limits,
    )
    ax_hodo_3d.set_title(
        f"{gondelach_label} trajectory\n"
        f"Departure = {format_departure_date(gondelach['departure_mjd2000'])}, "
        f"TOF = {float(gondelach['tof_days']) / YEAR_DAYS:.3f} yr",
        y=1.04,
        pad=0.0,
    )
    plot_3d_transfer_panel(
        ax_bspline_3d,
        bspline,
        earth_arc,
        bspline_target_arc,
        bspline_target_osculating_orbit,
        str(case["display"]),
        "b",
        "Trajectory",
        z_limits=z_limits,
        axis_limits=shared_3d_limits,
    )
    ax_bspline_3d.set_title(
        f"{bspline_label} trajectory\n"
        f"Departure = {format_departure_date(bspline['departure_mjd2000'])}, "
        f"TOF = {float(bspline['tof_days']) / YEAR_DAYS:.3f} yr",
        y=1.04,
        pad=0.0,
    )

    control_styles = [
        (r"$u_r$", "#d55e00", "--"),
        (r"$u_\theta$", "#0072b2", "--"),
        (r"$u_z$", "#009e73", "--"),
    ]

    for ax, profile, title in [
        (axes[0, 1], gondelach, f"{gondelach_label} control"),
        (axes[1, 1], bspline, f"{bspline_label} control"),
    ]:
        u = 1000.0 * np.asarray(profile["u_components_m_s2"], dtype=float)
        norm = 1000.0 * np.asarray(profile["u_norm_m_s2"], dtype=float)
        t = np.asarray(profile["t_days"], dtype=float) / YEAR_DAYS
        for component_idx, (label, component_color, linestyle) in enumerate(control_styles):
            ax.plot(
                t,
                u[:, component_idx],
                color=component_color,
                linestyle=linestyle,
                linewidth=1.6,
                label=label,
                zorder=3,
            )
        ax.plot(t, norm, color="0.15", linestyle="-", linewidth=2.2, label=r"$\|\mathbf{u}\|$", zorder=2)
        ax.axhline(0.0, color="0.75", linewidth=0.8)
        ax.set_xlim(0.0, float(profile["tof_days"]) / YEAR_DAYS)
        ax.set_xlabel("Time [yr]")
        ax.set_ylabel(r"$u$ [mm/s$^2$]")
        ax.set_title(f"{title}\n$\\Delta V$ = {float(profile['delta_v_km_s']):.3f} km/s")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=LEGEND_FONTSIZE, ncol=2)

    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.075, top=0.94, wspace=0.18, hspace=0.42)
    for ax in (axes[0, 1], axes[1, 1]):
        position = ax.get_position()
        reduced_height = 0.88 * position.height
        ax.set_position([position.x0, position.y0 + 0.5 * (position.height - reduced_height), position.width, reduced_height])
    handles, labels = ax_hodo_3d.get_legend_handles_labels()
    hodo_box = ax_hodo_3d.get_position()
    fig.legend(
        handles,
        labels,
        fontsize=LEGEND_FONTSIZE,
        loc="upper left",
        bbox_to_anchor=(hodo_box.x0, hodo_box.y0 - 0.015),
        ncol=2,
        frameon=False,
    )
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_control_history_panel(ax, profile: dict, title: str) -> None:
    control_styles = [
        (r"$u_r$", "#d55e00", "--"),
        (r"$u_\theta$", "#0072b2", "--"),
        (r"$u_z$", "#009e73", "--"),
    ]
    u = 1000.0 * np.asarray(profile["u_components_m_s2"], dtype=float)
    norm = 1000.0 * np.asarray(profile["u_norm_m_s2"], dtype=float)
    t = np.asarray(profile["t_days"], dtype=float) / YEAR_DAYS
    for component_idx, (label, component_color, linestyle) in enumerate(control_styles):
        ax.plot(
            t,
            u[:, component_idx],
            color=component_color,
            linestyle=linestyle,
            linewidth=1.35,
            label=label,
            zorder=3,
        )
    ax.plot(t, norm, color="0.15", linestyle="-", linewidth=2.0, label=r"$\|\mathbf{u}\|$", zorder=2)
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.set_xlim(0.0, float(profile["tof_days"]) / YEAR_DAYS)
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel(r"$u$ [mm/s$^2$]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=LEGEND_FONTSIZE, ncol=2)


def plot_bspline_best_profile_comparison(
    path: Path,
    profile_a: dict,
    label_a: str,
    profile_b: dict,
    label_b: str,
    case_name: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    configure_plot_fonts()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    case = CASE_CONFIGS[case_name]
    epoch_start = min(float(profile_a["departure_mjd2000"]), float(profile_b["departure_mjd2000"]))
    epoch_end = max(
        float(profile_a["departure_mjd2000"]) + float(profile_a["tof_days"]),
        float(profile_b["departure_mjd2000"]) + float(profile_b["tof_days"]),
    )
    earth_arc = ephemeris_arc("earth", epoch_start, epoch_end)
    target_arc, target_osculating_orbit = target_orbit_arcs(
        str(case["target"]),
        epoch_start,
        epoch_end - epoch_start,
    )
    z_limits = trajectory_z_limits(case_name)
    pos_a = np.asarray(profile_a["pos_au"], dtype=float)
    pos_b = np.asarray(profile_b["pos_au"], dtype=float)

    fig = plt.figure(figsize=(12.0, 8.2))
    ax_3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_norm = fig.add_subplot(2, 2, 2)
    ax_a = fig.add_subplot(2, 2, 3)
    ax_b = fig.add_subplot(2, 2, 4)

    ax_3d.plot(earth_arc[:, 0], earth_arc[:, 1], earth_arc[:, 2], color="0.68", linewidth=1.0, label="Earth orbit arc")
    if len(target_osculating_orbit):
        ax_3d.plot(
            target_osculating_orbit[:, 0],
            target_osculating_orbit[:, 1],
            target_osculating_orbit[:, 2],
            color="0.35",
            linestyle="--",
            linewidth=1.2,
            label="_nolegend_",
        )
    ax_3d.plot(
        target_arc[:, 0],
        target_arc[:, 1],
        target_arc[:, 2],
        color="0.35",
        linewidth=1.2,
        label=f"{case['display']} orbit arc",
    )
    ax_3d.plot(pos_a[:, 0], pos_a[:, 1], pos_a[:, 2], color="#009e73", linewidth=2.0, label=label_a)
    ax_3d.plot(pos_b[:, 0], pos_b[:, 1], pos_b[:, 2], color="#7f3c8d", linewidth=2.0, label=label_b)
    ax_3d.scatter(pos_a[0, 0], pos_a[0, 1], pos_a[0, 2], color="#009e73", marker="o", s=28, depthshade=False)
    ax_3d.scatter(pos_a[-1, 0], pos_a[-1, 1], pos_a[-1, 2], color="#009e73", marker="s", s=28, depthshade=False)
    ax_3d.scatter(pos_b[0, 0], pos_b[0, 1], pos_b[0, 2], color="#7f3c8d", marker="o", s=28, depthshade=False)
    ax_3d.scatter(pos_b[-1, 0], pos_b[-1, 1], pos_b[-1, 2], color="#7f3c8d", marker="s", s=28, depthshade=False)
    ax_3d.scatter([0.0], [0.0], [0.0], color="#f4d03f", edgecolor="black", s=55, depthshade=False, label="Sun")
    axis_limits = calculate_3d_axis_limits(
        [earth_arc, target_arc, target_osculating_orbit, pos_a, pos_b, np.zeros((1, 3))],
        z_limits=z_limits,
        xy_padding=trajectory_xy_padding(case_name),
    )
    set_3d_axis_limits(ax_3d, axis_limits, fixed_z_ticks=z_limits is not None)
    ax_3d.set_xlabel(r"$x\ [\mathrm{AU}]$")
    ax_3d.set_ylabel(r"$y\ [\mathrm{AU}]$")
    ax_3d.set_zlabel(r"$z\ [\mathrm{AU}]$")
    format_3d_axis_ticks(ax_3d)
    ax_3d.set_title(
        "Best B-spline trajectories\n"
        f"{label_a}: Departure = {format_departure_date(profile_a['departure_mjd2000'])}, "
        f"TOF = {float(profile_a['tof_days']) / YEAR_DAYS:.3f} yr\n"
        f"{label_b}: Departure = {format_departure_date(profile_b['departure_mjd2000'])}, "
        f"TOF = {float(profile_b['tof_days']) / YEAR_DAYS:.3f} yr",
        pad=2.0,
    )
    ax_3d.view_init(elev=24.0, azim=-58.0)
    ax_3d.grid(True, alpha=0.25)
    ax_3d.legend(fontsize=LEGEND_FONTSIZE, loc="upper left")

    for profile, label, color in [(profile_a, label_a, "#009e73"), (profile_b, label_b, "#7f3c8d")]:
        t = np.asarray(profile["t_days"], dtype=float) / YEAR_DAYS
        norm = 1000.0 * np.asarray(profile["u_norm_m_s2"], dtype=float)
        ax_norm.plot(t, norm, color=color, linewidth=2.0, label=label)
    ax_norm.set_xlabel("Time [yr]")
    ax_norm.set_ylabel(r"$\|\mathbf{u}\|$ [mm/s$^2$]")
    ax_norm.set_title("Thrust magnitude")
    ax_norm.grid(True, alpha=0.25)
    ax_norm.legend(fontsize=LEGEND_FONTSIZE)

    plot_control_history_panel(
        ax_a,
        profile_a,
        f"{label_a} control\n$\\Delta V$ = {float(profile_a['delta_v_km_s']):.3f} km/s",
    )
    plot_control_history_panel(
        ax_b,
        profile_b,
        f"{label_b} control\n$\\Delta V$ = {float(profile_b['delta_v_km_s']):.3f} km/s",
    )

    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.07, top=0.93, wspace=0.22, hspace=0.46)
    for ax in (ax_a, ax_b):
        position = ax.get_position()
        reduced_height = 0.88 * position.height
        ax.set_position([position.x0, position.y0 + 0.5 * (position.height - reduced_height), position.width, reduced_height])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_bspline_best_profile(
    source_dir: Path,
    variant: BsplineVariantSpec,
    args: argparse.Namespace,
) -> tuple[dict, dict[str, str]]:
    attempts_path = bspline_attempts_path(source_dir, variant)
    control_points_path = bspline_control_points_path(source_dir, variant)
    if not attempts_path.exists():
        raise FileNotFoundError(f"Missing attempt CSV for {variant.key}: {attempts_path}")
    if not control_points_path.exists():
        raise FileNotFoundError(f"Missing control point archive for {variant.key}: {control_points_path}")
    rows = filter_departure_rows(
        read_rows(attempts_path), args.departure_min, args.departure_max_exclusive
    )
    row = best_delta_v_row(rows, "usable")
    profile = reconstruct_bspline_profile_from_controls(control_points_path, row)
    return profile, row


def build_and_plot_bspline_best_profile_comparison(
    source_dir: Path, plots_output_dir: Path, args: argparse.Namespace
) -> list[Path]:
    if not args.compare_bspline_best_profiles:
        return []
    variant_a = parse_bspline_variant_spec(args.compare_bspline_best_profiles[0])
    variant_b = parse_bspline_variant_spec(args.compare_bspline_best_profiles[1])
    profile_a, _ = build_bspline_best_profile(source_dir, variant_a, args)
    profile_b, _ = build_bspline_best_profile(source_dir, variant_b, args)
    label_a = bspline_variant_label(variant_a)
    label_b = bspline_variant_label(variant_b)
    compare_key = f"{variant_a.key}_vs_{variant_b.key}"
    plot_path = output_figure_path(plots_output_dir, args.case, f"besttransfercontrol_compare_{compare_key}", args.figure_format)
    csv_a = plots_output_dir / f"best_profile_{variant_a.key}.csv"
    csv_b = plots_output_dir / f"best_profile_{variant_b.key}.csv"
    summary_csv = plots_output_dir / f"best_transfer_profile_summary_compare_{compare_key}.csv"
    write_profile_csv(csv_a, profile_a)
    write_profile_csv(csv_b, profile_b)
    write_best_profile_summary(summary_csv, [profile_a, profile_b])
    plot_bspline_best_profile_comparison(plot_path, profile_a, label_a, profile_b, label_b, args.case)
    return [plot_path, csv_a, csv_b, summary_csv]


def build_and_plot_stacked_bspline_best_profiles(
    source_dir: Path, plots_output_dir: Path, args: argparse.Namespace
) -> list[Path]:
    if not args.stacked_bspline_best_profiles:
        return []
    variant_a = parse_bspline_variant_spec(args.stacked_bspline_best_profiles[0])
    variant_b = parse_bspline_variant_spec(args.stacked_bspline_best_profiles[1])
    profile_a, _ = build_bspline_best_profile(source_dir, variant_a, args)
    profile_b, _ = build_bspline_best_profile(source_dir, variant_b, args)
    label_a = bspline_variant_label(variant_a)
    label_b = bspline_variant_label(variant_b)
    compare_key = f"{variant_a.key}_vs_{variant_b.key}"
    plot_path = output_figure_path(
        plots_output_dir,
        args.case,
        f"besttransfercontrol_stacked_{compare_key}",
        args.figure_format,
    )
    csv_a = plots_output_dir / f"best_profile_{variant_a.key}.csv"
    csv_b = plots_output_dir / f"best_profile_{variant_b.key}.csv"
    summary_csv = plots_output_dir / f"best_transfer_profile_summary_stacked_{compare_key}.csv"
    write_profile_csv(csv_a, profile_a)
    write_profile_csv(csv_b, profile_b)
    write_best_profile_summary(summary_csv, [profile_a, profile_b])
    plot_best_transfer_profiles(
        plot_path,
        profile_a,
        profile_b,
        args.case,
        bspline_label=label_b,
        gondelach_label=label_a,
    )
    return [plot_path, csv_a, csv_b, summary_csv]


def build_and_plot_best_profiles(
    source_dir: Path, plots_output_dir: Path, args: argparse.Namespace
) -> list[Path]:
    case = CASE_CONFIGS[args.case]
    variant = parse_bspline_variant_spec(args.bspline_variant) if args.bspline_variant else None
    suffix = variant_file_suffix(variant)
    fig3_path = source_dir / "fig3_attempts.csv"
    bspline_path = bspline_attempts_path(source_dir, variant)
    control_points_path = bspline_control_points_path(source_dir, variant)
    plot_path = output_figure_path(plots_output_dir, args.case, f"besttransfercontrol{suffix}", args.figure_format)
    gondelach_csv = plots_output_dir / "best_profile_gondelach_higher_order.csv"
    bspline_csv = plots_output_dir / f"best_profile_bspline{suffix}.csv"
    summary_csv = plots_output_dir / f"best_transfer_profile_summary{suffix}.csv"
    cache_paths = [gondelach_csv, bspline_csv, summary_csv]
    loaded_from_cache = False

    if all(path.exists() for path in cache_paths):
        try:
            summary_by_method = {row["method"]: row for row in read_rows(summary_csv)}
            if any(
                row.get("reference_metric_version") != REFERENCE_METRIC_VERSION
                for row in summary_by_method.values()
            ):
                raise ValueError("Cached profiles use an obsolete metric evaluator")
            if any(
                row.get("gondelach_formulation_version") != GONDELACH_FORMULATION_VERSION
                for row in summary_by_method.values()
            ):
                raise ValueError("Cached profiles use an obsolete Gondelach formulation")
            gondelach = load_profile_csv(gondelach_csv, summary_by_method["gondelach_higher_order"])
            bspline = load_profile_csv(bspline_csv, summary_by_method["bspline_cylindrical"])
            loaded_from_cache = True
        except (KeyError, ValueError, OSError):
            loaded_from_cache = False

    if not loaded_from_cache:
        if not fig3_path.exists() or not bspline_path.exists():
            missing = [str(path) for path in [fig3_path, bspline_path] if not path.exists()]
            raise FileNotFoundError("Missing attempt CSV(s): " + ", ".join(missing))

        fig3_rows = filter_departure_rows(
            read_rows(fig3_path), args.departure_min, args.departure_max_exclusive
        )
        bspline_rows = filter_departure_rows(
            read_rows(bspline_path), args.departure_min, args.departure_max_exclusive
        )
        fig3_best = best_delta_v_row(fig3_rows, "usable")
        bspline_best = best_delta_v_row(bspline_rows, "usable")
        gondelach = evaluate_time_driven_profile(
            parse_float(fig3_best["departure_mjd2000"]),
            parse_float(fig3_best["tof_days"]),
            int(parse_float(fig3_best["N"])),
            int(args.best_profile_n_quad),
            str(case["higher_basis"]),
            free_coefficients_from_row(fig3_best),
            target=str(case["target"]),
        )
        if not control_points_path.exists():
            raise FileNotFoundError(
                "Postprocessing reconstructs B-spline profiles only from saved control points; "
                f"missing archive: {control_points_path}"
            )
        bspline = reconstruct_bspline_profile_from_controls(control_points_path, bspline_best)
        write_profile_csv(gondelach_csv, gondelach)
        write_profile_csv(bspline_csv, bspline)
        write_best_profile_summary(summary_csv, [gondelach, bspline])

    bspline_label = bspline_variant_label(variant) if variant is not None else str(args.bspline_label)
    plot_best_transfer_profiles(plot_path, gondelach, bspline, args.case, bspline_label)
    return [plot_path, gondelach_csv, bspline_csv, summary_csv]


def plot_pareto_zoom(
    path: Path,
    points_by_method: dict[str, list[dict]],
    title: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    swap_axes: bool = False,
    method_order: list[str] | None = None,
    method_labels: dict[str, str] | None = None,
    auto_fit_x: bool = True,
    auto_fit_y: bool = True,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    configure_plot_fonts()
    import matplotlib.pyplot as plt

    if method_order is None:
        method_order = list(points_by_method)
    if method_labels is None:
        method_labels = {}
    colors = {
        "gondelach_fig3": "#d62728",
        "bspline_nctrl10_deg3": "#5dade2",
        "bspline_nctrl10_deg5": "#1f77b4",
        "bspline_nctrl10_deg5_gaussq6": "#2e86c1",
        "bspline_nctrl40_deg3": "#2874a6",
        "bspline_nctrl40_deg5": "#0b3c5d",
    }
    fallback_colors = ["#5dade2", "#1f77b4", "#2874a6", "#0b3c5d", "#999999"]
    markers = {
        "gondelach_fig3": "^",
        "bspline_nctrl10_deg3": "o",
        "bspline_nctrl10_deg5": "o",
        "bspline_nctrl10_deg5_gaussq6": "o",
        "bspline_nctrl40_deg3": "s",
        "bspline_nctrl40_deg5": "s",
    }
    fallback_markers = ["o", "s", "D", "X", "*"]
    linestyles = {
        "gondelach_fig3": "-",
        "bspline_nctrl10_deg3": "--",
        "bspline_nctrl10_deg5": "-",
        "bspline_nctrl10_deg5_gaussq6": "-.",
        "bspline_nctrl40_deg3": "--",
        "bspline_nctrl40_deg5": "-",
    }

    def padded_limits(values: np.ndarray, lower: float, upper: float, pad_fraction: float = 0.08) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return lower, upper
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if abs(vmax - vmin) < 1e-12:
            pad = max(abs(vmin) * pad_fraction, (upper - lower) * 0.05, 1e-6)
        else:
            pad = (vmax - vmin) * pad_fraction
        return vmin - pad, vmax + pad

    visible_front_x: list[np.ndarray] = []
    visible_front_y: list[np.ndarray] = []
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    for idx, method in enumerate(method_order):
        points = points_by_method.get(method, [])
        if not points:
            continue
        color = colors.get(method, fallback_colors[idx % len(fallback_colors)])
        marker = markers.get(method, fallback_markers[idx % len(fallback_markers)])
        linestyle = linestyles.get(method, "-")
        label = method_labels.get(method, method)
        front = pareto_front(points)
        if front:
            front_delta_v = np.asarray([float(point["delta_v_km_s"]) for point in front], dtype=float)
            front_u_max = 1000.0 * np.asarray([float(point["fmax_m_s2"]) for point in front], dtype=float)
            fx, fy = (front_u_max, front_delta_v) if swap_axes else (front_delta_v, front_u_max)
            order = np.argsort(fx)
            fx = fx[order]
            fy = fy[order]
            in_front_view = (fx >= x_min) & (fx <= x_max) & (fy >= y_min) & (fy <= y_max)
            visible_front_x.append(fx[in_front_view])
            visible_front_y.append(fy[in_front_view])
            ax.plot(
                fx[in_front_view],
                fy[in_front_view],
                marker=marker,
                markersize=3.8,
                linewidth=2.0,
                linestyle=linestyle,
                color=color,
                clip_on=False,
                label=label,
                zorder=3,
            )

    if visible_front_x and auto_fit_x:
        x_min, x_max = padded_limits(np.concatenate(visible_front_x), x_min, x_max)
    if visible_front_y and auto_fit_y:
        y_min, y_max = padded_limits(np.concatenate(visible_front_y), y_min, y_max)

    ax.set_xlim(x_min, x_max)
    ax.margins(x=0.0)
    ax.set_ylim(y_min, y_max)
    if swap_axes:
        ax.set_xlabel(r"$u_{\max}\ [\mathrm{mm/s^2}]$")
        ax.set_ylabel(r"$\Delta V\ [\mathrm{km/s}]$")
    else:
        ax.set_xlabel(r"$\Delta V\ [\mathrm{km/s}]$")
        ax.set_ylabel(r"$u_{\max}\ [\mathrm{mm/s^2}]$")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(
        fontsize=LEGEND_FONTSIZE,
        loc="upper right",
        ncol=1,
        frameon=True,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def best_delta_v_by_tof(points: list[dict]) -> list[dict]:
    best_by_tof: dict[float, dict] = {}
    for point in points:
        try:
            delta_v = float(point.get("delta_v_km_s", np.nan))
            tof_days = float(point.get("tof_days", np.nan))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(delta_v) or not np.isfinite(tof_days):
            continue
        best = best_by_tof.get(tof_days)
        if best is None or delta_v < float(best["delta_v_km_s"]):
            best_by_tof[tof_days] = point
    return [best_by_tof[tof] for tof in sorted(best_by_tof)]


def plot_pareto_delta_v_tof(
    path: Path,
    points_by_method: dict[str, list[dict]],
    title: str,
    method_order: list[str] | None = None,
    method_labels: dict[str, str] | None = None,
    log_delta_v: bool = False,
    legend_loc: str | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    configure_plot_fonts()
    import matplotlib.pyplot as plt

    if method_order is None:
        method_order = list(points_by_method)
    if method_labels is None:
        method_labels = {}
    colors = {
        "gondelach_fig2": "#d62728",
        "gondelach_fig3": "#d62728",
        "bspline10": "#1f77b4",
        "bspline_nctrl10_deg3": "#5dade2",
        "bspline_nctrl10_deg5": "#1f77b4",
        "bspline_nctrl40_deg3": "#2874a6",
        "bspline_nctrl40_deg5": "#0b3c5d",
    }
    fallback_colors = ["#5dade2", "#1f77b4", "#2874a6", "#0b3c5d", "#999999"]
    markers = {
        "gondelach_fig2": "v",
        "gondelach_fig3": "^",
        "bspline10": "o",
        "bspline_nctrl10_deg3": "o",
        "bspline_nctrl10_deg5": "o",
        "bspline_nctrl40_deg3": "s",
        "bspline_nctrl40_deg5": "s",
    }
    fallback_markers = ["o", "s", "D", "X", "*"]
    linestyles = {
        "gondelach_fig2": "-",
        "gondelach_fig3": "-",
        "bspline10": "-",
        "bspline_nctrl10_deg3": "--",
        "bspline_nctrl10_deg5": "-",
        "bspline_nctrl40_deg3": "--",
        "bspline_nctrl40_deg5": "-",
    }

    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    for idx, method in enumerate(method_order):
        points = best_delta_v_by_tof(points_by_method.get(method, []))
        if not points:
            continue
        color = colors.get(method, fallback_colors[idx % len(fallback_colors)])
        marker = markers.get(method, fallback_markers[idx % len(fallback_markers)])
        linestyle = linestyles.get(method, "-")
        label = method_labels.get(method, method)
        delta_v = np.asarray([float(point["delta_v_km_s"]) for point in points], dtype=float)
        tof_days = np.asarray([float(point["tof_days"]) for point in points], dtype=float)
        order = np.argsort(tof_days)
        ax.plot(
            tof_days[order],
            delta_v[order],
            linewidth=2.0,
            linestyle=linestyle,
            color=color,
            clip_on=False,
            label=label,
            zorder=3,
        )
        best_idx = int(np.nanargmin(delta_v))
        ax.scatter(
            [tof_days[best_idx]],
            [delta_v[best_idx]],
            marker=marker,
            s=55,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            clip_on=False,
            zorder=4,
        )

    ax.margins(x=0.0)
    if log_delta_v:
        ax.set_yscale("log")
        y_min, y_max = ax.get_ylim()
        preferred_ticks = np.asarray(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 200.0],
            dtype=float,
        )
        visible_ticks = preferred_ticks[
            (preferred_ticks >= y_min) & (preferred_ticks <= y_max)
        ]
        ax.set_yticks(visible_ticks)
        ax.set_yticklabels([f"{tick:g}" for tick in visible_ticks])
    ax.set_xlabel(r"Transfer time [days]")
    ax.set_ylabel(r"Lowest $\Delta V$ [km/s]")
    ax.set_title(title)
    ax.grid(True, which="both" if log_delta_v else "major", alpha=0.25)
    ax.legend(
        fontsize=LEGEND_FONTSIZE,
        loc=legend_loc or ("upper right" if log_delta_v else "upper left"),
        ncol=1,
        frameon=True,
        framealpha=0.9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_case_global_statistics(
    source_dir: Path, plots_output_dir: Path, args: argparse.Namespace
) -> list[Path]:
    from experiments.postprocess_global_statistics import (  # noqa: E402
        compute_case_rows,
        parse_bspline_variant_spec as parse_statistics_variant_spec,
        write_csv,
    )

    variants = [parse_statistics_variant_spec(str(spec)) for spec in args.pareto_bspline_variants]
    rows, best_row = compute_case_rows(args.case, source_dir, variants)
    stats_path = plots_output_dir / "global_statistics_delta_v.csv"
    best_path = plots_output_dir / "global_statistics_best_method_by_grid_point.csv"
    write_csv(stats_path, rows)
    write_csv(best_path, [best_row])
    return [stats_path, best_path]


def read_timing_method_row(path: Path, method_names: set[str]) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            method = str(row.get("method", ""))
            if any(method == name or method.startswith(f"{name}_") for name in method_names):
                return dict(row)
    return {}


def finite_csv_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def per_grid_point_attempt_effort(
    path: Path,
    departure_min: float | None,
    departure_max_exclusive: float | None,
) -> tuple[np.ndarray, int, int]:
    grouped: dict[tuple[float, float], list[float]] = {}
    branch_attempts = 0
    timed_attempts = 0
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            try:
                departure = float(row["departure_mjd2000"])
                tof = float(row["tof_days"])
            except (KeyError, TypeError, ValueError):
                continue
            if departure_min is not None and departure < departure_min:
                continue
            if departure_max_exclusive is not None and departure >= departure_max_exclusive:
                continue
            branch_attempts += 1
            elapsed = finite_csv_float(row.get("wall_time_s"))
            if not np.isfinite(elapsed) or elapsed < 0.0:
                continue
            grouped.setdefault((departure, tof), []).append(elapsed)
            timed_attempts += 1
    effort = np.asarray([sum(values) for values in grouped.values()], dtype=float)
    return effort, branch_attempts, timed_attempts


def write_fair_computational_time_comparison(
    source_dir: Path,
    plots_output_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    variants: list[BsplineVariantSpec] = []
    seen: set[str] = set()
    for spec in args.pareto_bspline_variants:
        parsed = parse_bspline_variant_spec(str(spec))
        if parsed.key not in seen:
            variants.append(parsed)
            seen.add(parsed.key)

    methods: list[tuple[str, str, Path, Path, set[str]]] = [
        (
            "gondelach_fig3",
            "High-order hodographic",
            source_dir / "fig3_attempts.csv",
            source_dir / "comparison_timing.csv",
            {"gondelach_fig3"},
        )
    ]
    for variant in variants:
        if is_nominal_baseline_variant(variant):
            timing_path = source_dir / "comparison_timing.csv"
            timing_methods = {"bspline_cylindrical_nctrl10", "bspline_nctrl10_deg5"}
        else:
            timing_path = resolve_variant_artifact(
                source_dir,
                f"comparison_timing_{variant.key}.csv",
                f"comparison_timing_{variant.key}_[0-9a-f]" + "[0-9a-f]" * 7 + ".csv",
            )
            timing_methods = {variant.key}
        methods.append(
            (
                variant.key,
                bspline_variant_label(variant),
                bspline_attempts_path(source_dir, variant),
                timing_path,
                timing_methods,
            )
        )

    rows: list[dict[str, object]] = []
    for method, label, attempts_path, timing_path, timing_methods in methods:
        if not attempts_path.exists():
            continue
        effort, branch_attempts, timed_attempts = per_grid_point_attempt_effort(
            attempts_path,
            args.departure_min,
            args.departure_max_exclusive,
        )
        if not effort.size:
            continue
        timing = read_timing_method_row(timing_path, timing_methods)
        used_workers_value = finite_csv_float(timing.get("used_workers")) if timing else float("nan")
        requested_workers_value = finite_csv_float(timing.get("requested_workers")) if timing else float("nan")
        used_workers = int(used_workers_value) if np.isfinite(used_workers_value) else 1
        requested_workers = (
            int(requested_workers_value) if np.isfinite(requested_workers_value) else used_workers
        )
        used_workers = max(used_workers, 1)
        requested_workers = max(requested_workers, 1)
        parallel_wall = finite_csv_float(timing.get("parallel_wall_time_s", timing.get("wall_time_s")))
        worker_normalized_wall = finite_csv_float(timing.get("worker_normalized_wall_time_s"))
        efficiency = finite_csv_float(timing.get("parallel_efficiency"))
        full_grid_points_value = finite_csv_float(timing.get("grid_points")) if timing else float("nan")
        full_grid_points = int(full_grid_points_value) if np.isfinite(full_grid_points_value) else 0
        observed_parallel_per_point = (
            parallel_wall / full_grid_points
            if np.isfinite(parallel_wall) and full_grid_points > 0
            else float("nan")
        )
        overhead_adjusted_worker_per_point = (
            worker_normalized_wall / full_grid_points
            if np.isfinite(worker_normalized_wall) and full_grid_points > 0
            else float("nan")
        )
        rows.append(
            {
                "method": method,
                "method_label": label,
                "grid_points_with_timing": int(effort.size),
                "branch_attempts": branch_attempts,
                "timed_branch_attempts": timed_attempts,
                "requested_workers": requested_workers,
                "used_workers": used_workers,
                "mean_worker_seconds_per_grid_point": float(np.mean(effort)),
                "median_worker_seconds_per_grid_point": float(np.median(effort)),
                "p95_worker_seconds_per_grid_point": float(np.percentile(effort, 95.0)),
                "observed_parallel_wall_seconds_per_grid_point": observed_parallel_per_point,
                "overhead_adjusted_worker_seconds_per_grid_point": overhead_adjusted_worker_per_point,
                "parallel_efficiency": efficiency,
                "timing_definition": "sum of worker-local N-branch wall times per departure/TOF point",
            }
        )

    if not rows:
        return []
    csv_path = plots_output_dir / "comparison_computational_time.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    labels = [str(row["method_label"]) for row in rows]
    means = np.asarray([row["mean_worker_seconds_per_grid_point"] for row in rows], dtype=float)
    medians = np.asarray([row["median_worker_seconds_per_grid_point"] for row in rows], dtype=float)
    x = np.arange(len(rows), dtype=float)
    width = 0.36
    fig_width = max(8.0, 1.55 * len(rows) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4.6))
    ax.bar(x - width / 2.0, means, width, label="Mean", color="tab:blue")
    ax.bar(x + width / 2.0, medians, width, label="Median", color="tab:orange")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylabel("Worker time per grid point [s]")
    ax.set_title("Fair computational effort")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    fig.tight_layout()
    figure_path = output_figure_path(
        plots_output_dir,
        args.case,
        "comparisoncomputationaltime",
        args.figure_format,
    )
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return [csv_path, figure_path]


def main() -> None:
    args = parse_args()
    case = CASE_CONFIGS[args.case]
    source_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    plots_output_dir = Path(args.plots_output_dir) if args.plots_output_dir else source_dir
    plots_output_dir.mkdir(parents=True, exist_ok=True)
    variant = parse_bspline_variant_spec(args.bspline_variant) if args.bspline_variant else None
    suffix = variant_file_suffix(variant)
    method_key = "bspline" if is_nominal_baseline_variant(variant) else variant.key
    grids_path = comparison_grids_path(source_dir, variant)
    if not grids_path.exists():
        raise FileNotFoundError(f"Missing saved grid file: {grids_path}")

    os.environ["FASTTRANSFER_USE_LATEX"] = "1" if args.use_tex else "0"
    os.environ.setdefault("MPLCONFIGDIR", str(plots_output_dir / ".matplotlib"))
    (plots_output_dir / ".matplotlib").mkdir(parents=True, exist_ok=True)

    data = np.load(grids_path)
    saved_formulation = (
        str(np.asarray(data["gondelach_formulation_version"]).item())
        if "gondelach_formulation_version" in data.files
        else ""
    )
    if saved_formulation != GONDELACH_FORMULATION_VERSION:
        raise ValueError(
            "Saved Gondelach results use an obsolete formulation and must be regenerated."
        )
    saved_ephemeris = str(np.asarray(data["ephemeris_source"]).item()) if "ephemeris_source" in data.files else "kepler"
    ephemeris_source = saved_ephemeris if args.ephemeris == "auto" else args.ephemeris
    saved_meta_kernel = str(np.asarray(data["spice_meta_kernel"]).item()) if "spice_meta_kernel" in data.files else ""
    saved_target_name = str(np.asarray(data["spice_target_name"]).item()) if "spice_target_name" in data.files else ""
    configure_ephemeris(
        ephemeris_source,
        args.spice_meta_kernel or saved_meta_kernel or None,
        args.spice_target_name or saved_target_name or None,
    )
    dep_grid = data["departure_mjd2000"]
    tof_grid = data["tof_days"]
    fig3_dv = data["fig3_delta_v_km_s"]
    if "variant_delta_v_km_s" in data.files:
        bspline_dv = data["variant_delta_v_km_s"]
    elif "bspline10_delta_v_km_s" in data.files:
        bspline_dv = data["bspline10_delta_v_km_s"]
    else:
        raise KeyError(f"No B-spline Delta-V grid found in {grids_path}")
    args.departure_min = None
    args.departure_max_exclusive = None
    if args.departure_window_days is not None:
        if args.departure_window_days <= 0.0:
            raise ValueError("--departure-window-days must be positive")
        args.departure_min = float(np.min(dep_grid))
        args.departure_max_exclusive = args.departure_min + float(args.departure_window_days)
        dep_mask = (dep_grid >= args.departure_min) & (dep_grid < args.departure_max_exclusive)
        if not np.any(dep_mask):
            raise ValueError("The departure window does not contain any saved departure dates")
        dep_grid = dep_grid[dep_mask]
        fig3_dv = fig3_dv[:, dep_mask]
        bspline_dv = bspline_dv[:, dep_mask]
    bspline_panel_label = bspline_variant_label(variant) if variant is not None else str(args.bspline_label)

    vmin, vmax = finite_limits(fig3_dv, bspline_dv)

    import matplotlib

    matplotlib.use("Agg")
    configure_plot_fonts()
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    dep_spacing = np.diff(np.asarray(dep_grid, dtype=float))
    tof_spacing = np.diff(np.asarray(tof_grid, dtype=float))
    is_20_day_grid = (
        dep_spacing.size > 0
        and tof_spacing.size > 0
        and np.allclose(dep_spacing, 20.0)
        and np.allclose(tof_spacing, 20.0)
    )
    vertical_porkchop = args.case in {"tempel1", "1989ml"} and is_20_day_grid
    if vertical_porkchop:
        departure_span_years = max(
            (float(dep_grid[-1]) - float(dep_grid[0])) / YEAR_DAYS,
            1.0 / YEAR_DAYS,
        )
        tof_span_years = max(
            (float(tof_grid[-1]) - float(tof_grid[0])) / YEAR_DAYS,
            1.0 / YEAR_DAYS,
        )
        panel_aspect = tof_span_years / departure_span_years
        figure_height = float(np.clip(2.0 * 5.0 * panel_aspect + 1.3, 3.4, 6.8))
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(7.2, figure_height),
            sharex=True,
            sharey=True,
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8), sharex=True, sharey=True)
    tailored_delta_v_bins = {
        "1989ml": [4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 30.0, 60.0],
        "mercury": [18.0, 20.0, 22.0, 24.0, 26.0, 30.0, 40.0, 80.0, 160.0],
        "tempel1": [12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 30.0, 60.0],
    }
    use_tailored_default_scale = args.case in tailored_delta_v_bins and args.delta_v_min is None and args.delta_v_max is None
    if use_tailored_default_scale:
        delta_v_bins = tailored_delta_v_bins[args.case]
        delta_v_bin_labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(delta_v_bins[:-1], delta_v_bins[1:])]
        vmin = delta_v_bins[0]
        vmax = delta_v_bins[-1]
        delta_v_extend = "max"
    else:
        if args.delta_v_min is not None:
            vmin = float(args.delta_v_min)
        if args.delta_v_max is not None:
            vmax = float(args.delta_v_max)
        if not vmax > vmin:
            raise ValueError(f"Delta-V color scale requires max > min, got {vmin:g} to {vmax:g}")
        delta_v_extend = "neither"

    if args.delta_v_min is not None or args.delta_v_max is not None:
        delta_v_bins = np.linspace(vmin, vmax, 9).tolist()
        delta_v_bin_labels = [f"{lo:g}-{hi:g}" for lo, hi in zip(delta_v_bins[:-1], delta_v_bins[1:])]
    elif use_tailored_default_scale:
        pass
    elif args.color_scale == "paper":
        delta_v_bins = [6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0]
        delta_v_bin_labels = ["6-7", "7-8", "8-10", "10-15", "15-20", "20-40"]
    else:
        delta_v_bins = [5.0, 6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0]
        delta_v_bin_labels = ["5-6", "6-7", "7-8", "8-10", "10-15", "15-20", "20-40"]
    if args.color_map == "gondelach-blue":
        delta_v_cmap = ListedColormap(
            [
                "#05008f",
                "#0035ff",
                "#087cff",
                "#12a7ee",
                "#17cfe2",
                "#00e0bd",
                "#78ff3a",
            ],
            name="gondelach_blue",
        )
    else:
        delta_v_cmap = "viridis_r"
    plot_panel(
        axes,
        dep_grid,
        tof_grid,
        [fig3_dv, bspline_dv],
        ["High-order hodographic", bspline_panel_label],
        output_figure_path(plots_output_dir, args.case, f"comparisonporkchop{suffix}", args.figure_format),
        r"$\Delta V\ [\mathrm{km/s}]$",
        delta_v_cmap,
        vmin,
        vmax,
        calendar_axes=True,
        bin_edges=delta_v_bins,
        bin_tick_style="boundaries",
        colorbar_extend=delta_v_extend,
        smooth_regions=args.smooth_regions,
        panel_layout="vertical" if vertical_porkchop else "horizontal",
        departure_label_x_offset=-0.025 if args.case in {"mars", "mercury"} else 0.0,
    )

    diff = bspline_dv - fig3_dv
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
        [transfer_title(args.case)],
        plots_output_dir / f"comparison_fig3_vs_{method_key}_delta_v_difference.{args.figure_format}",
        r"$\Delta V\ \mathrm{difference}\ [\mathrm{km/s}]$",
        "coolwarm",
        -diff_limit,
        diff_limit,
        calendar_axes=True,
        smooth_regions=args.smooth_regions,
        departure_label_x_offset=-0.025 if args.case in {"mars", "mercury"} else 0.0,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = 100.0 * (bspline_dv - fig3_dv) / fig3_dv
    rel_error[~np.isfinite(fig3_dv) | ~np.isfinite(bspline_dv) | (fig3_dv == 0.0)] = np.nan
    finite_rel_error = rel_error[np.isfinite(rel_error)]
    if args.relative_error_limit is not None:
        rel_limit = abs(float(args.relative_error_limit))
    else:
        rel_limit = float(np.nanmax(np.abs(finite_rel_error))) if finite_rel_error.size else 1.0
    if rel_limit < 1e-12:
        rel_limit = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.8))
    plot_panel(
        [ax],
        dep_grid,
        tof_grid,
        [rel_error],
        [transfer_title(args.case)],
        plots_output_dir / f"comparison_fig3_vs_{method_key}_delta_v_relative_error.{args.figure_format}",
        r"$\Delta V\ \mathrm{relative\ difference}\ [\%]$",
        "coolwarm",
        -rel_limit,
        rel_limit,
        calendar_axes=True,
        smooth_regions=args.smooth_regions,
        departure_label_x_offset=-0.025 if args.case in {"mars", "mercury"} else 0.0,
    )

    points_by_method, pareto_method_order, pareto_method_labels = read_publication_pareto_points(
        source_dir,
        [str(spec) for spec in args.pareto_bspline_variants],
    )
    points_by_method = filter_pareto_departures(
        points_by_method, args.departure_min, args.departure_max_exclusive
    )
    if points_by_method:
        pareto_swap_axes = False if args.pareto_swap_axes is None else bool(args.pareto_swap_axes)
        if pareto_swap_axes:
            pareto_x_min = 0.1 if args.pareto_x_min is None else float(args.pareto_x_min)
            pareto_x_max = 10.0 if args.pareto_x_max is None else float(args.pareto_x_max)
            pareto_y_min = 18.0 if args.pareto_y_min is None else float(args.pareto_y_min)
            pareto_y_max = 30.0 if args.pareto_y_max is None else float(args.pareto_y_max)
        elif args.case == "mercury":
            pareto_x_min = 18.0 if args.pareto_x_min is None else float(args.pareto_x_min)
            pareto_x_max = 30.0 if args.pareto_x_max is None else float(args.pareto_x_max)
            pareto_y_min = 0.1 if args.pareto_y_min is None else float(args.pareto_y_min)
            pareto_y_max = 10.0 if args.pareto_y_max is None else float(args.pareto_y_max)
        elif args.case == "tempel1":
            pareto_x_min = 12.0 if args.pareto_x_min is None else float(args.pareto_x_min)
            pareto_x_max = 20.0 if args.pareto_x_max is None else float(args.pareto_x_max)
            pareto_y_min = 0.1 if args.pareto_y_min is None else float(args.pareto_y_min)
            pareto_y_max = 1.5 if args.pareto_y_max is None else float(args.pareto_y_max)
        elif args.case == "1989ml":
            pareto_x_min = 3.85 if args.pareto_x_min is None else float(args.pareto_x_min)
            pareto_x_max = 6.0 if args.pareto_x_max is None else float(args.pareto_x_max)
            pareto_y_min = 0.1 if args.pareto_y_min is None else float(args.pareto_y_min)
            pareto_y_max = 1.6 if args.pareto_y_max is None else float(args.pareto_y_max)
        else:
            pareto_x_min = 5.6 if args.pareto_x_min is None else float(args.pareto_x_min)
            pareto_x_max = 7.0 if args.pareto_x_max is None else float(args.pareto_x_max)
            pareto_y_min = 0.075 if args.pareto_y_min is None else float(args.pareto_y_min)
            pareto_y_max = 0.2 if args.pareto_y_max is None else float(args.pareto_y_max)
        plot_pareto_zoom(
            output_figure_path(plots_output_dir, args.case, f"comparisonpareto{suffix}", args.figure_format),
            points_by_method,
            transfer_title(args.case),
            pareto_x_min,
            pareto_x_max,
            pareto_y_min,
            pareto_y_max,
            swap_axes=pareto_swap_axes,
            method_order=[method for method in pareto_method_order if method != "gondelach_fig2"],
            method_labels=pareto_method_labels,
            auto_fit_x=args.pareto_x_min is None and args.pareto_x_max is None,
            auto_fit_y=args.pareto_y_min is None and args.pareto_y_max is None,
        )
        plot_pareto_delta_v_tof(
            output_figure_path(plots_output_dir, args.case, f"comparisonpareto_tof{suffix}", args.figure_format),
            points_by_method,
            transfer_title(args.case),
            method_order=[method for method in pareto_method_order if method != "gondelach_fig2"],
            method_labels=pareto_method_labels,
            log_delta_v=args.case == "1989ml",
            legend_loc="upper right" if args.case == "tempel1" else None,
        )

    best_profile_outputs: list[Path] = []
    if args.plot_best_profiles:
        best_profile_outputs = build_and_plot_best_profiles(source_dir, plots_output_dir, args)
    bspline_compare_outputs = build_and_plot_bspline_best_profile_comparison(
        source_dir, plots_output_dir, args
    )
    stacked_bspline_outputs = build_and_plot_stacked_bspline_best_profiles(
        source_dir, plots_output_dir, args
    )
    statistics_outputs: list[Path] = []
    if args.write_global_statistics:
        statistics_outputs = write_case_global_statistics(source_dir, plots_output_dir, args)
    timing_outputs = write_fair_computational_time_comparison(
        source_dir, plots_output_dir, args
    )

    print(f"wrote {output_figure_path(plots_output_dir, args.case, f'comparisonporkchop{suffix}', args.figure_format)}")
    print(f"wrote {plots_output_dir / f'comparison_fig3_vs_{method_key}_delta_v_difference.{args.figure_format}'}")
    print(f"wrote {plots_output_dir / f'comparison_fig3_vs_{method_key}_delta_v_relative_error.{args.figure_format}'}")
    if points_by_method:
        print(f"wrote {output_figure_path(plots_output_dir, args.case, f'comparisonpareto{suffix}', args.figure_format)}")
        print(f"wrote {output_figure_path(plots_output_dir, args.case, f'comparisonpareto_tof{suffix}', args.figure_format)}")
    for path in best_profile_outputs:
        print(f"wrote {path}")
    for path in bspline_compare_outputs:
        print(f"wrote {path}")
    for path in stacked_bspline_outputs:
        print(f"wrote {path}")
    for path in statistics_outputs:
        print(f"wrote {path}")
    for path in timing_outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

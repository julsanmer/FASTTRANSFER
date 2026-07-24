"""Validate saved cylindrical B-splines by independent forward propagation.

For each requested B-spline architecture, this script selects the lowest
Delta-V usable solution from its saved fine-grid control-point archive. It
reconstructs the nominal spline and its inverse-dynamics thrust history, then
integrates the Cartesian two-body equations from the SPICE Earth departure
state with that open-loop inertial thrust history.

No optimization is performed.
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
from scipy.integrate import solve_ivp
from scipy.interpolate import BSpline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_gondelach_fig2_fig3_bspline10 import (  # noqa: E402
    CASE_CONFIGS,
    configure_plot_fonts,
)
from experiments.reproduce_gondelach_fig2 import (  # noqa: E402
    AU_KM,
    DAY_S,
    configure_ephemeris,
    planet_state,
)
from optimizer.canonical_units import UT_YR, UV_AU_PER_YR  # noqa: E402
from optimizer.optimization_Bspline_freetf import (  # noqa: E402
    cylindrical_position_velocity_accel_np,
)


YEAR_DAYS = 365.25
YEAR_S = YEAR_DAYS * DAY_S
CANONICAL_VELOCITY_TO_KM_S = UV_AU_PER_YR * AU_KM / YEAR_S


@dataclass(frozen=True)
class Variant:
    n_ctrl: int
    degree: int
    run_id: str = ""

    @property
    def key(self) -> str:
        base = f"bspline_nctrl{self.n_ctrl}_deg{self.degree}"
        return f"{base}_{self.run_id}" if self.run_id else base

    @property
    def label(self) -> str:
        degree_name = {3: "Cubic", 5: "Quintic"}.get(self.degree, f"Degree {self.degree}")
        return f"{degree_name} B-spline ($n_c={self.n_ctrl}$)"


def parse_variant(text: str) -> Variant:
    match = re.fullmatch(r"(\d+):(\d+)(?::([0-9a-fA-F]{8}))?", str(text).strip())
    if not match:
        raise argparse.ArgumentTypeError(f"Expected n_ctrl:degree, got {text!r}")
    variant = Variant(int(match.group(1)), int(match.group(2)), str(match.group(3) or "").lower())
    if variant.n_ctrl <= variant.degree:
        raise argparse.ArgumentTypeError("n_ctrl must be greater than degree")
    return variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASE_CONFIGS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        type=parse_variant,
        default=[Variant(10, 3), Variant(10, 5), Variant(40, 3), Variant(40, 5)],
    )
    parser.add_argument("--samples", type=int, default=2001)
    parser.add_argument("--rtol", type=float, default=1e-11)
    parser.add_argument("--atol", type=float, default=1e-13)
    parser.add_argument("--steps-per-knot-span", type=int, default=20)
    parser.add_argument("--figure-format", choices=["pdf", "png"], default="pdf")
    parser.add_argument("--use-tex", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spice-meta-kernel", default=None)
    parser.add_argument("--spice-target-name", default=None)
    return parser.parse_args()


def archive_path(output_dir: Path, variant: Variant) -> Path:
    if variant.n_ctrl == 10 and variant.degree == 5 and not variant.run_id:
        baseline = output_dir / "bspline10_control_points.npz"
        if baseline.is_file():
            return baseline
    exact = output_dir / f"{variant.key}_control_points.npz"
    if exact.is_file() or variant.run_id:
        return exact
    matches = sorted(output_dir.glob(f"{variant.key}_[0-9a-f]" + "[0-9a-f]" * 7 + "_control_points.npz"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple saved runs match {variant.key}; specify n_ctrl:degree:run_id"
        )
    return exact


def scalar(data, key: str, default=None):
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.size != 1:
        return default
    return value.reshape(-1)[0].item()


def best_usable_index(data) -> int:
    usable = np.asarray(data["usable"], dtype=bool)
    delta_v_key = (
        "delta_v_reference_km_s"
        if "delta_v_reference_km_s" in data.files
        else "delta_v_km_s"
    )
    delta_v = np.asarray(data[delta_v_key], dtype=float)
    valid = usable & np.isfinite(delta_v)
    indices = np.flatnonzero(valid)
    if not indices.size:
        raise ValueError("Control-point archive contains no usable finite solution")
    return int(indices[np.argmin(delta_v[indices])])


def canonical_velocity(velocity_au_day: np.ndarray) -> np.ndarray:
    return np.asarray(velocity_au_day, dtype=float) * YEAR_DAYS / UV_AU_PER_YR


def evaluate_nominal(
    spline: BSpline,
    derivative_1: BSpline,
    derivative_2: BSpline,
    tau: float | np.ndarray,
    tf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau_array = np.atleast_1d(np.asarray(tau, dtype=float))
    q = np.asarray(spline(tau_array), dtype=float)
    qdot = np.asarray(derivative_1(tau_array), dtype=float) / tf
    qddot = np.asarray(derivative_2(tau_array), dtype=float) / (tf * tf)
    pos = np.empty_like(q)
    vel = np.empty_like(q)
    acc = np.empty_like(q)
    for index in range(len(tau_array)):
        pos[index], vel[index], acc[index] = cylindrical_position_velocity_accel_np(
            q[index], qdot[index], qddot[index]
        )
    return pos, vel, acc


def propagate_at_saved_control(
    control_points: np.ndarray,
    knots: np.ndarray,
    degree: int,
    tf: float,
    mu: float,
    initial_state: np.ndarray,
    sample_tau: np.ndarray,
    rtol: float,
    atol: float,
    steps_per_span: int,
) -> tuple[np.ndarray, int, str]:
    spline = BSpline(knots, control_points, degree, extrapolate=False)
    derivative_1 = spline.derivative(1)
    derivative_2 = spline.derivative(2)

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        tau = float(np.clip(time / tf, 0.0, 1.0))
        nominal_pos, _, nominal_acc = evaluate_nominal(
            spline, derivative_1, derivative_2, tau, tf
        )
        nominal_position = nominal_pos[0]
        radius_nominal = float(np.linalg.norm(nominal_position))
        thrust = nominal_acc[0] + mu * nominal_position / (radius_nominal**3)

        position = state[:3]
        radius = float(np.linalg.norm(position))
        acceleration = -mu * position / (radius**3) + thrust
        return np.concatenate([state[3:], acceleration])

    propagated = np.full((len(sample_tau), 6), np.nan, dtype=float)
    current_state = np.asarray(initial_state, dtype=float)
    nfev = 0
    messages: list[str] = []

    breaks = np.unique(np.clip(np.asarray(knots, dtype=float), 0.0, 1.0))
    breaks = breaks[(breaks >= 0.0) & (breaks <= 1.0)]
    if breaks[0] > 0.0:
        breaks = np.insert(breaks, 0, 0.0)
    if breaks[-1] < 1.0:
        breaks = np.append(breaks, 1.0)

    for segment_index, (left, right) in enumerate(zip(breaks[:-1], breaks[1:])):
        if right <= left:
            continue
        max_step = tf * (right - left) / max(int(steps_per_span), 1)
        solution = solve_ivp(
            rhs,
            (tf * left, tf * right),
            current_state,
            method="DOP853",
            dense_output=True,
            rtol=float(rtol),
            atol=float(atol),
            max_step=max_step,
        )
        nfev += int(solution.nfev)
        if not solution.success:
            raise RuntimeError(
                f"Forward propagation failed on knot span {left:g}-{right:g}: "
                f"{solution.message}"
            )
        messages.append(str(solution.message))
        current_state = solution.y[:, -1]

        tolerance = 1e-12
        indices = np.flatnonzero(
            (sample_tau >= left - tolerance) & (sample_tau <= right + tolerance)
        )
        if indices.size:
            query_tau = np.clip(sample_tau[indices], left, right)
            propagated[indices] = solution.sol(tf * query_tau).T

    if not np.all(np.isfinite(propagated)):
        missing = np.flatnonzero(~np.all(np.isfinite(propagated), axis=1))
        raise RuntimeError(f"Forward propagation did not sample indices {missing.tolist()}")

    return propagated, nfev, "; ".join(dict.fromkeys(messages))


def write_trajectory_csv(
    path: Path,
    tof_days: float,
    sample_tau: np.ndarray,
    nominal_pos: np.ndarray,
    nominal_vel: np.ndarray,
    propagated: np.ndarray,
) -> None:
    position_error_km = AU_KM * np.linalg.norm(propagated[:, :3] - nominal_pos, axis=1)
    velocity_error_m_s = (
        1000.0
        * CANONICAL_VELOCITY_TO_KM_S
        * np.linalg.norm(propagated[:, 3:] - nominal_vel, axis=1)
    )
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "tau",
                "time_days",
                "nominal_x_au",
                "nominal_y_au",
                "nominal_z_au",
                "propagated_x_au",
                "propagated_y_au",
                "propagated_z_au",
                "position_difference_km",
                "velocity_difference_m_s",
            ]
        )
        for index, tau in enumerate(sample_tau):
            writer.writerow(
                [
                    float(tau),
                    float(tau * tof_days),
                    *nominal_pos[index].tolist(),
                    *propagated[index, :3].tolist(),
                    float(position_error_km[index]),
                    float(velocity_error_m_s[index]),
                ]
            )


def plot_validation(
    path: Path,
    label: str,
    tof_days: float,
    sample_tau: np.ndarray,
    nominal_pos: np.ndarray,
    nominal_vel: np.ndarray,
    propagated: np.ndarray,
    target_pos: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    configure_plot_fonts()
    import matplotlib.pyplot as plt

    position_error_km = AU_KM * np.linalg.norm(propagated[:, :3] - nominal_pos, axis=1)
    velocity_error_m_s = (
        1000.0
        * CANONICAL_VELOCITY_TO_KM_S
        * np.linalg.norm(propagated[:, 3:] - nominal_vel, axis=1)
    )
    time_days = sample_tau * tof_days

    fig = plt.figure(figsize=(12.5, 5.6))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.08, 0.92),
        hspace=0.36,
        wspace=0.42,
    )
    ax_trajectory = fig.add_subplot(grid[:, 0], projection="3d")
    ax_position = fig.add_subplot(grid[0, 1])
    ax_velocity = fig.add_subplot(grid[1, 1])

    ax_trajectory.plot(
        nominal_pos[:, 0],
        nominal_pos[:, 1],
        nominal_pos[:, 2],
        color="tab:blue",
        linewidth=2.0,
        label="Saved B-spline",
    )
    ax_trajectory.plot(
        propagated[:, 0],
        propagated[:, 1],
        propagated[:, 2],
        color="tab:red",
        linestyle="--",
        linewidth=1.6,
        label="Forward propagation",
    )
    ax_trajectory.scatter(
        [target_pos[0]],
        [target_pos[1]],
        [target_pos[2]],
        marker="x",
        color="black",
        s=55,
        linewidth=1.5,
        label="SPICE target at arrival",
    )
    ax_trajectory.set_xlabel(r"$x$ [AU]")
    ax_trajectory.set_ylabel(r"$y$ [AU]")
    ax_trajectory.set_zlabel(r"$z$ [AU]")
    ax_trajectory.set_title(label)
    ax_trajectory.legend(fontsize=12, loc="upper left", bbox_to_anchor=(0.0, 0.98))
    ax_trajectory.grid(True, alpha=0.25)

    positive_position = np.maximum(position_error_km, np.finfo(float).tiny)
    positive_velocity = np.maximum(velocity_error_m_s, np.finfo(float).tiny)
    ax_position.semilogy(time_days, positive_position, color="tab:blue", linewidth=1.8)
    ax_position.set_ylabel("Position difference [km]")
    ax_position.grid(True, which="both", alpha=0.25)
    ax_velocity.semilogy(time_days, positive_velocity, color="tab:red", linewidth=1.8)
    ax_velocity.set_xlabel("Time [days]")
    ax_velocity.set_ylabel("Velocity difference [m/s]")
    ax_velocity.grid(True, which="both", alpha=0.25)

    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.12, top=0.92)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_variant(
    output_dir: Path,
    variant: Variant,
    case_name: str,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[Path]]:
    path = archive_path(output_dir, variant)
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved control-point archive: {path}")

    with np.load(path) as data:
        index = best_usable_index(data)
        control_points = np.asarray(data["control_points"][index], dtype=float)
        if control_points.ndim != 2 or control_points.shape[1] != 3:
            raise ValueError(
                f"Saved control points have invalid shape {control_points.shape} at index {index}"
            )
        if not np.all(np.isfinite(control_points)):
            raise ValueError(f"Saved control points are non-finite at index {index}")
        knots = np.asarray(data["knots"], dtype=float)
        degree = int(scalar(data, "degree", variant.degree))
        tf = float(np.asarray(data["t_transfer_canonical"], dtype=float)[index])
        mu = float(np.asarray(data["mu"], dtype=float)[index])
        departure = float(np.asarray(data["departure_mjd2000"], dtype=float)[index])
        tof_days = float(np.asarray(data["tof_days"], dtype=float)[index])
        n_rev = int(np.asarray(data["N"], dtype=int)[index])
        delta_v = float(np.asarray(data["delta_v_reference_km_s"], dtype=float)[index])
        source_success = bool(np.asarray(data["success"], dtype=bool)[index])
        ephemeris_source = str(scalar(data, "ephemeris_source", "kepler"))
        saved_meta_kernel = str(scalar(data, "spice_meta_kernel", ""))
        saved_target_name = str(scalar(data, "spice_target_name", ""))
        target = str(scalar(data, "target", CASE_CONFIGS[case_name]["target"]))

    meta_kernel = str(args.spice_meta_kernel or saved_meta_kernel)
    target_name = str(args.spice_target_name or saved_target_name)
    configure_ephemeris(ephemeris_source, meta_kernel or None, target_name or None)

    earth_pos, earth_vel_au_day = planet_state("earth", departure)
    target_pos, target_vel_au_day = planet_state(target, departure + tof_days)
    earth_vel = canonical_velocity(earth_vel_au_day)
    target_vel = canonical_velocity(target_vel_au_day)

    spline = BSpline(knots, control_points, degree, extrapolate=False)
    derivative_1 = spline.derivative(1)
    derivative_2 = spline.derivative(2)
    sample_tau = np.linspace(0.0, 1.0, max(int(args.samples), 2))
    nominal_pos, nominal_vel, _ = evaluate_nominal(
        spline, derivative_1, derivative_2, sample_tau, tf
    )
    initial_state = np.concatenate([earth_pos, earth_vel])
    propagated, nfev, solver_message = propagate_at_saved_control(
        control_points,
        knots,
        degree,
        tf,
        mu,
        initial_state,
        sample_tau,
        float(args.rtol),
        float(args.atol),
        int(args.steps_per_knot_span),
    )

    position_difference_km = AU_KM * np.linalg.norm(
        propagated[:, :3] - nominal_pos, axis=1
    )
    velocity_difference_m_s = (
        1000.0
        * CANONICAL_VELOCITY_TO_KM_S
        * np.linalg.norm(propagated[:, 3:] - nominal_vel, axis=1)
    )
    shaped_arrival_position_error_km = AU_KM * float(
        np.linalg.norm(nominal_pos[-1] - target_pos)
    )
    shaped_arrival_velocity_error_m_s = (
        1000.0
        * CANONICAL_VELOCITY_TO_KM_S
        * float(np.linalg.norm(nominal_vel[-1] - target_vel))
    )
    propagated_arrival_position_error_km = AU_KM * float(
        np.linalg.norm(propagated[-1, :3] - target_pos)
    )
    propagated_arrival_velocity_error_m_s = (
        1000.0
        * CANONICAL_VELOCITY_TO_KM_S
        * float(np.linalg.norm(propagated[-1, 3:] - target_vel))
    )

    trajectory_path = output_dir / f"forward_propagation_{variant.key}.csv"
    figure_path = output_dir / (
        f"forward_propagation_{variant.key}_{CASE_CONFIGS[case_name]['display'].replace(' ', '')}."
        f"{args.figure_format}"
    )
    write_trajectory_csv(
        trajectory_path,
        tof_days,
        sample_tau,
        nominal_pos,
        nominal_vel,
        propagated,
    )
    plot_validation(
        figure_path,
        variant.label,
        tof_days,
        sample_tau,
        nominal_pos,
        nominal_vel,
        propagated,
        target_pos,
    )

    row: dict[str, object] = {
        "case": case_name,
        "variant": variant.key,
        "departure_mjd2000": departure,
        "tof_days": tof_days,
        "N": n_rev,
        "delta_v_km_s": delta_v,
        "source_success": source_success,
        "solver_method": "DOP853",
        "solver_rtol": float(args.rtol),
        "solver_atol": float(args.atol),
        "solver_nfev": nfev,
        "solver_message": solver_message,
        "initial_position_error_km": AU_KM
        * float(np.linalg.norm(nominal_pos[0] - earth_pos)),
        "initial_velocity_error_m_s": 1000.0
        * CANONICAL_VELOCITY_TO_KM_S
        * float(np.linalg.norm(nominal_vel[0] - earth_vel)),
        "shaped_arrival_position_error_km": shaped_arrival_position_error_km,
        "shaped_arrival_velocity_error_m_s": shaped_arrival_velocity_error_m_s,
        "propagated_arrival_position_error_km": propagated_arrival_position_error_km,
        "propagated_arrival_velocity_error_m_s": propagated_arrival_velocity_error_m_s,
        "maximum_propagated_vs_shaped_position_error_km": float(
            np.max(position_difference_km)
        ),
        "maximum_propagated_vs_shaped_velocity_error_m_s": float(
            np.max(velocity_difference_m_s)
        ),
        "reaches_target": bool(
            propagated_arrival_position_error_km < 1.0
            and propagated_arrival_velocity_error_m_s < 0.01
        ),
        "control_interpretation": "saved open-loop inertial inverse-dynamics history",
        "control_archive": str(path),
    }
    return row, [trajectory_path, figure_path]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    os.environ["FASTTRANSFER_USE_LATEX"] = "1" if args.use_tex else "0"
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    (output_dir / ".matplotlib").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    outputs: list[Path] = []
    for variant in args.variants:
        row, variant_outputs = validate_variant(output_dir, variant, args.case, args)
        rows.append(row)
        outputs.extend(variant_outputs)
        print(
            f"{variant.key}: arrival position error = "
            f"{float(row['propagated_arrival_position_error_km']):.6g} km, "
            f"velocity error = "
            f"{float(row['propagated_arrival_velocity_error_m_s']):.6g} m/s"
        )

    summary_path = output_dir / "bspline_forward_propagation_validation.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary_path}")
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

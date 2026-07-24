"""
Free-time low-thrust optimization in MEE with Radau direct collocation.

The state is x = [p, f, g, h, k, L] and the control is RTN acceleration
u = [uR, uT, uN].  A Cartesian B-spline solution can be supplied as an initial
guess; its inertial acceleration profile is converted to RTN controls.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import casadi as ca
import numpy as np

from .canonical_units import (
    MU_CANONICAL,
    accel_from_canonical,
    accel_to_canonical,
    dv_from_canonical,
    energy_from_canonical,
    time_from_canonical,
    time_to_canonical,
)
from .orbit_utils import get_kepler_substeps, kepler_coast_sym
from .targets import DEFAULT_MEE0, DEFAULT_MEE_TARGET_EPOCH, dionysus_target_mee, mars_target_mee, target_for_name
from utils.utils import kepler_coast_np, mee2rv, rv2mee


DEFAULT_MU = MU_CANONICAL
DEFAULT_TF_MIN_FACTOR = 0.5
DEFAULT_TF_MAX_FACTOR = 1.5
DIRECT_TERMINAL_ANGLE_MODE = "sincos"


@dataclass
class RadauCoefficients:
    degree: int
    tau_root: np.ndarray
    C: np.ndarray
    D: np.ndarray
    B: np.ndarray


def target_mee_at_time(t: float, mee_target_epoch: np.ndarray, mu: float) -> np.ndarray:
    return kepler_coast_np(
        np.asarray(mee_target_epoch, dtype=float),
        float(t),
        mu,
        n_iter=get_kepler_substeps(),
    )


def target_period(mee_target_epoch: np.ndarray, mu: float) -> float:
    p, f, g = np.asarray(mee_target_epoch, dtype=float)[:3]
    a = p / (1.0 - f * f - g * g)
    return float(2.0 * np.pi * np.sqrt(a**3 / mu))


def mee_semimajor_axis(mee: np.ndarray) -> float:
    p, f, g = np.asarray(mee, dtype=float)[:3]
    return float(p / (1.0 - f * f - g * g))


def mean_a_transfer_period(mee0: np.ndarray, mee_target_epoch: np.ndarray, mu: float) -> float:
    a0 = mee_semimajor_axis(mee0)
    af = mee_semimajor_axis(mee_target_epoch)
    a_mean = 0.5 * (a0 + af)
    return float(2.0 * np.pi * np.sqrt(a_mean**3 / mu))


def infer_longitude_branch_from_solution(state_nodes: np.ndarray, target_mee: np.ndarray) -> int:
    states = np.asarray(state_nodes, dtype=float)
    if states.ndim != 2:
        raise ValueError("state_nodes must be a 2D array")
    l_final = float(states[5, -1] if states.shape[0] == 6 else states[-1, 5])
    target_l = float(np.asarray(target_mee, dtype=float).reshape(6)[5])
    return int(np.rint((l_final - target_l) / (2.0 * np.pi)))


def fit_origin_kepler_cartesian_seed(
    mats,
    mee0: np.ndarray,
    mee_target_epoch: np.ndarray,
    tf: float,
    mu: float,
) -> np.ndarray:
    """Fit absolute B-spline control points to a Kepler coast from the origin.

    The fitted curve follows the uncontrolled origin orbit in the interior, but
    enforces the transfer boundary conditions at t=0 and t=tf.  It is intended
    as a robust second B-spline initialization when the usual transfer seed
    fails.
    """
    tf = float(tf)
    tau_fit = np.asarray(mats.tau_fine, dtype=float)
    mee_fit = np.vstack(
        [kepler_coast_np(mee0, float(tau) * tf, mu, n_iter=get_kepler_substeps()) for tau in tau_fit]
    )
    pos_fit = np.vstack([mee2rv(mee_fit[idx], mu)[0] for idx in range(len(tau_fit))])

    pos0, vel0 = mee2rv(mee0, mu)
    mee_target_final = target_mee_at_time(tf, mee_target_epoch, mu)
    posf, velf = mee2rv(mee_target_final, mu)

    a_eq = np.vstack(
        [
            mats.b0_start,
            mats.b1_start / tf,
            mats.b0_end,
            mats.b1_end / tf,
        ]
    )
    b_eq = np.vstack(
        [
            pos0.reshape(1, 3),
            vel0.reshape(1, 3),
            posf.reshape(1, 3),
            velf.reshape(1, 3),
        ]
    )

    ata = mats.b0_fine.T @ mats.b0_fine
    rhs_top = mats.b0_fine.T @ pos_fit
    lhs = np.block(
        [
            [2.0 * ata, a_eq.T],
            [a_eq, np.zeros((a_eq.shape[0], a_eq.shape[0]))],
        ]
    )
    rhs = np.vstack([2.0 * rhs_top, b_eq])
    sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return sol[: mats.b0_fine.shape[1], :]


def mee_gauss_rhs_sym(x, u, mu):
    """Walker-style MEE dynamics under RTN acceleration [uR, uT, uN]."""
    p, f, g, h, k, L = x[0], x[1], x[2], x[3], x[4], x[5]
    uR, uT, uN = u[0], u[1], u[2]

    cosL = ca.cos(L)
    sinL = ca.sin(L)
    w = ca.fmax(1.0 + f * cosL + g * sinL, 1e-3)
    s2 = 1.0 + h * h + k * k
    p_safe = ca.fmax(p, 1e-3)
    sqp = ca.sqrt(p_safe)
    A = sqp / (ca.sqrt(mu) * w)

    dp = 2.0 * p_safe * A * uT
    df = A * (
        sinL * uR
        + ((w + 1.0) * cosL + f) / w * uT
        - g * (h * sinL - k * cosL) / w * uN
    )
    dg = A * (
        -cosL * uR
        + ((w + 1.0) * sinL + g) / w * uT
        + f * (h * sinL - k * cosL) / w * uN
    )
    dh = A * s2 / 2.0 * cosL * uN
    dk = A * s2 / 2.0 * sinL * uN
    dL = (ca.sqrt(mu) / (sqp * p_safe)) * w**2 + A * (h * sinL - k * cosL) * uN

    return ca.vertcat(dp, df, dg, dh, dk, dL)


def mee_gauss_rhs_np(x: np.ndarray, u: np.ndarray, mu: float) -> np.ndarray:
    """Numerical version of the MEE dynamics for initial-seed construction."""
    p, f, g, h, k, L = np.asarray(x, dtype=float).reshape(6)
    uR, uT, uN = np.asarray(u, dtype=float).reshape(3)

    cosL = np.cos(L)
    sinL = np.sin(L)
    w = max(1.0 + f * cosL + g * sinL, 1e-3)
    s2 = 1.0 + h * h + k * k
    p_safe = max(p, 1e-3)
    sqp = np.sqrt(p_safe)
    A = sqp / (np.sqrt(mu) * w)

    dp = 2.0 * p_safe * A * uT
    df = A * (
        sinL * uR
        + ((w + 1.0) * cosL + f) / w * uT
        - g * (h * sinL - k * cosL) / w * uN
    )
    dg = A * (
        -cosL * uR
        + ((w + 1.0) * sinL + g) / w * uT
        + f * (h * sinL - k * cosL) / w * uN
    )
    dh = A * s2 / 2.0 * cosL * uN
    dk = A * s2 / 2.0 * sinL * uN
    dL = (np.sqrt(mu) / (sqp * p_safe)) * w**2 + A * (h * sinL - k * cosL) * uN

    return np.array([dp, df, dg, dh, dk, dL], dtype=float)


def build_radau_coefficients(degree: int = 3) -> RadauCoefficients:
    if degree < 1:
        raise ValueError("degree must be >= 1")

    tau_root = np.array([0.0] + ca.collocation_points(int(degree), "radau"), dtype=float)
    C = np.zeros((degree + 1, degree + 1))
    D = np.zeros(degree + 1)
    B = np.zeros(degree + 1)

    for j in range(degree + 1):
        poly = np.poly1d([1.0])
        for r in range(degree + 1):
            if r != j:
                poly *= np.poly1d([1.0, -tau_root[r]]) / (tau_root[j] - tau_root[r])

        D[j] = poly(1.0)
        integral = np.polyint(poly)
        B[j] = integral(1.0) - integral(0.0)
        dpoly = np.polyder(poly)
        for r in range(degree + 1):
            C[j, r] = dpoly(tau_root[r])

    return RadauCoefficients(degree=int(degree), tau_root=tau_root, C=C, D=D, B=B)


def state_bounds(mee0: np.ndarray, mee_target: np.ndarray, margin: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    lo = np.minimum(np.asarray(mee0, dtype=float), np.asarray(mee_target, dtype=float))
    hi = np.maximum(np.asarray(mee0, dtype=float), np.asarray(mee_target, dtype=float))
    span = np.maximum(hi - lo, 1e-6)

    lb = lo - float(margin) * span
    ub = hi + float(margin) * span

    lb[0] = max(lb[0], 1e-3)
    lb[1] = max(lb[1], -0.99)
    ub[1] = min(ub[1], 0.99)
    lb[2] = max(lb[2], -0.99)
    ub[2] = min(ub[2], 0.99)
    lb[3] = max(lb[3], -10.0)
    ub[3] = min(ub[3], 10.0)
    lb[4] = max(lb[4], -10.0)
    ub[4] = min(ub[4], 10.0)
    lb[5] = -np.inf
    ub[5] = np.inf

    return lb, ub


def shifted_norm(u, eps: float):
    eps = float(eps)
    return ca.sqrt(ca.sumsqr(u) + eps**2) - eps


def _interp_rows(tau_grid: np.ndarray, values: np.ndarray, tau_query: np.ndarray) -> np.ndarray:
    tau_grid = np.asarray(tau_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    tau_query = np.asarray(tau_query, dtype=float)

    return np.vstack(
        [np.interp(tau_query, tau_grid, values[:, idx]) for idx in range(values.shape[1])]
    ).T


def _cartesian_accel_to_rtn(r: np.ndarray, v: np.ndarray, u_cart: np.ndarray) -> np.ndarray:
    rhat = r / max(np.linalg.norm(r), 1e-12)
    hvec = np.cross(r, v)
    hhat = hvec / max(np.linalg.norm(hvec), 1e-12)
    that = np.cross(hhat, rhat)

    return np.array([np.dot(u_cart, rhat), np.dot(u_cart, that), np.dot(u_cart, hhat)])


def _align_seed_longitude(mee: np.ndarray, mee0: np.ndarray, mee_target_final: np.ndarray, tau: np.ndarray) -> np.ndarray:
    aligned = np.array(mee, dtype=float, copy=True)
    aligned[:, 5] = np.unwrap(aligned[:, 5])
    aligned[:, 5] += float(mee0[5] - aligned[0, 5])
    target_l = float(mee_target_final[5])
    target_l += 2.0 * np.pi * np.rint((aligned[-1, 5] - target_l) / (2.0 * np.pi))
    final_l_error = float(target_l - aligned[-1, 5])
    aligned[:, 5] += tau * final_l_error
    aligned[0, 5] = float(mee0[5])
    aligned[-1, 5] = float(target_l)
    return aligned


def build_seed_from_bspline_result(
    bspline_result: dict,
    mesh_tau: np.ndarray,
    collocation_tau: np.ndarray,
    mee0: np.ndarray,
    mee_target_final: np.ndarray,
    mu: float,
) -> dict:
    """Convert a Cartesian B-spline profile into MEE states and RTN controls."""
    profile = bspline_result["profile_fine"]
    tau_src = np.asarray(profile.get("tau", profile["t"] / bspline_result["t_transfer"]), dtype=float)
    pos_src = np.asarray(profile["pos"], dtype=float)
    vel_src = np.asarray(profile["vel"], dtype=float)
    u_src = np.asarray(profile["u"], dtype=float)

    state_tau = np.unique(np.concatenate([mesh_tau, collocation_tau]))
    pos_state = _interp_rows(tau_src, pos_src, state_tau)
    vel_state = _interp_rows(tau_src, vel_src, state_tau)
    mee_state = np.vstack([rv2mee(pos_state[i], vel_state[i], mu) for i in range(len(state_tau))])
    mee_state = _align_seed_longitude(mee_state, mee0, mee_target_final, state_tau)

    pos_col = _interp_rows(tau_src, pos_src, collocation_tau)
    vel_col = _interp_rows(tau_src, vel_src, collocation_tau)
    u_col = _interp_rows(tau_src, u_src, collocation_tau)
    u_rtn = np.vstack(
        [_cartesian_accel_to_rtn(pos_col[i], vel_col[i], u_col[i]) for i in range(len(collocation_tau))]
    )

    return {
        "tau_state": state_tau,
        "mee_state": mee_state,
        "tau_control": collocation_tau,
        "u_rtn": u_rtn,
        "t_transfer": float(bspline_result["t_transfer"]),
    }


def build_default_seed(
    mesh_tau: np.ndarray,
    collocation_tau: np.ndarray,
    mee0: np.ndarray,
    mee_target_final: np.ndarray,
) -> dict:
    state_tau = np.unique(np.concatenate([mesh_tau, collocation_tau]))
    s = state_tau[:, None]
    mee_state = (1.0 - s) * mee0[None, :] + s * mee_target_final[None, :]
    mee_state[:, 5] = mee0[5] + state_tau * (mee_target_final[5] - mee0[5])

    return {
        "tau_state": state_tau,
        "mee_state": mee_state,
        "tau_control": collocation_tau,
        "u_rtn": np.zeros((len(collocation_tau), 3)),
        "t_transfer": None,
    }


def build_kepler_origin_seed(
    mesh_tau: np.ndarray,
    collocation_tau: np.ndarray,
    mee0: np.ndarray,
    mee_target_final: np.ndarray,
    tf_guess: float,
    mu: float,
) -> dict:
    state_tau = np.unique(np.concatenate([mesh_tau, collocation_tau]))
    mee_state = np.vstack(
        [target_mee_at_time(float(tau) * float(tf_guess), mee0, mu) for tau in state_tau]
    )
    mee_state = _align_seed_longitude(mee_state, mee0, mee_target_final, state_tau)

    return {
        "tau_state": state_tau,
        "mee_state": mee_state,
        "tau_control": collocation_tau,
        "u_rtn": np.zeros((len(collocation_tau), 3)),
        "t_transfer": float(tf_guess),
    }


def _quintic_blend(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=float)
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    ds_dtau = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
    return s, ds_dtau


def _mee_inverse_dynamics_controls(
    mee_state: np.ndarray,
    dmee_dtau: np.ndarray,
    tf_guess: float,
    mu: float,
) -> np.ndarray:
    controls = []
    tf_safe = max(float(tf_guess), 1e-9)
    basis = np.eye(3)

    for x, dx_dtau in zip(np.asarray(mee_state, dtype=float), np.asarray(dmee_dtau, dtype=float)):
        desired = np.asarray(dx_dtau, dtype=float) / tf_safe
        drift = mee_gauss_rhs_np(x, np.zeros(3), mu)
        control_matrix = np.column_stack(
            [mee_gauss_rhs_np(x, basis[:, idx], mu) - drift for idx in range(3)]
        )
        residual = desired - drift
        try:
            control, *_ = np.linalg.lstsq(control_matrix, residual, rcond=1e-10)
        except np.linalg.LinAlgError:
            control = np.zeros(3)
        if not np.all(np.isfinite(control)):
            control = np.zeros(3)
        controls.append(control)

    return np.asarray(controls, dtype=float)


def build_mee_inverse_dynamics_seed(
    mesh_tau: np.ndarray,
    collocation_tau: np.ndarray,
    mee0: np.ndarray,
    mee_target_final: np.ndarray,
    tf_guess: float,
    mu: float,
) -> dict:
    """Smooth MEE seed with RTN controls from least-squares inverse dynamics."""
    state_tau = np.unique(np.concatenate([mesh_tau, collocation_tau]))
    state_s, _ = _quintic_blend(state_tau)
    delta = np.asarray(mee_target_final, dtype=float).reshape(6) - np.asarray(mee0, dtype=float).reshape(6)
    mee_state = np.asarray(mee0, dtype=float).reshape(1, 6) + state_s[:, None] * delta.reshape(1, 6)
    mee_state[0, :] = np.asarray(mee0, dtype=float).reshape(6)
    mee_state[-1, :] = np.asarray(mee_target_final, dtype=float).reshape(6)

    control_s, control_ds = _quintic_blend(collocation_tau)
    mee_control_state = np.asarray(mee0, dtype=float).reshape(1, 6) + control_s[:, None] * delta.reshape(1, 6)
    dmee_dtau = control_ds[:, None] * delta.reshape(1, 6)
    u_rtn = _mee_inverse_dynamics_controls(mee_control_state, dmee_dtau, tf_guess, mu)

    return {
        "tau_state": state_tau,
        "mee_state": mee_state,
        "tau_control": collocation_tau,
        "u_rtn": u_rtn,
        "t_transfer": float(tf_guess),
    }


def _seed_state(seed: dict, tau: float) -> np.ndarray:
    return _interp_rows(seed["tau_state"], seed["mee_state"], np.array([tau], dtype=float))[0]


def _seed_control(seed: dict, tau: float) -> np.ndarray:
    return _interp_rows(seed["tau_control"], seed["u_rtn"], np.array([tau], dtype=float))[0]


def solve_mee_radau_collocation(
    tf_guess: float,
    tf_min: float | None = None,
    tf_max: float | None = None,
    mee0: np.ndarray | None = None,
    mee_target_epoch: np.ndarray | None = None,
    mu: float = DEFAULT_MU,
    n_intervals: int = 30,
    degree: int = 3,
    objective: str = "dv",
    u_max: float | None = None,
    control_bound: float | None = None,
    state_bound_margin: float = 3.0,
    time_weight: float = 0.0,
    dv_eps: float = 1e-6,
    smoothness_weight: float = 0.0,
    bspline_seed: dict | None = None,
    fixed_tf: bool = False,
    max_iter: int = 2000,
    print_level: int = 5,
    nlp_scaling_method: str = "gradient-based",
    terminal_angle_mode: str = "unwrapped",
    longitude_branch: int = 0,
    seed_source: str = "linear_mee",
) -> dict:
    if mee0 is None:
        mee0 = DEFAULT_MEE0
    if mee_target_epoch is None:
        mee_target_epoch = DEFAULT_MEE_TARGET_EPOCH

    mee0 = np.asarray(mee0, dtype=float).reshape(6)
    mee_target_epoch = np.asarray(mee_target_epoch, dtype=float).reshape(6)
    tf_guess = float(tf_guess)

    if tf_min is None:
        tf_min = max(1e-3, 0.5 * tf_guess)
    if tf_max is None:
        tf_max = max(float(tf_min) + 1e-3, 1.5 * tf_guess)
    tf_min = float(tf_min)
    tf_max = float(tf_max)

    if tf_min <= 0.0:
        raise ValueError("tf_min must be positive")
    if tf_max <= tf_min:
        raise ValueError("tf_max must be greater than tf_min")
    if not (tf_min <= tf_guess <= tf_max):
        raise ValueError("tf_guess must be inside [tf_min, tf_max]")
    if n_intervals < 1:
        raise ValueError("n_intervals must be >= 1")
    if objective not in {"energy", "dv"}:
        raise ValueError("objective must be 'energy' or 'dv'")
    if terminal_angle_mode not in {"unwrapped", "sincos", "branch"}:
        raise ValueError("terminal_angle_mode must be 'unwrapped', 'sincos', or 'branch'")
    longitude_branch = int(longitude_branch)

    coeff = build_radau_coefficients(degree)
    mesh_tau = np.linspace(0.0, 1.0, n_intervals + 1)
    collocation_tau = np.array(
        [
            (k + coeff.tau_root[j]) / n_intervals
            for k in range(n_intervals)
            for j in range(1, degree + 1)
        ],
        dtype=float,
    )

    mee_target_guess = target_mee_at_time(tf_guess, mee_target_epoch, mu)
    if terminal_angle_mode == "branch":
        mee_target_guess = np.array(mee_target_guess, dtype=float, copy=True)
        mee_target_guess[5] += 2.0 * np.pi * longitude_branch
    normalized_seed_source = str(seed_source).replace("-", "_")
    if bspline_seed is not None:
        seed = build_seed_from_bspline_result(bspline_seed, mesh_tau, collocation_tau, mee0, mee_target_guess, mu)
        effective_seed_source = "bspline"
    elif normalized_seed_source == "kepler":
        seed = build_kepler_origin_seed(mesh_tau, collocation_tau, mee0, mee_target_guess, tf_guess, mu)
        effective_seed_source = "kepler"
    elif normalized_seed_source in {"mee_id", "mee_inverse_dynamics"}:
        seed = build_mee_inverse_dynamics_seed(mesh_tau, collocation_tau, mee0, mee_target_guess, tf_guess, mu)
        effective_seed_source = "mee_inverse_dynamics"
    else:
        seed = build_default_seed(mesh_tau, collocation_tau, mee0, mee_target_guess)
        effective_seed_source = "linear_mee"

    if seed["t_transfer"] is not None:
        tf_init = float(np.clip(seed["t_transfer"], tf_min, tf_max))
    else:
        tf_init = tf_guess

    seed_u_peak = float(np.max(np.linalg.norm(seed["u_rtn"], axis=1))) if seed["u_rtn"].size else 0.0
    if control_bound is None:
        control_bound = max(2.0 * seed_u_peak, float(u_max or 0.0) * 1.5, 1e-3)
    control_bound = float(control_bound)

    lb, ub = state_bounds(mee0, mee_target_guess, margin=state_bound_margin)

    opti = ca.Opti()
    X = opti.variable(6, n_intervals + 1)
    Xc = [opti.variable(6, degree) for _ in range(n_intervals)]
    Uc = [opti.variable(3, degree) for _ in range(n_intervals)]
    tf_fixed = float(tf_guess)
    tf = tf_fixed if fixed_tf else opti.variable()

    if not fixed_tf:
        opti.subject_to(opti.bounded(tf_min, tf, tf_max))
    initial_eq = X[:, 0] == ca.DM(mee0)
    opti.subject_to(initial_eq)
    collocation_eqs: list[list[ca.MX]] = []
    continuity_eqs: list[ca.MX] = []

    for row in range(5):
        opti.subject_to(opti.bounded(lb[row], X[row, :], ub[row]))
        for k in range(n_intervals):
            opti.subject_to(opti.bounded(lb[row], Xc[k][row, :], ub[row]))

    for k in range(n_intervals):
        opti.subject_to(opti.bounded(-control_bound, Uc[k], control_bound))
        if u_max is not None:
            for j in range(degree):
                opti.subject_to(ca.sumsqr(Uc[k][:, j]) <= float(u_max) ** 2)

    h_step = tf / n_intervals
    cost = ca.MX(0)
    u_values = []

    for k in range(n_intervals):
        x_all = [X[:, k]] + [Xc[k][:, j] for j in range(degree)]
        collocation_eqs_k = []

        for j in range(1, degree + 1):
            xp = ca.MX.zeros(6, 1)
            for r in range(degree + 1):
                xp += coeff.C[r, j] * x_all[r]

            u_j = Uc[k][:, j - 1]
            f_j = mee_gauss_rhs_sym(Xc[k][:, j - 1], u_j, mu)
            defect_eq = h_step * f_j == xp
            opti.subject_to(defect_eq)
            collocation_eqs_k.append(defect_eq)
            u_values.append(u_j)

            if objective == "energy":
                cost += h_step * coeff.B[j] * ca.sumsqr(u_j)
            else:
                cost += h_step * coeff.B[j] * shifted_norm(u_j, dv_eps)
        collocation_eqs.append(collocation_eqs_k)

        x_end = ca.MX.zeros(6, 1)
        for r in range(degree + 1):
            x_end += coeff.D[r] * x_all[r]
        continuity_eq = X[:, k + 1] == x_end
        opti.subject_to(continuity_eq)
        continuity_eqs.append(continuity_eq)

    target_epoch_dm = ca.DM(mee_target_epoch.reshape(6, 1))
    target_final = kepler_coast_sym(target_epoch_dm, tf, mu)
    if terminal_angle_mode == "branch":
        target_final = ca.vertcat(
            target_final[0],
            target_final[1],
            target_final[2],
            target_final[3],
            target_final[4],
            target_final[5] + 2.0 * np.pi * longitude_branch,
        )
    terminal_eq_orbital = X[0:5, -1] == target_final[0:5]
    opti.subject_to(terminal_eq_orbital)
    if terminal_angle_mode == "sincos":
        terminal_delta_l = X[5, -1] - target_final[5]
        terminal_eq_longitude = ca.sin(terminal_delta_l) == 0.0
        terminal_ineq_longitude = ca.cos(terminal_delta_l) >= 0.0
        opti.subject_to(terminal_eq_longitude)
        opti.subject_to(terminal_ineq_longitude)
    else:
        terminal_eq_longitude = X[5, -1] == target_final[5]
        terminal_ineq_longitude = None
        opti.subject_to(terminal_eq_longitude)

    if smoothness_weight > 0.0 and len(u_values) > 1:
        smooth_cost = ca.MX(0)
        for idx in range(1, len(u_values)):
            smooth_cost += ca.sumsqr(u_values[idx] - u_values[idx - 1])
        cost += float(smoothness_weight) * smooth_cost / (len(u_values) - 1)

    if time_weight > 0.0:
        cost += float(time_weight) * (tf / tf_guess)

    opti.minimize(cost)

    if not fixed_tf:
        opti.set_initial(tf, tf_init)
    for k, tau_k in enumerate(mesh_tau):
        opti.set_initial(X[:, k], _seed_state(seed, tau_k))
    for k in range(n_intervals):
        for j in range(1, degree + 1):
            tau_kj = (k + coeff.tau_root[j]) / n_intervals
            opti.set_initial(Xc[k][:, j - 1], _seed_state(seed, tau_kj))
            opti.set_initial(Uc[k][:, j - 1], _seed_control(seed, tau_kj))

    p_opts = {
        "expand": True,
        "print_time": bool(print_level > 0),
        "ipopt": {
            "max_iter": int(max_iter),
            "tol": 1e-8,
            "constr_viol_tol": 1e-8,
            "acceptable_tol": 1e-6,
            "acceptable_iter": 10,
            "mu_strategy": "adaptive",
            "nlp_scaling_method": str(nlp_scaling_method),
            "print_level": int(print_level),
        },
    }
    opti.solver("ipopt", p_opts)

    try:
        sol = opti.solve()
        success = True
        message = "Solve_Succeeded"
        tf_opt = tf_fixed if fixed_tf else float(sol.value(tf))
        x_nodes = np.array(sol.value(X), dtype=float)
        x_cols = np.stack([np.array(sol.value(Xc[k]), dtype=float) for k in range(n_intervals)], axis=0)
        u_cols = np.stack([np.array(sol.value(Uc[k]), dtype=float) for k in range(n_intervals)], axis=0)
        obj_val = float(sol.value(cost))
        value_source = sol
    except RuntimeError as exc:
        success = False
        message = str(exc).splitlines()[-1]
        tf_opt = tf_fixed if fixed_tf else float(opti.debug.value(tf))
        x_nodes = np.array(opti.debug.value(X), dtype=float)
        x_cols = np.stack([np.array(opti.debug.value(Xc[k]), dtype=float) for k in range(n_intervals)], axis=0)
        u_cols = np.stack([np.array(opti.debug.value(Uc[k]), dtype=float) for k in range(n_intervals)], axis=0)
        obj_val = float(opti.debug.value(cost))
        value_source = opti.debug

    def _dual_value(eq, shape: tuple[int, ...]) -> np.ndarray:
        try:
            return np.array(value_source.value(opti.dual(eq)), dtype=float).reshape(shape)
        except Exception:
            return np.full(shape, np.nan)

    dual_initial = _dual_value(initial_eq, (6,))
    dual_collocation = np.stack(
        [
            np.stack([_dual_value(collocation_eqs[k][j], (6,)) for j in range(degree)], axis=0)
            for k in range(n_intervals)
        ],
        axis=0,
    )
    dual_continuity = np.stack([_dual_value(continuity_eqs[k], (6,)) for k in range(n_intervals)], axis=0)
    dual_terminal = np.concatenate(
        [
            _dual_value(terminal_eq_orbital, (5,)),
            _dual_value(terminal_eq_longitude, (1,)),
        ]
    )

    u_flat = u_cols.transpose(0, 2, 1).reshape(n_intervals * degree, 3)
    u_norm = np.linalg.norm(u_flat, axis=1)
    h_opt = tf_opt / n_intervals
    weights = np.tile(coeff.B[1:], n_intervals) * h_opt
    dv_quad = float(np.sum(weights * u_norm))
    energy_quad = float(np.sum(weights * u_norm**2))

    target_final_np = target_mee_at_time(tf_opt, mee_target_epoch, mu)
    direct_longitude_branch = infer_longitude_branch_from_solution(x_nodes, target_final_np)
    if terminal_angle_mode == "branch":
        target_final_np = np.array(target_final_np, dtype=float, copy=True)
        target_final_np[5] += 2.0 * np.pi * longitude_branch
    endpoint = x_nodes[:, -1] - target_final_np
    endpoint[5] = np.arctan2(np.sin(endpoint[5]), np.cos(endpoint[5]))

    return {
        "success": success,
        "message": message,
        "objective": obj_val,
        "objective_type": objective,
        "state_nodes": x_nodes,
        "state_collocation": x_cols,
        "control_collocation": u_cols,
        "mesh_tau": mesh_tau,
        "collocation_tau": collocation_tau,
        "radau_tau_root": coeff.tau_root,
        "radau_weights": coeff.B,
        "radau_C": coeff.C,
        "radau_D": coeff.D,
        "t_transfer": tf_opt,
        "t_transfer_guess": tf_guess,
        "tf_min": tf_min,
        "tf_max": tf_max,
        "fixed_tf": bool(fixed_tf),
        "target_mee": target_final_np,
        "mee0": mee0,
        "mee_target_epoch": mee_target_epoch,
        "mu": float(mu),
        "n_intervals": int(n_intervals),
        "degree": int(degree),
        "u_max": u_max,
        "control_bound": control_bound,
        "max_u": float(np.max(u_norm)),
        "dv_quad": dv_quad,
        "energy_quad": energy_quad,
        "endpoint_error": endpoint,
        "endpoint_error_norm": float(np.linalg.norm(endpoint)),
        "dual_initial": dual_initial,
        "dual_collocation_defects": dual_collocation,
        "dual_continuity": dual_continuity,
        "dual_terminal": dual_terminal,
        "dual_extracted": bool(np.all(np.isfinite(dual_collocation))),
        "dv_eps": float(dv_eps),
        "smoothness_weight": float(smoothness_weight),
        "time_weight": float(time_weight),
        "nlp_scaling_method": str(nlp_scaling_method),
        "terminal_angle_mode": terminal_angle_mode,
        "longitude_branch": longitude_branch,
        "inferred_longitude_branch": direct_longitude_branch,
        "seed_source": effective_seed_source,
    }


def solve_with_internal_bspline_seed(
    tf_guess: float,
    tf_min: float | None = None,
    tf_max: float | None = None,
    mee0: np.ndarray | None = None,
    mee_target_epoch: np.ndarray | None = None,
    mu: float = DEFAULT_MU,
    bspline_options: dict | None = None,
    **radau_kwargs,
) -> dict:
    """Run the Cartesian B-spline solver first, then use it as Radau seed."""
    from .helpers_Bspline import build_bspline_matrices
    from .optimization_Bspline_freetf import solve_free_tf_cartesian_bspline

    options = dict(bspline_options or {})
    if "correction_bound" in options and "R_bound" not in options:
        options["R_bound"] = options.pop("correction_bound")

    bspline_first = solve_free_tf_cartesian_bspline(
        tf_guess,
        tf_min=tf_min,
        tf_max=tf_max,
        mee0=mee0,
        mee_target_epoch=mee_target_epoch,
        mu=mu,
        **options,
    )
    bspline_seed = bspline_first

    if not bool(bspline_first.get("success", False)):
        tf_retry = float(bspline_first.get("t_transfer", tf_guess))
        if not np.isfinite(tf_retry):
            tf_retry = float(tf_guess)
        if tf_min is not None or tf_max is not None:
            lo = max(1e-3, float(tf_min if tf_min is not None else 0.25 * tf_guess))
            hi = max(lo + 1e-3, float(tf_max if tf_max is not None else 2.0 * tf_guess))
            tf_retry = float(np.clip(tf_retry, lo, hi))

        n_ctrl = int(options.get("n_ctrl", 24))
        degree = int(options.get("degree", 3))
        n_fine = int(options.get("n_fine", 800))
        mats = build_bspline_matrices(n_ctrl, degree=degree, n_fine=n_fine)
        mee0_seed = DEFAULT_MEE0 if mee0 is None else np.asarray(mee0, dtype=float)
        mee_target_seed = (
            DEFAULT_MEE_TARGET_EPOCH if mee_target_epoch is None else np.asarray(mee_target_epoch, dtype=float)
        )
        kepler_control_points = fit_origin_kepler_cartesian_seed(
            mats,
            mee0_seed,
            mee_target_seed,
            tf_retry,
            mu,
        )
        retry_options = dict(options)
        retry_options["initial_control_points"] = kepler_control_points
        retry_options["initial_tf"] = tf_retry
        bspline_seed = solve_free_tf_cartesian_bspline(
            tf_guess,
            tf_min=tf_min,
            tf_max=tf_max,
            mee0=mee0,
            mee_target_epoch=mee_target_epoch,
            mu=mu,
            **retry_options,
        )
        bspline_seed["fallback_seed"] = "origin_kepler_absolute_bspline"
        bspline_seed["first_attempt_result"] = bspline_first

    radau_seed = bspline_seed if bool(bspline_seed.get("success", False)) else None

    result = solve_mee_radau_collocation(
        tf_guess,
        tf_min=tf_min,
        tf_max=tf_max,
        mee0=mee0,
        mee_target_epoch=mee_target_epoch,
        mu=mu,
        bspline_seed=radau_seed,
        **radau_kwargs,
    )
    result["bspline_seed_result"] = bspline_seed
    result["bspline_first_result"] = bspline_first
    result["bspline_seed_used"] = radau_seed is not None
    if radau_seed is None:
        result["seed_source"] = "linear_mee_after_failed_bspline"
    return result


def _print_result(result: dict) -> None:
    print("\n" + "=" * 80)
    print("  MEE RADAU COLLOCATION RESULT")
    print("=" * 80)
    print(f"  success              : {result['success']}")
    print(f"  message              : {result['message']}")
    print(f"  seed source          : {result['seed_source']}")
    print(f"  terminal angle mode  : {result['terminal_angle_mode']}")
    print(f"  longitude branch     : {result['longitude_branch']}")
    print(f"  objective            : {result['objective']:.8e}")
    print(f"  objective type       : {result['objective_type']}")
    print(f"  T guess              : {time_from_canonical(result['t_transfer_guess']):.6f} yr")
    print(f"  T optimized          : {time_from_canonical(result['t_transfer']):.6f} yr")
    print(f"  T bounds             : [{time_from_canonical(result['tf_min']):.6f}, {time_from_canonical(result['tf_max']):.6f}] yr")
    print(f"  intervals/degree     : {result['n_intervals']} / {result['degree']}")
    print(f"  max ||u||            : {accel_from_canonical(result['max_u']):.8e}")
    print(f"  DV quadrature        : {dv_from_canonical(result['dv_quad']):.8e}")
    print(f"  energy quadrature    : {energy_from_canonical(result['energy_quad']):.8e}")
    print(f"  endpoint error norm  : {result['endpoint_error_norm']:.8e}")
    print(f"  endpoint error       : {np.array2string(result['endpoint_error'], precision=3)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["default", "dionysus", "mars"], default="default")
    parser.add_argument("--T-guess", type=float, default=None)
    parser.add_argument("--T-base", type=float, default=1.0)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--tf-min", type=float, default=None)
    parser.add_argument("--tf-max", type=float, default=None)
    parser.add_argument("--n-intervals", type=int, default=30)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--objective", choices=["energy", "dv"], default="dv")
    parser.add_argument("--u-max", type=float, default=None)
    parser.add_argument("--control-bound", type=float, default=None)
    parser.add_argument("--time-weight", type=float, default=0.0)
    parser.add_argument("--dv-eps", type=float, default=1e-6)
    parser.add_argument("--smoothness-weight", type=float, default=0.0)
    parser.add_argument("--seed-from-bspline", action="store_true")
    parser.add_argument("--bspline-n-ctrl", type=int, default=40)
    parser.add_argument("--bspline-degree", type=int, default=5)
    parser.add_argument("--bspline-n-fine", type=int, default=600)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--print-level", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mee0, mee_target_epoch = target_for_name(args.target)
    tf_guess = None if args.T_guess is None else time_to_canonical(args.T_guess)
    if tf_guess is None:
        tf_guess = float(time_to_canonical(args.T_base) + int(args.k) * mean_a_transfer_period(mee0, mee_target_epoch, DEFAULT_MU))

    tf_min = None if args.tf_min is None else time_to_canonical(args.tf_min)
    tf_max = None if args.tf_max is None else time_to_canonical(args.tf_max)
    if tf_min is None:
        tf_min = max(1e-3, DEFAULT_TF_MIN_FACTOR * tf_guess)
    if tf_max is None:
        tf_max = max(tf_min + 1e-3, DEFAULT_TF_MAX_FACTOR * tf_guess)
    u_max = accel_to_canonical(args.u_max)
    control_bound = accel_to_canonical(args.control_bound)
    dv_eps = accel_to_canonical(args.dv_eps)

    if args.seed_from_bspline:
        result = solve_with_internal_bspline_seed(
            tf_guess,
            tf_min=tf_min,
            tf_max=tf_max,
            mee0=mee0,
            mee_target_epoch=mee_target_epoch,
            mu=DEFAULT_MU,
            n_intervals=args.n_intervals,
            degree=args.degree,
            objective=args.objective,
            u_max=u_max,
            control_bound=control_bound,
            time_weight=args.time_weight,
            dv_eps=dv_eps,
            smoothness_weight=args.smoothness_weight,
            terminal_angle_mode=DIRECT_TERMINAL_ANGLE_MODE,
            longitude_branch=0,
            max_iter=args.max_iter,
            print_level=args.print_level,
            bspline_options={
                "n_ctrl": args.bspline_n_ctrl,
                "degree": args.bspline_degree,
                "n_fine": args.bspline_n_fine,
                "u_max": u_max,
                "print_level": args.print_level,
            },
        )
    else:
        result = solve_mee_radau_collocation(
            tf_guess,
            tf_min=tf_min,
            tf_max=tf_max,
            mee0=mee0,
            mee_target_epoch=mee_target_epoch,
            mu=DEFAULT_MU,
            n_intervals=args.n_intervals,
            degree=args.degree,
            objective=args.objective,
            u_max=u_max,
            control_bound=control_bound,
            time_weight=args.time_weight,
            dv_eps=dv_eps,
            smoothness_weight=args.smoothness_weight,
            terminal_angle_mode=DIRECT_TERMINAL_ANGLE_MODE,
            longitude_branch=0,
            max_iter=args.max_iter,
            print_level=args.print_level,
        )

    _print_result(result)


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "results_casadi_cartesian_bspline", ".matplotlib"))
    main()

"""Direct Radau collocation in cylindrical coordinates.

The state is ``[rho, theta, z, rho_dot, theta_dot, z_dot]`` and the control is
the cylindrical inertial acceleration ``[u_r, u_theta, u_z]``. This mirrors the
cylindrical B-spline transcription, so saved B-spline control polygons can be
used as direct-collocation seeds without converting through MEE.
"""

from __future__ import annotations

import casadi as ca
import numpy as np

from .optimization_Bspline_freetf import (
    cartesian_to_cylindrical_state,
    cylindrical_state_from_cartesian_sym,
)
from .optimization_MEE_Radaucollocation import (
    build_radau_coefficients,
    shifted_norm,
    target_mee_at_time,
)
from .orbit_utils import kepler_coast_sym, mee_to_rv_sym
from .targets import DEFAULT_MEE0, DEFAULT_MEE_TARGET_EPOCH
from .canonical_units import MU_CANONICAL
from utils.utils import mee2rv


DEFAULT_MU = MU_CANONICAL


def _interp_rows(tau_grid: np.ndarray, values: np.ndarray, tau_query: np.ndarray) -> np.ndarray:
    tau_grid = np.asarray(tau_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    tau_query = np.asarray(tau_query, dtype=float)
    return np.vstack([np.interp(tau_query, tau_grid, values[:, idx]) for idx in range(values.shape[1])]).T


def cylindrical_control_from_profile(
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
    mu: float,
) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    qdot = np.asarray(qdot, dtype=float)
    qddot = np.asarray(qddot, dtype=float)
    rho = q[:, 0]
    z = q[:, 2]
    rho_dot = qdot[:, 0]
    theta_dot = qdot[:, 1]
    rho_ddot = qddot[:, 0]
    theta_ddot = qddot[:, 1]
    z_ddot = qddot[:, 2]
    radius = np.sqrt(rho * rho + z * z + 1e-12)
    return np.column_stack(
        [
            rho_ddot - rho * theta_dot * theta_dot + float(mu) * rho / (radius**3),
            2.0 * rho_dot * theta_dot + rho * theta_ddot,
            z_ddot + float(mu) * z / (radius**3),
        ]
    )


def cylindrical_rhs_sym(x, u, mu: float):
    rho = ca.fmax(x[0], 1e-5)
    z = x[2]
    rho_dot = x[3]
    theta_dot = x[4]
    z_dot = x[5]
    radius = ca.sqrt(rho * rho + z * z + 1e-12)
    return ca.vertcat(
        rho_dot,
        theta_dot,
        z_dot,
        rho * theta_dot * theta_dot - float(mu) * rho / (radius**3) + u[0],
        (u[1] - 2.0 * rho_dot * theta_dot) / rho,
        -float(mu) * z / (radius**3) + u[2],
    )


def endpoint_states(
    mee0: np.ndarray,
    mee_target_epoch: np.ndarray,
    tf_guess: float,
    mu: float,
    winding_target_rev: float | None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    pos0, vel0 = mee2rv(np.asarray(mee0, dtype=float), mu)
    target_guess = target_mee_at_time(float(tf_guess), np.asarray(mee_target_epoch, dtype=float), mu)
    posf, velf = mee2rv(target_guess, mu)
    q0_state = cartesian_to_cylindrical_state(pos0, vel0)
    thetaf_wrapped = float(np.arctan2(posf[1], posf[0]))
    if winding_target_rev is None:
        theta_ref = q0_state[1]
    else:
        theta_ref = q0_state[1] + 2.0 * np.pi * float(winding_target_rev)
    theta_offset = 2.0 * np.pi * float(np.rint((theta_ref - thetaf_wrapped) / (2.0 * np.pi)))
    thetaf_unwrapped = thetaf_wrapped + theta_offset
    qf_state = cartesian_to_cylindrical_state(posf, velf, theta_reference=thetaf_unwrapped)
    return q0_state, qf_state, theta_offset, thetaf_unwrapped


def build_seed_from_cylindrical_bspline(
    bspline_seed: dict,
    mesh_tau: np.ndarray,
    collocation_tau: np.ndarray,
    mu: float,
) -> dict:
    profile = bspline_seed["profile_fine"]
    tau_src = np.asarray(profile["tau"], dtype=float)
    q_src = np.asarray(profile["q_cyl"], dtype=float)
    qdot_src = np.asarray(profile["qdot_cyl"], dtype=float)
    qddot_src = np.asarray(profile["qddot_cyl"], dtype=float)
    state_src = np.column_stack([q_src, qdot_src])
    u_src = cylindrical_control_from_profile(q_src, qdot_src, qddot_src, mu)
    state_tau = np.unique(np.concatenate([mesh_tau, collocation_tau]))
    return {
        "tau_state": state_tau,
        "state": _interp_rows(tau_src, state_src, state_tau),
        "tau_control": collocation_tau,
        "control": _interp_rows(tau_src, u_src, collocation_tau),
        "t_transfer": float(bspline_seed["t_transfer"]),
        "source": "bspline_cylindrical",
    }


def build_linear_seed(
    mesh_tau: np.ndarray,
    collocation_tau: np.ndarray,
    q0_state: np.ndarray,
    qf_state: np.ndarray,
    tf_guess: float,
) -> dict:
    state_tau = np.unique(np.concatenate([mesh_tau, collocation_tau]))
    s = state_tau[:, None]
    state = (1.0 - s) * q0_state.reshape(1, 6) + s * qf_state.reshape(1, 6)
    return {
        "tau_state": state_tau,
        "state": state,
        "tau_control": collocation_tau,
        "control": np.zeros((len(collocation_tau), 3)),
        "t_transfer": float(tf_guess),
        "source": "linear_cylindrical",
    }


def _seed_state(seed: dict, tau: float) -> np.ndarray:
    return _interp_rows(seed["tau_state"], seed["state"], np.array([tau], dtype=float))[0]


def _seed_control(seed: dict, tau: float) -> np.ndarray:
    return _interp_rows(seed["tau_control"], seed["control"], np.array([tau], dtype=float))[0]


def solve_cylindrical_radau_collocation(
    tf_guess: float,
    tf_min: float | None = None,
    tf_max: float | None = None,
    mee0: np.ndarray | None = None,
    mee_target_epoch: np.ndarray | None = None,
    mu: float = DEFAULT_MU,
    winding_target_rev: float | None = None,
    n_intervals: int = 30,
    degree: int = 3,
    objective: str = "dv",
    u_max: float | None = None,
    control_bound: float | None = None,
    r_bound: float = 20.0,
    velocity_bound: float | None = None,
    time_weight: float = 0.0,
    dv_eps: float = 1e-6,
    smoothness_weight: float = 0.0,
    bspline_seed: dict | None = None,
    fixed_tf: bool = True,
    max_iter: int = 2000,
    print_level: int = 5,
    nlp_scaling_method: str = "gradient-based",
) -> dict:
    if mee0 is None:
        mee0 = DEFAULT_MEE0
    if mee_target_epoch is None:
        mee_target_epoch = DEFAULT_MEE_TARGET_EPOCH
    mee0 = np.asarray(mee0, dtype=float).reshape(6)
    mee_target_epoch = np.asarray(mee_target_epoch, dtype=float).reshape(6)
    tf_guess = float(tf_guess)
    if tf_min is None:
        tf_min = max(1e-8, 0.999999 * tf_guess) if fixed_tf else max(1e-3, 0.5 * tf_guess)
    if tf_max is None:
        tf_max = tf_guess + 1e-8 if fixed_tf else max(float(tf_min) + 1e-3, 1.5 * tf_guess)
    tf_min = float(tf_min)
    tf_max = float(tf_max)
    if tf_min <= 0.0 or tf_max <= tf_min:
        raise ValueError("invalid transfer-time bounds")
    if objective not in {"energy", "dv"}:
        raise ValueError("objective must be 'energy' or 'dv'")

    coeff = build_radau_coefficients(int(degree))
    mesh_tau = np.linspace(0.0, 1.0, int(n_intervals) + 1)
    collocation_tau = np.array(
        [
            (k + coeff.tau_root[j]) / int(n_intervals)
            for k in range(int(n_intervals))
            for j in range(1, int(degree) + 1)
        ],
        dtype=float,
    )
    q0_state, qf_guess_state, theta_offset, thetaf_unwrapped = endpoint_states(
        mee0,
        mee_target_epoch,
        tf_guess,
        mu,
        winding_target_rev,
    )
    if bspline_seed is not None:
        seed = build_seed_from_cylindrical_bspline(bspline_seed, mesh_tau, collocation_tau, mu)
    else:
        seed = build_linear_seed(mesh_tau, collocation_tau, q0_state, qf_guess_state, tf_guess)

    seed_state = np.asarray(seed["state"], dtype=float)
    seed_control = np.asarray(seed["control"], dtype=float)
    tf_init = float(np.clip(float(seed.get("t_transfer", tf_guess)), tf_min, tf_max))
    seed_u_peak = float(np.max(np.linalg.norm(seed_control, axis=1))) if seed_control.size else 0.0
    if control_bound is None:
        control_bound = max(2.0 * seed_u_peak, float(u_max or 0.0) * 1.5, 1e-4)
    if velocity_bound is None:
        velocity_bound = max(10.0, 2.0 * float(np.nanmax(np.abs(seed_state[:, 3:]))))

    rho_upper = max(float(r_bound), 2.0 * float(np.nanmax(seed_state[:, 0])), float(q0_state[0]), float(qf_guess_state[0]), 1.0)
    theta_margin = max(2.0 * np.pi, 0.35 * abs(float(thetaf_unwrapped) - float(q0_state[1])))
    theta_min = min(float(q0_state[1]), float(thetaf_unwrapped), float(np.nanmin(seed_state[:, 1]))) - theta_margin
    theta_max = max(float(q0_state[1]), float(thetaf_unwrapped), float(np.nanmax(seed_state[:, 1]))) + theta_margin

    opti = ca.Opti()
    X = opti.variable(6, int(n_intervals) + 1)
    Xc = [opti.variable(6, int(degree)) for _ in range(int(n_intervals))]
    Uc = [opti.variable(3, int(degree)) for _ in range(int(n_intervals))]
    tf_fixed = float(tf_guess)
    tf = tf_fixed if fixed_tf else opti.variable()
    if not fixed_tf:
        opti.subject_to(opti.bounded(tf_min, tf, tf_max))

    opti.subject_to(X[:, 0] == ca.DM(q0_state.reshape(6, 1)))
    for k in range(int(n_intervals) + 1):
        opti.subject_to(opti.bounded(1e-4, X[0, k], rho_upper))
        opti.subject_to(opti.bounded(theta_min, X[1, k], theta_max))
        opti.subject_to(opti.bounded(-float(r_bound), X[2, k], float(r_bound)))
        opti.subject_to(opti.bounded(-float(velocity_bound), X[3:6, k], float(velocity_bound)))
    for k in range(int(n_intervals)):
        opti.subject_to(opti.bounded(1e-4, Xc[k][0, :], rho_upper))
        opti.subject_to(opti.bounded(theta_min, Xc[k][1, :], theta_max))
        opti.subject_to(opti.bounded(-float(r_bound), Xc[k][2, :], float(r_bound)))
        opti.subject_to(opti.bounded(-float(velocity_bound), Xc[k][3:6, :], float(velocity_bound)))
        opti.subject_to(opti.bounded(-float(control_bound), Uc[k], float(control_bound)))
        if u_max is not None:
            for j in range(int(degree)):
                opti.subject_to(ca.sumsqr(Uc[k][:, j]) <= float(u_max) ** 2)

    h_step = tf / int(n_intervals)
    cost = ca.MX(0)
    u_values = []
    collocation_eqs: list[list[ca.MX]] = []
    continuity_eqs: list[ca.MX] = []

    for k in range(int(n_intervals)):
        x_all = [X[:, k]] + [Xc[k][:, j] for j in range(int(degree))]
        collocation_eqs_k = []
        for j in range(1, int(degree) + 1):
            xp = ca.MX.zeros(6, 1)
            for r in range(int(degree) + 1):
                xp += coeff.C[r, j] * x_all[r]
            u_j = Uc[k][:, j - 1]
            defect_eq = h_step * cylindrical_rhs_sym(Xc[k][:, j - 1], u_j, mu) == xp
            opti.subject_to(defect_eq)
            collocation_eqs_k.append(defect_eq)
            u_values.append(u_j)
            if objective == "energy":
                cost += h_step * coeff.B[j] * ca.sumsqr(u_j)
            else:
                cost += h_step * coeff.B[j] * shifted_norm(u_j, dv_eps)
        collocation_eqs.append(collocation_eqs_k)

        x_end = ca.MX.zeros(6, 1)
        for r in range(int(degree) + 1):
            x_end += coeff.D[r] * x_all[r]
        continuity_eq = X[:, k + 1] == x_end
        opti.subject_to(continuity_eq)
        continuity_eqs.append(continuity_eq)

    target_epoch_dm = ca.DM(mee_target_epoch.reshape(6, 1))
    target_final = kepler_coast_sym(target_epoch_dm, tf, mu)
    posf_sym, velf_sym = mee_to_rv_sym(target_final, mu)
    qf_sym, qdotf_sym = cylindrical_state_from_cartesian_sym(posf_sym, velf_sym, theta_offset)
    opti.subject_to(X[0:3, -1] == qf_sym)
    opti.subject_to(X[3:6, -1] == qdotf_sym)

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
    for k in range(int(n_intervals)):
        for j in range(1, int(degree) + 1):
            tau_kj = (k + coeff.tau_root[j]) / int(n_intervals)
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
        value_source = sol
    except RuntimeError as exc:
        success = False
        message = str(exc).splitlines()[-1]
        value_source = opti.debug

    tf_opt = tf_fixed if fixed_tf else float(value_source.value(tf))
    x_nodes = np.array(value_source.value(X), dtype=float)
    x_cols = np.stack([np.array(value_source.value(Xc[k]), dtype=float) for k in range(int(n_intervals))], axis=0)
    u_cols = np.stack([np.array(value_source.value(Uc[k]), dtype=float) for k in range(int(n_intervals))], axis=0)
    obj_val = float(value_source.value(cost))

    u_flat = u_cols.transpose(0, 2, 1).reshape(int(n_intervals) * int(degree), 3)
    u_norm = np.linalg.norm(u_flat, axis=1)
    weights = np.tile(coeff.B[1:], int(n_intervals)) * (tf_opt / int(n_intervals))
    dv_quad = float(np.sum(weights * u_norm))
    energy_quad = float(np.sum(weights * u_norm**2))

    target_final_np = target_mee_at_time(tf_opt, mee_target_epoch, mu)
    posf_np, velf_np = mee2rv(target_final_np, mu)
    qf_np = cartesian_to_cylindrical_state(posf_np, velf_np, theta_reference=thetaf_unwrapped)
    endpoint = x_nodes[:, -1] - qf_np
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
        "t_transfer": tf_opt,
        "t_transfer_guess": tf_guess,
        "tf_min": tf_min,
        "tf_max": tf_max,
        "fixed_tf": bool(fixed_tf),
        "mee0": mee0,
        "mee_target_epoch": mee_target_epoch,
        "mu": float(mu),
        "n_intervals": int(n_intervals),
        "degree": int(degree),
        "u_max": u_max,
        "control_bound": float(control_bound),
        "velocity_bound": float(velocity_bound),
        "max_u": float(np.max(u_norm)),
        "dv_quad": dv_quad,
        "energy_quad": energy_quad,
        "endpoint_error": endpoint,
        "endpoint_error_norm": float(np.linalg.norm(endpoint)),
        "winding_target_rev": float(winding_target_rev) if winding_target_rev is not None else float("nan"),
        "theta_offset": float(theta_offset),
        "seed_source": str(seed.get("source", "")),
    }

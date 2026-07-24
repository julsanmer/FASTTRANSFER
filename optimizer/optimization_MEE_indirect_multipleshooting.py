"""
Indirect MEE low-thrust solver using CasADi/IPOPT multiple shooting.

The decision variables are the state and costate at shooting nodes, plus the
final time when free_time=True.  Segment continuity is imposed by integrating
the canonical Hamiltonian system with a fixed-step RK4 map built in CasADi.

Radau direct-collocation solutions are used to seed both the MEE state history
and a full costate history.  The costates are estimated from the stationarity
relation u = -G(x)^T lambda for the energy objective; for smoothed delta-v the
corresponding smooth stationarity relation is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from .optimization_MEE_Radaucollocation import (
    DEFAULT_MEE0,
    DEFAULT_MEE_TARGET_EPOCH,
    DEFAULT_MU,
    mee_gauss_rhs_sym,
    state_bounds,
    target_mee_at_time,
)
from .optimization_MEE_indirect import (
    ShootingOptions,
    estimate_costates_from_radau,
    f0_fun,
    g_fun,
    indirect_fun,
    integrate_indirect_energy,
    solve_indirect_energy_from_radau,
    states_for_radau_seed,
    trapz,
    wrap_angle,
)
from .orbit_utils import kepler_coast_sym


@dataclass
class MultipleShootingOptions:
    n_segments: int = 10
    rk4_steps_per_segment: int = 4
    max_iter: int = 800
    print_level: int = 0
    lambda_bound: float = 1e4
    residual_tol: float = 1e-4
    state_bound_margin: float = 3.0
    regularization: float = 1e-12
    n_eval: int = 1000
    seed_shooting_max_nfev: int = 40
    free_time_mode: str = "bounded"


def _interp_rows(tau_grid: np.ndarray, values: np.ndarray, tau_query: np.ndarray) -> np.ndarray:
    tau_grid = np.asarray(tau_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    tau_query = np.asarray(tau_query, dtype=float)
    return np.vstack(
        [np.interp(tau_query, tau_grid, values[:, idx]) for idx in range(values.shape[1])]
    ).T


def _unique_average_by_tau(tau: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    order = np.argsort(tau)
    tau_sorted = tau[order]
    values_sorted = values[order]

    unique_tau = []
    unique_values = []
    start = 0
    while start < len(tau_sorted):
        end = start + 1
        while end < len(tau_sorted) and abs(tau_sorted[end] - tau_sorted[start]) <= 1e-12:
            end += 1
        unique_tau.append(float(np.mean(tau_sorted[start:end])))
        unique_values.append(np.mean(values_sorted[start:end], axis=0))
        start = end

    return np.asarray(unique_tau, dtype=float), np.asarray(unique_values, dtype=float)


def costate_profile_from_radau(
    radau_result: dict,
    tau_query: np.ndarray | None = None,
    objective: str = "energy",
    dv_eps: float = 1e-6,
    mu: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a full costate profile estimated from a Radau solution.

    The Radau NLP currently does not expose KKT multipliers, so this seed uses
    the indirect stationarity relation at Radau collocation points and
    interpolates it to the requested nodes.  IPOPT then enforces the true
    costate ODE during multiple shooting.
    """
    if mu is None:
        mu = float(radau_result["mu"])

    tau_lam, lambdas = estimate_costates_from_radau(
        radau_result,
        mu=mu,
        objective=objective,
        dv_eps=dv_eps,
    )
    tau_lam, lambdas = _unique_average_by_tau(tau_lam, lambdas)

    if tau_query is None:
        return tau_lam, lambdas

    tau_query = np.asarray(tau_query, dtype=float)
    lambda_query = _interp_rows(tau_lam, lambdas, tau_query)
    return tau_query, lambda_query


def _stationarity_target(control: np.ndarray, objective: str, dv_eps: float) -> np.ndarray:
    control = np.asarray(control, dtype=float).reshape(3)
    if objective == "energy":
        return -control
    if objective == "dv":
        return -control / np.sqrt(float(np.dot(control, control)) + float(dv_eps) ** 2)
    raise ValueError("objective must be 'energy' or 'dv'")


def _costate_stationarity_error(
    states: np.ndarray,
    controls: np.ndarray,
    lambdas: np.ndarray,
    mu: float,
    objective: str,
    dv_eps: float,
) -> float:
    G_eval = g_fun()
    err_sq = 0.0
    ref_sq = 0.0
    for state, control, lam in zip(states, controls, lambdas):
        G = np.asarray(G_eval(state, mu), dtype=float)
        q = G.T @ np.asarray(lam, dtype=float)
        q_ref = _stationarity_target(control, objective, dv_eps)
        err_sq += float(np.dot(q - q_ref, q - q_ref))
        ref_sq += float(np.dot(q_ref, q_ref))
    return float(np.sqrt(err_sq / max(ref_sq, 1e-16)))


def _project_lambdas_to_stationarity(
    states: np.ndarray,
    controls: np.ndarray,
    lambdas_ref: np.ndarray,
    mu: float,
    objective: str,
    dv_eps: float,
) -> np.ndarray:
    """Project each lambda_ref onto G(x)^T lambda = stationarity_target.

    The stationarity equation gives only 3 equations for 6 costates.  This
    projection keeps the component of the Radau dual seed in the nullspace of
    G(x)^T, while correcting the range-space component so the local optimal
    control law is satisfied.
    """
    G_eval = g_fun()
    projected = np.zeros_like(lambdas_ref, dtype=float)
    for idx, (state, control, lam_ref) in enumerate(zip(states, controls, lambdas_ref)):
        G = np.asarray(G_eval(state, mu), dtype=float)
        A = G.T
        q_ref = _stationarity_target(control, objective, dv_eps)
        residual = A @ lam_ref - q_ref
        gram = A @ A.T
        try:
            correction_coord = np.linalg.solve(gram + 1e-12 * np.eye(3), residual)
        except np.linalg.LinAlgError:
            correction_coord = np.linalg.lstsq(gram + 1e-12 * np.eye(3), residual, rcond=None)[0]
        projected[idx] = lam_ref - A.T @ correction_coord
    return projected


def costate_profile_from_radau_duals(
    radau_result: dict,
    tau_query: np.ndarray | None = None,
    objective: str | None = None,
    dv_eps: float | None = None,
    mu: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return a costate profile estimated from Radau defect multipliers.

    The defect constraints are written in the Radau NLP as

        h f(x_c, u_c) - x_tau = 0.

    For the energy objective used by the direct Radau solver,
    L = ||u||^2, whereas the indirect energy Hamiltonian uses
    0.5 ||u||^2.  The default expected mapping is therefore
    lambda ~= dual_defect / (2 * B_j) for energy and
    lambda ~= dual_defect / B_j for smoothed delta-v.  Since the sign
    convention of stored NLP constraints can be easy to get wrong, this
    routine tests both signs and a few scale conventions against the local
    stationarity relation and selects the best one.
    """
    if "dual_collocation_defects" not in radau_result:
        raise ValueError("radau_result does not contain dual_collocation_defects")
    if mu is None:
        mu = float(radau_result["mu"])
    if objective is None:
        objective = str(radau_result.get("objective_type", "energy"))
    if dv_eps is None:
        dv_eps = float(radau_result.get("dv_eps", 1e-6))

    dual = np.asarray(radau_result["dual_collocation_defects"], dtype=float)
    if dual.ndim != 3 or dual.shape[2] != 6:
        raise ValueError(f"dual_collocation_defects must have shape (N, degree, 6), got {dual.shape}")
    if not np.all(np.isfinite(dual)):
        raise ValueError("Radau duals are not finite")

    n_intervals, degree, _ = dual.shape
    tau_col = np.asarray(radau_result["collocation_tau"], dtype=float)
    states = np.asarray(radau_result["state_collocation"], dtype=float).transpose(0, 2, 1).reshape(-1, 6)
    controls = np.asarray(radau_result["control_collocation"], dtype=float).transpose(0, 2, 1).reshape(-1, 3)
    dual_flat = dual.reshape(n_intervals * degree, 6)

    weights = np.asarray(radau_result.get("radau_weights"), dtype=float)
    if weights.size < degree + 1:
        raise ValueError("radau_weights missing or inconsistent with degree")
    weights_flat = np.tile(weights[1:degree + 1], n_intervals)

    if objective == "energy":
        expected_den = 2.0 * weights_flat
    else:
        expected_den = weights_flat

    h_step = float(radau_result["t_transfer"]) / float(n_intervals)
    denom_candidates = {
        "expected": expected_den,
        "quadrature_weight": weights_flat,
        "raw": np.ones_like(weights_flat),
        "h_expected": h_step * expected_den,
        "h_weight": h_step * weights_flat,
    }

    best = None
    for scale_name, denom in denom_candidates.items():
        denom = np.where(np.abs(denom) > 1e-14, denom, 1.0)
        base_lambdas = dual_flat / denom[:, None]

        q_base = []
        q_ref = []
        G_eval = g_fun()
        for state, control, lam in zip(states, controls, base_lambdas):
            G = np.asarray(G_eval(state, mu), dtype=float)
            q_base.append(G.T @ lam)
            q_ref.append(_stationarity_target(control, objective, float(dv_eps)))
        q_base_vec = np.asarray(q_base, dtype=float).reshape(-1)
        q_ref_vec = np.asarray(q_ref, dtype=float).reshape(-1)
        alpha_den = float(np.dot(q_base_vec, q_base_vec))
        alpha = 0.0 if alpha_den <= 1e-30 else float(np.dot(q_base_vec, q_ref_vec) / alpha_den)
        lambdas = alpha * base_lambdas
        err = _costate_stationarity_error(states, controls, lambdas, mu, objective, float(dv_eps))
        if best is None or err < best["stationarity_error"]:
            best = {
                "scale": scale_name,
                "alpha": alpha,
                "stationarity_error": err,
                "lambdas": lambdas,
            }

    assert best is not None
    raw_lambdas = best["lambdas"]
    projected_lambdas = _project_lambdas_to_stationarity(
        states,
        controls,
        raw_lambdas,
        mu,
        objective,
        float(dv_eps),
    )
    projected_error = _costate_stationarity_error(
        states,
        controls,
        projected_lambdas,
        mu,
        objective,
        float(dv_eps),
    )
    tau_unique, lambda_unique = _unique_average_by_tau(tau_col, projected_lambdas)
    diagnostics = {
        "source": "radau_duals",
        "scale": best["scale"],
        "alpha": float(best["alpha"]),
        "sign": float(np.sign(best["alpha"])),
        "raw_stationarity_error": float(best["stationarity_error"]),
        "stationarity_error": float(projected_error),
        "projection": "closest_dual_seed_satisfying_stationarity",
    }

    if tau_query is None:
        return tau_unique, lambda_unique, diagnostics

    tau_query = np.asarray(tau_query, dtype=float)
    lambda_query = _interp_rows(tau_unique, lambda_unique, tau_query)
    return tau_query, lambda_query, diagnostics


def initial_guess_from_radau(
    radau_result: dict,
    n_segments: int,
    objective: str = "energy",
    dv_eps: float = 1e-6,
    mu: float | None = None,
    costate_source: str = "stationarity",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build node-wise state and costate guesses for multiple shooting."""
    if mu is None:
        mu = float(radau_result["mu"])
    if costate_source not in {"stationarity", "dual"}:
        raise ValueError("costate_source must be 'stationarity' or 'dual'")

    node_tau = np.linspace(0.0, 1.0, int(n_segments) + 1)
    tau_x, states = states_for_radau_seed(radau_result)
    tau_x, states = _unique_average_by_tau(tau_x, states)
    state_guess = _interp_rows(tau_x, states, node_tau)

    if costate_source == "dual":
        _, lambda_guess, _ = costate_profile_from_radau_duals(
            radau_result,
            tau_query=node_tau,
            objective=objective,
            dv_eps=dv_eps,
            mu=mu,
        )
    else:
        _, lambda_guess = costate_profile_from_radau(
            radau_result,
            tau_query=node_tau,
            objective=objective,
            dv_eps=dv_eps,
            mu=mu,
        )

    tf_guess = float(radau_result["t_transfer"])
    state_guess[0] = np.asarray(radau_result["mee0"], dtype=float)
    state_guess[-1] = target_mee_at_time(
        tf_guess,
        np.asarray(radau_result["mee_target_epoch"], dtype=float),
        mu,
    )
    return node_tau, state_guess, lambda_guess


def initial_guess_from_profile(profile: dict, tf: float, n_segments: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate an indirect profile to multiple-shooting nodes."""
    node_tau = np.linspace(0.0, 1.0, int(n_segments) + 1)
    tau = np.asarray(profile["t"], dtype=float) / float(tf)
    states = np.asarray(profile["state"], dtype=float)
    costates = np.asarray(profile["costate"], dtype=float)
    state_guess = _interp_rows(tau, states, node_tau)
    costate_guess = _interp_rows(tau, costates, node_tau)
    return node_tau, state_guess, costate_guess


def _state_scale(mee0: np.ndarray, target: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(target - mee0), 1.0)
    scale[3] = max(scale[3], 0.05)
    scale[4] = max(scale[4], 0.05)
    scale[5] = max(abs(wrap_angle(target[5] - mee0[5])), 1.0)
    return scale


def _u_max_value(u_max: float | None) -> float:
    return -1.0 if u_max is None else float(u_max)


def _rk4_step(rhs_fun, y, h, mu: float, u_max_num: float, dv_eps: float):
    def rhs(y_in):
        return rhs_fun(y_in[:6], y_in[6:], mu, u_max_num, float(dv_eps))[0]

    k1 = rhs(y)
    k2 = rhs(y + 0.5 * h * k1)
    k3 = rhs(y + 0.5 * h * k2)
    k4 = rhs(y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _propagate_segment(rhs_fun, y0, h_segment, n_steps: int, mu: float, u_max_num: float, dv_eps: float):
    y = y0
    h = h_segment / int(n_steps)
    for _ in range(int(n_steps)):
        y = _rk4_step(rhs_fun, y, h, mu, u_max_num, dv_eps)
    return y


def _node_profile(
    state_nodes: np.ndarray,
    costate_nodes: np.ndarray,
    tf: float,
    mu: float,
    objective: str,
    dv_eps: float,
    u_max: float | None,
) -> dict:
    rhs_fun = indirect_fun(objective)
    u_max_num = _u_max_value(u_max)
    controls = np.zeros((state_nodes.shape[0], 3))
    H = np.zeros(state_nodes.shape[0])
    for idx in range(state_nodes.shape[0]):
        _, u_i, H_i = rhs_fun(state_nodes[idx], costate_nodes[idx], mu, u_max_num, float(dv_eps))
        controls[idx] = np.asarray(u_i, dtype=float).reshape(3)
        H[idx] = float(H_i)
    return {
        "t": np.linspace(0.0, float(tf), state_nodes.shape[0]),
        "state": state_nodes,
        "costate": costate_nodes,
        "control": controls,
        "hamiltonian": H,
        "objective": objective,
        "dv_eps": float(dv_eps),
    }


def solve_indirect_multiple_shooting(
    mee0: np.ndarray = DEFAULT_MEE0,
    mee_target_epoch: np.ndarray = DEFAULT_MEE_TARGET_EPOCH,
    mu: float = DEFAULT_MU,
    tf_guess: float | None = None,
    tf_min: float | None = None,
    tf_max: float | None = None,
    state_guess: np.ndarray | None = None,
    costate_guess: np.ndarray | None = None,
    u_max: float | None = None,
    objective: str = "energy",
    dv_eps: float = 1e-6,
    free_time: bool = False,
    options: MultipleShootingOptions | None = None,
) -> dict:
    if objective not in {"energy", "dv"}:
        raise ValueError("objective must be 'energy' or 'dv'")
    if objective == "dv" and u_max is None:
        raise ValueError("Indirect delta-v multiple shooting requires u_max")
    if options is None:
        options = MultipleShootingOptions()
    if options.free_time_mode not in {"bounded", "interior", "lower-bound", "upper-bound"}:
        raise ValueError("free_time_mode must be 'bounded', 'interior', 'lower-bound', or 'upper-bound'")

    n_segments = int(options.n_segments)
    if n_segments < 1:
        raise ValueError("n_segments must be >= 1")
    rk4_steps = int(options.rk4_steps_per_segment)
    if rk4_steps < 1:
        raise ValueError("rk4_steps_per_segment must be >= 1")

    mee0 = np.asarray(mee0, dtype=float).reshape(6)
    mee_target_epoch = np.asarray(mee_target_epoch, dtype=float).reshape(6)
    if tf_guess is None:
        tf_guess = 1.0
    tf_guess = float(tf_guess)
    if tf_min is None:
        tf_min = max(1e-3, 0.5 * tf_guess)
    if tf_max is None:
        tf_max = max(float(tf_min) + 1e-3, 1.5 * tf_guess)
    tf_min = float(tf_min)
    tf_max = float(tf_max)

    target_guess = target_mee_at_time(tf_guess, mee_target_epoch, mu)
    if state_guess is None:
        tau = np.linspace(0.0, 1.0, n_segments + 1)
        state_guess = (1.0 - tau[:, None]) * mee0[None, :] + tau[:, None] * target_guess[None, :]
    if costate_guess is None:
        costate_guess = np.zeros((n_segments + 1, 6))
    state_guess = np.asarray(state_guess, dtype=float)
    costate_guess = np.asarray(costate_guess, dtype=float)
    expected_shape = (n_segments + 1, 6)
    if state_guess.shape != expected_shape:
        raise ValueError(f"state_guess must have shape {expected_shape}, got {state_guess.shape}")
    if costate_guess.shape != expected_shape:
        raise ValueError(f"costate_guess must have shape {expected_shape}, got {costate_guess.shape}")

    opti = ca.Opti()
    X = opti.variable(6, n_segments + 1)
    Lam = opti.variable(6, n_segments + 1)
    tf = opti.variable()
    time_mu_lower = None
    time_mu_upper = None

    opti.set_initial(X, state_guess.T)
    opti.set_initial(Lam, costate_guess.T)
    opti.set_initial(tf, tf_guess)

    if free_time:
        if options.free_time_mode == "lower-bound":
            opti.subject_to(opti.bounded(tf_min, tf, tf_min))
        elif options.free_time_mode == "upper-bound":
            opti.subject_to(opti.bounded(tf_max, tf, tf_max))
        else:
            opti.subject_to(opti.bounded(tf_min, tf, tf_max))
        if options.free_time_mode == "bounded":
            time_mu_lower = opti.variable()
            time_mu_upper = opti.variable()
            opti.set_initial(time_mu_lower, 0.0)
            opti.set_initial(time_mu_upper, 0.0)
            opti.subject_to(time_mu_lower >= 0.0)
            opti.subject_to(time_mu_upper >= 0.0)
    else:
        opti.subject_to(opti.bounded(tf_guess, tf, tf_guess))

    target_epoch_dm = ca.DM(mee_target_epoch.reshape(6, 1))
    target_final_sym = kepler_coast_sym(target_epoch_dm, tf, mu)
    target_final_guess = target_mee_at_time(tf_guess, mee_target_epoch, mu)
    state_scale = _state_scale(mee0, target_final_guess)
    lambda_scale = np.maximum(np.max(np.abs(costate_guess), axis=0), 1.0)
    y_scale = ca.DM(np.concatenate([state_scale, lambda_scale]).reshape(12, 1))

    lb, ub = state_bounds(mee0, target_final_guess, margin=float(options.state_bound_margin))
    for row in range(5):
        opti.subject_to(opti.bounded(lb[row], X[row, :], ub[row]))
    opti.subject_to(opti.bounded(-float(options.lambda_bound), Lam, float(options.lambda_bound)))

    opti.subject_to((X[:, 0] - ca.DM(mee0.reshape(6, 1))) / ca.DM(state_scale.reshape(6, 1)) == 0)
    opti.subject_to((X[:, -1] - target_final_sym) / ca.DM(state_scale.reshape(6, 1)) == 0)

    rhs_fun = indirect_fun(objective)
    u_max_num = _u_max_value(u_max)
    h_segment = tf / n_segments
    for idx in range(n_segments):
        y_i = ca.vertcat(X[:, idx], Lam[:, idx])
        y_next = ca.vertcat(X[:, idx + 1], Lam[:, idx + 1])
        y_prop = _propagate_segment(
            rhs_fun,
            y_i,
            h_segment,
            rk4_steps,
            mu,
            u_max_num,
            float(dv_eps),
        )
        opti.subject_to((y_next - y_prop) / y_scale == 0)

    if free_time and options.free_time_mode in {"bounded", "interior"}:
        zero_u = ca.MX.zeros(3, 1)
        target_dot = mee_gauss_rhs_sym(target_final_sym, zero_u, mu)
        _, _, Hf = rhs_fun(X[:, -1], Lam[:, -1], mu, u_max_num, float(dv_eps))
        trans_expr = Hf - ca.dot(Lam[:, -1], target_dot)
        if options.free_time_mode == "bounded":
            time_span = max(tf_max - tf_min, 1e-12)
            opti.subject_to(trans_expr + time_mu_upper - time_mu_lower == 0)
            opti.subject_to(time_mu_lower * (tf - tf_min) / time_span == 0)
            opti.subject_to(time_mu_upper * (tf_max - tf) / time_span == 0)
        else:
            opti.subject_to(trans_expr == 0)

    reg = float(options.regularization)
    if reg > 0.0:
        state_scale_mat = ca.repmat(ca.DM(state_scale.reshape(6, 1)), 1, n_segments + 1)
        lambda_scale_mat = ca.repmat(ca.DM(lambda_scale.reshape(6, 1)), 1, n_segments + 1)
        cost = reg * (
            ca.sumsqr((X - ca.DM(state_guess.T)) / state_scale_mat)
            + ca.sumsqr((Lam - ca.DM(costate_guess.T)) / lambda_scale_mat)
        )
    else:
        cost = ca.MX(0)
    opti.minimize(cost)

    p_opts = {
        "expand": True,
        "print_time": bool(options.print_level > 0),
        "ipopt": {
            "max_iter": int(options.max_iter),
            "tol": 1e-9,
            "constr_viol_tol": 1e-9,
            "acceptable_tol": 1e-7,
            "acceptable_iter": 10,
            "mu_strategy": "adaptive",
            "nlp_scaling_method": "gradient-based",
            "print_level": int(options.print_level),
        },
    }
    opti.solver("ipopt", p_opts)

    try:
        sol = opti.solve()
        solver_success = True
        message = "Solve_Succeeded"
        tf_opt = float(sol.value(tf))
        state_nodes = np.asarray(sol.value(X), dtype=float).T
        costate_nodes = np.asarray(sol.value(Lam), dtype=float).T
        obj_val = float(sol.value(cost))
        time_mu_lower_opt = float(sol.value(time_mu_lower)) if time_mu_lower is not None else 0.0
        time_mu_upper_opt = float(sol.value(time_mu_upper)) if time_mu_upper is not None else 0.0
    except RuntimeError as exc:
        solver_success = False
        message = str(exc).splitlines()[-1]
        tf_opt = float(opti.debug.value(tf))
        state_nodes = np.asarray(opti.debug.value(X), dtype=float).T
        costate_nodes = np.asarray(opti.debug.value(Lam), dtype=float).T
        obj_val = float(opti.debug.value(cost))
        try:
            time_mu_lower_opt = float(opti.debug.value(time_mu_lower)) if time_mu_lower is not None else 0.0
            time_mu_upper_opt = float(opti.debug.value(time_mu_upper)) if time_mu_upper is not None else 0.0
        except Exception:
            time_mu_lower_opt = np.nan
            time_mu_upper_opt = np.nan

    node_profile = _node_profile(
        state_nodes,
        costate_nodes,
        tf_opt,
        mu,
        objective,
        float(dv_eps),
        u_max,
    )

    try:
        profile = integrate_indirect_energy(
            costate_nodes[0],
            tf_opt,
            mee0=mee0,
            mu=mu,
            u_max=u_max,
            objective=objective,
            dv_eps=float(dv_eps),
            n_eval=int(options.n_eval),
        )
    except Exception as exc:
        profile = dict(node_profile)
        profile["success"] = False
        profile["message"] = str(exc)

    target_final = target_mee_at_time(tf_opt, mee_target_epoch, mu)
    node_endpoint = state_nodes[-1] - target_final
    node_endpoint[5] = wrap_angle(node_endpoint[5])
    endpoint = np.asarray(profile["state"][-1], dtype=float) - target_final
    endpoint[5] = wrap_angle(endpoint[5])
    target_dot = np.asarray(f0_fun()(target_final, mu), dtype=float).reshape(6)
    hamiltonian_final = float(node_profile["hamiltonian"][-1])
    lambda_target_dot = float(np.dot(costate_nodes[-1], target_dot))
    transversality = hamiltonian_final - lambda_target_dot
    transversality_plus = hamiltonian_final + lambda_target_dot

    max_defect = 0.0
    rhs_eval = indirect_fun(objective)
    for idx in range(n_segments):
        y_i = ca.DM(np.concatenate([state_nodes[idx], costate_nodes[idx]]))
        y_prop = _propagate_segment(
            rhs_eval,
            y_i,
            tf_opt / n_segments,
            rk4_steps,
            mu,
            u_max_num,
            float(dv_eps),
        )
        y_next = np.concatenate([state_nodes[idx + 1], costate_nodes[idx + 1]])
        defect = y_next - np.asarray(y_prop, dtype=float).reshape(12)
        scaled_defect = defect / np.concatenate([state_scale, lambda_scale])
        max_defect = max(max_defect, float(np.linalg.norm(scaled_defect, ord=np.inf)))

    u_norm = np.linalg.norm(profile["control"], axis=1)
    energy = 0.5 * trapz(u_norm**2, profile["t"])
    dv = trapz(u_norm, profile["t"])
    endpoint_norm = float(np.linalg.norm(endpoint))
    node_endpoint_norm = float(np.linalg.norm(node_endpoint))

    return {
        "success": bool(
            solver_success
            and bool(profile.get("success", True))
            and node_endpoint_norm <= float(options.residual_tol)
            and endpoint_norm <= float(options.residual_tol)
            and max_defect <= max(float(options.residual_tol), 1e-8)
        ),
        "solver_success": bool(solver_success),
        "integrator_success": bool(profile.get("success", True)),
        "message": message,
        "objective_value": obj_val,
        "t_transfer": tf_opt,
        "t_transfer_guess": tf_guess,
        "tf_min": tf_min,
        "tf_max": tf_max,
        "free_time": bool(free_time),
        "free_time_mode": str(options.free_time_mode),
        "time_mu_lower": time_mu_lower_opt,
        "time_mu_upper": time_mu_upper_opt,
        "objective": objective,
        "dv_eps": float(dv_eps),
        "target_mee": target_final,
        "mee0": mee0,
        "mee_target_epoch": mee_target_epoch,
        "mu": float(mu),
        "u_max": u_max,
        "state_nodes": state_nodes,
        "costate_nodes": costate_nodes,
        "node_profile": node_profile,
        "profile": profile,
        "node_endpoint_error": node_endpoint,
        "node_endpoint_error_norm": node_endpoint_norm,
        "endpoint_error": endpoint,
        "endpoint_error_norm": endpoint_norm,
        "max_scaled_defect": max_defect,
        "transversality": transversality,
        "transversality_plus": transversality_plus,
        "hamiltonian_final": hamiltonian_final,
        "lambda_target_dot": lambda_target_dot,
        "energy": energy,
        "dv": dv,
        "max_u": float(np.max(u_norm)),
        "n_segments": n_segments,
        "rk4_steps_per_segment": rk4_steps,
        "method": f"indirect_{objective}_multiple_shooting_ipopt",
    }


def solve_indirect_multiple_shooting_from_radau(
    radau_result: dict,
    objective: str = "energy",
    dv_eps: float = 1e-6,
    u_max: float | None = None,
    free_time: bool = False,
    costate_seed: str = "shooting",
    options: MultipleShootingOptions | None = None,
) -> dict:
    if options is None:
        options = MultipleShootingOptions()
    if costate_seed not in {"radau", "dual", "shooting"}:
        raise ValueError("costate_seed must be 'radau', 'dual', or 'shooting'")

    mu = float(radau_result["mu"])
    costate_source = "dual" if costate_seed == "dual" else "stationarity"
    dual_diagnostics = None
    _, state_guess, costate_guess = initial_guess_from_radau(
        radau_result,
        n_segments=int(options.n_segments),
        objective=objective,
        dv_eps=float(dv_eps),
        mu=mu,
        costate_source=costate_source,
    )
    if costate_source == "dual":
        _, _, dual_diagnostics = costate_profile_from_radau_duals(
            radau_result,
            tau_query=np.linspace(0.0, 1.0, int(options.n_segments) + 1),
            objective=objective,
            dv_eps=float(dv_eps),
            mu=mu,
        )
    radau_state_guess = state_guess.copy()
    radau_costate_guess = costate_guess.copy()
    shooting_seed_result = None

    if costate_seed == "shooting":
        shooting_options = ShootingOptions(
            max_nfev=int(options.seed_shooting_max_nfev),
            verbose=0,
            residual_tol=max(float(options.residual_tol), 1e-10),
        )
        shooting_seed_result = solve_indirect_energy_from_radau(
            radau_result,
            objective=objective,
            dv_eps=float(dv_eps),
            u_max=u_max,
            free_time=False,
            options=shooting_options,
        )
        if shooting_seed_result["success"]:
            _, state_guess, costate_guess = initial_guess_from_profile(
                shooting_seed_result["profile"],
                float(shooting_seed_result["t_transfer"]),
                int(options.n_segments),
            )

    result = solve_indirect_multiple_shooting(
        mee0=np.asarray(radau_result["mee0"], dtype=float),
        mee_target_epoch=np.asarray(radau_result["mee_target_epoch"], dtype=float),
        mu=mu,
        tf_guess=float(radau_result["t_transfer"]),
        tf_min=float(radau_result["tf_min"]),
        tf_max=float(radau_result["tf_max"]),
        state_guess=state_guess,
        costate_guess=costate_guess,
        u_max=u_max,
        objective=objective,
        dv_eps=float(dv_eps),
        free_time=bool(free_time),
        options=options,
    )
    result["radau_seed_result"] = radau_result
    result["radau_state_seed_nodes"] = radau_state_guess
    result["radau_costate_seed_nodes"] = radau_costate_guess
    result["costate_seed_nodes"] = costate_guess
    result["state_seed_nodes"] = state_guess
    result["costate_seed_source"] = (
        "shooting" if shooting_seed_result is not None and shooting_seed_result["success"] else costate_source
    )
    result["dual_costate_diagnostics"] = dual_diagnostics
    result["shooting_seed_result"] = shooting_seed_result
    return result

"""
Indirect MEE low-thrust solver using Radau collocation of the PMP system.

This is an indirect collocation method: the state and costate are collocated
together as y = [x, lambda], with the control recovered from Pontryagin's
stationarity relation inside the Hamiltonian RHS.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np

from .optimization_MEE_Radaucollocation import (
    DEFAULT_MEE0,
    DEFAULT_MEE_TARGET_EPOCH,
    DEFAULT_MU,
    build_radau_coefficients,
    mee_gauss_rhs_sym,
    target_mee_at_time,
)
from .optimization_MEE_indirect import (
    f0_fun,
    indirect_fun,
    integrate_indirect_energy,
    trapz,
    wrap_angle,
)
from .optimization_MEE_indirect_multipleshooting import initial_guess_from_radau
from .orbit_utils import kepler_coast_sym


@dataclass
class IndirectRadauOptions:
    n_intervals: int = 10
    degree: int = 3
    max_iter: int = 800
    print_level: int = 0
    lambda_bound: float = 1e4
    regularization: float = 1e-12
    n_eval: int = 1000
    terminal_angle_mode: str = "unwrapped"
    longitude_branch: int = 0
    nlp_scaling_method: str = "gradient-based"


def _u_max_value(u_max: float | None) -> float:
    return -1.0 if u_max is None else float(u_max)


def _interp_rows(tau_grid: np.ndarray, values: np.ndarray, tau_query: np.ndarray) -> np.ndarray:
    tau_grid = np.asarray(tau_grid, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    tau_query = np.asarray(tau_query, dtype=float).reshape(-1)
    return np.vstack([np.interp(tau_query, tau_grid, values[:, i]) for i in range(values.shape[1])]).T


def _seed_nodes_from_radau(radau_result: dict, n_intervals: int, degree: int, objective: str, dv_eps: float) -> dict:
    mesh_tau = np.linspace(0.0, 1.0, int(n_intervals) + 1)
    coeff = build_radau_coefficients(int(degree))
    collocation_tau = np.array(
        [(k + coeff.tau_root[j]) / int(n_intervals) for k in range(int(n_intervals)) for j in range(1, int(degree) + 1)],
        dtype=float,
    )

    _, state_nodes, costate_nodes = initial_guess_from_radau(
        radau_result,
        n_segments=int(n_intervals),
        objective=objective,
        dv_eps=float(dv_eps),
        mu=float(radau_result["mu"]),
        costate_source="dual" if radau_result.get("dual_extracted", False) else "stationarity",
    )

    state_collocation = _interp_rows(mesh_tau, state_nodes, collocation_tau).reshape(int(n_intervals), int(degree), 6)
    costate_collocation = _interp_rows(mesh_tau, costate_nodes, collocation_tau).reshape(int(n_intervals), int(degree), 6)
    return {
        "mesh_tau": mesh_tau,
        "collocation_tau": collocation_tau,
        "state_nodes": state_nodes,
        "costate_nodes": costate_nodes,
        "state_collocation": state_collocation,
        "costate_collocation": costate_collocation,
    }


def solve_indirect_mee_radau_collocation(
    radau_result: dict,
    objective: str = "energy",
    dv_eps: float = 1e-6,
    u_max: float | None = None,
    free_time: bool = True,
    tf_min: float | None = None,
    tf_max: float | None = None,
    costate_seed: str = "dual",
    options: IndirectRadauOptions | None = None,
) -> dict:
    if objective not in {"energy", "dv"}:
        raise ValueError("objective must be 'energy' or 'dv'")
    if objective == "dv" and u_max is None:
        raise ValueError("Indirect delta-v Radau collocation requires u_max")
    if costate_seed not in {"dual", "radau", "zero"}:
        raise ValueError("costate_seed must be 'dual', 'radau', or 'zero'")
    if options is None:
        options = IndirectRadauOptions()
    if options.terminal_angle_mode not in {"unwrapped", "sincos", "branch"}:
        raise ValueError("terminal_angle_mode must be 'unwrapped', 'sincos', or 'branch'")

    n_intervals = int(options.n_intervals)
    degree = int(options.degree)
    coeff = build_radau_coefficients(degree)
    tf_guess = float(radau_result["t_transfer"])
    tf_min = float(radau_result["tf_min"] if tf_min is None else tf_min)
    tf_max = float(radau_result["tf_max"] if tf_max is None else tf_max)
    mu = float(radau_result["mu"])
    mee0 = np.asarray(radau_result["mee0"], dtype=float).reshape(6)
    mee_target_epoch = np.asarray(radau_result["mee_target_epoch"], dtype=float).reshape(6)

    seed = _seed_nodes_from_radau(radau_result, n_intervals, degree, objective, float(dv_eps))
    if costate_seed == "zero":
        seed["costate_nodes"] = np.zeros_like(seed["costate_nodes"])
        seed["costate_collocation"] = np.zeros_like(seed["costate_collocation"])
    elif costate_seed == "radau":
        _, state_nodes, costate_nodes = initial_guess_from_radau(
            radau_result,
            n_segments=n_intervals,
            objective=objective,
            dv_eps=float(dv_eps),
            mu=mu,
            costate_source="stationarity",
        )
        seed["state_nodes"] = state_nodes
        seed["costate_nodes"] = costate_nodes
        seed["state_collocation"] = _interp_rows(seed["mesh_tau"], state_nodes, seed["collocation_tau"]).reshape(n_intervals, degree, 6)
        seed["costate_collocation"] = _interp_rows(seed["mesh_tau"], costate_nodes, seed["collocation_tau"]).reshape(n_intervals, degree, 6)

    opti = ca.Opti()
    X = opti.variable(6, n_intervals + 1)
    Lam = opti.variable(6, n_intervals + 1)
    Xc = [opti.variable(6, degree) for _ in range(n_intervals)]
    Lamc = [opti.variable(6, degree) for _ in range(n_intervals)]
    tf = opti.variable() if free_time else tf_guess

    if free_time:
        opti.subject_to(opti.bounded(tf_min, tf, tf_max))

    opti.subject_to(X[:, 0] == ca.DM(mee0))
    opti.subject_to(opti.bounded(-float(options.lambda_bound), Lam, float(options.lambda_bound)))
    for k in range(n_intervals):
        opti.subject_to(opti.bounded(-float(options.lambda_bound), Lamc[k], float(options.lambda_bound)))

    rhs_fun = indirect_fun(objective)
    u_max_num = _u_max_value(u_max)
    h_step = tf / n_intervals

    for k in range(n_intervals):
        y_all = [ca.vertcat(X[:, k], Lam[:, k])] + [
            ca.vertcat(Xc[k][:, j], Lamc[k][:, j]) for j in range(degree)
        ]
        for j in range(1, degree + 1):
            yp = ca.MX.zeros(12, 1)
            for r in range(degree + 1):
                yp += coeff.C[r, j] * y_all[r]
            ydot_j, _, _ = rhs_fun(Xc[k][:, j - 1], Lamc[k][:, j - 1], mu, u_max_num, float(dv_eps))
            opti.subject_to(h_step * ydot_j == yp)

        y_end = ca.MX.zeros(12, 1)
        for r in range(degree + 1):
            y_end += coeff.D[r] * y_all[r]
        opti.subject_to(ca.vertcat(X[:, k + 1], Lam[:, k + 1]) == y_end)

    target_epoch_dm = ca.DM(mee_target_epoch.reshape(6, 1))
    target_final = kepler_coast_sym(target_epoch_dm, tf, mu)
    if options.terminal_angle_mode == "branch":
        target_final = ca.vertcat(
            target_final[0],
            target_final[1],
            target_final[2],
            target_final[3],
            target_final[4],
            target_final[5] + 2.0 * np.pi * int(options.longitude_branch),
        )
    opti.subject_to(X[0:5, -1] == target_final[0:5])
    if options.terminal_angle_mode == "sincos":
        delta_l = X[5, -1] - target_final[5]
        opti.subject_to(ca.sin(delta_l) == 0.0)
        opti.subject_to(ca.cos(delta_l) >= 0.0)
    else:
        opti.subject_to(X[5, -1] == target_final[5])

    if free_time:
        zero_u = ca.MX.zeros(3, 1)
        target_dot = mee_gauss_rhs_sym(target_final, zero_u, mu)
        _, _, Hf = rhs_fun(X[:, -1], Lam[:, -1], mu, u_max_num, float(dv_eps))
        opti.subject_to(Hf - ca.dot(Lam[:, -1], target_dot) == 0)

    reg = ca.sumsqr(Lam)
    for k in range(n_intervals):
        reg += ca.sumsqr(Lamc[k])
    opti.minimize(float(options.regularization) * reg)

    if free_time:
        opti.set_initial(tf, tf_guess)
    opti.set_initial(X, seed["state_nodes"].T)
    opti.set_initial(Lam, seed["costate_nodes"].T)
    for k in range(n_intervals):
        opti.set_initial(Xc[k], seed["state_collocation"][k].T)
        opti.set_initial(Lamc[k], seed["costate_collocation"][k].T)

    p_opts = {
        "expand": True,
        "print_time": bool(options.print_level > 0),
        "ipopt": {
            "max_iter": int(options.max_iter),
            "tol": 1e-8,
            "constr_viol_tol": 1e-8,
            "acceptable_tol": 1e-6,
            "acceptable_iter": 10,
            "mu_strategy": "adaptive",
            "nlp_scaling_method": str(options.nlp_scaling_method),
            "print_level": int(options.print_level),
        },
    }
    opti.solver("ipopt", p_opts)

    try:
        sol = opti.solve()
        success = True
        message = "Solve_Succeeded"
        value = sol.value
    except RuntimeError as exc:
        success = False
        message = str(exc).splitlines()[-1]
        value = opti.debug.value

    tf_opt = float(value(tf)) if free_time else tf_guess
    state_nodes = np.asarray(value(X), dtype=float).T
    costate_nodes = np.asarray(value(Lam), dtype=float).T
    state_collocation = np.stack([np.asarray(value(Xc[k]), dtype=float).T for k in range(n_intervals)], axis=0)
    costate_collocation = np.stack([np.asarray(value(Lamc[k]), dtype=float).T for k in range(n_intervals)], axis=0)

    control_collocation = np.zeros((n_intervals, degree, 3))
    hamiltonian_collocation = np.zeros((n_intervals, degree))
    for k in range(n_intervals):
        for j in range(degree):
            _, u_j, h_j = rhs_fun(state_collocation[k, j], costate_collocation[k, j], mu, u_max_num, float(dv_eps))
            control_collocation[k, j] = np.asarray(u_j, dtype=float).reshape(3)
            hamiltonian_collocation[k, j] = float(h_j)

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
    target_final_np = target_mee_at_time(tf_opt, mee_target_epoch, mu)
    if options.terminal_angle_mode == "branch":
        target_final_np = np.array(target_final_np, dtype=float, copy=True)
        target_final_np[5] += 2.0 * np.pi * int(options.longitude_branch)
    node_endpoint = state_nodes[-1] - target_final_np
    node_endpoint[5] = wrap_angle(node_endpoint[5])
    endpoint = profile["state"][-1] - target_final_np
    endpoint[5] = wrap_angle(endpoint[5])

    target_dot = np.asarray(f0_fun()(target_final_np, mu), dtype=float).reshape(6)
    _, _, Hf_num = rhs_fun(state_nodes[-1], costate_nodes[-1], mu, u_max_num, float(dv_eps))
    hamiltonian_final = float(Hf_num)
    lambda_target_dot = float(np.dot(costate_nodes[-1], target_dot))
    u_norm = np.linalg.norm(profile["control"], axis=1)
    u_col_flat = control_collocation.reshape(n_intervals * degree, 3)
    u_col_norm = np.linalg.norm(u_col_flat, axis=1)
    weights = np.tile(coeff.B[1:], n_intervals) * (tf_opt / n_intervals)

    return {
        "success": bool(success and profile["success"]),
        "message": message,
        "method": f"indirect_{objective}_radau_collocation",
        "objective": objective,
        "dv_eps": float(dv_eps),
        "free_time": bool(free_time),
        "terminal_angle_mode": options.terminal_angle_mode,
        "longitude_branch": int(options.longitude_branch),
        "costate_seed_source": costate_seed,
        "t_transfer": tf_opt,
        "t_transfer_guess": tf_guess,
        "tf_min": tf_min,
        "tf_max": tf_max,
        "mee0": mee0,
        "mee_target_epoch": mee_target_epoch,
        "target_mee": target_final_np,
        "mu": mu,
        "u_max": u_max,
        "state_nodes": state_nodes,
        "costate_nodes": costate_nodes,
        "state_collocation": state_collocation,
        "costate_collocation": costate_collocation,
        "control_collocation": control_collocation,
        "hamiltonian_collocation": hamiltonian_collocation,
        "mesh_tau": seed["mesh_tau"],
        "collocation_tau": seed["collocation_tau"],
        "radau_tau_root": coeff.tau_root,
        "profile": profile,
        "node_endpoint_error": node_endpoint,
        "node_endpoint_error_norm": float(np.linalg.norm(node_endpoint)),
        "endpoint_error": endpoint,
        "endpoint_error_norm": float(np.linalg.norm(endpoint)),
        "transversality": hamiltonian_final - lambda_target_dot,
        "hamiltonian_final": hamiltonian_final,
        "lambda_target_dot": lambda_target_dot,
        "energy": 0.5 * trapz(u_norm**2, profile["t"]),
        "dv": trapz(u_norm, profile["t"]),
        "energy_collocation": float(0.5 * np.sum(weights * u_col_norm**2)),
        "dv_collocation": float(np.sum(weights * u_col_norm)),
        "max_u": float(np.max(u_norm)),
        "max_u_collocation": float(np.max(u_col_norm)),
        "n_intervals": n_intervals,
        "degree": degree,
        "rk4_reintegration_success": bool(profile["success"]),
        "radau_seed_result": radau_result,
    }

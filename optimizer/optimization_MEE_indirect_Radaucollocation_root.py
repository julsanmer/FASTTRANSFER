"""
Indirect MEE Radau collocation solved as a square root problem.

CasADi builds the Radau residual F(z) and analytic Jacobian dF/dz. SciPy's
root solver then solves F(z)=0 without IPOPT.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
from scipy.optimize import root

from .optimization_MEE_Radaucollocation import (
    build_radau_coefficients,
    mee_gauss_rhs_sym,
    target_mee_at_time,
)
from .optimization_MEE_indirect import f0_fun, indirect_fun, integrate_indirect_energy, trapz, wrap_angle
from .optimization_MEE_indirect_Radaucollocation import _interp_rows, _seed_nodes_from_radau, _u_max_value
from .optimization_MEE_indirect_multipleshooting import initial_guess_from_radau
from .orbit_utils import kepler_coast_sym


@dataclass
class IndirectRadauRootOptions:
    n_intervals: int = 10
    degree: int = 3
    maxfev: int = 200
    n_eval: int = 1000
    terminal_angle_mode: str = "unwrapped"
    longitude_branch: int = 0
    method: str = "hybr"


def _state_scale(mee0: np.ndarray, target: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(target - mee0), 1.0)
    scale[3] = max(scale[3], 0.05)
    scale[4] = max(scale[4], 0.05)
    scale[5] = max(abs(wrap_angle(target[5] - mee0[5])), 1.0)
    return scale


def _pack(y_nodes: np.ndarray, y_cols: np.ndarray, tf: float | None) -> np.ndarray:
    parts = [
        np.asarray(y_nodes, dtype=float).T.reshape(-1, order="F"),
        np.asarray(y_cols, dtype=float).reshape(-1, 12).T.reshape(-1, order="F"),
    ]
    if tf is not None:
        parts.append(np.array([float(tf)]))
    return np.concatenate(parts)


def _unpack(z: np.ndarray, n_intervals: int, degree: int, free_time: bool, tf_fixed: float):
    z = np.asarray(z, dtype=float).reshape(-1)
    n_nodes = int(n_intervals) + 1
    n_node_vars = 12 * n_nodes
    y_nodes = z[:n_node_vars].reshape((12, n_nodes), order="F").T
    y_cols = z[n_node_vars:n_node_vars + 12 * int(n_intervals) * int(degree)]
    y_cols = y_cols.reshape((12, int(n_intervals) * int(degree)), order="F").T.reshape(int(n_intervals), int(degree), 12)
    tf = float(z[-1]) if free_time else float(tf_fixed)
    return y_nodes, y_cols, tf


def _array_with_shape(source: dict, key: str, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(source[key], dtype=float)
    if value.shape != shape:
        raise ValueError(f"initial_guess['{key}'] has shape {value.shape}, expected {shape}")
    return np.array(value, dtype=float, copy=True)


def build_indirect_radau_root_functions(
    mee0: np.ndarray,
    mee_target_epoch: np.ndarray,
    mu: float,
    tf_guess: float,
    tf_min: float,
    tf_max: float,
    state_scale: np.ndarray,
    lambda_scale: np.ndarray,
    n_intervals: int,
    degree: int,
    objective: str,
    dv_eps: float,
    u_max: float | None,
    free_time: bool,
    terminal_angle_mode: str,
    longitude_branch: int,
) -> tuple[ca.Function, ca.Function]:
    if terminal_angle_mode not in {"unwrapped", "sin", "branch"}:
        raise ValueError("terminal_angle_mode must be 'unwrapped', 'sin', or 'branch'")
    longitude_branch = int(longitude_branch)

    n_nodes = int(n_intervals) + 1
    n_col = int(n_intervals) * int(degree)
    n_vars = 12 * n_nodes + 12 * n_col + (1 if free_time else 0)
    z = ca.MX.sym("z", n_vars)

    y_nodes = ca.reshape(z[: 12 * n_nodes], 12, n_nodes)
    y_cols = ca.reshape(z[12 * n_nodes: 12 * n_nodes + 12 * n_col], 12, n_col)
    tf = z[-1] if free_time else float(tf_guess)

    coeff = build_radau_coefficients(int(degree))
    rhs_fun = indirect_fun(objective)
    u_max_num = _u_max_value(u_max)
    h_step = tf / int(n_intervals)
    state_scale_dm = ca.DM(np.asarray(state_scale, dtype=float).reshape(6, 1))
    y_scale_dm = ca.DM(np.concatenate([state_scale, lambda_scale]).reshape(12, 1))

    residuals = [(y_nodes[0:6, 0] - ca.DM(mee0.reshape(6, 1))) / state_scale_dm]

    for k in range(int(n_intervals)):
        y_all = [y_nodes[:, k]] + [y_cols[:, k * int(degree) + j] for j in range(int(degree))]
        for j in range(1, int(degree) + 1):
            yp = ca.MX.zeros(12, 1)
            for r in range(int(degree) + 1):
                yp += coeff.C[r, j] * y_all[r]
            x_j = y_cols[0:6, k * int(degree) + j - 1]
            lam_j = y_cols[6:12, k * int(degree) + j - 1]
            ydot_j, _, _ = rhs_fun(x_j, lam_j, mu, u_max_num, float(dv_eps))
            residuals.append((h_step * ydot_j - yp) / y_scale_dm)

        y_end = ca.MX.zeros(12, 1)
        for r in range(int(degree) + 1):
            y_end += coeff.D[r] * y_all[r]
        residuals.append((y_nodes[:, k + 1] - y_end) / y_scale_dm)

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
    residuals.append((y_nodes[0:5, -1] - target_final[0:5]) / state_scale_dm[0:5])
    if terminal_angle_mode == "sin":
        residuals.append(ca.vertcat(ca.sin(y_nodes[5, -1] - target_final[5])))
    else:
        residuals.append(ca.vertcat((y_nodes[5, -1] - target_final[5]) / state_scale_dm[5]))

    if free_time:
        target_dot = mee_gauss_rhs_sym(target_final, ca.MX.zeros(3, 1), mu)
        _, _, Hf = rhs_fun(y_nodes[0:6, -1], y_nodes[6:12, -1], mu, u_max_num, float(dv_eps))
        residuals.append(ca.vertcat(Hf - ca.dot(y_nodes[6:12, -1], target_dot)))

    F = ca.vertcat(*residuals)
    J = ca.jacobian(F, z)
    return ca.Function("indirect_radau_root_F", [z], [F]), ca.Function("indirect_radau_root_J", [z], [J])


def solve_indirect_mee_radau_root(
    radau_result: dict,
    objective: str = "energy",
    dv_eps: float = 1e-6,
    u_max: float | None = None,
    free_time: bool = False,
    tf_min: float | None = None,
    tf_max: float | None = None,
    costate_seed: str = "dual",
    options: IndirectRadauRootOptions | None = None,
    initial_guess: dict | None = None,
) -> dict:
    if options is None:
        options = IndirectRadauRootOptions()
    if objective == "dv" and u_max is None:
        raise ValueError("Indirect delta-v Radau root requires u_max")
    if costate_seed not in {"dual", "radau", "zero"}:
        raise ValueError("costate_seed must be 'dual', 'radau', or 'zero'")

    n_intervals = int(options.n_intervals)
    degree = int(options.degree)
    tf_guess = float(radau_result["t_transfer"])
    tf_min = float(radau_result["tf_min"] if tf_min is None else tf_min)
    tf_max = float(radau_result["tf_max"] if tf_max is None else tf_max)
    mu = float(radau_result["mu"])
    mee0 = np.asarray(radau_result["mee0"], dtype=float).reshape(6)
    mee_target_epoch = np.asarray(radau_result["mee_target_epoch"], dtype=float).reshape(6)
    target_guess = target_mee_at_time(tf_guess, mee_target_epoch, mu)
    if options.terminal_angle_mode == "branch":
        target_guess = np.array(target_guess, dtype=float, copy=True)
        target_guess[5] += 2.0 * np.pi * int(options.longitude_branch)

    seed = _seed_nodes_from_radau(radau_result, n_intervals, degree, objective, float(dv_eps))
    initial_guess_source = "radau"
    if costate_seed == "zero":
        seed["costate_nodes"] = np.zeros_like(seed["costate_nodes"])
        seed["costate_collocation"] = np.zeros_like(seed["costate_collocation"])
        initial_guess_source = "zero"
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
        seed["state_collocation"] = _interp_rows(seed["mesh_tau"], state_nodes, seed["collocation_tau"]).reshape(
            n_intervals, degree, 6
        )
        seed["costate_collocation"] = _interp_rows(seed["mesh_tau"], costate_nodes, seed["collocation_tau"]).reshape(
            n_intervals, degree, 6
        )
        initial_guess_source = "radau_stationarity"

    if initial_guess is not None:
        seed["state_nodes"] = _array_with_shape(initial_guess, "state_nodes", (n_intervals + 1, 6))
        seed["costate_nodes"] = _array_with_shape(initial_guess, "costate_nodes", (n_intervals + 1, 6))
        seed["state_collocation"] = _array_with_shape(
            initial_guess, "state_collocation", (n_intervals, degree, 6)
        )
        seed["costate_collocation"] = _array_with_shape(
            initial_guess, "costate_collocation", (n_intervals, degree, 6)
        )
        seed["state_nodes"][0] = mee0
        seed["state_nodes"][-1] = target_guess
        initial_guess_source = str(initial_guess.get("source", "external"))

    y_nodes0 = np.hstack([seed["state_nodes"], seed["costate_nodes"]])
    y_cols0 = np.concatenate([seed["state_collocation"], seed["costate_collocation"]], axis=2)
    z0 = _pack(y_nodes0, y_cols0, tf_guess if free_time else None)

    state_scale = _state_scale(mee0, target_guess)
    lambda_scale = np.maximum(np.max(np.abs(seed["costate_nodes"]), axis=0), 1.0)
    angle_mode = "sin" if options.terminal_angle_mode == "sincos" else options.terminal_angle_mode
    F_fun, J_fun = build_indirect_radau_root_functions(
        mee0, mee_target_epoch, mu, tf_guess, tf_min, tf_max, state_scale, lambda_scale,
        n_intervals, degree, objective, float(dv_eps), u_max, bool(free_time), angle_mode,
        int(options.longitude_branch),
    )

    def residual(z):
        return np.asarray(F_fun(z), dtype=float).reshape(-1)

    def jacobian(z):
        return np.asarray(J_fun(z), dtype=float)

    sol = root(
        residual,
        z0,
        jac=jacobian,
        method=str(options.method),
        options={"maxfev": int(options.maxfev)},
    )

    y_nodes, y_cols, tf_opt = _unpack(sol.x, n_intervals, degree, bool(free_time), tf_guess)
    state_nodes = y_nodes[:, 0:6]
    costate_nodes = y_nodes[:, 6:12]
    state_collocation = y_cols[:, :, 0:6]
    costate_collocation = y_cols[:, :, 6:12]
    res_final = residual(sol.x)

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

    rhs_fun = indirect_fun(objective)
    u_max_num = _u_max_value(u_max)
    control_collocation = np.zeros((n_intervals, degree, 3))
    hamiltonian_collocation = np.zeros((n_intervals, degree))
    for k in range(n_intervals):
        for j in range(degree):
            _, u_j, h_j = rhs_fun(state_collocation[k, j], costate_collocation[k, j], mu, u_max_num, float(dv_eps))
            control_collocation[k, j] = np.asarray(u_j, dtype=float).reshape(3)
            hamiltonian_collocation[k, j] = float(h_j)

    target_final = target_mee_at_time(tf_opt, mee_target_epoch, mu)
    if options.terminal_angle_mode == "branch":
        target_final = np.array(target_final, dtype=float, copy=True)
        target_final[5] += 2.0 * np.pi * int(options.longitude_branch)
    endpoint = profile["state"][-1] - target_final
    endpoint[5] = wrap_angle(endpoint[5])
    node_endpoint = state_nodes[-1] - target_final
    node_endpoint[5] = wrap_angle(node_endpoint[5])
    target_dot = np.asarray(f0_fun()(target_final, mu), dtype=float).reshape(6)
    _, _, Hf = rhs_fun(state_nodes[-1], costate_nodes[-1], mu, u_max_num, float(dv_eps))
    hamiltonian_final = float(Hf)
    lambda_target_dot = float(np.dot(costate_nodes[-1], target_dot))
    delta_l_raw = float(state_nodes[-1, 5] - target_final[5])
    delta_l_wrapped = float(wrap_angle(delta_l_raw))
    sin_delta_l = float(np.sin(delta_l_raw))
    cos_delta_l = float(np.cos(delta_l_raw))
    if abs(cos_delta_l) > 1e-12:
        terminal_multiplier_sin = costate_nodes[-1].copy()
        terminal_multiplier_sin[5] = costate_nodes[-1, 5] / cos_delta_l
    else:
        terminal_multiplier_sin = np.full(6, np.nan)
    psi_t_sin = np.array(
        [
            -target_dot[0],
            -target_dot[1],
            -target_dot[2],
            -target_dot[3],
            -target_dot[4],
            -cos_delta_l * target_dot[5],
        ],
        dtype=float,
    )
    transversality_minus = hamiltonian_final - lambda_target_dot
    transversality_plus = hamiltonian_final + lambda_target_dot
    transversality_sin_lambda_eq_psix_nu = float(
        hamiltonian_final + np.dot(terminal_multiplier_sin, psi_t_sin)
    )
    transversality_sin_lambda_eq_minus_psix_nu = float(
        hamiltonian_final - np.dot(terminal_multiplier_sin, psi_t_sin)
    )
    u_norm = np.linalg.norm(profile["control"], axis=1)
    print(hamiltonian_final)
    print(np.dot(terminal_multiplier_sin, psi_t_sin))
    return {
        "success": bool(sol.success and profile["success"]),
        "solver_success": bool(sol.success),
        "integrator_success": bool(profile["success"]),
        "message": sol.message,
        "method": f"indirect_{objective}_radau_root",
        "objective": objective,
        "dv_eps": float(dv_eps),
        "free_time": bool(free_time),
        "terminal_angle_mode": options.terminal_angle_mode,
        "longitude_branch": int(options.longitude_branch),
        "costate_seed_source": costate_seed,
        "initial_guess_source": initial_guess_source,
        "t_transfer": tf_opt,
        "t_transfer_guess": tf_guess,
        "tf_min": tf_min,
        "tf_max": tf_max,
        "mee0": mee0,
        "mee_target_epoch": mee_target_epoch,
        "target_mee": target_final,
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
        "profile": profile,
        "node_endpoint_error": node_endpoint,
        "node_endpoint_error_norm": float(np.linalg.norm(node_endpoint)),
        "endpoint_error": endpoint,
        "endpoint_error_norm": float(np.linalg.norm(endpoint)),
        "transversality": transversality_minus,
        "transversality_minus": transversality_minus,
        "transversality_plus": transversality_plus,
        "transversality_sin_lambda_eq_psix_nu": transversality_sin_lambda_eq_psix_nu,
        "transversality_sin_lambda_eq_minus_psix_nu": transversality_sin_lambda_eq_minus_psix_nu,
        "hamiltonian_final": hamiltonian_final,
        "lambda_target_dot": lambda_target_dot,
        "target_dot_final": target_dot,
        "lambda_final": costate_nodes[-1],
        "delta_l_raw": delta_l_raw,
        "delta_l_wrapped": delta_l_wrapped,
        "sin_delta_l": sin_delta_l,
        "cos_delta_l": cos_delta_l,
        "terminal_multiplier_sin": terminal_multiplier_sin,
        "psi_t_sin": psi_t_sin,
        "energy": 0.5 * trapz(u_norm**2, profile["t"]),
        "dv": trapz(u_norm, profile["t"]),
        "max_u": float(np.max(u_norm)),
        "residual": res_final,
        "residual_norm": float(np.linalg.norm(res_final)),
        "residual_inf": float(np.linalg.norm(res_final, ord=np.inf)),
        "n_intervals": n_intervals,
        "degree": degree,
        "root_result": sol,
        "radau_seed_result": radau_result,
    }

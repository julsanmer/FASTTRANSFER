"""
Indirect MEE boundary-value solver using scipy.solve_bvp with analytic Jacobians.

The independent variable is the normalized time s in [0, 1].  For fixed final
time, the BVP has 12 states y = [x, lambda].  For free final time, tf is passed
as a solve_bvp unknown parameter and the boundary conditions include the
free-time transversality condition.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
from scipy.integrate import solve_bvp

from .optimization_MEE_Radaucollocation import (
    DEFAULT_MEE0,
    DEFAULT_MEE_TARGET_EPOCH,
    DEFAULT_MU,
    infer_longitude_branch_from_solution,
    target_mee_at_time,
)
from .optimization_MEE_indirect import (
    estimate_costates_from_radau,
    f0_fun,
    indirect_fun,
    states_for_radau_seed,
    trapz,
    wrap_angle,
)
from .orbit_utils import kepler_coast_sym


@dataclass
class SolveBVPJacOptions:
    tol: float = 1e-5
    bc_tol: float = 1e-7
    max_nodes: int = 20000
    n_eval: int = 1000
    residual_tol: float = 1e-5
    verbose: int = 1
    use_jacobians: bool = True
    longitude_branch: int | None = None


_RHS_CACHE: dict[tuple[str], tuple[ca.Function, ca.Function, ca.Function]] = {}
_BC_CACHE: dict[tuple[str, bool], tuple[ca.Function, ca.Function]] = {}


def _u_max_value(u_max: float | None) -> float:
    return -1.0 if u_max is None else float(u_max)


def _interp_rows(tau_grid: np.ndarray, values: np.ndarray, tau_query: np.ndarray) -> np.ndarray:
    tau_grid = np.asarray(tau_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    tau_query = np.asarray(tau_query, dtype=float)
    return np.vstack([np.interp(tau_query, tau_grid, values[:, idx]) for idx in range(values.shape[1])]).T


def _strict_unit_mesh(mesh: np.ndarray, min_step: float = 1e-8) -> np.ndarray:
    values = np.asarray(mesh, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    values = np.sort(values)
    values = values[(values >= -float(min_step)) & (values <= 1.0 + float(min_step))]
    values = np.clip(values, 0.0, 1.0)

    cleaned = [0.0]
    for value in values:
        value = float(value)
        if value <= cleaned[-1] + float(min_step):
            continue
        cleaned.append(value)
    if cleaned[-1] < 1.0 - float(min_step):
        cleaned.append(1.0)
    else:
        cleaned[-1] = 1.0

    if len(cleaned) < 3:
        return np.linspace(0.0, 1.0, 3)
    return np.asarray(cleaned, dtype=float)


def _sigmoid(value: float) -> float:
    value = float(value)
    if value >= 0.0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))


def _tf_to_q(tf: float, tf_min: float, tf_max: float) -> float:
    alpha = (float(tf) - float(tf_min)) / (float(tf_max) - float(tf_min))
    alpha = float(np.clip(alpha, 1e-10, 1.0 - 1e-10))
    return float(np.log(alpha / (1.0 - alpha)))


def _q_to_tf(q: float, tf_min: float, tf_max: float) -> float:
    alpha = _sigmoid(float(q))
    return float(tf_min + (tf_max - tf_min) * alpha)


def _dtf_dq(q: float, tf_min: float, tf_max: float) -> float:
    alpha = _sigmoid(float(q))
    return float((tf_max - tf_min) * alpha * (1.0 - alpha))


def _initial_guess_from_radau(
    radau_result: dict,
    objective: str,
    dv_eps: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    tau_x, states = states_for_radau_seed(radau_result)
    tau_lam, lambdas = estimate_costates_from_radau(
        radau_result,
        mu=mu,
        objective=objective,
        dv_eps=float(dv_eps),
    )
    mesh = _strict_unit_mesh(np.concatenate([tau_x, tau_lam]))
    state_guess = _interp_rows(tau_x, states, mesh)
    lambda_guess = _interp_rows(tau_lam, lambdas, mesh)
    return mesh, np.vstack([state_guess.T, lambda_guess.T])


def _build_rhs_functions(objective: str) -> tuple[ca.Function, ca.Function, ca.Function]:
    key = (objective,)
    if key in _RHS_CACHE:
        return _RHS_CACHE[key]

    y = ca.MX.sym("y", 12)
    tf = ca.MX.sym("tf")
    mu = ca.MX.sym("mu")
    u_max = ca.MX.sym("u_max")
    dv_eps = ca.MX.sym("dv_eps")

    rhs_fun = indirect_fun(objective)
    ydot, control, hamiltonian = rhs_fun(y[:6], y[6:], mu, u_max, dv_eps)
    f_scaled = tf * ydot
    jy = ca.jacobian(f_scaled, y)
    jtf = ca.jacobian(f_scaled, tf)

    f = ca.Function(f"solve_bvp_{objective}_fun", [y, tf, mu, u_max, dv_eps], [f_scaled])
    fjac = ca.Function(f"solve_bvp_{objective}_fun_jac", [y, tf, mu, u_max, dv_eps], [jy])
    fpjac = ca.Function(f"solve_bvp_{objective}_fun_pjac", [y, tf, mu, u_max, dv_eps], [jtf])
    _RHS_CACHE[key] = (f, fjac, fpjac)
    return _RHS_CACHE[key]


def _build_bc_functions(objective: str, free_time: bool) -> tuple[ca.Function, ca.Function]:
    key = (objective, bool(free_time))
    if key in _BC_CACHE:
        return _BC_CACHE[key]

    ya = ca.MX.sym("ya", 12)
    yb = ca.MX.sym("yb", 12)
    tf = ca.MX.sym("tf")
    mee0 = ca.MX.sym("mee0", 6)
    mee_target_epoch = ca.MX.sym("mee_target_epoch", 6)
    mu = ca.MX.sym("mu")
    u_max = ca.MX.sym("u_max")
    dv_eps = ca.MX.sym("dv_eps")
    branch = ca.MX.sym("branch")

    target = kepler_coast_sym(mee_target_epoch, tf, mu)
    target = ca.vertcat(target[0], target[1], target[2], target[3], target[4], target[5] + 2.0 * np.pi * branch)
    residual = [ya[:6] - mee0, yb[:6] - target]

    if free_time:
        rhs_fun = indirect_fun(objective)
        target_dot = ca.vertcat(ca.MX(f0_fun()(target, mu)))
        _, _, hamiltonian = rhs_fun(yb[:6], yb[6:], mu, u_max, dv_eps)
        residual.append(ca.vertcat(hamiltonian - ca.dot(yb[6:], target_dot)))

    bc = ca.vertcat(*residual)
    vars_all = ca.vertcat(ya, yb, tf)
    jac = ca.jacobian(bc, vars_all)
    bc_fun = ca.Function(
        f"solve_bvp_{objective}_{'free' if free_time else 'fixed'}_bc",
        [ya, yb, tf, mee0, mee_target_epoch, mu, u_max, dv_eps, branch],
        [bc],
    )
    bc_jac_fun = ca.Function(
        f"solve_bvp_{objective}_{'free' if free_time else 'fixed'}_bc_jac",
        [ya, yb, tf, mee0, mee_target_epoch, mu, u_max, dv_eps, branch],
        [jac],
    )
    _BC_CACHE[key] = (bc_fun, bc_jac_fun)
    return _BC_CACHE[key]


def _evaluate_profile(sol, tf: float, mu: float, objective: str, dv_eps: float, u_max: float | None, n_eval: int) -> dict:
    s_eval = np.linspace(0.0, 1.0, int(n_eval))
    y_eval = sol.sol(s_eval).T
    rhs_fun = indirect_fun(objective)
    u_max_num = _u_max_value(u_max)
    controls = np.zeros((len(s_eval), 3))
    hamiltonian = np.zeros(len(s_eval))
    for idx, row in enumerate(y_eval):
        _, u_i, h_i = rhs_fun(row[:6], row[6:], mu, u_max_num, float(dv_eps))
        controls[idx] = np.asarray(u_i, dtype=float).reshape(3)
        hamiltonian[idx] = float(h_i)
    return {
        "t": s_eval * float(tf),
        "s": s_eval,
        "state": y_eval[:, :6],
        "costate": y_eval[:, 6:],
        "control": controls,
        "hamiltonian": hamiltonian,
        "objective": objective,
        "dv_eps": float(dv_eps),
    }


def solve_indirect_bvp_jac(
    mee0: np.ndarray = DEFAULT_MEE0,
    mee_target_epoch: np.ndarray = DEFAULT_MEE_TARGET_EPOCH,
    mu: float = DEFAULT_MU,
    tf_guess: float = 1.0,
    u_max: float | None = None,
    objective: str = "energy",
    dv_eps: float = 1e-6,
    free_time: bool = False,
    tf_min: float | None = None,
    tf_max: float | None = None,
    s_mesh: np.ndarray | None = None,
    y_guess: np.ndarray | None = None,
    options: SolveBVPJacOptions | None = None,
) -> dict:
    if objective not in {"energy", "dv"}:
        raise ValueError("objective must be 'energy' or 'dv'")
    if objective == "dv" and u_max is None:
        raise ValueError("Indirect delta-v BVP requires u_max")
    if options is None:
        options = SolveBVPJacOptions()

    mee0 = np.asarray(mee0, dtype=float).reshape(6)
    mee_target_epoch = np.asarray(mee_target_epoch, dtype=float).reshape(6)
    tf_guess = float(tf_guess)
    if tf_min is None:
        tf_min = max(1e-3, 0.5 * tf_guess)
    if tf_max is None:
        tf_max = max(float(tf_min) + 1e-3, 1.5 * tf_guess)
    tf_min = float(tf_min)
    tf_max = float(tf_max)
    tf_guess = float(np.clip(tf_guess, tf_min + 1e-10, tf_max - 1e-10))
    u_max_num = _u_max_value(u_max)
    branch = int(0 if options.longitude_branch is None else options.longitude_branch)

    if s_mesh is None:
        s_mesh = np.linspace(0.0, 1.0, 50)
    s_mesh = _strict_unit_mesh(s_mesh)

    if y_guess is None:
        target = target_mee_at_time(tf_guess, mee_target_epoch, mu)
        target[5] += 2.0 * np.pi * branch
        states = (1.0 - s_mesh[:, None]) * mee0[None, :] + s_mesh[:, None] * target[None, :]
        y_guess = np.vstack([states.T, np.zeros_like(states.T)])
    y_guess = np.asarray(y_guess, dtype=float)
    if y_guess.shape[1] != s_mesh.size:
        source_mesh = _strict_unit_mesh(np.linspace(0.0, 1.0, y_guess.shape[1]))
        y_guess = np.vstack([np.interp(s_mesh, source_mesh, y_guess[row]) for row in range(y_guess.shape[0])])

    h_mesh = np.diff(s_mesh)
    if not np.all(h_mesh > 0.0):
        raise ValueError("solve_bvp mesh must be strictly increasing")

    rhs_fun, rhs_jac_fun, rhs_pjac_fun = _build_rhs_functions(objective)
    bc_fun, bc_jac_fun = _build_bc_functions(objective, bool(free_time))

    def _tf(p):
        return _q_to_tf(float(p[0]), tf_min, tf_max) if free_time else tf_guess

    def _tf_p_derivative(p):
        return _dtf_dq(float(p[0]), tf_min, tf_max) if free_time else 0.0

    def fun(_s, y, p=None):
        tf = _tf(p)
        out = np.empty_like(y)
        for idx in range(y.shape[1]):
            out[:, idx] = np.asarray(rhs_fun(y[:, idx], tf, mu, u_max_num, float(dv_eps)), dtype=float).reshape(12)
        return out

    def fun_jac(_s, y, p=None):
        tf = _tf(p)
        df_dy = np.empty((12, 12, y.shape[1]))
        df_dp = np.empty((12, 1, y.shape[1]))
        for idx in range(y.shape[1]):
            df_dy[:, :, idx] = np.asarray(rhs_jac_fun(y[:, idx], tf, mu, u_max_num, float(dv_eps)), dtype=float)
            df_dp[:, 0, idx] = (
                np.asarray(rhs_pjac_fun(y[:, idx], tf, mu, u_max_num, float(dv_eps)), dtype=float).reshape(12)
                * _tf_p_derivative(p)
            )
        if free_time:
            return df_dy, df_dp
        return df_dy

    def bc(ya, yb, p=None):
        tf = _tf(p)
        return np.asarray(
            bc_fun(ya, yb, tf, mee0, mee_target_epoch, mu, u_max_num, float(dv_eps), float(branch)),
            dtype=float,
        ).reshape(-1)

    def bc_jac(ya, yb, p=None):
        tf = _tf(p)
        jac = np.asarray(
            bc_jac_fun(ya, yb, tf, mee0, mee_target_epoch, mu, u_max_num, float(dv_eps), float(branch)),
            dtype=float,
        )
        dya = jac[:, :12]
        dyb = jac[:, 12:24]
        dp = jac[:, 24:25]
        if free_time:
            return dya, dyb, dp * _tf_p_derivative(p)
        return dya, dyb

    solve_kwargs = {
        "tol": float(options.tol),
        "bc_tol": float(options.bc_tol),
        "max_nodes": int(options.max_nodes),
        "verbose": int(options.verbose),
    }
    if options.use_jacobians:
        solve_kwargs["fun_jac"] = fun_jac
        solve_kwargs["bc_jac"] = bc_jac

    if free_time:
        q_guess = _tf_to_q(tf_guess, tf_min, tf_max)
        sol = solve_bvp(fun, bc, s_mesh, y_guess, p=np.array([q_guess], dtype=float), **solve_kwargs)
        tf_opt = _q_to_tf(float(sol.p[0]), tf_min, tf_max)
    else:
        sol = solve_bvp(fun, bc, s_mesh, y_guess, **solve_kwargs)
        tf_opt = tf_guess

    profile = _evaluate_profile(sol, tf_opt, mu, objective, float(dv_eps), u_max, int(options.n_eval))
    target_final = target_mee_at_time(tf_opt, mee_target_epoch, mu)
    target_final[5] += 2.0 * np.pi * branch
    endpoint = profile["state"][-1] - target_final
    endpoint[5] = wrap_angle(endpoint[5])
    bc_residual = bc(sol.y[:, 0], sol.y[:, -1], sol.p if free_time else None)
    u_norm = np.linalg.norm(profile["control"], axis=1)
    target_dot = np.asarray(f0_fun()(target_final, mu), dtype=float).reshape(6)
    lambda_final = profile["costate"][-1]
    hamiltonian_final = float(profile["hamiltonian"][-1])
    lambda_target_dot = float(np.dot(lambda_final, target_dot))
    delta_l_raw = float(profile["state"][-1, 5] - target_final[5])
    delta_l_wrapped = float(wrap_angle(delta_l_raw))
    sin_delta_l = float(np.sin(delta_l_raw))
    cos_delta_l = float(np.cos(delta_l_raw))
    transversality_minus = hamiltonian_final - lambda_target_dot
    transversality_plus = hamiltonian_final + lambda_target_dot

    return {
        "success": bool(sol.success and np.linalg.norm(bc_residual, ord=np.inf) <= float(options.residual_tol)),
        "solver_success": bool(sol.success),
        "message": sol.message,
        "status": int(sol.status),
        "niter": int(sol.niter),
        "method": "indirect_solve_bvp_analytic_jac",
        "objective": objective,
        "dv_eps": float(dv_eps),
        "free_time": bool(free_time),
        "t_transfer": tf_opt,
        "t_transfer_guess": tf_guess,
        "tf_min": tf_min,
        "tf_max": tf_max,
        "mee0": mee0,
        "mee_target_epoch": mee_target_epoch,
        "target_mee": target_final,
        "mu": float(mu),
        "u_max": u_max,
        "longitude_branch": branch,
        "profile": profile,
        "solution": sol,
        "endpoint_error": endpoint,
        "endpoint_error_norm": float(np.linalg.norm(endpoint)),
        "bvp_residual": bc_residual,
        "bvp_residual_norm": float(np.linalg.norm(bc_residual)),
        "bvp_residual_inf": float(np.linalg.norm(bc_residual, ord=np.inf)),
        "bvp_residual_tol": float(options.residual_tol),
        "hamiltonian_final": hamiltonian_final,
        "lambda_target_dot": lambda_target_dot,
        "target_dot_final": target_dot,
        "lambda_final": lambda_final,
        "transversality": transversality_minus,
        "transversality_minus": transversality_minus,
        "transversality_plus": transversality_plus,
        "delta_l_raw": delta_l_raw,
        "delta_l_wrapped": delta_l_wrapped,
        "sin_delta_l": sin_delta_l,
        "cos_delta_l": cos_delta_l,
        "energy": 0.5 * trapz(u_norm**2, profile["t"]),
        "dv": trapz(u_norm, profile["t"]),
        "max_u": float(np.max(u_norm)),
        "use_jacobians": bool(options.use_jacobians),
    }


def solve_indirect_bvp_from_radau_jac(
    radau_result: dict,
    objective: str = "energy",
    dv_eps: float = 1e-6,
    u_max: float | None = None,
    free_time: bool = False,
    options: SolveBVPJacOptions | None = None,
) -> dict:
    if options is None:
        options = SolveBVPJacOptions()
    mu = float(radau_result["mu"])
    branch = options.longitude_branch
    if branch is None:
        branch = int(radau_result.get("inferred_longitude_branch", 0))
        if "inferred_longitude_branch" not in radau_result:
            branch = infer_longitude_branch_from_solution(radau_result["state_nodes"], radau_result["target_mee"])
        options = SolveBVPJacOptions(
            tol=options.tol,
            bc_tol=options.bc_tol,
            max_nodes=options.max_nodes,
            n_eval=options.n_eval,
            residual_tol=options.residual_tol,
            verbose=options.verbose,
            use_jacobians=options.use_jacobians,
            longitude_branch=branch,
        )

    mesh, y_guess = _initial_guess_from_radau(
        radau_result,
        objective=objective,
        dv_eps=float(dv_eps),
        mu=mu,
    )
    result = solve_indirect_bvp_jac(
        mee0=np.asarray(radau_result["mee0"], dtype=float),
        mee_target_epoch=np.asarray(radau_result["mee_target_epoch"], dtype=float),
        mu=mu,
        tf_guess=float(radau_result["t_transfer"]),
        u_max=u_max,
        objective=objective,
        dv_eps=float(dv_eps),
        free_time=bool(free_time),
        tf_min=float(radau_result.get("tf_min", 0.5 * float(radau_result["t_transfer"]))),
        tf_max=float(radau_result.get("tf_max", 1.5 * float(radau_result["t_transfer"]))),
        s_mesh=mesh,
        y_guess=y_guess,
        options=options,
    )
    result["radau_seed_result"] = radau_result
    return result


def solve_indirect_bvp_from_bvp_jac(
    previous_result: dict,
    objective: str | None = None,
    dv_eps: float | None = None,
    u_max: float | None = None,
    free_time: bool = True,
    options: SolveBVPJacOptions | None = None,
) -> dict:
    if options is None:
        options = SolveBVPJacOptions()
    branch = int(previous_result["longitude_branch"] if options.longitude_branch is None else options.longitude_branch)
    if options.longitude_branch is None:
        options = SolveBVPJacOptions(
            tol=options.tol,
            bc_tol=options.bc_tol,
            max_nodes=options.max_nodes,
            n_eval=options.n_eval,
            residual_tol=options.residual_tol,
            verbose=options.verbose,
            use_jacobians=options.use_jacobians,
            longitude_branch=branch,
        )

    sol = previous_result["solution"]
    s_mesh = _strict_unit_mesh(sol.x)
    y_guess = sol.sol(s_mesh)

    result = solve_indirect_bvp_jac(
        mee0=np.asarray(previous_result["mee0"], dtype=float),
        mee_target_epoch=np.asarray(previous_result["mee_target_epoch"], dtype=float),
        mu=float(previous_result["mu"]),
        tf_guess=float(previous_result["t_transfer"]),
        u_max=previous_result["u_max"] if u_max is None else u_max,
        objective=str(previous_result["objective"] if objective is None else objective),
        dv_eps=float(previous_result["dv_eps"] if dv_eps is None else dv_eps),
        free_time=bool(free_time),
        tf_min=float(previous_result.get("tf_min", 0.5 * float(previous_result["t_transfer"]))),
        tf_max=float(previous_result.get("tf_max", 1.5 * float(previous_result["t_transfer"]))),
        s_mesh=s_mesh,
        y_guess=y_guess,
        options=options,
    )
    result["previous_bvp_result"] = previous_result
    return result

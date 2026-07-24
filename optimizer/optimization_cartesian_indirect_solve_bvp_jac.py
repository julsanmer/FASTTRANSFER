"""
Cartesian indirect low-thrust BVP solved with scipy.solve_bvp and CasADi Jacobians.

State y = [r, v, lambda_r, lambda_v].  This module currently targets the
minimum-energy problem with optional acceleration clipping.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
from scipy.integrate import solve_bvp

from .optimization_MEE_Radaucollocation import DEFAULT_MEE0, DEFAULT_MEE_TARGET_EPOCH, DEFAULT_MU, target_mee_at_time
from .optimization_MEE_indirect import controls_for_radau_seed, states_for_radau_seed, trapz
from .optimization_MEE_indirect_solve_bvp_jac import _dtf_dq, _q_to_tf, _strict_unit_mesh, _tf_to_q
from .orbit_utils import kepler_coast_sym, mee_to_rv_sym
from utils.utils import mee2rv


@dataclass
class CartesianBVPJacOptions:
    tol: float = 1e-5
    bc_tol: float = 1e-7
    max_nodes: int = 20000
    n_eval: int = 1000
    residual_tol: float = 1e-5
    verbose: int = 1
    use_jacobians: bool = True


_RHS_CACHE = None
_BC_CACHE: dict[bool, tuple[ca.Function, ca.Function]] = {}


def _u_max_value(u_max: float | None) -> float:
    return -1.0 if u_max is None else float(u_max)


def _interp_rows(tau_grid: np.ndarray, values: np.ndarray, tau_query: np.ndarray) -> np.ndarray:
    tau_grid = np.asarray(tau_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    tau_query = np.asarray(tau_query, dtype=float)
    return np.vstack([np.interp(tau_query, tau_grid, values[:, idx]) for idx in range(values.shape[1])]).T


def _rtn_to_cartesian(r: np.ndarray, v: np.ndarray, u_rtn: np.ndarray) -> np.ndarray:
    rhat = r / max(np.linalg.norm(r), 1e-12)
    hvec = np.cross(r, v)
    hhat = hvec / max(np.linalg.norm(hvec), 1e-12)
    that = np.cross(hhat, rhat)
    return u_rtn[0] * rhat + u_rtn[1] * that + u_rtn[2] * hhat


def _two_body_accel(r, mu):
    r_norm = ca.sqrt(ca.sumsqr(r) + 1e-18)
    return -mu * r / r_norm**3


def _target_cartesian_sym(mee_target_epoch, tf, mu):
    target_mee = kepler_coast_sym(mee_target_epoch, tf, mu)
    r_t, v_t = mee_to_rv_sym(target_mee, mu)
    a_t = _two_body_accel(r_t, mu)
    return ca.vertcat(r_t, v_t), ca.vertcat(v_t, a_t)


def _build_rhs_functions():
    global _RHS_CACHE
    if _RHS_CACHE is not None:
        return _RHS_CACHE

    x = ca.MX.sym("x", 6)
    lam = ca.MX.sym("lam", 6)
    y = ca.vertcat(x, lam)
    tf = ca.MX.sym("tf")
    mu = ca.MX.sym("mu")
    u_max = ca.MX.sym("u_max")
    r = x[0:3]
    v = x[3:6]
    lam_r = lam[0:3]
    lam_v = lam[3:6]

    raw_u = -lam_v
    raw_norm = ca.sqrt(ca.sumsqr(raw_u) + 1e-16)
    scale = ca.if_else(u_max > 0.0, ca.fmin(1.0, u_max / raw_norm), 1.0)
    u = scale * raw_u

    a_grav = _two_body_accel(r, mu)
    xdot = ca.vertcat(v, a_grav + u)
    running_cost = 0.5 * ca.sumsqr(u)
    hamiltonian = running_cost + ca.dot(lam_r, v) + ca.dot(lam_v, a_grav + u)
    lam_dot = -ca.jacobian(hamiltonian, x).T
    ydot = ca.vertcat(xdot, lam_dot)
    f_scaled = tf * ydot

    f = ca.Function("cart_indirect_energy_fun", [y, tf, mu, u_max], [f_scaled])
    fjac = ca.Function("cart_indirect_energy_fun_jac", [y, tf, mu, u_max], [ca.jacobian(f_scaled, y)])
    fpjac = ca.Function("cart_indirect_energy_fun_pjac", [y, tf, mu, u_max], [ca.jacobian(f_scaled, tf)])
    out = ca.Function("cart_indirect_energy_out", [y, mu, u_max], [ydot, u, hamiltonian])
    _RHS_CACHE = (f, fjac, fpjac, out)
    return _RHS_CACHE


def _build_bc_functions(free_time: bool):
    key = bool(free_time)
    if key in _BC_CACHE:
        return _BC_CACHE[key]

    ya = ca.MX.sym("ya", 12)
    yb = ca.MX.sym("yb", 12)
    tf = ca.MX.sym("tf")
    x0 = ca.MX.sym("x0", 6)
    mee_target_epoch = ca.MX.sym("mee_target_epoch", 6)
    mu = ca.MX.sym("mu")
    u_max = ca.MX.sym("u_max")

    target, target_dot = _target_cartesian_sym(mee_target_epoch, tf, mu)
    residual = [ya[0:6] - x0, yb[0:6] - target]
    if free_time:
        _, _, _, out_fun = _build_rhs_functions()
        _, _, hamiltonian = out_fun(yb, mu, u_max)
        residual.append(ca.vertcat(hamiltonian - ca.dot(yb[6:12], target_dot)))

    bc = ca.vertcat(*residual)
    vars_all = ca.vertcat(ya, yb, tf)
    jac = ca.jacobian(bc, vars_all)
    bc_fun = ca.Function("cart_indirect_bc_free" if free_time else "cart_indirect_bc_fixed",
                         [ya, yb, tf, x0, mee_target_epoch, mu, u_max], [bc])
    bc_jac_fun = ca.Function("cart_indirect_bc_jac_free" if free_time else "cart_indirect_bc_jac_fixed",
                             [ya, yb, tf, x0, mee_target_epoch, mu, u_max], [jac])
    _BC_CACHE[key] = (bc_fun, bc_jac_fun)
    return _BC_CACHE[key]


def initial_cartesian_guess_from_radau(radau_result: dict, mu: float) -> tuple[np.ndarray, np.ndarray]:
    tau_x, mee_states = states_for_radau_seed(radau_result)
    tau_u, controls_rtn = controls_for_radau_seed(radau_result)
    mesh = _strict_unit_mesh(np.concatenate([tau_x, tau_u]))
    mee_mesh = _interp_rows(tau_x, mee_states, mesh)
    rtn_mesh = _interp_rows(tau_u, controls_rtn, mesh)

    rv = np.zeros((len(mesh), 6))
    u_cart = np.zeros((len(mesh), 3))
    for idx, mee in enumerate(mee_mesh):
        r_i, v_i = mee2rv(mee, mu)
        rv[idx, :3] = r_i
        rv[idx, 3:] = v_i
        u_cart[idx] = _rtn_to_cartesian(r_i, v_i, rtn_mesh[idx])

    tf = float(radau_result["t_transfer"])
    t_mesh = mesh * tf
    lambda_v = -u_cart
    lambda_r = -np.gradient(lambda_v, t_mesh, axis=0, edge_order=1)
    y_guess = np.vstack([rv.T, lambda_r.T, lambda_v.T])
    return mesh, y_guess


def _evaluate_profile(sol, tf: float, mu: float, u_max: float | None, n_eval: int) -> dict:
    s_eval = np.linspace(0.0, 1.0, int(n_eval))
    y_eval = sol.sol(s_eval).T
    _, _, _, out_fun = _build_rhs_functions()
    u_max_num = _u_max_value(u_max)
    controls = np.zeros((len(s_eval), 3))
    hamiltonian = np.zeros(len(s_eval))
    for idx, row in enumerate(y_eval):
        _, u_i, h_i = out_fun(row, mu, u_max_num)
        controls[idx] = np.asarray(u_i, dtype=float).reshape(3)
        hamiltonian[idx] = float(h_i)
    return {
        "t": s_eval * float(tf),
        "s": s_eval,
        "state": y_eval[:, :6],
        "costate": y_eval[:, 6:],
        "control": controls,
        "hamiltonian": hamiltonian,
    }


def solve_cartesian_indirect_bvp_jac(
    x0: np.ndarray,
    mee_target_epoch: np.ndarray = DEFAULT_MEE_TARGET_EPOCH,
    mu: float = DEFAULT_MU,
    tf_guess: float = 1.0,
    tf_min: float | None = None,
    tf_max: float | None = None,
    u_max: float | None = None,
    free_time: bool = False,
    s_mesh: np.ndarray | None = None,
    y_guess: np.ndarray | None = None,
    options: CartesianBVPJacOptions | None = None,
) -> dict:
    if options is None:
        options = CartesianBVPJacOptions()
    x0 = np.asarray(x0, dtype=float).reshape(6)
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

    if s_mesh is None:
        s_mesh = np.linspace(0.0, 1.0, 50)
    s_mesh = _strict_unit_mesh(s_mesh)
    if y_guess is None:
        target_mee = target_mee_at_time(tf_guess, mee_target_epoch, mu)
        rf, vf = mee2rv(target_mee, mu)
        state_guess = (1.0 - s_mesh[:, None]) * x0[None, :] + s_mesh[:, None] * np.concatenate([rf, vf])[None, :]
        y_guess = np.vstack([state_guess.T, np.zeros((6, len(s_mesh)))])
    y_guess = np.asarray(y_guess, dtype=float)

    rhs_fun, rhs_jac_fun, rhs_pjac_fun, _ = _build_rhs_functions()
    bc_fun, bc_jac_fun = _build_bc_functions(bool(free_time))

    def _tf(p):
        return _q_to_tf(float(p[0]), tf_min, tf_max) if free_time else tf_guess

    def _tf_p_derivative(p):
        return _dtf_dq(float(p[0]), tf_min, tf_max) if free_time else 0.0

    def fun(_s, y, p=None):
        tf = _tf(p)
        out = np.empty_like(y)
        for idx in range(y.shape[1]):
            out[:, idx] = np.asarray(rhs_fun(y[:, idx], tf, mu, u_max_num), dtype=float).reshape(12)
        return out

    def fun_jac(_s, y, p=None):
        tf = _tf(p)
        df_dy = np.empty((12, 12, y.shape[1]))
        df_dp = np.empty((12, 1, y.shape[1]))
        for idx in range(y.shape[1]):
            df_dy[:, :, idx] = np.asarray(rhs_jac_fun(y[:, idx], tf, mu, u_max_num), dtype=float)
            df_dp[:, 0, idx] = (
                np.asarray(rhs_pjac_fun(y[:, idx], tf, mu, u_max_num), dtype=float).reshape(12)
                * _tf_p_derivative(p)
            )
        if free_time:
            return df_dy, df_dp
        return df_dy

    def bc(ya, yb, p=None):
        tf = _tf(p)
        return np.asarray(bc_fun(ya, yb, tf, x0, mee_target_epoch, mu, u_max_num), dtype=float).reshape(-1)

    def bc_jac(ya, yb, p=None):
        tf = _tf(p)
        jac = np.asarray(bc_jac_fun(ya, yb, tf, x0, mee_target_epoch, mu, u_max_num), dtype=float)
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
        sol = solve_bvp(fun, bc, s_mesh, y_guess, p=np.array([_tf_to_q(tf_guess, tf_min, tf_max)]), **solve_kwargs)
        tf_opt = _q_to_tf(float(sol.p[0]), tf_min, tf_max)
    else:
        sol = solve_bvp(fun, bc, s_mesh, y_guess, **solve_kwargs)
        tf_opt = tf_guess

    profile = _evaluate_profile(sol, tf_opt, mu, u_max, int(options.n_eval))
    target_mee = target_mee_at_time(tf_opt, mee_target_epoch, mu)
    rt, vt = mee2rv(target_mee, mu)
    target_state = np.concatenate([rt, vt])
    endpoint = profile["state"][-1] - target_state
    bc_residual = bc(sol.y[:, 0], sol.y[:, -1], sol.p if free_time else None)
    u_norm = np.linalg.norm(profile["control"], axis=1)
    target_dot = np.concatenate([vt, -mu * rt / max(np.linalg.norm(rt), 1e-12) ** 3])
    lambda_final = profile["costate"][-1]
    hamiltonian_final = float(profile["hamiltonian"][-1])
    lambda_target_dot = float(np.dot(lambda_final, target_dot))

    return {
        "success": bool(sol.success and np.linalg.norm(bc_residual, ord=np.inf) <= float(options.residual_tol)),
        "solver_success": bool(sol.success),
        "message": sol.message,
        "status": int(sol.status),
        "niter": int(sol.niter),
        "method": "cartesian_indirect_solve_bvp_analytic_jac",
        "objective": "energy",
        "free_time": bool(free_time),
        "t_transfer": tf_opt,
        "t_transfer_guess": tf_guess,
        "tf_min": tf_min,
        "tf_max": tf_max,
        "x0": x0,
        "mee_target_epoch": mee_target_epoch,
        "target_state": target_state,
        "target_mee": target_mee,
        "mu": float(mu),
        "u_max": u_max,
        "profile": profile,
        "solution": sol,
        "endpoint_error": endpoint,
        "endpoint_error_norm": float(np.linalg.norm(endpoint)),
        "bvp_residual": bc_residual,
        "bvp_residual_norm": float(np.linalg.norm(bc_residual)),
        "bvp_residual_inf": float(np.linalg.norm(bc_residual, ord=np.inf)),
        "energy": 0.5 * trapz(u_norm**2, profile["t"]),
        "dv": trapz(u_norm, profile["t"]),
        "max_u": float(np.max(u_norm)),
        "use_jacobians": bool(options.use_jacobians),
        "hamiltonian_final": hamiltonian_final,
        "lambda_target_dot": lambda_target_dot,
        "transversality_minus": hamiltonian_final - lambda_target_dot,
        "transversality_plus": hamiltonian_final + lambda_target_dot,
        "lambda_final": lambda_final,
        "target_dot_final": target_dot,
    }


def solve_cartesian_indirect_bvp_from_radau_jac(
    radau_result: dict,
    u_max: float | None = None,
    free_time: bool = False,
    options: CartesianBVPJacOptions | None = None,
) -> dict:
    mu = float(radau_result["mu"])
    mee0 = np.asarray(radau_result["mee0"], dtype=float)
    r0, v0 = mee2rv(mee0, mu)
    mesh, y_guess = initial_cartesian_guess_from_radau(radau_result, mu)
    result = solve_cartesian_indirect_bvp_jac(
        x0=np.concatenate([r0, v0]),
        mee_target_epoch=np.asarray(radau_result["mee_target_epoch"], dtype=float),
        mu=mu,
        tf_guess=float(radau_result["t_transfer"]),
        tf_min=float(radau_result.get("tf_min", 0.5 * float(radau_result["t_transfer"]))),
        tf_max=float(radau_result.get("tf_max", 1.5 * float(radau_result["t_transfer"]))),
        u_max=u_max,
        free_time=bool(free_time),
        s_mesh=mesh,
        y_guess=y_guess,
        options=options,
    )
    result["radau_seed_result"] = radau_result
    return result


def solve_cartesian_indirect_bvp_from_bvp_jac(
    previous_result: dict,
    free_time: bool = True,
    options: CartesianBVPJacOptions | None = None,
) -> dict:
    sol = previous_result["solution"]
    mesh = _strict_unit_mesh(sol.x)
    y_guess = sol.sol(mesh)
    result = solve_cartesian_indirect_bvp_jac(
        x0=np.asarray(previous_result["x0"], dtype=float),
        mee_target_epoch=np.asarray(previous_result["mee_target_epoch"], dtype=float),
        mu=float(previous_result["mu"]),
        tf_guess=float(previous_result["t_transfer"]),
        tf_min=float(previous_result["tf_min"]),
        tf_max=float(previous_result["tf_max"]),
        u_max=previous_result["u_max"],
        free_time=bool(free_time),
        s_mesh=mesh,
        y_guess=y_guess,
        options=options,
    )
    result["previous_bvp_result"] = previous_result
    return result

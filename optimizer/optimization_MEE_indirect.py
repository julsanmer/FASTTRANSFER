"""
Indirect MEE low-thrust solver using Pontryagin shooting.

This first implementation targets the smooth energy problem

    J = integral 0.5 * ||u||^2 dt

with MEE dynamics and RTN acceleration.  The optimal unconstrained control is

    u = -G(x)^T lambda

where xdot = f0(x) + G(x) u.  A Radau direct-collocation solution can be used to
build the initial costate guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .optimization_MEE_Radaucollocation import (
    DEFAULT_MEE0,
    DEFAULT_MEE_TARGET_EPOCH,
    DEFAULT_MU,
    mee_gauss_rhs_sym,
    target_mee_at_time,
)


_INDIRECT_FUNS = {}
_G_FUN = None
_F0_FUN = None


@dataclass
class ShootingOptions:
    max_nfev: int = 120
    rtol: float = 1e-9
    atol: float = 1e-11
    terminal_weight: float = 1.0
    transversality_weight: float = 1.0
    lambda_bound: float = 1e4
    residual_tol: float = 1e-6
    verbose: int = 1


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


def trapz(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _build_indirect_fun(objective: str):
    if objective not in {"energy", "dv"}:
        raise ValueError("objective must be 'energy' or 'dv'")

    x = ca.MX.sym("x", 6)
    lam = ca.MX.sym("lam", 6)
    mu = ca.MX.sym("mu")
    u_max = ca.MX.sym("u_max")
    dv_eps = ca.MX.sym("dv_eps")

    u_sym = ca.MX.sym("u", 3)
    f_u = mee_gauss_rhs_sym(x, u_sym, mu)
    G = ca.jacobian(f_u, u_sym)
    q = ca.mtimes(G.T, lam)

    if objective == "energy":
        raw_u = -q
        raw_norm = ca.sqrt(ca.sumsqr(raw_u) + 1e-16)
        scale = ca.if_else(u_max > 0.0, ca.fmin(1.0, u_max / raw_norm), 1.0)
        u = scale * raw_u
        running_cost = 0.5 * ca.sumsqr(u)
    else:
        q_norm = ca.sqrt(ca.sumsqr(q) + 1e-16)
        denom = ca.sqrt(ca.fmax(1.0 - q_norm**2, 1e-12))
        interior_mag = dv_eps * q_norm / denom
        bounded_mag = ca.if_else(q_norm < 1.0 - 1e-10, interior_mag, u_max)
        mag = ca.if_else(u_max > 0.0, ca.fmin(bounded_mag, u_max), interior_mag)
        u = -(mag / q_norm) * q
        running_cost = ca.sqrt(ca.sumsqr(u) + dv_eps**2) - dv_eps

    f = mee_gauss_rhs_sym(x, u, mu)
    H = running_cost + ca.dot(lam, f)
    lam_dot = -ca.jacobian(H, x).T
    y_dot = ca.vertcat(f, lam_dot)

    return ca.Function(
        f"mee_indirect_{objective}_rhs",
        [x, lam, mu, u_max, dv_eps],
        [y_dot, u, H],
    )


def _build_g_fun():
    x = ca.MX.sym("x", 6)
    mu = ca.MX.sym("mu")
    u = ca.MX.sym("u", 3)
    f = mee_gauss_rhs_sym(x, u, mu)
    G = ca.jacobian(f, u)
    return ca.Function("mee_control_jacobian", [x, mu], [G])


def _build_f0_fun():
    x = ca.MX.sym("x", 6)
    mu = ca.MX.sym("mu")
    f0 = mee_gauss_rhs_sym(x, ca.MX.zeros(3, 1), mu)
    return ca.Function("mee_unforced_rhs", [x, mu], [f0])


def indirect_fun(objective: str = "energy"):
    if objective not in _INDIRECT_FUNS:
        _INDIRECT_FUNS[objective] = _build_indirect_fun(objective)
    return _INDIRECT_FUNS[objective]


def g_fun():
    global _G_FUN
    if _G_FUN is None:
        _G_FUN = _build_g_fun()
    return _G_FUN


def f0_fun():
    global _F0_FUN
    if _F0_FUN is None:
        _F0_FUN = _build_f0_fun()
    return _F0_FUN


def states_for_radau_seed(radau_result: dict) -> tuple[np.ndarray, np.ndarray]:
    tau_nodes = np.asarray(radau_result["mesh_tau"], dtype=float)
    x_nodes = np.asarray(radau_result["state_nodes"], dtype=float).T
    tau_col = np.asarray(radau_result["collocation_tau"], dtype=float)
    x_col = np.asarray(radau_result["state_collocation"], dtype=float).transpose(0, 2, 1).reshape(-1, 6)

    tau = np.concatenate([tau_nodes, tau_col])
    states = np.vstack([x_nodes, x_col])
    order = np.argsort(tau)
    return tau[order], states[order]


def controls_for_radau_seed(radau_result: dict) -> tuple[np.ndarray, np.ndarray]:
    tau = np.asarray(radau_result["collocation_tau"], dtype=float)
    controls = np.asarray(radau_result["control_collocation"], dtype=float).transpose(0, 2, 1).reshape(-1, 3)
    order = np.argsort(tau)
    return tau[order], controls[order]


def estimate_costates_from_radau(
    radau_result: dict,
    mu: float = DEFAULT_MU,
    objective: str = "energy",
    dv_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate lambda(t) from the direct solution relation u = -G(x)^T lambda."""
    tau_u, controls = controls_for_radau_seed(radau_result)
    tau_x, states = states_for_radau_seed(radau_result)
    states_u = np.vstack([np.interp(tau_u, tau_x, states[:, idx]) for idx in range(6)]).T

    lambdas = np.zeros((len(tau_u), 6))
    G_eval = g_fun()
    for idx, (state, control) in enumerate(zip(states_u, controls)):
        G = np.array(G_eval(state, mu), dtype=float)
        A = G.T
        if objective == "energy":
            q_est = -control
        elif objective == "dv":
            q_est = -control / np.sqrt(float(np.dot(control, control)) + float(dv_eps) ** 2)
        else:
            raise ValueError("objective must be 'energy' or 'dv'")
        lambdas[idx] = np.linalg.lstsq(A, q_est, rcond=None)[0]

    return tau_u, lambdas


def lambda0_guess_from_radau(
    radau_result: dict,
    mu: float = DEFAULT_MU,
    fit_degree: int = 3,
    objective: str = "energy",
    dv_eps: float = 1e-6,
) -> np.ndarray:
    tau, lambdas = estimate_costates_from_radau(
        radau_result,
        mu=mu,
        objective=objective,
        dv_eps=dv_eps,
    )
    if len(tau) == 0:
        return np.zeros(6)

    degree = int(min(max(fit_degree, 0), len(tau) - 1))
    lam0 = np.zeros(6)
    for idx in range(6):
        coeff = np.polyfit(tau, lambdas[:, idx], degree)
        lam0[idx] = np.polyval(coeff, 0.0)

    return lam0


def _u_max_value(u_max: float | None) -> float:
    return -1.0 if u_max is None else float(u_max)


def integrate_indirect_energy(
    lambda0: np.ndarray,
    tf: float,
    mee0: np.ndarray = DEFAULT_MEE0,
    mu: float = DEFAULT_MU,
    u_max: float | None = None,
    objective: str = "energy",
    dv_eps: float = 1e-6,
    n_eval: int = 800,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> dict:
    lambda0 = np.asarray(lambda0, dtype=float).reshape(6)
    mee0 = np.asarray(mee0, dtype=float).reshape(6)
    y0 = np.concatenate([mee0, lambda0])
    if objective == "dv" and u_max is None:
        raise ValueError("Indirect delta-v shooting requires u_max")

    rhs_fun = indirect_fun(objective)
    u_max_num = _u_max_value(u_max)
    dv_eps = float(dv_eps)

    def rhs(_t, y):
        return np.array(rhs_fun(y[:6], y[6:], mu, u_max_num, dv_eps)[0], dtype=float).reshape(12)

    t_eval = np.linspace(0.0, float(tf), int(n_eval))
    sol = solve_ivp(
        rhs,
        (0.0, float(tf)),
        y0,
        method="DOP853",
        t_eval=t_eval,
        rtol=float(rtol),
        atol=float(atol),
    )

    controls = np.zeros((sol.y.shape[1], 3))
    H = np.zeros(sol.y.shape[1])
    for idx in range(sol.y.shape[1]):
        y_i = sol.y[:, idx]
        _, u_i, H_i = rhs_fun(y_i[:6], y_i[6:], mu, u_max_num, dv_eps)
        controls[idx] = np.array(u_i, dtype=float).reshape(3)
        H[idx] = float(H_i)

    return {
        "success": bool(sol.success),
        "message": sol.message,
        "t": sol.t,
        "state": sol.y[:6, :].T,
        "costate": sol.y[6:, :].T,
        "control": controls,
        "hamiltonian": H,
        "objective": objective,
        "dv_eps": dv_eps,
    }


def _state_scale(mee0: np.ndarray, target: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.abs(target - mee0), 1.0)
    scale[3] = max(scale[3], 0.05)
    scale[4] = max(scale[4], 0.05)
    scale[5] = max(abs(wrap_angle(target[5] - mee0[5])), 1.0)
    return scale


def _terminal_residual(
    final_state: np.ndarray,
    final_costate: np.ndarray,
    H_final: float,
    tf: float,
    mee0: np.ndarray,
    mee_target_epoch: np.ndarray,
    mu: float,
    include_transversality: bool,
    terminal_weight: float,
    transversality_weight: float,
) -> np.ndarray:
    target = target_mee_at_time(tf, mee_target_epoch, mu)
    state_res = final_state - target
    state_res[5] = wrap_angle(state_res[5])
    state_res = terminal_weight * state_res / _state_scale(mee0, target)

    if not include_transversality:
        return state_res

    target_dot = np.array(f0_fun()(target, mu), dtype=float).reshape(6)
    trans = H_final - float(np.dot(final_costate, target_dot))
    return np.concatenate([state_res, [transversality_weight * trans]])


def solve_indirect_energy_shooting(
    mee0: np.ndarray = DEFAULT_MEE0,
    mee_target_epoch: np.ndarray = DEFAULT_MEE_TARGET_EPOCH,
    mu: float = DEFAULT_MU,
    lambda0_guess: np.ndarray | None = None,
    tf_guess: float | None = None,
    tf_min: float | None = None,
    tf_max: float | None = None,
    u_max: float | None = None,
    objective: str = "energy",
    dv_eps: float = 1e-6,
    free_time: bool = True,
    options: ShootingOptions | None = None,
) -> dict:
    if options is None:
        options = ShootingOptions()

    mee0 = np.asarray(mee0, dtype=float).reshape(6)
    mee_target_epoch = np.asarray(mee_target_epoch, dtype=float).reshape(6)
    if objective == "dv" and u_max is None:
        raise ValueError("Indirect delta-v shooting requires u_max")

    if tf_guess is None:
        tf_guess = 1.0
    tf_guess = float(tf_guess)
    if tf_min is None:
        tf_min = max(1e-3, 0.5 * tf_guess)
    if tf_max is None:
        tf_max = max(float(tf_min) + 1e-3, 1.5 * tf_guess)

    if lambda0_guess is None:
        lambda0_guess = np.zeros(6)
    lambda0_guess = np.asarray(lambda0_guess, dtype=float).reshape(6)

    if free_time:
        z0 = np.concatenate([lambda0_guess, [tf_guess]])
        lower = np.concatenate([-options.lambda_bound * np.ones(6), [float(tf_min)]])
        upper = np.concatenate([options.lambda_bound * np.ones(6), [float(tf_max)]])
    else:
        z0 = lambda0_guess.copy()
        lower = -options.lambda_bound * np.ones(6)
        upper = options.lambda_bound * np.ones(6)

    last_profile = {}

    def residual(z):
        nonlocal last_profile
        lambda0 = np.asarray(z[:6], dtype=float)
        tf = float(z[6]) if free_time else tf_guess

        try:
            profile = integrate_indirect_energy(
                lambda0,
                tf,
                mee0=mee0,
                mu=mu,
                u_max=u_max,
                objective=objective,
                dv_eps=dv_eps,
                n_eval=250,
                rtol=options.rtol,
                atol=options.atol,
            )
        except Exception:
            return 1e6 * np.ones(7 if free_time else 6)

        last_profile = profile
        if not profile["success"]:
            return 1e5 * np.ones(7 if free_time else 6)

        return _terminal_residual(
            profile["state"][-1],
            profile["costate"][-1],
            profile["hamiltonian"][-1],
            tf,
            mee0,
            mee_target_epoch,
            mu,
            include_transversality=free_time,
            terminal_weight=options.terminal_weight,
            transversality_weight=options.transversality_weight,
        )

    lsq = least_squares(
        residual,
        z0,
        bounds=(lower, upper),
        max_nfev=int(options.max_nfev),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        x_scale="jac",
        verbose=int(options.verbose),
    )

    lambda0_opt = np.asarray(lsq.x[:6], dtype=float)
    tf_opt = float(lsq.x[6]) if free_time else tf_guess
    profile = integrate_indirect_energy(
        lambda0_opt,
        tf_opt,
        mee0=mee0,
        mu=mu,
        u_max=u_max,
        objective=objective,
        dv_eps=dv_eps,
        n_eval=1000,
        rtol=options.rtol,
        atol=options.atol,
    )
    residual_final = _terminal_residual(
        profile["state"][-1],
        profile["costate"][-1],
        profile["hamiltonian"][-1],
        tf_opt,
        mee0,
        mee_target_epoch,
        mu,
        include_transversality=free_time,
        terminal_weight=1.0,
        transversality_weight=1.0,
    )

    target_final = target_mee_at_time(tf_opt, mee_target_epoch, mu)
    endpoint = profile["state"][-1] - target_final
    endpoint[5] = wrap_angle(endpoint[5])
    u_norm = np.linalg.norm(profile["control"], axis=1)
    energy = 0.5 * trapz(u_norm**2, profile["t"])
    dv = trapz(u_norm, profile["t"])

    residual_norm = float(np.linalg.norm(residual_final))

    return {
        "success": bool(lsq.success and profile["success"] and residual_norm <= options.residual_tol),
        "optimizer_success": bool(lsq.success),
        "integrator_success": bool(profile["success"]),
        "message": lsq.message,
        "least_squares": lsq,
        "lambda0": lambda0_opt,
        "lambda0_guess": lambda0_guess,
        "t_transfer": tf_opt,
        "t_transfer_guess": tf_guess,
        "tf_min": float(tf_min),
        "tf_max": float(tf_max),
        "free_time": bool(free_time),
        "target_mee": target_final,
        "mee0": mee0,
        "mee_target_epoch": mee_target_epoch,
        "mu": float(mu),
        "u_max": u_max,
        "objective": objective,
        "dv_eps": float(dv_eps),
        "profile": profile,
        "endpoint_error": endpoint,
        "endpoint_error_norm": float(np.linalg.norm(endpoint)),
        "shooting_residual": residual_final,
        "shooting_residual_norm": residual_norm,
        "shooting_residual_tol": float(options.residual_tol),
        "energy": energy,
        "dv": dv,
        "max_u": float(np.max(u_norm)),
        "method": f"indirect_{objective}_shooting",
    }


def solve_indirect_energy_from_radau(
    radau_result: dict,
    mee0: np.ndarray | None = None,
    mee_target_epoch: np.ndarray | None = None,
    mu: float | None = None,
    tf_min: float | None = None,
    tf_max: float | None = None,
    u_max: float | None = None,
    objective: str = "energy",
    dv_eps: float = 1e-6,
    free_time: bool = True,
    options: ShootingOptions | None = None,
) -> dict:
    if mee0 is None:
        mee0 = np.asarray(radau_result["mee0"], dtype=float)
    if mee_target_epoch is None:
        mee_target_epoch = np.asarray(radau_result["mee_target_epoch"], dtype=float)
    if mu is None:
        mu = float(radau_result["mu"])

    lambda0_guess = lambda0_guess_from_radau(
        radau_result,
        mu=mu,
        objective=objective,
        dv_eps=dv_eps,
    )
    tf_guess = float(radau_result["t_transfer"])
    if tf_min is None:
        tf_min = float(radau_result.get("tf_min", max(1e-3, 0.5 * tf_guess)))
    if tf_max is None:
        tf_max = float(radau_result.get("tf_max", max(tf_min + 1e-3, 1.5 * tf_guess)))

    result = solve_indirect_energy_shooting(
        mee0=mee0,
        mee_target_epoch=mee_target_epoch,
        mu=mu,
        lambda0_guess=lambda0_guess,
        tf_guess=tf_guess,
        tf_min=tf_min,
        tf_max=tf_max,
        u_max=u_max,
        objective=objective,
        dv_eps=dv_eps,
        free_time=free_time,
        options=options,
    )
    result["radau_seed_result"] = radau_result
    return result

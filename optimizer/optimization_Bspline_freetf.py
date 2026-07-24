import ctypes
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.interpolate import BSpline

from .canonical_units import MU_CANONICAL, UA_AU_PER_YR2, UV_AU_PER_YR
from experiments.reference_metrics import evaluate_reference_metrics
from .helpers_Bspline import build_bspline_matrices, derivative_matrix, evaluate_cartesian_profile, fit_cartesian_seed, mee_profile
from .orbit_utils import get_kepler_substeps, kepler_coast_sym, mee_to_rv_sym
from .targets import DEFAULT_MEE0, DEFAULT_MEE_TARGET_EPOCH
from utils.utils import kepler_coast_np, mee2rv, rv2mee


DEFAULT_MU = MU_CANONICAL
AU_KM = 149_597_870.7
YEAR_S = 365.25 * 86_400.0
CANONICAL_DV_TO_KM_S = UV_AU_PER_YR * AU_KM / YEAR_S
CANONICAL_ACCEL_TO_M_S2 = UA_AU_PER_YR2 * AU_KM * 1000.0 / YEAR_S**2
IPOPT_LINEAR_SOLVERS = {"mumps", "ma27", "ma57", "ma77", "ma86", "ma97"}
_COINHSL_HANDLES: dict[str, ctypes.CDLL] = {}


def ipopt_linear_solver_options(linear_solver: str, coinhsl_library: str | None) -> dict:
    solver = str(linear_solver).lower()
    if solver not in IPOPT_LINEAR_SOLVERS:
        raise ValueError(f"Unsupported IPOPT linear solver: {linear_solver!r}")
    options = {"linear_solver": solver}
    if solver == "mumps":
        return options
    if not coinhsl_library:
        raise ValueError(f"IPOPT solver {solver} requires a CoinHSL library path")
    library = str(Path(coinhsl_library).expanduser().resolve())
    if not Path(library).is_file():
        raise FileNotFoundError(f"CoinHSL library not found: {library}")
    # Keep a global RTLD handle open. On macOS, allowing IPOPT's temporary
    # loader handle to unload CoinHSL can crash the process during shutdown.
    if library not in _COINHSL_HANDLES:
        _COINHSL_HANDLES[library] = ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
    options["hsllib"] = library
    return options


def build_bspline_objective_quadrature(
    mats,
    quadrature_order: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = int(quadrature_order)
    if order < 1:
        raise ValueError("quadrature_order must be >= 1")

    xi, wi = np.polynomial.legendre.leggauss(order)
    knots_unique = np.unique(np.asarray(mats.knots, dtype=float))
    tau_parts = []
    weight_parts = []
    for left, right in zip(knots_unique[:-1], knots_unique[1:]):
        a = max(float(left), 0.0)
        b = min(float(right), 1.0)
        if b <= a + 1e-14:
            continue
        tau_parts.append(0.5 * (b - a) * xi + 0.5 * (a + b))
        weight_parts.append(0.5 * (b - a) * wi)
    if not tau_parts:
        tau_parts = [0.5 * xi + 0.5]
        weight_parts = [0.5 * wi]

    tau = np.concatenate(tau_parts)
    weights = np.concatenate(weight_parts)
    d1 = derivative_matrix(mats.knots, mats.degree)
    d2 = derivative_matrix(mats.knots[1:-1], mats.degree - 1)
    b0 = BSpline.design_matrix(tau, mats.knots, mats.degree).toarray()
    b1 = BSpline.design_matrix(tau, mats.knots[1:-1], mats.degree - 1).toarray() @ d1
    b2 = BSpline.design_matrix(tau, mats.knots[2:-2], mats.degree - 2).toarray() @ d2 @ d1
    return tau, weights, b0, b1, b2


def integrate_cartesian_control_gauss(
    control_points: np.ndarray,
    mats,
    tf: float,
    mu: float,
    quadrature_order: int = 6,
) -> tuple[float, float]:
    _, weights, b0, b1, b2 = build_bspline_objective_quadrature(mats, quadrature_order)
    control_points = np.asarray(control_points, dtype=float)
    pos = b0 @ control_points
    acc = (b2 @ control_points) / (tf**2)
    radius = np.sqrt(np.sum(pos * pos, axis=1, keepdims=True) + 1e-12)
    u = acc + mu * pos / (radius**3)
    u_norm = np.linalg.norm(u, axis=1)
    dv = float(tf * np.sum(weights * u_norm))
    energy = float(tf * np.sum(weights * u_norm**2))
    return dv, energy


def integrate_cylindrical_control_gauss(
    control_points: np.ndarray,
    mats,
    tf: float,
    mu: float,
    quadrature_order: int = 6,
) -> tuple[float, float]:
    _, weights, b0, b1, b2 = build_bspline_objective_quadrature(mats, quadrature_order)
    control_points = np.asarray(control_points, dtype=float)
    q = b0 @ control_points
    qdot = (b1 @ control_points) / tf
    qddot = (b2 @ control_points) / (tf**2)
    u = np.zeros_like(q)
    for idx in range(len(weights)):
        pos, _, acc = cylindrical_position_velocity_accel_np(q[idx], qdot[idx], qddot[idx])
        radius = np.sqrt(np.sum(pos * pos) + 1e-12)
        u[idx] = acc + mu * pos / (radius**3)
    u_norm = np.linalg.norm(u, axis=1)
    dv = float(tf * np.sum(weights * u_norm))
    energy = float(tf * np.sum(weights * u_norm**2))
    return dv, energy


_REFERENCE_BSPLINE_BASIS_CACHE: dict[
    tuple[int, bytes, int], tuple[np.ndarray, np.ndarray, np.ndarray]
] = {}


def _bspline_basis_at_tau(mats, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=float)
    key = (int(mats.degree), np.asarray(mats.knots, dtype=float).tobytes(), id(tau))
    cached = _REFERENCE_BSPLINE_BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    d1 = derivative_matrix(mats.knots, mats.degree)
    d2 = derivative_matrix(mats.knots[1:-1], mats.degree - 1)
    b0 = BSpline.design_matrix(tau, mats.knots, mats.degree).toarray()
    b1 = BSpline.design_matrix(tau, mats.knots[1:-1], mats.degree - 1).toarray() @ d1
    b2 = BSpline.design_matrix(tau, mats.knots[2:-2], mats.degree - 2).toarray() @ d2 @ d1
    result = (b0, b1, b2)
    _REFERENCE_BSPLINE_BASIS_CACHE[key] = result
    return result


def integrate_cartesian_control_reference(
    control_points: np.ndarray,
    mats,
    tf: float,
    mu: float,
) -> dict:
    control_points = np.asarray(control_points, dtype=float)

    def acceleration_norm(tau: np.ndarray) -> np.ndarray:
        b0, _, b2 = _bspline_basis_at_tau(mats, tau)
        pos = b0 @ control_points
        acc = (b2 @ control_points) / tf**2
        radius_squared = np.sum(pos * pos, axis=1, keepdims=True)
        if np.any(radius_squared <= 0.0):
            raise ValueError("Reference trajectory reaches zero physical radius")
        radius = np.sqrt(radius_squared)
        control = acc + mu * pos / radius**3
        return np.linalg.norm(control, axis=1)

    return evaluate_reference_metrics(
        acceleration_norm,
        delta_v_scale_km_s=tf * CANONICAL_DV_TO_KM_S,
        acceleration_scale_m_s2=CANONICAL_ACCEL_TO_M_S2,
    ).as_dict()


def integrate_cylindrical_control_reference(
    control_points: np.ndarray,
    mats,
    tf: float,
    mu: float,
) -> dict:
    control_points = np.asarray(control_points, dtype=float)

    def acceleration_norm(tau: np.ndarray) -> np.ndarray:
        b0, b1, b2 = _bspline_basis_at_tau(mats, tau)
        q = b0 @ control_points
        qdot = (b1 @ control_points) / tf
        qddot = (b2 @ control_points) / tf**2
        control = np.zeros_like(q)
        for idx in range(len(tau)):
            pos, _, acc = cylindrical_position_velocity_accel_np(q[idx], qdot[idx], qddot[idx])
            radius_squared = float(np.sum(pos * pos))
            if radius_squared <= 0.0:
                raise ValueError("Reference trajectory reaches zero physical radius")
            radius = np.sqrt(radius_squared)
            control[idx] = acc + mu * pos / radius**3
        return np.linalg.norm(control, axis=1)

    return evaluate_reference_metrics(
        acceleration_norm,
        delta_v_scale_km_s=tf * CANONICAL_DV_TO_KM_S,
        acceleration_scale_m_s2=CANONICAL_ACCEL_TO_M_S2,
    ).as_dict()


def target_mee_at_time(t: float, mee_target_epoch: np.ndarray, mu: float) -> np.ndarray:
    return kepler_coast_np(mee_target_epoch, t, mu, n_iter=get_kepler_substeps())


def _row3(value) -> ca.MX:
    return ca.reshape(value, 1, 3)


def boundary_reduction_matrices(mats, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, int]:
    """Return right-inverse and nullspace matrices for endpoint constraints.

    The constant normalized-time endpoint equations are
    ``A @ C = [q0, tf*qdot0, qf, tf*qdotf]``.  Reducing with a numerical
    nullspace removes the boundary control-point variables without assuming
    a particular knot pattern beyond full row rank of ``A``.
    """

    a_eq = np.vstack([mats.b0_start, mats.b1_start, mats.b0_end, mats.b1_end])
    u, s, vt = np.linalg.svd(a_eq, full_matrices=True)
    if s.size == 0:
        raise ValueError("Empty B-spline endpoint constraint matrix")
    threshold = float(tol) * max(a_eq.shape) * float(s[0])
    rank = int(np.sum(s > threshold))
    if rank != a_eq.shape[0]:
        raise ValueError(
            "B-spline endpoint constraints are rank deficient; "
            f"rank={rank}, expected={a_eq.shape[0]}"
        )

    right_inverse = vt[:rank, :].T @ np.diag(1.0 / s[:rank]) @ u[:, :rank].T
    nullspace = vt[rank:, :].T
    return right_inverse, nullspace, rank


def build_reduced_control_points(
    opti: ca.Opti,
    mats,
    boundary_rhs,
    initial_control_points: np.ndarray,
    initial_boundary_rhs: np.ndarray,
) -> tuple[ca.MX, ca.MX | None, np.ndarray, int]:
    initial_control_points = np.asarray(initial_control_points, dtype=float)
    initial_boundary_rhs = np.asarray(initial_boundary_rhs, dtype=float)

    clamped = clamped_endpoint_reduction_coefficients(mats)
    if clamped is not None:
        n_ctrl = int(initial_control_points.shape[0])
        if initial_control_points.shape != (n_ctrl, 3):
            raise ValueError("initial_control_points must be a 2D array with three components")
        if initial_boundary_rhs.shape != (4, 3):
            raise ValueError(
                f"initial_boundary_rhs must have shape {(4, 3)}, "
                f"got {initial_boundary_rhs.shape}"
            )

        start_d0, start_d1, end_dprev, end_dlast = clamped
        n_free = max(n_ctrl - 4, 0)
        c0_initial = initial_boundary_rhs[0:1, :]
        c1_initial = (initial_boundary_rhs[1:2, :] - start_d0 * c0_initial) / start_d1
        c_last_initial = initial_boundary_rhs[2:3, :]
        c_prev_initial = (initial_boundary_rhs[3:4, :] - end_dlast * c_last_initial) / end_dprev
        if n_free > 0:
            free_initial = initial_control_points[2:-2, :]
            free_control_points = opti.variable(n_free, 3)
            opti.set_initial(free_control_points, free_initial)
            projected_initial = np.vstack([c0_initial, c1_initial, free_initial, c_prev_initial, c_last_initial])
        else:
            free_control_points = None
            projected_initial = np.vstack([c0_initial, c1_initial, c_prev_initial, c_last_initial])

        c0 = boundary_rhs[0:1, :]
        c1 = (boundary_rhs[1:2, :] - start_d0 * c0) / start_d1
        c_last = boundary_rhs[2:3, :]
        c_prev = (boundary_rhs[3:4, :] - end_dlast * c_last) / end_dprev
        if free_control_points is None:
            full_control_points = ca.vertcat(c0, c1, c_prev, c_last)
        else:
            full_control_points = ca.vertcat(c0, c1, free_control_points, c_prev, c_last)
        return full_control_points, free_control_points, projected_initial, n_free

    right_inverse, nullspace, rank = boundary_reduction_matrices(mats)
    expected_shape = (right_inverse.shape[0], 3)
    if initial_control_points.shape != expected_shape:
        raise ValueError(
            f"initial_control_points must have shape {expected_shape}, "
            f"got {initial_control_points.shape}"
        )
    if initial_boundary_rhs.shape != (rank, 3):
        raise ValueError(
            f"initial_boundary_rhs must have shape {(rank, 3)}, "
            f"got {initial_boundary_rhs.shape}"
        )

    boundary_part_initial = right_inverse @ initial_boundary_rhs
    free_initial = nullspace.T @ (initial_control_points - boundary_part_initial)
    projected_initial = boundary_part_initial + nullspace @ free_initial

    boundary_part = ca.mtimes(ca.DM(right_inverse), boundary_rhs)
    n_free = int(nullspace.shape[1])
    if n_free == 0:
        return boundary_part, None, projected_initial, n_free

    free_control_points = opti.variable(n_free, 3)
    opti.set_initial(free_control_points, free_initial)
    full_control_points = boundary_part + ca.mtimes(ca.DM(nullspace), free_control_points)
    return full_control_points, free_control_points, projected_initial, n_free


def clamped_endpoint_reduction_coefficients(mats, tol: float = 1e-12) -> tuple[float, float, float, float] | None:
    n_ctrl = int(mats.b0_start.shape[1])
    if n_ctrl < 4:
        return None

    start_pos = np.zeros((1, n_ctrl))
    start_pos[0, 0] = 1.0
    end_pos = np.zeros((1, n_ctrl))
    end_pos[0, -1] = 1.0
    if not np.allclose(mats.b0_start, start_pos, atol=tol, rtol=0.0):
        return None
    if not np.allclose(mats.b0_end, end_pos, atol=tol, rtol=0.0):
        return None

    start_nonzero = np.flatnonzero(np.abs(mats.b1_start.reshape(-1)) > tol)
    end_nonzero = np.flatnonzero(np.abs(mats.b1_end.reshape(-1)) > tol)
    if start_nonzero.tolist() != [0, 1]:
        return None
    if end_nonzero.tolist() != [n_ctrl - 2, n_ctrl - 1]:
        return None

    start_d0 = float(mats.b1_start[0, 0])
    start_d1 = float(mats.b1_start[0, 1])
    end_dprev = float(mats.b1_end[0, -2])
    end_dlast = float(mats.b1_end[0, -1])
    if abs(start_d1) <= tol or abs(end_dprev) <= tol:
        return None
    return start_d0, start_d1, end_dprev, end_dlast


def profile_true_longitude_delta_rev(profile: dict, mu: float) -> float:
    pos = np.asarray(profile.get("pos", []), dtype=float)
    vel = np.asarray(profile.get("vel", []), dtype=float)
    if pos.ndim != 2 or vel.ndim != 2 or len(pos) == 0 or len(pos) != len(vel):
        return float("nan")
    l_path = np.array([rv2mee(pos[i], vel[i], mu)[5] for i in range(len(pos))], dtype=float)
    l_path = np.unwrap(l_path)
    return float((l_path[-1] - l_path[0]) / (2.0 * np.pi))


def profile_xy_delta_rev(profile: dict) -> float:
    pos = np.asarray(profile.get("pos", []), dtype=float)
    if pos.ndim != 2 or len(pos) == 0 or pos.shape[1] < 2:
        return float("nan")
    theta = np.unwrap(np.arctan2(pos[:, 1], pos[:, 0]))
    return float((theta[-1] - theta[0]) / (2.0 * np.pi))


def cartesian_to_cylindrical_state(pos: np.ndarray, vel: np.ndarray, theta_reference: float | None = None) -> np.ndarray:
    pos = np.asarray(pos, dtype=float).reshape(3)
    vel = np.asarray(vel, dtype=float).reshape(3)
    x, y, z = pos
    vx, vy, vz = vel
    rho = max(float(np.hypot(x, y)), 1e-12)
    theta = float(np.arctan2(y, x))
    if theta_reference is not None:
        theta += 2.0 * np.pi * np.rint((float(theta_reference) - theta) / (2.0 * np.pi))
    rho_dot = float((x * vx + y * vy) / rho)
    theta_dot = float((x * vy - y * vx) / (rho**2))
    return np.array([rho, theta, z, rho_dot, theta_dot, vz], dtype=float)


def cylindrical_position_velocity_accel_np(q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho, theta, z = np.asarray(q, dtype=float).reshape(3)
    rho_dot, theta_dot, z_dot = np.asarray(qdot, dtype=float).reshape(3)
    rho_ddot, theta_ddot, z_ddot = np.asarray(qddot, dtype=float).reshape(3)
    c = np.cos(theta)
    s = np.sin(theta)
    pos = np.array([rho * c, rho * s, z], dtype=float)
    vel = np.array(
        [
            rho_dot * c - rho * theta_dot * s,
            rho_dot * s + rho * theta_dot * c,
            z_dot,
        ],
        dtype=float,
    )
    acc = np.array(
        [
            (rho_ddot - rho * theta_dot**2) * c - (2.0 * rho_dot * theta_dot + rho * theta_ddot) * s,
            (rho_ddot - rho * theta_dot**2) * s + (2.0 * rho_dot * theta_dot + rho * theta_ddot) * c,
            z_ddot,
        ],
        dtype=float,
    )
    return pos, vel, acc


def cylindrical_position_velocity_accel_sym(q, qdot, qddot):
    rho, theta, z = q[0], q[1], q[2]
    rho_dot, theta_dot, z_dot = qdot[0], qdot[1], qdot[2]
    rho_ddot, theta_ddot, z_ddot = qddot[0], qddot[1], qddot[2]
    c = ca.cos(theta)
    s = ca.sin(theta)
    pos = ca.vertcat(rho * c, rho * s, z)
    vel = ca.vertcat(
        rho_dot * c - rho * theta_dot * s,
        rho_dot * s + rho * theta_dot * c,
        z_dot,
    )
    acc = ca.vertcat(
        (rho_ddot - rho * theta_dot**2) * c - (2.0 * rho_dot * theta_dot + rho * theta_ddot) * s,
        (rho_ddot - rho * theta_dot**2) * s + (2.0 * rho_dot * theta_dot + rho * theta_ddot) * c,
        z_ddot,
    )
    return pos, vel, acc


def cylindrical_state_from_cartesian_sym(pos, vel, theta_offset: float = 0.0):
    x, y, z = pos[0], pos[1], pos[2]
    vx, vy, vz = vel[0], vel[1], vel[2]
    rho = ca.sqrt(x * x + y * y + 1e-12)
    theta = ca.atan2(y, x) + float(theta_offset)
    rho_dot = (x * vx + y * vy) / rho
    theta_dot = (x * vy - y * vx) / (rho**2)
    return ca.vertcat(rho, theta, z), ca.vertcat(rho_dot, theta_dot, vz)


def fit_cylindrical_seed(
    mats,
    mee0: np.ndarray,
    meef: np.ndarray,
    tf: float,
    seed_profile: str,
    mu: float,
    thetaf_unwrapped: float,
    n_fit: int = 900,
) -> np.ndarray:
    pos0, vel0 = mee2rv(mee0, mu)
    posf, velf = mee2rv(meef, mu)
    q0_state = cartesian_to_cylindrical_state(pos0, vel0)
    qf_state = cartesian_to_cylindrical_state(posf, velf, theta_reference=thetaf_unwrapped)

    tau_fit = np.linspace(0.0, 1.0, n_fit)
    mee_fit = mee_profile(tau_fit, mee0, meef, seed_profile)
    posvel_fit = [mee2rv(mee_fit[i], mu) for i in range(n_fit)]
    pos_fit = np.vstack([item[0] for item in posvel_fit])
    theta_fit = np.unwrap(np.arctan2(pos_fit[:, 1], pos_fit[:, 0]))
    theta_fit += q0_state[1] - theta_fit[0]
    theta_fit += tau_fit * (qf_state[1] - theta_fit[-1])
    q_fit = np.column_stack(
        [
            np.sqrt(np.sum(pos_fit[:, :2] * pos_fit[:, :2], axis=1)),
            theta_fit,
            pos_fit[:, 2],
        ]
    )
    q_fit[0, :] = q0_state[:3]
    q_fit[-1, :] = qf_state[:3]

    b_fit = BSpline.design_matrix(tau_fit, mats.knots, mats.degree).toarray()
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
            q0_state[:3].reshape(1, 3),
            q0_state[3:].reshape(1, 3),
            qf_state[:3].reshape(1, 3),
            qf_state[3:].reshape(1, 3),
        ]
    )

    ata = b_fit.T @ b_fit
    rhs_top = b_fit.T @ q_fit
    lhs = np.block(
        [
            [2.0 * ata, a_eq.T],
            [a_eq, np.zeros((a_eq.shape[0], a_eq.shape[0]))],
        ]
    )
    rhs = np.vstack([2.0 * rhs_top, b_eq])
    sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return sol[: b_fit.shape[1], :]


def evaluate_cylindrical_profile(control_points: np.ndarray, mats, tf: float, mu: float) -> dict:
    tau, b0, b1, b2 = mats.tau_fine, mats.b0_fine, mats.b1_fine, mats.b2_fine

    q = b0 @ control_points
    qdot = (b1 @ control_points) / tf
    qddot = (b2 @ control_points) / (tf**2)
    pos = np.zeros_like(q)
    vel = np.zeros_like(q)
    acc = np.zeros_like(q)
    for idx in range(len(tau)):
        pos[idx], vel[idx], acc[idx] = cylindrical_position_velocity_accel_np(q[idx], qdot[idx], qddot[idx])
    r = np.sqrt(np.sum(pos * pos, axis=1, keepdims=True) + 1e-12)
    u = acc + mu * pos / (r**3)
    return {
        "tau": tau,
        "t": tau * tf,
        "q_cyl": q,
        "qdot_cyl": qdot,
        "qddot_cyl": qddot,
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "u": u,
        "u_norm": np.linalg.norm(u, axis=1),
    }


def solve_free_tf_cartesian_bspline(
    tf_guess: float,
    tf_min: float | None = None,
    tf_max: float | None = None,
    mee0: np.ndarray | None = None,
    mee_target_epoch: np.ndarray | None = None,
    mu: float = DEFAULT_MU,
    n_ctrl: int = 24,
    degree: int = 3,
    n_fine: int = 800,
    seed_profile: str = "quintic",
    quadrature_order: int = 6,
    u_max: float | None = None,
    R_bound: float = 20.0,
    time_weight: float = 0.0,
    dv_eps: float = 1e-6,
    smoothness_weight: float = 0.0,
    endpoint_control_weight: float = 0.0,
    initial_control_points: np.ndarray | None = None,
    initial_tf: float | None = None,
    fixed_tf: bool = False,
    winding_target_rev: float | None = None,
    max_iter: int = 900,
    print_level: int = 5,
    linear_solver: str = "mumps",
    coinhsl_library: str | None = None,
) -> dict:
    if mee0 is None:
        mee0 = DEFAULT_MEE0
    if mee_target_epoch is None:
        mee_target_epoch = DEFAULT_MEE_TARGET_EPOCH

    mee0 = np.asarray(mee0, dtype=float)
    mee_target_epoch = np.asarray(mee_target_epoch, dtype=float)
    tf_guess = float(tf_guess)

    if fixed_tf:
        tf_min = tf_guess
        tf_max = tf_guess
    else:
        if tf_min is None:
            tf_min = max(1e-3, 0.25 * tf_guess)
        if tf_max is None:
            tf_max = max(float(tf_min) + 1e-3, 2.0 * tf_guess)
        tf_min = float(tf_min)
        tf_max = float(tf_max)
        if tf_min <= 0.0:
            raise ValueError("tf_min must be positive")
        if tf_max <= tf_min:
            raise ValueError("tf_max must be greater than tf_min")
        if not (tf_min <= tf_guess <= tf_max):
            raise ValueError("tf_guess must be inside [tf_min, tf_max]")

    # Seed with the target at tf_guess. The optimized target is recomputed
    # symbolically from the optimized tf inside the NLP.
    mee_target_guess = target_mee_at_time(tf_guess, mee_target_epoch, mu)
    pos0, vel0 = mee2rv(mee0, mu)
    posf_guess, velf_guess = mee2rv(mee_target_guess, mu)

    mats = build_bspline_matrices(n_ctrl, degree=degree, n_fine=n_fine)
    n_ctrl_actual = mats.b0_fine.shape[1]

    c_init = fit_cartesian_seed(
        mats,
        mee0,
        mee_target_guess,
        tf_guess,
        seed_profile,
        mu,
    )

    tf_init = tf_guess
    if initial_tf is not None:
        tf_init = float(initial_tf)
        tf_init = float(np.clip(tf_init, tf_min, tf_max))

    if initial_control_points is not None:
        c_init = np.asarray(initial_control_points, dtype=float)
        expected_shape = (n_ctrl_actual, 3)
        if c_init.shape != expected_shape:
            raise ValueError(
                f"initial_control_points must have shape {expected_shape}, "
                f"got {c_init.shape}"
            )

    mee0_dm = ca.DM(mee0.reshape(6, 1))
    pos0_sym, vel0_sym = mee_to_rv_sym(mee0_dm, mu)

    opti = ca.Opti()
    tf_fixed = float(tf_guess)
    tf = tf_fixed if fixed_tf else opti.variable()
    if not fixed_tf:
        opti.set_initial(tf, tf_init)
        opti.subject_to(opti.bounded(tf_min, tf, tf_max))

    mee_target_epoch_dm = ca.DM(mee_target_epoch.reshape(6, 1))
    mee_target_sym = kepler_coast_sym(mee_target_epoch_dm, tf, mu)
    posf_sym, velf_sym = mee_to_rv_sym(mee_target_sym, mu)

    mee_target_init = target_mee_at_time(tf_init, mee_target_epoch, mu)
    posf_init, velf_init = mee2rv(mee_target_init, mu)
    boundary_rhs = ca.vertcat(
        _row3(pos0_sym),
        _row3(tf * vel0_sym),
        _row3(posf_sym),
        _row3(tf * velf_sym),
    )
    boundary_rhs_init = np.vstack(
        [
            pos0.reshape(1, 3),
            (tf_init * vel0).reshape(1, 3),
            posf_init.reshape(1, 3),
            (tf_init * velf_init).reshape(1, 3),
        ]
    )
    C, C_free, c_init, n_free_control_points = build_reduced_control_points(
        opti,
        mats,
        boundary_rhs,
        c_init,
        boundary_rhs_init,
    )
    opti.subject_to(opti.bounded(-R_bound, C, R_bound))

    pos0_spline = ca.reshape(ca.mtimes(ca.DM(mats.b0_start), C), 3, 1)
    posf_spline = ca.reshape(ca.mtimes(ca.DM(mats.b0_end), C), 3, 1)

    _, objective_weights, objective_b0, objective_b1, objective_b2 = build_bspline_objective_quadrature(
        mats,
        int(quadrature_order),
    )

    cost = ca.MX(0)
    u_values = []
    for j in range(len(objective_weights)):
        b0j = ca.DM(objective_b0[j:j + 1, :])
        b1j = ca.DM(objective_b1[j:j + 1, :])
        b2j = ca.DM(objective_b2[j:j + 1, :])

        pos_j = ca.reshape(ca.mtimes(b0j, C), 3, 1)
        vel_j = ca.reshape(ca.mtimes(b1j, C), 3, 1) / tf
        acc_j = ca.reshape(ca.mtimes(b2j, C), 3, 1) / (tf**2)
        r_j = ca.sqrt(ca.sumsqr(pos_j) + 1e-12)
        u_j = acc_j + mu * pos_j / r_j**3
        u_values.append(u_j)
        eps = float(dv_eps)
        cost += tf * float(objective_weights[j]) * (ca.sqrt(ca.sumsqr(u_j) + eps**2) - eps)
        if u_max is not None:
            opti.subject_to(ca.sumsqr(u_j) / (u_max**2) <= 1.0)

    if smoothness_weight > 0.0 and len(u_values) > 1:
        smoothness_cost = ca.MX(0)
        dt_quad = tf / max(len(u_values) - 1, 1)
        for j in range(1, len(u_values)):
            smoothness_cost += ca.sumsqr(u_values[j] - u_values[j - 1]) / dt_quad
        cost += float(smoothness_weight) * smoothness_cost

    if endpoint_control_weight > 0.0 and len(u_values) > 1:
        cost += float(endpoint_control_weight) * tf * (ca.sumsqr(u_values[0]) + ca.sumsqr(u_values[-1]))

    if time_weight > 0.0:
        cost += float(time_weight) * (tf / tf_guess)

    opti.minimize(cost)

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
            "nlp_scaling_method": "gradient-based",
            "print_level": int(print_level),
        },
    }
    p_opts["ipopt"].update(ipopt_linear_solver_options(linear_solver, coinhsl_library))

    opti.solver("ipopt", p_opts)

    try:
        sol = opti.solve()
        success = True
        message = "Solve_Succeeded"
        c_opt = np.array(sol.value(C), dtype=float)
        tf_opt = tf_fixed if fixed_tf else float(sol.value(tf))
        obj_val = float(sol.value(cost))
    except RuntimeError as exc:
        success = False
        message = str(exc).splitlines()[-1]
        c_opt = np.array(opti.debug.value(C), dtype=float)
        tf_opt = tf_fixed if fixed_tf else float(opti.debug.value(tf))
        obj_val = float(opti.debug.value(cost))
    solver_stats = opti.stats()

    mee_target = target_mee_at_time(tf_opt, mee_target_epoch, mu)
    posf, velf = mee2rv(mee_target, mu)

    prof_fine = evaluate_cartesian_profile(c_opt, mats, tf_opt, mu)
    prof_init = evaluate_cartesian_profile(c_init, mats, tf_init, mu)
    dv_pure_fine, energy_pure_fine = integrate_cartesian_control_gauss(
        c_opt,
        mats,
        tf_opt,
        mu,
        int(quadrature_order),
    )
    reference_metrics = integrate_cartesian_control_reference(c_opt, mats, tf_opt, mu)

    endpoint = {
        "r0": float(np.linalg.norm(prof_fine["pos"][0] - pos0)),
        "v0": float(np.linalg.norm(prof_fine["vel"][0] - vel0)),
        "rf": float(np.linalg.norm(prof_fine["pos"][-1] - posf)),
        "vf": float(np.linalg.norm(prof_fine["vel"][-1] - velf)),
    }

    return {
        "success": success,
        "message": message,
        "objective": obj_val,
        "control_points": c_opt,
        "control_points_initial": c_init,
        "profile_fine": prof_fine,
        "profile_initial_fine": prof_init,
        "winding_coordinate": "mee_true_longitude",
        "winding_target_rev": float(winding_target_rev) if winding_target_rev is not None else float("nan"),
        "winding_sum_rev": profile_true_longitude_delta_rev(prof_fine, mu),
        "winding_error_rev": (
            profile_true_longitude_delta_rev(prof_fine, mu) - float(winding_target_rev)
            if winding_target_rev is not None
            else float("nan")
        ),
        "winding_fine_rev": profile_true_longitude_delta_rev(prof_fine, mu),
        "endpoint_errors": endpoint,
        "t_transfer": tf_opt,
        "t_transfer_guess": tf_guess,
        "t_transfer_initial": tf_init,
        "tf_min": tf_min,
        "tf_max": tf_max,
        "fixed_tf": bool(fixed_tf),
        "target_mee": mee_target,
        "target_mee_guess": mee_target_guess,
        "mee0": mee0,
        "mee_target_epoch": mee_target_epoch,
        "mu": float(mu),
        "u_max": u_max,
        "max_u_fine": float(np.max(prof_fine["u_norm"])),
        "max_u_initial_fine": float(np.max(prof_init["u_norm"])),
        "dv_pure_fine": dv_pure_fine,
        "delta_v_optimizer_canonical": dv_pure_fine,
        "energy_pure_fine": energy_pure_fine,
        **reference_metrics,
        "n_ctrl": int(n_ctrl),
        "degree": int(degree),
        "boundary_control_points_fixed": True,
        "boundary_control_points_eliminated": True,
        "n_free_control_points": int(n_free_control_points),
        "objective_type": "dv",
        "linear_solver": str(linear_solver).lower(),
        "ipopt_iterations": int(solver_stats.get("iter_count", -1)),
        "objective_quadrature": "gauss",
        "quadrature_order": int(quadrature_order),
        "time_weight": float(time_weight),
        "dv_eps": float(dv_eps),
        "smoothness_weight": float(smoothness_weight),
        "endpoint_control_weight": float(endpoint_control_weight),
        "seed_endpoint_guess": {
            "rf": posf_guess,
            "vf": velf_guess,
        },
    }


def solve_free_tf_cylindrical_bspline(
    tf_guess: float,
    tf_min: float | None = None,
    tf_max: float | None = None,
    mee0: np.ndarray | None = None,
    mee_target_epoch: np.ndarray | None = None,
    mee_target_final: np.ndarray | None = None,
    mu: float = DEFAULT_MU,
    n_ctrl: int = 24,
    degree: int = 3,
    n_fine: int = 800,
    seed_profile: str = "quintic",
    quadrature_order: int = 6,
    u_max: float | None = None,
    R_bound: float = 20.0,
    time_weight: float = 0.0,
    dv_eps: float = 1e-6,
    smoothness_weight: float = 0.0,
    endpoint_control_weight: float = 0.0,
    initial_control_points: np.ndarray | None = None,
    initial_tf: float | None = None,
    fixed_tf: bool = False,
    winding_target_rev: float | None = None,
    max_iter: int = 900,
    print_level: int = 5,
    linear_solver: str = "mumps",
    coinhsl_library: str | None = None,
) -> dict:
    if mee0 is None:
        mee0 = DEFAULT_MEE0
    if mee_target_epoch is None:
        mee_target_epoch = DEFAULT_MEE_TARGET_EPOCH

    mee0 = np.asarray(mee0, dtype=float)
    mee_target_epoch = np.asarray(mee_target_epoch, dtype=float)
    if mee_target_final is not None:
        mee_target_final = np.asarray(mee_target_final, dtype=float)
        if not fixed_tf:
            raise ValueError("mee_target_final is only valid for fixed_tf=True")
    tf_guess = float(tf_guess)

    if fixed_tf:
        tf_min = tf_guess
        tf_max = tf_guess
    else:
        if tf_min is None:
            tf_min = max(1e-3, 0.25 * tf_guess)
        if tf_max is None:
            tf_max = max(float(tf_min) + 1e-3, 2.0 * tf_guess)
        tf_min = float(tf_min)
        tf_max = float(tf_max)
        if tf_min <= 0.0:
            raise ValueError("tf_min must be positive")
        if tf_max <= tf_min:
            raise ValueError("tf_max must be greater than tf_min")
        if not (tf_min <= tf_guess <= tf_max):
            raise ValueError("tf_guess must be inside [tf_min, tf_max]")

    mee_target_guess = (
        mee_target_final.copy()
        if mee_target_final is not None
        else target_mee_at_time(tf_guess, mee_target_epoch, mu)
    )
    pos0, vel0 = mee2rv(mee0, mu)
    posf_guess, velf_guess = mee2rv(mee_target_guess, mu)
    q0_state = cartesian_to_cylindrical_state(pos0, vel0)

    thetaf_guess_wrapped = float(np.arctan2(posf_guess[1], posf_guess[0]))
    if winding_target_rev is not None:
        theta_ref = q0_state[1] + 2.0 * np.pi * float(winding_target_rev)
    else:
        theta_ref = q0_state[1]
    theta_offset = 2.0 * np.pi * float(np.rint((theta_ref - thetaf_guess_wrapped) / (2.0 * np.pi)))
    thetaf_unwrapped_guess = thetaf_guess_wrapped + theta_offset

    mats = build_bspline_matrices(n_ctrl, degree=degree, n_fine=n_fine)
    n_ctrl_actual = mats.b0_fine.shape[1]
    c_init = fit_cylindrical_seed(
        mats,
        mee0,
        mee_target_guess,
        tf_guess,
        seed_profile,
        mu,
        thetaf_unwrapped_guess,
    )

    tf_init = tf_guess
    if initial_tf is not None:
        tf_init = float(np.clip(float(initial_tf), tf_min, tf_max))

    if initial_control_points is not None:
        c_init = np.asarray(initial_control_points, dtype=float)
        expected_shape = (n_ctrl_actual, 3)
        if c_init.shape != expected_shape:
            raise ValueError(
                f"initial_control_points must have shape {expected_shape}, "
                f"got {c_init.shape}"
            )

    opti = ca.Opti()
    tf_fixed = float(tf_guess)
    tf = tf_fixed if fixed_tf else opti.variable()
    if not fixed_tf:
        opti.set_initial(tf, tf_init)
        opti.subject_to(opti.bounded(tf_min, tf, tf_max))

    if mee_target_final is not None:
        final_pos, final_vel = mee2rv(mee_target_final, mu)
        final_state = cartesian_to_cylindrical_state(
            final_pos,
            final_vel,
            theta_reference=float(np.arctan2(final_pos[1], final_pos[0])) + theta_offset,
        )
        qf_sym = ca.DM(final_state[:3].reshape(1, 3))
        qdotf_sym = ca.DM(final_state[3:].reshape(1, 3))
        mee_target_init = mee_target_final.copy()
    else:
        mee_target_epoch_dm = ca.DM(mee_target_epoch.reshape(6, 1))
        mee_target_sym = kepler_coast_sym(mee_target_epoch_dm, tf, mu)
        posf_sym, velf_sym = mee_to_rv_sym(mee_target_sym, mu)
        qf_sym, qdotf_sym = cylindrical_state_from_cartesian_sym(posf_sym, velf_sym, theta_offset)
        mee_target_init = target_mee_at_time(tf_init, mee_target_epoch, mu)
    posf_init, velf_init = mee2rv(mee_target_init, mu)
    thetaf_init_unwrapped = float(np.arctan2(posf_init[1], posf_init[0])) + theta_offset
    qf_init_state = cartesian_to_cylindrical_state(posf_init, velf_init, theta_reference=thetaf_init_unwrapped)
    boundary_rhs = ca.vertcat(
        _row3(ca.DM(q0_state[:3].reshape(3, 1))),
        _row3(tf * ca.DM(q0_state[3:].reshape(3, 1))),
        _row3(qf_sym),
        _row3(tf * qdotf_sym),
    )
    boundary_rhs_init = np.vstack(
        [
            q0_state[:3].reshape(1, 3),
            (tf_init * q0_state[3:]).reshape(1, 3),
            qf_init_state[:3].reshape(1, 3),
            (tf_init * qf_init_state[3:]).reshape(1, 3),
        ]
    )
    C, C_free, c_init, n_free_control_points = build_reduced_control_points(
        opti,
        mats,
        boundary_rhs,
        c_init,
        boundary_rhs_init,
    )

    rho_upper = max(float(R_bound), 2.0 * float(np.max(c_init[:, 0])), 1.0)
    theta_margin = max(2.0 * np.pi, abs(thetaf_unwrapped_guess - q0_state[1]) * 0.35)
    theta_min = min(float(q0_state[1]), float(thetaf_unwrapped_guess)) - theta_margin
    theta_max = max(float(q0_state[1]), float(thetaf_unwrapped_guess)) + theta_margin
    opti.subject_to(opti.bounded(1e-4, C[:, 0], rho_upper))
    opti.subject_to(opti.bounded(theta_min, C[:, 1], theta_max))
    opti.subject_to(opti.bounded(-R_bound, C[:, 2], R_bound))

    q0_spline = ca.reshape(ca.mtimes(ca.DM(mats.b0_start), C), 3, 1)
    qdot0_spline = ca.reshape(ca.mtimes(ca.DM(mats.b1_start), C), 3, 1) / tf
    qf_spline = ca.reshape(ca.mtimes(ca.DM(mats.b0_end), C), 3, 1)
    qdotf_spline = ca.reshape(ca.mtimes(ca.DM(mats.b1_end), C), 3, 1) / tf

    _, objective_weights, objective_b0, objective_b1, objective_b2 = build_bspline_objective_quadrature(
        mats,
        int(quadrature_order),
    )

    cost = ca.MX(0)
    u_values = []

    for j in range(len(objective_weights)):
        b0j = ca.DM(objective_b0[j:j + 1, :])
        b1j = ca.DM(objective_b1[j:j + 1, :])
        b2j = ca.DM(objective_b2[j:j + 1, :])

        q_j = ca.reshape(ca.mtimes(b0j, C), 3, 1)
        qdot_j = ca.reshape(ca.mtimes(b1j, C), 3, 1) / tf
        qddot_j = ca.reshape(ca.mtimes(b2j, C), 3, 1) / (tf**2)
        pos_j, _, acc_j = cylindrical_position_velocity_accel_sym(q_j, qdot_j, qddot_j)
        r_j = ca.sqrt(ca.sumsqr(pos_j) + 1e-12)
        u_j = acc_j + mu * pos_j / r_j**3
        u_values.append(u_j)

        eps = float(dv_eps)
        cost += tf * float(objective_weights[j]) * (ca.sqrt(ca.sumsqr(u_j) + eps**2) - eps)
        if u_max is not None:
            opti.subject_to(ca.sumsqr(u_j) / (u_max**2) <= 1.0)

    theta_delta = qf_spline[1] - q0_spline[1]
    winding_sum = theta_delta
    winding_error = None
    if winding_target_rev is not None:
        winding_error = theta_delta - 2.0 * np.pi * float(winding_target_rev)

    if smoothness_weight > 0.0 and len(u_values) > 1:
        smoothness_cost = ca.MX(0)
        dt_quad = tf / max(len(u_values) - 1, 1)
        for j in range(1, len(u_values)):
            smoothness_cost += ca.sumsqr(u_values[j] - u_values[j - 1]) / dt_quad
        cost += float(smoothness_weight) * smoothness_cost

    if endpoint_control_weight > 0.0 and len(u_values) > 1:
        cost += float(endpoint_control_weight) * tf * (ca.sumsqr(u_values[0]) + ca.sumsqr(u_values[-1]))

    if time_weight > 0.0:
        cost += float(time_weight) * (tf / tf_guess)

    opti.minimize(cost)

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
            "nlp_scaling_method": "gradient-based",
            "print_level": int(print_level),
        },
    }
    p_opts["ipopt"].update(ipopt_linear_solver_options(linear_solver, coinhsl_library))
    opti.solver("ipopt", p_opts)

    try:
        sol = opti.solve()
        success = True
        message = "Solve_Succeeded"
        c_opt = np.array(sol.value(C), dtype=float)
        tf_opt = tf_fixed if fixed_tf else float(sol.value(tf))
        obj_val = float(sol.value(cost))
        winding_sum_value = float(sol.value(winding_sum))
        winding_error_value = float(sol.value(winding_error)) if winding_error is not None else float("nan")
    except RuntimeError as exc:
        success = False
        message = str(exc).splitlines()[-1]
        c_opt = np.array(opti.debug.value(C), dtype=float)
        tf_opt = tf_fixed if fixed_tf else float(opti.debug.value(tf))
        obj_val = float(opti.debug.value(cost))
        winding_sum_value = float(opti.debug.value(winding_sum))
        winding_error_value = float(opti.debug.value(winding_error)) if winding_error is not None else float("nan")
    solver_stats = opti.stats()

    mee_target = (
        mee_target_final.copy()
        if mee_target_final is not None
        else target_mee_at_time(tf_opt, mee_target_epoch, mu)
    )
    posf, velf = mee2rv(mee_target, mu)

    prof_fine = evaluate_cylindrical_profile(c_opt, mats, tf_opt, mu)
    prof_init = evaluate_cylindrical_profile(c_init, mats, tf_init, mu)
    dv_pure_fine, energy_pure_fine = integrate_cylindrical_control_gauss(
        c_opt,
        mats,
        tf_opt,
        mu,
        int(quadrature_order),
    )
    reference_metrics = integrate_cylindrical_control_reference(c_opt, mats, tf_opt, mu)

    endpoint = {
        "r0": float(np.linalg.norm(prof_fine["pos"][0] - pos0)),
        "v0": float(np.linalg.norm(prof_fine["vel"][0] - vel0)),
        "rf": float(np.linalg.norm(prof_fine["pos"][-1] - posf)),
        "vf": float(np.linalg.norm(prof_fine["vel"][-1] - velf)),
    }

    return {
        "success": success,
        "message": message,
        "objective": obj_val,
        "control_points": c_opt,
        "control_points_initial": c_init,
        "control_point_coordinates": "cylindrical",
        "profile_fine": prof_fine,
        "profile_initial_fine": prof_init,
        "winding_coordinate": "cylindrical_theta_xy",
        "winding_target_rev": float(winding_target_rev) if winding_target_rev is not None else float("nan"),
        "winding_sum_rev": float(winding_sum_value / (2.0 * np.pi)) if np.isfinite(winding_sum_value) else float("nan"),
        "winding_error_rev": float(winding_error_value / (2.0 * np.pi)) if np.isfinite(winding_error_value) else float("nan"),
        "winding_fine_rev": profile_xy_delta_rev(prof_fine),
        "endpoint_errors": endpoint,
        "t_transfer": tf_opt,
        "t_transfer_guess": tf_guess,
        "t_transfer_initial": tf_init,
        "tf_min": tf_min,
        "tf_max": tf_max,
        "fixed_tf": bool(fixed_tf),
        "target_mee": mee_target,
        "target_mee_guess": mee_target_guess,
        "mee0": mee0,
        "mee_target_epoch": mee_target_epoch,
        "mee_target_final": mee_target_final if mee_target_final is not None else np.full(6, np.nan),
        "mu": float(mu),
        "u_max": u_max,
        "max_u_fine": float(np.max(prof_fine["u_norm"])),
        "max_u_initial_fine": float(np.max(prof_init["u_norm"])),
        "dv_pure_fine": dv_pure_fine,
        "delta_v_optimizer_canonical": dv_pure_fine,
        "energy_pure_fine": energy_pure_fine,
        **reference_metrics,
        "n_ctrl": int(n_ctrl),
        "degree": int(degree),
        "boundary_control_points_fixed": True,
        "boundary_control_points_eliminated": True,
        "n_free_control_points": int(n_free_control_points),
        "objective_type": "dv",
        "linear_solver": str(linear_solver).lower(),
        "ipopt_iterations": int(solver_stats.get("iter_count", -1)),
        "objective_quadrature": "gauss",
        "quadrature_order": int(quadrature_order),
        "time_weight": float(time_weight),
        "dv_eps": float(dv_eps),
        "smoothness_weight": float(smoothness_weight),
        "endpoint_control_weight": float(endpoint_control_weight),
        "seed_endpoint_guess": {
            "rf": posf_guess,
            "vf": velf_guess,
        },
    }

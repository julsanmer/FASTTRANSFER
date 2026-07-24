import numpy as np

from dataclasses import dataclass
from scipy.interpolate import BSpline

from utils.utils import mee2rv


@dataclass
class BSplineMatrices:
    degree: int
    knots: np.ndarray
    tau_fine: np.ndarray
    tau_start: float
    tau_end: float
    b0_fine: np.ndarray
    b1_fine: np.ndarray
    b2_fine: np.ndarray
    b0_start: np.ndarray
    b1_start: np.ndarray
    b0_end: np.ndarray
    b1_end: np.ndarray


def evaluate_cartesian_profile(control_points: np.ndarray,
                               mats,
                               tf: float,
                               mu: float) -> dict:
    tau, b0, b1, b2 = mats.tau_fine, mats.b0_fine, mats.b1_fine, mats.b2_fine

    # Compute positions-velocities-accelerations from splines
    pos_spline = b0 @ control_points
    vel_spline = (b1 @ control_points) / tf
    acc_spline = (b2 @ control_points) / (tf**2)

    # Orbital radius
    r = np.sqrt(np.sum(pos_spline * pos_spline, axis=1, keepdims=True) + 1e-12)
    u = acc_spline + mu * pos_spline / (r**3)

    return {
        "tau": tau,
        "t": tau * tf,
        "pos": pos_spline,
        "vel": vel_spline,
        "acc": acc_spline,
        "u": u,
        "u_norm": np.linalg.norm(u, axis=1),
    }


def fit_cartesian_seed(
    mats,
    mee0: np.ndarray,
    meef: np.ndarray,
    tf: float,
    seed_profile: str,
    mu: float,
    n_fit: int = 900,
) -> np.ndarray:
    # MEEs to position-velocity
    pos0, vel0 = mee2rv(mee0, mu)
    posf, velf = mee2rv(meef, mu)

    # Normalized grid in time
    tau_fit = np.linspace(0.0, 1.0, n_fit)

    # Do an a-priori MEE profile
    mee_fit = mee_profile(tau_fit, mee0, meef, seed_profile)

    # Position and velocities from MEE profile
    posvel_fit = [mee2rv(mee_fit[i], mu) for i in range(n_fit)]
    pos_fit = np.vstack([item[0] for item in posvel_fit])

    # B-spline basis matrix at tau_fit points
    b_fit = BSpline.design_matrix(tau_fit,
                                  mats.knots,
                                  mats.degree).toarray()

    # Endpoint constraints (a_eq @ C = b_eq)
    a_eq = np.vstack(
        [
            mats.b0_start,
            mats.b1_start / tf,
            mats.b0_end,
            mats.b1_end / tf
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

    # Normal equations of least-squares fit
    # (b_fit @ C - pos_fit).T (b_fit @ C - pos_fit)
    ata = b_fit.T @ b_fit
    rhs_top = b_fit.T @ pos_fit

    # Left and right sides least-squares.
    # We form it with Lagrange multipliers
    # [2 AᵀA   Eᵀ][C] = [2Aᵀ y]
    # [E        0][λ] = [b_eq]
    lhs = np.block(
        [
            [2.0*ata, a_eq.T],
            [a_eq, np.zeros((a_eq.shape[0], a_eq.shape[0]))],
        ]
    )
    rhs = np.vstack([2.0*rhs_top,
                     b_eq])

    # Solve the linear least-squares with boundary constraints
    sol = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    return sol[:b_fit.shape[1], :]


def mee_profile(
    tau: np.ndarray,
    mee0: np.ndarray,
    meef: np.ndarray,
    profile: str,
) -> np.ndarray:
    # Adimensional times
    tau = np.asarray(tau, dtype=float)

    # Initial and final MEE
    mee0 = np.asarray(mee0, dtype=float)
    meef = np.asarray(meef, dtype=float)

    if profile == "linear":
        # Linear profile
        s = tau
    elif profile == "cubic":
        # Cubic profile
        s = 3.0*tau**2 - 2.0*tau**3
    elif profile == "quintic":
        # Quintic profile
        s = 10.0*tau**3 - 15.0*tau**4 + 6.0*tau**5
    else:
        raise ValueError("profile debe ser 'linear', 'cubic' o 'quintic'")

    # Modified equinoctial elements evolution
    mee = mee0[None, :] + s[:, None] * (meef[None, :] - mee0[None, :])

    return mee


def build_bspline_matrices(
    n_ctrl: int,
    degree: int = 3,
    n_fine: int = 600,
) -> BSplineMatrices:
    tau_start = 0.0
    tau_end = 1.0
    knots = make_clamped_knots_interval(n_ctrl, degree, tau_start, tau_end)

    D1 = derivative_matrix(knots, degree)
    D2 = derivative_matrix(knots[1:-1], degree - 1)

    tau_fine = np.linspace(0.0, 1.0, n_fine)

    b0_fine = BSpline.design_matrix(tau_fine, knots, degree).toarray()
    b1_fine = BSpline.design_matrix(tau_fine, knots[1:-1], degree - 1).toarray() @ D1
    b2_fine = BSpline.design_matrix(tau_fine, knots[2:-2], degree - 2).toarray() @ D2 @ D1

    b0_start = BSpline.design_matrix([0.0], knots, degree).toarray()
    b1_start = BSpline.design_matrix([0.0], knots[1:-1], degree - 1).toarray() @ D1
    b0_end = BSpline.design_matrix([1.0], knots, degree).toarray()
    b1_end = BSpline.design_matrix([1.0], knots[1:-1], degree - 1).toarray() @ D1

    return BSplineMatrices(
        degree=degree,
        knots=knots,
        tau_fine=tau_fine,
        tau_start=tau_start,
        tau_end=tau_end,
        b0_fine=b0_fine,
        b1_fine=b1_fine,
        b2_fine=b2_fine,
        b0_start=b0_start,
        b1_start=b1_start,
        b0_end=b0_end,
        b1_end=b1_end,
    )


def derivative_matrix(knots: np.ndarray, degree: int) -> np.ndarray:
    n_ctrl = len(knots) - degree - 1
    D = np.zeros((n_ctrl - 1, n_ctrl))

    for j in range(n_ctrl - 1):
        denom = knots[j + degree + 1] - knots[j + 1]

        if abs(denom) > 1e-14:
            D[j, j] = -degree / denom
            D[j, j + 1] = degree / denom

    return D


def make_clamped_knots_interval(
    n_ctrl: int,
    degree: int,
    tau_start: float,
    tau_end: float,
) -> np.ndarray:

    knots = make_clamped_knots(n_ctrl, degree)
    return tau_start + (tau_end - tau_start) * knots


def make_clamped_knots(n_ctrl: int, degree: int) -> np.ndarray:
    if n_ctrl < degree + 1:
        raise ValueError("n_ctrl must be >= degree + 1")

    n_inner = n_ctrl - degree - 1

    if n_inner > 0:
        inner = np.linspace(0.0, 1.0, n_inner + 2)[1:-1]
    else:
        inner = np.array([])

    return np.concatenate(
        [
            np.zeros(degree + 1),
            inner,
            np.ones(degree + 1),
        ]
    )

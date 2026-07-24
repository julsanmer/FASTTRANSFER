"""Gondelach-style hodographic/cylindrical shaping for low-thrust transfers.

This is a practical comparator/seed generator inspired by hodographic shaping:
the angular branch is explicit through an unwrapped cylindrical angle, endpoint
states are matched analytically, and the required low-thrust acceleration is
recovered from the shaped trajectory.
"""

from __future__ import annotations

import numpy as np

from .canonical_units import MU_CANONICAL
from .orbit_utils import get_kepler_substeps
from utils.utils import kepler_coast_np, mee2rv, rv2mee


DEFAULT_MU = MU_CANONICAL
DEFAULT_POLAR_BASIS = "CPow2Pow4-CPow3Pow5-CosPCosPSin"


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def target_mee_at_time(t: float, mee_target_epoch: np.ndarray, mu: float) -> np.ndarray:
    return kepler_coast_np(np.asarray(mee_target_epoch, dtype=float), float(t), mu, n_iter=get_kepler_substeps())


def cartesian_to_cylindrical_state(
    pos: np.ndarray,
    vel: np.ndarray,
    theta_reference: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    return np.array([rho, theta, z], dtype=float), np.array([rho_dot, theta_dot, vz], dtype=float)


def cylindrical_to_cartesian_profile(
    q: np.ndarray,
    qdot: np.ndarray,
    qddot: np.ndarray,
    mu: float,
) -> dict:
    rho = q[:, 0]
    theta = q[:, 1]
    z = q[:, 2]
    rho_dot = qdot[:, 0]
    theta_dot = qdot[:, 1]
    z_dot = qdot[:, 2]
    rho_ddot = qddot[:, 0]
    theta_ddot = qddot[:, 1]
    z_ddot = qddot[:, 2]

    c = np.cos(theta)
    s = np.sin(theta)
    pos = np.column_stack([rho * c, rho * s, z])
    vel = np.column_stack(
        [
            rho_dot * c - rho * theta_dot * s,
            rho_dot * s + rho * theta_dot * c,
            z_dot,
        ]
    )
    acc = np.column_stack(
        [
            (rho_ddot - rho * theta_dot**2) * c - (2.0 * rho_dot * theta_dot + rho * theta_ddot) * s,
            (rho_ddot - rho * theta_dot**2) * s + (2.0 * rho_dot * theta_dot + rho * theta_ddot) * c,
            z_ddot,
        ]
    )
    r = np.sqrt(np.sum(pos * pos, axis=1, keepdims=True) + 1e-12)
    u = acc + float(mu) * pos / (r**3)
    return {
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "u": u,
        "u_norm": np.linalg.norm(u, axis=1),
    }


def _quintic_endpoint_shape(tau: np.ndarray, q0: np.ndarray, qf: np.ndarray, qtau0: np.ndarray, q_tauf: np.ndarray):
    tau = np.asarray(tau, dtype=float)
    q0 = np.asarray(q0, dtype=float).reshape(3)
    qf = np.asarray(qf, dtype=float).reshape(3)
    qtau0 = np.asarray(qtau0, dtype=float).reshape(3)
    q_tauf = np.asarray(q_tauf, dtype=float).reshape(3)

    coeffs = np.zeros((6, 3))
    coeffs[0] = q0
    coeffs[1] = qtau0
    coeffs[2] = 0.0
    rhs = np.vstack(
        [
            qf - coeffs[0] - coeffs[1] - coeffs[2],
            q_tauf - coeffs[1] - 2.0 * coeffs[2],
            -2.0 * coeffs[2],
        ]
    )
    mat = np.array(
        [
            [1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0],
            [6.0, 12.0, 20.0],
        ],
        dtype=float,
    )
    coeffs[3:6] = np.linalg.solve(mat, rhs)

    powers = np.vstack([tau**i for i in range(6)]).T
    dpowers = np.vstack(
        [
            np.zeros_like(tau),
            np.ones_like(tau),
            2.0 * tau,
            3.0 * tau**2,
            4.0 * tau**3,
            5.0 * tau**4,
        ]
    ).T
    ddpowers = np.vstack(
        [
            np.zeros_like(tau),
            np.zeros_like(tau),
            2.0 * np.ones_like(tau),
            6.0 * tau,
            12.0 * tau**2,
            20.0 * tau**3,
        ]
    ).T
    return powers @ coeffs, dpowers @ coeffs, ddpowers @ coeffs


def _shape_basis(tau: np.ndarray, n_shape: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau = np.asarray(tau, dtype=float)
    if int(n_shape) <= 0:
        return np.zeros((len(tau), 0)), np.zeros((len(tau), 0)), np.zeros((len(tau), 0))

    envelope = tau**2 * (1.0 - tau) ** 2
    denvelope = 2.0 * tau * (1.0 - tau) * (1.0 - 2.0 * tau)
    ddenvelope = 2.0 - 12.0 * tau + 12.0 * tau**2
    cols = []
    dcols = []
    ddcols = []
    x = 2.0 * tau - 1.0
    for idx in range(int(n_shape)):
        p = x**idx
        if idx == 0:
            dp = np.zeros_like(tau)
            ddp = np.zeros_like(tau)
        elif idx == 1:
            dp = 2.0 * np.ones_like(tau)
            ddp = np.zeros_like(tau)
        else:
            dp = 2.0 * idx * x ** (idx - 1)
            ddp = 4.0 * idx * (idx - 1) * x ** (idx - 2)
        cols.append(envelope * p)
        dcols.append(denvelope * p + envelope * dp)
        ddcols.append(ddenvelope * p + 2.0 * denvelope * dp + envelope * ddp)
    return np.column_stack(cols), np.column_stack(dcols), np.column_stack(ddcols)


def _evaluate_shape(
    tau: np.ndarray,
    tf: float,
    q0: np.ndarray,
    qf: np.ndarray,
    qdot0: np.ndarray,
    qdotf: np.ndarray,
    coeff: np.ndarray,
    mu: float,
) -> dict:
    tau = np.asarray(tau, dtype=float)
    tf = float(tf)
    coeff = np.asarray(coeff, dtype=float)
    q_base, qtau_base, qtautau_base = _quintic_endpoint_shape(tau, q0, qf, tf * qdot0, tf * qdotf)
    basis, dbasis, ddbasis = _shape_basis(tau, coeff.shape[0] if coeff.ndim == 2 else 0)
    if basis.shape[1]:
        q = q_base + basis @ coeff
        qtau = qtau_base + dbasis @ coeff
        qtautau = qtautau_base + ddbasis @ coeff
    else:
        q, qtau, qtautau = q_base, qtau_base, qtautau_base
    qdot = qtau / tf
    qddot = qtautau / (tf**2)
    cart = cylindrical_to_cartesian_profile(q, qdot, qddot, mu)
    t = tau * tf
    return {
        "tau": tau,
        "t": t,
        "q_cyl": q,
        "qdot_cyl": qdot,
        "qddot_cyl": qddot,
        **cart,
    }


def simpson_integral(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return trapezoid_integral(y, x)
    if (n - 1) % 2 == 1:
        return simpson_integral(y[:-1], x[:-1]) + trapezoid_integral(y[-2:], x[-2:])
    h = float((x[-1] - x[0]) / (n - 1))
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])))


def cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
    return out


def cumulative_simpson_integral(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return cumulative_trapezoid(y, x)
    h = float((x[-1] - x[0]) / (n - 1))
    out = np.zeros_like(y, dtype=float)
    out[1] = 0.5 * h * (y[0] + y[1])
    for idx in range(2, n):
        if idx % 2 == 0:
            out[idx] = out[idx - 2] + h / 3.0 * (y[idx - 2] + 4.0 * y[idx - 1] + y[idx])
        else:
            out[idx] = out[idx - 1] + 0.5 * h * (y[idx - 1] + y[idx])
    return out


class PolarBasis:
    def __init__(self, kind: str, power: int = 0, trig: str | None = None, freq_tag: str | None = None):
        self.kind = str(kind)
        self.power = int(power)
        self.trig = trig
        self.freq_tag = freq_tag

    def frequency(self, n_revolutions: float) -> float:
        tag = self.freq_tag
        if tag is None:
            return 1.0
        if tag in {"05", "0.5"}:
            return 0.5
        if tag in {"15", "1.5"}:
            return 1.5
        if tag in {"25", "2.5"}:
            return 2.5
        if tag in {"N5", "R5"}:
            return float(n_revolutions) + 0.5
        return float(tag)

    def evaluate(self, theta_rel: np.ndarray, theta_final: float, n_revolutions: float) -> tuple[np.ndarray, np.ndarray]:
        theta_rel = np.asarray(theta_rel, dtype=float)
        theta_final = float(theta_final)
        x = theta_rel / theta_final if abs(theta_final) > 1e-12 else theta_rel
        dx = 1.0 / theta_final if abs(theta_final) > 1e-12 else 1.0

        if self.kind == "constant":
            return np.ones_like(theta_rel), np.zeros_like(theta_rel)
        if self.kind == "power":
            p = self.power
            value = x**p
            deriv = np.zeros_like(theta_rel) if p == 0 else p * x ** (p - 1) * dx
            return value, deriv

        p = self.power
        amp = x**p
        damp = np.zeros_like(theta_rel) if p == 0 else p * x ** (p - 1) * dx
        freq = self.frequency(n_revolutions)
        arg = freq * theta_rel
        if self.trig == "sin":
            trig = np.sin(arg)
            dtrig = freq * np.cos(arg)
        elif self.trig == "cos":
            trig = np.cos(arg)
            dtrig = -freq * np.sin(arg)
        else:
            raise ValueError(f"Unknown trigonometric basis: {self.trig}")
        return amp * trig, damp * trig + amp * dtrig

    def __repr__(self) -> str:
        if self.kind == "constant":
            return "C"
        if self.kind == "power":
            return "Pow" if self.power == 1 else f"Pow{self.power}"
        prefix = "" if self.power == 0 else ("P" if self.power == 1 else f"P{self.power}")
        trig = "Sin" if self.trig == "sin" else "Cos"
        return f"{prefix}{trig}{self.freq_tag or ''}"


def _match_token(text: str, idx: int) -> tuple[PolarBasis, int]:
    if text.startswith("Pow", idx):
        j = idx + 3
        while j < len(text) and text[j].isdigit():
            j += 1
        power = int(text[idx + 3:j]) if j > idx + 3 else 1
        return PolarBasis("power", power=power), j

    if text[idx] == "C" and not text.startswith("Cos", idx):
        return PolarBasis("constant"), idx + 1

    if text[idx] == "P":
        j = idx + 1
        while j < len(text) and text[j].isdigit():
            j += 1
        power = int(text[idx + 1:j]) if j > idx + 1 else 1
        if text.startswith("Sin", j):
            trig = "sin"
            j += 3
        elif text.startswith("Cos", j):
            trig = "cos"
            j += 3
        else:
            raise ValueError(f"Cannot parse basis token near '{text[idx:]}'")
        suffix_start = j
        while j < len(text) and (text[j].isdigit() or text[j] in {"N", "R"}):
            j += 1
        suffix = text[suffix_start:j] or None
        return PolarBasis("trig", power=power, trig=trig, freq_tag=suffix), j

    if text.startswith("Sin", idx) or text.startswith("Cos", idx):
        trig = "sin" if text.startswith("Sin", idx) else "cos"
        j = idx + 3
        suffix_start = j
        while j < len(text) and (text[j].isdigit() or text[j] in {"N", "R"}):
            j += 1
        suffix = text[suffix_start:j] or None
        return PolarBasis("trig", power=0, trig=trig, freq_tag=suffix), j

    raise ValueError(f"Cannot parse basis token near '{text[idx:]}'")


def parse_polar_basis_group(text: str) -> list[PolarBasis]:
    out: list[PolarBasis] = []
    idx = 0
    while idx < len(text):
        token, idx = _match_token(text, idx)
        out.append(token)
    if len(out) < 3:
        raise ValueError(f"At least three base functions are required, got {text!r}")
    return out


def parse_polar_basis_string(basis: str) -> tuple[list[PolarBasis], list[PolarBasis], list[PolarBasis]]:
    parts = [item.strip() for item in str(basis).split("-")]
    if len(parts) != 3:
        raise ValueError("Gondelach basis must have three groups: radial-time-axial")
    return tuple(parse_polar_basis_group(part) for part in parts)  # type: ignore[return-value]


def _basis_matrix(
    bases: list[PolarBasis],
    theta_rel: np.ndarray,
    theta_final: float,
    n_revolutions: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = []
    derivs = []
    for basis in bases:
        value, deriv = basis.evaluate(theta_rel, theta_final, n_revolutions)
        values.append(value)
        derivs.append(deriv)
    return np.column_stack(values), np.column_stack(derivs)


def _solve_paper_coefficients(
    bases: list[PolarBasis],
    theta_rel: np.ndarray,
    theta_final: float,
    n_revolutions: float,
    y0: float,
    yf: float,
    integral_target: float,
    extra_coeff: np.ndarray,
) -> np.ndarray:
    values, _ = _basis_matrix(bases, theta_rel, theta_final, n_revolutions)
    integrals = np.array([simpson_integral(values[:, idx], theta_rel) for idx in range(values.shape[1])])
    extra_coeff = np.asarray(extra_coeff, dtype=float)
    rhs = np.array([y0, yf, integral_target], dtype=float)
    if len(bases) > 3:
        extra_values = values[:, 3:]
        rhs -= np.array(
            [
                float(extra_values[0] @ extra_coeff),
                float(extra_values[-1] @ extra_coeff),
                float(integrals[3:] @ extra_coeff),
            ]
        )
    mat = np.array(
        [
            values[0, :3],
            values[-1, :3],
            integrals[:3],
        ],
        dtype=float,
    )
    coeff_first = np.linalg.solve(mat, rhs)
    return np.concatenate([coeff_first, extra_coeff])


def _paper_polar_profile(
    theta_rel: np.ndarray,
    theta_final: float,
    n_revolutions: float,
    tf_target: float,
    q0: np.ndarray,
    qf: np.ndarray,
    qdot0: np.ndarray,
    qdotf: np.ndarray,
    basis_groups: tuple[list[PolarBasis], list[PolarBasis], list[PolarBasis]],
    free_coeff: np.ndarray,
    mu: float,
) -> tuple[dict, np.ndarray]:
    theta_dot0 = qdot0[1]
    theta_dotf = qdotf[1]
    if abs(theta_dot0) < 1e-10 or abs(theta_dotf) < 1e-10:
        raise ValueError("Endpoint angular rate is too small for polar-angle hodographic shaping")

    boundary = [
        (qdot0[0] / theta_dot0, qdotf[0] / theta_dotf, qf[0] - q0[0]),
        (1.0 / theta_dot0, 1.0 / theta_dotf, float(tf_target)),
        (qdot0[2] / theta_dot0, qdotf[2] / theta_dotf, qf[2] - q0[2]),
    ]
    n_extra = [max(0, len(group) - 3) for group in basis_groups]
    free_coeff = np.asarray(free_coeff, dtype=float)
    if free_coeff.size != sum(n_extra):
        raise ValueError("free_coeff size does not match selected basis")
    coeffs = []
    cursor = 0
    for idx, group in enumerate(basis_groups):
        extra = free_coeff[cursor:cursor + n_extra[idx]]
        cursor += n_extra[idx]
        target = boundary[idx][2]
        coeffs.append(
            _solve_paper_coefficients(
                group,
                theta_rel,
                theta_final,
                n_revolutions,
                float(boundary[idx][0]),
                float(boundary[idx][1]),
                float(target),
                extra,
            )
        )

    values = []
    derivs = []
    for group in basis_groups:
        val, der = _basis_matrix(group, theta_rel, theta_final, n_revolutions)
        values.append(val)
        derivs.append(der)
    R = values[0] @ coeffs[0]
    Rp = derivs[0] @ coeffs[0]

    T_raw = values[1] @ coeffs[1]
    Tp_raw = derivs[1] @ coeffs[1]
    # Eq. (21) uses |T| for time. The usual forward-transfer case has T > 0,
    # so the equality integral above directly imposes the requested tf.
    tf_integral = simpson_integral(np.abs(T_raw), theta_rel)
    if tf_integral <= 0.0 or not np.isfinite(tf_integral):
        raise ValueError("Invalid Gondelach T profile")
    T = T_raw
    Tp = Tp_raw

    Z = values[2] @ coeffs[2]
    Zp = derivs[2] @ coeffs[2]

    rho = q0[0] + cumulative_simpson_integral(R, theta_rel)
    z = q0[2] + cumulative_simpson_integral(Z, theta_rel)
    t = cumulative_simpson_integral(T, theta_rel)
    theta_abs = q0[1] + theta_rel
    theta_dot = 1.0 / T
    q = np.column_stack([rho, theta_abs, z])
    qdot = np.column_stack([R / T, theta_dot, Z / T])
    qddot = np.column_stack(
        [
            (Rp * T - R * Tp) / (T**3),
            -Tp / (T**3),
            (Zp * T - Z * Tp) / (T**3),
        ]
    )
    cart = cylindrical_to_cartesian_profile(q, qdot, qddot, mu)
    profile = {
        "tau": theta_rel / theta_final,
        "theta_rel": theta_rel,
        "t": t,
        "q_cyl": q,
        "qdot_cyl": qdot,
        "qddot_cyl": qddot,
        "R": R,
        "T": T,
        "Z": Z,
        "Rp": Rp,
        "Tp": Tp,
        "Zp": Zp,
        **cart,
    }
    return profile, np.concatenate(coeffs)


def solve_gondelach_polar_angle_hodographic(
    tf: float,
    mee0: np.ndarray,
    mee_target_epoch: np.ndarray,
    mu: float = DEFAULT_MU,
    sc_rev: float | None = None,
    n_fine: int = 801,
    basis: str = DEFAULT_POLAR_BASIS,
    optimize_shape: bool = False,
    objective: str = "dv",
    u_max: float | None = None,
    max_iter: int = 5000,
    print_level: int = 0,
) -> dict:
    tf = float(tf)
    mee0 = np.asarray(mee0, dtype=float).reshape(6)
    mee_target_epoch = np.asarray(mee_target_epoch, dtype=float).reshape(6)
    mee_target = target_mee_at_time(tf, mee_target_epoch, mu)
    pos0, vel0 = mee2rv(mee0, mu)
    posf, velf = mee2rv(mee_target, mu)

    q0, qdot0 = cartesian_to_cylindrical_state(pos0, vel0)
    theta_ref = q0[1] + 2.0 * np.pi * float(sc_rev) if sc_rev is not None else q0[1]
    qf, qdotf = cartesian_to_cylindrical_state(posf, velf, theta_reference=theta_ref)
    theta_final = float(qf[1] - q0[1])
    if theta_final <= 1e-8:
        raise ValueError("Gondelach polar method requires positive unwrapped transfer angle")
    n_revolutions = float(theta_final / (2.0 * np.pi))
    n_fine = int(n_fine)
    if n_fine % 2 == 0:
        n_fine += 1
    theta_rel = np.linspace(0.0, theta_final, n_fine)
    basis_groups = parse_polar_basis_string(basis)
    n_free = sum(max(0, len(group) - 3) for group in basis_groups)

    def build(flat: np.ndarray) -> tuple[dict, np.ndarray]:
        return _paper_polar_profile(
            theta_rel,
            theta_final,
            n_revolutions,
            tf,
            q0,
            qf,
            qdot0,
            qdotf,
            basis_groups,
            flat,
            mu,
        )

    def merit(flat: np.ndarray) -> float:
        try:
            prof, _ = build(flat)
        except Exception:
            return 1e30
        T = prof["T"]
        rho = prof["q_cyl"][:, 0]
        if np.any(~np.isfinite(T)) or np.any(T <= 0.0) or np.min(rho) <= 1e-4:
            return 1e30
        u_norm = prof["u_norm"]
        if np.any(~np.isfinite(u_norm)):
            return 1e30
        value = trapezoid_integral(u_norm, prof["t"]) if objective == "dv" else trapezoid_integral(u_norm**2, prof["t"])
        if u_max is not None:
            violation = np.maximum(u_norm / float(u_max) - 1.0, 0.0)
            value += max(1.0, value) * 1e4 * float(np.mean(violation**2))
        return float(value)

    free0 = np.zeros(n_free, dtype=float)
    opt_success = True
    opt_message = "paper_polar_lowest_order"
    free_opt = free0
    if bool(optimize_shape) and n_free > 0:
        from scipy.optimize import minimize

        res = minimize(
            merit,
            free0,
            method="Nelder-Mead",
            options={
                "maxiter": int(max_iter),
                "maxfev": int(max_iter),
                "disp": bool(print_level > 0),
                "xatol": 1e-8,
                "fatol": 1e-8,
            },
        )
        opt_success = bool(res.success)
        opt_message = str(res.message)
        free_opt = np.asarray(res.x, dtype=float)

    prof, coeffs = build(free_opt)

    u_norm = prof["u_norm"]
    dv = trapezoid_integral(u_norm, prof["t"])
    energy = trapezoid_integral(u_norm**2, prof["t"])
    endpoint = {
        "r0": float(np.linalg.norm(prof["pos"][0] - pos0)),
        "v0": float(np.linalg.norm(prof["vel"][0] - vel0)),
        "rf": float(np.linalg.norm(prof["pos"][-1] - posf)),
        "vf": float(np.linalg.norm(prof["vel"][-1] - velf)),
    }
    endpoint_norm = float(max(endpoint.values()))
    max_u = float(np.max(u_norm))
    T_positive = bool(np.all(prof["T"] > 0.0))
    u_ok = True if u_max is None else max_u <= float(u_max) * (1.0 + 1e-6)
    success = bool(endpoint_norm <= 1e-6 and T_positive and u_ok and np.all(np.isfinite(u_norm)))

    l_path = np.array([rv2mee(prof["pos"][i], prof["vel"][i], mu)[5] for i in range(len(prof["t"]))], dtype=float)
    l_unwrapped = np.unwrap(l_path)
    mee_delta_rev = float((l_unwrapped[-1] - l_unwrapped[0]) / (2.0 * np.pi))
    theta_delta_rev = float((prof["q_cyl"][-1, 1] - prof["q_cyl"][0, 1]) / (2.0 * np.pi))

    return {
        "success": success,
        "message": opt_message,
        "optimizer_success": opt_success,
        "method": "gondelach_paper_polar_angle",
        "basis": str(basis),
        "shape_coefficients": coeffs,
        "free_coefficients": free_opt,
        "profile_fine": prof,
        "endpoint_errors": endpoint,
        "endpoint_error_norm": endpoint_norm,
        "t_transfer": float(prof["t"][-1]),
        "target_mee": mee_target,
        "mee0": mee0,
        "mee_target_epoch": mee_target_epoch,
        "mu": float(mu),
        "sc_rev_target": float(sc_rev) if sc_rev is not None else float("nan"),
        "xy_delta_L_rev": theta_delta_rev,
        "raw_spacecraft_delta_L_rev": mee_delta_rev,
        "rev_error_rev": mee_delta_rev - float(sc_rev) if sc_rev is not None else float("nan"),
        "max_u": max_u,
        "dv_quad": dv,
        "energy_quad": energy,
        "objective_type": objective,
        "n_shape": int(n_free),
        "optimize_shape": bool(optimize_shape),
        "T_positive": T_positive,
    }


def solve_gondelach_hodographic(
    tf: float,
    mee0: np.ndarray,
    mee_target_epoch: np.ndarray,
    mu: float = DEFAULT_MU,
    sc_rev: float | None = None,
    n_fine: int = 800,
    n_shape: int = 0,
    optimize_shape: bool = False,
    objective: str = "energy",
    u_max: float | None = None,
    max_iter: int = 120,
    print_level: int = 0,
    mode: str = "paper-polar",
    basis: str = DEFAULT_POLAR_BASIS,
) -> dict:
    """Build and optionally optimize a Gondelach-style shaped trajectory."""
    mode_normalized = str(mode).strip().lower().replace("_", "-")
    if mode_normalized in {"paper", "paper-polar", "polar", "polar-angle"}:
        return solve_gondelach_polar_angle_hodographic(
            tf,
            mee0=mee0,
            mee_target_epoch=mee_target_epoch,
            mu=mu,
            sc_rev=sc_rev,
            n_fine=n_fine,
            basis=basis,
            optimize_shape=optimize_shape,
            objective=objective,
            u_max=u_max,
            max_iter=max_iter,
            print_level=print_level,
        )
    if mode_normalized not in {"quintic", "quintic-cylindrical", "legacy"}:
        raise ValueError(f"Unknown Gondelach mode: {mode!r}")

    tf = float(tf)
    mee0 = np.asarray(mee0, dtype=float).reshape(6)
    mee_target_epoch = np.asarray(mee_target_epoch, dtype=float).reshape(6)
    mee_target = target_mee_at_time(tf, mee_target_epoch, mu)
    pos0, vel0 = mee2rv(mee0, mu)
    posf, velf = mee2rv(mee_target, mu)

    q0, qdot0 = cartesian_to_cylindrical_state(pos0, vel0)
    theta_ref = q0[1] + 2.0 * np.pi * float(sc_rev) if sc_rev is not None else q0[1]
    qf, qdotf = cartesian_to_cylindrical_state(posf, velf, theta_reference=theta_ref)

    tau = np.linspace(0.0, 1.0, int(n_fine))
    n_shape = int(max(n_shape, 0))
    coeff0 = np.zeros((n_shape, 3), dtype=float)

    def eval_with_flat(flat_coeff: np.ndarray) -> dict:
        coeff = np.asarray(flat_coeff, dtype=float).reshape(n_shape, 3) if n_shape else coeff0
        return _evaluate_shape(tau, tf, q0, qf, qdot0, qdotf, coeff, mu)

    def merit(flat_coeff: np.ndarray) -> float:
        prof = eval_with_flat(flat_coeff)
        rho = prof["q_cyl"][:, 0]
        if np.any(~np.isfinite(rho)) or np.min(rho) <= 1e-4:
            return 1e30 + 1e30 * float(np.sum(np.maximum(1e-4 - rho, 0.0) ** 2))
        u_norm = prof["u_norm"]
        if np.any(~np.isfinite(u_norm)):
            return 1e30
        if objective == "dv":
            value = trapezoid_integral(u_norm, prof["t"])
        else:
            value = trapezoid_integral(u_norm**2, prof["t"])
        if u_max is not None:
            violation = np.maximum(u_norm / float(u_max) - 1.0, 0.0)
            value += max(1.0, value) * 1e4 * float(np.mean(violation**2))
        return float(value)

    coeff_opt = coeff0
    opt_success = True
    opt_message = "analytic_no_free_shape"
    if bool(optimize_shape) and n_shape > 0:
        from scipy.optimize import minimize

        rscale = max(float(q0[0]), float(qf[0]), 1.0)
        theta_scale = max(abs(float(qf[1] - q0[1])), 2.0 * np.pi)
        zscale = max(abs(float(q0[2])), abs(float(qf[2])), 0.2)
        bounds = []
        for _ in range(n_shape):
            bounds.extend(
                [
                    (-0.5 * rscale, 0.5 * rscale),
                    (-0.25 * theta_scale, 0.25 * theta_scale),
                    (-0.5 * zscale, 0.5 * zscale),
                ]
            )
        res = minimize(
            merit,
            coeff0.reshape(-1),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_iter), "disp": bool(print_level > 0)},
        )
        opt_success = bool(res.success)
        opt_message = str(res.message)
        coeff_opt = np.asarray(res.x, dtype=float).reshape(n_shape, 3)

    prof = _evaluate_shape(tau, tf, q0, qf, qdot0, qdotf, coeff_opt, mu)
    u_norm = prof["u_norm"]
    dv = trapezoid_integral(u_norm, prof["t"])
    energy = trapezoid_integral(u_norm**2, prof["t"])
    endpoint = {
        "r0": float(np.linalg.norm(prof["pos"][0] - pos0)),
        "v0": float(np.linalg.norm(prof["vel"][0] - vel0)),
        "rf": float(np.linalg.norm(prof["pos"][-1] - posf)),
        "vf": float(np.linalg.norm(prof["vel"][-1] - velf)),
    }
    endpoint_norm = float(max(endpoint.values()))
    max_u = float(np.max(u_norm))
    u_ok = True if u_max is None else max_u <= float(u_max) * (1.0 + 1e-6)
    success = bool(endpoint_norm <= 1e-7 and u_ok and np.all(np.isfinite(u_norm)))

    l_path = np.array([rv2mee(prof["pos"][i], prof["vel"][i], mu)[5] for i in range(len(tau))], dtype=float)
    l_unwrapped = np.unwrap(l_path)
    mee_delta_rev = float((l_unwrapped[-1] - l_unwrapped[0]) / (2.0 * np.pi))
    theta_delta_rev = float((prof["q_cyl"][-1, 1] - prof["q_cyl"][0, 1]) / (2.0 * np.pi))

    return {
        "success": success,
        "message": opt_message,
        "optimizer_success": opt_success,
        "method": "gondelach_style_hodographic",
        "basis": "quintic_endpoint_cylindrical",
        "shape_coefficients": coeff_opt,
        "profile_fine": prof,
        "endpoint_errors": endpoint,
        "endpoint_error_norm": endpoint_norm,
        "t_transfer": tf,
        "target_mee": mee_target,
        "mee0": mee0,
        "mee_target_epoch": mee_target_epoch,
        "mu": float(mu),
        "sc_rev_target": float(sc_rev) if sc_rev is not None else float("nan"),
        "xy_delta_L_rev": theta_delta_rev,
        "raw_spacecraft_delta_L_rev": mee_delta_rev,
        "rev_error_rev": mee_delta_rev - float(sc_rev) if sc_rev is not None else float("nan"),
        "max_u": max_u,
        "dv_quad": dv,
        "energy_quad": energy,
        "objective_type": objective,
        "n_shape": int(n_shape),
        "optimize_shape": bool(optimize_shape),
    }

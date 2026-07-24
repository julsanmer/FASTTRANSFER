"""Reproduce Gondelach & Noomen 2015, Fig. 2, as closely as possible.

Figure 2 is a Mars porkchop for the lowest-order time-driven hodographic
solution

    CPowPow2-CPowPow2-CosN5P3CosN5P3SinN5

with integer revolution parameter N = 0..5.  This script uses the secular
J2000 Keplerian elements and date/TOF ranges printed in the paper tables, then
selects the best Delta V over N for each departure-date/TOF grid point.

This is intentionally independent of CasADi/Ipopt: it is a direct evaluator of
the shaped trajectory, not an optimal-control solve.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.reference_metrics import evaluate_reference_metrics  # noqa: E402


AU_KM = 149_597_870.7
DAY_S = 86_400.0
AU_PER_DAY_TO_KM_PER_S = AU_KM / DAY_S
AU_PER_DAY2_TO_M_PER_S2 = (AU_KM * 1000.0) / (DAY_S**2)
MU_SUN_AU_DAY = 0.01720209895**2
GONDELACH_FORMULATION_VERSION = "analytic_basis_derivatives_integrals_v1"


@dataclass(frozen=True)
class SecularElements:
    a: float
    a_rate: float
    e: float
    e_rate: float
    inc_deg: float
    inc_rate_deg: float
    mean_long_deg: float
    mean_long_rate_deg: float
    lon_peri_deg: float
    lon_peri_rate_deg: float
    raan_deg: float
    raan_rate_deg: float


@dataclass(frozen=True)
class FixedKeplerianElements:
    a: float
    e: float
    inc_deg: float
    raan_deg: float
    arg_peri_deg: float
    mean_anomaly_deg: float
    epoch_mjd: float


# Table 4 in Gondelach & Noomen, with respect to mean ecliptic/equinox J2000.
ELEMENTS = {
    "mercury": SecularElements(
        a=0.38710,
        a_rate=0.00000,
        e=0.20564,
        e_rate=0.00002,
        inc_deg=7.00559,
        inc_rate_deg=-0.00590,
        mean_long_deg=252.25167,
        mean_long_rate_deg=149_472.67487,
        lon_peri_deg=77.45772,
        lon_peri_rate_deg=0.15940,
        raan_deg=48.33962,
        raan_rate_deg=-0.12214,
    ),
    "earth": SecularElements(
        a=1.00000,
        a_rate=0.00000,
        e=0.01673,
        e_rate=-0.00004,
        inc_deg=-0.00054,
        inc_rate_deg=-0.01337,
        mean_long_deg=100.46692,
        mean_long_rate_deg=35_999.37306,
        lon_peri_deg=102.93006,
        lon_peri_rate_deg=0.31795,
        raan_deg=-5.11260,
        raan_rate_deg=-0.24124,
    ),
    "mars": SecularElements(
        a=1.52371,
        a_rate=0.00000,
        e=0.09337,
        e_rate=0.00009,
        inc_deg=1.85182,
        inc_rate_deg=-0.00725,
        mean_long_deg=-4.56813,
        mean_long_rate_deg=19_140.29934,
        lon_peri_deg=-23.91745,
        lon_peri_rate_deg=0.45224,
        raan_deg=49.71321,
        raan_rate_deg=-0.26852,
    ),
}


# Table 5 in Gondelach & Noomen, with respect to mean ecliptic/equinox J2000.
FIXED_ELEMENTS = {
    "1989ml": FixedKeplerianElements(
        a=1.27254,
        e=0.13671,
        inc_deg=4.37800,
        raan_deg=104.38253,
        arg_peri_deg=183.23998,
        mean_anomaly_deg=287.92582,
        epoch_mjd=56_000.0,
    ),
    "tempel1": FixedKeplerianElements(
        a=3.12338,
        e=0.51734,
        inc_deg=10.52975,
        raan_deg=68.93384,
        arg_peri_deg=178.91137,
        mean_anomaly_deg=162.40622,
        epoch_mjd=54_466.0,
    ),
}


@dataclass(frozen=True)
class BasisTerm:
    kind: str
    power: int = 0
    trig: str | None = None
    freq_tag: str | None = None

    def frequency(self, n_rev: int) -> float:
        tag = self.freq_tag
        if tag is None:
            return 1.0
        if tag == "05":
            return 0.5
        if tag == "15":
            return 1.5
        if tag == "25":
            return 2.5
        if tag in {"N5", "R5"}:
            return float(n_rev) + 0.5
        return float(tag)

    def evaluate(self, tau: np.ndarray, n_rev: int) -> tuple[np.ndarray, np.ndarray]:
        tau = np.asarray(tau, dtype=float)
        if self.kind == "constant":
            return np.ones_like(tau), np.zeros_like(tau)
        if self.kind == "power":
            value = tau**self.power
            deriv = np.zeros_like(tau) if self.power == 0 else self.power * tau ** (self.power - 1)
            return value, deriv

        amp = tau**self.power
        damp = np.zeros_like(tau) if self.power == 0 else self.power * tau ** (self.power - 1)
        omega = 2.0 * math.pi * self.frequency(n_rev)
        arg = omega * tau
        if self.trig == "sin":
            trig = np.sin(arg)
            dtrig = omega * np.cos(arg)
        elif self.trig == "cos":
            trig = np.cos(arg)
            dtrig = -omega * np.sin(arg)
        else:
            raise ValueError(f"Unknown trigonometric basis: {self.trig}")
        return amp * trig, damp * trig + amp * dtrig


def wrap_0_2pi(angle: float) -> float:
    return float(angle % (2.0 * math.pi))


def solve_kepler(mean_anomaly: float, ecc: float) -> float:
    mean_anomaly = (mean_anomaly + math.pi) % (2.0 * math.pi) - math.pi
    ecc = float(ecc)
    estimate = mean_anomaly if ecc < 0.8 else math.pi
    for _ in range(30):
        f = estimate - ecc * math.sin(estimate) - mean_anomaly
        fp = 1.0 - ecc * math.cos(estimate)
        step = f / fp
        estimate -= step
        if abs(step) < 1e-13:
            break
    return float(estimate)


def rot3(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rot1(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def keplerian_state(
    a: float,
    e: float,
    inc: float,
    raan: float,
    arg_peri: float,
    mean_anomaly: float,
) -> tuple[np.ndarray, np.ndarray]:
    ecc_anomaly = solve_kepler(mean_anomaly, e)
    cos_e = math.cos(ecc_anomaly)
    sin_e = math.sin(ecc_anomaly)
    radius_factor = 1.0 - e * cos_e
    mean_motion = math.sqrt(MU_SUN_AU_DAY / (a**3))
    r_perifocal = np.array(
        [
            a * (cos_e - e),
            a * math.sqrt(1.0 - e * e) * sin_e,
            0.0,
        ],
        dtype=float,
    )
    v_perifocal = np.array(
        [
            -a * mean_motion * sin_e / radius_factor,
            a * mean_motion * math.sqrt(1.0 - e * e) * cos_e / radius_factor,
            0.0,
        ],
        dtype=float,
    )
    rotation = rot3(raan) @ rot1(inc) @ rot3(arg_peri)
    return rotation @ r_perifocal, rotation @ v_perifocal


_EPHEMERIS_SOURCE = "kepler"
_SPICE_META_KERNEL = ""
_SPICE_TARGET_NAME = ""


def configure_ephemeris(
    source: str = "kepler",
    spice_meta_kernel: str | None = None,
    spice_target_name: str | None = None,
) -> None:
    global _EPHEMERIS_SOURCE, _SPICE_META_KERNEL, _SPICE_TARGET_NAME
    source = str(source).lower()
    if source not in {"kepler", "spice"}:
        raise ValueError(f"Unsupported ephemeris source: {source}")
    if source == "spice" and not spice_meta_kernel:
        raise ValueError("--spice-meta-kernel is required with --ephemeris spice")
    _EPHEMERIS_SOURCE = source
    _SPICE_META_KERNEL = str(spice_meta_kernel or "")
    _SPICE_TARGET_NAME = str(spice_target_name or "")
    if source == "spice":
        from experiments.spice_ephemeris import ensure_kernels_loaded

        ensure_kernels_loaded(_SPICE_META_KERNEL)


def ephemeris_metadata() -> dict[str, str]:
    if _EPHEMERIS_SOURCE == "spice":
        from experiments.spice_ephemeris import spice_metadata

        metadata = spice_metadata(_SPICE_META_KERNEL)
        metadata["spice_target_name"] = _SPICE_TARGET_NAME
        return metadata
    return {"ephemeris_source": "kepler"}


def keplerian_body_state(body: str, mjd2000: float) -> tuple[np.ndarray, np.ndarray]:
    body = str(body).lower()
    if body in FIXED_ELEMENTS:
        elements_fixed = FIXED_ELEMENTS[body]
        epoch_mjd2000 = elements_fixed.epoch_mjd - 51_544.5
        mean_motion = math.sqrt(MU_SUN_AU_DAY / (elements_fixed.a**3))
        mean_anomaly = math.radians(elements_fixed.mean_anomaly_deg) + mean_motion * (float(mjd2000) - epoch_mjd2000)
        return keplerian_state(
            elements_fixed.a,
            elements_fixed.e,
            math.radians(elements_fixed.inc_deg),
            math.radians(elements_fixed.raan_deg),
            math.radians(elements_fixed.arg_peri_deg),
            mean_anomaly,
        )

    elements = ELEMENTS[body]
    centuries = float(mjd2000) / 36525.0
    a = elements.a + elements.a_rate * centuries
    e = elements.e + elements.e_rate * centuries
    inc = math.radians(elements.inc_deg + elements.inc_rate_deg * centuries)
    mean_long = math.radians(elements.mean_long_deg + elements.mean_long_rate_deg * centuries)
    lon_peri = math.radians(elements.lon_peri_deg + elements.lon_peri_rate_deg * centuries)
    raan = math.radians(elements.raan_deg + elements.raan_rate_deg * centuries)
    arg_peri = lon_peri - raan
    mean_anomaly = mean_long - lon_peri
    return keplerian_state(a, e, inc, raan, arg_peri, mean_anomaly)


def planet_state(body: str, mjd2000: float) -> tuple[np.ndarray, np.ndarray]:
    body_key = str(body).lower()
    if _EPHEMERIS_SOURCE == "kepler":
        return keplerian_body_state(body_key, mjd2000)
    from experiments.spice_ephemeris import spice_state

    spice_name = _SPICE_TARGET_NAME if _SPICE_TARGET_NAME and body_key not in {"earth", "sun"} else body.upper()
    return spice_state(spice_name, float(mjd2000), _SPICE_META_KERNEL)


def inclusive_grid(lower: float, upper: float, step: float) -> np.ndarray:
    step = float(step)
    if step <= 0.0:
        raise ValueError("Grid spacing must be positive")
    values = np.arange(float(lower), float(upper) + 0.5 * step, step, dtype=float)
    values = values[values <= float(upper) + 1.0e-9]
    if values.size == 0:
        raise ValueError("Grid spacing produced an empty grid")
    return values


def cartesian_to_cylindrical(pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = np.asarray(pos, dtype=float)
    vx, vy, vz = np.asarray(vel, dtype=float)
    rho = float(math.hypot(x, y))
    theta = float(math.atan2(y, x))
    vr = float((x * vx + y * vy) / rho)
    vtheta = float((x * vy - y * vx) / rho)
    return np.array([rho, theta, z], dtype=float), np.array([vr, vtheta, vz], dtype=float)


def simpson(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return float(np.trapezoid(y, x))
    if (len(x) - 1) % 2:
        return simpson(y[:-1], x[:-1]) + float(np.trapezoid(y[-2:], x[-2:]))
    h = float((x[-1] - x[0]) / (len(x) - 1))
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-1:2])))


def cumulative_simpson(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(y)
    if len(x) < 3:
        out[1:] = np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))
        return out
    h = float((x[-1] - x[0]) / (len(x) - 1))
    out[1] = 0.5 * h * (y[0] + y[1])
    for idx in range(2, len(x)):
        if idx % 2 == 0:
            out[idx] = out[idx - 2] + h / 3.0 * (y[idx - 2] + 4.0 * y[idx - 1] + y[idx])
        else:
            out[idx] = out[idx - 1] + 0.5 * h * (y[idx - 1] + y[idx])
    return out


def parse_basis_group(text: str) -> list[BasisTerm]:
    terms: list[BasisTerm] = []
    idx = 0
    while idx < len(text):
        if text.startswith("Pow", idx):
            end = idx + 3
            while end < len(text) and text[end].isdigit():
                end += 1
            power = int(text[idx + 3 : end]) if end > idx + 3 else 1
            terms.append(BasisTerm("power", power=power))
            idx = end
            continue

        if text.startswith("Sin", idx) or text.startswith("Cos", idx):
            trig = "sin" if text.startswith("Sin", idx) else "cos"
            end = idx + 3
            suffix_start = end
            while end < len(text) and (text[end].isdigit() or text[end] in {"N", "R"}):
                end += 1
            terms.append(BasisTerm("trig", power=0, trig=trig, freq_tag=text[suffix_start:end] or None))
            idx = end
            continue

        if text[idx] == "P":
            end = idx + 1
            while end < len(text) and text[end].isdigit():
                end += 1
            power = int(text[idx + 1 : end]) if end > idx + 1 else 1
            if text.startswith("Sin", end):
                trig = "sin"
                end += 3
            elif text.startswith("Cos", end):
                trig = "cos"
                end += 3
            else:
                raise ValueError(f"Cannot parse basis token near {text[idx:]!r}")
            suffix_start = end
            while end < len(text) and (text[end].isdigit() or text[end] in {"N", "R"}):
                end += 1
            terms.append(BasisTerm("trig", power=power, trig=trig, freq_tag=text[suffix_start:end] or None))
            idx = end
            continue

        if text[idx] == "C":
            terms.append(BasisTerm("constant"))
            idx += 1
            continue

        raise ValueError(f"Cannot parse basis token near {text[idx:]!r}")

    if len(terms) < 3:
        raise ValueError(f"Expected at least three basis functions per component, got {text!r}")
    return terms


def basis_matrix(terms: list[BasisTerm], tau: np.ndarray, n_rev: int) -> tuple[np.ndarray, np.ndarray]:
    values = []
    derivs = []
    for term in terms:
        value, deriv = term.evaluate(tau, n_rev)
        values.append(value)
        derivs.append(deriv)
    return np.column_stack(values), np.column_stack(derivs)


def basis_integral_matrix(terms: list[BasisTerm], tau: np.ndarray, n_rev: int) -> np.ndarray:
    """Exact integrals of each basis term from zero to every tau value."""
    tau = np.asarray(tau, dtype=float)
    integrals = []
    for term in terms:
        if term.kind == "constant":
            integrals.append(tau)
            continue
        if term.kind == "power":
            integrals.append(tau ** (term.power + 1) / float(term.power + 1))
            continue

        omega = 2.0 * math.pi * term.frequency(n_rev)
        sin_integral = (1.0 - np.cos(omega * tau)) / omega
        cos_integral = np.sin(omega * tau) / omega
        for power in range(1, term.power + 1):
            next_sin = -(tau**power) * np.cos(omega * tau) / omega + power * cos_integral / omega
            next_cos = (tau**power) * np.sin(omega * tau) / omega - power * sin_integral / omega
            sin_integral, cos_integral = next_sin, next_cos
        if term.trig == "sin":
            integrals.append(sin_integral)
        elif term.trig == "cos":
            integrals.append(cos_integral)
        else:
            raise ValueError(f"Unknown trigonometric basis: {term.trig}")
    return np.column_stack(integrals)


_REFERENCE_BASIS_CACHE: dict[
    tuple[tuple[BasisTerm, ...], int, int],
    tuple[np.ndarray, np.ndarray, np.ndarray],
] = {}


def reference_basis_matrices(
    terms: list[BasisTerm], tau: np.ndarray, n_rev: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cache basis data at the shared evaluator's immutable node arrays."""
    key = (tuple(terms), int(n_rev), id(tau))
    cached = _REFERENCE_BASIS_CACHE.get(key)
    if cached is None:
        values, derivatives = basis_matrix(terms, tau, n_rev)
        cached = (values, derivatives, basis_integral_matrix(terms, tau, n_rev))
        _REFERENCE_BASIS_CACHE[key] = cached
    return cached


def solve_component_coefficients(
    terms: list[BasisTerm],
    tau: np.ndarray,
    n_rev: int,
    tf_days: float,
    y0: float,
    yf: float,
    integral_target: float,
    free_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    values, _ = basis_matrix(terms, tau, n_rev)
    free_coefficients = np.asarray(
        [] if free_coefficients is None else free_coefficients,
        dtype=float,
    )
    n_free = values.shape[1] - 3
    if len(free_coefficients) != n_free:
        raise ValueError(f"Expected {n_free} free coefficients, got {len(free_coefficients)}")
    integrals = basis_integral_matrix(terms, np.asarray([1.0]), n_rev)[0]
    matrix = np.vstack([values[0, :3], values[-1, :3], tf_days * integrals[:3]])
    rhs = np.array([y0, yf, integral_target], dtype=float)
    if n_free:
        free_values = values[:, 3:]
        rhs -= np.array(
            [
                free_values[0] @ free_coefficients,
                free_values[-1] @ free_coefficients,
                tf_days * (integrals[3:] @ free_coefficients),
            ],
            dtype=float,
        )
    constrained = np.linalg.solve(matrix, rhs)
    return np.concatenate([constrained, free_coefficients])


def solve_transverse_coefficients(
    terms: list[BasisTerm],
    tau: np.ndarray,
    n_rev: int,
    tf_days: float,
    vtheta0: float,
    vthetaf: float,
    theta_target: float,
    rho: np.ndarray,
    free_coefficients: np.ndarray | None = None,
) -> np.ndarray:
    values, _ = basis_matrix(terms, tau, n_rev)
    free_coefficients = np.asarray(
        [] if free_coefficients is None else free_coefficients,
        dtype=float,
    )
    n_free = values.shape[1] - 3
    if len(free_coefficients) != n_free:
        raise ValueError(f"Expected {n_free} free coefficients, got {len(free_coefficients)}")
    angle_integrals = np.array([tf_days * simpson(values[:, idx] / rho, tau) for idx in range(values.shape[1])])
    matrix = np.vstack([values[0, :3], values[-1, :3], angle_integrals[:3]])
    rhs = np.array([vtheta0, vthetaf, theta_target], dtype=float)
    if n_free:
        free_values = values[:, 3:]
        rhs -= np.array(
            [
                free_values[0] @ free_coefficients,
                free_values[-1] @ free_coefficients,
                angle_integrals[3:] @ free_coefficients,
            ],
            dtype=float,
        )
    constrained = np.linalg.solve(matrix, rhs)
    return np.concatenate([constrained, free_coefficients])


def count_free_coefficients(radial_terms: list[BasisTerm], transverse_terms: list[BasisTerm], axial_terms: list[BasisTerm]) -> int:
    return sum(max(0, len(terms) - 3) for terms in [radial_terms, transverse_terms, axial_terms])


def split_free_coefficients(
    free_coefficients: np.ndarray | None,
    radial_terms: list[BasisTerm],
    transverse_terms: list[BasisTerm],
    axial_terms: list[BasisTerm],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = [max(0, len(terms) - 3) for terms in [radial_terms, transverse_terms, axial_terms]]
    total = sum(counts)
    if free_coefficients is None:
        free = np.zeros(total, dtype=float)
    else:
        free = np.asarray(free_coefficients, dtype=float)
    if len(free) != total:
        raise ValueError(f"Expected {total} free coefficients, got {len(free)}")
    out = []
    offset = 0
    for count in counts:
        out.append(free[offset : offset + count])
        offset += count
    return tuple(out)  # type: ignore[return-value]


def evaluate_time_driven_metrics(
    dep_mjd2000: float,
    tof_days: float,
    n_rev: int,
    n_quad: int,
    basis: str = "CPowPow2-CPowPow2-CosN5P3CosN5P3SinN5",
    free_coefficients: np.ndarray | None = None,
    target: str = "mars",
) -> dict:
    if n_quad % 2 == 0:
        n_quad += 1
    tau = np.linspace(0.0, 1.0, n_quad)
    radial_terms, transverse_terms, axial_terms = [parse_basis_group(part) for part in basis.split("-")]

    pos0, vel0 = planet_state("earth", dep_mjd2000)
    posf, velf = planet_state(target, dep_mjd2000 + tof_days)
    q0, qdot0 = cartesian_to_cylindrical(pos0, vel0)
    qf, qdotf = cartesian_to_cylindrical(posf, velf)
    radial_free, transverse_free, axial_free = split_free_coefficients(
        free_coefficients,
        radial_terms,
        transverse_terms,
        axial_terms,
    )

    theta_target = wrap_0_2pi(float(qf[1] - q0[1])) + 2.0 * math.pi * int(n_rev)
    coeff_r = solve_component_coefficients(
        radial_terms,
        tau,
        n_rev,
        tof_days,
        qdot0[0],
        qdotf[0],
        qf[0] - q0[0],
        radial_free,
    )
    radial_values, radial_derivs = basis_matrix(radial_terms, tau, n_rev)
    vr = radial_values @ coeff_r
    radial_integrals = basis_integral_matrix(radial_terms, tau, n_rev)
    rho = q0[0] + tof_days * (radial_integrals @ coeff_r)
    if np.any(~np.isfinite(rho)) or np.min(rho) <= 1e-6:
        return float("nan")

    coeff_theta = solve_transverse_coefficients(
        transverse_terms,
        tau,
        n_rev,
        tof_days,
        qdot0[1],
        qdotf[1],
        theta_target,
        rho,
        transverse_free,
    )
    coeff_z = solve_component_coefficients(
        axial_terms,
        tau,
        n_rev,
        tof_days,
        qdot0[2],
        qdotf[2],
        qf[2] - q0[2],
        axial_free,
    )

    theta_values, theta_derivs = basis_matrix(transverse_terms, tau, n_rev)
    axial_values, axial_derivs = basis_matrix(axial_terms, tau, n_rev)
    vtheta = theta_values @ coeff_theta
    vz = axial_values @ coeff_z
    axial_integrals = basis_integral_matrix(axial_terms, tau, n_rev)
    z = q0[2] + tof_days * (axial_integrals @ coeff_z)

    vr_dot = (radial_derivs @ coeff_r) / tof_days
    vtheta_dot = (theta_derivs @ coeff_theta) / tof_days
    vz_dot = (axial_derivs @ coeff_z) / tof_days

    s = np.sqrt(rho * rho + z * z)
    f_r = vr_dot - vtheta * vtheta / rho + MU_SUN_AU_DAY * rho / (s**3)
    f_theta = vtheta_dot + vr * vtheta / rho
    f_z = vz_dot + MU_SUN_AU_DAY * z / (s**3)
    accel = np.sqrt(f_r * f_r + f_theta * f_theta + f_z * f_z)
    return {
        "delta_v_km_s": tof_days * simpson(accel, tau) * AU_PER_DAY_TO_KM_PER_S,
        "fmax_m_s2": float(np.nanmax(accel)) * AU_PER_DAY2_TO_M_PER_S2,
    }


def evaluate_time_driven(
    dep_mjd2000: float,
    tof_days: float,
    n_rev: int,
    n_quad: int,
    basis: str = "CPowPow2-CPowPow2-CosN5P3CosN5P3SinN5",
    free_coefficients: np.ndarray | None = None,
    target: str = "mars",
) -> float:
    return float(
        evaluate_time_driven_metrics(
            dep_mjd2000,
            tof_days,
            n_rev,
            n_quad,
            basis,
            free_coefficients,
            target,
        )["delta_v_km_s"]
    )


def reconstruct_coefficients_from_free(
    dep_mjd2000: float,
    tof_days: float,
    n_rev: int,
    n_quad: int,
    basis: str,
    free_coefficients: np.ndarray | None = None,
    target: str = "mars",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n_quad % 2 == 0:
        n_quad += 1
    tau = np.linspace(0.0, 1.0, n_quad)
    radial_terms, transverse_terms, axial_terms = [parse_basis_group(part) for part in basis.split("-")]

    pos0, vel0 = planet_state("earth", dep_mjd2000)
    posf, velf = planet_state(target, dep_mjd2000 + tof_days)
    q0, qdot0 = cartesian_to_cylindrical(pos0, vel0)
    qf, qdotf = cartesian_to_cylindrical(posf, velf)
    radial_free, transverse_free, axial_free = split_free_coefficients(
        free_coefficients,
        radial_terms,
        transverse_terms,
        axial_terms,
    )

    theta_target = wrap_0_2pi(float(qf[1] - q0[1])) + 2.0 * math.pi * int(n_rev)
    coeff_r = solve_component_coefficients(
        radial_terms,
        tau,
        n_rev,
        tof_days,
        qdot0[0],
        qdotf[0],
        qf[0] - q0[0],
        radial_free,
    )
    radial_integrals = basis_integral_matrix(radial_terms, tau, n_rev)
    rho = q0[0] + tof_days * (radial_integrals @ coeff_r)
    if np.any(~np.isfinite(rho)) or np.min(rho) <= 1e-6:
        raise ValueError("Invalid radial profile while reconstructing coefficients")

    coeff_theta = solve_transverse_coefficients(
        transverse_terms,
        tau,
        n_rev,
        tof_days,
        qdot0[1],
        qdotf[1],
        theta_target,
        rho,
        transverse_free,
    )
    coeff_z = solve_component_coefficients(
        axial_terms,
        tau,
        n_rev,
        tof_days,
        qdot0[2],
        qdotf[2],
        qf[2] - q0[2],
        axial_free,
    )
    return coeff_r, coeff_theta, coeff_z


def evaluate_time_driven_reference_metrics(
    dep_mjd2000: float,
    tof_days: float,
    n_rev: int,
    n_quad: int,
    basis: str,
    free_coefficients: np.ndarray | None = None,
    target: str = "mars",
    coefficients: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Evaluate Delta V and peak acceleration with the shared reference rule."""
    radial_terms, transverse_terms, axial_terms = [parse_basis_group(part) for part in basis.split("-")]
    if coefficients is None:
        coefficients = reconstruct_coefficients_from_free(
            dep_mjd2000,
            tof_days,
            n_rev,
            n_quad,
            basis,
            free_coefficients,
            target,
        )
    coeff_r, coeff_theta, coeff_z = [np.asarray(values, dtype=float) for values in coefficients]

    pos0, vel0 = planet_state("earth", dep_mjd2000)
    q0, _ = cartesian_to_cylindrical(pos0, vel0)

    def acceleration_norm(tau: np.ndarray) -> np.ndarray:
        radial_values, radial_derivs, radial_integrals = reference_basis_matrices(
            radial_terms, tau, n_rev
        )
        transverse_values, transverse_derivs, _ = reference_basis_matrices(
            transverse_terms, tau, n_rev
        )
        axial_values, axial_derivs, axial_integrals = reference_basis_matrices(
            axial_terms, tau, n_rev
        )

        v_r = radial_values @ coeff_r
        v_theta = transverse_values @ coeff_theta
        v_z = axial_values @ coeff_z
        rho = q0[0] + tof_days * (radial_integrals @ coeff_r)
        z = q0[2] + tof_days * (axial_integrals @ coeff_z)
        if np.any(~np.isfinite(rho)) or np.any(rho <= 1.0e-10):
            raise ValueError("Invalid radial profile in reference evaluator")

        v_r_dot = (radial_derivs @ coeff_r) / tof_days
        v_theta_dot = (transverse_derivs @ coeff_theta) / tof_days
        v_z_dot = (axial_derivs @ coeff_z) / tof_days
        radius = np.sqrt(rho * rho + z * z)
        f_r = v_r_dot - v_theta * v_theta / rho + MU_SUN_AU_DAY * rho / radius**3
        f_theta = v_theta_dot + v_r * v_theta / rho
        f_z = v_z_dot + MU_SUN_AU_DAY * z / radius**3
        return np.sqrt(f_r * f_r + f_theta * f_theta + f_z * f_z)

    metrics = evaluate_reference_metrics(
        acceleration_norm,
        delta_v_scale_km_s=tof_days * AU_PER_DAY_TO_KM_PER_S,
        acceleration_scale_m_s2=AU_PER_DAY2_TO_M_PER_S2,
    )
    return metrics.as_dict()


def build_coefficient_rows(dep_grid: np.ndarray, tof_grid: np.ndarray, args: argparse.Namespace) -> list[dict]:
    basis = str(getattr(args, "basis", "CPowPow2-CPowPow2-CosN5P3CosN5P3SinN5"))
    target = str(getattr(args, "target", "mars"))
    n_quad = int(getattr(args, "n_quad", 401))
    n_min = int(getattr(args, "n_min", 0))
    n_max = int(getattr(args, "n_max", 5))
    radial_terms, transverse_terms, axial_terms = [parse_basis_group(part) for part in basis.split("-")]
    free = np.zeros(count_free_coefficients(radial_terms, transverse_terms, axial_terms), dtype=float)

    rows: list[dict] = []
    for tof in np.asarray(tof_grid, dtype=float):
        for dep in np.asarray(dep_grid, dtype=float):
            for n_rev in range(n_min, n_max + 1):
                branch_t0 = perf_counter()
                row = {
                    "departure_mjd2000": float(dep),
                    "tof_days": float(tof),
                    "N": int(n_rev),
                    "delta_v_km_s": float("nan"),
                    "fmax_m_s2": float("nan"),
                    "delta_v_optimizer_km_s": float("nan"),
                    "u_max_optimizer_m_s2": float("nan"),
                    "source_success": False,
                    "usable": False,
                    "message": "",
                    "free_coefficients": free.copy(),
                    "coefficient_reconstruction_success": False,
                    "coefficient_reconstruction_message": "",
                }
                try:
                    metrics = evaluate_time_driven_metrics(
                        float(dep),
                        float(tof),
                        int(n_rev),
                        n_quad,
                        basis,
                        free,
                        target=target,
                    )
                    if not isinstance(metrics, dict):
                        raise ValueError("Invalid trajectory metrics")
                    row["delta_v_km_s"] = float(metrics["delta_v_km_s"])
                    row["fmax_m_s2"] = float(metrics["fmax_m_s2"])
                    row["delta_v_optimizer_km_s"] = row["delta_v_km_s"]
                    row["u_max_optimizer_m_s2"] = row["fmax_m_s2"]
                    row["source_success"] = bool(
                        np.isfinite(row["delta_v_km_s"]) and np.isfinite(row["fmax_m_s2"])
                    )
                    row["usable"] = row["source_success"]
                except Exception as exc:
                    row["message"] = str(exc).splitlines()[-1]

                try:
                    coeff_r, coeff_theta, coeff_z = reconstruct_coefficients_from_free(
                        float(dep),
                        float(tof),
                        int(n_rev),
                        n_quad,
                        basis,
                        free,
                        target=target,
                    )
                    row["radial_coefficients"] = coeff_r
                    row["transverse_coefficients"] = coeff_theta
                    row["axial_coefficients"] = coeff_z
                    row["coefficient_reconstruction_success"] = True
                    reference = evaluate_time_driven_reference_metrics(
                        float(dep),
                        float(tof),
                        int(n_rev),
                        n_quad,
                        basis,
                        free,
                        target=target,
                        coefficients=(coeff_r, coeff_theta, coeff_z),
                    )
                    row.update(reference)
                    row["delta_v_km_s"] = float(reference["delta_v_reference_km_s"])
                    row["fmax_m_s2"] = float(reference["u_max_reference_m_s2"])
                    row["source_success"] = bool(
                        np.isfinite(row["delta_v_km_s"])
                        and np.isfinite(row["fmax_m_s2"])
                    )
                    row["usable"] = row["source_success"]
                except Exception as exc:
                    row["coefficient_reconstruction_message"] = str(exc).splitlines()[-1]
                    if not row["message"]:
                        row["message"] = row["coefficient_reconstruction_message"]

                row["wall_time_s"] = perf_counter() - branch_t0
                rows.append(row)
    return rows


def write_coefficients_npz(
    path: Path,
    rows: list[dict],
    dep_grid: np.ndarray,
    tof_grid: np.ndarray,
    args: argparse.Namespace,
) -> None:
    ordered = sorted(rows, key=lambda item: (item["tof_days"], item["departure_mjd2000"], item["N"]))
    basis = str(getattr(args, "basis", "CPowPow2-CPowPow2-CosN5P3CosN5P3SinN5"))
    radial_terms, transverse_terms, axial_terms = [parse_basis_group(part) for part in basis.split("-")]
    free_count = count_free_coefficients(radial_terms, transverse_terms, axial_terms)
    coeff_shapes = (len(radial_terms), len(transverse_terms), len(axial_terms))

    free_coefficients = np.full((len(ordered), free_count), np.nan, dtype=float)
    radial_coefficients = np.full((len(ordered), coeff_shapes[0]), np.nan, dtype=float)
    transverse_coefficients = np.full((len(ordered), coeff_shapes[1]), np.nan, dtype=float)
    axial_coefficients = np.full((len(ordered), coeff_shapes[2]), np.nan, dtype=float)
    coefficient_reconstruction_success = np.zeros(len(ordered), dtype=bool)
    coefficient_reconstruction_message = [""] * len(ordered)

    for idx, row in enumerate(ordered):
        free = np.asarray(row.get("free_coefficients", np.zeros(free_count)), dtype=float).reshape(-1)
        if free.shape[0] == free_count:
            free_coefficients[idx] = free
        for key, target_array, width in [
            ("radial_coefficients", radial_coefficients, coeff_shapes[0]),
            ("transverse_coefficients", transverse_coefficients, coeff_shapes[1]),
            ("axial_coefficients", axial_coefficients, coeff_shapes[2]),
        ]:
            values = np.asarray(row.get(key, []), dtype=float).reshape(-1)
            if values.shape[0] == width:
                target_array[idx] = values
        coefficient_reconstruction_success[idx] = bool(row.get("coefficient_reconstruction_success", False))
        coefficient_reconstruction_message[idx] = str(row.get("coefficient_reconstruction_message", ""))

    np.savez(
        path,
        departure_mjd2000=np.asarray([row["departure_mjd2000"] for row in ordered], dtype=float),
        tof_days=np.asarray([row["tof_days"] for row in ordered], dtype=float),
        N=np.asarray([row["N"] for row in ordered], dtype=int),
        delta_v_km_s=np.asarray([row.get("delta_v_km_s", np.nan) for row in ordered], dtype=float),
        delta_v_optimizer_km_s=np.asarray(
            [row.get("delta_v_optimizer_km_s", np.nan) for row in ordered],
            dtype=float,
        ),
        delta_v_reference_km_s=np.asarray(
            [row.get("delta_v_reference_km_s", np.nan) for row in ordered],
            dtype=float,
        ),
        delta_v_reference_error_km_s=np.asarray(
            [row.get("delta_v_reference_error_km_s", np.nan) for row in ordered],
            dtype=float,
        ),
        start_delta_v_km_s=np.asarray(
            [row.get("delta_v_optimizer_km_s", np.nan) for row in ordered],
            dtype=float,
        ),
        fmax_m_s2=np.asarray([row.get("fmax_m_s2", np.nan) for row in ordered], dtype=float),
        u_max_optimizer_m_s2=np.asarray(
            [row.get("u_max_optimizer_m_s2", np.nan) for row in ordered],
            dtype=float,
        ),
        u_max_reference_m_s2=np.asarray(
            [row.get("u_max_reference_m_s2", np.nan) for row in ordered],
            dtype=float,
        ),
        u_max_reference_error_m_s2=np.asarray(
            [row.get("u_max_reference_error_m_s2", np.nan) for row in ordered],
            dtype=float,
        ),
        reference_quadrature_order=np.asarray(
            [row.get("reference_quadrature_order", -1) for row in ordered],
            dtype=int,
        ),
        reference_evaluations=np.asarray(
            [row.get("reference_evaluations", 0) for row in ordered],
            dtype=int,
        ),
        reference_converged=np.asarray(
            [bool(row.get("reference_converged", False)) for row in ordered],
            dtype=bool,
        ),
        reference_metric_version=np.asarray(
            [str(row.get("reference_metric_version", "")) for row in ordered],
        ),
        wall_time_s=np.asarray([row.get("wall_time_s", np.nan) for row in ordered], dtype=float),
        source_success=np.asarray([bool(row.get("source_success", False)) for row in ordered], dtype=bool),
        optimizer_success=np.asarray([bool(row.get("source_success", False)) for row in ordered], dtype=bool),
        usable=np.asarray([bool(row.get("usable", False)) for row in ordered], dtype=bool),
        nfev=np.zeros(len(ordered), dtype=int),
        message=np.asarray([str(row.get("message", "")) for row in ordered]),
        free_coefficients=free_coefficients,
        radial_coefficients=radial_coefficients,
        transverse_coefficients=transverse_coefficients,
        axial_coefficients=axial_coefficients,
        coefficient_reconstruction_success=coefficient_reconstruction_success,
        coefficient_reconstruction_message=np.asarray(coefficient_reconstruction_message),
        grid_departure_mjd2000=np.asarray(dep_grid, dtype=float),
        grid_tof_days=np.asarray(tof_grid, dtype=float),
        basis=np.asarray(basis),
        radial_basis=np.asarray(basis.split("-")[0]),
        transverse_basis=np.asarray(basis.split("-")[1]),
        axial_basis=np.asarray(basis.split("-")[2]),
        target=np.asarray(str(getattr(args, "target", "mars"))),
        n_quad=np.asarray(int(getattr(args, "n_quad", 401)), dtype=int),
        n_min=np.asarray(int(getattr(args, "n_min", 0)), dtype=int),
        n_max=np.asarray(int(getattr(args, "n_max", 5)), dtype=int),
        free_coefficient_count=np.asarray(free_count, dtype=int),
        radial_coefficient_count=np.asarray(coeff_shapes[0], dtype=int),
        transverse_coefficient_count=np.asarray(coeff_shapes[1], dtype=int),
        axial_coefficient_count=np.asarray(coeff_shapes[2], dtype=int),
        **{key: np.asarray(value) for key, value in ephemeris_metadata().items()},
        coefficient_components=np.asarray(["v_r", "v_theta", "v_z"]),
        source_kind=np.asarray("gondelach_low_order_time_driven"),
        gondelach_formulation_version=np.asarray(GONDELACH_FORMULATION_VERSION),
    )


def compute_grid(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dep_grid = inclusive_grid(args.dep_min, args.dep_max, args.dep_step)
    tof_grid = inclusive_grid(args.tof_min, args.tof_max, args.tof_step)
    dv_grid = np.full((len(tof_grid), len(dep_grid)), np.nan)
    best_n_grid = np.full((len(tof_grid), len(dep_grid)), -1, dtype=int)

    total = len(dep_grid) * len(tof_grid)
    count = 0
    for i_tof, tof in enumerate(tof_grid):
        for i_dep, dep in enumerate(dep_grid):
            count += 1
            best_dv = float("inf")
            best_n = -1
            for n_rev in range(args.n_min, args.n_max + 1):
                try:
                    dv = evaluate_time_driven(
                        dep,
                        tof,
                        n_rev,
                        args.n_quad,
                        args.basis,
                        target=getattr(args, "target", "mars"),
                    )
                except Exception:
                    dv = float("nan")
                if np.isfinite(dv) and dv < best_dv:
                    best_dv = float(dv)
                    best_n = int(n_rev)
            if best_n >= 0:
                dv_grid[i_tof, i_dep] = best_dv
                best_n_grid[i_tof, i_dep] = best_n
        if args.progress:
            print(f"row {i_tof + 1}/{len(tof_grid)} complete ({count}/{total})")
    return dep_grid, tof_grid, dv_grid, best_n_grid


def write_csv(path: Path, dep_grid: np.ndarray, tof_grid: np.ndarray, dv_grid: np.ndarray, best_n_grid: np.ndarray) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["departure_mjd2000", "tof_days", "delta_v_km_s", "best_N"])
        for i_tof, tof in enumerate(tof_grid):
            for i_dep, dep in enumerate(dep_grid):
                writer.writerow([dep, tof, dv_grid[i_tof, i_dep], best_n_grid[i_tof, i_dep]])


def write_timing_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "method",
        "wall_time_s",
        "grid_points",
        "branch_attempts",
        "seconds_per_grid_point",
        "seconds_per_branch_attempt",
        "finite_points",
        "usable_attempts",
        "formal_success_attempts",
        "optimizer_function_evaluations",
        "notes",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_with_matplotlib(path: Path, dep_grid: np.ndarray, tof_grid: np.ndarray, dv_grid: np.ndarray, best_n_grid: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels = [6.0, 7.0, 8.0, 10.0, 15.0, 20.0, 40.0]
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    cf = ax.contourf(dep_grid, tof_grid, dv_grid, levels=levels, cmap="viridis_r", extend="both")
    cs = ax.contour(dep_grid, tof_grid, dv_grid, levels=levels, colors="black", linewidths=0.85)
    ax.clabel(cs, inline=True, fmt="%g", fontsize=9)
    ax.set_xlabel("Departure date [MJD2000]")
    ax.set_ylabel("Time of flight [days]")
    ax.set_title("Gondelach Fig. 2 reproduction: Mars, time-driven, N=0-5")
    cbar = fig.colorbar(cf, ax=ax, ticks=levels)
    cbar.set_label("Delta V [km/s]")
    ax.set_xlim(float(dep_grid[0]), float(dep_grid[-1]))
    ax.set_ylim(float(tof_grid[0]), float(tof_grid[-1]))
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    n_path = path.with_name(path.stem + "_best_N.png")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    image = ax.imshow(
        best_n_grid,
        origin="lower",
        aspect="auto",
        extent=[float(dep_grid[0]), float(dep_grid[-1]), float(tof_grid[0]), float(tof_grid[-1])],
        cmap="tab10",
        vmin=-0.5,
        vmax=5.5,
    )
    ax.set_xlabel("Departure date [MJD2000]")
    ax.set_ylabel("Time of flight [days]")
    ax.set_title("Best revolution parameter N selected at each grid point, N=0-5")
    cbar = fig.colorbar(image, ax=ax, ticks=range(0, 6))
    cbar.set_label("best N")
    fig.tight_layout()
    fig.savefig(n_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dep-min", type=float, default=7304.0)
    parser.add_argument("--dep-max", type=float, default=10225.0)
    parser.add_argument("--tof-min", type=float, default=500.0)
    parser.add_argument("--tof-max", type=float, default=2000.0)
    parser.add_argument("--dep-step", type=float, default=20.0)
    parser.add_argument("--tof-step", type=float, default=20.0)
    parser.add_argument("--n-quad", type=int, default=401)
    parser.add_argument("--n-min", type=int, default=0)
    parser.add_argument("--n-max", type=int, default=5)
    parser.add_argument("--basis", default="CPowPow2-CPowPow2-CosN5P3CosN5P3SinN5")
    parser.add_argument("--target", choices=["mars", "1989ml", "tempel1", "mercury"], default="mars")
    parser.add_argument("--ephemeris", choices=["kepler", "spice"], default="kepler")
    parser.add_argument("--spice-meta-kernel", default=None)
    parser.add_argument("--spice-target-name", default=None)
    parser.add_argument("--output-dir", default="output/gondelach_fig2")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    configure_ephemeris(args.ephemeris, args.spice_meta_kernel, args.spice_target_name)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = perf_counter()
    dep_grid, tof_grid, dv_grid, best_n_grid = compute_grid(args)
    grid_time = perf_counter() - t0

    t0 = perf_counter()
    coefficient_rows = build_coefficient_rows(dep_grid, tof_grid, args)
    coefficient_time = perf_counter() - t0

    npz_path = output_dir / "gondelach_fig2_reproduction.npz"
    coeff_path = output_dir / "gondelach_fig2_reproduction_coefficients.npz"
    csv_path = output_dir / "gondelach_fig2_reproduction.csv"
    timing_path = output_dir / "gondelach_fig2_reproduction_timing.csv"
    png_path = output_dir / "gondelach_fig2_reproduction.png"
    np.savez(
        npz_path,
        departure_mjd2000=dep_grid,
        tof_days=tof_grid,
        delta_v_km_s=dv_grid,
        best_N=best_n_grid,
        basis=args.basis,
    )
    write_coefficients_npz(coeff_path, coefficient_rows, dep_grid, tof_grid, args)
    write_csv(csv_path, dep_grid, tof_grid, dv_grid, best_n_grid)
    grid_points = int(len(dep_grid) * len(tof_grid))
    branch_attempts = int(grid_points * (args.n_max - args.n_min + 1))
    write_timing_csv(
        timing_path,
        [
            {
                "method": "gondelach_fig2_grid",
                "wall_time_s": grid_time,
                "grid_points": grid_points,
                "branch_attempts": branch_attempts,
                "seconds_per_grid_point": grid_time / max(grid_points, 1),
                "seconds_per_branch_attempt": grid_time / max(branch_attempts, 1),
                "finite_points": int(np.isfinite(dv_grid).sum()),
                "usable_attempts": "",
                "formal_success_attempts": "",
                "optimizer_function_evaluations": "",
                "notes": "direct lower-order evaluator",
            },
            {
                "method": "gondelach_fig2_coefficients",
                "wall_time_s": coefficient_time,
                "grid_points": grid_points,
                "branch_attempts": branch_attempts,
                "seconds_per_grid_point": coefficient_time / max(grid_points, 1),
                "seconds_per_branch_attempt": coefficient_time / max(branch_attempts, 1),
                "finite_points": "",
                "usable_attempts": int(sum(bool(row.get("usable", False)) for row in coefficient_rows)),
                "formal_success_attempts": int(sum(bool(row.get("source_success", False)) for row in coefficient_rows)),
                "optimizer_function_evaluations": "",
                "notes": "low-order coefficient archive reconstruction",
            },
            {
                "method": "overall_compute",
                "wall_time_s": grid_time + coefficient_time,
                "grid_points": grid_points,
                "branch_attempts": branch_attempts,
                "seconds_per_grid_point": (grid_time + coefficient_time) / max(grid_points, 1),
                "seconds_per_branch_attempt": (grid_time + coefficient_time) / max(branch_attempts, 1),
                "finite_points": int(np.isfinite(dv_grid).sum()),
                "usable_attempts": int(sum(bool(row.get("usable", False)) for row in coefficient_rows)),
                "formal_success_attempts": "",
                "optimizer_function_evaluations": "",
                "notes": "grid plus coefficient archive reconstruction",
            },
        ],
    )
    plot_with_matplotlib(png_path, dep_grid, tof_grid, dv_grid, best_n_grid)

    finite = dv_grid[np.isfinite(dv_grid)]
    if finite.size:
        best_idx = np.unravel_index(int(np.nanargmin(dv_grid)), dv_grid.shape)
        print(f"wrote {png_path}")
        print(f"wrote {csv_path}")
        print(f"wrote {coeff_path}")
        print(f"wrote {timing_path}")
        print(f"wrote {npz_path}")
        print(
            "best grid point: "
            f"dep={dep_grid[best_idx[1]]:.1f} MJD2000, "
            f"TOF={tof_grid[best_idx[0]]:.1f} days, "
            f"DeltaV={dv_grid[best_idx]:.3f} km/s, "
            f"N={best_n_grid[best_idx]}"
        )
    else:
        print("No finite solutions found.")


if __name__ == "__main__":
    main()

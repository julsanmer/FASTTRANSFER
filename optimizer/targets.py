"""Shared target and angle helpers for FASTTRANSFER command-line tools."""

from __future__ import annotations

import numpy as np


DEFAULT_MEE0 = np.array([1.0, 0.01, -0.01, 0.001, -0.001, 0.0])
DEFAULT_MEE_TARGET_EPOCH = np.array([1.5, 0.2, 0.05, 0.002, 0.001, np.pi / 4])


def dionysus_target_mee() -> np.ndarray:
    a = 2.20
    e = 0.54
    inc = np.radians(13.6)
    raan = np.radians(82.8)
    argp = np.radians(204.5)
    L0 = np.pi

    f = e * np.cos(argp + raan)
    g = e * np.sin(argp + raan)
    h = np.tan(inc / 2.0) * np.cos(raan)
    k = np.tan(inc / 2.0) * np.sin(raan)
    p = a * (1.0 - e**2)

    return np.array([p, f, g, h, k, L0])


def mars_target_mee() -> np.ndarray:
    """Approximate heliocentric Mars MEE elements."""
    a = 1.52371034
    e = 0.09339410
    inc = np.radians(1.84969142)
    raan = np.radians(49.55953891)
    lon_peri = np.radians(-23.94362959)
    L0 = 0.0

    f = e * np.cos(lon_peri)
    g = e * np.sin(lon_peri)
    h = np.tan(inc / 2.0) * np.cos(raan)
    k = np.tan(inc / 2.0) * np.sin(raan)
    p = a * (1.0 - e**2)

    return np.array([p, f, g, h, k, L0])


def target_for_name(target: str) -> tuple[np.ndarray, np.ndarray]:
    """Return fresh departure and target MEE arrays for a named case."""
    if target == "default":
        return DEFAULT_MEE0.copy(), DEFAULT_MEE_TARGET_EPOCH.copy()
    if target == "dionysus":
        return DEFAULT_MEE0.copy(), dionysus_target_mee()
    if target == "mars":
        return DEFAULT_MEE0.copy(), mars_target_mee()
    raise ValueError(f"Unknown target: {target}")


def wrap_0_2pi(angle: float) -> float:
    return float(np.mod(angle, 2.0 * np.pi))


def wrap_minus_pi_pi(angle: float) -> float:
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def rev_defaults(_mee0: np.ndarray, _mee_target_epoch: np.ndarray) -> tuple[float, float]:
    return 1.0, 2.0

"""Canonical unit conversions for heliocentric transfers.

External CLI units remain AU and years.  Internal canonical units use
UD = 1 AU, UV = sqrt(mu_sun / UD) = 2*pi AU/yr,
UT = UD / UV = 1/(2*pi) yr, UA = UV / UT, and mu = 1.
"""

from __future__ import annotations

import numpy as np


MU_AU_YR = 4.0 * np.pi**2
MU_CANONICAL = 1.0
UD_AU = 1.0
UV_AU_PER_YR = float(np.sqrt(MU_AU_YR / UD_AU))
UT_YR = UD_AU / UV_AU_PER_YR
UA_AU_PER_YR2 = UV_AU_PER_YR / UT_YR


def time_to_canonical(t_year: float) -> float:
    return float(t_year) / UT_YR


def time_from_canonical(t_canonical: float) -> float:
    return float(t_canonical) * UT_YR


def velocity_to_canonical(v_au_per_yr):
    return np.asarray(v_au_per_yr, dtype=float) / UV_AU_PER_YR


def velocity_from_canonical(v_canonical):
    return np.asarray(v_canonical, dtype=float) * UV_AU_PER_YR


def accel_to_canonical(a_au_per_yr2: float | None):
    if a_au_per_yr2 is None:
        return None
    return float(a_au_per_yr2) / UA_AU_PER_YR2


def accel_from_canonical(a_canonical: float | None):
    if a_canonical is None:
        return None
    return float(a_canonical) * UA_AU_PER_YR2


def dv_from_canonical(dv_canonical: float) -> float:
    return float(dv_canonical) * UV_AU_PER_YR


def dv_to_canonical(dv_au_per_yr: float) -> float:
    return float(dv_au_per_yr) / UV_AU_PER_YR


def energy_from_canonical(energy_canonical: float) -> float:
    return float(energy_canonical) * UA_AU_PER_YR2**2 * UT_YR


def energy_to_canonical(energy_au_yr: float) -> float:
    return float(energy_au_yr) / (UA_AU_PER_YR2**2 * UT_YR)


def time_array_from_canonical(t_canonical):
    return np.asarray(t_canonical, dtype=float) * UT_YR


def accel_array_from_canonical(a_canonical):
    return np.asarray(a_canonical, dtype=float) * UA_AU_PER_YR2

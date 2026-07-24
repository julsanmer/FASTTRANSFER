"""Lazy, process-local SPICE state access for fixed-time transfer grids."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np


AU_KM = 149_597_870.7
DAY_S = 86_400.0
MJD2000_STANDARD_MJD = 51_544.5

_loaded_pid: int | None = None
_loaded_meta_kernel: str | None = None


def _spice_module():
    try:
        import spiceypy as spice
    except ImportError as exc:
        raise ImportError(
            "SPICE ephemerides require SpiceyPy. Install it in the active Python "
            "environment with: python3.13 -m pip install spiceypy"
        ) from exc
    return spice


def ensure_kernels_loaded(meta_kernel: str | Path) -> str:
    global _loaded_pid, _loaded_meta_kernel
    path = str(Path(meta_kernel).expanduser().resolve())
    if not Path(path).is_file():
        raise FileNotFoundError(f"SPICE meta-kernel not found: {path}")
    pid = os.getpid()
    if _loaded_pid != pid or _loaded_meta_kernel != path:
        spice = _spice_module()
        spice.kclear()
        spice.furnsh(path)
        _loaded_pid = pid
        _loaded_meta_kernel = path
        spice_state.cache_clear()
    return path


@lru_cache(maxsize=16_384)
def spice_state(
    body: str,
    mjd2000: float,
    meta_kernel: str,
    frame: str = "ECLIPJ2000",
    aberration: str = "NONE",
    observer: str = "SUN",
) -> tuple[np.ndarray, np.ndarray]:
    """Return heliocentric position [AU] and velocity [AU/day]."""
    path = ensure_kernels_loaded(meta_kernel)
    spice = _spice_module()
    standard_mjd = MJD2000_STANDARD_MJD + float(mjd2000)
    jd_utc = standard_mjd + 2_400_000.5
    et = spice.str2et(f"JD {jd_utc:.16f} UTC")
    state, _ = spice.spkezr(
        str(body),
        et,
        str(frame),
        str(aberration),
        str(observer),
    )
    state = np.asarray(state, dtype=float)
    return state[:3] / AU_KM, state[3:] * DAY_S / AU_KM


def spice_metadata(meta_kernel: str | Path) -> dict[str, str]:
    path = ensure_kernels_loaded(meta_kernel)
    spice = _spice_module()
    return {
        "ephemeris_source": "spice",
        "spice_meta_kernel": path,
        "spice_frame": "ECLIPJ2000",
        "spice_observer": "SUN",
        "spice_aberration": "NONE",
        "spiceypy_version": str(getattr(spice, "__version__", "unknown")),
        "cspice_version": str(spice.tkvrsn("TOOLKIT")),
    }

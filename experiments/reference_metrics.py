"""Shared, accuracy-controlled metrics for trajectory comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np


REFERENCE_METRIC_VERSION = "composite_gauss_legendre_v1"
REFERENCE_LOW_ORDER = 8
REFERENCE_HIGH_ORDER = 12
REFERENCE_FALLBACK_ORDER = 16
REFERENCE_BASE_INTERVALS = 32
REFERENCE_DV_ABS_TOL_KM_S = 1.0e-6
REFERENCE_DV_REL_TOL = 1.0e-8
REFERENCE_UMAX_ABS_TOL_M_S2 = 1.0e-9
REFERENCE_UMAX_REL_TOL = 1.0e-6
STANDARD_BSPLINE_VARIANTS = ((10, 3), (10, 5), (40, 3), (40, 5))


@dataclass(frozen=True)
class ReferenceMetrics:
    delta_v_km_s: float
    u_max_m_s2: float
    delta_v_error_km_s: float
    u_max_error_m_s2: float
    quadrature_order: int
    evaluations: int
    converged: bool
    metric_version: str = REFERENCE_METRIC_VERSION

    def as_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "delta_v_reference_km_s": self.delta_v_km_s,
            "u_max_reference_m_s2": self.u_max_m_s2,
            "delta_v_reference_error_km_s": self.delta_v_error_km_s,
            "u_max_reference_error_m_s2": self.u_max_error_m_s2,
            "reference_quadrature_order": self.quadrature_order,
            "reference_evaluations": self.evaluations,
            "reference_converged": self.converged,
            "reference_metric_version": self.metric_version,
        }


def _standard_internal_knots(n_ctrl: int, degree: int) -> np.ndarray:
    spans = int(n_ctrl) - int(degree)
    if spans <= 1:
        return np.empty(0, dtype=float)
    return np.arange(1, spans, dtype=float) / float(spans)


@lru_cache(maxsize=1)
def reference_breakpoints() -> np.ndarray:
    parts = [np.linspace(0.0, 1.0, REFERENCE_BASE_INTERVALS + 1)]
    parts.extend(_standard_internal_knots(n_ctrl, degree) for n_ctrl, degree in STANDARD_BSPLINE_VARIANTS)
    points = np.unique(np.round(np.concatenate(parts), 15))
    points[0] = 0.0
    points[-1] = 1.0
    return points


@lru_cache(maxsize=None)
def composite_gauss_legendre_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    order = int(order)
    if order < 1:
        raise ValueError("Gauss-Legendre order must be positive")
    xi, wi = np.polynomial.legendre.leggauss(order)
    breaks = reference_breakpoints()
    nodes = []
    weights = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        nodes.append(0.5 * (right - left) * xi + 0.5 * (left + right))
        weights.append(0.5 * (right - left) * wi)
    return np.concatenate(nodes), np.concatenate(weights)


def _within_tolerance(value: float, error: float, absolute: float, relative: float) -> bool:
    return bool(np.isfinite(value) and np.isfinite(error) and error <= absolute + relative * abs(value))


def evaluate_reference_metrics(
    acceleration_norm: Callable[[np.ndarray], np.ndarray],
    delta_v_scale_km_s: float,
    acceleration_scale_m_s2: float,
) -> ReferenceMetrics:
    breaks = reference_breakpoints()
    break_values = np.asarray(acceleration_norm(breaks), dtype=float).reshape(-1)
    if break_values.shape != breaks.shape:
        raise ValueError("Acceleration sampler returned an unexpected shape")

    evaluations = len(breaks)
    previous_dv = float("nan")
    previous_umax = float("nan")
    final_dv = float("nan")
    final_umax = float("nan")
    dv_error = float("inf")
    umax_error = float("inf")
    final_order = REFERENCE_HIGH_ORDER
    converged = False

    for order in (REFERENCE_LOW_ORDER, REFERENCE_HIGH_ORDER, REFERENCE_FALLBACK_ORDER):
        nodes, weights = composite_gauss_legendre_rule(order)
        values = np.asarray(acceleration_norm(nodes), dtype=float).reshape(-1)
        if values.shape != nodes.shape:
            raise ValueError("Acceleration sampler returned an unexpected shape")
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Acceleration sampler returned invalid values")

        evaluations += len(nodes)
        final_dv = float(delta_v_scale_km_s * np.sum(weights * values))
        final_umax = float(acceleration_scale_m_s2 * max(np.max(values), np.max(break_values)))
        final_order = order

        if np.isfinite(previous_dv):
            dv_error = abs(final_dv - previous_dv)
            umax_error = abs(final_umax - previous_umax)
            converged = _within_tolerance(
                final_dv,
                dv_error,
                REFERENCE_DV_ABS_TOL_KM_S,
                REFERENCE_DV_REL_TOL,
            ) and _within_tolerance(
                final_umax,
                umax_error,
                REFERENCE_UMAX_ABS_TOL_M_S2,
                REFERENCE_UMAX_REL_TOL,
            )
            if converged and order >= REFERENCE_HIGH_ORDER:
                break

        previous_dv = final_dv
        previous_umax = final_umax

    return ReferenceMetrics(
        delta_v_km_s=final_dv,
        u_max_m_s2=final_umax,
        delta_v_error_km_s=dv_error,
        u_max_error_m_s2=umax_error,
        quadrature_order=final_order,
        evaluations=evaluations,
        converged=converged,
    )

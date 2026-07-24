"""O'Neill AS47 variant of the Nelder-Mead simplex method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class OneillResult:
    x: np.ndarray
    fun: float
    nfev: int
    nit: int
    num_restarts: int
    success: bool
    message: str


class _EvaluationBudgetExceeded(RuntimeError):
    pass


def minimize_oneill(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    step: float | np.ndarray,
    reqmin: float = 1.0e-8,
    konvge: int = 10,
    maxfev: int = 5000,
    factorial_epsilon: float = 1.0e-3,
) -> OneillResult:
    """Minimize ``fun`` using O'Neill's Algorithm AS47.

    This follows the public AS47 implementation, including convergence based
    on simplex-value variance and coordinate factorial tests with restarts.
    Function evaluations are capped strictly at ``maxfev``.
    """
    start = np.asarray(x0, dtype=float).reshape(-1).copy()
    n = int(start.size)
    step_array = np.broadcast_to(np.asarray(step, dtype=float), start.shape).copy()
    if n < 1:
        raise ValueError("x0 must contain at least one variable")
    if reqmin <= 0.0:
        raise ValueError("reqmin must be positive")
    if konvge < 1:
        raise ValueError("konvge must be at least 1")
    if maxfev < n + 1:
        raise ValueError("maxfev must allow evaluation of the initial simplex")
    if factorial_epsilon <= 0.0:
        raise ValueError("factorial_epsilon must be positive")
    if np.any(~np.isfinite(step_array)) or np.any(step_array <= 0.0):
        raise ValueError("all simplex steps must be finite and positive")

    nfev = 0
    nit = 0
    num_restarts = 0
    best_x = start.copy()
    best_fun = float("inf")

    def evaluate(x: np.ndarray) -> float:
        nonlocal nfev, best_x, best_fun
        if nfev >= maxfev:
            raise _EvaluationBudgetExceeded
        value = float(fun(np.asarray(x, dtype=float)))
        nfev += 1
        if value < best_fun:
            best_fun = value
            best_x = np.asarray(x, dtype=float).copy()
        return value

    delta = 1.0
    variance_limit = float(reqmin) * n

    try:
        while True:
            simplex = np.tile(start, (n + 1, 1))
            values = np.empty(n + 1, dtype=float)
            values[n] = evaluate(start)
            for axis in range(n):
                simplex[axis, axis] += step_array[axis] * delta
                values[axis] = evaluate(simplex[axis])

            checks_remaining = int(konvge)
            converged = False
            while nfev < maxfev:
                highest = int(np.argmax(values))
                lowest = int(np.argmin(values))
                centroid = (np.sum(simplex, axis=0) - simplex[highest]) / n
                reflected = 2.0 * centroid - simplex[highest]
                reflected_value = evaluate(reflected)

                if reflected_value < values[lowest]:
                    expanded = 3.0 * centroid - 2.0 * simplex[highest]
                    expanded_value = evaluate(expanded)
                    if reflected_value < expanded_value:
                        simplex[highest] = reflected
                        values[highest] = reflected_value
                    else:
                        simplex[highest] = expanded
                        values[highest] = expanded_value
                else:
                    better_count = int(np.count_nonzero(reflected_value < values))
                    if better_count > 1:
                        simplex[highest] = reflected
                        values[highest] = reflected_value
                    elif better_count == 0:
                        contracted = 0.5 * (centroid + simplex[highest])
                        contracted_value = evaluate(contracted)
                        if values[highest] < contracted_value:
                            lowest = int(np.argmin(values))
                            for vertex in range(n + 1):
                                simplex[vertex] = 0.5 * (simplex[vertex] + simplex[lowest])
                                values[vertex] = evaluate(simplex[vertex])
                        else:
                            simplex[highest] = contracted
                            values[highest] = contracted_value
                    else:
                        contracted = 0.5 * (centroid + reflected)
                        contracted_value = evaluate(contracted)
                        if contracted_value <= reflected_value:
                            simplex[highest] = contracted
                            values[highest] = contracted_value
                        else:
                            simplex[highest] = reflected
                            values[highest] = reflected_value

                nit += 1
                checks_remaining -= 1
                if checks_remaining > 0:
                    continue
                checks_remaining = int(konvge)
                mean_value = float(np.mean(values))
                if float(np.sum((values - mean_value) ** 2)) <= variance_limit:
                    converged = True
                    break

            if not converged:
                raise _EvaluationBudgetExceeded

            lowest = int(np.argmin(values))
            candidate_x = simplex[lowest].copy()
            candidate_value = float(values[lowest])
            restart_x: np.ndarray | None = None
            for axis in range(n):
                perturbation = step_array[axis] * factorial_epsilon
                positive = candidate_x.copy()
                positive[axis] += perturbation
                if evaluate(positive) < candidate_value:
                    restart_x = positive
                    break
                negative = candidate_x.copy()
                negative[axis] -= perturbation
                if evaluate(negative) < candidate_value:
                    restart_x = negative
                    break

            if restart_x is None:
                return OneillResult(
                    x=candidate_x,
                    fun=candidate_value,
                    nfev=nfev,
                    nit=nit,
                    num_restarts=num_restarts,
                    success=True,
                    message="O'Neill factorial tests passed",
                )

            start = restart_x
            delta = float(factorial_epsilon)
            num_restarts += 1
    except _EvaluationBudgetExceeded:
        return OneillResult(
            x=best_x,
            fun=best_fun,
            nfev=nfev,
            nit=nit,
            num_restarts=num_restarts,
            success=False,
            message="Maximum number of function evaluations exceeded",
        )

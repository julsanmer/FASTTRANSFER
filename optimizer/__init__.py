"""Public API for the active FASTTRANSFER optimization pipeline.

The package exposes the solver entry points lazily so lightweight helpers such
as unit conversion and target selection do not import every optimization backend.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "DEFAULT_MEE0": ("optimizer.targets", "DEFAULT_MEE0"),
    "DEFAULT_MEE_TARGET_EPOCH": ("optimizer.targets", "DEFAULT_MEE_TARGET_EPOCH"),
    "DEFAULT_MU": ("optimizer.canonical_units", "MU_CANONICAL"),
    "MU_AU_YR": ("optimizer.canonical_units", "MU_AU_YR"),
    "MU_CANONICAL": ("optimizer.canonical_units", "MU_CANONICAL"),
    "UA_AU_PER_YR2": ("optimizer.canonical_units", "UA_AU_PER_YR2"),
    "UD_AU": ("optimizer.canonical_units", "UD_AU"),
    "UT_YR": ("optimizer.canonical_units", "UT_YR"),
    "UV_AU_PER_YR": ("optimizer.canonical_units", "UV_AU_PER_YR"),
    "CartesianBVPJacOptions": ("optimizer.optimization_cartesian_indirect_solve_bvp_jac", "CartesianBVPJacOptions"),
    "IndirectRadauRootOptions": ("optimizer.optimization_MEE_indirect_Radaucollocation_root", "IndirectRadauRootOptions"),
    "RadauCoefficients": ("optimizer.optimization_MEE_Radaucollocation", "RadauCoefficients"),
    "SolveBVPJacOptions": ("optimizer.optimization_MEE_indirect_solve_bvp_jac", "SolveBVPJacOptions"),
    "accel_array_from_canonical": ("optimizer.canonical_units", "accel_array_from_canonical"),
    "accel_from_canonical": ("optimizer.canonical_units", "accel_from_canonical"),
    "accel_to_canonical": ("optimizer.canonical_units", "accel_to_canonical"),
    "apply_dv_sym": ("optimizer.orbit_utils", "apply_dv_sym"),
    "build_radau_coefficients": ("optimizer.optimization_MEE_Radaucollocation", "build_radau_coefficients"),
    "dionysus_target_mee": ("optimizer.targets", "dionysus_target_mee"),
    "dv_from_canonical": ("optimizer.canonical_units", "dv_from_canonical"),
    "dv_to_canonical": ("optimizer.canonical_units", "dv_to_canonical"),
    "energy_from_canonical": ("optimizer.canonical_units", "energy_from_canonical"),
    "energy_to_canonical": ("optimizer.canonical_units", "energy_to_canonical"),
    "get_kepler_substeps": ("optimizer.orbit_utils", "get_kepler_substeps"),
    "infer_longitude_branch_from_solution": (
        "optimizer.optimization_MEE_Radaucollocation",
        "infer_longitude_branch_from_solution",
    ),
    "kepler_coast_sym": ("optimizer.orbit_utils", "kepler_coast_sym"),
    "mars_target_mee": ("optimizer.targets", "mars_target_mee"),
    "mean_a_transfer_period": ("optimizer.optimization_MEE_Radaucollocation", "mean_a_transfer_period"),
    "mee_gauss_rhs_sym": ("optimizer.optimization_MEE_Radaucollocation", "mee_gauss_rhs_sym"),
    "mee_semimajor_axis": ("optimizer.optimization_MEE_Radaucollocation", "mee_semimajor_axis"),
    "mee_to_rv_sym": ("optimizer.orbit_utils", "mee_to_rv_sym"),
    "propagate_L": ("optimizer.orbit_utils", "propagate_L"),
    "rev_defaults": ("optimizer.targets", "rev_defaults"),
    "rv_to_mee_sym": ("optimizer.orbit_utils", "rv_to_mee_sym"),
    "solve_cartesian_indirect_bvp_from_bvp_jac": (
        "optimizer.optimization_cartesian_indirect_solve_bvp_jac",
        "solve_cartesian_indirect_bvp_from_bvp_jac",
    ),
    "solve_cartesian_indirect_bvp_from_radau_jac": (
        "optimizer.optimization_cartesian_indirect_solve_bvp_jac",
        "solve_cartesian_indirect_bvp_from_radau_jac",
    ),
    "solve_cartesian_indirect_bvp_jac": (
        "optimizer.optimization_cartesian_indirect_solve_bvp_jac",
        "solve_cartesian_indirect_bvp_jac",
    ),
    "solve_free_tf_cartesian_bspline": ("optimizer.optimization_Bspline_freetf", "solve_free_tf_cartesian_bspline"),
    "solve_free_tf_cylindrical_bspline": (
        "optimizer.optimization_Bspline_freetf",
        "solve_free_tf_cylindrical_bspline",
    ),
    "solve_gondelach_hodographic": ("optimizer.optimization_Gondelach_hodographic", "solve_gondelach_hodographic"),
    "solve_cylindrical_radau_collocation": (
        "optimizer.optimization_cylindrical_Radaucollocation",
        "solve_cylindrical_radau_collocation",
    ),
    "solve_indirect_bvp_from_bvp_jac": (
        "optimizer.optimization_MEE_indirect_solve_bvp_jac",
        "solve_indirect_bvp_from_bvp_jac",
    ),
    "solve_indirect_bvp_from_radau_jac": (
        "optimizer.optimization_MEE_indirect_solve_bvp_jac",
        "solve_indirect_bvp_from_radau_jac",
    ),
    "solve_indirect_bvp_jac": ("optimizer.optimization_MEE_indirect_solve_bvp_jac", "solve_indirect_bvp_jac"),
    "solve_indirect_mee_radau_root": (
        "optimizer.optimization_MEE_indirect_Radaucollocation_root",
        "solve_indirect_mee_radau_root",
    ),
    "solve_mee_radau_collocation": ("optimizer.optimization_MEE_Radaucollocation", "solve_mee_radau_collocation"),
    "solve_with_internal_bspline_seed": ("optimizer.optimization_MEE_Radaucollocation", "solve_with_internal_bspline_seed"),
    "target_for_name": ("optimizer.targets", "target_for_name"),
    "target_mee_at_time": ("optimizer.optimization_MEE_Radaucollocation", "target_mee_at_time"),
    "time_array_from_canonical": ("optimizer.canonical_units", "time_array_from_canonical"),
    "time_from_canonical": ("optimizer.canonical_units", "time_from_canonical"),
    "time_to_canonical": ("optimizer.canonical_units", "time_to_canonical"),
    "velocity_from_canonical": ("optimizer.canonical_units", "velocity_from_canonical"),
    "velocity_to_canonical": ("optimizer.canonical_units", "velocity_to_canonical"),
    "wrap_0_2pi": ("optimizer.targets", "wrap_0_2pi"),
    "wrap_minus_pi_pi": ("optimizer.targets", "wrap_minus_pi_pi"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'optimizer' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

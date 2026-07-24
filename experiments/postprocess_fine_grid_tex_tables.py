"""Build publication-ready LaTeX tables for the four saved fine-grid campaigns."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_CASE_DIRS = {
    "mars": Path(
        "output/compare_gondelach_mars_2028_spice_bspline_dep20d_tof20d/"
        "first_synodic_departure_window/fine_dep5d_tof10d"
    ),
    "mercury": Path(
        "output/compare_gondelach_mercury_2028_spice_bspline_dep20d_tof20d/"
        "first_synodic_departure_window/fine_dep1d_tof6p5d"
    ),
    "tempel1": Path(
        "output/compare_gondelach_tempel1_2028_spice_bspline_dep20d_tof20d/"
        "full_orbital_period/fine_dep10d_tof10d"
    ),
    "1989ml": Path(
        "output/compare_gondelach_1989ml_2028_spice_bspline_dep20d_tof20d/"
        "first_synodic_departure_window/fine_dep5d_tof5d"
    ),
}

CASE_ORDER = ["mars", "mercury", "tempel1", "1989ml"]
CASE_LABELS = {
    "mars": "Mars",
    "mercury": "Mercury",
    "tempel1": "Tempel 1",
    "1989ml": "1989ML",
}
METHOD_ORDER = [
    "gondelach_fig3",
    "bspline_nctrl10_deg3",
    "bspline_nctrl10_deg5",
    "bspline_nctrl40_deg3",
    "bspline_nctrl40_deg5",
]
METHOD_LABELS = {
    "gondelach_fig3": "High-order hodographic",
    "bspline_nctrl10_deg3": r"Cubic B-spline ($n_c=10$)",
    "bspline_nctrl10_deg5": r"Quintic B-spline ($n_c=10$)",
    "bspline_nctrl40_deg3": r"Cubic B-spline ($n_c=40$)",
    "bspline_nctrl40_deg5": r"Quintic B-spline ($n_c=40$)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for case_name in CASE_ORDER:
        parser.add_argument(
            f"--{case_name}-dir",
            type=Path,
            default=DEFAULT_CASE_DIRS[case_name],
            help=f"Saved {CASE_LABELS[case_name]} fine-grid output directory.",
        )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/global_fine_grid_postprocessing"),
    )
    return parser.parse_args()


def read_keyed_rows(path: Path, key: str = "method") -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required postprocessing file: {path}")
    with path.open(newline="") as file:
        return {str(row[key]): dict(row) for row in csv.DictReader(file)}


def finite_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "--"


def format_integer(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{int(round(value)):,}".replace(",", r"\,")


def write_delta_v_table(path: Path, case_dirs: dict[str, Path]) -> None:
    lines = [
        r"\begin{table*}[htbp!]",
        r"\caption{Global $\Delta V$ statistics for the fine-grid transfer campaigns.}",
        r"\label{tab:fine_grid_delta_v_statistics}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Target & Method & Mean & Median & Minimum \\",
        r" & & \multicolumn{3}{c}{$\Delta V$ [km/s]} \\",
        r"\cmidrule(lr){3-5}",
        r"\midrule",
    ]
    for case_index, case_name in enumerate(CASE_ORDER):
        rows = read_keyed_rows(case_dirs[case_name] / "global_statistics_delta_v.csv")
        for method in METHOD_ORDER:
            if method not in rows:
                raise ValueError(f"Missing method {method} in {case_dirs[case_name] / 'global_statistics_delta_v.csv'}")
            row = rows[method]
            lines.append(
                " & ".join(
                    [
                        CASE_LABELS[case_name],
                        METHOD_LABELS[method],
                        format_float(finite_float(row, "mean_delta_v_km_s"), 3),
                        format_float(finite_float(row, "median_delta_v_km_s"), 3),
                        format_float(finite_float(row, "min_delta_v_km_s"), 3),
                    ]
                )
                + r" \\"
            )
        if case_index != len(CASE_ORDER) - 1:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines))


def write_timing_table(path: Path, case_dirs: dict[str, Path]) -> None:
    lines = [
        r"\begin{table*}[htbp!]",
        r"\caption{Computational performance for the fine-grid transfer campaigns. A grid attempt denotes one departure-date/TOF/revolution-branch evaluation. The estimated serial wall time is the sum of branch-local runtimes, removing the effect of B-spline worker parallelization, and the mean is the corresponding time per grid attempt.}",
        r"\label{tab:fine_grid_computational_times}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Target & Method & Grid attempts & Mean [s] & Serial time [h] \\",
        r"\midrule",
    ]
    for case_index, case_name in enumerate(CASE_ORDER):
        rows = read_keyed_rows(case_dirs[case_name] / "comparison_computational_time.csv")
        for method in METHOD_ORDER:
            if method not in rows:
                raise ValueError(
                    f"Missing method {method} in {case_dirs[case_name] / 'comparison_computational_time.csv'}"
                )
            row = rows[method]
            grid_points = finite_float(row, "grid_points_with_timing")
            branch_attempts = finite_float(row, "branch_attempts")
            branches_per_grid = branch_attempts / grid_points
            mean_grid_effort = finite_float(row, "mean_worker_seconds_per_grid_point")
            serial_hours = grid_points * mean_grid_effort / 3600.0
            lines.append(
                " & ".join(
                    [
                        CASE_LABELS[case_name],
                        METHOD_LABELS[method],
                        format_integer(branch_attempts),
                        format_float(mean_grid_effort / branches_per_grid, 3),
                        format_float(serial_hours, 2),
                    ]
                )
                + r" \\"
            )
        if case_index != len(CASE_ORDER) - 1:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    case_dirs = {case_name: Path(getattr(args, f"{case_name}_dir")) for case_name in CASE_ORDER}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    delta_path = args.output_dir / "delta_v_statistics_fine_grids.tex"
    timing_path = args.output_dir / "computational_times_fine_grids.tex"
    write_delta_v_table(delta_path, case_dirs)
    write_timing_table(timing_path, case_dirs)
    print(f"wrote {delta_path}")
    print(f"wrote {timing_path}")


if __name__ == "__main__":
    main()

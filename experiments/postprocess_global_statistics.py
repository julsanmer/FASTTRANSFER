"""Compute global grid statistics for Gondelach and B-spline comparisons.

The script reads saved ``comparison_grids*.npz`` files and summarizes each
method over departure-date/TOF grid points. B-spline variants are compared
pointwise against the Gondelach high-order grid (Fig. 3 basis).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_gondelach_fig2_fig3_bspline10 import CASE_CONFIGS  # noqa: E402
from experiments.postprocess_compare_gondelach_plots import (  # noqa: E402
    BsplineVariantSpec,
    bspline_variant_label,
    comparison_grids_path,
    parse_bspline_variant_spec,
)


DEFAULT_VARIANTS = ["10:5", "10:3", "40:5", "40:3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASE_CONFIGS),
        default=["mars", "1989ml", "tempel1", "mercury"],
    )
    parser.add_argument("--dep-step", type=float, default=20.0)
    parser.add_argument("--tof-step", type=float, default=20.0)
    parser.add_argument(
        "--output-root",
        default="output",
        help="Root containing compare_gondelach_* folders.",
    )
    parser.add_argument(
        "--output-dir-template",
        default="compare_gondelach_{output_name}_bspline_dep20d_tof20d",
        help="Folder template relative to --output-root. Fields: case, output_name.",
    )
    parser.add_argument(
        "--bspline-variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help="B-spline variants to include as n_ctrl:degree[:run_id].",
    )
    parser.add_argument(
        "--out",
        default="output/global_statistics_delta_v.csv",
        help="CSV path for method-vs-Gondelach statistics.",
    )
    parser.add_argument(
        "--best-out",
        default="output/global_statistics_best_method_by_grid_point.csv",
        help="CSV path for winner counts across methods at each common grid point.",
    )
    return parser.parse_args()


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def pct(count: int, total: int) -> float:
    return 100.0 * float(count) / float(total) if total else float("nan")


def method_grid_from_npz(path: Path, variant: BsplineVariantSpec | None) -> np.ndarray:
    data = np.load(path)
    if variant is None:
        return np.asarray(data["bspline10_delta_v_km_s"], dtype=float)
    if "variant_delta_v_km_s" not in data.files and variant.n_ctrl == 10 and variant.degree == 5 and not variant.run_id:
        return np.asarray(data["bspline10_delta_v_km_s"], dtype=float)
    return np.asarray(data["variant_delta_v_km_s"], dtype=float)


def grid_summary(
    case_name: str,
    body_label: str,
    method_key: str,
    method_label: str,
    grid: np.ndarray,
    gondelach_grid: np.ndarray,
    total_grid_points: int,
) -> dict[str, object]:
    finite = np.isfinite(grid)
    common = finite & np.isfinite(gondelach_grid)
    values = finite_values(grid)
    common_values = np.asarray(grid[common], dtype=float)
    common_gondelach = np.asarray(gondelach_grid[common], dtype=float)
    diff = common_values - common_gondelach
    rel = 100.0 * diff / common_gondelach
    better = diff < 0.0
    equal = np.isclose(diff, 0.0, rtol=1e-10, atol=1e-10)
    worse = diff > 0.0
    return {
        "case": case_name,
        "target": body_label,
        "method": method_key,
        "method_label": method_label,
        "total_grid_points": total_grid_points,
        "finite_points": int(finite.sum()),
        "finite_percent": pct(int(finite.sum()), total_grid_points),
        "common_with_gondelach_points": int(common.sum()),
        "mean_delta_v_km_s": float(np.nanmean(values)) if values.size else float("nan"),
        "median_delta_v_km_s": float(np.nanmedian(values)) if values.size else float("nan"),
        "min_delta_v_km_s": float(np.nanmin(values)) if values.size else float("nan"),
        "max_delta_v_km_s": float(np.nanmax(values)) if values.size else float("nan"),
        "mean_minus_gondelach_km_s": float(np.nanmean(diff)) if diff.size else float("nan"),
        "median_minus_gondelach_km_s": float(np.nanmedian(diff)) if diff.size else float("nan"),
        "mean_percent_minus_gondelach": float(np.nanmean(rel)) if rel.size else float("nan"),
        "median_percent_minus_gondelach": float(np.nanmedian(rel)) if rel.size else float("nan"),
        "better_than_gondelach_points": int(better.sum()),
        "better_than_gondelach_percent": pct(int(better.sum()), int(common.sum())),
        "equal_to_gondelach_points": int(equal.sum()),
        "equal_to_gondelach_percent": pct(int(equal.sum()), int(common.sum())),
        "worse_than_gondelach_points": int(worse.sum()),
        "worse_than_gondelach_percent": pct(int(worse.sum()), int(common.sum())),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def output_dir_for_case(args: argparse.Namespace, case_name: str) -> Path:
    case = CASE_CONFIGS[case_name]
    folder = args.output_dir_template.format(
        case=case_name,
        output_name=case["output_name"],
    )
    return Path(args.output_root) / folder


def compute_case_rows(
    case_name: str,
    output_dir: Path,
    variants: list[BsplineVariantSpec],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    case = CASE_CONFIGS[case_name]
    body_label = str(case["display"])
    base_path = comparison_grids_path(output_dir, None)
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base grid: {base_path}")
    base = np.load(base_path)
    fig3 = np.asarray(base["fig3_delta_v_km_s"], dtype=float)
    total_points = int(fig3.size)

    rows: list[dict[str, object]] = []
    rows.append(
        grid_summary(
            case_name,
            body_label,
            "gondelach_fig3",
            "Gondelach high-order",
            fig3,
            fig3,
            total_points,
        )
    )
    if "fig2_delta_v_km_s" in base.files:
        rows.append(
            grid_summary(
                case_name,
                body_label,
                "gondelach_fig2",
                "Gondelach low-order",
                np.asarray(base["fig2_delta_v_km_s"], dtype=float),
                fig3,
                total_points,
            )
        )

    method_grids: dict[str, np.ndarray] = {"gondelach_fig3": fig3}
    method_labels: dict[str, str] = {"gondelach_fig3": "Gondelach high-order"}

    for variant in variants:
        key = variant.key
        path = comparison_grids_path(output_dir, variant)
        if not path.exists():
            print(f"warning: missing {path}", file=sys.stderr)
            continue
        grid = method_grid_from_npz(path, variant)
        label = bspline_variant_label(variant)
        rows.append(grid_summary(case_name, body_label, key, label, grid, fig3, total_points))
        method_grids[key] = grid
        method_labels[key] = label

    common_mask = np.isfinite(fig3)
    for grid in method_grids.values():
        common_mask &= np.isfinite(grid)
    common_count = int(common_mask.sum())
    winner_counts = {key: 0 for key in method_grids}
    if common_count:
        keys = list(method_grids)
        stack = np.stack([method_grids[key][common_mask] for key in keys], axis=0)
        winners = np.nanargmin(stack, axis=0)
        for idx, key in enumerate(keys):
            winner_counts[key] = int((winners == idx).sum())

    best_row: dict[str, object] = {
        "case": case_name,
        "target": body_label,
        "common_all_methods_points": common_count,
    }
    for key, count in winner_counts.items():
        safe_key = key.replace("bspline_", "bs_")
        best_row[f"{safe_key}_best_points"] = count
        best_row[f"{safe_key}_best_percent"] = pct(count, common_count)
    return rows, best_row


def main() -> None:
    args = parse_args()
    variants = [parse_bspline_variant_spec(spec) for spec in args.bspline_variants]
    rows: list[dict[str, object]] = []
    best_rows: list[dict[str, object]] = []
    for case_name in args.cases:
        output_dir = output_dir_for_case(args, case_name)
        case_rows, best_row = compute_case_rows(case_name, output_dir, variants)
        rows.extend(case_rows)
        best_rows.append(best_row)

    write_csv(Path(args.out), rows)
    write_csv(Path(args.best_out), best_rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.best_out}")


if __name__ == "__main__":
    main()

"""Compare high-order Gondelach grids from two completed output folders."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def read_attempt_counts(path: Path) -> dict[str, int]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return {
        "branch_attempts": len(rows),
        "formal_success_attempts": sum(
            row.get("optimizer_success", "").strip().lower() == "true" for row in rows
        ),
        "usable_attempts": sum(
            row.get("usable", "").strip().lower() == "true" for row in rows
        ),
        "total_nfev": sum(int(row.get("nfev", 0) or 0) for row in rows),
    }


def scalar_text(archive: np.lib.npyio.NpzFile, key: str) -> str:
    return str(np.asarray(archive[key]).item()) if key in archive.files else ""


def saved_basis(directory: Path, archive: np.lib.npyio.NpzFile) -> str:
    basis = scalar_text(archive, "fig3_basis")
    if basis:
        return basis
    coefficient_path = directory / "gondelach_high_order_coefficients.npz"
    if coefficient_path.exists():
        with np.load(coefficient_path, allow_pickle=False) as coefficients:
            return scalar_text(coefficients, "basis")
    return ""


def read_high_order_pareto(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row.get("method") == "gondelach_fig3"
            and row.get("is_pareto", "").strip().lower() == "true"
        ]
    delta_v = np.asarray([float(row["delta_v_km_s"]) for row in rows], dtype=float)
    u_max = np.asarray([float(row["fmax_m_s2"]) for row in rows], dtype=float)
    finite = np.isfinite(delta_v) & np.isfinite(u_max)
    order = np.argsort(delta_v[finite])
    return delta_v[finite][order], u_max[finite][order]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-label", default="Current 6-DoF basis")
    parser.add_argument("--candidate-label", default="Former 4-DoF basis")
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    reference = np.load(reference_dir / "comparison_grids.npz", allow_pickle=False)
    candidate = np.load(candidate_dir / "comparison_grids.npz", allow_pickle=False)
    dep = np.asarray(reference["departure_mjd2000"], dtype=float)
    tof = np.asarray(reference["tof_days"], dtype=float)
    if not np.array_equal(dep, candidate["departure_mjd2000"]) or not np.array_equal(
        tof, candidate["tof_days"]
    ):
        raise ValueError("Reference and candidate departure/TOF grids differ")

    reference_dv = np.asarray(reference["fig3_delta_v_km_s"], dtype=float)
    candidate_dv = np.asarray(candidate["fig3_delta_v_km_s"], dtype=float)
    reference_n = np.asarray(reference["fig3_best_N"], dtype=int)
    candidate_n = np.asarray(candidate["fig3_best_N"], dtype=int)
    valid = np.isfinite(reference_dv) & np.isfinite(candidate_dv)
    difference = candidate_dv - reference_dv
    absolute = np.abs(difference[valid])

    reference_counts = read_attempt_counts(reference_dir / "fig3_attempts.csv")
    candidate_counts = read_attempt_counts(candidate_dir / "fig3_attempts.csv")
    reference_best = np.unravel_index(int(np.nanargmin(reference_dv)), reference_dv.shape)
    candidate_best = np.unravel_index(int(np.nanargmin(candidate_dv)), candidate_dv.shape)
    rows = []
    for label, directory, archive, values, best, counts in [
        (args.reference_label, reference_dir, reference, reference_dv, reference_best, reference_counts),
        (args.candidate_label, candidate_dir, candidate, candidate_dv, candidate_best, candidate_counts),
    ]:
        rows.append(
            {
                "label": label,
                "basis": saved_basis(directory, archive),
                "formulation": scalar_text(archive, "gondelach_formulation_version"),
                **counts,
                "finite_grid_points": int(np.isfinite(values).sum()),
                "best_delta_v_km_s": float(values[best]),
                "best_departure_mjd2000": float(dep[best[1]]),
                "best_tof_days": float(tof[best[0]]),
                "best_N": int((reference_n if archive is reference else candidate_n)[best]),
            }
        )
    summary_path = output_dir / "basis_comparison_summary.csv"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    statistics = {
        "matched_grid_points": int(valid.sum()),
        "best_N_mismatches": int(np.sum(valid & (reference_n != candidate_n))),
        "mean_signed_delta_v_difference_km_s": float(np.mean(difference[valid])),
        "median_absolute_delta_v_difference_km_s": float(np.median(absolute)),
        "p95_absolute_delta_v_difference_km_s": float(np.percentile(absolute, 95)),
        "p99_absolute_delta_v_difference_km_s": float(np.percentile(absolute, 99)),
        "max_absolute_delta_v_difference_km_s": float(np.max(absolute)),
        "count_absolute_difference_gt_0p01_km_s": int(np.sum(absolute > 0.01)),
        "count_absolute_difference_gt_0p1_km_s": int(np.sum(absolute > 0.1)),
        "count_absolute_difference_gt_1_km_s": int(np.sum(absolute > 1.0)),
    }
    stats_path = output_dir / "basis_comparison_statistics.csv"
    with stats_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(statistics))
        writer.writeheader()
        writer.writerow(statistics)

    np.savez(
        output_dir / "basis_comparison_grids.npz",
        departure_mjd2000=dep,
        tof_days=tof,
        reference_delta_v_km_s=reference_dv,
        candidate_delta_v_km_s=candidate_dv,
        delta_v_difference_km_s=difference,
        reference_best_N=reference_n,
        candidate_best_N=candidate_n,
        reference_label=np.asarray(args.reference_label),
        candidate_label=np.asarray(args.candidate_label),
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extent = [float(dep[0]), float(dep[-1]), float(tof[0]), float(tof[-1])]
    finite_values = np.concatenate([reference_dv[np.isfinite(reference_dv)], candidate_dv[np.isfinite(candidate_dv)]])
    vmax = float(np.percentile(finite_values, 95))
    diff_limit = max(float(np.percentile(absolute, 98)), 1.0e-9)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), sharex=True, sharey=True)
    for ax, values, label in zip(
        axes[:2], [reference_dv, candidate_dv], [args.reference_label, args.candidate_label]
    ):
        image = ax.imshow(values, origin="lower", aspect="auto", extent=extent, cmap="viridis_r", vmin=0.0, vmax=vmax)
        ax.set_title(label)
        fig.colorbar(image, ax=ax, label=r"$\Delta V$ [km/s]")
    image = axes[2].imshow(
        difference,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        vmin=-diff_limit,
        vmax=diff_limit,
    )
    axes[2].set_title(f"{args.candidate_label} - {args.reference_label}")
    fig.colorbar(image, ax=axes[2], label=r"$\Delta V$ difference [km/s]")
    for ax in axes:
        ax.set_xlabel("Departure date [MJD2000]")
    axes[0].set_ylabel("Time of flight [days]")
    fig.tight_layout()
    plot_path = output_dir / "basis_delta_v_comparison.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    reference_pareto = read_high_order_pareto(reference_dir / "comparison_pareto_points.csv")
    candidate_pareto = read_high_order_pareto(candidate_dir / "comparison_pareto_points.csv")
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    for (delta_v, u_max), label, color, marker in [
        (reference_pareto, args.reference_label, "tab:blue", "o"),
        (candidate_pareto, args.candidate_label, "tab:orange", "s"),
    ]:
        ax.plot(
            delta_v,
            u_max,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4.5,
            label=label,
        )
    ax.set_xlabel(r"$\Delta V$ [km/s]")
    ax.set_ylabel(r"$u_{\max}$ [m/s$^2$]")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    pareto_png = output_dir / "basis_pareto_umax_delta_v.png"
    pareto_pdf = output_dir / "basis_pareto_umax_delta_v.pdf"
    fig.savefig(pareto_png, dpi=180)
    fig.savefig(pareto_pdf)
    plt.close(fig)

    for path in [
        summary_path,
        stats_path,
        output_dir / "basis_comparison_grids.npz",
        plot_path,
        pareto_png,
        pareto_pdf,
    ]:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

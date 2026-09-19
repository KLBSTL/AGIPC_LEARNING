"""Plot the single-system AGIPC criterion threshold diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.analysis.read_text(encoding="utf-8"))
    rows = sorted(result["threshold_ablation_same_increment_field"].values(),
                  key=lambda row: row["threshold"])
    x = np.array([row["threshold"] for row in rows])
    active = 100 * np.array([row["active_fem_dof_ratio"] for row in rows])
    protected = 100 * np.array([row["protected_edge_ratio"] for row in rows])
    fit_error = 100 * np.array([row["best_euclidean_subspace_fit_relative_error"] for row in rows])
    nodes = result["captured"]["mapping"]["fine_nodes"]
    largest = 100 * np.array([row["largest_group_sizes"][0] / nodes for row in rows])
    threshold = result["captured"]["threshold"]
    runtime = result["captured"]["runtime_mapping_subspace_fit"]

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.2), sharex=True, constrained_layout=True)
    axes[0].semilogx(x, active, "o-", label="CC-oracle active FEM DoF")
    axes[0].semilogx(x, protected, "s-", label="protected edges")
    axes[0].scatter([threshold], [100 * runtime["active_fem_dof_ratio"]], marker="*", s=120,
                    color="black", zorder=4, label="runtime stable mapping DoF")
    axes[0].set_ylabel("fraction (%)")
    axes[0].set_ylim(-3, 103)
    axes[0].grid(True, which="both", alpha=.25)
    axes[0].legend(fontsize=8, loc="best")

    axes[1].semilogx(x, fit_error, "o-", label="CC-oracle best direction-fit error")
    axes[1].semilogx(x, largest, "s-", label="largest CC coverage")
    axes[1].scatter([threshold], [100 * runtime["best_euclidean_subspace_fit_relative_error"]],
                    marker="*", s=120, color="black", zorder=4,
                    label="runtime mapping best-fit error")
    axes[1].axvline(threshold, color="0.45", linestyle="--", linewidth=1,
                    label=r"captured threshold $5\times10^{-5}$")
    axes[1].set_xlabel("Green-strain increment threshold")
    axes[1].set_ylabel("fraction (%)")
    axes[1].set_ylim(-3, 103)
    axes[1].grid(True, which="both", alpha=.25)
    axes[1].legend(fontsize=8, loc="best")
    fig.suptitle("AGIPC criterion replay on one accepted contact system")

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=200)
    fig.savefig(args.output_stem.with_suffix(".pdf"))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

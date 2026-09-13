#!/usr/bin/env python3
"""Plot frozen group membership and fine-direction error in rest coordinates."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from analyze_frozen_fallback import read_vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("--output-stem", type=Path, required=True)
    parser.add_argument("--frame-index", type=int)
    args = parser.parse_args()
    metadata = json.loads((args.sample / "metadata.json").read_text(encoding="utf-8"))
    nodes = int(metadata["mapping_fine_nodes"])
    dofs = int(metadata["fine_dofs"])
    prefix = 3 * (int(metadata["fine_block_nodes"]) - nodes)
    rest = read_vector(args.sample / "fine_rest_positions.f64x3.bin", np.float64, 3 * nodes).reshape(nodes, 3)
    owners = read_vector(args.sample / "fine_to_coarse.i32.bin", np.int32, nodes)
    masks = read_vector(args.sample / "coarse_basis_masks.i32.bin", np.int32, int(metadata["mapping_coarse_nodes"]))
    reference = read_vector(args.sample / "cpu_direct_solution.f64.bin", np.float64, dofs)[prefix:].reshape(nodes, 3)
    candidate = read_vector(args.sample / "agipc_candidate.f64.bin", np.float64, dofs)[prefix:].reshape(nodes, 3)
    error = np.linalg.norm(candidate - reference, axis=1)
    if not np.isfinite(rest).all() or not np.isfinite(error).all() or not (error > 0).any():
        raise ValueError("invalid coordinates/direction error")
    sizes = np.bincount(owners, minlength=masks.size)
    affine = [i for i, mask in enumerate(masks) if int(mask).bit_count() > 1]
    affine.sort(key=lambda i: sizes[i], reverse=True)
    axes = sorted(np.argsort(np.ptp(rest, axis=0))[-2:])
    x, y = rest[:, axes[0]], rest[:, axes[1]]
    figure, panels = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    small = ~np.isin(owners, affine)
    panels[0].scatter(x[small], y[small], s=3, c="#555555", label="Other groups", rasterized=True)
    colors = plt.get_cmap("tab10", max(10, len(affine)))
    for index, group in enumerate(affine):
        selected = owners == group
        panels[0].scatter(x[selected], y[selected], s=2, c=[colors(index)],
                          label=f"Group {group}: {sizes[group]:,}", rasterized=True)
    fraction = 100 * sizes[affine].sum() / nodes
    panels[0].set_title(f"{len(affine)} affine groups: {fraction:.2f}% of cloth nodes")
    panels[0].legend(fontsize=7, loc="upper right", markerscale=2, framealpha=0.9)
    lower = max(float(error[error > 0].min()), float(error.max()) * 1e-4)
    dots = panels[1].scatter(x, y, s=2, c=error, cmap="magma",
                             norm=LogNorm(vmin=lower, vmax=float(error.max())), rasterized=True)
    panels[1].set_title("GPU10 minus fine direct direction")
    figure.colorbar(dots, ax=panels[1], label="FEM direction error norm (m)")
    for panel in panels:
        panel.set_aspect("equal")
        panel.set_xlabel(f"Rest {'XYZ'[axes[0]]} (m)")
        panel.set_ylabel(f"Rest {'XYZ'[axes[1]]} (m)")
    frame = f" / frame {args.frame_index}" if args.frame_index is not None else ""
    figure.suptitle(f"Frozen update {metadata['update_index']}{frame}; coordinates show rest geometry")
    for extension in ("png", "pdf"):
        output = args.output_stem.with_suffix("." + extension)
        figure.savefig(output, dpi=240)
        print(output.resolve())
    plt.close(figure)


if __name__ == "__main__":
    main()

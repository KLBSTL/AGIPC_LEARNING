#!/usr/bin/env python3
"""Create a deterministic SPD block-chain fixture with an exact reference."""

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--blocks", type=int, default=257)
    args = parser.parse_args()
    if args.blocks < 2:
        parser.error("at least two blocks are required")
    count = args.blocks
    positions = np.arange(count)
    reference = np.column_stack((np.sin(0.37 * positions) + 0.1 * positions / count,
                                 np.cos(0.23 * positions), np.sin(0.17 * positions + 0.3)))
    diagonal = np.tile(np.diag([1.0, 2.0, 3.0]), (count, 1, 1))
    rotations = []
    for i in range(count - 1):
        angle = 0.13 * (-1 if i % 2 else 1)
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        rotations.append(-100.0 * rotation)
        diagonal[i] += 100.0 * np.eye(3)
        diagonal[i + 1] += 100.0 * np.eye(3)
    # A = block mass + sum 100 * ||x_i - R_i x_(i+1)||^2, hence SPD.
    rhs = np.einsum("nij,nj->ni", diagonal, reference)
    for i, block in enumerate(rotations):
        rhs[i] += block @ reference[i + 1]
        rhs[i + 1] += block.T @ reference[i]
    values = np.concatenate((diagonal, np.asarray(rotations)))
    rows = np.concatenate((np.arange(count), np.arange(count - 1))).astype(np.int32)
    cols = np.concatenate((np.arange(count), np.arange(1, count))).astype(np.int32)
    args.directory.mkdir(parents=True, exist_ok=True)
    values.transpose(0, 2, 1).tofile(args.directory / "coarse_A_values.f64x9.bin")
    rows.tofile(args.directory / "coarse_A_rows.i32.bin")
    cols.tofile(args.directory / "coarse_A_cols.i32.bin")
    rhs.tofile(args.directory / "coarse_rhs.f64.bin")
    reference.tofile(args.directory / "coarse_solution.f64.bin")
    metadata = {"format": "agipc_coarse_system_v1", "sample_name": args.directory.name,
                "category": "synthetic_spd_chain", "reference_kind": "analytic exact solution",
                "coarse_block_nodes": count, "coarse_unique_blocks": 2 * count - 1,
                "coarse_matrix_half_storage": True,
                "coarse_solve": {"max_iterations": min(512, 3 * count), "relative_tolerance": 1e-3}}
    (args.directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.directory / "manifest.json").write_text(
        json.dumps({"format": "agipc_coarse_samples_v1",
                    "samples": [{**metadata, "sample_name": "."}]}, indent=2) + "\n",
        encoding="utf-8")
    print(f"created {count}-block SPD fixture with {2 * count - 1} half-storage blocks")


if __name__ == "__main__":
    main()

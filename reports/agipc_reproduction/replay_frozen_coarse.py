#!/usr/bin/env python3
"""Independently replay FP64 block-Jacobi PCG on frozen coarse systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse


def read_array(path: Path, dtype, count: int) -> np.ndarray:
    result = np.fromfile(path, dtype=dtype)
    if result.size != count:
        raise ValueError(f"{path}: expected {count} values, found {result.size}")
    if not np.isfinite(result).all():
        raise ValueError(f"{path}: nonfinite data")
    return result


def load_system(directory: Path, metadata: dict):
    blocks = int(metadata["coarse_block_nodes"])
    unique = int(metadata["coarse_unique_blocks"])
    if blocks <= 0 or unique <= 0:
        raise ValueError("empty coarse matrix")
    if not metadata.get("coarse_matrix_half_storage", True):
        raise ValueError("only symmetric half-storage coarse snapshots are supported")
    values = read_array(directory / "coarse_A_values.f64x9.bin", np.float64, 9 * unique)
    values = values.reshape(unique, 3, 3).transpose(0, 2, 1)
    rows = read_array(directory / "coarse_A_rows.i32.bin", np.int32, unique)
    cols = read_array(directory / "coarse_A_cols.i32.bin", np.int32, unique)
    if rows.min() < 0 or cols.min() < 0 or rows.max() >= blocks or cols.max() >= blocks:
        raise ValueError("coarse matrix index out of bounds")
    offsets = np.arange(3)
    scalar_rows = np.broadcast_to(3 * rows[:, None, None] + offsets[None, :, None], values.shape)
    scalar_cols = np.broadcast_to(3 * cols[:, None, None] + offsets[None, None, :], values.shape)
    off = rows != cols
    matrix = sparse.coo_matrix(
        (
            np.concatenate((values.ravel(), values[off].transpose(0, 2, 1).ravel())),
            (
                np.concatenate((scalar_rows.ravel(), scalar_cols[off].transpose(0, 2, 1).ravel())),
                np.concatenate((scalar_cols.ravel(), scalar_rows[off].transpose(0, 2, 1).ravel())),
            ),
        ),
        shape=(3 * blocks, 3 * blocks),
    ).tocsr()
    diagonal = np.zeros((blocks, 3, 3))
    on = rows == cols
    found = np.zeros(blocks, dtype=np.int32)
    np.add.at(diagonal, rows[on], values[on])
    np.add.at(found, rows[on], 1)
    if not (found > 0).all():
        raise ValueError("missing coarse diagonal blocks")
    skew = np.max(np.abs(diagonal - diagonal.transpose(0, 2, 1)))
    scale = max(float(np.max(np.abs(diagonal))), np.finfo(np.float64).tiny)
    if skew / scale > 1e-12:
        raise ValueError("nonsymmetric coarse diagonal block")
    eigenvalues = np.linalg.eigvalsh(diagonal)
    if not (eigenvalues > 0).all():
        raise ValueError("coarse block-Jacobi diagonal is not positive definite")
    rhs = read_array(directory / "coarse_rhs.f64.bin", np.float64, 3 * blocks)
    saved = read_array(directory / "coarse_solution.f64.bin", np.float64, 3 * blocks)
    return matrix, np.linalg.inv(diagonal), rhs, saved, float(eigenvalues.min())


def pcg(matrix, inverse: np.ndarray, rhs: np.ndarray, limit: int, tolerance: float):
    solution = np.zeros_like(rhs)
    residual = rhs.copy()
    initial2 = float(residual @ residual)
    target2 = tolerance * tolerance * initial2
    apply = lambda vector: np.einsum("nij,nj->ni", inverse, vector.reshape(-1, 3)).ravel()
    z = apply(residual)
    direction = z.copy()
    rz = float(residual @ z)
    residual2 = initial2
    iterations = 0
    failure = ""
    while iterations < limit and residual2 > target2:
        product = matrix @ direction
        curvature = float(direction @ product)
        if not np.isfinite(curvature) or curvature <= 0 or not np.isfinite(rz):
            failure = "nonpositive_or_nonfinite_curvature"
            break
        alpha = rz / curvature
        solution += alpha * direction
        residual -= alpha * product
        iterations += 1
        residual2 = float(residual @ residual)
        if not np.isfinite(residual2):
            failure = "nonfinite_residual"
            break
        if residual2 <= target2:
            break
        z = apply(residual)
        next_rz = float(residual @ z)
        if not np.isfinite(next_rz) or abs(rz) < 1e-30:
            failure = "invalid_preconditioned_residual"
            break
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    converged = residual2 <= target2
    if not converged and not failure:
        failure = "iteration_cap"
    return solution, {"converged": converged, "iterations": iterations,
                      "failure_reason": failure,
                      "recursive_relative_residual": float(np.sqrt(residual2 / initial2))
                      if initial2 else 0.0}


def metrics(matrix, rhs: np.ndarray, solution: np.ndarray) -> dict:
    product = matrix @ solution
    residual_norm = float(np.linalg.norm(rhs - product))
    return {"residual_norm": residual_norm,
            "relative_residual": residual_norm / max(float(np.linalg.norm(rhs)), np.finfo(np.float64).tiny),
            "predicted_quadratic_decrease": float(rhs @ solution - 0.5 * solution @ product)}


def replay(directory: Path) -> dict:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    matrix, inverse, rhs, saved, minimum = load_system(directory, metadata)
    solve = metadata.get("coarse_solve", {})
    limit = int(solve.get("max_iterations", min(512, rhs.size)))
    tolerance = float(solve.get("relative_tolerance", 1e-3))
    solution, result = pcg(matrix, inverse, rhs, limit, tolerance)
    return {"sample": directory.name,
            "coarse_block_nodes": rhs.size // 3,
            "coarse_unique_blocks": metadata["coarse_unique_blocks"],
            "diagonal_minimum_eigenvalue": minimum,
            "matrix_integrity_passed": True,
            "saved_gpu": metrics(matrix, rhs, saved),
            "cpu_block_jacobi": {**result, **metrics(matrix, rhs, solution)},
            "saved_vs_replay_relative_direction_difference":
                float(np.linalg.norm(saved - solution))
                / max(float(np.linalg.norm(saved)), np.finfo(np.float64).tiny),
            "relative_tolerance": tolerance, "max_iterations": limit}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    manifest = json.loads((arguments.sample_root / "manifest.json").read_text(encoding="utf-8"))
    # Compatibility with the earlier fallback snapshots, which also froze Hc/bc/dc.
    if manifest["format"] not in ("agipc_coarse_samples_v1", "agipc_fallback_direction_v1"):
        raise ValueError("unsupported snapshot manifest")
    report = {"format": "agipc_coarse_replay_v1",
              "samples": [replay(arguments.sample_root / item["sample_name"])
                          for item in manifest["samples"]]}
    output = json.dumps(report, indent=2, allow_nan=False)
    if arguments.output:
        arguments.output.write_text(output + "\n", encoding="utf-8")
    if not arguments.quiet:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

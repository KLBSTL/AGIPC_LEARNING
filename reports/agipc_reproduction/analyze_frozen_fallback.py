#!/usr/bin/env python3
"""Independently verify and compare frozen AGIPC fallback directions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_vector(path: Path, dtype: np.dtype, count: int | None = None) -> np.ndarray:
    values = np.fromfile(path, dtype=dtype)
    if count is not None and values.size != count:
        raise ValueError(f"{path}: expected {count} values, found {values.size}")
    return values


def read_block_matrix(sample: Path, prefix: str, unique: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = read_vector(sample / f"{prefix}_values.f64x9.bin", np.float64, 9 * unique)
    # Eigen::Matrix3d is column-major in the binary artifact.
    values = raw.reshape(unique, 3, 3).transpose(0, 2, 1)
    rows = read_vector(sample / f"{prefix}_rows.i32.bin", np.int32, unique)
    cols = read_vector(sample / f"{prefix}_cols.i32.bin", np.int32, unique)
    return values, rows, cols


def symmetric_half_spmv(
    values: np.ndarray, rows: np.ndarray, cols: np.ndarray, vector: np.ndarray, blocks: int
) -> np.ndarray:
    if rows.size and (rows.min() < 0 or cols.min() < 0 or rows.max() >= blocks or cols.max() >= blocks):
        raise ValueError("matrix block index outside the frozen vector domain")
    x = vector.reshape(blocks, 3)
    y = np.zeros_like(x)
    np.add.at(y, rows, np.einsum("nij,nj->ni", values, x[cols]))
    off_diagonal = rows != cols
    np.add.at(
        y,
        cols[off_diagonal],
        np.einsum(
            "nji,nj->ni",
            values[off_diagonal],
            x[rows[off_diagonal]],
        ),
    )
    return y.reshape(-1)


def reconstruct_prolongated(sample: Path, metadata: dict) -> np.ndarray:
    fine_dofs = int(metadata["fine_dofs"])
    fine_blocks = int(metadata["fine_block_nodes"])
    fine_nodes = int(metadata["mapping_fine_nodes"])
    coarse_nodes = int(metadata["mapping_coarse_nodes"])
    prefix_blocks = fine_blocks - fine_nodes
    mapping = read_vector(sample / "fine_to_coarse.i32.bin", np.int32, fine_nodes)
    bases = read_vector(sample / "coarse_block_bases.i32.bin", np.int32, coarse_nodes)
    masks = read_vector(sample / "coarse_basis_masks.i32.bin", np.int32, coarse_nodes)
    rest = read_vector(sample / "fine_rest_positions.f64x3.bin", np.float64, 3 * fine_nodes).reshape(-1, 3)
    coarse = read_vector(sample / "coarse_solution.f64.bin", np.float64)
    prolonged = np.zeros((fine_blocks, 3), dtype=np.float64)
    prolonged[:prefix_blocks] = coarse[: 3 * prefix_blocks].reshape(prefix_blocks, 3)
    for local in range(fine_nodes):
        owner = int(mapping[local])
        mask = int(masks[owner])
        columns = [column for column in range(4) if mask & (1 << column)]
        phi = np.array([1.0, *rest[local]], dtype=np.float64)[columns]
        base = prefix_blocks + int(bases[owner])
        block_values = coarse[3 * base : 3 * (base + len(columns))].reshape(-1, 3)
        prolonged[prefix_blocks + local] = phi @ block_values
    result = prolonged.reshape(-1)
    if result.size != fine_dofs or not np.isfinite(result).all():
        raise ValueError("invalid reconstructed prolongated direction")
    return result


def direction_metrics(
    values: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    rhs: np.ndarray,
    direction: np.ndarray,
) -> dict[str, float]:
    ax = symmetric_half_spmv(values, rows, cols, direction, rhs.size // 3)
    residual = rhs - ax
    rhs_dot = float(rhs @ direction)
    quadratic = float(direction @ ax)
    rhs_norm = float(np.linalg.norm(rhs))
    residual_norm = float(np.linalg.norm(residual))
    return {
        "residual_norm": residual_norm,
        "relative_residual": residual_norm / max(rhs_norm, np.finfo(np.float64).tiny),
        "direction_norm": float(np.linalg.norm(direction)),
        "rhs_dot_direction": rhs_dot,
        "direction_dot_A_direction": quadratic,
        "predicted_quadratic_decrease": rhs_dot - 0.5 * quadratic,
    }


def compare(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    denominator = reference_norm * candidate_norm
    cosine = float(np.clip(reference @ candidate / denominator, -1.0, 1.0)) if denominator else 0.0
    return {
        "cosine": cosine,
        "angle_degrees": float(np.degrees(np.arccos(cosine))),
        "relative_direction_difference": float(np.linalg.norm(candidate - reference))
        / max(reference_norm, np.finfo(np.float64).tiny),
    }


def analyze_sample(sample: Path) -> dict:
    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    fine_dofs = int(metadata["fine_dofs"])
    unique = int(metadata["fine_unique_blocks"])
    values, rows, cols = read_block_matrix(sample, "fine_A", unique)
    rhs = read_vector(sample / "fine_rhs.f64.bin", np.float64, fine_dofs)
    agipc = read_vector(sample / "agipc_candidate.f64.bin", np.float64, fine_dofs)
    fallback = read_vector(sample / "fine_fallback.f64.bin", np.float64, fine_dofs)
    prolonged = reconstruct_prolongated(sample, metadata)
    agipc_metrics = direction_metrics(values, rows, cols, rhs, agipc)
    fallback_metrics = direction_metrics(values, rows, cols, rhs, fallback)
    prolonged_metrics = direction_metrics(values, rows, cols, rhs, prolonged)
    guard_uses_prolongated = metadata["post_residual_reduction_ratio"] > 1.0 + 1e-6
    guarded = prolonged if guard_uses_prolongated else agipc
    guarded_metrics = prolonged_metrics if guard_uses_prolongated else agipc_metrics
    return {
        "sample": sample.name,
        "quality_category": metadata["quality_category"],
        "post_residual_reduction_ratio": metadata["post_residual_reduction_ratio"],
        "coarse_fem_nodes": metadata["coarse_fem_nodes"],
        "coarse_iterations": metadata["coarse_iterations"],
        "prolongated": prolonged_metrics,
        "post_corrected_candidate": agipc_metrics,
        "fine_fallback": fallback_metrics,
        "residual_guard_selection": "prolongated" if guard_uses_prolongated else "post_corrected",
        "residual_guard": guarded_metrics,
        "prolongated_vs_fine": compare(fallback, prolonged),
        "candidate_vs_fine": compare(fallback, agipc),
        "residual_guard_vs_fine": compare(fallback, guarded),
        "candidate_to_fine_residual_ratio": agipc_metrics["residual_norm"]
        / max(fallback_metrics["residual_norm"], np.finfo(np.float64).tiny),
        "prolongated_to_fine_residual_ratio": prolonged_metrics["residual_norm"]
        / max(fallback_metrics["residual_norm"], np.finfo(np.float64).tiny),
        "post_to_prolongated_residual_ratio": agipc_metrics["residual_norm"]
        / max(prolonged_metrics["residual_norm"], np.finfo(np.float64).tiny),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_root", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    manifest = json.loads((arguments.sample_root / "manifest.json").read_text(encoding="utf-8"))
    rows = [analyze_sample(arguments.sample_root / item["sample_name"]) for item in manifest["samples"]]
    report = {
        "format": "agipc_fallback_direction_analysis_v1",
        "sample_count": len(rows),
        "samples": rows,
    }
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if arguments.output:
        arguments.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

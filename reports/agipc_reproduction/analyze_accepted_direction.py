#!/usr/bin/env python3
"""Compare an accepted GPU direction with an independent same-H/g SparseLU solve."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from analyze_frozen_fallback import (
    compare, direction_metrics, read_block_matrix, read_vector,
    reconstruct_prolongated, symmetric_half_spmv,
)
from replay_frozen_coarse import load_system


def bounded_post(matrix, values, rows, cols, rhs, initial):
    """One offline FP64 block-Jacobi PCG trajectory, checkpointed at 10/20 steps."""
    blocks = rhs.size // 3
    diagonal = np.zeros((blocks, 3, 3))
    on = rows == cols
    np.add.at(diagonal, rows[on], values[on])
    if not (np.linalg.eigvalsh(diagonal) > 0).all():
        raise ValueError("fine block-Jacobi diagonal is not SPD")
    inverse = np.linalg.inv(diagonal)
    apply = lambda x: np.einsum("nij,nj->ni", inverse, x.reshape(-1, 3)).ravel()
    x = initial.copy()
    r = rhs - matrix @ x
    z = apply(r)
    p = z.copy()
    rz = float(r @ z)
    checkpoints = {}
    target2 = 1e-6 * float(rhs @ rhs)
    for step in range(1, 21):
        if float(r @ r) <= target2:
            for budget in (10, 20):
                if budget not in checkpoints:
                    checkpoints[budget] = x.copy()
            return checkpoints, step - 1
        ap = matrix @ p
        curvature = float(p @ ap)
        if not np.isfinite(curvature) or curvature <= 0 or not np.isfinite(rz) or rz <= 0:
            raise ValueError("offline post-PCG lost finite positive curvature/rMz")
        alpha = rz / curvature
        x += alpha * p
        r -= alpha * ap
        z = apply(r)
        next_rz = float(r @ z)
        p = z + (next_rz / rz) * p
        rz = next_rz
        if step in (10, 20):
            checkpoints[step] = x.copy()
    return checkpoints, 20


def prolongation_operator(sample, metadata):
    fine_nodes = int(metadata["mapping_fine_nodes"])
    coarse_nodes = int(metadata["mapping_coarse_nodes"])
    prefix = int(metadata["fine_block_nodes"]) - fine_nodes
    owners = read_vector(sample / "fine_to_coarse.i32.bin", np.int32, fine_nodes)
    bases = read_vector(sample / "coarse_block_bases.i32.bin", np.int32, coarse_nodes)
    masks = read_vector(sample / "coarse_basis_masks.i32.bin", np.int32, coarse_nodes)
    rest = read_vector(sample / "fine_rest_positions.f64x3.bin", np.float64, 3 * fine_nodes).reshape(-1, 3)
    if owners.min() < 0 or owners.max() >= coarse_nodes or not np.isfinite(rest).all():
        raise ValueError("invalid mapping domain/rest positions")
    row_parts = [np.arange(3 * prefix)]
    col_parts = [np.arange(3 * prefix)]
    value_parts = [np.ones(3 * prefix)]
    phi = np.column_stack((np.ones(fine_nodes), rest))
    popcounts = np.array([i.bit_count() for i in range(16)])
    xyz = np.arange(3)
    for column in range(4):
        selected = np.flatnonzero(masks[owners] & (1 << column))
        base = prefix + bases[owners[selected]] + popcounts[masks[owners[selected]] & ((1 << column) - 1)]
        row_parts.append((3 * (prefix + selected[:, None]) + xyz).ravel())
        col_parts.append((3 * base[:, None] + xyz).ravel())
        value_parts.append(np.repeat(phi[selected, column], 3))
    return sparse.coo_matrix((np.concatenate(value_parts),
        (np.concatenate(row_parts), np.concatenate(col_parts))),
        shape=(int(metadata["fine_dofs"]), 3 * int(metadata["coarse_block_nodes"]))).tocsr()


def analyze(sample: Path) -> dict:
    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    if metadata["format"] != "agipc_accepted_direction_v1" or not metadata["adoption"]["adopted"]:
        raise ValueError("expected an accepted direction snapshot")
    dofs = int(metadata["fine_dofs"])
    blocks = int(metadata["fine_block_nodes"])
    if dofs != 3 * blocks or not metadata["fine_matrix_half_storage"]:
        raise ValueError("unsupported fine matrix dimensions/storage")
    values, rows, cols = read_block_matrix(sample, "fine_A", int(metadata["fine_unique_blocks"]))
    rhs = read_vector(sample / "fine_rhs.f64.bin", np.float64, dofs)
    candidate = read_vector(sample / "agipc_candidate.f64.bin", np.float64, dofs)
    prolonged = read_vector(sample / "prolongated.f64.bin", np.float64, dofs)
    if not all(np.isfinite(v).all() for v in (values, rhs, candidate, prolonged)):
        raise ValueError("nonfinite frozen data")
    # This also validates the block-index domain independently of sparse assembly.
    symmetric_half_spmv(values, rows, cols, rhs, blocks)
    offsets = np.arange(3)
    ri = np.broadcast_to(3 * rows[:, None, None] + offsets[None, :, None], values.shape)
    ci = np.broadcast_to(3 * cols[:, None, None] + offsets[None, None, :], values.shape)
    off = rows != cols
    matrix = sparse.coo_matrix((
        np.concatenate((values.ravel(), values[off].transpose(0, 2, 1).ravel())),
        (np.concatenate((ri.ravel(), ci[off].transpose(0, 2, 1).ravel())),
         np.concatenate((ci.ravel(), ri[off].transpose(0, 2, 1).ravel()))),
    ), shape=(dofs, dofs)).tocsc()
    matrix.eliminate_zeros()
    skew = matrix - matrix.T
    symmetry_error = float(np.max(np.abs(skew.data))) if skew.nnz else 0.0
    symmetry_relative = symmetry_error / max(float(np.max(np.abs(matrix.data))), np.finfo(float).tiny)
    if symmetry_relative > 1e-12:
        raise ValueError("frozen Hessian is not symmetric")
    start = time.perf_counter()
    # Preserve the symmetric elimination order for the physical SPD Hessian.
    # No shift/regularization is applied; independently checked true residuals
    # still reject an inaccurate reference, including a failed factorization.
    factor = splu(matrix, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                  options={"SymmetricMode": True})
    direct = factor.solve(rhs)
    direct_seconds = time.perf_counter() - start
    if not np.isfinite(direct).all():
        raise ValueError("nonfinite direct reference")
    reconstructed = reconstruct_prolongated(sample, metadata)
    reconstruction_relative = float(np.linalg.norm(prolonged - reconstructed)) / max(
        float(np.linalg.norm(prolonged)), np.finfo(float).tiny)
    coarse_matrix, _, coarse_rhs, coarse_saved, _ = load_system(sample, metadata)
    prolongation = prolongation_operator(sample, metadata)
    cpu_coarse = (prolongation.T @ matrix @ prolongation).tocsr()
    difference = cpu_coarse - coarse_matrix
    galerkin_matrix_error = float(np.linalg.norm(difference.data)) / max(
        float(np.linalg.norm(cpu_coarse.data)), np.finfo(float).tiny)
    restricted_rhs = prolongation.T @ rhs
    galerkin_rhs_error = float(np.linalg.norm(restricted_rhs - coarse_rhs)) / max(
        float(np.linalg.norm(restricted_rhs)), np.finfo(float).tiny)
    coarse_prefix_dofs = 3 * (blocks - int(metadata["mapping_fine_nodes"]))
    cpu_fem = cpu_coarse[coarse_prefix_dofs:, coarse_prefix_dofs:]
    fem_difference = difference[coarse_prefix_dofs:, coarse_prefix_dofs:]
    galerkin_fem_error = float(np.linalg.norm(fem_difference.data)) / max(
        float(np.linalg.norm(cpu_fem.data)), np.finfo(float).tiny)
    galerkin_fem_rhs_error = float(np.linalg.norm((restricted_rhs - coarse_rhs)[coarse_prefix_dofs:])) / max(
        float(np.linalg.norm(restricted_rhs[coarse_prefix_dofs:])), np.finfo(float).tiny)
    coarse_direct = splu(coarse_matrix.tocsc(), permc_spec="MMD_AT_PLUS_A").solve(coarse_rhs)
    coarse_direct_residual = float(np.linalg.norm(coarse_rhs - coarse_matrix @ coarse_direct)) / max(
        float(np.linalg.norm(coarse_rhs)), np.finfo(float).tiny)
    exact_coarse_prolonged = reconstruct_prolongated(sample, metadata, coarse_direct)
    cpu_post, cpu_steps = bounded_post(matrix, values, rows, cols, rhs, prolonged)
    vectors = {"direct": direct, "prolongated": prolonged, "candidate": candidate,
               "exact_coarse_prolongated": exact_coarse_prolonged,
               "cpu_post10": cpu_post[10], "cpu_post20": cpu_post[20]}
    metrics = {name: direction_metrics(values, rows, cols, rhs, vector)
               for name, vector in vectors.items()}
    gpu_post = metadata["post_correction"]
    residual_agreement = {}
    for name, key in (("prolongated", "initial_residual_norm"), ("candidate", "final_residual_norm")):
        cpu_value = metrics[name]["residual_norm"]
        gpu_value = float(gpu_post[key])
        residual_agreement[name] = abs(cpu_value - gpu_value) / max(abs(cpu_value), abs(gpu_value), np.finfo(float).tiny)
    direct_energy = metrics["direct"]["direction_dot_A_direction"]
    comparisons = {}
    for name, vector in vectors.items():
        if name == "direct":
            continue
        error = vector - direct
        error_energy = float(error @ (matrix @ error))
        comparisons[name] = {
            **compare(direct, vector),
            "error_A_energy": error_energy,
            "relative_A_energy_error": float(np.sqrt(max(0.0, error_energy) / direct_energy))
                if direct_energy > 0 else None,
            "predicted_decrease_fraction_of_direct": metrics[name]["predicted_quadratic_decrease"]
                / metrics["direct"]["predicted_quadratic_decrease"],
        }
        prefix = blocks - int(metadata["mapping_fine_nodes"])
        fem_error = error[3 * prefix:].reshape(-1, 3)
        fem_reference = direct[3 * prefix:]
        comparisons[name]["fem_relative_direction_error"] = float(np.linalg.norm(fem_error)) / max(
            float(np.linalg.norm(fem_reference)), np.finfo(float).tiny)
        comparisons[name]["fem_direction_error_rms"] = float(np.sqrt(np.mean(np.sum(fem_error ** 2, axis=1))))
        comparisons[name]["fem_direction_error_max"] = float(np.max(np.linalg.norm(fem_error, axis=1)))
    direct.tofile(sample / "cpu_direct_solution.f64.bin")
    gate = (metrics["direct"]["relative_residual"] <= 1e-10
            and reconstruction_relative <= 1e-12
            and max(residual_agreement.values()) <= 1e-9
            and coarse_direct_residual <= 1e-10
            and galerkin_matrix_error <= 1e-12 and galerkin_rhs_error <= 1e-12
            and galerkin_fem_error <= 1e-12 and galerkin_fem_rhs_error <= 1e-12
            and direct_energy > 0
            and all(v["error_A_energy"] >= 0 for v in comparisons.values()))
    return {
        "format": "agipc_accepted_direction_analysis_v1", "sample": str(sample.resolve()),
        "update_index": metadata["update_index"], "fine_dofs": dofs,
        "fine_unique_blocks": metadata["fine_unique_blocks"], "scalar_nonzeros": matrix.nnz,
        "reference_method": "SciPy SuperLU FP64 MMD_AT_PLUS_A SymmetricMode, diag_pivot_thresh=0; same frozen H/g, no regularization",
        "direct_factor_nonzeros": factor.L.nnz + factor.U.nnz,
        "direct_seconds": direct_seconds, "symmetry_relative_error": symmetry_relative,
        "prolongation_reconstruction_relative_error": reconstruction_relative,
        "cpu_gpu_residual_relative_difference": residual_agreement,
        "coarse_direct_relative_residual": coarse_direct_residual,
        "independent_galerkin_matrix_relative_error": galerkin_matrix_error,
        "independent_galerkin_rhs_relative_error": galerkin_rhs_error,
        "independent_galerkin_fem_matrix_relative_error": galerkin_fem_error,
        "independent_galerkin_fem_rhs_relative_error": galerkin_fem_rhs_error,
        "coarse_gpu_vs_direct": compare(coarse_direct, coarse_saved),
        "cpu_post_iterations": cpu_steps,
        "gpu_candidate_vs_cpu_post10": compare(cpu_post[10], candidate),
        "independent_reference_gate_passed": bool(gate),
        "directions": metrics, "vs_direct": comparisons,
        "post_to_prolongated_residual_ratio": metrics["candidate"]["residual_norm"]
            / max(metrics["prolongated"]["residual_norm"], np.finfo(float).tiny),
        "gpu_coarse_solve": metadata["coarse_solve"],
        "gpu_post_correction": metadata["post_correction"],
        "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(sample.glob("*.bin"))},
        "scope": "one frozen linear system; offline CPU10/20 are unguarded budget ablations; no nonlinear trajectory equivalence or speed claim",
    }


def geometry_summary(sample: Path, bbox_squared: float, dt: float) -> dict:
    """Cheap FEM stopping/mapping scan; reuses the verified direct vector."""
    if bbox_squared <= 0 or dt <= 0:
        raise ValueError("positive bbox squared and dt are required")
    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    nodes = int(metadata["mapping_fine_nodes"])
    dofs = int(metadata["fine_dofs"])
    prefix = 3 * (int(metadata["fine_block_nodes"]) - nodes)
    threshold = float(1e-3 * np.sqrt(bbox_squared) * dt)
    directions = {}
    for name, filename in (("direct", "cpu_direct_solution.f64.bin"),
                           ("candidate", "agipc_candidate.f64.bin"),
                           ("prolongated", "prolongated.f64.bin")):
        vector = read_vector(sample / filename, np.float64, dofs)[prefix:].reshape(nodes, 3)
        if not np.isfinite(vector).all():
            raise ValueError("nonfinite FEM direction")
        maximum = float(np.max(np.linalg.norm(vector, axis=1)))
        directions[name] = {"fem_direction_max": maximum,
                            "max_to_threshold_ratio": maximum / threshold,
                            "fem_stop_component_passed": bool(maximum <= threshold)}
    owners = read_vector(sample / "fine_to_coarse.i32.bin", np.int32, nodes)
    masks = read_vector(sample / "coarse_basis_masks.i32.bin", np.int32,
                        int(metadata["mapping_coarse_nodes"]))
    sizes = np.bincount(owners, minlength=masks.size)
    affine = np.array([int(mask).bit_count() > 1 for mask in masks])
    reference = read_vector(sample / "cpu_direct_solution.f64.bin", np.float64, dofs)[prefix:].reshape(nodes, 3)
    candidate = read_vector(sample / "agipc_candidate.f64.bin", np.float64, dofs)[prefix:].reshape(nodes, 3)
    node_error2 = np.sum((candidate - reference) ** 2, axis=1)
    grouped_error2 = np.bincount(owners, weights=node_error2, minlength=masks.size)
    rest = read_vector(sample / "fine_rest_positions.f64x3.bin", np.float64, 3 * nodes).reshape(nodes, 3)
    phi = np.column_stack((np.ones(nodes), rest))
    fitted = np.zeros_like(reference)
    for group, mask in enumerate(masks):
        selected = np.flatnonzero(owners == group)
        columns = [bit for bit in range(4) if int(mask) & (1 << bit)]
        basis = phi[selected][:, columns]
        coefficients = np.linalg.lstsq(basis, reference[selected], rcond=None)[0]
        fitted[selected] = basis @ coefficients
    return {
        "bbox_diagonal_squared_input": bbox_squared, "dt": dt,
        "newton_fem_displacement_threshold": threshold, "directions": directions,
        "mapping": {"coarse_nodes": int(sizes.size), "group_size_min": int(sizes.min()),
                    "group_size_max": int(sizes.max()), "group_size_median": float(np.median(sizes)),
                    "affine_groups": int(affine.sum()), "affine_group_fine_nodes": int(sizes[affine].sum()),
                    "largest_group_sizes": sorted(map(int, sizes), reverse=True)[:10],
                    "affine_group_fraction_of_candidate_error_squared": float(grouped_error2[affine].sum() / node_error2.sum()),
                    "best_euclidean_subspace_fit_relative_error": float(np.linalg.norm(fitted - reference) / np.linalg.norm(reference))},
        "scope": "FEM norms only, not a mixed-world stop certificate; sampled solve stops before Newton test; bbox may be rounded in the log",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--bbox-diagonal-squared", type=float)
    parser.add_argument("--dt", type=float)
    args = parser.parse_args()
    if args.geometry_only:
        if args.bbox_diagonal_squared is None or args.dt is None:
            parser.error("geometry-only requires bbox-diagonal-squared and dt")
        result = geometry_summary(args.sample, args.bbox_diagonal_squared, args.dt)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    result = analyze(args.sample)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "update_index", "fine_dofs", "direct_seconds", "independent_reference_gate_passed",
        "prolongation_reconstruction_relative_error", "directions", "vs_direct")}, indent=2))
    return 0 if result["independent_reference_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Replay a frozen AGIPC strain criterion and analyze its coarse components."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from analyze_frozen_fallback import read_vector


def component_labels(edges: np.ndarray, allowed: np.ndarray, nodes: int) -> np.ndarray:
    selected = edges[allowed]
    graph = sparse.coo_matrix(
        (np.ones(2 * selected.shape[0], dtype=np.int8),
         (np.concatenate((selected[:, 0], selected[:, 1])),
          np.concatenate((selected[:, 1], selected[:, 0])))),
        shape=(nodes, nodes),
    ).tocsr()
    return connected_components(graph, directed=False, return_labels=True)[1]


def subspace_fit(rest: np.ndarray, reference: np.ndarray, labels: np.ndarray) -> dict:
    groups = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=groups)
    sums = np.zeros((groups, 3))
    np.add.at(sums, labels, reference)
    fitted = sums[labels] / sizes[labels, None]
    blocks = groups
    affine_groups = 0
    affine_nodes = 0
    phi = np.column_stack((np.ones(rest.shape[0]), rest))
    for group in np.flatnonzero(sizes > 32):
        selected = labels == group
        basis = phi[selected]
        coefficients, _, rank, _ = np.linalg.lstsq(basis, reference[selected], rcond=1e-10)
        fitted[selected] = basis @ coefficients
        blocks += int(rank) - 1
        affine_groups += 1
        affine_nodes += int(sizes[group])
    return {
        "coarse_nodes": groups,
        "coarse_block_nodes": int(blocks),
        "active_fem_dof_ratio": float(blocks / rest.shape[0]),
        "affine_groups": affine_groups,
        "affine_group_nodes": affine_nodes,
        "largest_group_sizes": sorted(map(int, sizes), reverse=True)[:10],
        "best_euclidean_subspace_fit_relative_error": float(
            np.linalg.norm(fitted - reference) / max(np.linalg.norm(reference), np.finfo(float).tiny)
        ),
    }


def analyze(sample: Path, thresholds: list[float]) -> dict:
    metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
    snapshot = metadata["criterion_snapshot"]
    if snapshot["format"] != "agipc_criterion_snapshot_v1":
        raise ValueError("unsupported criterion snapshot")
    nodes = int(snapshot["fine_nodes"])
    fine_offset = int(snapshot["fine_offset"])
    elements = int(snapshot["element_count"])
    edge_count = int(snapshot["edge_count"])
    dofs = int(metadata["fine_dofs"])
    prefix = 3 * (int(metadata["fine_block_nodes"]) - nodes)
    vertices = read_vector(sample / "criterion_element_vertices.u32x4.bin", np.uint32,
                           4 * elements).reshape(elements, 4).astype(np.int64)
    dims = read_vector(sample / "criterion_element_dims.i32.bin", np.int32, elements)
    inverses = read_vector(sample / "criterion_element_inverse.f64x9.bin", np.float64,
                           9 * elements).reshape(elements, 3, 3)
    saved_green = read_vector(sample / "criterion_green_current.f64x9.bin", np.float64,
                              9 * elements).reshape(elements, 3, 3)
    increments = read_vector(sample / "criterion_green_increment.f64.bin", np.float64, elements)
    edges = read_vector(sample / "criterion_edges.u32x2.bin", np.uint32,
                        2 * edge_count).reshape(edge_count, 2).astype(np.int64) - fine_offset
    offsets = read_vector(sample / "criterion_edge_offsets.i32.bin", np.int32, edge_count + 1)
    adjacent = read_vector(sample / "criterion_edge_elements.i32.bin", np.int32,
                           int(snapshot["edge_element_entries"]))
    tags = read_vector(sample / "criterion_edge_tags.i32.bin", np.int32, edge_count)
    reasons = read_vector(sample / "criterion_edge_reasons.i32.bin", np.int32, edge_count)
    boundary = read_vector(sample / "criterion_boundary.i32.bin", np.int32, nodes)
    positions = read_vector(sample / "criterion_current_positions.f64x3.bin", np.float64,
                            3 * nodes).reshape(nodes, 3)
    rest = read_vector(sample / "fine_rest_positions.f64x3.bin", np.float64, 3 * nodes).reshape(nodes, 3)
    owners = read_vector(sample / "fine_to_coarse.i32.bin", np.int32, nodes)
    reference = read_vector(sample / "cpu_direct_solution.f64.bin", np.float64, dofs)[prefix:].reshape(nodes, 3)
    if not np.isin(dims, (2, 3)).all() or edges.min() < 0 or edges.max() >= nodes:
        raise ValueError("invalid element dimensions or edge domain")
    if offsets[0] != 0 or offsets[-1] != adjacent.size or (np.diff(offsets) <= 0).any():
        raise ValueError("invalid edge-element offsets")
    if adjacent.min() < 0 or adjacent.max() >= elements:
        raise ValueError("invalid incident-element index")

    computed_green = np.zeros_like(saved_green)
    for dim in (2, 3):
        selected = np.flatnonzero(dims == dim)
        if selected.size == 0:
            continue
        local_vertices = vertices[selected, : dim + 1] - fine_offset
        if local_vertices.min() < 0 or local_vertices.max() >= nodes:
            raise ValueError("element escapes FEM domain")
        ds = positions[local_vertices[:, 1:]] - positions[local_vertices[:, :1]]
        deformation = np.einsum("nkr,nkc->nrc", ds, inverses[selected, :dim, :dim])
        green = 0.5 * (np.einsum("nrc,nrd->ncd", deformation, deformation)
                       - np.eye(dim)[None, :, :])
        computed_green[selected, :dim, :dim] = green
    green_absolute_error = float(np.max(np.abs(computed_green - saved_green)))
    green_relative_error = green_absolute_error / max(float(np.max(np.abs(saved_green))), np.finfo(float).tiny)

    edge_maximum = np.maximum.reduceat(increments[adjacent], offsets[:-1])
    threshold = float(snapshot["criterion"]["threshold"])
    expected_reasons = (edge_maximum > threshold).astype(np.int32)
    expected_reasons |= (np.logical_or(boundary[edges[:, 0]] != 0,
                                       boundary[edges[:, 1]] != 0).astype(np.int32) * 2)
    expected_tags = (expected_reasons == 0).astype(np.int32)
    reason_mismatches = int(np.count_nonzero(reasons != expected_reasons))
    tag_mismatches = int(np.count_nonzero(tags != expected_tags))

    labels = component_labels(edges, tags != 0, nodes)
    group_component_pairs = np.unique(np.column_stack((owners, labels)), axis=0)
    merges_distinct_allowed_components = int(group_component_pairs.shape[0] - np.unique(owners).size)
    remaining = int(np.count_nonzero((tags != 0) & (owners[edges[:, 0]] != owners[edges[:, 1]])))
    protected_internal = int(np.count_nonzero((tags == 0) & (owners[edges[:, 0]] == owners[edges[:, 1]])))
    protected_cut = int(np.count_nonzero((tags == 0) & (owners[edges[:, 0]] != owners[edges[:, 1]])))
    mapping_info = snapshot["mapping"]
    runtime_mapping_fit = subspace_fit(rest, reference, owners)
    exact_component_partition = (
        bool(mapping_info["globally_resolved"])
        and np.unique(labels).size == np.unique(owners).size
        and merges_distinct_allowed_components == 0
        and remaining == 0
    )

    threshold_rows = {}
    for value in sorted(set(thresholds + [threshold])):
        allowed = np.logical_not(np.logical_or(
            edge_maximum > value,
            np.logical_or(boundary[edges[:, 0]] != 0, boundary[edges[:, 1]] != 0),
        ))
        row_labels = component_labels(edges, allowed, nodes)
        row = subspace_fit(rest, reference, row_labels)
        row["threshold"] = value
        row["protected_edges"] = int(np.count_nonzero(~allowed))
        row["protected_edge_ratio"] = float(np.mean(~allowed))
        threshold_rows[f"{value:.12g}"] = row

    criterion = snapshot["criterion"]
    gate = (
        green_relative_error <= 1e-12
        and reason_mismatches == 0 and tag_mismatches == 0
        and merges_distinct_allowed_components == 0
        and remaining == int(mapping_info["remaining_collapsible_edges"])
        and int(np.count_nonzero(tags == 0)) == int(criterion["protected_edge_count"])
        and (not bool(mapping_info["globally_resolved"]) or exact_component_partition)
    )
    return {
        "format": "agipc_criterion_snapshot_analysis_v1",
        "sample": str(sample.resolve()),
        "independent_replay_gate_passed": bool(gate),
        "green_current_relative_error": green_relative_error,
        "reason_mismatches": reason_mismatches,
        "tag_mismatches": tag_mismatches,
        "captured": {
            "threshold": threshold,
            "green_increment_quantiles": dict(zip(
                ("min", "p50", "p90", "p99", "max"),
                map(float, np.quantile(increments, (0, .5, .9, .99, 1))),
            )),
            "protected_edges": int(np.count_nonzero(tags == 0)),
            "strain_protected_edges": int(np.count_nonzero(reasons & 1)),
            "boundary_protected_edges": int(np.count_nonzero(reasons & 2)),
            "protected_internal_to_same_final_group": protected_internal,
            "protected_cut_between_final_groups": protected_cut,
            "allowed_connected_components": int(np.unique(labels).size),
            "mapping_groups": int(np.unique(owners).size),
            "merges_distinct_allowed_components": merges_distinct_allowed_components,
            "remaining_allowed_cross_group_edges": remaining,
            "exact_allowed_component_partition": bool(exact_component_partition),
            "runtime_mapping_subspace_fit": runtime_mapping_fit,
            "mapping": mapping_info,
        },
        "threshold_ablation_same_increment_field": threshold_rows,
        "history_scope": "current Green tensor independently replayed; signed prior tensor was not captured, so delta norm history itself is not independently reconstructed",
        "interpretation_scope": "single Newton system; threshold retagging/subspace fits are offline diagnostics, not rerun trajectories",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sample", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, action="append", default=[])
    args = parser.parse_args()
    result = analyze(args.sample, args.threshold or [1e-6, 2.5e-6, 5e-6, 1e-5, 2.5e-5, 1e-4])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["independent_replay_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

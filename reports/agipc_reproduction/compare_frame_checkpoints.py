"""Compare two compact full-FEM frame checkpoint streams."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_checkpoint(manifest_path: Path) -> tuple[dict, dict[int, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format"] != "fem_frame_checkpoints_f64x3_v1":
        raise ValueError(f"unsupported checkpoint format: {manifest['format']}")
    binary_path = manifest_path.parent / manifest["binary_file"]
    frame_ids = list(map(int, manifest["record_frame_ids"]))
    vertices = int(manifest["fem_vertices"])
    raw = np.fromfile(binary_path, dtype=np.float64)
    expected = len(frame_ids) * vertices * 3
    if raw.size != expected:
        raise ValueError(f"{binary_path}: expected {expected} scalars, found {raw.size}")
    states = raw.reshape(len(frame_ids), vertices, 3)
    if not np.isfinite(states).all() or len(set(frame_ids)) != len(frame_ids):
        raise ValueError("non-finite checkpoint or duplicate frame id")
    return manifest, dict(zip(frame_ids, states))


def first_crossing(rows: list[dict], field: str, threshold: float) -> int | None:
    return next((row["frame"] for row in rows if row[field] >= threshold), None)


def nested(record: dict, *keys):
    value = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def compare_newton_stats(reference_path: Path, comparison_path: Path) -> dict:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))["frames"]
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))["frames"]

    def first_mismatch(keys: tuple[str, ...], applied_only: bool = False) -> dict | None:
        for frame_index, (ref_frame, cmp_frame) in enumerate(zip(reference, comparison)):
            for newton, (ref_row, cmp_row) in enumerate(
                    zip(ref_frame.get("newton", []), cmp_frame.get("newton", []))):
                if applied_only and (nested(ref_row, "line_search", "accepted_alpha") is None
                                     or nested(cmp_row, "line_search", "accepted_alpha") is None):
                    continue
                ref_value, cmp_value = nested(ref_row, *keys), nested(cmp_row, *keys)
                if ref_value != cmp_value:
                    return {"frame": frame_index + 1, "newton": newton,
                            "reference": ref_value, "comparison": cmp_value}
        return None

    protected = first_mismatch(("agipc_criterion", "protected_edge_count"))
    mapping = first_mismatch(("agipc_mapping", "coarse_nodes"))
    adoption = first_mismatch(("agipc_galerkin", "last_adoption", "adopted"), applied_only=True)
    focal = protected["frame"] if protected else None
    focal_rows = []
    if focal is not None:
        ref_rows = reference[focal - 1].get("newton", [])
        cmp_rows = comparison[focal - 1].get("newton", [])
        for newton, (ref_row, cmp_row) in enumerate(zip(ref_rows[:4], cmp_rows[:4])):
            ref_direction = nested(ref_row, "current_direction_norm")
            cmp_direction = nested(cmp_row, "current_direction_norm")
            focal_rows.append({
                "newton": newton,
                "update": nested(ref_row, "agipc_galerkin", "updates"),
                "direction_norm_reference": ref_direction,
                "direction_norm_comparison": cmp_direction,
                "direction_norm_relative_difference": abs(ref_direction - cmp_direction)
                    / max(abs(ref_direction), np.finfo(float).tiny),
                "accepted_alpha_reference": nested(ref_row, "line_search", "accepted_alpha"),
                "accepted_alpha_comparison": nested(cmp_row, "line_search", "accepted_alpha"),
                "protected_edges_reference": nested(ref_row, "agipc_criterion", "protected_edge_count"),
                "protected_edges_comparison": nested(cmp_row, "agipc_criterion", "protected_edge_count"),
                "coarse_nodes_reference": nested(ref_row, "agipc_mapping", "coarse_nodes"),
                "coarse_nodes_comparison": nested(cmp_row, "agipc_mapping", "coarse_nodes"),
            })
    return {
        "first_protected_edge_count_mismatch": protected,
        "first_coarse_node_count_mismatch": mapping,
        "first_applied_adoption_mismatch": adoption,
        "focal_frame_first_four_newton_records": focal_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_manifest", type=Path)
    parser.add_argument("comparison_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-stem", type=Path)
    parser.add_argument("--reference-stats", type=Path)
    parser.add_argument("--comparison-stats", type=Path)
    args = parser.parse_args()

    ref_manifest, ref = load_checkpoint(args.reference_manifest)
    cmp_manifest, cmp = load_checkpoint(args.comparison_manifest)
    if ref_manifest["fem_vertices"] != cmp_manifest["fem_vertices"]:
        raise ValueError("FEM vertex counts differ")
    if 0 not in ref or 0 not in cmp:
        raise ValueError("initial state record is missing")
    initial_difference = float(np.sqrt(np.mean(np.sum((ref[0] - cmp[0]) ** 2, axis=1))))
    common = sorted((set(ref) & set(cmp)) - {0})
    if not common:
        raise ValueError("no common captured frames")
    ref_summary = {int(row["frame"]): row for row in ref_manifest["frames"]}
    cmp_summary = {int(row["frame"]): row for row in cmp_manifest["frames"]}
    rows = []
    for frame in common:
        delta = cmp[frame] - ref[frame]
        point_error = np.linalg.norm(delta, axis=1)
        ref_displacement = ref[frame] - ref[0]
        rms_difference = float(np.sqrt(np.mean(point_error ** 2)))
        rms_reference_displacement = float(
            np.sqrt(np.mean(np.sum(ref_displacement ** 2, axis=1)))
        )
        row = {
            "frame": frame,
            "rms_position_difference": rms_difference,
            "relative_rms_difference": rms_difference / max(
                rms_reference_displacement, np.finfo(float).tiny
            ),
            "maximum_pointwise_difference": float(point_error.max()),
            "rms_reference_displacement": rms_reference_displacement,
            "mean_position_difference": float(np.linalg.norm(delta.mean(axis=0))),
            "reference_summary": ref_summary.get(frame),
            "comparison_summary": cmp_summary.get(frame),
        }
        rows.append(row)

    absolute_thresholds = (1e-6, 1e-5, 1e-4, 1e-3)
    relative_thresholds = (1e-4, 1e-3, 1e-2)
    result = {
        "format": "fem_frame_checkpoint_comparison_v1",
        "reference_manifest": str(args.reference_manifest.resolve()),
        "comparison_manifest": str(args.comparison_manifest.resolve()),
        "structural_gate_passed": bool(initial_difference <= 1e-14),
        "fem_vertices": int(ref_manifest["fem_vertices"]),
        "initial_rms_difference": initial_difference,
        "common_frames": common,
        "first_absolute_rms_crossing": {
            f"{value:.0e}": first_crossing(rows, "rms_position_difference", value)
            for value in absolute_thresholds
        },
        "first_relative_rms_crossing": {
            f"{value:.0e}": first_crossing(rows, "relative_rms_difference", value)
            for value in relative_thresholds
        },
        "maximum_rms_difference": max(row["rms_position_difference"] for row in rows),
        "maximum_relative_rms_difference": max(row["relative_rms_difference"] for row in rows),
        "final_common_frame": rows[-1],
        "frames": rows,
        "performance_claim": False,
    }
    if bool(args.reference_stats) != bool(args.comparison_stats):
        raise ValueError("reference and comparison stats must be supplied together")
    if args.reference_stats:
        result["newton_diagnostics"] = compare_newton_stats(
            args.reference_stats, args.comparison_stats
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if args.plot_stem:
        x = np.array(common)
        relative = 100 * np.array([row["relative_rms_difference"] for row in rows])
        absolute = np.array([row["rms_position_difference"] for row in rows])
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        ax.semilogy(x, np.maximum(relative, np.finfo(float).tiny), "o-",
                    label="relative RMS difference")
        ax.set_xlabel("completed frame")
        ax.set_ylabel("relative RMS position difference (%)")
        ax.grid(True, which="both", alpha=.25)
        other = ax.twinx()
        other.semilogy(x, np.maximum(absolute, np.finfo(float).tiny), "s--", color="tab:orange",
                       label="absolute RMS difference")
        other.set_ylabel("absolute RMS position difference")
        handles = ax.lines + other.lines
        ax.legend(handles, [line.get_label() for line in handles], loc="best")
        fig.suptitle("FEM trajectory divergence from full-frame checkpoints")
        args.plot_stem.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_stem.with_suffix(".png"), dpi=200)
        fig.savefig(args.plot_stem.with_suffix(".pdf"))
        plt.close(fig)

    print(json.dumps({
        "structural_gate_passed": result["structural_gate_passed"],
        "initial_rms_difference": initial_difference,
        "common_frames": common,
        "first_absolute_rms_crossing": result["first_absolute_rms_crossing"],
        "first_relative_rms_crossing": result["first_relative_rms_crossing"],
        "final_common_frame": rows[-1],
        "newton_diagnostics": result.get("newton_diagnostics"),
    }, indent=2))
    return 0 if result["structural_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

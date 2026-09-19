"""Analyze one preliminary StiffGIPC/AGIPC performance pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def state(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if not all(np.isfinite(data[name]).all() for name in ("vertex", "x", "y", "z")):
        raise ValueError(f"non-finite state: {path}")
    return data


def frame_iterations(path: Path, expected_frames: int, expected_total: int) -> dict:
    frames, pending = [], 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        iteration = re.search(r"iteration k:\s*(\d+)", line)
        frame = re.search(r"frame id:\s*(\d+)", line)
        if iteration:
            pending += int(iteration.group(1))
        if frame:
            frames.append({"frame": int(frame.group(1)), "applied_newton_iterations": pending})
            pending = 0
    if pending or len(frames) != expected_frames or sum(x["applied_newton_iterations"] for x in frames) != expected_total:
        raise ValueError(f"frame iteration log disagrees with metrics: {path}")
    return {"frames": frames,
            "worst_frames": sorted(frames, key=lambda row: row["applied_newton_iterations"], reverse=True)[:5]}


def gpu_summary(path: Path) -> dict:
    rows = load_json(path)
    if isinstance(rows, dict):
        rows = [rows]
    gpu = np.array([row["gpu"] for row in rows], dtype=float)
    memory = np.array([row["memory_util"] for row in rows], dtype=float)
    temperature = np.array([row["temperature"] for row in rows], dtype=float)
    return {
        "samples": len(rows),
        "gpu_utilization_min_percent": float(gpu.min()),
        "gpu_utilization_median_percent": float(np.median(gpu)),
        "gpu_utilization_max_percent": float(gpu.max()),
        "memory_utilization_median_percent": float(np.median(memory)),
        "temperature_max_c": float(temperature.max()),
        "pstates": sorted(set(row["pstate"] for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_metrics", type=Path)
    parser.add_argument("agipc_metrics", type=Path)
    parser.add_argument("--baseline-state", type=Path, required=True)
    parser.add_argument("--agipc-state", type=Path, required=True)
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--agipc-log", type=Path, required=True)
    parser.add_argument("--baseline-gpu", type=Path, required=True)
    parser.add_argument("--agipc-gpu", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline, agipc = load_json(args.baseline_metrics), load_json(args.agipc_metrics)
    matched_keys = ("scene", "cloth_mesh", "framework", "preconditioner", "young_modulus", "dt",
                    "frames_requested", "frames_completed", "newton_stop_effective", "linear_system_dof")
    mismatches = {key: [baseline.get(key), agipc.get(key)] for key in matched_keys
                  if baseline.get(key) != agipc.get(key)}
    if baseline["solver"] != "stiffgipc" or agipc["solver"] != "agipc-core":
        raise ValueError("expected StiffGIPC baseline and AGIPC-Core comparison")

    baseline_state, agipc_state = state(args.baseline_state), state(args.agipc_state)
    if not np.array_equal(baseline_state["vertex"], agipc_state["vertex"]):
        raise ValueError("FEM vertex IDs differ")
    delta = np.column_stack([agipc_state[name] - baseline_state[name] for name in ("x", "y", "z")])
    lengths = np.linalg.norm(delta, axis=1)
    rms = float(np.sqrt(np.mean(lengths ** 2)))
    baseline_ms, agipc_ms = baseline["simulation_time_ms"], agipc["simulation_time_ms"]
    coarse = agipc["agipc_criterion"]["galerkin_shadow"]
    baseline_gpu, agipc_gpu = gpu_summary(args.baseline_gpu), gpu_summary(args.agipc_gpu)
    environment_idle = max(baseline_gpu["gpu_utilization_median_percent"],
                           agipc_gpu["gpu_utilization_median_percent"]) <= 5
    result = {
        "format": "agipc_single_preliminary_performance_pair_v1",
        "configuration_gate_passed": not mismatches,
        "configuration_mismatches": mismatches,
        "executable_sha256": sha256(args.executable),
        "mesh_sha256": sha256(args.mesh),
        "baseline": {
            "simulation_time_ms": baseline_ms,
            "simulation_wall_time_ms": baseline["simulation_wall_time_ms"],
            "milliseconds_per_frame": baseline_ms / baseline["frames_completed"],
            "milliseconds_per_applied_newton": baseline_ms / baseline["newton_iterations"],
            "newton_iterations": baseline["newton_iterations"],
            "pcg_iterations": baseline["pcg_iterations"],
            "finite_vertices": baseline["finite_vertices"],
            "ground_penetration": baseline["ground_penetration"],
            "maximum_self_collision_pairs": baseline["maximum_self_collision_pairs"],
        },
        "agipc": {
            "simulation_time_ms": agipc_ms,
            "simulation_wall_time_ms": agipc["simulation_wall_time_ms"],
            "milliseconds_per_frame": agipc_ms / agipc["frames_completed"],
            "milliseconds_per_applied_newton": agipc_ms / agipc["newton_iterations"],
            "newton_iterations": agipc["newton_iterations"],
            "pcg_iterations": agipc["pcg_iterations"],
            "finite_vertices": agipc["finite_vertices"],
            "ground_penetration": agipc["ground_penetration"],
            "maximum_self_collision_pairs": agipc["maximum_self_collision_pairs"],
            "adoption_attempts": coarse["adoption_attempts"],
            "adoptions": coarse["adoptions"],
            "fallbacks": coarse["fallbacks"],
            "fallback_reason_counts": coarse["fallback_reason_counts"],
            "residual_guard_restores": coarse["residual_guard_restores"],
            "mas_local_failures": coarse["mas_local_failures"],
            "mas_graph_reuses": coarse["mas_graph_reuses"],
            "stage_timing_ms": coarse["total_stage_timing_ms"],
        },
        "ratios_agipc_over_baseline": {
            "simulation_time": agipc_ms / baseline_ms,
            "simulation_wall_time": agipc["simulation_wall_time_ms"] / baseline["simulation_wall_time_ms"],
            "milliseconds_per_applied_newton": (agipc_ms / agipc["newton_iterations"])
                / (baseline_ms / baseline["newton_iterations"]),
            "newton_iterations": agipc["newton_iterations"] / baseline["newton_iterations"],
            "pcg_iterations": agipc["pcg_iterations"] / baseline["pcg_iterations"],
        },
        "terminal_state": {
            "vertices": len(lengths),
            "rms_position_difference": rms,
            "maximum_position_difference": float(lengths.max()),
            "relative_to_baseline_rms_displacement": rms / max(baseline["fem_rms_displacement"], 1e-300),
        },
        "frame_iterations": {
            "baseline": frame_iterations(args.baseline_log, baseline["frames_completed"], baseline["newton_iterations"]),
            "agipc": frame_iterations(args.agipc_log, agipc["frames_completed"], agipc["newton_iterations"]),
        },
        "gpu_before_runs": {"baseline": baseline_gpu, "agipc": agipc_gpu},
        "formal_performance_claim_eligible": False,
        "performance_gate_failures": [
            "one measured pair; fewer than three interleaved repetitions",
            "GPU idle gate failed" if not environment_idle else "GPU idle gate needs longer verification",
            "methods followed different nonlinear trajectories",
            "scaled local asset and RTX 3070 Laptop do not match the paper benchmark",
        ],
        "performance_claim": False,
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["configuration_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compare matched runtime cases and their original-vertex FEM states."""

import argparse
import json
from pathlib import Path

import numpy as np


def same_parameter(key: str, left, right) -> bool:
    return Path(left) == Path(right) if key == "cloth_mesh" else left == right


def state(metrics: dict, report_path: Path) -> np.ndarray:
    stored = Path(metrics["fem_final_state_path"])
    path = stored if stored.is_file() else report_path.parent / stored.name
    result = np.genfromtxt(path, delimiter=",", names=True)
    if not all(np.isfinite(result[name]).all() for name in ("vertex", "x", "y", "z")):
        raise ValueError("nonfinite final-state CSV")
    return result


def difference(left: np.ndarray, right: np.ndarray, reference_displacement: float) -> dict:
    if not np.array_equal(left["vertex"], right["vertex"]):
        raise ValueError("FEM vertex IDs differ")
    delta = np.column_stack([left[name] - right[name] for name in ("x", "y", "z")])
    lengths = np.linalg.norm(delta, axis=1)
    rms = float(np.sqrt(np.mean(lengths**2)))
    return {"vertices": len(left), "rms_position_difference": rms,
            "maximum_position_difference": float(lengths.max()),
            "relative_to_reference_rms_displacement": rms / max(reference_displacement, 1e-300)}


def row(metrics: dict) -> dict:
    coarse = metrics["agipc_criterion"]["galerkin_shadow"]
    return {**{name: metrics[name] for name in (
        "agipc_coarse_preconditioner", "frames_completed", "newton_iterations", "pcg_iterations",
        "finite_vertices", "ground_penetration", "abd_maximum_displacement", "maximum_self_collision_pairs")},
        **{name: coarse[name] for name in (
            "adoption_attempts", "adoptions", "fallbacks", "fallback_reason_counts", "residual_guard_restores",
            "mas_attempts", "mas_local_failures", "total_mas_setup_wall_ms", "total_mas_validation_wall_ms")},
        "stage_timing_ms": coarse["total_stage_timing_ms"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jacobi", type=Path)
    parser.add_argument("mas", type=Path)
    parser.add_argument("--fine-baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    jacobi = json.loads(args.jacobi.read_text(encoding="utf-8"))
    mas = json.loads(args.mas.read_text(encoding="utf-8"))
    for key in ("scene", "solver", "cloth_mesh", "framework", "preconditioner", "young_modulus", "dt",
                "frames_requested", "frames_completed", "newton_stop_effective", "linear_system_dof"):
        if not same_parameter(key, jacobi[key], mas[key]):
            raise ValueError(f"runtime pair differs in {key}")
    if jacobi["agipc_coarse_preconditioner"] != "block-jacobi" or mas["agipc_coarse_preconditioner"] != "mas32":
        raise ValueError("expected Jacobi/MAS32 runtime modes")
    jacobi_state, mas_state = state(jacobi, args.jacobi), state(mas, args.mas)
    report = {"jacobi": row(jacobi), "mas32": row(mas),
              "mas_vs_jacobi": difference(mas_state, jacobi_state, jacobi["fem_rms_displacement"]),
              "performance_claim": False}
    if args.fine_baseline:
        fine = json.loads(args.fine_baseline.read_text(encoding="utf-8"))
        for key in ("scene", "cloth_mesh", "framework", "preconditioner", "linear_system_dof",
                    "young_modulus", "dt", "frames_completed", "newton_stop_effective"):
            if not same_parameter(key, fine[key], jacobi[key]):
                raise ValueError(f"fine baseline differs in {key}")
        fine_state = state(fine, args.fine_baseline)
        report["jacobi_vs_fine"] = difference(jacobi_state, fine_state, fine["fem_rms_displacement"])
        report["mas_vs_fine"] = difference(mas_state, fine_state, fine["fem_rms_displacement"])
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

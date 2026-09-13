#!/usr/bin/env python3
"""Check matched MAS runtime ablations and compare original-vertex states."""

import argparse
import hashlib
import json
from pathlib import Path

from compare_coarse_runtime import difference, frame_iterations, row, same_parameter, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path, nargs="+")
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    loaded = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in args.cases]
    reference_path, reference = loaded[0]
    reference_state = state(reference, reference_path)
    rows, differences, frame_counts = {}, {}, {}
    for path, metrics in loaded:
        for key in ("scene", "solver", "cloth_mesh", "framework", "preconditioner",
                    "young_modulus", "dt", "frames_requested", "frames_completed",
                    "newton_stop_effective", "linear_system_dof", "agipc_criterion_threshold"):
            if not same_parameter(key, reference[key], metrics[key]):
                raise ValueError(f"{path.name} differs in {key}")
        if (metrics["agipc_criterion"]["galerkin_shadow"]["post_correction"]["max_iterations"]
                != reference["agipc_criterion"]["galerkin_shadow"]["post_correction"]["max_iterations"]):
            raise ValueError(f"{path.name} differs in fine correction cap")
        if metrics["agipc_coarse_preconditioner"] != "mas32":
            raise ValueError("expected MAS32 ablation")
        if not metrics["finite_vertices"] or metrics["frames_completed"] != metrics["frames_requested"]:
            raise ValueError(f"incomplete or nonfinite case: {path}")
        if metrics["agipc_criterion"]["galerkin_shadow"]["mas_local_failures"]:
            raise ValueError(f"local diagnostic rejection in {path}; inspect before acceptance")
        rows[path.stem] = row(metrics)
        frame_counts[path.stem] = frame_iterations(metrics, path)
        differences[path.stem] = difference(state(metrics, path), reference_state,
                                            reference["fem_rms_displacement"])
    report = {"reference": str(reference_path), "cases": rows,
              "terminal_differences_to_reference": differences,
              "frame_iterations": frame_counts,
              "performance_claim": False,
              "hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in (("executable_sha256", args.executable),
                                            ("mesh_sha256", args.mesh))}}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

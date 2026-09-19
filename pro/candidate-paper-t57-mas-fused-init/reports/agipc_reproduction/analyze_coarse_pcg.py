#!/usr/bin/env python3
"""Summarize coarse-PCG size, iteration, timing, and outcome distributions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "sum": sum(values),
    }


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_energy = sum((x - left_mean) ** 2 for x in left)
    right_energy = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_energy * right_energy)
    return numerator / denominator if denominator else None


def coarse_bin(blocks: int) -> str:
    if blocks <= 16:
        return "001-016"
    if blocks <= 64:
        return "017-064"
    if blocks <= 256:
        return "065-256"
    if blocks <= 1024:
        return "257-1024"
    return "1025+"


def records_from(path: Path) -> list[dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict] = []
    for frame_index, frame in enumerate(document["frames"], start=1):
        for newton_index, newton in enumerate(frame.get("newton", []), start=1):
            galerkin = newton.get("agipc_galerkin")
            if not isinstance(galerkin, dict):
                continue
            coarse = galerkin.get("coarse_solve")
            if not isinstance(coarse, dict) or not coarse.get("attempted"):
                continue
            adoption = galerkin.get("last_adoption", {})
            stage = galerkin.get("stage_timing_ms", {})
            records.append(
                {
                    "frame": frame_index,
                    "newton": newton_index,
                    "coarse_fem_nodes": int(galerkin.get("coarse_fem_nodes", 0)),
                    "coarse_block_nodes": int(galerkin.get("coarse_block_nodes", 0)),
                    "coarse_unique_blocks": int(galerkin.get("coarse_unique_blocks", 0)),
                    "expanded_blocks": int(galerkin.get("expanded_blocks_before_reduction", 0)),
                    "iterations": int(coarse.get("iterations", 0)),
                    "max_iterations": int(coarse.get("max_iterations", 0)),
                    "coarse_solve_ms": float(stage.get("coarse_solve", coarse.get("solve_ms", 0.0))),
                    "fine_residual_ratio": (
                        float(coarse["fine_residual_ratio"])
                        if "fine_residual_ratio" in coarse
                        else None
                    ),
                    "coarse_converged": bool(coarse.get("converged", False)),
                    "coarse_failure": str(coarse.get("failure_reason", "")),
                    "adopted": bool(adoption.get("adopted", False)),
                    "adoption_reason": str(adoption.get("reason", "")),
                }
            )
    return records


def group_summary(records: list[dict]) -> dict:
    return {
        "attempts": len(records),
        "coarse_block_nodes": distribution([row["coarse_block_nodes"] for row in records]),
        "coarse_unique_blocks": distribution([row["coarse_unique_blocks"] for row in records]),
        "iterations": distribution([row["iterations"] for row in records]),
        "coarse_solve_ms": distribution([row["coarse_solve_ms"] for row in records]),
    }


def summarize(path: Path) -> dict:
    records = records_from(path)
    total_ms = sum(row["coarse_solve_ms"] for row in records)
    ranked = sorted(records, key=lambda row: row["coarse_solve_ms"], reverse=True)
    bins: dict[str, list[dict]] = {}
    for row in records:
        bins.setdefault(coarse_bin(row["coarse_block_nodes"]), []).append(row)

    outcomes: dict[str, list[dict]] = {}
    for row in records:
        key = "adopted" if row["adopted"] else f"fallback:{row['adoption_reason'] or 'unknown'}"
        outcomes.setdefault(key, []).append(row)

    blocks = [float(row["coarse_block_nodes"]) for row in records]
    unique = [float(row["coarse_unique_blocks"]) for row in records]
    iterations = [float(row["iterations"]) for row in records]
    solve_ms = [row["coarse_solve_ms"] for row in records]
    return {
        "source": str(path),
        **group_summary(records),
        "converged": sum(row["coarse_converged"] for row in records),
        "cost_concentration": {
            f"top_{count}_share": sum(row["coarse_solve_ms"] for row in ranked[:count])
            / total_ms
            if total_ms
            else 0.0
            for count in (1, 5, 10)
        },
        "correlations": {
            "log_coarse_blocks_vs_iterations": pearson(
                [math.log1p(value) for value in blocks], iterations
            ),
            "log_coarse_blocks_vs_solve_ms": pearson(
                [math.log1p(value) for value in blocks], solve_ms
            ),
            "log_unique_blocks_vs_solve_ms": pearson(
                [math.log1p(value) for value in unique], solve_ms
            ),
            "iterations_vs_solve_ms": pearson(iterations, solve_ms),
        },
        "size_bins": {
            key: group_summary(bins[key])
            for key in ("001-016", "017-064", "065-256", "257-1024", "1025+")
            if key in bins
        },
        "outcomes": {key: group_summary(rows) for key, rows in sorted(outcomes.items())},
        "highest_cost_events": ranked[:10],
        "highest_iteration_events": sorted(
            records, key=lambda row: (row["iterations"], row["coarse_solve_ms"]), reverse=True
        )[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    arguments = parser.parse_args()
    report = {
        "format": "agipc_coarse_pcg_analysis_v1",
        "runs": [summarize(path) for path in arguments.stats],
    }
    output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if arguments.output:
        arguments.output.write_text(output + "\n", encoding="utf-8")
    if not arguments.quiet:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

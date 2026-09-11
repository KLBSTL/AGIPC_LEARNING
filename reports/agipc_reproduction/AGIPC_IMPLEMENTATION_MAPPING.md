# AGIPC Implementation Mapping

Status: 2026-09-11, validated shadow path. Adaptive solver dispatch remains intentionally unavailable.

| Paper component | Implementation | Current status |
|---|---|---|
| Green strain increment, Eq. (3) | `StiffGIPC/agipc/agipc_criterion.cu`: `green_increment`, `tag_edges` | GPU, FP64, tet and triangle, strict `>` threshold, timestep reset plus Newton history |
| Recursive warp hash | `agipc_criterion.cu`: `build_mapping` and mapping kernels | GPU, immutable fine graph, 32-lane closure, hierarchy composition, deterministic owner checks |
| 3/12-DoF classification | `agipc_criterion.cu`: `classify_coarse_nodes`, `write_coarse_layout` | `child_count > 32` affine rule; translational nodes sorted first |
| Restriction and Galerkin assembly | `StiffGIPC/agipc/agipc_galerkin.cu` | Mixed `1x1/1x4/4x1/4x4` block expansion, rest basis `[1,X,Y,Z]`, ABD prefix identity, half-storage transpose handling |
| Coarse solve and prolongation | `agipc_galerkin.cu`: `solve_coarse_shadow`, `prolongate_mixed` | FP64 block-Jacobi PCG shadow, relative residual `1e-3`, signed-curvature checks; does not control simulation |
| Fine system hook | `StiffGIPC/linear_system/linear_system/global_linear_system.cu` | Runs after the fine BCOO converter, using `h_unique_key_number`; fine solver input is not modified |
| Runtime truthfulness | `StiffGIPC/gipc/runtime_options.*`, `StiffGIPC/gl_main.cu` | Parses Core/SymHessian/Paper names but exits with code 2 until adaptive adoption gates pass |
| MAS capacity prerequisite | `StiffGIPC/gipc/hierarchy_capacity.h`, `StiffGIPC/MASPreconditioner.*` | Semantic capacity alignment and bounds checks backported |

Still required before enabling `--solver agipc-core`: nonzero-initial-state post-PCG, actual direction adoption/fallback accounting, fine-contact regressions, paper Newton criterion, and performance experiments. MAS on the coarse system remains a later comparison; the current block-Jacobi coarse preconditioner is an intermediate numerical gate.

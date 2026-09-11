# AGIPC Implementation Mapping

Status: 2026-09-11, experimental AGIPC-Core dispatch validated on the frozen SPD fixture and one-frame cube. Symmetric-Hessian and paper-BVH modes remain unavailable.

| Paper component | Implementation | Current status |
|---|---|---|
| Green strain increment, Eq. (3) | `StiffGIPC/agipc/agipc_criterion.cu`: `green_increment`, `tag_edges` | GPU, FP64, tet and triangle, strict `>` threshold, timestep reset plus Newton history |
| Recursive warp hash | `agipc_criterion.cu`: `build_mapping` and mapping kernels | GPU, immutable fine graph, 32-lane closure, hierarchy composition, deterministic owner checks |
| 3/12-DoF classification | `agipc_criterion.cu`: `classify_coarse_nodes`, `write_coarse_layout` | `child_count > 32` affine rule; translational nodes sorted first |
| Restriction and Galerkin assembly | `StiffGIPC/agipc/agipc_galerkin.cu` | Mixed `1x1/1x4/4x1/4x4` block expansion, rest basis `[1,X,Y,Z]`, ABD prefix identity, half-storage transpose handling |
| Coarse solve and prolongation | `agipc_galerkin.cu`: `solve_coarse_shadow`, `prolongate_mixed` | FP64 block-Jacobi PCG, relative residual `1e-3`, signed-curvature checks |
| Fine post-PCG | `agipc_galerkin.cu`: `post_correct_shadow` | Starts from the nonzero prolongated solution, uses the unique fine BCOO and block-Jacobi preconditioner, relative residual `1e-3`, runtime cap (default 10), exact final residual plus recurrence/curvature traces |
| Guarded candidate adoption | `agipc_galerkin.cu`: `adopt_galerkin_candidate`; `global_linear_system.cu`: `solve_linear_system` | Requires a complete map and passes dimension, finite-value, descent and residual-growth gates; otherwise runs the original fine solver and records the fallback |
| Fine system hook | `StiffGIPC/linear_system/linear_system/global_linear_system.cu` | Runs after the fine BCOO converter, using `h_unique_key_number`; `agipc-core` adopts the guarded candidate while baseline mode remains unchanged |
| Runtime truthfulness | `StiffGIPC/gipc/runtime_options.*`, `StiffGIPC/gl_main.cu` | `agipc-core` dispatches to the implemented route; SymHessian/Paper exit code 2 with the missing stages named |
| MAS capacity prerequisite | `StiffGIPC/gipc/hierarchy_capacity.h`, `StiffGIPC/MASPreconditioner.*` | Semantic capacity alignment and bounds checks backported |

Still required before treating `--solver agipc-core` as broadly validated: ground/self-contact transitions, fallback fixtures, equal-frame state comparisons on larger meshes, and repeated performance experiments. The paper Newton criterion, symmetric Hessian and stackless BVH are separate remaining stages. MAS on the coarse system remains a later comparison; the current block-Jacobi coarse preconditioner is an intermediate route.

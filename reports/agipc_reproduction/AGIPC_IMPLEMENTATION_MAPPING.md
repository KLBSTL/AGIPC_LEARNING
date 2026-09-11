# AGIPC Implementation Mapping

Status: 2026-09-11, experimental AGIPC-Core dispatch validated on the frozen SPD fixture, one-frame cube, a 30-frame ground-contact transition, a reduced mixed ABD/FEM scene, a 35-frame reduced cloth-on-fixed-ABD self-contact scene, and a 35-frame 16,641-node contact gate. Large-contact warp-hash completion, symmetric-Hessian, and paper-BVH work remain open.

| Paper component | Implementation | Current status |
|---|---|---|
| Green strain increment, Eq. (3) | `StiffGIPC/agipc/agipc_criterion.cu`: `green_increment`, `tag_edges` | GPU, FP64, tet and triangle, strict `>` threshold, timestep reset plus Newton history |
| Recursive warp hash | `agipc_criterion.cu`: `build_mapping` and mapping kernels | GPU, immutable fine graph, 32-lane closure, hierarchy composition, deterministic owner checks |
| 3/12-DoF classification | `agipc_criterion.cu`: `affine_basis_mask`, `write_coarse_layout` | `child_count > 32` affine rule; translational nodes sorted first; dependent rest-coordinate columns removed for planar/linear aggregates |
| Restriction and Galerkin assembly | `StiffGIPC/agipc/agipc_galerkin.cu` | Mixed variable-width block expansion, rank-aware subset of rest basis `[1,X,Y,Z]`, ABD prefix identity, half-storage transpose handling |
| Coarse solve and prolongation | `agipc_galerkin.cu`: `solve_coarse_shadow`, `prolongate_mixed` | FP64 block-Jacobi PCG, relative residual `1e-3`, signed-curvature checks |
| Fine post-PCG | `agipc_galerkin.cu`: `post_correct_shadow` | Starts from the nonzero prolongated solution, uses the unique fine BCOO and block-Jacobi preconditioner, relative residual `1e-3`, runtime cap (default 10), exact final residual plus recurrence/curvature traces |
| Guarded candidate adoption | `agipc_galerkin.cu`: `adopt_galerkin_candidate`; `global_linear_system.cu`: `solve_linear_system` | Requires a complete map and passes dimension, finite-value, descent and residual-growth gates; otherwise runs the original fine solver and records the fallback |
| Fine system hook | `StiffGIPC/linear_system/linear_system/global_linear_system.cu` | Runs after the fine BCOO converter, using `h_unique_key_number`; `agipc-core` adopts the guarded candidate while baseline mode remains unchanged |
| Runtime truthfulness | `StiffGIPC/gipc/runtime_options.*`, `StiffGIPC/gl_main.cu` | `agipc-core` dispatches to the implemented route; SymHessian/Paper exit code 2 with the missing stages named |
| Newton termination | `StiffGIPC/GIPC.cu`: `solve_subIP` | AGIPC-Core checks the newly solved direction against the paper `1e-3 * bbox_diagonal * dt`; StiffGIPC retains its prior previous-direction check and scene threshold |
| Reduced Figure 15 fixture and state output | `StiffGIPC/gl_main.cu`: `set_case_fig15_cloth_abd_scaled`, `run_headless`; `StiffGIPC/gipc/runtime_options.*` | Fixed ABD sphere plus independently falling FEM cloth at `E=1e6`, `dt=.01`; JSON contact/terminal summaries and optional FP64 FEM CSV support direct paired-state error |
| MAS capacity prerequisite | `StiffGIPC/gipc/hierarchy_capacity.h`, `StiffGIPC/MASPreconditioner.*` | Semantic capacity alignment and bounds checks backported |

Still required before treating `--solver agipc-core` as broadly validated: explicit contact appearance/disappearance checks within Newton iterations, a paper-consistent resolution for cross-group warp-hash fixed points, downstream candidate-failure accounting, and repeated performance experiments after adoption improves. Symmetric Hessian and stackless BVH are separate remaining stages. MAS on the coarse system remains a later comparison; the current block-Jacobi coarse preconditioner is an intermediate route.

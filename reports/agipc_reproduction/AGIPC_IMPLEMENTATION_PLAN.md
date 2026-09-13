# AGIPC implementation plan

> Execution: keep every numerical gate explicit and run only the focused checks needed for each stage.

**Goal:** reproduce adaptive GPU IPC on the validated StiffGIPC branch with independently demonstrated numerical correctness, adaptive behavior, solver speedup and end-to-end speedup.

**Architecture:** preserve the fine physics pipeline and introduce a separate adaptive linear-solver workspace. Split topology/history, hash mapping, mixed Galerkin assembly and coarse/post-PCG into small modules. Enable a runtime mode only when it actually dispatches to a validated implementation.

**Tech stack:** C++17, CUDA FP64/sm86, existing GIPCTripletMatrix/Converter/SRBK/CEMAS16, CUB scans and persistent buffers; small independent Python/NumPy references for tests.

**Specification:** [AGIPC_PAPER_CONTRACT.md](AGIPC_PAPER_CONTRACT.md), [AGIPC_EXISTING_CODE_AUDIT.md](AGIPC_EXISTING_CODE_AUDIT.md), and the user's complete AGIPC 2026 task brief.

## Design decision

Considered routes:

1. **Recommended: new bounded modules in the performance branch, selectively reuse audited arithmetic.** Keeps the validated baseline and isolates new semantics. Requires fresh adapters for the branch's current CUDA buffer/converter APIs.
2. Port the entire old prototype, then repair it. Initially fewer files to write, but imports wrong defaults, fallback semantics, unsafe collectives and a different pipeline; hard to attribute changes.
3. Rebase onto current official Stiff-GIPC. Imports memory infrastructure, but invalidates the user's fixed starting point and entangles a major refactor; rejected by task scope.

No second approval for benchmarks after numerical gates is proposed. Source integration begins after review of this concrete design. No Git push or external publication is included.

## Global invariants

- Starting commit 06b34e7dc6ba28339e4adffb48f260d959a269f1; do not modify the sibling dirty worktree or existing assets/perf records.
- Keep full-resolution FEM and original scene-defined ABD, friction, CCD and line search; FP64 throughout.
- Default strain threshold 5e-5; strict greater-than protection; default post-PCG cap10; affine enrichment only for child_count>32.
- Main paper PCG relative residual1e-3, Newton `||d||inf <= 1e-3*l*dt`; adapters must not silently weaken only AGIPC.
- No performance optimization before its numerical gate. Formal tests interleave methods with >=3 measured repetitions.
- No success status for an unimplemented mode or level cap reached without completion.

## Proposed file/function boundaries

All paths in this table are relative to `code/Stiff-GIPC-c499-performance`. Function signatures are proposed interfaces; they do not yet exist. The shared types are declared in `agipc_types.cuh` and referenced below.

| File | Interface / responsibility |
|---|---|
| `StiffGIPC/agipc/agipc_types.cuh` | `CriterionStats`, `MappingStats`, `SolveStats`, `FineSystemView`, `MixedMapView`; integer counts/offsets with capacity and scalar-vs-block units |
| `StiffGIPC/agipc/agipc_workspace.cuh` | `Workspace`: persistent topology, strain history, maps, triplets, CUB scratch, vectors, events; `reserve()`/`release()` |
| `StiffGIPC/agipc/agipc_criterion.cu/.cuh` | `initialize_topology(Workspace&, const tetrahedra_obj&)`; `begin_timestep(Workspace&, const device_TetraData&)`; `update_criterion(Workspace&, const device_TetraData&, int)` -> CriterionStats |
| `StiffGIPC/agipc/agipc_mapping.cu/.cuh` | `build_mapping(Workspace&)` -> MappingStats; masked warp closure, representative ranking, hierarchy composition |
| `StiffGIPC/agipc/agipc_galerkin.cu/.cuh` | `classify_affine_nodes(Workspace&)`; `assemble_coarse(Workspace&, const FineSystemView&)`; restriction and block expansion using rest coordinates |
| `StiffGIPC/agipc/agipc_solver.cu/.cuh` | `solve_coarse(Workspace&)`; `prolongate(Workspace&)`; `post_correct(Workspace&, const FineSystemView&, int)`; `solve_adaptive(...)` -> SolveStats |
| `StiffGIPC/gipc/runtime_options.h/.cpp` | Solver enum: StiffGIPC/Core/SymHessian/Paper; post cap0 allowed; diagnostic flags; explicit unavailable-mode errors |
| `StiffGIPC/gl_main.cu` | Initialize workspace only in adaptive mode; scene and executable provenance; numerical-self-test dispatch; truthful mode JSON |
| `StiffGIPC/GIPC.cu` | History reset at timestep start; current-direction termination; full fine contacts, CCD and line search preserved |
| `StiffGIPC/linear_system/linear_system/global_linear_system.cu/.h` | Fine unique BCOO/RHS view after convert_new; baseline solve unchanged; adaptive solver dispatch; frozen H/g/map dump |
| `StiffGIPC/linear_system/utils/converter.cu` and existing converter headers | Reuse unique-triplet reduction with explicit storage convention and active counts; only extend API if required |
| `StiffGIPC/MASPreconditioner.cu/.cuh` | Narrow aligned hierarchy capacity fix, if needed for exercised shared path |
| `StiffGIPC/mlbvh.cu/.cuh` | Independent stackless traversal switch after core gates; compare candidate sets |
| `tests/agipc_criterion_tests.cu`, `agipc_mapping_tests.cu`, `agipc_galerkin_tests.cu`, `agipc_solver_tests.cu` | GPU gates with independent CPU expected results and explicit exit codes |
| `tests/agipc_reference.py` | Element tensor oracle, graph partitions, dense mixed U and SPD fixtures; never used in production |
| `CMakeLists.txt` | Explicit test targets separate from simulation source glob; Release/sm86 build and CTest registration |
| `perf_tools/run_agipc_paper.py` | Paired warm-up/interleaved experiments, schema validation, immutable hashes and statistical aggregation |

`FineSystemView` identifies `const double* rhs`, unique 3x3 block values, row/column pointers, unique count, block-node count, FEM offset and symmetric-half flag. `MixedMapView` identifies fine-to-coarse ids, coarse node bases, n3/n12, fine rest coordinates and fine/FEM domain. A wrapper must not reinterpret raw global_triplet_offset as unique count.

## Task 0: source and baseline verification

- [x] Freeze baseline and prototype commits, read main/supplement, inspect official source availability and capacity patch.
- [x] Rerun available GPU SpMV and CPU reference probes; record that executable is pre-existing.
- [ ] Configure a fresh Release build using the existing vcpkg/toolchain; record actual compiler/CUDA/executable hashes and baked asset path.
- [ ] Run existing runtime and SpMV tests from the fresh build; baseline small headless scene with CEMAS16+SRBK and frozen fine-system diagnostics.
- [ ] Verify the aligned MAS domain before any larger scene; if changed, add 38386->38400 and 1001/1024 grouped capacity cases. Preserve anchor comparison and label corrected common baseline separately.
- [ ] Change unimplemented adaptive modes from silent relabeling to explicit rejection; add CLI test requiring nonzero exit with clear reason, while stiffgipc retains previous behavior.

Use a dedicated build directory `build-agipc-reproduction`. Existing shortened-path recipe may be needed on Windows, but verify drive mapping before reuse. Initial tools found: cmake at D:/computer/cmake/bin/cmake.exe, CUDA13.0 nvcc, DL Python. Existing build artifacts are not substitutes for a new build.

## Task A: tensor criterion and history

- [ ] Write independent test fixtures for tet and triangle F, including rigid rotation, nonuniform stretch, shear, localized two-element deformation and mixed incident adjacency.
- [ ] Test threshold equality separately: norm==threshold leaves edge collapsible; larger protects. Test NaN as explicit failure, never a collapsible edge.
- [ ] Implement static topology and persistent strain history. At timestep start store G at initial positions, then compare each Newton state's G to the previous visited Newton state. Record initialization convention.
- [ ] Produce strain-only counters and constrained-edge counters; min/mean/max must exclude no-history sentinel values.
- [ ] Run CPU/GPU Gate A: deterministic tags and tensor increments, finite data, monotonic protection under 1e-6/5e-5/5e-4. No coarse solver before pass.

Concrete reference arithmetic:

```python
F = Ds @ np.linalg.inv(Dm)
G = 0.5 * (F.T @ F - np.eye(F.shape[1]))
increment = np.linalg.norm(G - previous_G, ord="fro")
tag[e] = int(all(increments[t] <= threshold for t in incident[e]))
```

Test rest tet `(0,0,0),(1,0,0),(0,1,0),(0,0,1)`; triangle uses first three points. A uniform x stretch `s` gives norm `abs((s*s-1)/2)` against rest. A rotated F leaves F^T F unchanged.

## Task B: hash mapping

- [ ] Independent CPU BFS/union-find on allowed edges serves only as the expected partition; production remains CUDA.
- [ ] Implement one group first: self bit, neighbor bits, transitive closure, first-bit representative, rank among elected representatives. Use shared immutable hashes plus explicit synchronization or a uniform shuffle schedule.
- [ ] Cover isolated components within a full warp, chain length32, groups1/7/16/31/32 and partial last group. Example allowed edges `(0,2),(2,4),(1,3)` must produce sets `{0,2,4},{1,3}`.
- [x] Add exclusive scans, compose mappings, persistent child counts and next-level edges; restart identity every Newton iteration.
- [x] Record per-level counts and fixed-point/cap reason; accept stable cross-group partitions, retain unresolved-link diagnostics, use a 16-level default, and cover a disconnected boundary-edge fixed point.
- [x] Run Gate B with valid indices, exactly one owner, nonempty coarse nodes and child sum equality. The focused GPU gate includes hierarchy, determinism, tail, protected-edge and stable cross-group cases.

## Task C: translational Galerkin

- [ ] Generate SPD H via `Q.T@Q+I`, independent dense U, and randomized nonmonotonic maps; include collapsed off-diagonal nonsymmetric blocks.
- [ ] Restrict RHS and transform unique fine BCOO. Preserve full-vs-half storage semantics; canonicalize with transpose. Use the existing converter's reduction primitives.
- [ ] CPU dense checks `norm(Hgpu-U@H@U.T)/norm(U@H@U.T)<=1e-12`, same for RHS, plus symmetry, Cholesky and positive quadratic forms.
- [ ] Keep PCG disabled until Gate C passes; no dense matrix in production.

## Task D: mixed affine system

- [x] Add child_count33 versus32 boundary tests. Sort nodes by 3/12 DoF class and build coarse block bases, with rank-aware 4/3/2/1-column layouts for volumetric/planar/linear/point aggregates.
- [x] Implement rest basis `[1,X,Y,Z]`, variable-width mixed block expansion and generalized q-column indexing. Include ABD prefix and cross terms without aggregating ABD bodies.
- [x] Compare all four full-rank transformed block shapes to dense U; pass the adjoint test and planar/collinear rank gates. The reduced Figure 12 mixed scene now produces 7/7 valid coarse diagonal blocks and adopts an AGIPC direction.
- [ ] Run rotating free-fall object with full-space/3-only/mixed routes; save equal-frame states and angular/dynamic comparisons. Gate D fails if mixed indexing or physical restoration is invalid.

## Task E/G: actual solver dispatch and termination

- [x] Implement coarse PCG with explicit norm tolerance1e-3, signed curvature checks and diagnosed nonconvergence. The validated intermediate preconditioner is block-Jacobi; paper MAS semantics remain later work.
- [x] Prolongate using the exact adjoint; post-PCG on full fine H starts from that solution, uses a block diagonal preconditioner and honors the runtime cap including cap0.
- [x] Test a nonzero initial correction state on a frozen SPD H/g and a real cube system; record projected residual reduction, finite direction, signed curvature and the residual trace.
- [x] Add experimental `--solver agipc-core` dispatch after A–D/E pass. Capture attempted/adopted/fallback counters plus aggregate typed reasons, and retain the original solver as the guarded fallback.
- [x] Apply the current-direction Newton criterion with dt factor after the AGIPC-Core solve; retain the baseline's previous-direction behavior as frozen provenance and record the current norm and threshold per Newton step.
- [x] Add a baseline-only `--newton-stop paper-current` ablation and per-Newton Galerkin/CCD/line-search diagnostics. On the 27-frame 16K contact gate, the matched fine baseline needed 177 applied updates versus AGIPC's 94; neither route performed energy or intersection backtracks.
- [x] Freeze representative mild/moderate/severe post-residual fallbacks and independently compare prolonged, corrected and fine directions. The first divergent stage is post-correction selection; restore the prolongated direction when exact residual grows beyond `1e-6` relative tolerance.
- [x] Analyze coarse-PCG distributions on both completed 27-frame stage-timed runs. Systems above 1,024 coarse blocks consume 65%--81% of coarse-solve time, and iteration count has greater than 0.99 correlation with solve time; large-system preconditioning is the next optimization target.
- [x] Implement explicit diagnostic freezing for medium/large converged and iteration-cap coarse systems. The 27-frame guard/capture diagnostic passes with 3 prolongation restores, finite states and zero ground penetration; real 1415/741-block converged snapshots pass CPU and GPU replay. Iteration-cap capture and a run without explicit diagnostics remain pending.
- [x] Add standalone frozen coarse GPU Jacobi/MAS32 replay. Real 7-block and synthetic 257-block cases pass; real contact matrices use 121/78 Jacobi versus 28/34 MAS32 iterations. Direct CPU reference confirms near-optimal quadratic decrease, but saved Jacobi directions themselves have about 4.5% Euclidean error. Keep default dispatch unchanged until MAS vector quality, exceptional cases and setup/application costs are validated.
- [x] Export original-block GPU solutions and independently compare them to sparse direct references: MAS direction errors 4.15%/1.15% versus Jacobi 4.41%/4.30% on real 1415/741-block matrices. Add separate setup/PCG timing and a cap1 negative gate (exit3, both report iteration_cap). Controlled timing, real exceptional inputs and runtime integration remain pending.
- [x] Add explicit runtime coarse MAS32 selection with guarded local diagnostics, signed rMz and exact coarse-residual checks. The same-build 27-frame MAS/Jacobi pair is finite and ground-penetration-free, with83/83 versus77/85 adoptions and0.0001947 RMS terminal difference. Default Jacobi passes without capture diagnostics. Full CPU MAS diagnostics cost1431.870ms and dominate the experimental coarse path; optimize their implementation before performance claims.
- [ ] Reduce runtime local-check cost while retaining validated rejection behavior; compare against full CPU diagnostics and preserve true-residual/fine-adoption gates. Cache/reuse and GPU adjacency are separate later ablations.
- [ ] Strict diagnostics compute full gradient RMS and full-space displacement. Per-stage CUDA timing is implemented and has isolated coarse-PCG and unused fine-preconditioner costs; low-perturbation benchmark timing remains open.

## Task F: fine contact preservation

- [ ] For no-contact, ground, self-contact and contact-transition fixtures, freeze complete fine H/g and compare the mapped contact contributions to dense references.
- [ ] Preserve all fine barrier/friction contributions before restriction. Execute original fine CCD and line search after solve.
- [ ] Pair baseline/adaptive runs with equal scene parameters; verify finite vertices, minimum separation/ground penetration, final-state error and accepted line-search steps. Record failed cases, never suppress them in summaries.
- [x] Run the first paired ground-contact transition with the paper current-direction threshold: 30-frame cube, zero penetration, minimum-y delta `1.30e-6`, 42 AGIPC applied steps versus 66 baseline steps, and 72/72 AGIPC linear candidate adoptions. Self-contact and frozen contact-matrix checks remain open.
- [x] Run a reduced mixed ABD/FEM cloth gate: one ABD bunny plus 289-node planar cloth, zero penetration, minimum-y delta `2.77e-10`, 7 independent coarse blocks, and one adopted physical direction. The final near-zero-residual check conservatively used the baseline fallback.
- [x] Run a dedicated reduced Figure 15 self-contact gate with a fixed ABD sphere: both 35-frame routes remained finite with zero penetration; direct 289-vertex RMS final-state error was `0.03314` (`5.78%` of baseline RMS displacement). AGIPC used 21/879 active DoF, but was `1.90x` slower in the single reduced-scale pair.

## Task H: metrics and paper pipeline

- [ ] Extend JSON with all fields from task brief §16, keep old fields, FEM and whole-system DoF distinct. Contact counts, FEM/ABD terminal summaries, optional final-state CSV, and stage timing are present; the remaining paper fields are still required.
- [ ] Add persistent CUDA events and NVTX labels; remove only adaptive-path allocations/sync after correctness evidence.
- [ ] Implement independent symmetric fine FEM/contact/friction assembly; compare complete matrices before allowing `agipc-symhessian`.
- [ ] Implement `--collision-traversal stack|stackless`; compare sorted candidate and contact sets on equal states before enabling `agipc-paper`.
- [ ] Keep each engineering optimization in a separate ablation and commit.

## Experiments and final acceptance

- [ ] Verify exact meshes and physical durations; if unavailable, label paper-inspired/reduced workloads and ASSET_BLOCKED for exact reproduction. Do not infer asset identity from scene names.
- [ ] Fig13 first: same ball, E=1e4/1e5/1e6/1e7, dt=.01. Establish active-DoF behavior and stage costs before comparing paper speedup.
- [ ] Fig14: same dragon E=3e5, dt=.005/.01/.02/.04, physical duration1.5s, document fractional-frame policy. Lower dt should yield lower active ratios.
- [x] Fig15 reduced preflight: 289-node cloth on a fixed ABD sphere at E=1e6, dt=.01; active ratio `0.02389`, direct final-state error recorded, negative single-pair runtime retained.
- [x] Fig15 capacity preflight: existing 16,641-node cloth on the fixed ABD sphere at E=1e6, dt=.01; both CEMAS16 routes pass one no-contact frame at 49,935 fine DoF, with 21 final AGIPC coarse DoF. This is not an exact paper asset.
- [x] Fig15 larger contact gate: both routes pass 35 frames with finite states and zero penetration; paired RMS terminal error is `0.01995` (`3.44%` of baseline displacement). The original AGIPC run is `2.98x` slower. A focused 27-frame rerun validates stable partitions, and a matched paper-current baseline shows that strict termination itself creates a 101-update contact-frame tail. Stage timing identifies coarse PCG as the largest adaptive cost and finds `331.314 ms` of unused fine-preconditioner work on accepted candidates; that assembly is now deferred to fallback.
- [ ] Fig15 exact assets: cloth on ABD sphere E=1e6, dt=.01, 10K/51K/92K first; only larger runs after measured capacity headroom.
- [ ] Warm up and interleave S/A/A/S until each method has >=3 measured trials. Hash every executable/mesh; store immutable JSON in perf_history and CSV/JSON summaries.
- [ ] Complete AGIPC_IMPLEMENTATION_MAPPING.md, AGIPC_NUMERICAL_VALIDATION.md, AGIPC_PERFORMANCE_ANALYSIS.md, AGIPC_FAILURES_AND_FIXES.md and AGIPC_PAPER_COMPARISON.md with actual results. Separate numerical/trend/quantitative levels; retain negative results.

Each source-stage commit requires a fresh Release build, relevant CPU/GPU tests and numerical self-test. Proposed sequence follows task brief: audit, criterion, map, coarse3, affine, postcg, contact, metrics, hessian, bvh, exp13, exp14, exp15, final. The current audit documents are uncommitted because that commit gate has not yet been run.

## First reviewable implementation increment

Begin with Task0+A only: truthful runtime dispatch, verified build/capacity prerequisites, tensor criterion/history and independent GPU gate. Mapping, coarse PCG and performance work remain gated behind their predecessors. This isolates the first falsifiable claim and prevents importing a running but mathematically different prototype.

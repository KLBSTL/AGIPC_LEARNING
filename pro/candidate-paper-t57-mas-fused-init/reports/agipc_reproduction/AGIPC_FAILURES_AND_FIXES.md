# AGIPC Failures and Fixes

## Empty edge set launched a zero-block kernel

The 7-isolated-node mapping fixture failed inside Thrust with `cudaErrorInvalidDevice`. The pending error came from launching the edge kernels with a zero-sized grid. Both edge launches now have explicit empty-input guards, followed by a CUDA launch check. The isolated-tail fixture passes.

## Rank-deficient affine test fixture

The first mixed test assigned only two fine nodes to a 12-DoF affine aggregate. Galerkin matrix and RHS errors were at machine precision, but Cholesky correctly reported a singular coarse system. The positive fixture now uses four noncoplanar child points. Separate planar and collinear fixtures retain rank 3 and rank 2 evidence so singular geometry is diagnosed rather than hidden.

## Planar cloth produced a singular runtime affine aggregate

The first reduced mixed ABD/FEM scene mapped all 289 cloth vertices to one fixed four-column affine basis `[1,x,y,z]`. Because the rest cloth lies exactly on `y=0`, one coarse diagonal block was zero: only 7 of 8 diagonal blocks were invertible, `invalid_entries` became 1, and both candidates fell back. Mapping now computes a pivoted rank of centered rest coordinates and stores a compact basis mask per aggregate. The planar cloth uses `[1,x,z]`, producing 7 total mixed coarse blocks with all 7 diagonals invertible and `invalid_entries=0`; volumetric aggregates retain all four columns. Planar and line fixtures are part of the mapping self-test.

## Symmetric-half expansion of affine diagonal blocks

Naively emitting and canonicalizing all 16 sub-blocks of an affine fine diagonal doubles off-diagonal coarse blocks. Fine diagonal blocks now emit only the upper triangular affine sub-blocks. Off-diagonal fine blocks still emit the full product, and collapsed coarse diagonals explicitly add the transposed half. Dense `P^T A P` comparison is below `1e-16` relative error.

## New CUDA source was absent from the first parallel link

After CMake detected `agipc_galerkin.cu`, the first parallel build reached the link before its object was available. A sequential rebuild compiled the new unit and linked successfully. Subsequent incremental builds are stable.

## Absolute post-PCG breakdown threshold rejected a tiny valid system

The first real cube run reduced the fine residual to `0.07735` of its prolongated value, then stopped because `r^T M^-1 r` was below the fixed `1e-30` guard. The right-hand side itself was only about `5.8e-14`, so the fixed guard was not scale invariant. The post-PCG recurrence now checks finite positive preconditioned residuals instead. The repeated run reached the relative residual tolerance in 10 iterations, reduced the residual by `2.3749251015407196e-5`, and all recorded curvatures were finite and positive.

## AGIPC Newton gate initially inherited the scene's 1e-2 threshold

The first current-direction integration reused `Newton_solver_threshold`, which the small headless scene sets to `1e-2`. The paper contract requires `1e-3 * bbox_diagonal * dt`; the 30-frame probe consequently stopped after 38 Newton iterations and was rejected as evidence. AGIPC-Core now uses an explicit `1e-3` constant while StiffGIPC preserves its scene threshold and previous-direction behavior.

## Figure 12 free fall did not exercise mixed contact

A 30-frame reduced Figure 12 run completed with finite vertices, but all 49,600 edge samples remained collapsible and the protected-edge count stayed zero. The ABD bunny and FEM cloth accelerated together, so extending this scene did not create the intended cloth-on-rigid contact. This run is retained in `perf_diag/agipc_core_fig12_hybrid_contact30.json` as negative scene evidence and is not accepted as Gate F contact validation. A dedicated reduced Figure 15 setup needs a stationary ABD obstacle and an independently falling cloth.

## Reduced Figure 15 is a correctness gate, not a speedup result

The replacement fixture fixes an ABD sphere and drops a 289-node cloth onto it. It exercises changing strain tags and thousands of self-collision pair samples, and both solvers finish 35 frames without finite-value or penetration failures. Direct final-state export shows `0.03314` RMS paired-vertex error (`5.78%` of baseline RMS displacement), which is compatible with the deliberately different AGIPC and StiffGIPC Newton stopping semantics but rules out claiming identical trajectories. The measured AGIPC-Core run was `1.90x` slower, with 204 versus 141 Newton iterations and 4,908 versus 1,890 linear iterations. The result is retained as reduced-scale overhead evidence; paper-scale speedup remains untested.

## Partial reduction blocks skipped a required barrier

The movement-norm maximum kernels returned out-of-range lanes before a block-wide `__syncthreads()` and used a full-warp shuffle mask when only the low lanes of warp zero remained active. Meshes whose vertex counts are not multiples of the 256-thread block size therefore relied on undefined CUDA synchronization behavior. Out-of-range lanes now contribute the neutral maximum value, every block lane reaches the barrier, and the second-level warp reduction uses its exact active mask. The 18,288-vertex preflight exercises a 112-lane final block and completes in both solver modes.

## Local-code-page paths failed strict JSON output

The first 16K attempts completed their GPU frame but exited with code 1 while serializing metrics. A Chinese `--cloth-mesh` path arrived through `char** argv` as non-UTF-8 local-code-page bytes, and nlohmann JSON's strict dump rejected the string. Metrics output now uses the replacement error handler, which prevents the run from being lost. Because replacement cannot reconstruct the original Unicode characters, recorded benchmark paths should use the ASCII `S:`/`T:` aliases; hashes remain the authoritative asset identity.

## Stable cross-group partitions were incorrectly rejected

The first 16,641-node, 35-frame contact run treated every remaining collapsible edge as proof that the map was incomplete. This rejected 334 stable warp-hash fixed points and 49 eight-level-cap states, causing 383 mapping-stage fallbacks. The fixed points were not corrupt maps: their coarse counts were always at least 33, and the unchanged final level showed that all remaining collapsible edges crossed 32-node group boundaries. The supplement defines only in-group hash closure followed by recursive compaction; it does not require those cross-group links to collapse before the ownership partition can define a Galerkin projection.

Mapping completion now distinguishes a valid stable partition from an artificial level cap. A fixed point is eligible for Galerkin assembly, while `remaining_collapsible_edges` and `globally_resolved` preserve the quality distinction. The default hierarchy budget increased from 8 to 16 levels so ordinary recursion is less likely to stop at the cap. A 64-node boundary-edge GPU fixture verifies the stable identity partition deterministically. In the focused 27-frame 16K contact rerun, all 114 maps were eligible: 78 globally resolved and 36 stable cross-group partitions. All 36 stable maps reached the candidate path; the only 26 fallbacks were typed `post_residual_not_reduced`. The run remained finite with zero penetration and fixed-ABD displacement below `5.56e-17`.

## Post-correction residual rejection selected the wrong fallback

Three frozen `post_residual_not_reduced` cases show that the rejected directions were within `0.55` degrees and `0.96%` relative norm difference of the fine fallback, while their exact residuals were 14--34 times smaller. One zero-iteration case was rejected solely because repeated SpMV reductions changed the residual ratio by `4.53e-8`, above the former `1e-12` growth gate. In the other two cases, ten post iterations enlarged the prolonged residual by `1.51x` and `2.30x`; the prolonged directions still had only `4.58%` and `2.91%` of the fine fallback residual. The post-correction path now keeps its raw diagnostics but restores the prolonged solution when exact residual growth exceeds `1e-6`. Nonfinite or abnormal stops still fall back. Full evidence and the independent binary reader are in `AGIPC_FALLBACK_DIRECTION_ANALYSIS.md`.

## Incremental CUDA build retained an incompatible class layout

The first implementation of the baseline stopping-rule ablation added a Boolean data member to `GIPC`. The incremental CUDA build did not rebuild every translation unit that instantiated the class, so linked objects disagreed on its layout. A forced baseline run then failed with CUDA error 700, and a subsequent one-frame legacy run produced invalid numerical behavior. Those runs are rejected as algorithm evidence.

The option now uses file-local state in `GIPC.cu`; `GIPC.cuh` adds only a setter declaration and does not change object layout. The build target was cleaned and rebuilt serially so every CUDA object used the same headers. The rebuilt executable passed a one-frame legacy-stop health check with two applied Newton updates, 32 PCG iterations, finite vertices, zero penetration, and fixed-ABD displacement below `5.56e-17`, followed by the successful 27-frame paper-current baseline. For future changes to widely included CUDA class definitions, a clean rebuild is required before interpreting runtime failures.

## Frozen coarse replay omitted matrix lifecycle initialization

The first standalone coarse replay failed at destruction with `global_matrix.h` reporting `cudaFree invalid argument`. The new adapter had constructed `GIPCTripletMatrix` without calling its required `init_var()`. Calling it immediately after construction initializes the raw pointers that the destructor owns. The final incremental Release build, real seven-block replay, synthetic 257-block replay and core GPU self-test all pass. The initial failure is rejected as MAS numerical evidence; the shared matrix class and default solver path were not changed.

Adding the new CUDA source also triggered slow CMake `FindCUDAToolkit` searches for optional static libraries. Setting `CMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=OFF` in the existing local build cache completed reconfiguration in about 27 seconds while retaining already located toolchain paths. The repository build configuration is unchanged; exact commands are in `AGIPC_COARSE_MAS32_REPLAY.md`.

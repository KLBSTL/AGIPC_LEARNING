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

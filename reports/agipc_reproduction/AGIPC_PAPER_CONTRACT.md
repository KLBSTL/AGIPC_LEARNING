# AGIPC paper contract

Audit date: 2026-09-10. Status: source contract established; GPU reproduction gates are not passed.

## Frozen sources and scope

- Main paper: [arXiv 2605.04773v1](https://arxiv.org/html/2605.04773v1), local `实验十三/2605.04773v1.pdf`, SHA256 `602573B0AA50B61E56D6EFFEFBA7A14A96F1A0AE89D3EAFB21C3534ACC734218`.
- Read all 18 PDF pages: main pp. 1–11 and supplement pp. 1–7 (physical pp. 12–18). Existing page-marked extraction: `实验十三/RePro/paper/agipc.txt`. Visually checked supplemental Algorithms 2/4 and mixed-block indexing against physical pp. 13/15.
- Development starting tree: `code/Stiff-GIPC-c499-performance`, branch `stiffgipc/c499-performance-reproduction`, HEAD `06b34e7dc6ba28339e4adffb48f260d959a269f1`. Retain its CEMAS16/SRBK/ABD/full fine IPC infrastructure.
- Old prototype is a DIFFERENT tree: `code/Stiff-GIPC`, branch `agipc/performance-optimization`, HEAD `e5903cd2c94d12c96488419cf16ef6b86aea5589`.
- Official [Adaptive-GIPC](https://github.com/KemengHuang/Adaptive-GIPC) unauthenticated GitHub repository API returned HTTP 404 on this date. This establishes present access failure, not whether a private repository exists. No official AGIPC commit can currently be frozen.
- Official Stiff-GIPC fix inspected: [a5bf7e548529181f0222f773e6fe330e3b8d881c](https://github.com/KemengHuang/Stiff-GIPC/commit/a5bf7e548529181f0222f773e6fe330e3b8d881c). Backport only its capacity/bounds semantics if needed, not its enclosing refactor.

## Algorithm 1 and sign convention

Main p. 3, Algorithm 1 and Eqs. (1)–(2): minimize the full incremental potential, including inertia, elasticity, contact barrier and friction. At each Newton iteration:

1. Save current energy and positions.
2. Assemble complete fine gradient and SPD proxy Hessian.
3. Recompute edge tags and fine-to-coarse mapping.
4. Construct `g_c = U g_f`, `H_c = U H_f U^T`.
5. Solve `H_c d_c = -g_c`.
6. Prolongate and post-correct on the complete fine system.
7. Update BVH, obtain CCD feasible step, backtrack energy, then test convergence using the corrected direction.

Repository adapters may use `b = -g`; this must be explicit. Restricting `b` does not require a second negation. Actual sign must be verified at subsystem assembly/retrieval, rather than inferred from variable names.

## A: per-element criterion

Main pp. 4–5, Eq. (3): `G = (F^T F - I)/2`, `DeltaG = G_current - G_previous_Newton`, criterion `||DeltaG||_F`. Tets use 3x3 F; shells use their 3x2 deformation gradient and 2x2 G. Count off-diagonal squares twice when storing only symmetric entries. No division by dt, edge length proxy, material weighting, or relative normalization.

Initialize edge tag 1; set 0 if ANY incident element has norm strictly greater than threshold. Static edge-element adjacency is computed once. Default threshold `5e-5`; Figure 10 specifically uses `5e-4`. Recompute from original fine topology each Newton iteration.

The paper does not specify first-iteration strain history initialization. Record this as an implementation convention, not a paper constant: initialize history at the current timestep's initial positions; the first comparison with those positions is zero. Do not silently carry a stale pre-line-search state from the previous timestep. Test first-iteration, accepted-step and timestep-reset transitions independently. Boundary constraints remain authoritative; protection of constrained endpoints must be recorded separately from strain-triggered protection.

Gate A: CPU/GPU agreement, deterministic tags, finite values, zero increment, rigid motion invariance, localized protection, threshold equality, monotonic thresholds, all incident elements, timestep reset; record min/mean/max norm and protection counts.

## B: recursive parallel hashing

Supplement pp. 1–3, Algorithms 1–2: connectivity/METIS ordering; contiguous groups of at most 32 nodes; own lane bit plus collapsible neighbors in the same group; iterative OR closure; first-set-bit representative; component election and exclusive scan to global coarse ids. Compose level mappings back to fine nodes; carry remaining inter-group connectivity to the next level. Restart from fine nodes on every Newton iteration. No remeshing and no CPU connected-component implementation in production.

Important pseudocode clarifications, to be tested rather than copied literally:

- Algorithm 2 lines 28–29 count elected representatives before `lane_id`, but nonrepresentative members must use the FIRST SET BIT representative's rank. Literal lane rank maps a connected pair to different ids. Follow the prose's component identity.
- Avoid `1 << 32`; use unsigned masks and `0xffffffffu` for a full warp.
- Shared-memory phases require synchronization. For shuffles, all named active lanes must execute each collective; a variable-length per-lane neighbor loop does not establish that contract.
- Supplement says recurse to a minimal representation but gives no numerical level cap or fallback ordering when a fixed partition stalls. Stop at a fixed point under the documented ordering and retain unresolved inter-group collapsible links as a coarsening-quality diagnostic. The resulting ownership partition is still a valid Galerkin map. A cap reached while another level still reduces the node count is incomplete.
- Protected edges cannot be traversed; their endpoints can still connect through a different wholly collapsible path. Do not add a global endpoint-separation rule absent from the paper.

Persistent arrays: edge tags, connectivity hashes, group counts, local indices, fine map, child counts, level offsets/sizes. Allocate for padded group/node domains; bounds-check all composed maps. Begin with 128/256 threads on sm86.

Gate B: isolated, complete, disconnected and indirect graphs; protected edges; group sizes 1/7/16/31/32; partial warps; hierarchy; independent CPU component oracle modulo numbering. Each fine node maps exactly once, indices valid, no empty aggregate, child sum equals fine count.

## C/D: Galerkin and mixed affine embedding

Main pp. 4–5, Eq. (4); supplement §§2–3, Algorithms 3–4. For each unique fine BCOO block `(i,j,B)`, form `A_i B A_j^T` with `A_i = I3` or `[1,X,Y,Z]^T tensor I3` (12x3), using rest-pose coordinates. Sum all transformed contributions. The main equation's abbreviated single-f notation must not discard off-diagonal i,j terms; supplemental Algorithm 4 makes this explicit.

An aggregate with **more than 32 fine nodes** gets 12 DoF; otherwise 3 DoF. Sort/reindex coarse nodes so 3-DoF nodes precede affine nodes. For coarse node c, its base 3x3-block index is `c` if c<n3, else `n3+4*(c-n3)`. For a transformed block with q column blocks (q=1 or 4), flattened sub-block k has offsets `floor(k/q)` and `k%q`. Supplement Eq. (2)'s division by 4 directly fits a four-column block; a 12x3 block instead requires q=1. Test all four combinations.

Block multiplicities: 1, 4, 4, 16. Count, exclusive-scan, emit uniform 3x3 triplets, use 64-bit `(row<<32)|col` sort and duplicate reduction. Do not allocate dense coarse matrices or 12 DoF for every node. Use an inverse permutation when translating old coarse ids to sorted ids.

Fine matrices stored as one symmetric half need special treatment: swapping mapped row/column transposes B; an off-diagonal fine block collapsed onto one coarse diagonal contributes both B and B^T. With affine expansion apply the equivalent identity to transformed blocks, avoiding duplicate diagonal contributions. Full-storage and half-storage routes must be explicitly distinguished.

Mixed-scene ABD coordinates remain scene-defined. Preserve their identity block region and all ABD/FEM cross terms; do not label ABD blocks as adaptive affine FEM nodes. Report FEM fine DoF separately from whole-system DoF.

Gate C: dense CPU `U H U^T` and `U g`, relative error <=1e-12, symmetry, finite values, SPD for full-row-rank U. Gate D: all four mixed block types, permutations and scalar indexing, rotating/free-fall comparison. More than 32 points is not by itself proof of rank 4 of homogeneous coordinates: planar/collinear affine aggregates can be rank deficient. Detect and diagnose; do not silently regularize or assert SPD solely because H_f is SPD.

## E/F/G: solve, post-correction, contact, termination

Main §4.3 and supplement pp. 5–6: coarse PCG; `d_f = U^T d_c`; fine post-PCG starts from d_f, max 10 by default, can stop early on residual tolerance. Supplement explicitly identifies a diagonal preconditioner for post-correction. Main performance experiments use MAS for both frameworks' main solve; a block-Jacobi-only coarse implementation is an intermediate ablation, not the final MAS comparison.

A forced full-space residual acceptance threshold followed by a full StiffGIPC solve is NOT the paper's post-correction algorithm. Numerical-invalidity safeguards may exist, but must record failure/fallback and cannot count as successful AGIPC adoption. Negative/nonfinite p^T H p must stop and freeze evidence.

Keep full fine contact and friction before restriction; keep full fine CCD/line search after prolongation. Validate no contact, ground contact, self contact, changing contact sets and mixed ABD/FEM coupling. Coarsening preserves barrier evaluation continuity; it does not alone prove that a chosen direction is feasible without CCD.

**Newton units:** Main Algorithm 1 line 21 and supplement convergence analysis require `||d||_infinity / dt <= epsilon_d`, with `epsilon_d=1e-3 * scene_bbox_diagonal` in m/s. Equivalently compare displacement with `1e-3*l*dt`. StiffGIPC uses the full-space solve direction; AGIPC uses prolonged, post-corrected direction. Do not omit dt. Do not substitute line-search-scaled alpha*d or the prior iteration's direction without labeling the deviation.

PCG relative residual-norm tolerance `1e-3`, double precision in both modes. When comparing squared Euclidean residuals the multiplier is `1e-6`, not `1e-3`. Verify whether existing baseline uses Euclidean or preconditioned residual before assigning a paper label. Existing validated baseline remains separately identifiable; any common paper-tolerance adapter must apply identically and be disclosed.

Strict full-space diagnostics report gradient RMS and direction norm, separate from paper-mode acceptance. Full-space solves performed solely for diagnostics must not enter timed core speedup.

## H: separate mechanisms and timing

`stiffgipc`: unchanged baseline. `agipc-core`: fine pipeline unchanged, adaptive solver only. `agipc-symhessian`: additionally symmetric FEM/contact/friction assembly. `agipc-paper`: additionally stackless BVH and only documented paper optimizations. An unimplemented mode must fail explicitly, never relabel StiffGIPC output.

Main §5.2 confirms total speedups include upper-triangular assembly and stackless BVH. Existing symmetric global SpMV alone does not establish the upstream assembly optimization. Stackless implementation details are not supplied in the paper; validate its candidate/contact sets independently.

CUDA events/NVTX cover fine build, tags, mapping, gradient restriction, Hessian remap/reduce, coarse preconditioner/PCG, prolongation, post-PCG, CCD/BVH, line search, misc, total. Avoid global synchronization introduced solely for timing. Existing baseline synchronizations remain part of its frozen provenance; do not attribute removing them to coarsening alone.

## Experiment contract

Paper hardware: i9-14900K, 64 GB RAM, RTX4090 24 GB. Local current query: RTX3070 Laptop 8192 MiB, driver 581.57. Warm up, interleave S/A, >=3 measured repetitions per method, equal frames/physical duration and mesh/physics, Release/FP64. Record mean/median/std/CV and immutable source/executable/mesh hashes. Final speedups use medians.

| Experiment | E | dt | Paper vertices | Stiff seconds | AGIPC seconds | Speedup | Active ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fig13 ball | 1e4 | .01 | 62K | 695.586 | 372.919 | 1.87 | .23 |
| Fig13 ball | 1e5 | .01 | 62K | 560.241 | 233.338 | 2.40 | .24 |
| Fig13 ball | 1e6 | .01 | 62K | 356.813 | 118.589 | 3.01 | .26 |
| Fig13 ball | 1e7 | .01 | 62K | 489.948 | 148.417 | 3.30 | .15 |
| Fig14 dragon | 3e5 | .005 | 50K | 397.349 | 163.609 | 2.43 | .12 |
| Fig14 dragon | 3e5 | .01 | 50K | 85.7406 | 41.5055 | 2.07 | .32 |
| Fig14 dragon | 3e5 | .02 | 50K | 57.8737 | 26.7386 | 2.16 | .48 |
| Fig14 dragon | 3e5 | .04 | 50K | 56.1352 | 27.1783 | 2.07 | .68 |
| Fig15 cloth/ABD | 1e6 | .01 | 10K | 43.3452 | 24.4691 | 1.77 | .35 |
| Fig15 cloth/ABD | 1e6 | .01 | 51K | 190.082 | 93.4818 | 2.03 | .30 |
| Fig15 cloth/ABD | 1e6 | .01 | 92K | 430.575 | 176.535 | 2.44 | .32 |
| Fig15 cloth/ABD | 1e6 | .01 | 133K | 675.641 | 296.692 | 2.28 | .27 |
| Fig15 cloth/ABD | 1e6 | .01 | 174K | 949.067 | 350.928 | 2.70 | .30 |

Fig14 physical duration is 1.5 seconds; rounding/frame policy for .04 must be explicit. Fig13 prose says 61K, main table 62K, supplement reports 185007 fine DoF (61669 vertices). Preserve this rounding difference. Active ratios are per-Newton average FEM DoF ratios, not vertex ratios once affine nodes exist. Ratios in Fig13 need not decrease monotonically for all four stiffnesses.

Fig12(a) uses balls E=1e7/5e6/1e6, ~185K vertices, speedup 3.15; Fig12(b) tree E=1e8/toys E=1e7, ribbon triangles and ABD snowflakes, ~127K, speedup 2.24. Both relative d_hat=3e-4, active ratio ~.15. Exact assets/initial conditions not yet established locally: label asset verification pending, never relabel a reconstructed mesh as exact.

Fig10 mat: 45K, E=1e7, dt=.01, three full turns, d_hat=3e-4, strain threshold=5e-4. Fair diagonal comparison 10284.5/3579.15 seconds=2.87x. Unstable MAS-based 8x is not a target. Post-PCG ablation caps 0/1/5/10/50/100; 0 must be accepted as an explicit ablation.

## Acceptance and stop rules

Numerical reproduction, trend reproduction, quantitative reproduction are separate. No speed claim until relevant A–G gates pass. Stop for invalid indices/nonfinite data, Galerkin relative error >1e-12, unexplained negative curvature, lost contacts, catastrophic post-residual growth, unexplained >2x Newton growth or >5x stage regression. Freeze H/g/U/map, record the first divergent stage. 3070 targets and ±20% paper speedup are goals, not permissions to weaken accuracy or baseline.

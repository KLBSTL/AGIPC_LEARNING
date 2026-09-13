# AGIPC Numerical Validation

Date: 2026-09-12. Build: Release, CUDA 13.0, `sm_86`, Visual Studio 2022. The executable supports the experimental `agipc-core` route; symmetric-Hessian and paper-BVH modes remain unavailable.

## Focused GPU gate

Command:

```text
build-agipc-reproduction/Release/gipc.exe --agipc-self-test
```

Result: exit 0.

| Gate | Evidence |
|---|---|
| Criterion | 10 cases; max absolute tensor error `4.098284211995207e-17`; history, strict equality, monotonic thresholds, boundary and NaN protection passed |
| Mapping | 8 cases; indirect connectivity, protected edge, 7-node tail, 32-versus-33 affine boundary, hierarchy, stable cross-group partition, repeat determinism and rank-aware planar/linear bases passed |
| Mixed Galerkin | relative matrix error `8.747661209037898e-17`; RHS error `0`; symmetry error `9.308113199029161e-18`; SPD Cholesky passed |
| Mixed indexing | all `1x1/1x4/4x1/4x4` shapes passed; adjoint error `5.551115123125783e-17` |
| Geometry rank fixtures | planar affine basis rank `3`; collinear rank `2`; the runtime basis now removes dependent coordinate columns instead of constructing a singular 12-DoF aggregate |
| Coarse PCG | converged; projected residual ratio `6.689865418268851e-4` under the paper `1e-3` tolerance; fine residual ratio after prolongation `0.29712101078523767`; `rhs_dot_direction=4.335198384840098` |
| Fine post-PCG | starts from the nonzero prolonged direction; 9/10 iterations; residual reduction ratio `0.0023250257841851774`; relative error to the dense fine solution improved from `0.8234604603462874` to `0.0006660592178736782` |
| Adoption/fallback | exact device copy error `0`; dimension mismatch selected the fallback; incomplete mappings are withheld from the candidate path |

## One-frame real-system gate

Command used the existing short drive because the CUDA/CMake cache was configured at `S:`:

```text
S:/build-agipc-reproduction/Release/gipc.exe --scene stiff-bunny-drop --tet-mesh S:/Assets/sorted_mesh/cube_sorted.16.msh --frames 1 --headless --agipc-diagnostics --agipc-fine-correction-iterations 10 --metrics-path S:/perf_diag/agipc_post_pcg_cube.json
```

Result: exit 0. Eight FEM nodes mapped to one translational coarse node; `child_sum=8`, remaining collapsible edges `0`, 26 fine unique blocks reduced to one coarse block, and invalid entries `0`. Coarse PCG converged in one iteration. Fine post-PCG used the nonzero prolongated direction, reached the `1e-3` relative residual tolerance on iteration 10, reduced the residual by a factor of `2.3749251015407196e-5`, and recorded ten finite positive curvatures.

Compared with `perf_diag/stiffgipc_cube_baseline.json`: `minimum_y` delta `0`, identical Newton count `2`, identical fine PCG total `5`, identical penetration `0`, and finite vertices in both runs. This confirms that the current shadow path does not modify the production direction.

These focused checks establish criterion, mapping, mixed Galerkin, coarse PCG, prolongation, and limited fine post-PCG arithmetic. The separate adoption gate below checks the simulation integration.

## AGIPC-Core adoption gate

Command:

```text
S:/build-agipc-reproduction/Release/gipc.exe --scene stiff-bunny-drop --solver agipc-core --tet-mesh S:/Assets/sorted_mesh/cube_sorted.16.msh --frames 1 --headless --agipc-fine-correction-iterations 10 --metrics-path S:/perf_diag/agipc_core_cube.json
```

Result: exit 0. Both linear solves adopted the validated candidate (`adoptions=2`, `fallbacks=0`). The last solve used one coarse iteration plus seven fine correction iterations, recomputed the true fine residual before adoption, and reached a residual reduction ratio of `6.421278659914055e-5`. The original fine-system solution distribution, CCD and line-search path then completed with finite vertices and zero ground penetration.

Against `perf_diag/stiffgipc_cube_baseline.json`, both routes used two Newton iterations. The minimum-y difference was `1.11e-16` and penetration difference was `0`. AGIPC-Core reported 9 aggregate coarse/post iterations versus 5 baseline PCG iterations. Its single measured simulation time was `94.10 ms` versus the stored baseline's `26.04 ms`; this tiny diagnostic workload is slower and is not a performance result. Warm-up and repeated interleaved experiments remain required.

The pending mode check `--solver agipc-symhessian` exits with code 2 and names the unimplemented symmetric-Hessian and paper-BVH stages. At this point the validated scope includes criterion, mapping, mixed Galerkin, coarse and post-PCG arithmetic, guarded direction adoption, and a one-frame cube smoke run. The next section adds a ground-contact transition; self-contact and paper-scale performance remain open.

## Thirty-frame ground-contact transition

The focused contact comparison used the same Release executable and cube for 30 frames, once with `--solver stiffgipc` and once with `--solver agipc-core`. Accepted metrics are in `perf_diag/stiffgipc_cube_contact30.json` and `perf_diag/agipc_core_cube_contact30_paper_stop.json`.

Both runs completed 30 frames with finite vertices and zero ground penetration. Their final minimum-y values differed by `1.3044243131199451e-6` at approximately `-0.999943`. StiffGIPC reported 66 applied Newton steps with its previous-direction/scene-threshold convention; AGIPC-Core reported 42 applied steps using the newly solved direction and paper `1e-3 * bbox_diagonal * dt` threshold. AGIPC performed 72 linear solves because each timestep needs a final current-direction solve before it can stop; all 72 candidates were adopted and none fell back. The criterion protected 112 of 1296 accumulated edge samples, so the run exercised changing strain tags and the ground-contact transition.

The measured simulation times were `582.13 ms` for AGIPC-Core and `333.50 ms` for StiffGIPC, a ratio of `1.75x` on this eight-node mesh. This is a negative overhead result, not a paper comparison: coarse assembly, diagnostics and fine post-PCG dominate at this scale. It confirms contact-path compatibility under the paper Newton threshold but leaves self-contact, larger contact systems and repeated performance trials open.

## Rank-aware mixed ABD/FEM gate

The reduced Figure 12 coupling scene used one ABD bunny plus the existing 17x17 FEM cloth fixture for one frame. The first run exposed a real mixed-space defect: the planar cloth was assigned the fixed basis `[1,x,y,z]`, whose `y` column is zero, so only 7 of 8 coarse diagonal blocks were invertible and both AGIPC candidates fell back. The runtime now selects independent coordinate columns from each aggregate's centered rest positions. The same planar aggregate uses three blocks, while volumetric aggregates retain four.

The repaired run in `perf_diag/agipc_core_fig12_hybrid_rankaware1.json` completed with `rank_reduced_affine_nodes=1`, 7 coarse blocks, 7 invertible diagonal blocks, `invalid_entries=0`, one adopted physical Newton direction and one conservative fallback during the final near-zero-residual check. It remained finite with zero penetration. Against `perf_diag/stiffgipc_fig12_hybrid_smoke1.json`, minimum-y differed by `2.77417179506134e-10`; applied Newton steps were 1 versus 2 and reported PCG iterations were 1 versus 7. Simulation times were `99.80 ms` and `99.86 ms`, respectively, which is effectively tied at this scale and is not a paper speedup measurement.

## Reduced Figure 15 self-contact gate

The dedicated contact fixture `paper-fig15-cloth-abd-scaled` places the existing 289-vertex, 512-triangle cloth above a fixed ABD sphere. It uses the paper Figure 15 material and timestep values, `E=1e6` and `dt=.01`, but is explicitly labeled `REDUCED_SCALE`; it is not one of the paper's 10K--174K cloth meshes. The sphere contains 1,647 vertices and 7,056 tetrahedra. The final Release executable SHA-256 is `1D79C61BEDD3D659B81E751C1EF9CDC9042CDFE48071C61305C898C0E5952B5F`, and the cloth OBJ SHA-256 is `9408047E3F11AEED2EA20DC72F65ED349EB959899A3005FDE7B67FB961DC6CBF`.

The accepted 35-frame pair is stored in `perf_diag/agipc_core_fig15_scaled_contact35_final.json` and `perf_diag/stiffgipc_fig15_scaled_contact35_final.json`. Both completed all frames with finite vertices and zero ground penetration. The fixed sphere's maximum displacement was `5.55e-17` in both runs. Contact was active: AGIPC accumulated 8,593 self-collision pair samples with peak 84, while StiffGIPC accumulated 4,122 with peak 89; the difference follows the distinct paper AGIPC and legacy StiffGIPC termination paths, rather than a matched-iteration contact-set comparison.

AGIPC's final mixed system had 879 fine DoF and 21 coarse DoF, an active ratio of `0.0238908`. Its criterion protected 45,618 of 191,200 edge samples (`0.238588`). Of 239 candidate attempts, 207 were adopted and 32 conservatively fell back; the final map contained seven valid coarse 3x3 blocks and no invalid entries.

The optional `--fem-final-state-path` output records stable vertex ids and FP64 positions. Directly pairing all 289 final FEM vertices gave RMS position error `0.0331395`, or `5.7802%` of the baseline RMS displacement, with maximum vertex error `0.0826706`. Aggregate checks agree with the same conclusion: FEM RMS displacement differed by `3.00e-5`, final minimum-y by `0.0289877`, and maximum displacement by `0.0299130`. These are comparable visible dynamics with a measurable localized trajectory difference, so this run passes the reduced Gate F contact check but does not establish equal trajectories.

The single measured pair took `4511.68 ms` for AGIPC-Core and `2371.67 ms` for StiffGIPC, a `1.90x` slowdown. AGIPC used 204 Newton iterations and 4,908 aggregate coarse/post-PCG iterations, versus 141 Newton iterations and 1,890 PCG iterations for the baseline. This is retained as a negative reduced-scale result. Repeated interleaved trials and the paper mesh sizes remain necessary before a performance claim.

## 16,641-node Figure 15 capacity preflight

The next preflight replaced the 289-node cloth with an existing 129x129 grid containing 16,641 vertices and 32,768 triangles. This is within the paper's broad Figure 15 scale range but is not an exact paper asset. Together with the fixed ABD sphere, the runtime contains 18,288 vertices and a 49,935-DoF mixed linear system. The cloth OBJ SHA-256 is `2B2F1EE3279DD38EE778694AEC5386A101C77EAE7F1A18D020C9E9AB816017F8`; the validated Release executable SHA-256 is `11BB85E0DEDC109DB4888A3A7EF05BD9E56477C422569B832A3BF05B8679404D`.

The accepted one-frame CEMAS16 runs are `perf_diag/stiffgipc_fig15_16k_cemas16_final.json` and `perf_diag/agipc_core_fig15_16k_cemas16_final.json`. Both exited 0 with finite vertices, zero ground penetration, two applied Newton steps, and a fixed-sphere maximum displacement of `5.55e-17`. The first frame has no cloth-obstacle contact, so this is a capacity and no-contact arithmetic gate; the 35-frame reduced fixture remains the contact gate.

StiffGIPC used 32 PCG iterations and `162.87 ms` of reported simulation time. AGIPC-Core used 19 aggregate coarse/post iterations and `193.48 ms`; two of three candidates were adopted and the final near-zero check fell back. The all-collapsible cloth mapped through four levels to one rank-aware planar affine aggregate. Its FEM aggregate uses three 3-DoF blocks; including the four ABD prefix blocks gives 21 final coarse DoF, or `0.0421%` of 49,935 fine DoF. The final post-correction reduced its input residual to `0.89385`. These single timings show successful scale-up and lower iteration count, but AGIPC remained `1.19x` slower and no speedup is claimed.

The first attempts appeared to fail after Newton assembly because the Chinese `--cloth-mesh` argument was stored as local-code-page bytes and strict JSON dumping threw after the entire frame had completed. Headless metrics now use nlohmann JSON's replacement error handler. `perf_diag/stiffgipc_fig15_16k_unicode_path_final.json` verifies exit 0 with the original Chinese path; invalid path bytes are visibly replaced in the recorded string, so the ASCII `T:` alias is retained for exact traceable experiment records.

## 16,641-node Figure 15 contact gate

The same non-paper-exact cloth was then advanced for 35 frames with CEMAS16+SRBK. The accepted pair is `perf_diag/stiffgipc_fig15_16k_contact35.json` and `perf_diag/agipc_core_fig15_16k_contact35.json`; FP64 terminal states use the matching `.csv` names. Both routes completed all frames with finite vertices, zero ground penetration, and fixed-sphere displacement below `5.56e-17`. Contact became active near frame 27. StiffGIPC accumulated 36,478 self-collision pair samples with peak 259, while AGIPC accumulated 110,475 with peak 230. Different Newton termination paths make these totals diagnostic rather than directly matched contact-set counts.

Direct comparison of all 16,641 FEM vertices gives RMS terminal position error `0.0199525`, or `3.4385%` of the baseline RMS displacement, and maximum error `0.0671798`. FEM RMS displacement differs by `5.18e-5`; final FEM minimum-y differs by `0.00118892`. The larger run therefore preserves the same visible cloth-on-obstacle motion with a measurable local trajectory difference.

The run also exposes the current large-contact mapping limit. Of 734 Newton mapping attempts, 351 completed and 383 retained collapsible inter-group edges. The incomplete cases comprise 334 warp-hash fixed points and 49 eight-level-cap stops, exactly accounting for 383 conservative candidate fallbacks. Among the 351 assembled candidates, 186 were adopted and 165 failed the downstream candidate gate. The final completed map has 13 translational aggregates plus one rank-reduced planar affine aggregate; together with the ABD prefix it contains 20 coarse blocks, or 60 DoF (`0.1202%` of the 49,935 fine DoF). The full per-Newton evidence is preserved in `perf_diag/agipc_core_fig15_16k_contact35_stats.json`.

AGIPC reported `57.678 s`, 699 applied Newton steps, and 50,224 aggregate coarse/post or fallback-PCG iterations. StiffGIPC reported `19.365 s`, 270 Newton steps, and 18,265 PCG iterations. The `2.98x` slowdown is a negative single-pair result. Its dominant causes are the 74.7% candidate fallback rate and 2.59x Newton-count increase; repeated timing is not justified until mapping completion and candidate acceptance improve.

The runtime now accumulates typed `fallback_reason_counts` instead of retaining only the last adoption result. A focused one-frame rerun in `perf_diag/agipc_core_fig15_16k_fallback_reasons1.json` reports its single conservative fallback as `post_residual_not_reduced`; both other candidates were adopted. The full AGIPC GPU self-test still passes, including the dimension-mismatch fallback counter. The saved 35-frame snapshot predates these counters, so its 165 downstream candidate failures are not retroactively assigned a cause.

## Stable-partition contact rerun

The old 35-frame evidence classified 334 stable warp-hash fixed points as incomplete because they retained collapsible edges across 32-node group boundaries. This was stricter than the paper's local hash recursion and unnecessarily withheld valid ownership maps from Galerkin assembly. The implementation now treats no-progress as a valid `stable_group_partition`, reports `globally_resolved=false` plus the remaining-edge count, and reserves incomplete status for a hierarchy cap that is still reducing. The default cap is now 16 levels.

The GPU self-test adds a 64-node graph whose only edge crosses the 31/32 group boundary. Its deterministic identity partition is valid, retains one diagnostic edge, and passes all ownership invariants. The Release build and the complete criterion/mapping/Galerkin self-test exited 0.

The focused runtime command advanced the 16,641-node Figure 15 cloth through the first contact frame:

```text
S:/build-agipc-reproduction/Release/gipc.exe --scene paper-fig15-cloth-abd-scaled --solver agipc-core --cloth-mesh T:/cloth_129x129.obj --framework abd-cemas-srbk --frames 27 --headless --agipc-fine-correction-iterations 10 --metrics-path S:/perf_diag/agipc_core_fig15_16k_stable_mapping27.json --fem-final-state-path S:/perf_diag/agipc_core_fig15_16k_stable_mapping27.csv
```

It exited 0 with 27 completed frames, finite vertices, zero ground penetration, and fixed-ABD maximum displacement `5.55e-17`. All 114 per-solve mappings were eligible for Galerkin assembly: 78 eliminated every collapsible edge and 36 terminated as stable cross-group partitions. The stable maps spanned 43 to 16,109 coarse nodes and retained 145 to 3,183 diagnostic cross-group links. Candidate adoption rose to 88/114; all 26 fallbacks were downstream `post_residual_not_reduced`, with no `mapping_incomplete` fallback. Full evidence is stored in `perf_diag/agipc_core_fig15_16k_stable_mapping27_stats.json`. Because accepting these maps changes the Newton trajectory, this run is a control-flow and stability validation rather than a direct timing comparison with the earlier 35-frame trajectory.

A single follow-up ablation used `--agipc-fine-correction-iterations 20`. It stayed finite and penetration-free, but took 159 applied Newton steps and `11.518 s`, compared with 87 steps and `8.053 s` at the default cap 10. Its 186 attempts produced 176 adoptions, two post-residual fallbacks and eight coarse-solve residual failures. The higher adoption count did not improve nonlinear convergence, so the default was not changed. Evidence is in `perf_diag/agipc_core_fig15_16k_post20_27.json` and its matching stats snapshot.

## Per-Newton direction quality and matched stopping rule

The next 27-frame AGIPC run added one record per Newton solve for the Galerkin candidate result, feasible/CCD/accepted step lengths, line-search backtracks, and initial/tested energy. Evidence is in `perf_diag/agipc_core_fig15_16k_direction_quality27.json` and `perf_diag/agipc_core_fig15_16k_direction_quality27_stats.json`. It completed with finite vertices, zero ground penetration, fixed-ABD displacement below `5.56e-17`, 94 applied Newton updates plus 27 terminal direction checks, 8,689 aggregate linear iterations, and `8.879 s` reported simulation time. Of 121 Galerkin attempts, 73 were adopted and 48 fell back: 40 `post_residual_not_reduced`, seven `coarse_invalid_preconditioned_residual`, and one `coarse_iteration_cap`.

Line search did not explain the long contact-frame tail. Of the 94 applied AGIPC updates, 89 accepted alpha 1; the minimum accepted alpha was `0.139561`. No update performed an energy or intersection backtrack. Frame 27 contained 67 applied updates and one terminal check. Its last 20 applied updates were full alpha fine-solver fallbacks, while the terminal direction norm reached `0.973890` of the paper tolerance.

To separate solver behavior from termination semantics, the baseline gained `--newton-stop solver-default|paper-current`. The matched command was:

```text
S:/build-agipc-reproduction/Release/gipc.exe --scene paper-fig15-cloth-abd-scaled --solver stiffgipc --newton-stop paper-current --cloth-mesh T:/cloth_129x129.obj --framework abd-cemas-srbk --frames 27 --headless --metrics-path S:/perf_diag/stiffgipc_fig15_16k_paper_stop27.json --fem-final-state-path S:/perf_diag/stiffgipc_fig15_16k_paper_stop27.csv
```

After a clean rebuild, a one-frame legacy-stop health check returned the expected two applied Newton updates and 32 PCG iterations with finite, penetration-free output. The matched 27-frame baseline then completed with 177 applied updates plus 27 terminal checks, 7,179 PCG iterations, and `11.648 s` reported simulation time. Frame 27 alone required 101 applied updates; 172/177 updates accepted alpha 1, the minimum accepted alpha was `0.183667`, and no energy or intersection backtrack occurred. Its terminal direction norm was `0.996033` of the same tolerance. The strict current-direction gate therefore produces a large late-contact tail even in the original fine solver. In this single matched-stop pair AGIPC used 34 fewer frame-27 updates and 83 fewer updates overall, but cost more per update; this is diagnostic evidence, not a repeated speedup measurement.

## Stage timing and deferred fine preconditioner

Evidence is stored in `perf_diag/agipc_core_fig15_16k_stage_timing27.json` and its matching stats snapshot. The run completed all 27 frames with finite vertices, zero ground penetration, and fixed-ABD displacement below `5.56e-17`. Its 146 direction solves comprised 91 candidate adoptions and 55 fallbacks. CUDA-event component sums agree with the complete linear-system path to `0.316 ms` over the run, providing a consistency check for the new timing fields.

The initial diagnostic showed `331.314 ms` of fine-preconditioner assembly on accepted candidates, although those candidates bypass the fine PCG. The implementation now defers that assembly until fallback. The follow-up evidence in `perf_diag/agipc_core_fig15_16k_deferred_preconditioner27.json` and its stats snapshot completed 27 frames with finite vertices and zero penetration. All 14 fallbacks still built the fine preconditioner; 85 accepted candidates accumulated only `0.208 ms` in the preconditioner interval.

The two diagnostics followed different nonlinear trajectories: 146 versus 99 direction solves and 119 versus 72 applied updates. Aggregate terminal FEM RMS displacements were `0.37009939` and `0.37009733`, while FEM minima differed by `3.85e-5`. These aggregate checks do not establish vertexwise equivalence, and the total runtime difference is not treated as a controlled speedup. They do show that deferred construction preserves the fallback mechanism and passes the existing 27-frame finite/penetration gate.

The final Release executable also passed `perf_diag/agipc_deferred_preconditioner_frozen.json`, which forces the Figure 12 frozen linear diagnostic to consume the fine GPU-MAS preconditioner. The report has `passed=true`, SpMV relative error `3.63e-18`, positive `b^T M^-1 b`, and 775/775 SPD local and inverse blocks. This specifically guards the diagnostic path that must assemble the preconditioner before returning early.

The final one-frame StiffGIPC health check in `perf_diag/stiffgipc_deferred_preconditioner_baseline_smoke1.json` retained the expected two Newton updates and 32 PCG iterations, with finite vertices and zero penetration. This confirms that deferred assembly is restricted to the AGIPC adoption path.

## Frozen fallback direction comparison

The explicit diagnostic path captured one mild, moderate, and severe `post_residual_not_reduced` case, including fine/coarse matrices, right-hand sides, mapping data, coarse solution, post-corrected candidate, and fine fallback. `analyze_frozen_fallback.py` independently reconstructs prolongation and recomputes exact residuals and quadratic decreases. All three candidates are within `0.55` degrees of the fine fallback and have lower exact residual. The moderate and severe post-corrections worsen the prolonged residual by `1.51x` and `2.30x`, locating the first divergent stage at post-correction result selection. The new residual guard restores prolongation in those cases while preserving abnormal-stop fallbacks. Detailed evidence is in `AGIPC_FALLBACK_DIRECTION_ANALYSIS.md`.

The resulting Release build passes the complete AGIPC GPU self-test. A 16K one-frame health run is finite and penetration-free. Its timing is rejected because concurrent external GPU load raised Galerkin assembly from the normal millisecond range to hundreds of milliseconds; the full 27-frame guard validation remains pending an idle GPU.

## Coarse snapshot replay preparation

The explicit diagnostic path now prepares one medium-converged (257--1024 blocks), one large-converged (>1024 blocks), and one iteration-cap coarse snapshot. `replay_frozen_coarse.py` independently rebuilds the symmetric half-storage coarse matrix and runs FP64 block-Jacobi PCG. On the three existing seven-block snapshots, all replays converge in one iteration and the maximum relative difference from the saved GPU coarse solution is `2.22e-16`. Large snapshots have not yet been collected. Details and the pending commands are in `AGIPC_COARSE_SNAPSHOT_PREPARATION.md`.

## Frozen coarse GPU MAS32 replay, 2026-09-13

The final Release build and both `--agipc-coarse-replay` inputs exit 0 with `passed=true`. The real seven-block snapshot uses 1 Jacobi iteration and 2 MAS32 iterations, with true relative residuals `6.57e-7` and `8.21e-9`. The deterministic 257-block rotated-chain SPD fixture uses 47 and 14 iterations, with residuals `9.431245e-4` and `9.428987e-4`. GPU/CPU SpMV errors are `4.32e-23` and `3.27e-16`; all 2/2 and 12/12 MAS local matrices and inverses pass SPD checks, and maximum inverse residuals are below `7.06e-7`. GPU Jacobi agrees with the independent CPU fixture replay. The fixture reference is analytic, not a saved real scene solution.

The same executable passes `--agipc-self-test` (exit 0, overall true, mixed Galerkin error `8.75e-17`). Combining replay with `--solver agipc-paper` still exits 2 naming the unavailable stages. Exact commands, JSON paths, the adapter initialization fix and limitations are recorded in `AGIPC_COARSE_MAS32_REPLAY.md`. No large real contact snapshot or post-guard 27-frame validation is claimed.

## Real contact coarse replay and residual guard

The subsequent 27-frame explicit guard/capture diagnostic exits 0: finite vertices, zero ground penetration, fixed ABD displacement `5.55e-17`, 3 guard restores, 84/93 adoptions, and 9 coarse-invalid-preconditioned-residual fallbacks. No post-residual fallback occurs. Direct 16,641-vertex comparison against the matched-stop fine baseline gives RMS position error `0.0004800351` (0.1297% of baseline RMS displacement). The run includes capture overhead and external GPU contention; a run without explicit diagnostics remains pending.

Real 1415/741-block snapshots both pass CPU and GPU replay. GPU Jacobi takes 121/78 iterations, MAS32 28/34, and all true residuals satisfy `1e-3`. All 53/53 and 32/32 local matrices/inverses are SPD; maximum inverse residuals are `5.15e-5` and `2.14e-5`. CPU sparse LU references have true residuals below `1.67e-15`, while saved Jacobi directions have approximately 4.5% Euclidean error. MAS32 quadratic decreases are within 0.0061%/0.0067% of the direct reference, but MAS vector errors have not yet been measured directly. Exact commands, hashes and input limits are in `AGIPC_REAL_COARSE_MAS32_VALIDATION.md`. This analysis increment reuses the unchanged, previously validated C++ Release executable.

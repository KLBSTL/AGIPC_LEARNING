# AGIPC Numerical Validation

Date: 2026-09-11. Build: Release, CUDA 13.0, `sm_86`, Visual Studio 2022. The executable supports the experimental `agipc-core` route; symmetric-Hessian and paper-BVH modes remain unavailable.

## Focused GPU gate

Command:

```text
build-agipc-reproduction/Release/gipc.exe --agipc-self-test
```

Result: exit 0.

| Gate | Evidence |
|---|---|
| Criterion | 10 cases; max absolute tensor error `4.098284211995207e-17`; history, strict equality, monotonic thresholds, boundary and NaN protection passed |
| Mapping | 7 cases; indirect connectivity, protected edge, 7-node tail, 32-versus-33 affine boundary, hierarchy, repeat determinism and rank-aware planar/linear bases passed |
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

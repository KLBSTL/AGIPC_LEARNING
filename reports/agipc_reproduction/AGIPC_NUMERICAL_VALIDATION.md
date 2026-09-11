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
| Mapping | 5 cases; indirect connectivity, protected edge, 7-node tail, 32-versus-33 affine boundary, hierarchy and repeat determinism passed |
| Mixed Galerkin | relative matrix error `8.747661209037898e-17`; RHS error `0`; symmetry error `9.308113199029161e-18`; SPD Cholesky passed |
| Mixed indexing | all `1x1/1x4/4x1/4x4` shapes passed; adjoint error `5.551115123125783e-17` |
| Geometry rank fixtures | planar affine basis rank `3`; collinear rank `2`, both detected as rank deficient |
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

The pending mode check `--solver agipc-symhessian` exits with code 2 and names the unimplemented symmetric-Hessian and paper-BVH stages. The validated scope is therefore criterion, mapping, mixed Galerkin, coarse and post-PCG arithmetic, guarded direction adoption, and a no-penetration cube smoke run. Fine-contact regression suites and paper-scale performance are still open.

# AGIPC Performance Analysis

Status: provisional reduced-scale evidence as of 2026-09-11. No paper-scale speedup has been reproduced.

## Reduced Figure 15 preflight

The first true mixed-contact timing fixture uses a 289-vertex FEM cloth, one fixed ABD sphere, `E=1e6`, `dt=.01`, and 35 frames. It keeps the paper material/timestep semantics but is far below the paper's 10K--174K Figure 15 meshes.

| Metric | AGIPC-Core | StiffGIPC |
|---|---:|---:|
| Simulation time | 4511.68 ms | 2371.67 ms |
| Newton iterations | 204 | 141 |
| Linear iterations | 4908 | 1890 |
| Self-collision pair samples | 8593 | 4122 |
| Peak self-collision pairs | 84 | 89 |
| Fine linear-system DoF | 879 | 879 |
| Final coarse DoF | 21 | n/a |

The observed time ratio is `1.90`, so the current AGIPC-Core route is slower on this fixture. Its final active-DoF ratio is only `2.389%`, but the present implementation pays for mixed Galerkin expansion, block-Jacobi coarse PCG, up to ten fine post-PCG iterations, diagnostics, and conservative fallback. It also follows the paper current-direction Newton gate, while the baseline retains the original previous-direction scene threshold; iteration and collision-pair totals are therefore expected to differ.

This is one paired measurement, not a statistical benchmark. Timing varied materially across process warm-up in earlier probes, although all observed pairs showed AGIPC overhead at this size. Performance acceptance still requires warm-up and interleaved StiffGIPC/AGIPC trials, at least three measured runs per method, exact executable and mesh hashes, stage timing, and the 10K/51K/92K Figure 15 meshes before larger cases.

## 16,641-node capacity preflight

An existing 129x129 cloth provides a larger but non-paper-exact Figure 15 input: 16,641 FEM vertices, 32,768 cloth triangles, 18,288 total vertices, and 49,935 mixed fine DoF. One Release frame with the same CEMAS16+SRBK framework produced:

| Metric | AGIPC-Core | StiffGIPC |
|---|---:|---:|
| Simulation time | 193.48 ms | 162.87 ms |
| Applied Newton steps | 2 | 2 |
| Linear iterations | 19 | 32 |
| Final coarse DoF | 21 | n/a |
| Candidate adoptions / fallbacks | 2 / 1 | n/a |
| Finite / ground penetration | yes / 0 | yes / 0 |

The AGIPC time ratio is `1.19`, despite a lower reported iteration count. The final active ratio is `21 / 49935 = 0.0421%`, because the no-contact first frame collapses the full planar cloth to one rank-aware affine aggregate. This is a scale/capacity smoke result with no contact and only one timing sample. It does not replace the required warm, interleaved multi-frame measurements on exact paper meshes.

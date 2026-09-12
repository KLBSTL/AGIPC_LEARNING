# AGIPC Performance Analysis

Status: provisional reduced-scale evidence as of 2026-09-12. No paper-scale speedup has been reproduced.

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

## 16,641-node contact pair

The same cloth was run for 35 frames so that it contacted the fixed ABD sphere. This remains a non-paper-exact asset and a single diagnostic pair.

| Metric | AGIPC-Core | StiffGIPC |
|---|---:|---:|
| Simulation time | 57.678 s | 19.365 s |
| Applied Newton steps | 699 | 270 |
| Linear iterations | 50,224 | 18,265 |
| Self-collision pair samples | 110,475 | 36,478 |
| Peak self-collision pairs | 230 | 259 |
| Mapping completions / attempts | 351 / 734 | n/a |
| Candidate adoptions / fallbacks | 186 / 548 | n/a |
| Final coarse DoF | 60 | n/a |

The measured time ratio is `2.98`; AGIPC also used 2.59x the Newton steps and 2.75x the reported linear iterations. Mapping remained incomplete on 383 attempts, while another 165 completed mappings failed candidate validation. The final active ratio of `60 / 49935 = 0.1202%` shows that DoF reduction itself is aggressive, but low mapping completion and adoption rates erase its benefit. The accumulated adaptive shadow pipeline time was `6.607 s`, about 11.5% of AGIPC simulation time; the larger cost comes from extra Newton work and fine-solver fallbacks. Repeated timing is deferred until those controls improve.

This 35-frame pair predates the stable-partition correction. The shortest runtime check that reaches contact is the 27-frame `agipc_core_fig15_16k_stable_mapping27` run. Its 114 mapping attempts comprise 78 globally resolved maps and 36 stable cross-group partitions, with no mapping-stage fallback. It adopted 88 candidates and typed all 26 remaining fallbacks as `post_residual_not_reduced`. Reported simulation time was `8.053 s`, but this unpaired run validates control flow and numerical stability only; it is not a replacement speed measurement. The next performance work should target post-correction quality before repeating the full paired benchmark.

Raising the fine post-PCG cap from 10 to 20 is a rejected ablation. On the same 27-frame setup it reduced total fallbacks from 26 to 10, but candidate-path changes increased attempts from 114 to 186, applied Newton steps from 87 to 159, and simulation time from `8.053 s` to `11.518 s`. Eight of the remaining failures moved to `coarse_invalid_preconditioned_residual`, while two remained `post_residual_not_reduced`. More candidate adoptions therefore did not produce better nonlinear progress; the default remains 10, and the next investigation must use direction quality and Newton work rather than adoption rate alone.

## Matched paper-current stopping ablation

The earlier timing pairs used different nonlinear termination semantics. A focused 27-frame ablation now forces the original StiffGIPC fine solver to use the same newly solved direction and paper tolerance as AGIPC-Core. The AGIPC row is the direction-quality diagnostic run; both rows reach the first large contact frame and are single measurements.

| Metric | AGIPC-Core | StiffGIPC, paper-current |
|---|---:|---:|
| Simulation time | 8.879 s | 11.648 s |
| Applied Newton updates | 94 | 177 |
| Frame-27 applied updates | 67 | 101 |
| Linear iterations | 8,689 | 7,179 |
| Full-alpha updates | 89 / 94 | 172 / 177 |
| Energy / intersection backtracks | 0 / 0 | 0 / 0 |
| Terminal direction / tolerance, frame 27 | 0.973890 | 0.996033 |
| Finite / ground penetration | yes / 0 | yes / 0 |

The matched gate changes the interpretation of the earlier Newton totals. Long late-contact convergence is not specific to the adaptive direction: the baseline itself needs 101 frame-27 updates under the strict paper gate, versus 67 for AGIPC in this pair. Line search accepts almost every update at full alpha for both solvers, so rejection is not the source of the tail.

The remaining performance problem is per-update cost. AGIPC averages `94.46 ms` per applied update (`8.879 s / 94`), while the fine baseline averages `65.81 ms` (`11.648 s / 177`), making the current AGIPC update about `1.44x` as expensive. AGIPC finishes this single run sooner only because it takes 83 fewer updates. Candidate quality and nonlinear work must therefore be reported separately from coarse-system overhead; a speedup claim still requires repeated interleaved trials and exact paper assets.

## Per-stage GPU diagnostic and deferred preconditioner

A CUDA-event diagnostic splits each AGIPC linear-system attempt into fine assembly, adaptive pipeline, fine preconditioner, candidate decision, fallback fine solve, and distribution. It also splits the adaptive pipeline into Galerkin assembly, coarse PCG, and fine post-correction. Events synchronize per stage, so these results identify cost centers but are not benchmark timings.

The initial run made 146 attempts and reported `7.121 s` of simulation time. The complete measured linear-system path was `3.555 s` (49.92%). Its principal components were adaptive assembly `0.496 s`, coarse PCG `1.180 s`, post-correction `0.293 s`, fine assembly `0.337 s`, fine preconditioner `0.535 s`, and fallback fine solve `0.697 s`. Mapping accumulated only `0.191 s`. Component sums close to within `0.316 ms` of the outer total, and the two adaptive-pipeline timers differ by `8.219 ms` over the whole run.

Of the fine-preconditioner cost, `331.314 ms` occurred on 91 accepted candidates and was never consumed by the fine solver. The AGIPC path now defers this assembly until the candidate gate chooses fallback. In the follow-up run, 85 accepted candidates accumulated only `0.208 ms` in that event interval, while all 14 fallbacks still assembled the preconditioner and solved normally. The complete measured path averaged `18.746 ms` per attempt versus `24.347 ms` before the change.

Both runs completed 27 frames with finite vertices and zero ground penetration. Their attempt counts differed materially (99 versus 146), so the follow-up's `4.180 s` total cannot be divided by the initial `7.121 s` as a controlled speedup. The defensible optimization result is removal of about `3.64 ms` of unused GPU work per accepted attempt on the measured trajectory. Coarse PCG remains the largest adaptive cost center and is the next performance target.

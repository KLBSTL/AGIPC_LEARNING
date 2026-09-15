# AGIPC Performance Analysis

## Numerical prerequisite update (2026-09-13)

The full fine H/g freeze at update300/frame31 passes independent reference checks but exposes weak accepted-direction quality:98.33% Euclidean error and9.50% predicted quadratic decrease versus fine direct. Exact coarse solve changes almost nothing; six giant affine groups dominate node coverage/error and the best coarse-space Euclidean fit still has93.87% error. CPU20 does not resolve it. Prioritize criterion/protected-edge/history and merging diagnosis; preserve default post10 and paper threshold. No timing from the contended capture or CPU reference, and no higher-adoption count, establishes a speedup. See `AGIPC_ACCEPTED_DIRECTION_QUALITY.md`.

The 2026-09-15 performance-environment audit also fails the controlled-timing gate: a10-second idle probe reports10%--63% GPU utilization (median29%),18% median memory activity, P5/P8 transitions and300--450MHz graphics clocks, with unowned WDDM/compute processes. The new checkpoint pair is instrumented, follows different Newton trajectories and uses scaled assets, so all of its timing is excluded. It does establish the numerical branch location: frame27 Newton2/update56 is the first tag/mapping mismatch, and frame35 Jacobi/MAS state difference is2.13368%. See `AGIPC_FRAME_CHECKPOINT_DIVERGENCE.md`.

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

Frozen fallback analysis further separates correctness from coarse-PCG performance. The three representative `post_residual_not_reduced` cases each have one coarse FEM aggregate and one coarse iteration, so they do not explain the accumulated coarse-PCG cost. Their failure occurs after prolongation: bounded post-correction worsens an already stronger direction. The residual guard addresses that selection issue; coarse-PCG profiling remains a separate task for the large accepted systems.

A post-guard performance run was rejected before completion because unrelated processes held the GPU near 66% utilization. The one-frame Galerkin assembly event increased from the normal millisecond range to `320.7 ms`, and reported simulation time reached `8.375 s`. These values diagnose external contention and are excluded from every speed comparison.

Offline size/iteration analysis of the two completed stage-timed runs confirms that large coarse systems dominate the remaining adaptive cost. Systems above 1,024 coarse blocks account for only 12.3% and 20.2% of attempts but 64.7% and 81.3% of coarse-solve time. Iteration count and coarse-solve time have Pearson correlations `0.9954` and `0.9933`; the slowest ten events alone account for 44.9% and 55.6% of cost. Small systems up to 16 blocks are more than 60% of attempts but no more than 6.1% of cost. The next performance experiment should therefore freeze representative large coarse systems and compare the current block-Jacobi PCG against a coarse MAS preconditioner. Full distributions are in `AGIPC_COARSE_PCG_ANALYSIS.md`.

Explicit medium/large/iteration-cap coarse snapshot capture and an independent CPU replay tool are now implemented. Three existing small snapshots replay within `2.22e-16` relative direction difference. No large snapshot has yet been collected, and no coarse MAS comparison or post-guard speed measurement is claimed. Snapshot writing is outside the inner Galerkin stage events but can perturb the outer adaptive and total runtime; capture runs must be excluded from timing comparisons. The pending commands are in `AGIPC_COARSE_SNAPSHOT_PREPARATION.md`.

The subsequent standalone GPU comparison passes on a real seven-block snapshot and a synthetic 257-block SPD matrix. On the synthetic matrix, MAS32 needs 14 iterations versus block-Jacobi's 47 (70.2% fewer), at the same `1e-3` true-residual target. The small real snapshot needs 2 MAS32 iterations versus 1 Jacobi iteration. These are iteration results, not speedups: MAS setup/application costs are unmeasured, the ordering is an intermediate identity mapping, and no large real contact matrix has been tested. Default coarse dispatch remains block-Jacobi. See `AGIPC_COARSE_MAS32_REPLAY.md` for results and interpretation.

The next explicit 27-frame capture produced real 1415/741-block converged contact matrices. Fixed-matrix GPU MAS32 replay takes 28/34 iterations versus Jacobi's 121/78, reducing iterations by 76.9%/56.4% at the same true-residual gate. Direct sparse references show MAS quadratic-decrease gaps of about 0.006%, but do not yet establish its Euclidean direction error. The capture run triggers 3 residual-guard restores and remains finite and ground-penetration-free. These results extend the iteration trend to real contact matrices; they still exclude setup/update costs and nonlinear trajectory changes, and are not scene speedups. Full results and remaining comparisons are in `AGIPC_REAL_COARSE_MAS32_VALIDATION.md`.

Complete GPU vector export now confirms that MAS direct-reference errors are 4.15%/1.15%, versus Jacobi's 4.41%/4.30%. Single-run setup+PCG wall intervals are 13.835/15.600 ms for MAS and 27.047/21.227 ms for replay Jacobi. MAS settings cost more, while PCG costs less. These intervals exclude common preparation and diagnostics, were collected under 27%--37% external load, and always run Jacobi before MAS. Replay Jacobi uses CPU LLT, unlike production coarse setup. The values are cost diagnostics, not controlled speedups or scene predictions; repeated idle-GPU timings remain pending. See `AGIPC_COARSE_VECTOR_AND_COST_ANALYSIS.md`.

Initial runtime MAS32 integration validates83/83 candidates and uses2367 total linear iterations versus Jacobi's6150 in a same-build27-frame pair. Their RMS terminal state difference is0.0001947. However, the complete experimental MAS coarse interval is2430.358ms versus Jacobi1240.539ms: full CPU local-matrix diagnostics consume1431.870ms (58.9% of the MAS interval), while setup consumes407.239ms (16.8%). These are included subintervals, not additive costs on top of the total. External load and differing trajectories exclude a controlled speed ratio, but the internal breakdown clearly identifies diagnostic cost as the next engineering target. Preserve the negative result and validated gates; optimize check implementation before caching/reuse or claims. Details are in `AGIPC_RUNTIME_COARSE_MAS32.md`.

GPU local checks now avoid full local-matrix host copies and preserve the same FP64 finite/SPD/inverse-residual gates. Separate CPU/GPU/reuse-crosscheck27-frame ablations pass, with57 real hierarchy reuse calls in the last case. Diagnostic wall totals are1748.761/659.579/2857.577ms, but graph/setup and Galerkin assembly fluctuate by tens of seconds under near-full GPU memory and93%--100% utilization. The crosscheck case includes both CPU/GPU checks and has more candidate attempts. These values are excluded from controlled speedup claims; no scene runtime ratio is valid. Keep exact-structure reuse explicit, always refresh coefficients/inverses, and defer controlled interleaved measurements until the GPU is idle. See `AGIPC_MAS_BATCH_OPTIMIZATION.md`.

The35-frame pair has107570/98750 Jacobi/MAS linear iterations, but observed complete coarse intervals49447.267/170356.489ms. MAS settings/checks consume43580.315/17919.110ms and reuse only54/640 structures (8.44%). Different fallback routes (89 versus1), near-full external GPU load, and2.46746% terminal-state divergence rule out a controlled speed comparison or numerical-equivalence claim. Do not proceed to quantitative speed work until same-H/g direction quality and the local rejection are independently resolved; raw cost remains a diagnostic negative result.

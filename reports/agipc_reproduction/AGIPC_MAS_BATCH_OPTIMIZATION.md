# AGIPC 粗层 MAS32：诊断迁移与拓扑复用

日期：2026-09-13。本轮把后续工作合并为诊断迁移、拓扑复用、27 帧消融与 35 帧接触延长四步。初版结果及负面成本保留在 `AGIPC_RUNTIME_COARSE_MAS32.md`。

结论：局部 GPU/CPU 检查一致性及 27 帧消融有证据；35 帧两条路线都能完成，但终态 RMS 差占参考位移 2.47%，MAS 有一次局部拒绝。当前不能宣称长接触场景数值等价或场景加速，下一批优先复核方向质量与失败粗系统。

## 1. 为什么按这个顺序推进

初版 MAS32 在 27 帧中减少了累计线性迭代，但完整粗层成本高于 Jacobi。CPU 局部诊断占粗层区间的 58.9%，因此先优化检查实现，再评估复用收益。两项由独立参数控制，可以分别关闭；默认粗层仍为 block-Jacobi。

诊断只改变检查的执行位置：`--agipc-mas-validation gpu|cpu|crosscheck`，默认 GPU；CPU 保留完整原诊断，crosscheck 同时执行并要求分类一致、最大逆残差绝对差不超过 `1e-9`。真实粗残差、正 rMz、signed curvature、迭代上限及细层候选采用门槛继续执行。

复用由 `--agipc-mas-reuse` 显式启用。仅在真实粗块数与 BCOO 行列数组逐项相同时保留 MAS 层级和工作区；结构变化立即释放并重建。每次复制当前矩阵值、清空局部矩阵并重新组装和求逆，没有沿用旧系数。邻接数组仍在 CPU 生成，匹配检查仍需复制索引；这不是 GPU 邻接实现。

## 2. GPU 局部检查与边界

每个 32 节点局部矩阵对应一个 128 线程 CUDA block，在 FP64 中展开 96×96 矩阵，检查有限性、局部矩阵/逆的 Cholesky 正定性，以及 `||A Inv-I||_F/sqrt(96)<=1e-3`。局部零标量对角置一的规则沿用 CPU 诊断。矩阵及粗 PCG 仍是 FP64，原 MAS 局部逆仍是 FP32。

主机只接收每个局部矩阵的紧凑检查结果，不再复制完整局部矩阵；CPU 诊断中的逐层 rTz 仍由 CPU/crosscheck 模式提供，求解器本身继续逐次检查正 rMz。GPU kernel 需要 73,728 字节动态共享内存及约 3 KiB 静态共享内存；当前 sm_86 构建支持 opt-in 共享内存。其他设备的共享内存兼容性尚未验证，CPU 模式保留作对照。

重要检查：`gipc.exe --mas32-self-test` 退出 0，原层级测试和六个局部检查用例通过。单位矩阵被接受；局部非正定、逆非正定、逆 NaN、SPD 但逆残差超标、局部 NaN 均被拒绝。这是人工负例，不能写成捕获了真实异常粗系统。

真实冻结输入重放均退出 0：

| 真实粗块数 | CPU/GPU 诊断一致 | GPU 最大逆残差 |
|---:|---|---:|
| 1415 | 是 | 5.15083045e-5 |
| 741 | 是 | 2.14053288e-5 |

对应 `perf_diag/agipc_mas_batch_frozen1415.json`、`agipc_mas_batch_frozen741.json`；每次重放都执行原 Jacobi/MAS PCG 和真实残差门槛。

## 3. 运行时消融

三组 27 帧使用同一 Release 可执行文件、16,641 节点 cloth、E=1e6、dt=.01、`abd-cemas-srbk`、当前方向 Newton 门槛与 10 次细层后校正。均不启用捕获诊断。CPU 不复用、GPU 不复用、crosscheck 复用分别隔离诊断位置与复用行为；crosscheck 的 CPU 成本计入验证时间，不能作为 GPU 复用的性能结果。

后续结果由 `analyze_mas_ablations.py` 检查匹配参数、完整帧数、有限顶点及原 FEM 顶点编号，并记录累计阶段成本和终态差异。35 帧 Jacobi/GPU+复用配对由 `compare_coarse_runtime.py` 分析。

CPU 不复用与 GPU 不复用均退出 0。前者 94/94 次采用、67 次应用 Newton 更新、3201 次累计线性迭代；后者 89/89 次采用、62 次更新、2803 次累计线性迭代。两者局部失败/回退均为 0、残差保护恢复各 1 次，地面穿透为 0，固定 ABD 最大位移 `5.55e-17`，最大自接触对数均为 139。

GPU 对 CPU 的 16,641 顶点 RMS 终态差 `0.0002265518`，最大差 `0.002012696`，占 CPU RMS 位移 `0.06121%`。本轮 CPU 对历史初版 MAS 的 RMS 差 `0.0002317811`（占历史位移 `0.06263%`），最大差 `0.002690901`。相同参数仍出现不同更新轨迹，因此不能宣称逐位一致，也不能把更新数的变化归因于诊断优化。

crosscheck+复用也退出 0：117/117 次采用、90 次应用 Newton 更新、3276 次累计线性迭代，57 次层级复用（48.72%），局部失败/回退/残差保护恢复均为 0。每个 MAS 粗解同时执行 CPU/GPU 检查，失败计数为 0；最后一次 `cpu_gpu_agreement=true`。结构变化后的新层级也重新检查，没有假设整段仿真结构恒定。

该组对 CPU 的 RMS 终态差 `0.0002394546`，最大差 `0.002224998`，占 CPU 位移 `0.06470%`；顶点有限、穿透为 0、固定 ABD 位移 `5.55e-17`、最大自接触对数 139。复用覆盖了真实场景中的相同结构调用，但异常输入覆盖仍来自上述人工局部负例，不能推广为所有接触条件安全。

| 累计区间，ms | CPU 不复用 | GPU 不复用 | crosscheck+复用 |
|---|---:|---:|---:|
| Galerkin 组装 | 28532.485 | 66992.249 | 18319.300 |
| 完整粗层区间 | 28900.071 | 12570.398 | 26678.511 |
| MAS 设置 wall-clock | 24783.248 | 10288.116 | 21296.001 |
| MAS 局部检查 wall-clock | 1748.761 | 659.579 | 2857.577 |
| 细层后校正 | 424.288 | 437.593 | 577.588 |

GPU 模式的完整局部矩阵主机复制已被紧凑结果复制替代，但这张表不是受控性能对照。组装和设置区间相对此前毫秒级记录大幅膨胀，且不同运行之间变化很大；它们反映显存/调度竞争，不能作为算法性能结论。

### 3.1 延长至 35 帧：保留局部拒绝结果

GPU+复用 MAS 完成 35/35 帧并退出 0：605 次应用 Newton 更新、98750 次累计线性迭代，640 次候选尝试、639 次采用、1 次回退；54 次同结构复用，11 次残差保护恢复。顶点有限、地面穿透为 0、固定 ABD 位移 `5.55e-17`，最大自接触对数由 27 帧的 139 增至 216。

这次回退的汇总原因是 `coarse_mas_local_diagnostics_failed`，局部失败计数为 1。因此不能宣称 35 帧全程通过局部检查。该运行没有开启矩阵捕获，只保留累计原因；失败发生于哪次调用、具体哪个检查项、是否落在复用调用以及 CPU 是否同样拒绝，均尚未确定，不能把它直接解释成逆残差超限或缓存错误。

复用比例在这条延长轨迹上为 54/640（8.44%），明显低于 crosscheck27 的 48.72%；同时接触阶段 Newton 工作量显著增长。不同运行轨迹及更多接触使结构匹配频率变化，这说明同结构复用只解决部分设置成本。后续需要保存拒绝时的紧凑诊断、当前 Ac/RHS 和结构，独立 CPU 复核之后才判断局部逆、GPU 检查或复用是否存在问题。

同一场景可执行文件的默认 Jacobi 配对也完成 35/35 帧、退出 0：

| 指标 | 默认 Jacobi | GPU MAS+复用 |
|---|---:|---:|
| 应用 Newton 更新 | 643 | 605 |
| 累计线性迭代（候选/后校正/回退合计） | 107570 | 98750 |
| 尝试 / 采用 / 回退 | 678 / 589 / 89 | 640 / 639 / 1 |
| 粗迭代上限回退 | 78 | 0 |
| 无效预条件残差回退 | 11 | 0 |
| MAS 局部拒绝 | 0 | 1 |
| 残差保护恢复 | 7 | 11 |
| 最大自接触对数 | 221 | 216 |
| 顶点有限 / 地面穿透 | 是 / 0 | 是 / 0 |
| 固定 ABD 最大位移 | 5.55e-17 | 5.55e-17 |

逐点配对 16,641 个 FEM 顶点：RMS 终态差 `0.0143181991`，最大差 `0.0726423534`，占 Jacobi RMS 位移 `2.46746%`。这远大于此前 27 帧的差异，不能只凭有限性、穿透为零或采用率更高就接受数值等价。这里比较的是不同实验的最终状态，没有逐帧状态检查点，不能准确定位首次分化帧。

| 帧 | Jacobi 应用更新 | MAS 应用更新 |
|---:|---:|---:|
| 27 | 102 | 42 |
| 28 | 58 | 54 |
| 29 | 19 | 16 |
| 30 | 26 | 38 |
| 31 | 139 | 131 |
| 32 | 89 | 85 |
| 33 | 43 | 74 |
| 34 | 57 | 84 |
| 35 | 83 | 54 |

两条路线前 26 帧的应用更新合计都为 27 次；逐帧日志总和与 JSON 的 643/605 次吻合。Jacobi 有大量粗上限回退而 MAS 主要采用粗解+有限后校正，两者实际使用的细层方向组合不同。终态差不能单独归因于 GPU 检查、FP32 局部逆或缓存。需在相同 H/g 上比较延拓、后校正与完整细层参考方向的真实残差、夹角及 Newton 停止量，区分方向质量与非线性接触轨迹放大。

| 累计区间，ms | 默认 Jacobi | GPU MAS+复用 |
|---|---:|---:|
| Galerkin 组装 | 24015.713 | 13731.766 |
| 完整粗层区间 | 49447.267 | 170356.489 |
| 细层后校正 | 3403.411 | 10690.370 |
| MAS 设置 wall-clock | — | 43580.315 |
| MAS 局部检查 wall-clock | — | 17919.110 |

迭代数略少没有带来更低的观测粗层区间；复杂接触下，局部检查迁移和有限复用也不足以证明实际场景改善。GPU 持续占用且两条轨迹/回退组合不同，仍不报告受控速度比。

### 3.2 为下一次拒绝准备有界诊断

针对汇总信息不足，后续代码增加 `mas_failure_samples`：最多保存前 4 次局部拒绝的更新序号、真实粗块数/非零块数、是否复用、主诊断及 CPU/GPU 对照。GPU/CPU 单一模式只在这些拒绝发生时补查另一个后端；crosscheck 直接保留已有结果。补查时间计入 MAS 验证区间，拒绝仍进入原细层回退，CPU 结果不会覆盖原先的拒绝决定。

开启现有 `--agipc-diagnostics` 时，粗系统冻结新增首个 `local_diagnostics_failure` 类别，保存当前 Ac/RHS、结构与诊断。拒绝时保存的粗解是零或部分迭代值，元数据 `reference_solution_valid=false`，不能视为有效参考方向。默认运行不写矩阵快照。

这属于失败可观测性的准备；上述 35 帧拒绝发生于补充记录之前，不能倒填本次缺失的诊断，也不能宣称新冻结类别已经捕获并验证了真实异常输入。

## 4. 计时边界与剩余问题

运行前 GPU 利用率约 36%，实验期间达到 93%—100%，显存接近 8 GiB；系统中有其他图形程序。因此本轮计时仅定位检查/设置成本，不报告受控加速比。验证时间和设置时间已经包含在完整粗层区间，不能重复加到总时间上。不同 Newton 轨迹也不能用总迭代数直接推断每次粗解性能。

本轮不代表 AGIPC 论文完整复现。真实 iteration-cap 异常矩阵、独立对称 Hessian、stackless BVH、完整自交/最小间距检查、精确论文网格与空闲 GPU 交错重复计时仍待完成。

下一批优先两项数值工作：用新失败记录/冻结复核局部拒绝；冻结代表性接触状态的完整细层 H/g，在同一系统比较延拓、10 次后校正和完整细层参考方向，并增加少量状态检查点。先确定 35 帧状态差的来源，再建立独立对称 Hessian 装配逐项等价检查。stackless BVH 随后用相同冻结状态比较排序后的候选与接触集合，均通过后才允许 `agipc-paper`。空闲 GPU 的至少三次交错计时单独进行，不以本轮拥挤环境下的耗时替代。

## 5. 命令与追溯

公共运行命令与三组区别：

```powershell
$common=@('--scene','paper-fig15-cloth-abd-scaled','--solver','agipc-core','--cloth-mesh','T:/cloth_129x129.obj','--framework','abd-cemas-srbk','--frames','27','--headless','--agipc-fine-correction-iterations','10','--agipc-coarse-preconditioner','mas32')
S:/build-agipc-reproduction/Release/gipc.exe @common --agipc-mas-validation cpu --metrics-path S:/perf_diag/agipc_mas_batch_cpu27.json --fem-final-state-path S:/perf_diag/agipc_mas_batch_cpu27.csv
S:/build-agipc-reproduction/Release/gipc.exe @common --agipc-mas-validation gpu --metrics-path S:/perf_diag/agipc_mas_batch_gpu27.json --fem-final-state-path S:/perf_diag/agipc_mas_batch_gpu27.csv
S:/build-agipc-reproduction/Release/gipc.exe @common --agipc-mas-validation crosscheck --agipc-mas-reuse --metrics-path S:/perf_diag/agipc_mas_batch_reuse_cross27.json --fem-final-state-path S:/perf_diag/agipc_mas_batch_reuse_cross27.csv
& 'E:\Anaconda\envs\DL\python.exe' reports/agipc_reproduction/analyze_mas_ablations.py perf_diag/agipc_mas_batch_cpu27.json perf_diag/agipc_mas_batch_gpu27.json perf_diag/agipc_mas_batch_reuse_cross27.json --executable perf_diag/agipc_mas_batch_scene_fe0a4421.exe --mesh T:/cloth_129x129.obj --output perf_diag/agipc_mas_batch_ablations.json
S:/perf_diag/agipc_mas_batch_scene_fe0a4421.exe --scene paper-fig15-cloth-abd-scaled --solver agipc-core --agipc-coarse-preconditioner mas32 --agipc-mas-validation gpu --agipc-mas-reuse --cloth-mesh T:/cloth_129x129.obj --framework abd-cemas-srbk --frames 35 --headless --agipc-fine-correction-iterations 10 --metrics-path S:/perf_diag/agipc_mas_batch_reuse35.json --fem-final-state-path S:/perf_diag/agipc_mas_batch_reuse35.csv
S:/perf_diag/agipc_mas_batch_scene_fe0a4421.exe --scene paper-fig15-cloth-abd-scaled --solver agipc-core --cloth-mesh T:/cloth_129x129.obj --framework abd-cemas-srbk --frames 35 --headless --agipc-fine-correction-iterations 10 --metrics-path S:/perf_diag/agipc_mas_batch_jacobi35.json --fem-final-state-path S:/perf_diag/agipc_mas_batch_jacobi35.csv
& 'E:\Anaconda\envs\DL\python.exe' reports/agipc_reproduction/compare_coarse_runtime.py perf_diag/agipc_mas_batch_jacobi35.json perf_diag/agipc_mas_batch_reuse35.json --output perf_diag/agipc_mas_batch_contact35_analysis.json
```

S: 映射当前代码仓库，T: 映射实验输出的 `meshes` 文件夹。27 帧三组和最终分析均退出 0；非法诊断名称、未选择 MAS 却使用复用参数均退出 2。Release 增量构建退出 0：`cmake --build S:/build-agipc-reproduction --config Release --target gipc -j1`，日志 `perf_diag/agipc_mas_batch_build.log`。

可执行文件 SHA256：`FE0A442178D48C4BC5BE4541C308C99DC8DC6310E25AFF588979B22354B61FBD`；网格 SHA256：`2B2F1EE3279DD38EE778694AEC5386A101C77EAE7F1A18D020C9E9AB816017F8`，记录于 `perf_diag/agipc_mas_batch_ablations.json`。原始 JSON/CSV/log 在 `perf_diag/`，不加入 Git。

五个场景实际均运行于上述 FE0A... 可执行文件，原文件保存在 `perf_diag/agipc_mas_batch_scene_fe0a4421.exe`；35 帧命令列出的副本与原调用 `build-agipc-reproduction/Release/gipc.exe` 字节相同。加入失败记录后的 Release 构建退出 0，日志 `perf_diag/agipc_mas_batch_observability_build.log`；新可执行文件 SHA256 `A7FF3D2D5E9D69E5C64EB2D809AC08326E741B7E6FC6DCA98E6F026160E4FD92`，保存于 `perf_diag/agipc_mas_batch_final_hashes.json`。

新构建最后仅做一次综合 `gipc.exe --agipc-self-test`：退出 0、整体 `passed=true`，输出 `perf_diag/agipc_mas_batch_final_core_selftest.json`。本轮没有为新记录代码重复全场景；新失败记录/冻结类别的真实触发验证仍待下一批完成。早先的局部六用例与真实重放验证的是未改动的 GPU 检查实现，不能充当新冻结类别的运行时验证。

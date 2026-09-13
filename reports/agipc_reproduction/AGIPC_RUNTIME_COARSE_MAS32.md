# AGIPC 粗层 MAS32 运行时实验

日期：2026-09-13。新增显式 `--agipc-coarse-preconditioner block-jacobi|mas32`，默认 `block-jacobi`；MAS32 是矩阵邻接与原始粗块顺序下的实验适配，不标为论文 MAS 完整实现。

这是初版接入的历史记录。后续 GPU 局部检查、显式同结构复用及成组消融见 [AGIPC_MAS_BATCH_OPTIMIZATION.md](AGIPC_MAS_BATCH_OPTIMIZATION.md)；下文 CPU 检查成本与每次重建描述保留为初版证据。

## 1. 接入范围

选择 `--solver agipc-core --agipc-coarse-preconditioner mas32` 后，粗层 PCG 使用已有 Traditional GPU MAS32，细层矩阵、细层后校正、CCD、线搜索与细层回退保持现有路径。非 AGIPC 的 StiffGIPC 模式拒绝 MAS32 粗层选项，避免静默忽略。

适配器从实际粗 BCOO 的行列索引生成对称邻接，保留原粗编号。只对 MAS 存储补齐到 32 的倍数，并添加隔离的单位对角块；原 Galerkin 矩阵、PCG 向量、点积、迭代上限和延拓索引始终使用真实维度。每次预条件应用在适配器内部填充零 RHS 尾部并拷回真实分量。

每个粗解重新构建并释放 MAS。设置阶段的邻接处理在 CPU 上，矩阵值在 GPU 上复制与补齐；没有 CPU 稠密全矩阵。局部逆及 MAS 中间向量保留原 FP32 实现，粗矩阵/PCG 保持 FP64。

## 2. 数值门槛与指标

保留原粗系统对角检查。MAS 首次应用后检查所有局部矩阵和逆的有限性/正定性、最大局部逆残差 `<=1e-3`。设置或局部诊断失败以 `coarse_mas_setup_failed` 或 `coarse_mas_local_diagnostics_failed` 进入现有细层回退。MAS PCG 还要求正的预条件残差内积，保留 signed curvature 和迭代上限检查。

MAS 求解完成后重新计算真实粗残差，要求相对目标 `1e-3`，仅允许 `1e-6` 相对舍入裕量；失败记录 `coarse_true_residual_not_converged`。随后仍须通过细层后校正与候选采用门槛。默认 Jacobi 的迭代与停止语义不改变。

JSON 新增顶层 `agipc_coarse_preconditioner`；粗解记录实际 `preconditioner`、`true_residual_checked`、递推与真实残差、MAS 局部诊断及设置/验证 wall-clock 区间。汇总增加 MAS 尝试/局部失败次数和累计设置/验证时间。冻结元数据从实际粗解记录预条件器。

这是正确性优先的初始接入：原粗 Jacobi 对角检查仍执行，MAS 每次重建、CPU 邻接复制和局部矩阵诊断均有额外成本；尚未做缓存、GPU 邻接或同结构复用。不得将冻结矩阵 PCG 的速度趋势直接当成运行时加速。

## 3. 同可执行文件的 27 帧数值对照

使用同一网格、E、dt、细层框架和后校正上限，均不启用 `--agipc-diagnostics`。MAS 与默认 Jacobi 顺序运行，各一次：

```powershell
S:/build-agipc-reproduction/Release/gipc.exe --scene paper-fig15-cloth-abd-scaled --solver agipc-core --agipc-coarse-preconditioner mas32 --cloth-mesh T:/cloth_129x129.obj --framework abd-cemas-srbk --frames 27 --headless --agipc-fine-correction-iterations 10 --metrics-path S:/perf_diag/agipc_runtime_mas32_27.json --fem-final-state-path S:/perf_diag/agipc_runtime_mas32_27.csv
S:/build-agipc-reproduction/Release/gipc.exe --scene paper-fig15-cloth-abd-scaled --solver agipc-core --cloth-mesh T:/cloth_129x129.obj --framework abd-cemas-srbk --frames 27 --headless --agipc-fine-correction-iterations 10 --metrics-path S:/perf_diag/agipc_runtime_jacobi_27.json --fem-final-state-path S:/perf_diag/agipc_runtime_jacobi_27.csv
& 'E:\Anaconda\envs\DL\python.exe' reports/agipc_reproduction/compare_coarse_runtime.py perf_diag/agipc_runtime_jacobi_27.json perf_diag/agipc_runtime_mas32_27.json --fine-baseline perf_diag/stiffgipc_fig15_16k_paper_stop27.json --output perf_diag/agipc_runtime_mas32_pair_analysis.json
```

均退出 0。比较脚本检查匹配参数和 FEM 顶点编号；旧细层基线的 `T:\...` 与当前 `T:/...` 按 Windows 路径语义规范化后比较，未改变输入网格。

| 指标 | 默认 Jacobi | 实验 MAS32 |
|---|---:|---:|
| 完成帧数 | 27/27 | 27/27 |
| 应用 Newton 更新 | 58 | 56 |
| 累计线性迭代（含候选与回退） | 6150 | 2367 |
| 尝试 / 采用 / 回退 | 85 / 77 / 8 | 83 / 83 / 0 |
| MAS 设置尝试 / 局部失败 | 0 / 0 | 83 / 0 |
| 残差保护恢复 | 2 | 1 |
| 顶点有限 / 地面穿透 | 是 / 0 | 是 / 0 |
| 固定 ABD 最大位移 | 5.55e-17 | 5.55e-17 |
| 最大自接触对数 | 139 | 139 |

Jacobi 的 8 次回退均为 `coarse_invalid_preconditioned_residual`。本次 MAS 无局部诊断失败，也没有粗求解/后校正导致的细层回退；83 个方向都通过真实粗残差及细层采用门槛。该结果只覆盖本次接触轨迹，不能证明所有异常输入均有效。

16,641 个 FEM 顶点的逐点终态比较：

| 对照 | RMS 位置差 | 最大位置差 | 占参考 RMS 位移 |
|---|---:|---:|---:|
| MAS 对 Jacobi | 0.0001946583 | 0.001633093 | 0.05260% |
| Jacobi 对已有细层基线 | 0.0004728363 | 0.003214337 | 0.12776% |
| MAS 对已有细层基线 | 0.0004698645 | 0.003437604 | 0.12696% |

两条路线都接近已有同阈值细层基线，但 MAS 的最大逐点差稍大，不能只看 RMS 宣称状态完全等价。地面穿透检查和接触对数不替代完整自交或最小间距检查。本次也补齐了无显式采样的 Jacobi 残差保护 27 帧稳定性复验。

## 4. 成本反思

启动前 GPU 利用率为 60%，验证结束后为 26%；单次顺序运行，不报告受控场景加速比。以下阶段计时用于定位成本，包含主机间隙和数值检查：

| 累计区间，ms | 默认 Jacobi | MAS32 |
|---|---:|---:|
| Galerkin 组装 | 360.598 | 352.330 |
| 粗系统求解完整区间 | 1240.539 | 2430.358 |
| 细层后校正 | 167.328 | 121.357 |
| MAS 设置 wall-clock | — | 407.239 |
| MAS 局部诊断 wall-clock | — | 1431.870 |

线性迭代减少没有自动转化为更低的粗层完整成本。局部 CPU 诊断约占本次 MAS 粗层区间的 58.9%，设置约占 16.8%；二者合计约 75.7%。这些子项已包含在完整粗层区间，不能再相加到总仿真时间上。剩余区间还包含共同对角检查、首次应用、PCG、真实残差、延拓/细层残差评估、分配释放与发射间隙，不能直接视为纯 MAS PCG。

每次局部诊断复制 MAS 局部矩阵和逆到 CPU，并执行 FP64 SPD/逆残差检查。它在首次实验中提供必要证据，但成为初版成本热点。下一步优先优化数值门槛的实现，减少完整 CPU 复制与检查成本，同时保留异常拒绝、正内积、真实残差和细层采用门槛；不要直接删除检查来制造加速。设置复用和 GPU 邻接应作为后续独立优化。

## 5. 验证与可追溯性

- Release 增量构建退出 0：`cmake --build S:/build-agipc-reproduction --config Release --target gipc -j 1`，日志 `perf_diag/agipc_runtime_mas32_build.log`。
- `gipc.exe --agipc-self-test` 退出 0、整体通过：`perf_diag/agipc_runtime_mas32_core_selftest.json`。它保护默认 criterion/mapping/Galerkin 路线；MAS 路线由上述真实场景验证。
- 非法粗层名称和 StiffGIPC 下选择粗层 MAS 均按预期退出 2：`agipc_runtime_mas32_invalid_cli.log`、`agipc_runtime_mas32_baseline_cli.log`。
- 可执行文件 SHA256：`AEF4E8BFFD1DB893CAA58FCBAB93A61695EF1F8B98B1901DEFC1E090EB9EE4BB`；网格 SHA256：`2B2F1EE3279DD38EE778694AEC5386A101C77EAE7F1A18D020C9E9AB816017F8`，记录于 `perf_diag/agipc_runtime_mas32_pair_hashes.json`。

默认粗层仍为 Jacobi；MAS 是显式实验选项。本轮未重复冻结矩阵测试或额外长场景。真实异常粗系统捕获、局部检查优化、空闲 GPU 的交错重复计时，以及论文对称 Hessian/stackless BVH 仍待完成。

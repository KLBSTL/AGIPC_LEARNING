# AGIPC 残差保护与真实粗系统 MAS32 验证

后续进展：完整 MAS 向量已导出并通过直接参考比较，设置/PCG 计时及人工迭代上限门槛已完成。本文保留首次真实样本验证记录；新增证据见 `AGIPC_COARSE_VECTOR_AND_COST_ANALYSIS.md`。

日期：2026-09-13。使用提交 `ce73af4` 对应的 Release 可执行文件，完成一次 16,641 节点 cloth-on-fixed-ABD 的 27 帧显式诊断，并重放两个真实粗系统。当前结果支持残差保护的数值行为和粗层 MAS32 的进一步接入研究；不构成场景加速测量。

## 1. 运行条件与边界

本次启动前 GPU 利用率为 40%，显存占用 6487/8192 MiB，P3。显式诊断包含 GPU 数据复制与写盘，因此仿真耗时不用于性能比较。此次合并采样和残差保护的数值复验，避免在有外部负载时重复运行同一场景；无显式诊断的 27 帧对照仍待完成。

场景为已有 16,641 节点布料与固定 ABD 球体，`E=1e6`、`dt=.01`，框架 `abd-cemas-srbk`，细层 CEMAS16，粗层 block-Jacobi，后校正上限 10。该资产仍为 reduced/paper-inspired workload。

```powershell
S:/build-agipc-reproduction/Release/gipc.exe --scene paper-fig15-cloth-abd-scaled --solver agipc-core --cloth-mesh T:/cloth_129x129.obj --framework abd-cemas-srbk --frames 27 --headless --agipc-fine-correction-iterations 10 --agipc-diagnostics --metrics-path S:/perf_diag/agipc_core_fig15_16k_guard_capture27.json --fem-final-state-path S:/perf_diag/agipc_core_fig15_16k_guard_capture27.csv
```

运行退出 0。二进制和网格 SHA256 保存于 `perf_diag/agipc_guard_capture27_hashes.json`：

- 可执行文件：`D25A78A10D5D23A1B4A0BD4DAF596B51C2E7638FD3735F39FE6567E4A8D2A3C8`
- 布料：`2B2F1EE3279DD38EE778694AEC5386A101C77EAE7F1A18D020C9E9AB816017F8`

## 2. 残差保护与终态

| 指标 | 本次诊断 |
|---|---:|
| 完成帧数 | 27/27 |
| 应用 Newton 更新 | 66 |
| 累计线性迭代 | 6617 |
| 候选尝试 / 采用 / 回退 | 93 / 84 / 9 |
| 恢复延拓方向次数 | 3 |
| 后校正残差回退 | 0 |
| 粗层无效预条件残差回退 | 9 |
| 顶点有限 | 是 |
| 地面穿透 | 0 |
| 固定 ABD 最大位移 | 5.55e-17 |
| 最大自接触对数 | 139 |

3 次恢复说明残差保护在真实接触轨迹上触发；本次没有 `post_residual_not_reduced` 回退。剩余 9 次回退均为 `coarse_invalid_preconditioned_residual`，保护没有掩盖粗 PCG 异常。本次 66 次 Newton 更新与之前轨迹的次数不同，不能仅据此证明保护减少了 Newton 工作量，仍需受控对照。

与已有同场景、同帧数、`paper-current` 停止阈值的细层基线 `stiffgipc_fig15_16k_paper_stop27.csv` 按相同顶点编号逐点比较：16,641 个 FEM 顶点的 RMS 位置差 `0.0004800351`，最大位置差 `0.0031220934`，RMS 差占基线 RMS 位移的 `0.1297%`。结果为 `perf_diag/agipc_guard_capture27_state_comparison.json`。这支持本次终态接近细层基线，但地面穿透为零与接触对数不能替代完整自交/最小间距验证。

## 3. 实际粗系统采样

`perf_diag/agipc_core_fig15_16k_guard_capture27_coarse_samples/manifest.json` 保存两个样本：

| 样本 | 粗块 / 标量 DoF | 半存储非零块 | 保存粗 PCG 迭代 | 保存解真实相对残差 |
|---|---:|---:|---:|---:|
| `large_converged_56` | 1415 / 4245 | 9368 | 119 | 9.476146e-4 |
| `medium_converged_57` | 741 / 2223 | 4860 | 77 | 9.308946e-4 |

两个样本的粗 PCG 收敛，候选均有效并进入采用路径。本次未出现迭代上限冻结样本；这类输入覆盖仍待补齐。

独立 CPU block-Jacobi 重放分别需要 116 和 78 次，真实残差均满足 `1e-3`，相对保存粗解差为 `0.3816%`、`0.2386%`。原运行、CPU 重放和独立 GPU 重放的迭代数略有不同；不能像小型一步系统一样要求逐位一致。阈值停止位置与浮点归约路径对真实矩阵较敏感。

## 4. 固定矩阵 GPU 对照

```powershell
S:/build-agipc-reproduction/Release/gipc.exe --agipc-coarse-replay S:/perf_diag/agipc_core_fig15_16k_guard_capture27_coarse_samples/large_converged_56 --metrics-path S:/perf_diag/agipc_coarse_mas32_real1415.json
S:/build-agipc-reproduction/Release/gipc.exe --agipc-coarse-replay S:/perf_diag/agipc_core_fig15_16k_guard_capture27_coarse_samples/medium_converged_57 --metrics-path S:/perf_diag/agipc_coarse_mas32_real741.json
```

两次均退出 0、整体 `passed=true`。两种预条件器使用同一矩阵、RHS、零初值和 `1e-3` 真实残差门槛，MAS 保留原始粗块顺序、矩阵邻接和 FP32 局部逆。

| 指标 | 1415 块 | 741 块 |
|---|---:|---:|
| GPU/CPU SpMV 相对误差 | 2.25e-16 | 1.81e-16 |
| GPU Jacobi 迭代 | 121 | 78 |
| GPU MAS32 迭代 | 28 | 34 |
| 迭代减少 | 76.9% | 56.4% |
| Jacobi 真实相对残差 | 9.296077e-4 | 9.019927e-4 |
| MAS32 真实相对残差 | 9.938150e-4 | 8.983797e-4 |
| 局部矩阵及逆 SPD 数 | 53/53 | 32/32 |
| 最大局部逆相对残差 | 5.15e-5 | 2.14e-5 |
| MAS32 相对保存方向差 | 6.3654% | 4.8775% |

这些真实接触矩阵支持此前合成矩阵的迭代下降趋势。局部逆误差比合成矩阵更大，但仍通过 `1e-3` 门槛。尚未测量 MAS 初始化、更新和每步应用成本；迭代减少不能换算为场景加速比。

## 5. 方向差与直接解复核

残差达标并不保证欧氏方向误差同样小。为避免把保存的近似 Jacobi 解视为精确解，新增 CPU 重放选项 `--direct-reference`，用 SciPy 稀疏 LU 生成独立参考，要求真实相对残差不超过 `1e-10`。

```powershell
& 'E:\Anaconda\envs\DL\python.exe' reports/agipc_reproduction/replay_frozen_coarse.py perf_diag/agipc_core_fig15_16k_guard_capture27_coarse_samples --direct-reference --quiet --output perf_diag/agipc_guard_capture27_coarse_direct.json
```

两例直接解残差为 `1.67e-15` 和 `1.21e-15`。保存 Jacobi 解相对直接解的方向误差为 `4.4922%`、`4.5312%`，CPU Jacobi 为 `4.6271%`、`4.4728%`。因此 MAS32 与保存解的 4.9%–6.4% 差不能全部解释为 MAS 精度下降。当前 GPU 诊断未导出 MAS 向量，尚未直接测量其相对直接解的欧氏方向误差。

同时比较 `D(d)=b^T d - 0.5 d^T A d` 与直接解的 `D*`，报告 `(D*-D)/D*`：

| 预测二次下降量相对差距 | 1415 块 | 741 块 |
|---|---:|---:|
| GPU Jacobi | 0.006105% | 0.013255% |
| GPU MAS32 | 0.006018% | 0.006639% |

MAS32 的该项质量未退化，中型样本还更接近直接解。但这是粗系统局部二次模型，不能保证细层后校正、线搜索和非线性轨迹相同。复核结果为 `perf_diag/agipc_guard_capture27_coarse_quality.json`。

## 6. 下一阶段

1. 为冻结重放补充 MAS 向量输出或直接参考比较，量化完整方向误差，并覆盖迭代上限/异常粗残差样本。
2. 在真实冻结矩阵上分开测量 MAS 设置与求解成本；GPU 空闲后执行必要的重复对照。
3. 以显式实验选项接入粗层 MAS，保持 Jacobi 默认与异常回退，复验细层残差、终态和 Newton 轨迹。
4. 无显式诊断的残差保护 27 帧对照、论文对称 Hessian 和 stackless BVH 仍待完成。

本轮未修改 CUDA/C++ 模拟路径，复用已通过构建和核心自检的 Release 文件；实际场景、两次 GPU 重放及带直接参考的 CPU 重放均成功。未重复完整构建或核心自检。

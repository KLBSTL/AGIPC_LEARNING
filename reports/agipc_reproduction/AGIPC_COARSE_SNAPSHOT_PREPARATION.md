# AGIPC 大型粗系统采样与离线重放准备

后续进展：已完成 27 帧显式残差保护/采样诊断，取得真实 1415/741 块样本并通过 CPU 与 GPU MAS32 重放。下文保留准备阶段的历史记录；最新结果及仍待完成的无诊断对照见 `AGIPC_REAL_COARSE_MAS32_VALIDATION.md`。

日期：2026-09-13。已实现显式诊断模式下的大型粗系统冻结，并完成已有三个小型粗系统的独立 CPU 重放。大型样本和残差保护的 27 帧复验尚未完成；当前没有新增性能结果。

## 1. 采样范围

`--agipc-diagnostics` 在原有回退方向采样之外增加 `<metrics-stem>_coarse_samples/`。每次运行最多保存以下三类各一个样本：

| 类别 | 条件 |
|---|---|
| `medium_converged` | 粗 PCG 收敛，粗块数为 257–1024 |
| `large_converged` | 粗 PCG 收敛，粗块数大于 1024 |
| `iteration_cap` | 粗 PCG 停止原因是达到迭代上限 |

类别中的收敛仅指粗层 PCG；元数据另外记录后校正结束后的 `candidate_ready` 和失败原因。采样保存该次粗层 `Hc/bc/dc`，不额外执行细层或粗层求解。三类样本位于不同目录，索引使用本次运行的 Galerkin 更新编号。

每个样本包含五个二进制文件和一个元数据 JSON：矩阵 3×3 块值、行索引、列索引、RHS、粗解、求解记录。矩阵为对称半存储，块值是 Eigen 列优先 FP64；根目录的 `manifest.json` 列出已完成样本。文件写入失败会报告异常，避免将空文件作为有效冻结证据。

## 2. 计时边界

冻结发生在既有 Galerkin 阶段 CUDA event 已结束之后，因此不进入内部的组装、粗求解和后校正计时。完整自适应路径和总运行时间仍可能包含诊断复制及写盘等待；显式诊断运行不能用于速度比较。元数据中的 `binary_freeze_ms` 只覆盖二进制复制与写入，不含随后 JSON 元数据和清单写入。

不启用显式诊断时，采样函数只检查空目录并返回。默认求解方向、PCG 阈值、后校正和候选采用规则均未改变。

## 3. 独立重放

`replay_frozen_coarse.py` 重建标量 CSR 对称粗矩阵，检查尺寸、索引、有限值、对角块对称性和正定性，然后从零初值独立执行 FP64 block-Jacobi PCG。它使用相对残差目标 `1e-3`，最大迭代数与运行时一致，并记录保存粗解及 CPU 重放解的真实残差、预测二次下降量和方向差。

脚本兼容之前回退样本中已保存的粗矩阵，因而可以在本机 GPU 忙碌时验证读取与重放：

```powershell
& 'E:\Anaconda\envs\DL\python.exe' reports/agipc_reproduction/replay_frozen_coarse.py perf_diag/agipc_core_fig15_16k_fallback_quality27_fallback_samples --output perf_diag/agipc_coarse_replay_small.json
```

三个已有样本均为 7 个粗块、16 个半存储非零块。独立重放均一次迭代收敛；保存解的真实相对粗残差分别为 `4.20e-15`、`6.57e-7`、`1.32e-6`，CPU 解与保存 GPU 解的最大相对方向差为 `2.22e-16`。这些结果验证小型冻结数据的布局与重放基线，不代表大型粗系统已经通过验证。

## 4. 待完成的运行

GPU 检查为利用率 51%、显存 7,829/8,192 MiB、P0，故未启动 27 帧运行。GPU 空闲后先运行不含显式采样的保护复验，以获得不含写盘扰动的控制流证据；再用相同场景启用采样，获取大型系统。

```powershell
S:/build-agipc-reproduction/Release/gipc.exe --scene paper-fig15-cloth-abd-scaled --solver agipc-core --cloth-mesh T:/cloth_129x129.obj --framework abd-cemas-srbk --frames 27 --headless --agipc-fine-correction-iterations 10 --metrics-path S:/perf_diag/agipc_core_fig15_16k_residual_guard27.json --fem-final-state-path S:/perf_diag/agipc_core_fig15_16k_residual_guard27.csv

S:/build-agipc-reproduction/Release/gipc.exe --scene paper-fig15-cloth-abd-scaled --solver agipc-core --cloth-mesh T:/cloth_129x129.obj --framework abd-cemas-srbk --frames 27 --headless --agipc-fine-correction-iterations 10 --agipc-diagnostics --metrics-path S:/perf_diag/agipc_core_fig15_16k_coarse_capture27.json
```

当前尚未实现粗层 MAS。大型样本取得后，先确认保存解的真实粗残差，再在固定 `Hc/bc` 上比较预条件器，避免把非线性轨迹差异误当作预条件器性能差异。

## 5. 当前验证

- `cmake --build S:/build-agipc-reproduction --config Release --target gipc -j 1`：最终增量构建退出码为 0；日志为 `perf_diag/agipc_coarse_freeze_build.log`。沙箱内构建因 Windows SDK 目录访问被拒绝而失败，获准在沙箱外构建后通过。
- `gipc.exe --agipc-self-test`：criterion、mapping、mixed Galerkin 全部通过；相对矩阵误差 `8.75e-17`，RHS 误差为 0，后校正残差比 `2.33e-3`，候选复制误差为 0。结果为 `perf_diag/agipc_coarse_freeze_selftest.json`。
- 上述 CPU 重放命令：退出码为 0，三个小型样本均收敛；结果为 `perf_diag/agipc_coarse_replay_small.json`。

核心自检不触发大型冻结类别，因而不能替代大型样本生成路径的实际运行验证。该路径仍需后续接触场景诊断运行确认。

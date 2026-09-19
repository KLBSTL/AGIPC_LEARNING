# AGIPC 粗系统完整方向与设置成本分析

日期：2026-09-13。本轮扩展冻结粗系统重放，输出完整 Jacobi/MAS32 解，并分别记录预条件器设置和 PCG 成本。两例真实接触矩阵均通过独立向量复核；默认模拟仍使用 block-Jacobi。

## 1. 接口与检查范围

`--agipc-coarse-replay` 的结果新增 `solution_f64`：仅原始粗系统的 `3 * coarse_block_nodes` 个 FP64 分量，按 xyz 块顺序存储，补齐节点不写入向量。另记录 `padding_solution_norm`，GPU 原始残差仍在补齐后的完整系统上复算。原有通过/失败门槛保留。

CPU 重放新增可重复的 `--gpu-replay <JSON>`，与 `--direct-reference` 联用。独立读取 GPU 向量，检查尺寸、有限值及真实残差，要求其满足 `1e-3` 目标，并与 GPU 报告的真实相对残差在绝对 `1e-10` 内一致。随后比较欧氏方向误差、夹角、A 加权误差和预测二次下降量。

```powershell
S:/build-agipc-reproduction/Release/gipc.exe --agipc-coarse-replay S:/perf_diag/agipc_core_fig15_16k_guard_capture27_coarse_samples/large_converged_56 --metrics-path S:/perf_diag/agipc_coarse_vectors1415.json
S:/build-agipc-reproduction/Release/gipc.exe --agipc-coarse-replay S:/perf_diag/agipc_core_fig15_16k_guard_capture27_coarse_samples/medium_converged_57 --metrics-path S:/perf_diag/agipc_coarse_vectors741.json
& 'E:\Anaconda\envs\DL\python.exe' reports/agipc_reproduction/replay_frozen_coarse.py perf_diag/agipc_core_fig15_16k_guard_capture27_coarse_samples --direct-reference --gpu-replay perf_diag/agipc_coarse_vectors1415.json --gpu-replay perf_diag/agipc_coarse_vectors741.json --quiet --output perf_diag/agipc_coarse_vectors_independent.json
```

三个命令均退出 0。GPU 整体 `passed=true`，四个导出向量均通过独立真实残差门槛。直接参考仍为同一矩阵的 SciPy 稀疏 LU，残差约 `1e-15`。输出向量长度为 4245/2223，四个补齐区向量范数均为零；原始快照未修改。

## 2. 完整方向结果

以下误差均相对直接参考；A 加权误差为 `sqrt((d-d*)^T A (d-d*) / (d*^T A d*))`。

| 指标 | 1415 块 Jacobi | 1415 块 MAS32 | 741 块 Jacobi | 741 块 MAS32 |
|---|---:|---:|---:|---:|
| PCG 迭代 | 121 | 28 | 81 | 34 |
| 独立真实相对残差 | 9.255637e-4 | 9.909898e-4 | 8.224590e-4 | 8.968951e-4 |
| 欧氏方向相对误差 | 4.4101% | 4.1474% | 4.2962% | 1.1550% |
| 相对直接解夹角 | 2.5229° | 2.3742° | 2.4559° | 0.6561° |
| A 加权相对误差 | 0.7813% | 0.7757% | 1.1074% | 0.8148% |
| 二次下降量相对差距 | 0.006105% | 0.006018% | 0.012263% | 0.006639% |

在这两个输入上，MAS32 的完整方向精度没有退化，中型系统的欧氏误差明显较小。大系统仍有约 4.1% 的欧氏方向误差，说明固定 `1e-3` 残差目标不能保证同等大小的方向误差；它不只影响 Jacobi。当前不因这些结果修改论文残差目标。

本次中型 Jacobi 为 81 次，上一轮为 78 次，两者均在相同目标下停止。真实系统的浮点归约与停止位置存在波动；迭代数量不能要求逐位一致，独立复算的真实残差和解质量是主要检查。此前仅用 MAS 与保存 Jacobi 解的距离讨论方向差；本轮补齐直接参考比较，收窄了这一不确定性。

## 3. 设置与 PCG 成本

CUDA event 和 steady-clock 分别记录两个区间：

- Jacobi 设置：CPU 对角块 LLT、逆块生成和上传。
- MAS32 设置：邻接排序/展开、分区映射上传、邻接及层级矩阵分配、BCOO 设置与局部逆准备。
- PCG：工作向量已分配并初始化后，从首次预条件应用至 PCG 迭代停止，包含初始范数和循环归约。

共同的快照读取、矩阵上传、前置 SpMV 检查，以及求解后的真实残差/方向复核、MAS 局部诊断、解向量复制与 JSON 输出，均不计入上述设置+PCG之和。CUDA event 区间含主机发射间隙及外部 GPU 争用，不是纯 kernel busy time。

| wall-clock，ms | 1415 块 Jacobi | 1415 块 MAS32 | 741 块 Jacobi | 741 块 MAS32 |
|---|---:|---:|---:|---:|
| 设置 | 1.153 | 7.105 | 1.861 | 4.760 |
| PCG | 25.894 | 6.730 | 19.366 | 10.840 |
| 设置 + PCG | 27.047 | 13.835 | 21.227 | 15.600 |

PCG 的 CUDA event 时间为 `25.884/6.610 ms` 和 `19.359/10.594 ms`，与 wall-clock 接近。MAS 设置明显比重放 Jacobi 昂贵，PCG 部分较少；单次诊断提示设置成本未完全抵消迭代下降，但不能计算受控加速比。

开始检查时 GPU 有 27% 负载，复核后为 37%；每个输入仅重放一次，顺序固定为 Jacobi 后 MAS，缓存与初始化状态不对称。重放 Jacobi 使用 CPU LLT，生产粗 PCG 的对角逆构造路径也不同；MAS 使用原粗编号及 CPU 矩阵邻接，没有动态 METIS 重排。因此这些值用于判断成本构成，不代表生产设置成本或实际场景性能。暂不做额外重复计时。

## 4. 失败门槛与核心验证

将大型真实快照复制到独立 `perf_diag/coarse_replay_cap1/`，仅把元数据迭代上限改为 1，原矩阵/RHS/参考解保持一致。这个派生输入是人工停止门槛检查，不是从场景捕获的真实迭代上限失败样本。

```powershell
S:/build-agipc-reproduction/Release/gipc.exe --agipc-coarse-replay S:/perf_diag/coarse_replay_cap1 --metrics-path S:/perf_diag/agipc_coarse_vectors_cap1.json
S:/build-agipc-reproduction/Release/gipc.exe --agipc-self-test
```

第一条按预期退出 3，整体 `passed=false`；两种 PCG 均执行恰好 1 次并报告 `iteration_cap`，没有将近似方向误标为已通过。第二条退出 0、核心 GPU 自检整体通过，结果为 `perf_diag/agipc_coarse_vectors_core_selftest.json`。

Release 增量构建退出 0，命令为 `cmake --build S:/build-agipc-reproduction --config Release --target gipc -j 1`，详细日志 `perf_diag/agipc_coarse_vectors_build.log`。可执行文件 SHA256：`D2D64BE7687D08C972963029ECBAEBF2A5C119EF8F0BDADD73D47F89F802CF97`。

## 5. 下一步

完整向量精度和人工迭代上限门槛已经得到当前证据。下一阶段可用显式实验选项接入粗层 MAS，保留 Jacobi 默认和异常回退，检查细层后校正、候选采用、终态及 Newton 工作量。真实异常粗残差/迭代上限输入、无显式诊断的残差保护对照，以及空闲 GPU 上的重复设置/求解成本比较仍待完成。

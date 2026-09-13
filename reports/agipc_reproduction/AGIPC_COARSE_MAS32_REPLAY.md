# AGIPC 冻结粗系统 GPU MAS32 重放

日期：2026-09-13。该阶段新增独立 GPU 诊断入口，在固定粗矩阵上比较 block-Jacobi 与 Traditional GPU MAS32。默认模拟路径仍使用原粗层 block-Jacobi，当前不报告场景加速比，也不将该适配器标为论文 MAS 完整复现。

## 1. 接口与适配范围

```powershell
S:/build-agipc-reproduction/Release/gipc.exe --agipc-coarse-replay <样本目录> --metrics-path <结果JSON>
```

入口读取样本的粗矩阵、RHS、参考粗解和元数据，在加载模拟场景之前完成诊断并退出。两种 PCG 使用同一 FP64 对称半存储矩阵、零初值、相对残差目标 `1e-3` 和原粗系统的最大迭代数。

MAS 适配使用本仓库已有 `gpu_mas32::TraditionalMAS32Preconditioner`，从粗矩阵的结构非零块生成对称邻接，保留冻结粗块编号顺序，以恒等排列初始化分区映射。所有 ABD 前缀和自适应 affine 子块都按矩阵的 3×3 块参与诊断。碰撞耦合已经包含在粗矩阵及邻接内，因此不再向 MAS 单独添加细层碰撞对。

节点数向上补齐为 32 的倍数。补齐节点为相互独立的单位对角块，RHS 和参考解均为零；真实粗系统的迭代上限保持原值，不因补齐增加。矩阵与 PCG 为 FP64，MAS 保留现有实现的 FP32 局部逆与中间向量。

这是矩阵图与原始粗编号顺序下的中间适配。尚未实现动态粗网格的 METIS 重排，也未与当前细层 CEMAS16 路径形成同预条件器的场景性能对照。后续仍需验证真实大型接触矩阵，才能考虑运行时接入。

## 2. 检查与通过条件

诊断先检查文件尺寸、索引、有限值和 block-Jacobi 对角块正定性。用独立 CPU 块 SpMV 对照 GPU 对称 SpMV。MAS 首次应用后检查已有局部矩阵诊断：所有局部及局部逆块必须有限且正定，局部逆相对残差上限为 `1e-3`；失败时不继续运行 MAS PCG。

PCG 停止时重新计算真实粗残差，分别输出递推残差、真实残差、迭代数、失败原因、`b^T d`、预测二次下降量和相对参考方向差。整体通过要求 GPU/CPU SpMV 相对误差不超过 `1e-12`，两种 PCG 均正常收敛且真实相对残差不超过 `1e-3`，方向有限并满足下降检查。

## 3. 必要输入

- 已有真实小系统：`post_residual_moderate_2`，7 个粗块、16 个半存储非零块，包含原场景的 ABD 前缀与 rank-reduced affine 聚合块。
- 确定性合成系统：257 个粗块、513 个半存储非零块。矩阵由正定块质量加相邻节点旋转弹簧构成，即 `A = M + sum 100 * ||x_i - R_i x_(i+1)||^2`，具有解析参考解；非对称 3×3 耦合检查转置路径，257 节点检查多层聚合和尾部补齐。

```powershell
& 'E:\Anaconda\envs\DL\python.exe' reports/agipc_reproduction/make_coarse_replay_fixture.py perf_diag/coarse_mas32_fixture257
& 'E:\Anaconda\envs\DL\python.exe' reports/agipc_reproduction/replay_frozen_coarse.py perf_diag/coarse_mas32_fixture257 --output perf_diag/coarse_mas32_fixture257_cpu.json
```

CPU 基线在合成系统上需要 47 次 block-Jacobi PCG，真实相对残差为 `9.431245e-4`，相对解析解的方向差为 `0.3456%`。这只验证诊断基线，不代表实际 AGIPC 接触系统。

## 4. GPU 重放结果

最终 Release 构建退出 0；两个输入均退出 0，JSON 的整体 `passed=true`。结果保存在 `perf_diag/agipc_coarse_mas32_small.json` 和 `perf_diag/agipc_coarse_mas32_fixture257.json`。

| 指标 | 真实 7 块快照 | 合成 257 块系统 |
|---|---:|---:|
| 补齐后块数 | 32 | 288 |
| GPU/CPU SpMV 相对误差 | 4.32e-23 | 3.27e-16 |
| block-Jacobi PCG 迭代 | 1 | 47 |
| MAS32 PCG 迭代 | 2 | 14 |
| Jacobi 真实相对残差 | 6.57e-7 | 9.431245e-4 |
| MAS32 真实相对残差 | 8.21e-9 | 9.428987e-4 |
| Jacobi 相对参考方向差 | 2.30e-17 | 0.3456% |
| MAS32 相对参考方向差 | 3.09e-7 | 0.1798% |
| MAS 局部矩阵及逆 SPD 数 | 2/2 | 12/12 |
| MAS 最大局部逆相对残差 | 3.52e-8 | 7.06e-7 |

257 块系统的 GPU Jacobi 与独立 CPU 重放同为 47 次，残差吻合至约 `1e-17` 的绝对差。MAS32 在相同残差目标下减少 70.2% 的迭代，且方向差较小，支持进一步测试真实大型粗系统。该输入为链式合成矩阵；CPU 旧脚本中的 `saved_gpu` 字段在此指解析参考解，不能解释为实际大型 GPU 快照。

真实小系统中 Jacobi 一步已满足目标，MAS32 两步达到更小残差。小系统没有支持替换默认预条件器；初始化与每步应用成本尚未计时。当前只完成数值与迭代诊断，未得到场景速度证据。

实际验证命令：

```powershell
S:/build-agipc-reproduction/Release/gipc.exe --agipc-coarse-replay S:/perf_diag/agipc_core_fig15_16k_fallback_quality27_fallback_samples/post_residual_moderate_2 --metrics-path S:/perf_diag/agipc_coarse_mas32_small.json
S:/build-agipc-reproduction/Release/gipc.exe --agipc-coarse-replay S:/perf_diag/coarse_mas32_fixture257 --metrics-path S:/perf_diag/agipc_coarse_mas32_fixture257.json
S:/build-agipc-reproduction/Release/gipc.exe --agipc-self-test
S:/build-agipc-reproduction/Release/gipc.exe --agipc-coarse-replay S:/perf_diag/coarse_mas32_fixture257 --solver agipc-paper
```

核心 GPU 自检退出 0、整体通过，混合 Galerkin 矩阵误差 `8.75e-17`，日志为 `perf_diag/agipc_mas32_replay_core_selftest.json`。最后一条按预期退出 2，报告未实现 symmetric Hessian 和 paper BVH；诊断入口不会绕过不可用求解器门槛。

## 5. 构建与失败记录

新增 CUDA 源触发 CMake 重配置。`FindCUDAToolkit` 搜索系统环境路径中的可选静态库耗时过长；仅在本地构建缓存设置 `-DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=OFF`，保留已定位的 VS、CUDA 13 和 sm_86 配置，重配置约 27 秒完成。未修改仓库 CMake 配置。

```powershell
D:/computer/cmake/bin/cmake.exe -S S:/ -B S:/build-agipc-reproduction -DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=OFF
D:/computer/cmake/bin/cmake.exe --build S:/build-agipc-reproduction --config Release --target gipc -j 1 -- /verbosity:quiet /fl '/flp:logfile=S:/perf_diag/agipc_mas32_replay_final_build.log;verbosity=normal'
```

首个小系统重放在析构时触发 `cudaFree invalid argument`，原因是新适配器创建 `GIPCTripletMatrix` 后遗漏 `init_var()`，导致析构释放未初始化指针。局部补上初始化后重新构建，上述两个重放与核心自检均通过；首次失败不作为 MAS 数值结论。

后续已完成 27 帧显式残差保护/采样诊断，并取得真实 1415/741 块粗系统，MAS32 重放均通过。详细结果、方向差与直接解复核见 `AGIPC_REAL_COARSE_MAS32_VALIDATION.md`。无显式诊断的 27 帧对照与粗层 MAS 场景对照仍待完成。显式 GPU 重放不进入模拟统计，也不能将 GPU 忙碌时的耗时用于速度比较。

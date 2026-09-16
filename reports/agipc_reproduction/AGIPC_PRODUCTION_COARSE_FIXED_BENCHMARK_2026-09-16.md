# AGIPC 生产粗求解固定输入基准（2026-09-16）

## 目的与路径

此前 27/35 帧接触运行的 Newton 轨迹会波动，不能直接用场景总时间筛选毫秒级优化。现有 `--agipc-coarse-replay` 使用独立实现的 PCG；本轮在其原有诊断后增加 `production_coarse`，把快照中**未 padding** 的粗矩阵和右端传给生产 `solve_coarse_shadow()`，连续运行三次。首次建立 MAS 图，后两次在逐项行列索引比较通过后复用图。计时分别记录 MAS setup、GPU 局部校验、PCG（含真实残差复核）及整个 `solve_coarse_shadow()`；后者还包含对恒等 fine 映射的延长和诊断内积。矩阵上传、构造和 CLI 原有独立重放不在计时内。

基准只从现有快照读取，不改变模拟器的调度、碰撞、方向门或预条件器。它还逐次检查收敛、真实残差门、迭代数、图复用和与首次解的相对方向差。`production_coarse.passed` 只表示数值门通过，**不表示重复方向或计时稳定**。

运行方式（在仓库根目录，先将仓库映射为 `S:`）：

```powershell
& 'S:\build-agipc-reproduction\Release\gipc.exe' `
  --agipc-coarse-replay 'S:\perf_diag\agipc_core_fig15_16k_guard_capture27_coarse_samples\large_converged_56' `
  --metrics-path 'S:\perf_diag\production_coarse_large_run1.json'
```

本次 Release 可执行文件 SHA-256 为 `4F6E303507844CC66EC886A2F7A23532B04D1A004EAD9927BCAEB31043BCB6BB`。原始三份大样本、三份中样本 JSON/日志均在 `perf_diag/production_coarse_*`，不纳入 Git。

## 大样本：可用于下一轮局部性能对照

固定输入为 1,415 个粗块节点、9,368 个唯一块。三次独立 CLI 调用各包含 1 次新建图和 2 次精确复用，共 9 次生产粗解；全部 28 轮收敛且真实残差通过，复用状态依次为 `false,true,true`。与各次首次方向的相对差均小于 `5e-7`。计时的中位数如下，单位 ms：

| 情况 | 样本数 | 总墙钟 | MAS setup | MAS 校验 | PCG |
|---|---:|---:|---:|---:|---:|
| 新建图 | 3 | 19.505 | 3.315 | 7.369 | 6.516 |
| 精确复用 | 6 | 14.742 | 0.096 | 7.696 | 6.510 |

复用时总墙钟范围 `12.737–15.314 ms`，校验 `5.918–8.473 ms`，PCG `5.179–8.100 ms`。这表明固定图复用已使 setup 较小，而 **GPU 局部校验与 PCG 成为主要成本**；校验中位数甚至高于 PCG。这个结论是同一输入、同一实现的阶段定位，不是相对旧二进制的提速结论。独立重放器的 MAS PCG 同样为 28 轮，但它不计入上述生产阶段计时。

## 中样本：数值门通过但不适合作为精细计时主样本

741 块、4,860 个唯一块的快照也完成三次独立调用、共 9 次生产粗解，全部通过真实残差门。迭代数却在 `34–42` 之间变化；一次复用解与同调用首次方向的相对差达到 `0.01043`。该次真实相对残差仍为约 `9.29e-4`，低于 `1e-3` 门。其他调用也出现 34/39/42 轮和最高约 1% 的方向差。因此不能把这个样本的毫秒差直接归因于代码优化；它更适合监视收敛和方向稳定性。当前证据尚不能判定差异来自 GPU 非确定性、预条件器数值敏感性还是外部负载，后续不要从它推算速度比。

## 验证与下一步

`--agipc-self-test` 退出 0，`passed=true`，粗 PCG 收敛、细层 post-PCG 9 轮。大/中快照原有 `--agipc-coarse-replay` 数值门和新增 `production_coarse.passed` 均为 true。未重复 35 帧接触测试：本轮新增的是只在粗快照重放 CLI 调用的基准入口，模拟调度未改。

下一轮优先在 1,415 块主样本上优化并交错比较 MAS GPU 局部校验，同时保存 `production_coarse` 的逐次 JSON。必须同时核对局部诊断字段、28 轮收敛、真实残差和方向差；若校验变快但总墙钟不降，应报告同步等待转移，不宣称生产求解加速。

## 后续局部优化筛选（同日）

使用上述 1,415 块快照，以基线可执行文件 `gipc_before_validation_fastpath.exe`（SHA-256 同上）和各候选 Release 构建比较。每份 Nsight Systems trace 包含重放诊断及 3 次生产粗解，所以两个相关 kernel 各调用 4 次；`perf_diag/validation_*` 和 `perf_diag/inverse_sync_*` 保存原始 JSON、日志及 trace。所有候选的 CLI 数值门、`production_coarse.passed` 和 GPU 局部诊断均通过。下面的时间是受 profiling 影响的 GPU kernel 累计时间，用于**筛选候选**，不等同正常运行的生产墙钟。

| 候选 | 修改思路 | 局部校验 kernel，4 次合计 | 求逆 kernel，4 次合计 | 决定 |
|---|---|---:|---:|---|
| 基线 | 原实现 | 14.581 ms | 10.453 ms | 对照 |
| 严格对角占优快速证书 | 在 Cholesky 前跳过可直接证明 SPD 的矩阵 | 14.657 ms | 未针对求逆计时 | 撤回；53 个局部矩阵仅 3 个被证书覆盖 |
| 两次 barrier 的 Cholesky | 用 warp 内广播减少同步 | 14.364 / 17.456 ms | 不作为比较目标 | 撤回；交错基线为 13.428 / 15.474 ms，未见改善 |
| shared memory 缓存完整逆矩阵 | 减少求逆残差检查的全局内存读取 | 17.504 ms | 不作为比较目标 | 撤回；缓存增加 shared memory 和访问成本 |
| 求逆消元去一次 barrier | 各线程使用本列的枢轴行值，下一轮开头再同步 | 18.149 / 15.672 ms | 14.814 / 15.730 ms | 撤回；交错基线分别为 11.931 / 14.449 ms 和 13.358 / 12.914 ms |

最后一项采用 `old,new,new,old` 顺序做四份 trace；求逆 kernel 的 8 次平均从基线约 3.284 ms 增至候选约 3.818 ms。虽然去掉的 barrier 在数据依赖上看似冗余，实际调度和内存访问结果并不支持这个优化。阶段墙钟受环境波动影响较大，不据此宣称总求解速度变化。四项候选源码均已撤回，`gipc.exe` 恢复为上述基线 SHA-256；未触碰 `Assets/sorted_mesh/` 和未跟踪的 `perf_diag/`。

这次 profiling 的其他明显成本是 `prepare_hessian_bcoo_sum_kernel` 约 3.697 ms/4 次，以及生产/诊断 PCG 中大量短 kernel 调用。基线完整 CLI trace 的 CUDA API 汇总记录了 2,827 次 kernel launch、1,298 次 stream sync、768 次 `cudaMalloc` 和 979 次 `cudaFree`；这些计数包含独立诊断重放，**不能直接全部归因于生产 PCG**。GPU 局部校验和求逆仍是单 kernel 热点，但简单增加证书、shared memory 或减少 barrier 都没有收益。下一轮应先拆分 PCG 的 CPU 等待与 GPU 执行时间，再针对重复的归约/数据传输选择一个较小改动；该方向目前只是性能定位，尚无提速结果。

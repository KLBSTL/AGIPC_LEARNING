# AGIPC 判据快照、层级连通与阈值分析

日期：2026-09-15

## 目的与结论

上一轮在一个第 300 次 Galerkin 更新的冻结系统中发现 6 个巨大仿射组，且采用的 GPU10 方向相对细层直接解有 98.33% 欧氏误差。本轮只补充一次必要捕获，保存同一采用方向对应的 Green 应变判据、边标签、边—单元邻接和当前位置，用独立 CPU 脚本回放判据并分析层级映射。

独立回放通过：当前 Green 张量相对误差为 `8.20e-15`，49,408 条边的原因位和折叠标签均为零不一致。静态代码审计也确认，应变历史在每个 subIP 入口重置一次，此后每次 Newton 比较当前与紧邻前一次迭代并覆盖历史，符合论文式 `||G_i-G_{i-1}||_F` 的时序。快照没有保存带符号的前一 Green 张量，因此只独立重建了当前张量，未独立重建增量范数本身。

当前证据不支持“巨大组来自错误的 Green 计算或边标签”。在论文默认阈值 `5e-5` 下，允许边图的全局连通分量中有一个 14,379 节点分量；巨大可合并区域已经由判据产生。运行时的 32-lane 递归哈希停在 2,202 组，最大组 1,031 节点，并保留 2,193 条跨组允许边。论文补充材料描述连续 32 节点分组、递归和 METIS 排序，但没有规定自适应标签造成局部稳定且仍有跨组边时如何继续，因此这是保真度歧义，不能据此认定实现错误。

强行把每个允许边全局连通分量合成一个组会进一步损失方向表达能力：其最佳欧氏粗空间拟合误差为 84.96%，而当前稳定映射为 62.86%。所以不能用全局 union-find 取代论文的 warp-hash 映射。当前采用方向仍较弱，但本次样本明显好于上一冻结样本，说明“第 300 次更新”不是跨运行可比的确定性状态检查点。

## 捕获与验证边界

运行配置沿用 16,641 节点布料、固定 ABD、`agipc-core`、运行时 MAS32、GPU 局部验证与结构复用、默认 Green 阈值 `5e-5`、默认 10 步细层后校正。显式冻结发生在第 30 帧完成后、下一帧内的第 300 次 Galerkin 更新，并在 CCD、线搜索和状态更新前退出；故意指定的完整 metrics 文件不存在。

Release 构建和一次综合核心自检均通过。可执行文件 SHA-256 为 `E795E40EACBDD357D6B928BC54934743CF91C901FBD67F080130FD7D526E47C9`；输入布料 SHA-256 为 `2B2F1EE3279DD38EE778694AEC5386A101C77EAE7F1A18D020C9E9AB816017F8`。捕获带有大量读回和写盘，运行期间 GPU 负载也变化，本轮不使用任何耗时作性能结论。

快照新增内容只在显式 accepted-direction freeze 路径执行：元素顶点/维度/静止逆矩阵、当前 Green 张量及增量范数、边与边—单元邻接、标签与原因位、边界标志和当前 FEM 位置。普通模拟路径不写这些文件。

## 独立判据回放

| 检查 | 结果 |
|---|---:|
| FEM 节点 / 三角形 / 边 | 16,641 / 32,768 / 49,408 |
| 当前 Green 最大相对误差 | `8.2003e-15` |
| 边原因位不一致 | 0 |
| 边标签不一致 | 0 |
| 非法元素 | 0 |
| 保护边 / 比例 | 12,485 / 25.2692% |
| 应变保护 / 边界保护 | 12,485 / 0 |
| 独立回放门 | **通过，exit0** |

增量范数的最小值、中位数、90%、99% 和最大值依次为 `6.57e-7`、`2.96e-5`、`6.78e-5`、`2.38e-4`、`1.05e-3`。默认阈值位于分布中部，而非只筛出极少数离群单元。

12,485 条保护边中，4,049 条的两个端点仍属于同一最终聚合组。这不等于直接跨过保护边：两端可以经其他允许边路径连通，论文的哈希闭包按允许边连通传播，并没有规定 tag0 必须成为全局 must-separate 约束。其余 8,436 条保护边跨越最终组。

## 层级映射与方向表达

运行时层级节点数为 `16641 -> 3893 -> 2496 -> 2300 -> 2245 -> 2219 -> 2203 -> 2202 -> 2202`。末级已达到当前连续编号下的局部稳定状态：

| 项目 | 运行时稳定映射 | 允许边全局连通分量参考 |
|---|---:|---:|
| 聚合组 | 2,202 | 1,958 |
| 混合粗块 | 2,294 | 1,960 |
| FEM 活跃块比例 | 13.7852% | 11.7781% |
| 最大组 | 1,031 | 14,379 |
| 仿射组 / 覆盖节点 | 46 / 13,892 | 1 / 14,379 |
| 对细直接方向的最佳欧氏拟合误差 | 62.8589% | 84.9642% |

运行时映射没有合并互不连通的允许边分量，但把部分全局分量拆成多个局部组；仍有 2,193 条允许边横跨最终组。论文声称全允许时在极限可收缩到单节点，并依赖 METIS 排序保证组内连接；当前全允许布料自检此前确实收缩到单个平面仿射组。自适应标签改变粗编号和局部连接后出现稳定拆分，补充材料没有给出重排或跨稳定点继续合并的规则。保留稳定分组是目前更可审计的选择，也比全局连通合并保留更多方向自由度。

## 同一冻结系统的方向质量

本次细层对称 SparseLU 参考真实相对残差 `1.09e-13`，独立 Galerkin/CPU 后校正门通过。以下数字只描述该 H/g 的线性方向：

| 方向 | 欧氏误差 | H 能量误差 | 相对直接解预测下降 |
|---|---:|---:|---:|
| 精确粗层解延拓 | 92.5185% | 81.2737% | 33.9459% |
| 实际 GPU10 候选 | 87.5464% | 73.2104% | 46.4023% |
| 独立 CPU20 消融 | 78.6277% | 62.5540% | 60.8699% |

粗层直接解是给定粗空间的 H-正交投影；它仍有 81.27% 能量误差，表明粗空间本身造成主要损失。后校正改善了方向，但 10 步仍未得到接近细直接解的方向。20 步结果仅为同一矩阵的离线预算消融，不能支持修改论文默认值或速度结论。

上一捕获与本次捕获的细 H、g 和映射 SHA-256 均不同，不能复用直接解。上一样本只有 261 个组、FEM 活跃块 1.64%、最大组 7,162、最佳欧氏拟合误差 93.87%、GPU10 方向误差 98.33%；本次分别为 2,202、13.79%、1,031、62.86%、87.55%。GPU 并行浮点轨迹和非线性迭代路径会让相同更新序号落在不同状态，后续首次分叉定位必须使用同一运行内的稀疏状态检查点，不能比较两个独立运行的 update300。

## 同一增量场的阈值消融

下表把保存的增量场离线重新打标签，再用允许边的**全局连通分量**计算理想化粗空间。它不是论文 warp-hash 的逐阈值运行结果，也没有重新执行非线性仿真。

| 阈值 | 保护边 | 活跃 FEM 块 | 最大连通分量覆盖 | 最佳欧氏拟合误差 |
|---:|---:|---:|---:|---:|
| `1e-6` | 99.9980% | 99.9940% | 0.0120% | 0.1842% |
| `2.5e-6` | 99.9676% | 99.9099% | 0.0240% | 0.4802% |
| `5e-6` | 99.7288% | 99.2669% | 0.0781% | 1.7170% |
| `1e-5` | 97.1563% | 93.0593% | 0.3305% | 7.3260% |
| `2.5e-5` | 70.3591% | 48.9814% | 14.0196% | 44.4480% |
| `5e-5` | 25.2692% | 11.7781% | 86.4071% | 84.9642% |
| `1e-4` | 5.2320% | 2.7282% | 97.2057% | 92.9218% |

该样本在 `2.5e-5` 到 `5e-5` 之间出现明显连通跃迁：活跃块从约 49% 降到 12%，最大分量从 14% 跳到 86%，方向拟合误差从 44% 升到 85%。这说明论文固定 `5e-5` 阈值在此接触状态附近十分敏感，也与论文把固定阈值列为限制的讨论一致。单快照不能支持直接修改阈值；需要在同一运行内记录少量状态，并同时考察 Newton 收敛、接触安全和终态误差。

阈值诊断图：`perf_diag/agipc_criterion_snapshot_contact300_thresholds.png` 和 `.pdf`。黑色星号是默认阈值下的实际稳定 warp-hash 映射，其余曲线是全局连通分量参考。

## 复现命令与证据

```text
cmake --build S:/build-agipc-reproduction --config Release --target gipc -j1
S:/build-agipc-reproduction/Release/gipc.exe --agipc-self-test
S:/build-agipc-reproduction/Release/gipc.exe --scene paper-fig15-cloth-abd-scaled --solver agipc-core --agipc-coarse-preconditioner mas32 --agipc-mas-validation gpu --agipc-mas-reuse --cloth-mesh T:/cloth_129x129.obj --framework abd-cemas-srbk --frames 35 --headless --agipc-fine-correction-iterations 10 --agipc-direction-freeze-path S:/perf_diag/agipc_criterion_snapshot_contact300 --agipc-direction-freeze-after-update 300 --metrics-path S:/perf_diag/agipc_criterion_snapshot_intentionally_incomplete.json
E:/Anaconda/envs/DL/python.exe reports/agipc_reproduction/analyze_accepted_direction.py perf_diag/agipc_criterion_snapshot_contact300 --output perf_diag/agipc_criterion_snapshot_contact300_direction_analysis.json
E:/Anaconda/envs/DL/python.exe reports/agipc_reproduction/analyze_criterion_snapshot.py perf_diag/agipc_criterion_snapshot_contact300 --output perf_diag/agipc_criterion_snapshot_contact300_analysis.json
E:/Anaconda/envs/DL/python.exe reports/agipc_reproduction/plot_criterion_snapshot.py perf_diag/agipc_criterion_snapshot_contact300_analysis.json --output-stem perf_diag/agipc_criterion_snapshot_contact300_thresholds
```

核心证据为 `perf_diag/agipc_criterion_snapshot_contact300/metadata.json`、`perf_diag/agipc_criterion_snapshot_contact300_analysis.json`、`perf_diag/agipc_criterion_snapshot_contact300_direction_analysis.json`、捕获/构建日志和自检 JSON。`perf_diag` 继续排除在 Git 之外。

## 下一步

1. 保留默认阈值、默认 post10 和当前稳定 warp-hash 语义；本轮没有足够证据修改算法参数或把全局连通分量当作论文映射。
2. 在一次完整 35 帧运行中增加少量、低写盘量的状态摘要检查点，定位 MAS/Jacobi 或 AGIPC/细解轨迹首次显著分叉的帧与 Newton 更新，再只冻结该处的同一 H/g。
3. 在首次分叉附近同时记录判据分布、映射规模、方向预测下降和接触集合摘要，检验阈值连通跃迁是否与终态 2.47% 差异相关。
4. 数值分叉解释清楚之前继续暂停性能优化、论文对称 Hessian/BVH 整合和速度结论。

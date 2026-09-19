# AGIPC 接触系统的采用方向质量诊断

日期：2026-09-13。

## 为什么做这一轮

上一轮 35 帧 Jacobi/MAS 配对虽然均完成、顶点有限且无地面穿透，但终态相对 RMS 差异达到 2.46746%，最大点差异 0.0726423534。MAS 采用 639/640 个候选方向，Jacobi 采用 589/678 个方向，后者还触发 78 次粗层迭代上限回退。采用率不能证明方向接近同一系统的完整细层解，也不能单凭终态差异归因于 GPU 局部诊断或拓扑复用。

本轮只回答同一 H/g 上的问题：被采用的方向有多准确？粗层迭代误差、粗空间截断和细层后校正预算各自表现如何？这不能定位上一轮非线性轨迹首次分叉的帧，也不能替代失效矩阵捕获。

## 捕获方式与边界

- 场景：Figure 15 缩减布料/固定 ABD 球，16,641 个布料顶点、49,935 个混合细层 DoF，E=1e6、dt=.01、`abd-cemas-srbk`。
- 求解：实验性 coarse MAS32、GPU 局部检查、显式结构复用、默认 10 次细层后校正。粗层容差、精度、回退及 Newton 门槛均保持原配置。
- 选择：从第 300 次 Galerkin 更新起，捕获首次实际被采用的候选方向；这是预先指定的单样本选择，不根据方向误差挑选结果。
- 保存唯一细层 BCOO H/g、粗层 H/g/解、完整映射和静止坐标、GPU 延拓方向及实际采用方向。
- 写入完整 `metadata.json` 后，沿现有冻结诊断出口停止，尚未执行该方向的 CCD、线搜索、位置更新和帧计数。此运行不是完整 35 帧结果，故不生成常规最终状态/指标。
- 普通运行不启用捕获。目录必须为空，避免覆盖之前的矩阵。

## 独立 CPU 分析

`analyze_accepted_direction.py` 使用 NumPy/SciPy，而不调用 GPU 求解器。它将 Eigen 列主序 3x3 BCOO 的对称半存储恢复为稀疏细层矩阵，并执行：

1. 从映射和基函数构造稀疏延拓 P，独立验证粗层 Hc=PᵀHP、gc=Pᵀg。除整个混合系统外，单独检查 FEM 区域，避免固定 ABD 的大数值掩盖 FEM 装配误差。
2. 从保存的粗层解重建 Pdc，与 GPU 保存的延拓方向比较。
3. 使用 FP64 SuperLU 对称模式直接求解完整细层 H/g，保持 MMD_AT_PLUS_A 消元顺序、`diag_pivot_thresh=0`；不移位、不正则化、不修改 H/g，独立以半存储 SpMV 检查真实残差。另求粗层直接解，比较 GPU 粗层迭代误差及精确粗层解的延拓误差。
4. 比较实际采用方向、原始延拓和精确粗层延拓与细层直接解的欧氏方向误差、夹角、H 能量范数误差和二次模型预测下降。
5. 从同一个保存的延拓方向出发，离线运行一次 FP64 block-Jacobi PCG，分别保存 10/20 步结果。它们是无 GPU 残差保护选择的预算消融，未修改运行时默认值，也未重跑场景。
6. 单独报告 FEM 方向误差；混合 ABD/FEM 广义坐标的整体欧氏范数不能直接解释为场景最大位移。
7. 复用保存的直接解，按聚合组统计方向平方误差，并逐组做小规模最小二乘，计算当前粗空间对细层方向的最佳欧氏拟合误差。额外扫描 FEM 最大方向与停止门槛，不重复直接分解。

独立参考门槛：细层与粗层直接解真实相对残差 ≤1e-10；延拓重建相对误差 ≤1e-12；GPU/CPU 初末真实残差相对差 ≤1e-9；完整/FEM Galerkin 矩阵与右端项误差 ≤1e-12；参考方向能量及误差能量有效。此门槛验证分析参考，**不把细层 1e-3 残差阈值新增为运行时采用条件**。论文默认有限后校正与强制完整细层解回退是不同算法，见 `AGIPC_PAPER_CONTRACT.md` §E/F/G。

## 可复现命令

`S:` 映射本仓库，`T:` 映射本实验的 mesh 输出目录。

```text
cmake --build S:/build-agipc-reproduction --config Release --target gipc -j1
S:/build-agipc-reproduction/Release/gipc.exe --agipc-self-test
S:/build-agipc-reproduction/Release/gipc.exe --scene paper-fig15-cloth-abd-scaled --solver agipc-core --agipc-coarse-preconditioner mas32 --agipc-mas-validation gpu --agipc-mas-reuse --cloth-mesh T:/cloth_129x129.obj --framework abd-cemas-srbk --frames 35 --headless --agipc-fine-correction-iterations 10 --agipc-direction-freeze-path S:/perf_diag/agipc_direction_quality_contact300 --agipc-direction-freeze-after-update 300 --metrics-path S:/perf_diag/agipc_direction_quality_intentionally_incomplete.json
E:/Anaconda/envs/DL/python.exe reports/agipc_reproduction/analyze_accepted_direction.py perf_diag/agipc_direction_quality_contact300 --output perf_diag/agipc_direction_quality_contact300_analysis.json
E:/Anaconda/envs/DL/python.exe reports/agipc_reproduction/analyze_accepted_direction.py perf_diag/agipc_direction_quality_contact300 --geometry-only --bbox-diagonal-squared 4.57 --dt .01 --output perf_diag/agipc_direction_quality_contact300_stop_and_mapping.json
E:/Anaconda/envs/DL/python.exe reports/agipc_reproduction/plot_direction_groups.py perf_diag/agipc_direction_quality_contact300 --frame-index 31 --output-stem perf_diag/agipc_direction_quality_contact300_groups
```

Release 构建 exit0；核心自检 exit0、`passed=true`。可执行文件 SHA-256：`924C41530D62A31377E1D6A337C8FAFEBAE8C2F395160F9CD9547BCBC4DFAB81`。布料 SHA-256：`2B2F1EE3279DD38EE778694AEC5386A101C77EAE7F1A18D020C9E9AB816017F8`。

重要 CLI 拒绝检查：`--solver stiffgipc --headless --agipc-direction-freeze-path ...` 在场景初始化前 exit2，未创建捕获目录；这一入口只用于明确启用的 AGIPC-Core 诊断。

构建日志：`perf_diag/agipc_direction_quality_build.log`；核心自检：`perf_diag/agipc_direction_quality_selftest.json`；捕获日志：`perf_diag/agipc_direction_quality_capture.log`。冻结二进制的逐文件 SHA-256 由 CPU 分析记录。

本轮运行期间 GPU 达到 100% 利用率，显存接近 8GB；所有耗时仅记录诊断过程，不用于性能结论。

## 捕获与独立验证结果

场景捕获 exit0，实际为**第 31 帧内、第 300 次 Galerkin 更新**。第 30 帧已完成，采样方向尚未执行 Newton 停止判断、CCD、线搜索或物理位置更新。`agipc_direction_quality_intentionally_incomplete.json` 不存在，符合故意提前停止的设计。

| 项目 | 结果 |
|---|---:|
| 细层 DoF / 唯一 3x3 块 | 49,935 / 98,959 |
| 粗层块 / 唯一块 / FEM 聚合组 | 277 / 1,497 / 261 |
| FEM 活跃 DoF 比例 | 1.640526%（819 / 49,923） |
| GPU 粗层迭代 / 后校正 | 78 收敛 / 10 达到上限，采用 `post_corrected` |
| 完整细层直接解真实相对残差 | 3.3381253e-13 |
| 粗层直接解真实相对残差 | 1.3767759e-14 |
| 完整矩阵 / 右端项 Galerkin 相对误差 | 3.6871880e-15 / 4.2721535e-15 |
| FEM 矩阵 / 右端项 Galerkin 相对误差 | 1.0756408e-14 / 4.2721535e-15 |
| GPU 延拓与独立 CPU 重建相对误差 | 0 |
| GPU/CPU 初末真实残差相对差 | 0 / 0 |
| GPU 候选与 CPU10 方向相对差 | 6.2218671e-15 |
| GPU 粗层解与粗层直接解相对方向差 | 0.0900061% |
| 独立参考门槛 | **通过，分析 exit0** |

最初的通用 LU 分解未完成，工作内存持续增长至约 2.7GB，已停止本任务自己启动的参考进程（退出状态 -1，不是数值验证结果）。随后采用上述对称模式、单线程 BLAS/OMP 配置完成参考；直接分解与求解记录为 12.0762 秒，L/U 合计 18,654,126 个非零项。两次均使用相同 H/g；这只是诊断工具的求解配置调整，不能算 AGIPC 或 GPU 加速。

本轮成功参考命令执行前设置 PowerShell 环境变量：`$env:OPENBLAS_NUM_THREADS='1'`、`$env:OMP_NUM_THREADS='1'`。成功日志为 `perf_diag/agipc_direction_quality_cpu_symmetric_analysis.log`；旧通用参考日志 `perf_diag/agipc_direction_quality_cpu_analysis.log` 不包含成功结论。

## 方向质量与误差来源

以下均与**同一细层 H/g 的直接解**比较。预测下降为 gᵀd−½dᵀHd，是该线性化二次模型的下降，未包含线搜索或实际非线性能量评估。

| 方向 | 真实相对残差 | 欧氏方向误差 | H 能量范数误差 | 相对直接解预测下降 |
|---|---:|---:|---:|---:|
| 原始 GPU 延拓 | 约 1.279 | 99.757457% | 99.045320% | 1.900247% |
| 精确粗层解的延拓 | 约 1.279 | 99.757573% | 99.045319% | 1.900248% |
| 实际采用的 GPU10 | 0.961881 | 98.334084% | 95.132264% | 9.498523% |
| 独立 CPU10 | 0.961881 | 98.334084% | 95.132264% | 9.498523% |
| 独立 CPU20（预算消融） | 0.811702 | 93.725688% | 88.339978% | 21.960483% |

1. **装配及已测后校正实现与 CPU 一致，采用率却不能代表方向质量。** GPU10 与 CPU10 几乎相同，因此本样本没有支持“GPU 后校正计算错误”的证据。真实残差确实从 2.5433043e-6 降到 1.9131699e-6，减少约 24.78%；但相对完整细层直接解，98.33% 欧氏误差、72.84° 夹角和只有 9.50% 预测下降表明它是很弱的近似方向。全细层参考的预测下降为 8.5885476e-10，GPU10 为 8.1578520e-11，绝对量也一并保留。
2. **此样本的主要损失来自粗空间，粗层迭代精度不是主要来源。** GPU 粗层解本身只偏离粗层直接解约 0.09%；精确粗层解的延拓仍有 99.76% 欧氏误差和 99.05% 能量范数误差。改善 MAS 迭代/局部诊断不会改变给定粗空间的表达上限。
3. **少数巨大仿射组覆盖了绝大多数节点。** 261 个组的中位规模为 1；6 个仿射组的规模分别为 7,162、4,116、4,101、800、146、42，共 16,367 个节点，占 98.353464%。这些组占 GPU10 方向平方误差的 96.489340%。独立逐组最小二乘得到的最佳欧氏粗空间拟合误差仍为 **93.869739%**，直接支持当前粗空间无法充分表达该细层方向，而不是仅有一个不准确的粗层迭代解。
4. **翻倍后校正预算并未解决这一样本。** CPU20 提高预测下降至 21.96%，但方向误差仍有 93.73%。保留默认 10 步；该消融不能支持直接改成 20 步，更不能作为速度提升证据。

组分布/误差图已生成并目视检查：[PNG](<E:/university_class/代码收集&杂项/课程作业/暑期实验/实验十三/code/Stiff-GIPC-c499-performance/perf_diag/agipc_direction_quality_contact300_groups.png>)、[PDF](<E:/university_class/代码收集&杂项/课程作业/暑期实验/实验十三/code/Stiff-GIPC-c499-performance/perf_diag/agipc_direction_quality_contact300_groups.pdf>)。左图显示巨大聚合组覆盖的区域；右图为 GPU10 与细层直接方向之差的节点范数，采用对数色标。坐标使用**静止 X/Z 平面**，不是采样时的变形布料形状；误差是该次线性方向误差，不是 35 帧终态位置误差。

这是单个接触阶段冻结系统的结果，说明此时的粗空间/有限后校正效果很弱。它尚不能判定这种现象是论文算法本身的近似代价，还是当前判据历史、保护传播、warp-hash 层级收缩等实现与论文的偏差；也不能直接证明它造成上一轮 2.46746% 终态差异。

## Newton 停止敏感性

日志给出 `bboxDiagSize2=4.570000`，dt=.01，对应位移门槛约 2.1377558e-5。该 bbox 来源只有六位小数，因此按诊断近似输入解释。

| 方向 | 最大 FEM 节点方向 | 相对门槛 |
|---|---:|---:|
| 完整细层直接解 | 2.7409474e-4 | 12.8216 倍 |
| 实际 GPU10 | 7.5306222e-5 | 3.5227 倍 |
| 原始延拓 | 4.9222346e-5 | 2.3025 倍 |

三者的 FEM 部分均高于门槛；本样本**没有**展示“采用方向通过停止判断、直接解不通过”的分叉。此扫描只检查 FEM 范数，未重建混合 ABD 的世界空间方向；且采样提前停止，完整运行的 Newton 判定尚未执行。仍需在真实停止附近记录方向/状态检查点。

## 接下来做什么

- 暂停速度优化、对称 Hessian/BVH 的论文整合与速度结论。当前独立参考通过，但该单样本候选方向接近细层解的质量不足；上一轮长接触数值等价问题也仍未解决。
- 优先追查 6 个大组：核对增量 Green 应变历史重置/累积、保护边在层级收缩中的传播，以及合并规则是否跨过应保留的细层变化。保留论文阈值和默认后校正预算。
- 现有冻结数据的误差分布图与组统计已完成；下一次必要捕获补充当前细层位置、保护边/判据值和层级映射记录，使大组形成过程可重放。先缩小疑点，再做一次针对性捕获，避免盲目重复完整场景。
- 后续用少量稀疏状态检查点定位 Jacobi/MAS 的首次非线性分叉；新局部诊断失败类别的实际触发和失败矩阵 CPU/GPU 对照仍待验证。

完整数值与逐文件哈希：`perf_diag/agipc_direction_quality_contact300_analysis.json`；组分布/停止敏感性：`perf_diag/agipc_direction_quality_contact300_stop_and_mapping.json`。这些诊断产物不纳入 Git。

## 判据追踪后续结果（2026-09-15）

后续一次显式判据快照已独立重建当前 Green 张量并逐边回放原因位/标签，Green 相对误差 `8.20e-15`、边标签零不一致。默认阈值下的允许边图本身包含 14,379 节点的全局连通分量，因此没有证据把上一样本的大组归因于 Green 或标签计算错误。运行时 32-lane 递归映射在自适应标签下停于更细的局部稳定分区；强行按全局连通分量合并反而提高保存直接方向的最佳拟合误差。

后续捕获的 H/g 与本报告样本哈希不同，方向误差也从 98.33% 变为 87.55%。相同 update300 不是跨运行确定性状态，不能直接配对比较。完整判据、映射与阈值分析见 `AGIPC_CRITERION_SNAPSHOT_ANALYSIS.md`；下一步改为同一运行内的稀疏状态检查点。

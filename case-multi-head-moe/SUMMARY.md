# multi-head-moe（SLA2）适配工作总结（给人看的版本）

> 周期：2026-11-05 – 2026-11-11。模型：multi-head-moe（SLA2Attention + Multi-Head-MoE，
> MAGI-2 preview 风格，text-only）。
> 一句话结论：**模型迁入 titan030 仿真栈并全绿；SLA2 算子建模入库算子建模仓 opm
> （operator-performance-models，本地 `/home/l30079348/opm-dev`，gdn-sla2@0dd983e）
> 并推送；黄金参照法发现并修复捕获测试全部漏掉的真实架构 bug；owner 保真裁定后恢复源码实现 +
> 融合开关（默认关）；sim-platform 端到端 + 真机实测对比完成，唯一实质算子缺口为 repeat_interleave
> （已给出修复方案与影响量化）。**

## 1. 五阶段脉络

### 阶段一：迁移模型代码到仿真仓库（样例 `02-model-migration.md`）

- 参考实现从未运行成功，含 3 个真实 bug（RoPE 广播 / einsum 下标 / 布局转置），修复经 owner 批准保留；
  pristine 原版存档 `_mhoe_scratch/multi-head_moe_ORIGINAL.py`。
- 结构映射：(head,expert) 展平复用 common `RoutedExperts`；EP2 换 AllToAll dispatcher；
  TP/CP/MTP fail-fast；两处 plain 参数给 Replicate sharding；preset `multi_head_moe_debugmodel`。
- debug 几何 63,395,104 参数 / 139 张量，公式精确一致。

### 阶段二：算子建模入库 opm 仓（样例 `03-op-modeling.md`）

- 三轮架构指示收敛：simulator 自建 handler → 只认算子建模仓 → 目标仓 opm-dev + DV120 真实硬件。
- SLA2 = 库里没有才新写的实例：语义 4 族 + adaptor 4 个 + roofline 4 个（闭式 cube + 声明式 vec，
  `max(cube+vec, memory)+head_tail`），9 文件 +724，commit `0dd983e` 推送 gdn-sla2 分支。
- 三个源契约（grad 规范名 `Sla2*Backward` / dy 末尾 / linear 标量=补集块数）由"失败→复现→对源码核实"确立。

### 阶段三：自我验证（样例 `04-self-verification.md`）

- 捕获测试最终 **14/14**（保真整改后重写参数化）；opm-dev **1793** 全绿 + mypy strict + ruff 全绿。
- **黄金参照法发现 TransformerBlock 残差丢失 bug**（第二残差接回 x_BLD 而非 h；15 个捕获测试
  全部无法发现）——修复后 logits 1.7e-6 / 123 组梯度 ≤7.6e-5。无基准模型必做黄金校准。
- CPU dispatch 对照、内存 oracle（253,580,416 字节精确一致）、EP2 通信字节逐节点对账、
  48 存量失败经临时 worktree 证实为基线预存（零回归）。

### 阶段四：审视与修改（样例 `05-review-revisions.md`）

- 三轮评审闭环 + owner **保真裁定**："必须和我给你的源码一模一样"——恢复源码实现（RoPE/RMSNorm/
  缩放/分支序列逐字还原）+ 双版本 + `SimulationConfig.enable_sla2_fusion`（默认 False）+
  MoE GMM 例外（bincount/where 无法 Meta 捕获，探针证实）。
- 整改增量 9 文件 +348/−306 未提交未推送（待指示）。

### 阶段五：sim-platform 端到端 + 真机实测对比（样例 `06-sim-vs-real-diff.md`）

- sim-platform 链路跑通（toml + `run_sim.py` 导入手术），L1 导出 85 归一化算子族、
  end_to_end_us=4519（F 1980 / B 1738 / O 800）。
- 真机：新建 `mhoe-real` 容器（同镜像 8 卡）+ scipy 回退 1.13.1 修复，EP2/DP2/seq128 跑通。
- 对比：主计算算子双侧语义一一对应；**唯一实质缺口 repeat_interleave**（复合算子在 Meta 捕获下
  分解为零成本 Expand，真机执行直连 ACLNN 核 64 kernel/步 = 0.33% 设备时间，修复三件套已给方案）；
  确认真机为 bf16 混精度（口径修正）。

## 2. 关键决策记录（后续接手者须知）

| 决策 | 内容 | 依据 |
| --- | --- | --- |
| 保真铁律 | 源码一模一样；融合仅限两 seam 且默认关；例外三项（GMM/3 bugs/初始化）经批准 | owner 裁定（2026-11-10） |
| MoE 专家计算 | 两种模式均 grouped-GEMM | bincount 无 fake impl、where 数据依赖形状——Meta 不可捕获（探针实测） |
| 建模只认 opm 仓 | simulator 自建 handler 属待废弃路径 | 需求方架构指示 |
| 硬件档 | DV120 真实规格（只翻译不标定），不沿用 qwen3.5 的 npu_910b3 标定档 | 需求方指示 |
| 黄金测试入仓 | 忽略（涉及 CPU 结果对比不予执行） | owner 指示；校准脚本留 `_mhoe_scratch/` |

## 3. 跨芯片性能对比口径（需求：同一模型在不同芯片上比性能）

"口径"即对比时遵循的统一约定：

- **对比走 sim-platform 链路**——端到端耗时由平台 DES 输出（样例 end_to_end_us=4519）、
  offload 传输耗时由算子建模仓的 host-device transfer model（`6b7a871` 起）覆盖，
  这些能力平台/算子建模仓已具备，**无需再开发**。（仅当单独使用原 simulator、不走 platform 时，
  L2 `SchedulePlan.duration_est` 未填充、无端到端输出。）
- **不需要**标定/DVFS 校准——跨芯片是相对对比；各芯片各用其真实规格档（如 DV120），
  "只翻译不标定"。
- 报告中先声明已知偏差：① 复合算子缺口（repeat_interleave，见上）；② cast 粒度差异；
  ③ dtype（真机 bf16 vs 仿真捕获）。
- AC（full/selective）覆盖完整；EP2 通信位置：20 a2a + 11 RS（stream 3）；fsdp-hidden=0 为
  bucket 聚合的正常表现，非异常。

## 4. 仓库 / 分支 / 提交 / 推送记录

| 仓 | 分支 | 提交/状态 | 推送 |
| --- | --- | --- | --- |
| simulator（worktree `mhm-wip-wt`，正仓 `torchtitan-npu-simulator030`） | `multi-head-moe-wip` | `5747770` 基线 | ✅ fork https://github.com/luying0612/torchtitan-npu-simulator 分支 `030-multiheadmoe-1109`（token 推送） |
| 同上（保真整改+开关） | 同上 | 9 文件 +348/−306 | ❌ 未提交未推送（待指示） |
| opm-dev `/home/l30079348/opm-dev` | `gdn-sla2` | `0dd983e`（分支后被并行会话推进至 `21a35b6`） | ✅ https://github.com/ZhanluModelSim/operator-performance-models 分支 `gdn-sla2`（SSH 推送） |
| torchtitan-npu-real（同 simulator 仓的隔离 clone @70a29ba） | — | 模型包 + examples **untracked** | ❌ 未版本化（可复现性需要时再提交，待指示） |

注：titan030 主线已前进到 `9c4454e`（dsv4.1 rope 对齐），worktree 分支基于 `fca58c5`——
合入主线需先 rebase。

## 5. 待办与合入路径

1. mhm-wip-wt 9 文件：提交 → `multi-head-moe-wip` rebase 到 `titan030`（9c4454e）→ 合入 → 推送（待指示）。
2. torchtitan-npu-real untracked 两目录：需要实测可复现性时提交（待指示）。
3. repeat_interleave 修复三件套（register_fake + op_mapping + opm cost handler，不动模型源码）——待 owner 决策。
4. 若做耗时级对比：仿真侧切 bf16 混精度重捕获。
5. 超边界未做：`aclnnBlockSparseAttention` microbench 锚点校准、`dv120-topo.txt` 通信链路模型、L2cache。

## 6. 复验命令速查

```bash
# 捕获测试（容器 titan030-qwen35，期望 14/14）
docker exec -w /workspace/mhm-wip-wt/experiments/simulator -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  titan030-qwen35 bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
  python -m pytest tests/test_multi_head_moe_capture.py -q"

# opm 仓测试（容器内 /workspace/opm-dev，期望全绿；host 编辑后需 docker cp 重同步）
docker exec titan030-qwen35 bash -c "cd /workspace/opm-dev && python -m pytest tests/ -q"

# 端到端 CLI（静默成功，PYEXIT=0 即通过）
python -m torchtitan_simulator --config multi_head_moe_debugmodel

# sim-platform 端到端（导入手术见 06 §6.2）
python run_sim.py   # mhoe-platform-run/

# 真机实测（容器 mhoe-real；profiler active 步 = 3、7）
cd /workspace/torchtitan-npu-real && NGPU=2 bash examples/multi_head_moe/mhoe_debugmodel_moe_2p_128.sh \
  --profiler.enable-profiling --profiler.profile-freq 4 --profiler.profiler-warmup 2 \
  --profiler.profiler-active 1 --profiler.save-traces-folder <dir>
```

注意：容器 `/workspace` bind-mount 宿主机 `/home/l30079348/titan030-qwen3.5/`（文件改动双向即时生效）；
opm-dev 例外（host 编辑后需 `docker cp` 重同步——发生过容器内副本被并行会话回退的事故）。

## 7. 产物索引

| 产物 | 路径 |
| --- | --- |
| 仿真 toml + 导入手术 + L1 导出 | `titan030-qwen3.5/mhoe-platform-run/` |
| 实测 kernel CSV / 聚合 / 对比报告 | `titan030-qwen3.5/mhoe-real-bench/` |
| 实测运行脚本 | `torchtitan-npu-real/examples/multi_head_moe/`（untracked） |
| 黄金校准/探针脚本 | `titan030-qwen3.5/_mhoe_scratch/` |
| 原始文档逐字归档 | 本目录 `archive/` |

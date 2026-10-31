# FSDP2 通信与计算掩盖分析 —— DeepSeek V4 实测（torchtitan-npu）

> 面向同事的学习总结：如何在 8 卡 NPU 上用 torch profiler 定位 FSDP2 通信问题，
> 以及针对 DSV4 的一组受控实验（E0–E5）的完整结论。
>
> 分析日期：2026-10-29

## 0. 环境与版本（重要：结论对应此版本）

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/hicann/torchtitan-npu |
| 分支 / Commit | 实验分支 `real-dsv4`，**与 `master` 完全一致**（同一 commit `2374945f4f04871f1019403f0dcdb7301d1e66ef`，2026-10-20，"[refactor](dsv4): delete the golden attention override"；`git rev-list --count master..real-dsv4` = 0）。即：**本分析的代码基线 = master @ `2374945f`** |
| 未提交改动 | `models/deepseek_v4/__init__.py`、`config_registry.py`（新增 `deepseek_v4_flash_4layers_8experts` 调试配置，仅 E3 用到；E0/E1/E2/E4/E5 均为 master 原生配置，不受影响） |
| torch / torch_npu | 2.14.0+cpu / 2.14.0.dev20260918 |
| CANN Toolkit | 9.2.0（V100R001C12SPC001B237），MindStudio profiler 组件随包 |
| NPU | 8 × Ascend 910B3 64GB（Atlas A3 节点），驱动 25.5.2，固件 7.8.0.7.220 |
| 容器 | `titan-real-dsv4`（镜像 `torchtitan-npu-muon:9.0.0-py3.11`），仓库挂载 `/home/l30079348/titan-real ↔ /workspace` |
| 模型 | DeepSeek V4 flash：原生 43 层 / dim 4096 / 256 专家（top_k=6, moe_inter 2048）；实验用 16 专家版 ≈ **29–30B**（原生 256 专家 ≈ 290B，bf16 分片后 72.5GB/卡，61GB 卡放不下，故减专家） |

---

## 1. Profiling 采集的正确姿势（含踩坑）

### 1.1 启动命令模板

```bash
docker exec -d titan-real-dsv4 bash -c "source /usr/local/Ascend/cann/set_env.sh && \
cd /workspace/torchtitan-npu && NGPU=8 \
HF_ASSETS_PATH=/workspace/assets/deepseek_v4_tokenizer \
CKPT_SAVE_LOAD_PATH=/workspace/ckpt_prof \
bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh \
  --compile.no-enable \
  --training.steps 5 --training.global-batch-size 8 \
  --parallelism.expert-parallel-degree 4 \
  --profiler.enable-profiling \
  --profiler.extension.profiler-start 3 --profiler.extension.profiler-end 4 \
  --profiler.extension.profile-ranks 0 1 2 3 4 5 6 7 \
  --profiler.save-traces-folder <name> --dump-folder /workspace/prof_x"
```

### 1.2 三个坑

1. **`profile_ranks` 默认只落 rank0**（默认值 `[-1]` 实际行为异常），做跨卡分析必须显式传 `0..7`。
2. **`--profiler.extension.profile-with-stack / profile-with-memory` 严禁在多卡全开**：
   实测把 profiled step 从 19s 拖到 83.6s（4.4 倍），host 被采集拖垮后所有集合通信算子 95% 时间在等 peers，
   产生"假快慢卡"和"假零掩盖"（见 E0）。需要 Python 栈时只对个别 rank 开。
3. **原始产物是二进制**，ELF 版 `msprof --export=on` 会误判为 App 模式直接跑失败。
   正确导出方式（CANN 9.2）：

```bash
MSP=/usr/local/Ascend/ascend-toolkit/latest/tools/profiler/profiler_tool/analysis/msprof/msprof.py
python3 $MSP import -dir <...>_ascend_pt          # 解析出 device_0/sqlite/*.db + host/sqlite/hccl.db
python3 $MSP export summary -dir <...>_ascend_pt --format csv   # 生成 op_summary_*.csv 等
```

关键数据源：`op_summary_*.csv`（全部设备算子：名称/类型/start/duration/queue wait）、
`hccl_single_device.db`（HCCL 算子字节数+起止）、`host/sqlite/hccl.db`（逐 rank 通信字节数，全 rank 可用）。

### 1.3 采集策略结论

**8 卡 + 轻量采集（无 stack/memory）**，开销≈0（E2 实测 10.0s ≈ 无采集 10–11s）。
跨卡偏斜分析必须 8 卡都在。

---

## 2. FSDP2 做了哪些通信（结合代码）

代码位置：`torchtitan_npu/extensions/torchtitan/distributed/fsdp.py`（apply_fsdp_to_decoder）。

| 通信 | 次数/步 | 数据量/步 | 作用 | 语义要点 |
|---|---|---|---|---|
| allGather | 178 | 7.25GiB | FSDP2 参数 unshard：每卡广播自己的 1/8 分片，**每卡各自重建该层完整参数副本**（临时，用完释放） | fwd + 反向重算 fwd + bwd 各一次；参数本体始终分片，从不聚合到单卡 |
| reduceScatter | 90 | 2.86GiB | 反向把该层完整参数的**梯度均值**按分片切回 8 卡（同时完成 1/8 缩放） | 作用对象是梯度不是参数；优化器直接更新本地分片 |
| alltoallv | 561 | 25.1GiB | EP 的 MoE token dispatch/combine（**不属于 FSDP2**，`models/deepseek_v4/token_dispatcher.py`） | dispatch→专家计算→combine 结构性串行 |
| allReduce | 7 | 标量 | grad-norm 全局范数 / 裁剪系数同步 | count=1 的标量；主梯度同步是 RS 不是 AR |

注意：E0 曾被误记为"4 层 8 专家"，实际是 **43 层 16 专家完整 flash**（日志 `n_layers: 43` 为准）。
43 层是 flash 原生层数（`_deepseek_v4_flash: n_layers=43`），该配置只减了专家 256→16。

### 2.1 预取机制（为什么 AG 能被掩盖）

`fsdp.py:272-306`：EP 开启时显式设置 **1 层 lookahead** 预取——
forward 中每层提前 all-gather 下一层参数（`set_modules_to_forward_prefetch`），
backward 提前 gather 前一层（`set_modules_to_backward_prefetch`）。
预取与层数无关（4 层与 43 层效果同量级），只要求 host 能提前把 AG 发出去。

### 2.2 与 MindSpeed-LLM 的对比（fsdp2 路径，源码查证）

**MindSpeed 前向 MoE 流程**（megatron MoE + `expert_fully_shard_parallel.py`）：
router → shared_experts（dense MLP，稠密 mesh）→ permute → **a2a dispatch（EP 组）** →
experts 模块（GroupedMLP）：**efsdp AG**（权重在 efsdp 轴切分——默认 `Shard(1)` 切 F 维，
可配 shard_by_dim_0；`fully_shard(expert, mesh=(ep,ep_fsdp))`）→ GMM → **a2a combine** →
unpermute。即 **dispatch 与 GMM 之间插了一次 efsdp 权重 AG**（efsdp 组内，payload =
本地专家权重的一半量级）；反向对称（efsdp RS + edp 梯度除数 `gradient_divide_factor`）。
prefetch：`num_to_forward/backward_prefetch` 默认 1（与本仓库 fsdp.py:272-306 同款 1-unit
lookahead）。mesh 展平 `('edp','efsdp','ep')` → efsdp 对 = 连续对 {0,1}/{2,3}…（ep-major），
与本仓库实测的 {0,4} 型（efsdp-major）不同。

**三条路径**：

| | MindSpeed fsdp2 | 本仓库 eager（fsdp.py:174-183，E8 实测） | 本仓库 spmd（本次实测） |
|---|---|---|---|
| 专家权重 | ep 组内 4 完整专家 × **efsdp 切 F 半**（默认 efsdp=1 不切 = 4 完整专家；开启 = 2 专家当量） | edp mesh 上 Shard(0) 切 dim0(E)；**E8 实测通信结构与 spmd 完全一致**（AG 178/RS 90，无额外单元级 AG） | **4 完整专家/rank，efsdp 复制** |
| 用前通信 | efsdp AG（2 卡，~小；默认关则无） | 与 spmd 相同的 AG 骨架（仅 mesh 不同） | 无 |
| a2a 组 | 4 卡（ep 组） | 4 卡（ep 组，`token_dispatcher.py:209 ep_mesh.get_group()`） | 4 卡（ep 组，实测） |
| 优化器状态 | 分散式优化器 + ep 除数 | 标准 FSDP2 分片 | flex_shard 按 ('ep','efsdp') 8 路切 E + 组间 RS/AG |
| 选择方式 | — | spmd-backend 未设 spmd_types | `--parallelism.spmd-backend spmd_types`（dsv4 调试脚本默认） |

**开 EP 的收益（两仓库同理）**：ep=1 时专家权重虽 8 路切（2 专家/rank），但每次 GMM 前
要 AG 全量 16 专家（805MB bf16 × 43 层 × 3 相位 ≈ 100GB/step）——EP 让专家驻留、
零权重 AG，用 +202MB/rank 驻留换掉 100GB 流量与 805MB 瞬时缓冲。

### 2.3 专家切分与 EP/efsdp mesh（实测更正版，E2 rank0）

模型维度（GMM 实测反推）：D=4096，moe_intermediate F=2048，16 专家，ep=4，efsdp=2；
8 卡展平 **rank = efsdp×4 + ep**（efsdp 对 = {0,4}/{1,5}/{2,6}/{3,7}，由 2 卡集合通信配对实测确认）。

**计算布局（GMM 实测 shape 为准）**：

| 参数 | 全局 shape | 切分后每 rank | 证据 |
|---|---|---|---|
| routed w1/w3 (E,F,D) | (16, 2048, 4096) | **(4, 2048, 4096)** | GMM 258 次 |
| routed w2 (E,D,F) | (16, 4096, 2048) | **(4, 4096, 2048)** | GMM 129 次（43层×3相位） |
| dense（attention 等 2D 权重） | 如 (2048, 4096) | (256, 4096)（FSDP2 切 dim0，8 卡） | 97× 8 卡 AG |
| shared expert + gate | 3×(2048,4096) ≈ 50MB | (512,4096)×3（ep 组 4 卡切） | 38× 4 卡 AG，payload≈47MB |

要点：
- **E 维只被 ep 切（4 路）**：每 rank 独占 4 个**完整**专家（F/D 全维在位）⇒ 前向/反向
  **专家权重无 AG、梯度无 RS**（对比 dense FSDP 每层要 AG 重建完整参数）。
- **efsdp 轴不切 compute 权重的任何一维**（hidden D=4096、F=2048 全维在位；若切 hidden
  GMM 应为 (4,1024,4096)，实测否定）。efsdp 是 **EP 组间复制轴（HSDP 式）**：8 个 DP rank
  分成 2 个独立 EP 组（{0-3} 与 {4-7}），**各组持有全部 16 专家（4 完整专家/rank），服务组内
  4 个 rank 的 token**（a2a 组局域性实测：rank0 的 a2a 与 rank1 重叠 200/200、与 rank4 仅
  85/200 为墙钟巧合）——两组处理不同数据，计算零冗余。
- **efsdp 的作用在优化器状态**：状态按 ('ep','efsdp') 8 路切 E（2 专家状态/rank，fp32 状态
  减半）；组间对（{0,4} 等）承担跨组梯度均值与权重/状态再同步——**实测每 step 数次、
  时间戳散布在反向后段（grad-norm AR 之前，即优化器更新之前）**，RS ~19-25ms、AG ~3-6ms，
  且 AG→RS 常相邻成对（疑似重分片的 gather/scatter 对，AG 的确切语义未定位）。
  专家数 < ep×efsdp 时该布局有 Shard(1)/BlockShard 兜底（`fsdp.py:180` eager 路径同款逻辑）。
- **可扩展性**：E 只需被 ep 切（≤256 路），多出的卡全部充当组间副本——
  1024 卡 = ep256 × efsdp4（每 rank 1 个完整专家 + 4 组副本）完全成立，
  不存在"专家不够切"的问题。
- **sparse 4 卡 AG 是预取**（shared expert/gate）：在 attention 期间发出，完成于 router
  前（AG 结束→下一个 TopkV2 中位间隔 3.79ms，min 0.04ms），94% 被 attention 掩盖；
  若"用到才发"则 +14ms×43层 全暴露。
- **MoE 层参数的三种待遇**：dense 部分（attention 块）8 卡 FSDP（8 卡 AG）；shared
  expert+gate（~50MB，payload 实测吻合）在 EP 组 4 卡 mesh 上切（EP 轴 4 份，组内 AG，
  efsdp 轴复制，随 sparse 域一起组间同步）；routed experts EP 驻留无 AG。
- **与 MindSpeed-LLM 的 efsdp 同名不同义**：MindSpeed fsdp2 路径（`expert_fully_shard_parallel.py`）
  的 efsdp 是**真·专家 FSDP**——`fully_shard(expert, mesh=(ep, ep_fsdp))`，默认
  `Shard(1)`（切 F 维，可配 'shard_by_dim_0'），权重真切、用前 AG；本仓库 v4 spmd
  路径的 efsdp 是组间复制轴（计算权重不切，优化器状态切）。本仓库 eager 路径
  （`fsdp.py:174-183`）与 MindSpeed 同款（efsdp×ep 在 dim0 切，efsdp_ep_size >
  num_experts 时降 Shard(1)）。

按参与集合分类的通信（union 口径，8 卡平均，busy/掩盖率）：

| 类别 | AG | RS | a2a |
|---|---|---|---|
| 8 卡（dense FSDP） | 0.52s / 63%* | 0.58s / 54%* | —（分类噪音） |
| 4 卡 ep 组（sparse FSDP / dispatch+combine） | 0.59s / **94%*** | 0.30s / 80%* | ~2.4s / **11%*** |
| 2 卡 efsdp 对 | 0.03s / 77%* | 0.10s / 38%* | — |

*带星号为 rank0 数值；8 卡平均见 §4.2。**efsdp 对内通信每 step 仅 ~25ms**，
MoE 段的暴露通信几乎全部来自 a2a（EP token 移动）。

---

## 3. 实验矩阵与结果

统一条件：8 卡，FSDP=8，EP=4，TP/PP=1，GBS=8（mbs=1），seq 4096，5 步，采 step3，`compile` 关闭。

| 实验 | 参数 offload | 优化器卸载 | 采集 | step 窗口 | 总掩盖率(ov/comm) | AG 掩盖 | RS 掩盖 | a2a 掩盖 | 设备空隙(>1ms) | 快慢卡 |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | ✓ | ✓ | **重型(stack+mem)** | 83.5s | 0.3–9.3% | 8–9% | 5–13% | **0%** | — | **严重**（0–3 号卡每次集合晚到 58–275ms） |
| E1 | ✓ | ✓ | 轻量 | 21.4s | 20–31% | 49–67% | 51–64% | 0–2% | 2078 个/7.04s | 无 |
| E2 | ✗ | ✗ | 轻量 | 10.0s | **55–60%** | **81–92%** | **69–77%** | 22–33% | 8 个/0.03s | 无 |
| E3 (4L) | ✗ | ✗ | 轻量 | 1.09s | 43–50% | 60% | 54% | 21–32% | — | 无 |
| E4 =E0+强制负载均衡 | ✓ | ✓ | 重型 | 83.0s | 1.3–10.9% | — | — | — | — | **仍在**（与 E0 同模式） |
| E5 =E2+优化器卸载 | ✗ | ✓ | 轻量 | 14.8s | 33–41% | **82%（不变）** | 72%（不变） | ~30% | 442 个/3.52s | 无 |
| E7 =E2+AdamW（非 Muon） | ✗ | ✗ | 轻量 | 15.6–18.2s/步 | — | — | — | — | 显存重试 6–8 次/步 | 无 |
| E8 =E7+eager 路径（partial_dtensor） | ✗ | ✗ | 轻量 | 14.5–17.4s/步 | — | — | — | — | 显存重试 6–8 次/步 | 无 |
| E9 =E8+Muon（消停顿） | ✗ | ✗ | 轻量 | 9.8–12.2s/步 | — | — | — | — | 0 重试 | 无 |

E7/E8/E9 补充（优化器栈与路径对照，**实测**）：
- **通信结构与优化器、路径无关**：AG 178 / RS 90 / AR 7 在 E2/E7/E8 完全一致，A2A 561 在
  E2/E9（同为 Muon、同窗口口径）**精确一致**；**eager（partial_dtensor）与 spmd 产生相同的
  集合通信序列**——此前"eager 专家会有额外单元级 AG"的推测被否定（per-param mesh 复用
  同一 AG/RS 骨架，仅 mesh 不同）。
- **AdamW vs Muon 的代价是状态内存**：AdamW m+v（2×fp32）→ 显存分配重试 6–8 次/step
  （Muon 为 0）→ 宿主停顿 → 步时 10.7s → 15.6–18.2s（**+46–70%**），RS/A2A 单实例
  膨胀至 max 2.7s（peer-wait）。
- spmd vs eager（同 Muon，E2 vs E9 干净对比）：中位时长 AG 5.70→5.64ms、A2A 4.77→4.69ms、
  RS 16.5→13.2ms，步时一致。分组分布主体一致（8r AG ~100 主导 + 4r AG + 少量 2r + 噪声）；
  唯一超出方法噪声的信号：eager 的 2r AG/RS 偏多（13–14 vs 5）、8r RS 偏少（16 vs 32）——
  疑似 eager 逐参数单元在 edp 子组上归约而 spmd 融合成 8r，**待字节级验证，不断言**。

E4 补充：强制 round-robin 路由后 a2a 字节差 7%→0.12%，但快慢卡模式原样保留 →
**快慢卡与负载均衡无关**（真实路由下 a2a 字节不均衡本来只有 7%，hash 层确定性路由 + `load_balance_coeff=1e-3` 辅助损失 + expert_bias 自适应在起作用）。

---

## 4. 核心结论

### 4.1 E0 的异常全部是采集 artifact

重型采集拖垮 host → 算子发射间隔拉大 → 各卡到达集合通信的时间偏斜 →
通信算子 95% 时间在等 peers（AG 真实传输 ~5ms/次，慢卡上显示 95ms）→
伪快慢卡（0–3 号卡 host 慢：NUMA/采集落盘争用）+ 伪零掩盖。
**换轻量采集后一切恢复正常，训练栈本身没有问题。**

### 4.2 无干扰时（E2）FSDP2 通信是健康的

10s 窗口的临界路径分解（8 卡平均）：

| 构成 | 时间 | 占比 |
|---|---|---|
| 暴露的计算（有效功） | 4.73s | 47.3% |
| **暴露的 a2a（EP 结构性）** | **2.18s** | **21.9%** |
| 暴露的 RS | 0.43s | 4.3% |
| 暴露的 AG | 0.15s | 1.5% |
| 暴露的 AR | 0.03s | 0.3% |
| 空闲 | 0.07s | 0.7% |

通信总量与掩盖率（union 口径，8 卡平均；暴露 = 总量 × (1−掩盖率)）：

| 通信 | 总时间（busy） | 掩盖率 | 暴露时间 |
|---|---|---|---|
| allGather（FSDP 参数） | 1.12s | **77%** | 0.26s |
| alltoallv（EP token） | 3.05s | **28%** | 2.18s |
| reduceScatter（梯度） | 1.55s | 66% | 0.53s |
| allReduce（grad-norm） | 0.03s | **0%** | 0.03s |

- AG 81–92% 被预取掩盖、RS 69–77% 与后续反向重叠——**FSDP2 不是瓶颈**
- 唯一的大损耗是 **a2a 的结构性暴露**（dispatch 完成才能算专家、combine 完成才能出结果）
- 等待 vs 带宽：E2 中 AG 时长的 32%、RS 的 11%、a2a 的 46% 是"等最慢 rank"；
  单次传输只有几 ms、消息 8–66MB，未跑满 HCCS——**等待是主，带宽不是当前矛盾**
- **AR 的 0.03s 既是总量也是暴露量**：实测 7 个 AR 实例与计算重叠全部为 0ms、
  与其他通信重叠也为 0ms——grad-norm AR 必须等所有 RS 完成，**完全无法掩盖**（8 卡总量
  0.007–0.044s，avg 0.026s；快的 rank 在 AR 里等慢的 rank，等待计入 AR 时长但仍 100% 暴露）

### 4.3 两种卸载的影响模式完全不同

| | 参数 offload（FSDP2 cpu_offload） | 优化器卸载（swap_optimizer/novaswap） |
|---|---|---|
| step 时间 | 10.0 → 21.4s（+114%） | 10.0 → 14.8s（+48%） |
| host 空隙 | 0.03 → 7.04s | 0.03 → 3.52s |
| **FSDP2 AG 掩盖** | **82% → 49%（被破坏）** | **82% → 82%（不变）** |
| 作用位置 | 前向/反向层间（每层参数回存/回灌 + event 同步插在预取管线上） | **只在优化器段**（RS 之后的更新循环） |
| 显存收益 | 50.8 → 18.0GiB | 50.8 → 39.0GiB |

**确切结论：优化器卸载不影响 FSDP2 通信。**
E5 的 AG/RS 时长与掩盖率和 E2 完全一致，前向/反向分段逐毫秒不变。
优化器段 +4.84s 的构成：**3.51s 纯 host 空隙（与通信重叠 0ms、与计算重叠 0ms）**
+ ~0.9s 亚毫秒空隙累积 + 0.24s 段内残余 EP a2a 等待微涨（次生）+ AR +9ms。
设备算子本身零变化（Baddbmm/Matmul 数量相同；Muon 的 Newton-Schulz 两种配置都在 NPU 上跑，
卸载的是 AdamW 状态分桶与 momentum）。

空隙的固定位置：`ApplyAdamWV2`/`Axpy`/`Baddbmm` 之后等 `ForeachAddScalarV2`/`InplaceLerps`——
即 novaswap 桶切换的 `swap_api.wait_for_device_release` 阻塞 + 裁剪系数按桶重缩放（`rescale_pending`）提交。
这是"纯尾巴"：只把下一步整体推后，不侵蚀前向预取管线。

参数 offload 的机制（为什么会打断 FSDP2）：每层 unsharded 参数用完 D2H 回 CPU pinned、
反向用前 H2D 回来（3248 次拷贝/步），拷贝本体仅 0.11s，但每次提交的
pinned 池分配 + event record/wait（`staging.py: TransferHandle.wait` / `stream.wait_event`）
插在 77k 个算子发射之间；`fsdp.py:272` 注释自己也承认 "D2H syncs in EP could interfere
with implicit prefetching in FSDP"。

### 4.4 判定方法（可复用）

1. **临界路径分解**：窗口 = 暴露计算 + 暴露各类型通信 + 空隙；区分"有用功"和"损耗"。
2. **等待 vs 带宽**：同一集合通信实例跨 rank 配对（按类型+顺序），
   `单卡时长 − 全 rank 最短时长 ≈ 等 peers`；剩余与消息量/时长对照带宽。
3. **空隙归因**：对每个 >1ms 空隙检查期间是否有通信/计算 kernel 在跑——
   E5 的 3.5s 空隙重叠均为 0ms → 纯 host 停顿，一锤定音。
4. **负载均衡核查**：host 侧 `hccl.db` 的 HCCLOP.count 按 rank 汇总 a2a 字节量。
5. **字节反推模型规模**：AG 总字节 = 分片参数量 → 全模型参数 = ×8/2B。

---

### 4.5 激活卸载对 FSDP2 通信的影响（r 系列实测，2026-10-31）

在 4L/8exp 缩小版（4 卡，EP2/dp_shard4，Muon，GBS8）上用 MindSpeed 式块级
`saved_tensors_hooks`（`override/common/act_offload.py`，流量忠实/显存中性）做了
0 → 1.88 → 8.32 → 10.54 GB/步 的卸载量扫描（详见
`verify_act_offload/OFFLOAD_ANALYSIS.md` §3.6）：

- **通信结构完全不变**：AG 13f/31b、RS 24、A2A 16f/50b、AR 7 在所有卸载档计数一致；
  compute busy 持平（1.50–1.55s）。激活卸载不改变 FSDP2 的通信算子集合，只影响时序。
- **受影响算子 = 反向 ReduceScatter**（梯度分片回写）：med 25.7 → 39.4（+53% @1.88GB）
  → 47.1ms（+83% @10.54GB），max 至 179ms；busy 仅在 10.54GB/步跳变 0.66→1.57s（+138%）。
- 前向 A2A med 微增 ≤9%（2.75→3.00ms）；AG/AR 无感。
- **步时近线性 +0.05~0.08 s/GB，无悬崖**；通信显著劣化临界量 ≈ 9–10GB/步
  （本配置 = 每层前向卸载 ~1.3GB 边界激活的 100% 场景）。
- 机理（推断）：unpack H2D 与侧流事件插在 bwd 计算间，推迟梯度就绪 → RS 等 peers
  拉长；超过 bwd 可藏窗口后等待实质占用通信流。
- 对 43L 的外推（推断）：只卸边界张量 ≈2.4GB/步，落在"被掩盖"档，预计步时 +5~10%。

判定方法沿用 §4.4（等待 vs 带宽、空隙归因）；本次的增量判据是
**"单次集合通信时长膨胀但 busy 持平 = 被掩盖的劣化；busy 跳变 = 实质劣化"**。

---

## 5. 优化建议（按收益排序）

1. **采集规范**：全 rank 只开设备侧采集；stack/memory 仅个别 rank、仅在需要看 Python 栈时。
2. **a2a 重叠**（最大结构损耗 22%）：尝试 `moe_comm_backend` 非 standard 后端 /
   dispatch 分块与计算重叠 / 双 micro-batch（TBO）思路。
3. **offload 决策**：显存够（本配置 50.8/61GiB）就全关，是无损收益；
   显存紧张优先只开优化器卸载（代价集中在前向反向之外的尾巴），避免参数 offload 破坏预取。
4. **优化器卸载调优**（若必须用）：加大常驻桶预算减少 `wait_for_device_release`、
   合并按桶重缩放提交、桶搬运与 NS 计算流水化。
5. **batch 效率**：E2 有效计算仅 ~12% MFU（GBS=8 单核太小，60k 个小向量算子），
   加大 batch/seq 让计算稠密是另一个正交方向。

---

## 6. 附录：产物与脚本

| 路径 | 内容 |
|---|---|
| `titan-real/prof_out2/profiling_dsv4_ep4_v2/` | E0 原始 8 rank 数据（15GB，重型采集） |
| `titan-real/prof_e1|e2|e3|e4|e5/` | E1–E5 各组 8 rank 数据 |
| `titan-real/scripts/run_e{1..5}.sh` | 实验启动脚本 |
| `titan-real/scripts/exp_export.sh` | 批量 import + export summary |
| `titan-real/scripts/fsdp2_overlap.py` | 总掩盖率/快慢卡对比（接受数据 glob 参数） |
| `titan-real/scripts/fsdp2_comm_overlap.py` | 按通信类型拆分掩盖率 |
| `titan-real/scripts/fsdp2_skew.py` | 跨 rank 集合通信到达偏斜 + 大空隙定位 |
| `titan-real/scripts/e2_deepdive.py` / `e2_refine.py` / `e2_bw.py` | 字节量/负载均衡/带宽 vs 等待 |
| `titan-real/scripts/e5_phases2.py` / `e5_phases3.py` / `e5_gapwait.py` | 前向/反向/优化器分段 + 空隙归因 |
| `titan-real/logs/e{1..5}_*.log` | 各实验训练日志（含 config 打印） |

### 复现命令

```bash
# 训练 + 采集（以 E2 为例）：见 scripts/run_e2.sh
docker exec -d titan-real-dsv4 bash /workspace/scripts/run_e2.sh
# 导出 + 分析
docker exec titan-real-dsv4 bash /workspace/scripts/exp_export.sh prof_e2
docker exec titan-real-dsv4 python3 /workspace/scripts/fsdp2_overlap.py \
  "/workspace/prof_e2/e2_nooffload/profiling_data/*_ascend_pt/PROF_*/mindstudio_profiler_output"
docker exec titan-real-dsv4 python3 /workspace/scripts/e5_phases3.py   # 前向/反向/优化器分段
```

注意事项：连续多轮异常退出可能造成 NPU 显存泄漏（无进程持有但显存不释放），
处理方式为重启容器后 `npu-smi set -t reset -i 0 -c <chip>`（会重启整个 HCCS fabric，需确认 8 卡空闲）。

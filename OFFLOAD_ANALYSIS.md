# 激活卸载与优化器卸载：机制、实测与框架对比

> 实测环境：titan-real-dsv4 容器（CANN 9.2.0 / torch 2.14 / torch_npu，Ascend 910B3，8 卡共享）。
> 测量脚本与原始数据：`titan-real/verify_act_offload/`（`bench_bw_sync.py`、`bench_critical.py`、`bench_bw_sync.json`、`bench_critical.json`）。
> 框架卸载代码分析基于 torchtitan-npu `real-dsv4` 分支与 MindSpeed-LLM 当前 checkout。

---

## 1. 结论摘要

| 问题 | 结论 |
|---|---|
| 当前框架有无激活卸载 | **无原生实现**（激活靠 FullAC 重计算）；PyTorch 2.14 图级路线（`torch._functorch._activation_offloading`）在本容器 NPU 上验证可用 |
| 优化器卸载有无同步点 | 有。CPU-canonical 档每个参数一次 H2D→算→D2H 往返 + `step()` 末尾 `staging.wait()`（host 同步）；NPU-resident 档仅参数往返，无 host 同步 |
| 传输能否藏进前/反向 | 能，但约束不是"总 fw/bwd 时间"，而是**每个张量 last-use 之后的剩余计算时间**。等价 MindSpeed 块边界释放语义（`sink_wait=True`）下：卸载量 ≤ fw 时间×19GB/s（本测试模型 ≈0.39GB）时完全隐藏 |
| 是否增加端到端耗时 | 超过临界卸载量后线性增加：本测试模型边际代价 ≈ **+70%/GB（双向）**；DSV4 真实场景（层计算 ~34ms、边界张量 56MB）远低于临界值，可完全隐藏 |
| 有效带宽 | D2H ≈ **18.9 GB/s**、H2D ≈ **20.2 GB/s**（256MB 以上饱和）；1MB 小块仅 3.4–3.8 GB/s（延迟主导）；pageable 与 pinned 带宽相同 |
| 同步开销 | NPU 平台同步原语本身很贵：event record+wait 对 **124µs**、wait_stream 对 **140µs**、单次 ao.offload 发射 **218µs**、pinned 新分配 **223µs**（池化后 0.24µs）；host 侧 event.synchronize 往返（HybridDeviceOptimizer 模式）64MB D2H 共 **4.1ms** |
| 完全暴露时的瓶颈 | 大块：PCIe 单向 ~19–20GB/s 物理墙；小块/多事件：**host 侧同步原语与发射开销**（每次 120–220µs）主导，等效带宽骤降到 <10GB/s |
| 与 MindSpeed-LLM 的显著区别 | ① MindSpeed 有 eager 原生激活卸载且与重计算联动筛选，torchtitan-npu 只有重计算；② 优化器卸载哲学相反：MindSpeed 在 CPU 上执行更新（只传 grads/params），torchtitan-npu 在 NPU 上执行更新（CPU-canonical 档每步往返 momentum+params）或 NPU 常驻；③ MindSpeed 按 block 边界释放+单一 wait_stream，天然避开 torch 图 pass 的 sink_wait 两难 |

---

## 2. 当前框架（torchtitan-npu）的卸载机制

### 2.1 优化器卸载（三档容器，`override/common/optimizer.py`）

`swap_optimizer` override 按 `--training.enable-cpu-offload` 分流：

| 档位 | 数据布局 | 每步传输 | 同步点 |
|---|---|---|---|
| `CpuOffloadOptimizersContainer`（CPU canonical，列出 swap_optimizer 时） | 参数/梯度/动量 pinned CPU | **每参数**：H2D momentum→scratch → NPU 计算 → D2H 回写（`cpu_offload_muon.py:637-725`）；参数同样往返 | `step()` 末尾 `self._staging.wait()` = **host event.synchronize**；`zero_grad` 同样 wait |
| `CpuOffloadNpuStateOptimizersContainer`（默认，仅开 enable-cpu-offload） | 动量 NPU 常驻 | 仅 CPU 参数 H2D、更新后 D2H 回写，动量零往返 | 无 host 同步（事件链） |
| `OptimizerStateSwapContainer`（NovaSwap，未开 cpu-offload） | 动量按 FlexShard bucket swap | bucket 粒度 D2H/H2D，prefetch_plan 前移 | bucket 边界 stream-order |

底层 `CpuStaging`（`extensions/cpu_offload/staging.py`）：4-slot pinned bounce ring + 专用传输流 + 单线程 pageable-commit executor；`TransferHandle.wait()` 是唯一的 host 同步入口，稳态全部事件链（`wait_on(stream)` 无 host 同步）。梯度累加（`grad_accum.py`）为 64MB 分块 H2D→add→D2H 双缓冲流水，`grad_offload_event` 事件链保证同 slice 先写后读。clip 在 NPU 侧（`GradientClipChannel`），避免 host norm。

### 2.2 FSDP2 参数/梯度 offload（数据面，`CPUOffloadPolicy`）

`apply_fsdp_to_decoder` 挂 `CPUOffloadPolicy()`：sharded 参数 pinned CPU 常驻，forward 前 H2D（多 rank 上游内建、单 rank 本仓预取补丁），backward 后 RS 结果 D2H 落盘并由 NPU 分块累加。`PinnedFlatPool` 统一 pow2 chunk 分配。

### 2.3 激活卸载（现状 + 本次验证的路线）

- 现状：**无**。激活内存由 FullAC 全量重计算管理（`config_registry.py:313`），官方文档明确"AC 管激活，offload 管参数/梯度/状态"。
- 本次验证补上的路线：PyTorch 2.14 `torch._functorch._activation_offloading`（图级 pass，与后端无关）。`joint_custom_pass` 给大激活打 `should_offload` 标记 → partitioner 在 fw/bw 图插入 `ao.offload/ao.wait_tensor/ao.reload`（每设备一条专用 `torch.Stream` + data_ptr 事件注册表 + `pinned_memory_pool` 固定内存池）。**policy 要点：必须排除 placeholder（参数/输入），否则 keepalive 语义会释放权重存储，第二步即崩（实测复现）**。
- 验证结果（aot_eager，8 块 MLP H2048×seq8192 bf16）：24 对 offload/reload 插入、梯度逐位一致（rel err=0.0）、多步稳定；fw 峰值 2.91→0.91 GiB（-69%，`sink_wait=False`）。
- 已知限制：本容器 inductor 有两个与 offload 无关的既有错配（`make_reduction` 缺 `strict_sum`、C++17/20 头文件），任何含归约的编译都失败；offload pass 本身与后端无关，镜像对齐后可直接接 inductor。

---

## 3. 实测数据

### 3.1 有效带宽（pinned，专用流，GB/s）

| 方向 | 1MB | 4MB | 16MB | 64MB | 256MB | 1GB | 2GB |
|---|---|---|---|---|---|---|---|
| D2H | 3.8 | 15.0 | 18.1 | 18.5 | **18.9** | 19.0 | 18.3 |
| H2D | 3.4 | – | 19.4 | – | **20.2** | 18.9 | – |

- **平台墙 ≈ 单向 19–20 GB/s，64MB 起饱和**；1MB 块被延迟主导（3.4–3.8 GB/s）。
- pageable 与 pinned 同速（1GB：18.9 / 20.2）——本平台 pinned 不增益带宽，增益在异步语义。
- `ao.offload+wait_tensor+reload+wait_tensor` 完整往返 256MB：25.4ms（双向聚合 20.1 GB/s）——custom op 机制在大块下不增加开销。
- `CpuStaging.submit_d2h` 256MB（源重填、验证过内容）：约 13–19 GB/s，比裸 copy_ 多几 ms 事务开销（producer 事件 + 校验）。

### 3.2 同步与发射开销（host 侧，单次）

| 原语 | 开销 |
|---|---|
| `torch_npu` Event record + wait_event 对 | **124µs** |
| wait_stream 双向对 | **140µs** |
| 单次 `ao.offload` 发射（dispatch+池分配+enqueue） | **218µs** |
| pinned 32MB 新分配 | **223µs**（池复用 0.24µs，差 1000 倍） |
| 64MB D2H + host `event.synchronize()`（HybridDeviceOptimizer 模式） | **4.1ms**（其中传输 3.4ms） |

推论：本平台同步原语比 CUDA 贵约一个数量级，**每次事件/流同步 ≈ 100–200µs**——细粒度（小张量、每层多事件）卸载的 host 开销下限由它决定；`activation_offload_cpu_gpu_bw=50`（PyTorch prefetch 用的默认带宽假设）比实测 19–20GB/s 高 2.5 倍，前移量会被低估。

### 3.3 隐藏能力与临界卸载量（单步，128MB 粒度，min-of-7）

基线（无 offload）：fw 20.5ms / bwd 43.6ms / step 64.1ms。理论隐藏预算 = 阶段时间 × 19GB/s：**fw 0.39GB、bwd 0.82GB（reload-prefetch 可前移）、合计 1.21GB**。

| 卸载量 | `sink_wait=False`（wait 原地，早释放，fw 峰 -69%） | `sink_wait=True`（wait 沉图末，全隐藏，fw 峰不降） |
|---|---|---|
| 0.25GB | +16.5% | **+0.8%（全隐藏临界点）** |
| 0.50GB | +31.8% | +14.3% |
| 0.75GB | +47.4% | +34.3% |
| 0.88GB | +54.8% | +48.2% |
| 1.00GB | +69.1% | +66.1% |

三条规律：

1. **`sink_wait=True` 的临界卸载量 ≈ 0.25–0.39GB**，与理论 fw 预算吻合；0.5GB 起 D2H(26ms) > fw(20.5ms) 开始暴露。
2. **`sink_wait=False` 没有全隐藏点**：wait 留在 offload 原地把计算流串行化到该 D2H 完成，实际约束是"该张量 last-use 之后的剩余计算"——本模型每块仅 2.6ms，可藏 ~48MB < 128MB 粒度，故最小粒度也暴露。这是图级 pass 的结构性两难：早释放 vs 不串行化。**MindSpeed 的块边界释放设计（§4.1）等价于把 wait 沉到每块末尾，兼得两者**——理想集成应做 per-layer/分块 compile，或在 pass 中支持块边界 sink。
3. 边际暴露斜率 ≈ **+70%/GB（双向）**，即每卸载 100MB 约增加 3.5–4.5ms/步（两次传输 @ ~20GB/s，部分与计算重叠后）。

### 3.4 端到端耗时影响

- 本测试模型：0.25–1.0GB 卸载 → step +0.8%…+66%（见上表）；aot_eager 下另有每 op ~218µs 的 python dispatch 开销（inductor 下为图内节点，可忽略）。
- DSV4 实测佐证（E 系列 profiling）：E1（参数/梯度/优化器全 offload + FullAC）19s/step vs E2（无 offload）10–11s/step，tflops 19.1 vs 35.9；损失主要来自 **host 发射干扰**（设备空转 0.07s→27–46%、a2a 重叠 30%→0%、allReduce 等待 +4.8s），而非带宽——8 rank 累计 offload 事务使 host 发射速率下降，集合通信退化为等 peers。
- DSV4 激活卸载预算推算：E2 每 rank 每步设备忙 ~1.48s（7.4s/5 步），43 层 → 单层 fwd+bwd ≈ 34ms；每层可藏 D2H ≈ 34ms×19GB/s ≈ **0.65GB**。FullAC 下只需卸每层边界张量 seq4096×h7168×bf16 ≈ **56MB**（43 层共 2.4GB/步），远低于预算 → **DSV4 场景激活卸载（边界张量）可完全隐藏，且节省的 HBM 可换 micro batch 或关掉部分重计算**。
- 优化器卸载的"可藏性"最差：优化器 step 是串行阶段，CPU-canonical 档每参数的 H2D→算→D2H 往返只能藏进 step 内部（算子间隙），不能藏进前反向；NPU-resident 档动量零往返、参数往返与 grad-clip/step 流水重叠，是当前框架的推荐档。

### 3.5 完全暴露时的瓶颈

- 传输 ≥64MB：PCIe 单向 19–20GB/s 物理墙（双向全双工聚合 ~39GB/s）。
- 传输 <16MB：**host 侧发射与同步原语主导**（124–218µs/事件），等效带宽 <10GB/s 且随粒度下降；一次 64MB 传输的 host sync 往返固定 ~0.7–1ms。
- 多张量多事件累积：24 个 128MB 块 ≈ 24×(124µs 同步 + 218µs 发射) ≈ 8ms/步纯 host 开销——粒度越细越贵，**MindSpeed 每块一个 wait_stream 的粗粒度设计在本平台上更经济**。

### 3.6 DSV4 真实模型实测：激活卸载量 vs FSDP2 通信（r 系列，2026-10-31）

**实现**（`torchtitan_npu/override/common/act_offload.py`，MindSpeed 式）：类级 patch `DeepSeekV4TransformerBlock.forward`，块级 `saved_tensors_hooks`（pack：事件+侧流异步 D2H→pinned；unpack：H2D→原张量，双流+事件序）。pack 返回 `(k,shape,dtype)` 载荷使 checkpoint 计数归零、recompute 不触发；`store` 持有原张量 → **流量忠实、显存中性**（本次只测通信/耗时影响，不验证 HBM 节省）。模块树/FSDP2/spmd 结构零改动。

**实验设计**：4L/8exp 缩小版，4 卡，EP2/dp_shard4，Muon，GBS8（2 microbatch × 4 层 = 8 block-fwd/步）；梯度累积下每步卸载量由 MIN_MB/PACK_N 控制。对照 r0m = 钩子全挂+透传（0 卸载，含 recompute 的结构对照）。

| 卸载量 (GB/步) | 0 (r0m) | 1.88 (r2c) | 8.32 (r1r) | 10.54 (r3n) |
|---|---|---|---|---|
| 步时 (s, 稳态) | 2.43–2.47 | 2.52–2.58 | 3.04–3.10 | 3.11–3.18 |
| RS med / max (ms, bwd) | 25.7 / 62.9 | 39.4 / 66.9 | 40.4 / 53.9 | **47.1 / 179** |
| RS busy (s/步) | 0.66 | 0.72 | 0.71 | **1.57** |
| A2A fwd med (ms) | 2.75 | 2.90 | 2.96 | 3.00 |
| AG med f/b (ms) | 5.55/5.79 | 5.66/5.83 | 5.62/5.58 | 5.62/5.76 |
| 通信计数 f/b | AG 13/31、RS 24、A2A 16/50、AR 7 —— **四档完全一致** ||||
| compute busy (s) | 1.55 | 1.53 | 1.50 | 1.50 |

**回答四个问题**：

1. **哪些通信算子受影响**：只有**反向 ReduceScatter**（FSDP2 梯度分片回写）显著受影响；前向 A2A（MoE dispatch）微增 ≤9%；AG（参数 all-gather）与 AR 无感。通信**计数与结构在所有卸载量下完全不变**。
2. **前向还是反向**：RS 100% 在反向（busy f/b = 0/0.66–1.57）；受影响的窗口是 bwd。
3. **减小卸载量是否缓解**：是，且近线性——步时 +0.05~0.08 s/GB（1.88GB→+5%、8.32GB→+25%、10.54GB→+27%），无突变悬崖；RS busy 1.57→0.72→0.66 递减。
4. **临界卸载量**：**通信显著劣化点 ≈ 9–10GB/步**（RS busy +138%、RS max 179ms）；≤8.3GB/步时 RS busy 与基线持平（劣化被计算/通信重叠掩盖）。单次 RS 时长劣化更早出现（<2GB/步即 med +53%）但同样被掩盖。纯传输对照：8.32GB/步 = 双向 16.6GB ≈ 0.87s @19GB/s，实测步时仅 +0.63s → 大部分传输被侧流掩盖，与 §3.3 预算一致。

**机理（推断）**：unpack 的 H2D 恢复流量与侧流事件插在 bwd 计算之间，推迟梯度就绪 → RS 等待 peers 拉长（单次时长膨胀）；超过 bwd 内可藏窗口后（~9-10GB/步），等待实质占用通信流 → RS busy 跳变。

**对照 43L 全量的外推（推断，未实测）**：§3.4 的"只卸边界张量"方案 ≈ 2.4GB/步，落在 r2c（1.88GB）档附近 → 预计 RS med +~50%（被掩盖）、步时 +5~10%。43L 实测因外部任务占卡未执行（`scripts/run_e10.sh` 就绪）。

**诚实备注**：① loss 轻微偏移 ≤0.1（r3n 起点 12.316 vs r0m 12.228），未发散，机制未定（数值扰动或 run 间波动）；② 每层打包体积不均（470–1040MB/层·fwd，层内容差异，未深究）；③ r0v3（更早 vanilla 基线，步时 4.1–4.7s）受外部任务干扰不可作基线，以 r0m 为准；④ r2m（0.94GB/步，EVERY_N 模式）混入 3 层 recompute，不进主曲线；⑤ 采集窗口=1 个完整 step（AG 44/A2A 66/RS 24/AR 7 完整）。

---

## 4. MindSpeed-LLM 的卸载实现

### 4.1 FSDP2 异步激活卸载（`fsdp2/features/async_offload.py`，eager）

- 机制：`saved_tensors_hooks`（pack/unpack）+ `async_save_on_cpu(h2d_stream, d2h_stream, block_idx, depth)` 按 block 包装前向；qwen3_moe 实际接线为每层一个上下文、`custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr()` —— **只卸每层边界 hidden_states**，单条双向 offload 流。
- pack：为每个张量现场 `pin_memory` 分配 CPU buffer，D2H 异步；**下一 block 开始时 `wait_stream` + 对上一块 GPU 张量 `resize_(0)`**（块粒度释放，等价 per-block sink_wait）。
- unpack：H2D 异步，`working_stream.wait_stream(h2d_stream)` 保证正确性；**固定 1-block lookahead 预取**（`prefetch_get` 提前加载上一块的张量）。
- 稳态无 host 同步（全部 stream-order）；参数天然被 `base_check_fn` 排除（Parameter 检查）。
- 策略文档明确启发式：**计算贵、传输小的张量 offload（并跳过对应重计算，如 attention）；激活大、计算便宜的走重计算**——offload 与重计算是同一决策的两面。

### 4.2 mcore 优化器卸载（`HybridDeviceOptimizer`，来自 Megatron core）

- 按 `optimizer_offload_fraction` 把参数分成 GPU/CPU 两组：CPU 组 grads D2H（专用流）→ **host `d2h_event.synchronize()`** → **CPUAdam(fused) 在 CPU 上执行更新**（fp32 主参数）→ params H2D 回写（optimizer hook + 事件序）。
- 选项：`overlap_cpu_optimizer_d2h_h2d`、`pin_cpu_grads/params`。FSDP2 路径（`fsdp2/optim/`）**没有**优化器卸载（原生 AdamW/Muon 在 NPU）。

---

## 5. 显著区别对比

| 维度 | torchtitan-npu | MindSpeed-LLM |
|---|---|---|
| 激活卸载存在性 | 无原生（FullAC 重计算）；本次验证补 PyTorch 图级路线 | FSDP2 eager 原生（saved_tensors_hooks），mcore 路径另有配套 |
| 激活策略 | offload 与重计算正交，各自开关 | **同一决策两面**：按"计算贵传输小→offload，反之重计算"筛选（custom_check_fn），可跳过 attention 重算 |
| 集成粒度 | 图 pass + 标记 policy，模型零侵入，但依赖 compile | 模型代码显式 per-block 包装（侵入 qwen3_moe），eager 可用 |
| 传输流 | 每设备一条专用流，事件注册表（data_ptr 键） | 单条双向 offload 流 + 逐块 wait_stream |
| GPU 释放时机 | `sink_wait` 二选一：早释放（计算流被串行化）或沉图末（峰值不降，全模型 compile 下） | **块边界释放**（下一块开始时 resize_0），天然兼得早释放与不串行化 |
| 预取 | reload 按估算传输时间前移（bw 可配，默认 50GB/s 偏高） | 固定 1-block lookahead |
| host 内存 | `pinned_memory_pool`（按 (numel,dtype) 复用）/ PinnedFlatPool | 每张量现场 pin_memory（~223µs/次，无池） |
| 参数保护 | 需 policy 排除 placeholder（踩过坑） | `base_check_fn` 检查 Parameter 天然排除 |
| host 同步点 | CpuStaging `wait()`（step/zero_grad 边界）；图路线无 | 稳态无；HybridDeviceOptimizer 有 per-group `d2h_event.synchronize()` |
| 优化器卸载哲学 | **状态 CPU、数学在 NPU**（CPU-canonical 档每步 momentum+params 往返）或**动量 NPU 常驻**（默认，零往返） | **数学在 CPU**（CPUAdam fused），状态全 CPU，只传 grads(D2H)+params(H2D)，momentum 不迁移 |
| 优化器卸载粒度 | 容器级三档 + NovaSwap bucket | fraction 混合（按参数比例切分 GPU/CPU 优化器） |

**三点本质差异**：

1. **激活卸载的成熟度与形态**：MindSpeed 是现成的 eager 实现，且把 offload/recompute 做成统一决策；torchtitan-npu 走 FullAC 重计算，激活卸载需要外接（PyTorch 图路线已验证可行，但需 policy 正确性——排除 placeholder、处理 sink_wait 权衡）。
2. **优化器卸载方向相反**：MindSpeed 把计算搬到 CPU（PCIe 流量 = grads+params，momentum 不动，适合 host 算力空闲的大规模场景，代价是 host 同步与 CPU 算力）；torchtitan-npu 把计算留在 NPU（CPU-canonical 档 PCIe 流量 = params+momentum 双倍往返，NPU-resident 档最小）。**传输量上 NPU-resident 档 < MindSpeed CPU-optimizer < torchtitan CPU-canonical 档**。
3. **同步设计哲学**：MindSpeed 以块为生命周期单位（块边界释放、单 wait_stream、固定 lookahead），粗粒度但与 NPU 的贵同步原语匹配；torch 图 pass 以张量为单位（keepalive/事件注册表），细粒度灵活但暴露了 sink_wait 两难与 per-op 发射开销。

---

## 6. 建议

1. DSV4 若上激活卸载：优先边界张量（FullAC checkpoint 入口，~56MB/层），块粒度生命周期（对齐 MindSpeed 语义），可完全隐藏传输并腾出 ~2.4GB/步 HBM；eager 路线可直接借鉴 `async_offload.py` 的块包装，compile 路线需 per-layer compile 或给 pass 加块边界 sink。
2. 优化器卸载维持 NPU-resident 默认档；CPU-canonical 档仅在 host 空闲且 PCIe 预算充足时使用，且应仿 MindSpeed 把"动量驻留 CPU、只传 grads"作为第三形态评估（省一半 PCIe 流量）。
3. `activation_offload_cpu_gpu_bw` 应校准为实测 19–20GB/s，否则 prefetch 前移量不足。
4. 卸载粒度 ≥64MB（带宽饱和下限），避免 <16MB 的小块（延迟与同步原语主导）。

## 附：产物

- `verify_act_offload/test_e2e.py` / `probe_min.py`：激活卸载功能验证（ correctness + 峰值）
- `verify_act_offload/bench_bw_sync.py` / `bench_bw_sync.json`：带宽与同步开销
- `verify_act_offload/bench_critical.py` / `bench_critical.json`：临界卸载量扫描（两条 sink_wait 曲线）
- 本文档：`verify_act_offload/OFFLOAD_ANALYSIS.md`

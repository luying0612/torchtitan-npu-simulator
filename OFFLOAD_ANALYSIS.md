# 激活卸载与优化器卸载：机制、实测与框架对比

> 实测环境：titan-real-dsv4 容器（CANN 9.2.0 / torch 2.14 / torch_npu，Ascend 910B3，8 卡共享）。
> 测量脚本与原始数据：`titan-real/verify_act_offload/`（`bench_bw_sync.py`、`bench_critical.py`、`bench_bw_sync.json`、`bench_critical.json`）。
> 框架卸载代码分析基于 torchtitan-npu `real-dsv4` 分支与 MindSpeed-LLM 当前 checkout。

---

## 1. 结论摘要

| 问题 | 结论 |
|---|---|
| 当前框架有无激活卸载 | **无原生实现**（激活靠 FullAC 重计算）；PyTorch 2.14 图级路线（`torch._functorch._activation_offloading`）在本容器 NPU 上验证可用 |
| 优化器卸载有无同步点 | 有（两档都有）。容器级 `step()` 末尾 `staging.wait()`（AdamW 无条件执行，`cpu_offload_adamw.py:233`；Muon `:611-616`）+ `zero_grad()`/`load_state_dict()` 也 wait（`optimizer.py:596,604-607`）——每步 2 个 CPU 停等点；且该平台 `Event.synchronize()` 可能提前返回，实际是"弱停等"，正确性由设备侧流序兜底。换入换出均被预取/事件链覆盖，无 MindSpeed AdamWSwap 那种**块间强制串行**（其换入分块=总量/16，块边界整块 H2D 暴露）；titan 每步传输总量与 MindSpeed 暴露量相当（bf16 状态 14B/元素 vs fp32 状态换入 12B/元素） |
| 传输能否藏进前/反向 | 能，但约束不是"总 fw/bwd 时间"，而是**每个张量 last-use 之后的剩余计算时间**。等价 MindSpeed 块边界释放语义（`sink_wait=True`）下：卸载量 ≤ fw 时间×19GB/s（本测试模型 ≈0.39GB）时完全隐藏 |
| 是否增加端到端耗时 | 超过临界卸载量后线性增加：本测试模型边际代价 ≈ **+70%/GB（双向）**；DSV4 真实场景（层计算 ~34ms、边界张量 ~57MB）远低于临界值，可完全隐藏 |
| 有效带宽 | 大块（≥64MB）D2H ≈ **19–27 GB/s**、H2D ≈ **19–24 GB/s**（两次测量区间，随机器负载波动）；1MB 小块仅 ~4 GB/s（延迟主导）；pageable 与 pinned 带宽相同 |
| 同步开销 | NPU 平台同步原语本身很贵：event record+wait 对 **115–124µs**、wait_stream 对 **127–140µs**、单次 ao.offload 发射 **201–218µs**、pinned 新分配 **220–227µs**（池化后 0.23–0.26µs）；host 同步往返（HybridDeviceOptimizer 式）64MB D2H 共 **3.3–4.3ms**。**平台陷阱**：D2H 后紧接的 `Event.synchronize()` 在空闲流上可能提前返回（§3.2） |
| 完全暴露时的瓶颈 | 大块：PCIe 单向 19–27GB/s 物理墙（随机器负载波动）；小块/多事件：**host 侧同步原语与发射开销**（每次 115–220µs）主导，等效带宽骤降到 <10GB/s |
| 与 MindSpeed-LLM 的显著区别 | ① MindSpeed 有 eager 原生激活卸载且与重计算联动筛选，torchtitan-npu 只有重计算；② 优化器卸载同为"状态常驻 CPU、NPU 执行更新"（AdamWSwap 型 swap），差异在流水粒度（MindSpeed chunk=总量/16 + fused kernel vs torchtitan 逐参数 + 公开算子）与 host 同步（MindSpeed 无 vs torchtitan step 末有）；③ MindSpeed 按 block 边界释放+单一 wait_stream，天然避开 torch 图 pass 的 sink_wait 两难 |

---

## 2. 当前框架（torchtitan-npu）的卸载机制

### 2.1 优化器卸载（三档容器，`override/common/optimizer.py`）

档位由两级选择决定（`optimizer.py:651-661` 类 docstring + `:1045-1052`）：

- 仅开 `--training.enable-cpu-offload` → `CpuOffloadNpuStateOptimizersContainer`（`_offload_states=False`，NPU-resident，`optimizer.py:651-661`）；
- 再显式列出 `swap_optimizer` override → 重 derive 为 `CpuOffloadOptimizersContainer`（`_offload_states=True`，CPU-canonical，`:1045-1052` + `cpu_offload()` `:712-723`）。**E1 实跑的是 CPU-canonical**（`logs/e1_lightprof.log:25784`：`swap_optimizer: CpuOffloadNpuStateOptimizersContainer.Config -> CpuOffloadOptimizersContainer.Config`）。

| 档位 | 数据布局 | 每步传输 | 同步点 |
|---|---|---|---|
| `CpuOffloadOptimizersContainer`（CPU canonical） | 参数/梯度/动量 pinned CPU（状态 `zeros_like(param)` 随参数 dtype，`cpu_offload_adamw.py:126-132`） | **每参数**：H2D 参数+梯度+双动量 → NPU 计算 → D2H 参数+双动量（AdamW，`cpu_offload_adamw.py:278-312`）；Muon 的 D2H 提交走独立 commit 流（`cpu_offload_muon.py:499,672-677`） | AdamW `step()` 末尾**无条件** `staging.wait()`（`cpu_offload_adamw.py:233`，两种模式都有）；Muon step 末 wait（`cpu_offload_muon.py:611-616`）；容器级 `zero_grad()` 与 `load_state_dict()` 也 wait（`optimizer.py:596,604-607`） |
| `CpuOffloadNpuStateOptimizersContainer`（NPU-resident，仅开 enable-cpu-offload 时） | 动量 NPU 常驻（`_prepare_muon_input` 原地更新 `cpu_offload_muon.py:679-687`；AdamW 同理 `cpu_offload_adamw.py:304-305` 注释） | 仅 CPU 参数 H2D、更新后 D2H 回写，动量零往返 | 同上——**step 末 `staging.wait()` 无条件执行，NPU-resident 也有 CPU 停等**；但等待的是参数回写尾巴（传输量小） |
| `OptimizerStateSwapContainer`（NovaSwap，未开 cpu-offload） | 动量按模型分片（part）注册 swap_api，FlexShard bucket 粒度 | D2H/H2D 经 swap_api execute，`prefetch_plan` 钩子提前提交 H2D（`optimizer.py:726-800`） | bucket 边界 stream-order |

底层 `CpuStaging`（`extensions/cpu_offload/staging.py`）：4-slot pinned bounce ring（`:25`）+ 专用传输流（`:98`）+ 单线程 ThreadPoolExecutor 处理 pageable bounce 提交（`:102-104`）；`TransferHandle.wait()` 是 host 同步入口（注意 §3.2 平台陷阱：空闲流上可能提前返回），`wait_on(stream)` 为设备侧事件链（无 host 同步）。梯度累加（`grad_accum.py`）为 64MB 分块（`_CHUNK_BYTES` `:75`）H2D→add→D2H 双缓冲流水（每 dtype 2 个 `_AccumSlot` `:100-107`），`grad_offload_event` 事件链保证同 slice 先写后读（`:195-208`）。clip 在 NPU 侧（`stage_gradient` 携带 `clip_channel`，`cpu_offload_adamw.py:284-291`），避免 host norm。

### 2.2 FSDP2 参数/梯度 offload（数据面，`CPUOffloadPolicy`）

`apply_fsdp_to_decoder` 挂 `CPUOffloadPolicy()`：sharded 参数 pinned CPU 常驻，forward 前 H2D（多 rank 上游内建、单 rank 本仓预取补丁），backward 后 RS 结果 D2H 落盘并由 NPU 分块累加。`PinnedFlatPool` 统一 pow2 chunk 分配。

### 2.3 激活卸载（现状 + 本次验证的路线）

- 现状：**无**。激活内存由 FullAC 全量重计算管理（`config_registry.py:313`），官方文档明确"AC 管激活，offload 管参数/梯度/状态"。
- 本次验证补上的路线：PyTorch 2.14 `torch._functorch._activation_offloading`（图级 pass，与后端无关）。`joint_custom_pass` 给大激活打 `should_offload` 标记 → partitioner 在 fw/bw 图插入 `ao.offload/ao.wait_tensor/ao.reload`（每设备一条专用 `torch.Stream` + data_ptr 事件注册表 + `pinned_memory_pool` 固定内存池）。**policy 要点：必须排除 placeholder（参数/输入），否则 keepalive 语义会释放权重存储，第二步即崩（实测复现）**。
- 验证结果（aot_eager，8 块 MLP H2048×seq8192 bf16）：24 对 offload/reload 插入、梯度逐位一致（rel err=0.0）、多步稳定；fw 峰值 2.91→0.91 GiB（-69%，`sink_wait=False`）。
- 已知限制：本容器 inductor 有两个与 offload 无关的既有错配（`make_reduction` 缺 `strict_sum`、C++17/20 头文件），任何含归约的编译都失败；offload pass 本身与后端无关，镜像对齐后可直接接 inductor。

---

## 3. 实测数据

> **测量方法声明**：所有带宽/耗时数字均以 `torch.npu.synchronize()`（设备级同步）计时的脚本测得（`bench_bw_sync.py`、`bench_critical.py`，`bench_critical.py:54,94,127`）。两次独立运行：**10-30（机器被其他容器占用）** 与 **11-03（空闲）**，下表给出区间。
> **平台陷阱（11-03 发现）**：D2H 拷贝后紧接的 `Event.synchronize()` 在空闲流上可能**提前返回**（256MB 实测 0.2-0.5ms，物理不可能）；在事件链（跨流 wait_event）之后才可靠阻塞。因此①一切计时勿用逐次 ev.sync；②依赖 `event.synchronize()` 做 host 侧完成保证的模式（titan `staging.wait()`、MindSpeed `d2h_event.synchronize()`）在本平台的阻塞强度存疑，正确性实际由设备侧流序保证。另：向同一 pinned dst 重复发起 D2H 并用 ev.sync 计时会出现亚毫秒假象（原因未定位），已弃用该路径数据。

### 3.1 有效带宽（pinned，专用流，GB/s，设备级同步）

| 方向 | 1MB | 4MB | 16MB | 64MB | 256MB | 1GB | 2GB |
|---|---|---|---|---|---|---|---|
| D2H | 3.8–4.0 | 15.0–16.2 | 18.1–25.2 | 18.5–25.7 | **18.9–26.4** | 19.0–26.5 | 18.3–26.6 |
| H2D | 3.4–3.9 | – | 19.4–23.1 | – | **20.2–24.1** | 18.9–24.3 | – |

- 大块有效带宽 **19–27 GB/s，随机器负载波动**（区间下限=10-30 重载，上限=11-03 空闲）；64MB 起饱和；1MB 块被延迟主导（~4 GB/s）。
- pageable 与 pinned 同速（1GB 两次运行均为 22.7–26.0）——本平台 pinned 不增益带宽，增益在异步语义。
- `ao.offload+wait_tensor+reload+wait_tensor` 完整往返 256MB：25.4ms（重载）/ 19.7ms（空闲），双向聚合 20–26 GB/s——custom op 机制在大块下不增加开销（`bench_bw_sync.py:61-77`）。
- `CpuStaging.submit_d2h` 与裸 `copy_` 同构（producer 事件 + copy + complete 事件，`staging.py:204-211`）：设备级同步同构对比实测 13.2 vs 12.8 ms/256MB（11-03），**staging 路径无显著额外开销**；内容校验通过（`track=False` 直传路径）。bench 脚本内的 staging 数字（0.29-0.31ms）因同 dst 复用 + ev.sync 计时落入上述陷阱，已弃用。

### 3.2 同步与发射开销（host 侧，单次，两次运行区间）

| 原语 | 开销 |
|---|---|
| `torch_npu` Event record + wait_event 对 | **114.7–124.2µs** |
| wait_stream 双向对 | **127.4–140.2µs** |
| 单次 `ao.offload` 发射（dispatch+池分配+enqueue） | **200.6–217.6µs** |
| pinned 32MB 新分配 | **220.3–226.7µs**（池复用 0.23–0.26µs，差 ~1000 倍） |
| 64MB D2H + host 同步（HybridDeviceOptimizer 式 host sync 模式） | **3.3–4.3ms**（其中传输 ~2.5–3.4ms） |

推论：本平台同步原语比 CUDA 贵约一个数量级，**每次事件/流同步 ≈ 100–200µs**——细粒度（小张量、每层多事件）卸载的 host 开销下限由它决定；`activation_offload_cpu_gpu_bw=50`（PyTorch prefetch 默认带宽假设，`_functorch/config.py:248`）比实测 19–27 GB/s 高约 2 倍，前移量会被低估。

### 3.3 隐藏能力与临界卸载量（单步，128MB 粒度，min-of-7，设备级同步）

基线（无 offload，两次运行一致）：fw 20.5ms / bwd 43.6–43.7ms / step 64.1–64.2ms。理论隐藏预算 = 阶段时间 × 有效带宽：按 19GB/s 计 **fw 0.39GB、bwd 0.82GB、合计 1.21GB**（若按空闲机 26GB/s 计则 fw 预算抬至 ~0.53GB）。

| 卸载量 | `sink_wait=False`（wait 原地，早释放，fw 峰 -69%；10-30 运行） | `sink_wait=True`（wait 沉图末，全隐藏，fw 峰不降；**两次运行**） |
|---|---|---|
| 0.25GB | +16.5% | **+0.8% / +1.2%（全隐藏临界点）** |
| 0.50GB | +31.8% | +14.3% / +14.5% |
| 0.75GB | +47.4% | +34.3% / +33.9% |
| 0.88GB | +54.8% | +48.2% / +47.9% |
| 1.00GB | +69.1% | +66.1% / +66.1% |

`sink_wait=True` 曲线经两次独立运行（不同机器负载）复现，逐点偏差 <0.5 个百分点——经验临界点可信。三条规律：

1. **`sink_wait=True` 的临界卸载量 ≈ 0.25–0.39GB**（≈ fw 时间 × 有效带宽）；0.5GB 起开始暴露。
2. **`sink_wait=False` 没有全隐藏点**：wait 留在 offload 原地把计算流串行化到该 D2H 完成，实际约束是"该张量 last-use 之后的剩余计算"——本模型每块仅 2.6ms（20.5ms/8 块），按 19–26GB/s 可藏 48–66MB < 128MB 粒度，故最小粒度也暴露。这是图级 pass 的结构性两难：早释放 vs 不串行化。**MindSpeed 的块边界释放设计（§4.1）等价于把 wait 沉到每块末尾，兼得两者**——理想集成应做 per-layer/分块 compile，或在 pass 中支持块边界 sink。
3. 边际暴露斜率 ≈ **+70%/GB（双向）**，即每卸载 100MB 约增加 3.5–4.5ms/步（两次传输 @ 19–27GB/s，部分与计算重叠后）。

### 3.4 端到端耗时影响

- 本测试模型：0.25–1.0GB 卸载 → step +1%…+66%（见上表）；aot_eager 下另有每 op ~200–218µs 的 python dispatch 开销（inductor 下为图内节点，可忽略）。
- DSV4 实测佐证（E 系列 profiling，`logs/e1_lightprof.log` / `logs/e2_nooffload.log` 及 `prof_e1/` `prof_e2/` 8-rank op_summary 聚合）：E1（参数/梯度/优化器全 offload + FullAC）19s/step vs E2（无 offload）10–11s/step，tflops 19.1 vs 35.9；损失主要来自 **host 发射干扰**（设备空转 0.07s→27–46%、a2a 重叠 30%→0%、allReduce 等待 +4.8s），而非带宽——8 rank 累计 offload 事务使 host 发射速率下降，集合通信退化为等 peers。
- DSV4 激活卸载预算推算：E2 每 rank 每步设备忙 ~1.48s（7.4s/5 步，prof_e2 op_summary 聚合），43 层 → 单层 fwd+bwd ≈ 34ms；每层可藏 D2H ≈ 34ms×(19–27GB/s) ≈ **0.65–0.9GB**。FullAC 下只需卸每层边界张量 seq4096×h7168×bf16 ≈ **57MB**（43 层共 ~2.5GB/步），远低于预算 → **DSV4 场景激活卸载（边界张量）可完全隐藏，且节省的 HBM 可换 micro batch 或关掉部分重计算**。
- 优化器卸载的"可藏性"最差：优化器 step 是串行阶段，CPU-canonical 档每参数的 H2D→算→D2H 往返只能藏进 step 内部（算子间隙），不能藏进前反向；NPU-resident 档动量零往返、参数往返与 grad-clip/step 流水重叠，是当前框架的推荐档。

### 3.5 完全暴露时的瓶颈

- 传输 ≥64MB：PCIe 单向 19–27GB/s 物理墙（随机器负载波动）。
- 传输 <16MB：**host 侧发射与同步原语主导**（115–218µs/事件），等效带宽 <10GB/s 且随粒度下降；一次 64MB 传输的 host sync 往返固定 ~0.8–1.8ms。
- 多张量多事件累积：24 个 128MB 块 ≈ 24×(115–124µs 同步 + 200–218µs 发射) ≈ 7.5–8.2ms/步纯 host 开销——粒度越细越贵，**MindSpeed 每块一个 wait_stream 的粗粒度设计在本平台上更经济**。

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

### 4.2 优化器状态卸载（AdamWSwap：`SwapDistributedOptimizer` + `swap_adamw_step`，`--swap-optimizer`）

来源：MindSpeed core 包 `mindspeed/core/optimizer/swap_optimizer/swap_optimizer.py`，经 `features_manager/optimizer/swap_optimizer_feature.py` 以 patch 启用（替换 `DistributedOptimizer` 类 + patch `AdamW.step = swap_adamw_step`）。参数 `--swap-optimizer-times`（默认 16）控制 chunk 粒度。核心语义正是"**平时驻留 CPU，step 时按 chunk 搬回 NPU 计算，算完搬回 CPU**"。

- **布局**：fp32 主参数、`exp_avg`、`exp_avg_sq` 平时**常驻 CPU pinned buffer**（`param_to_cpu_states_map`，逐张量独立 pinned 分配、无池化）；对应 NPU storage 在换出后 `resize_(0)`（与激活卸载同一释放技巧），初始化时即 zeros→CPU→`resize_(0)`。**step 间隙 NPU 上不存在这些状态**，峰值 HBM 只需驻留一个 chunk。
- **step 流水**（`swap_adamw_step`，`swap_numel = 总元素数 / swap_optimizer_times` 为一个 chunk）：
  1. 首 chunk：计算流先 `wait_stream(swap_to_host_stream)`（保证上一步的 D2H 已读完同一批 CPU buffer），随后在专用 `swap_to_device_stream` 上整 chunk 换入（`resize_` + `copy_(non_blocking=True)`，参数+双状态一次提交）；
  2. 逐参数：`wait_swap_to_device_event`（事件等待，**无 host 同步**）→ **`npu_apply_fused_adamw_v2` 单 fused kernel 在 NPU 完成更新** → 主参数→bf16 model param 回拷（挂事件）→ 转 `swap_to_host_stream`：等该事件后 D2H 换出 + `resize_(0)`；
  3. **换出与计算重叠、换入不与上一块计算重叠**：换出（D2H）挂在独立 host 流上，被后续计算覆盖；但下一块换入的发起条件是 `swap_count` 归零（上一块全部算完并挂上换出），且换入前 `wait_stream(swap_to_host_stream)` 经"换出流 D2H ← copy 事件 ← 上一块计算"的依赖链传导——**换入被压到上一块计算结束之后，整块 H2D 在每个块边界暴露**（即实测的块间串行等待，详见 §5.2）。
- **同步模型**：三条流（current / `swap_to_device_stream` / `swap_to_host_stream`）+ 三张事件表，全程 stream-order，**没有任何 host synchronize**——下一步的依赖由步骤入口的 `wait_stream` 在流上表达。
- **每步流量**：每元素 H2D 12B（fp32 主参数+双状态）+ D2H 12B，共 24B/元素/步（DistributedOptimizer 按 DP 分片）；**梯度不参与 swap**（RS 后梯度本就位于 NPU grad buffer）。
- 附注：mcore 还有另一条 CPU 上执行更新的路线（`HybridDeviceOptimizer`，CPUAdam fused），与 swap 型不同，本文不展开。

---

## 5. 激活卸载与优化器卸载：分别对比

两类机制数据对象与生命周期完全不同，分开比较：

### 5.1 激活卸载对比（torch 图路线 vs MindSpeed `async_offload`）

先说两类机制本身的区别（同一框架内也不同）：

| 维度 | 激活卸载 | 优化器卸载 |
|---|---|---|
| 数据对象 | 每步新生成的临时张量，生命周期 = 生成 → bwd 最后一次使用 | 跨步常驻张量（fp32 主参数、动量），生命周期 = 全程 |
| 卸载本质 | 把"last-use 前的驻留"临时搬到 CPU，用完即弃 | 改变驻留介质：平时 CPU，step 时窗口性换入换出 |
| 触发点 | fw 图内（hooks / 图 pass 自动插入） | step 边界 |
| 每步流量 | 卸载量 ×2（出+回），**可与 fw/bwd 计算大范围重叠** | 状态字节 ×2/步（fp32），只能藏在 step 内部或 bwd 尾部 |
| 释放/回收 | 一次性（用完即弃，涉及 keepalive 与释放时机设计） | 循环复用（resize_(0) 释放 NPU storage，CPU buffer 常驻） |
| host 同步容忍度 | fw/bwd 内不能 host sync（会卡住计算流提交） | step 边界可以承受（torchtitan 就这么做的），MindSpeed 选择不承受 |
| 可隐藏窗口 | 有，且是核心约束（last-use 后剩余计算 × 带宽） | 无窗口概念：状态量 > step 计算量 × 带宽时必然暴露，靠"少传"（NPU-resident / 不换动量）缓解 |
| HBM 收益 | 削激活峰值（= 本步卸载量） | 恒定省去状态常驻（与 batch/seq 无关） |

再对比两框架的激活卸载实现：

| 维度 | torch 图路线（本次验证） | MindSpeed FSDP2 `async_offload` |
|---|---|---|
| 集成方式 | 图 pass + 标记 policy，模型零侵入，依赖 compile | eager `saved_tensors_hooks`，per-block 显式包装（侵入模型代码） |
| 筛选策略 | 用户写 `joint_custom_pass` 标记 | `custom_check_fn` 启发式："计算贵、传输小→offload 并跳过对应重计算；反之重计算"——**offload 与重计算是同一决策** |
| 传输流 | 每设备一条专用流 + data_ptr 事件注册表 | 单条双向 offload 流 + 逐块 `wait_stream` |
| GPU 释放时机 | `sink_wait` 二选一：早释放（wait 原地，计算流被串行化）或沉图末（峰值不降） | **块边界释放**（下一块开始时 `resize_(0)`），兼得早释放与不串行化 |
| 预取 | reload 按传输时间前移（bw 可配，默认 50GB/s 偏高，实测 19–27） | 固定 1-block lookahead |
| host 内存 | `pinned_memory_pool`（按 (numel,dtype) 复用） | 每张量现场 `pin_memory`（~220–227µs/次，无池） |
| 参数保护 | 需 policy 排除 placeholder（踩过坑） | `base_check_fn` 检查 Parameter，天然排除 |
| host 同步 | 无（事件链） | 无（事件链） |

激活卸载要点：**核心差异是生命周期管理**。MindSpeed 以块为单位（块边界释放 + 单 wait_stream + 固定 lookahead），粗粒度但与 NPU 的贵同步原语（115–218µs/事件）匹配，且把 offload/recompute 统一决策；torch 图 pass 以张量为单位，灵活但暴露 sink_wait 两难（§3.3 第 2 条）与 per-op 发射开销，理想集成应做 per-layer/分块 compile 或给 pass 加块边界 sink。

### 5.2 优化器卸载对比（torchtitan-npu 三档 vs MindSpeed AdamWSwap）

#### 先弄清"等"有几种等法

优化器卸载全程都在抢同一条 PCIe，两套框架的差别本质上只有一个：**在哪里等、用什么姿势等、等多少次**。等法有三种：

1. **排队等**（`wait_stream` / `wait_event`）：CPU 把"前面那个做完我再做"写进设备的任务队列就走人，继续发后面的算子。不卡 CPU，但每次排队这个动作本身要花 ~115–140µs（NPU 平台的发射开销）。
2. **CPU 停等**（`event.synchronize()` / `staging.wait()`）：CPU 原地卡死，直到设备把队列排空才继续。代价 = 此刻队列里还没做完的部分，期间一个算子都发不出去。
3. **不等**：把先后依赖直接排进流里（事件链），一步都不用专门等——MindSpeed 的理想形态。

#### 同步点设置对比

| 位置 | MindSpeed AdamWSwap | titan CPU-canonical | titan NPU-resident（仅开 enable-cpu-offload） |
|---|---|---|---|
| 换入数据到达后计算前 | 排队等（每参数一次 `wait_event`） | 排队等（`handle.wait_on`，且数据已提前一步在预取） | 同左（只有参数+梯度要换入） |
| 计算完换出 | 排队等（计算流记事件 → 换出流等它） | 排队等（D2H 挂 producer 事件；Muon 走独立 commit 流） | 排队等 |
| 块/步边界 | 下一块换入前排队等一次换出流 | **`step()` 末尾 CPU 停等一次**（AdamW 无条件，`cpu_offload_adamw.py:233`；Muon `:611-616`）；`zero_grad()` 再停等一次（`optimizer.py:604-607`） | 同左（同一代码路径，`_offload_states` 不改变 wait 调用） |
| 每步 CPU 停等次数 | **0 次** | **2 次** | **2 次** |
| 传输流条数 | **2 条**（换入流、换出流各一条，双向并行） | Muon **2 条**（H2D 走 staging 流、D2H 走独立 commit 流，`cpu_offload_muon.py:499`）；AdamW **1 条**（全部排 `staging.stream` FIFO） | 1 条 |
| 每参数事务数 | chunk 化（默认 1/16 总量一批），事务少 | 逐参数，事务多（每次排队 ~115–124µs 级发射成本线性累积） | 逐参数 |

注意：titan 的"CPU 停等"（`staging.wait()` → 逐 handle `event.synchronize()`）在本平台可能提前返回（§3.2 平台陷阱），实测语义偏"弱停等"；训练正确性不依赖它（设备侧同流 FIFO / producer 事件链已保证顺序），但它挡 host 发射的保护作用也相应减弱。

#### MindSpeed 实测的"块间串行长等待"是怎么来的

实测现象：优化器搬运分块后，块与块之间必须串行，等待时间长。原因是一个**依赖传导链**：

1. 下一块的换入要等 `swap_count` 归零才发起——即**上一块所有参数都算完并挂上换出**之后；
2. 发起换入前还统一 `wait_stream(换出流)`，而换出流上排着上一块每个参数的 D2H，每个 D2H 又事件依赖上一块的计算——依赖一级级传回去；
3. 结果：**每个块边界，计算流整块卡住，等整块 H2D 做完才能开始算**。块越大（默认=总量/16）卡得越久；只有换出（D2H）全程被后面的计算盖住不暴露。

#### titan 会不会有类似情况

**不会出现"块边界突发卡顿"这种形态**——titan 没有块的概念，逐参数深度 1 预取（算参数 i 时，i+1 的换入已经在流上跑），换入永远提前一步在路上，等待被摊平成连续的流排队，不会集中在几个点上突然卡很久。

但 titan CPU-canonical 档有**平滑版的同等问题**：

1. **AdamW 的换入和换出挤在一条传输流上**：全部流量（参数+梯度+双动量换入、参数+双动量换出，bf16 状态下约 14B/元素）排成一条 FIFO，优化器步长 ≈ 全部字节 ÷ 有效带宽。MindSpeed 换入/换出两条流并行、且梯度根本不换（梯度本就在 NPU buffer 里），暴露的是换入 12B/元素（fp32 状态）——**titan 的暴露总量与 MindSpeed 相当（约 1.2 倍）**，但形态不同：MindSpeed"少数几次、每次很长"的突发卡顿，titan"从不长卡、整步平滑排队"。Muon 则已把 D2H 分到独立 commit 流，双向并行，比 AdamW 更接近 MindSpeed 的流拓扑。
2. **step 末 + zero_grad 各一次 CPU 停等**：本意是等传输尾巴、挡住 host 继续发射（E1 实测的 host 干扰、集合通信等待 +4.8s 与此模型一致）；但受 §3.2 平台陷阱影响，`Event.synchronize()` 可能提前返回，保护强度存疑。
3. 逐参数事务的发射成本线性累积：每次事件/句柄操作 ~115–124µs 级，参数条目多时（E1 实测 Addcdiv/Addcmul 3191 次发射）host 发射压力显著；MindSpeed 的 chunk 化把事务数压到 1/16，外加 fused kernel 少发射。

**结论**：titan 的 NPU-resident 档把传输量降到最小（动量零往返），是仅开 enable-cpu-offload 时的默认；若要用 CPU-canonical 档，需要对齐 AdamWSwap 的四个设计：chunk 化换入换出、换入/换出分两条流并行（Muon 已做到，AdamW 未）、`npu_apply_fused_adamw_v2` 类 fused kernel、以及把跨步/容器级依赖全部事件链化。

#### 机制细节对照（供查阅）

| 维度 | torchtitan-npu | MindSpeed AdamWSwap（`--swap-optimizer`） |
|---|---|---|
| 卸载对象 | CPU-canonical 档：参数+梯度+双动量全 CPU 常驻；NPU-resident 档：仅参数往返 | fp32 主参数 + `exp_avg`/`exp_avg_sq`；**梯度不 swap**（在 NPU grad buffer） |
| 更新执行位置 | NPU（公开算子组合：`lerp_/addcmul_/addcdiv_`，E1 实测新增 3191 次 Addcdiv/Addcmul 发射） | NPU（`npu_apply_fused_adamw_v2` 单 fused kernel） |
| 换入换出粒度 | 逐参数，深度 1 双缓冲预取（`pending_work`，2 组 work buffer） | **chunk = 总元素/`swap_optimizer_times`(16)**，chunk 级流水但块间串行（见上） |
| NPU storage 回收 | work buffer 常驻（双缓冲复用） | 换出即 `resize_(0)`，峰值 HBM 仅一个 chunk |
| 状态精度 | 随参数 dtype（bf16，`zeros_like`，`cpu_offload_adamw.py:126-132`） | fp32 主参数 + fp32 双状态 |
| 每步流量（每元素） | 换入 8B + 换出 6B（bf16），单流（AdamW）或分流（Muon） | 换入 12B + 换出 12B（fp32），双流并行，暴露约 12B |
| 跨步顺序保障 | host `wait()` 兜底 | 事件链（`swap_to_host_events_map` 等），无兜底同步 |
| 对应档位 | **NovaSwap `OptimizerStateSwapContainer`**（FlexShard bucket + prefetch_plan）是其直接同类 | chunk 版本，粒度更粗、发射次数更少 |

附注：mcore 另有一条 CPU 上执行更新的路线（`HybridDeviceOptimizer`：grads D2H + host `d2h_event.synchronize()` + CPUAdam fused），与 swap 型不同；MindSpeed-LLM FSDP2 路径（`fsdp2/optim/`）则无优化器卸载。

---

## 6. 建议

1. DSV4 若上激活卸载：优先边界张量（FullAC checkpoint 入口，~57MB/层），块粒度生命周期（对齐 MindSpeed 语义），可完全隐藏传输并腾出 ~2.5GB/步 HBM；eager 路线可直接借鉴 `async_offload.py` 的块包装，compile 路线需 per-layer compile 或给 pass 加块边界 sink。
2. 优化器卸载维持 NPU-resident 档（仅开 enable-cpu-offload 时的默认，动量零往返）；若评估 CPU-canonical 档，应补齐四件事对齐 AdamWSwap：chunk 级换入换出（现逐参数）、换入/换出分两条传输流并行（Muon 已做到，AdamW 现共一条 FIFO）、`npu_apply_fused_adamw_v2` 类 fused kernel（现公开算子逐个发射）、跨步依赖全部事件链化；或直接扩展 NovaSwap bucket 档（已是 chunk 级同类）。另建议在镜像上实测验证 `Event.synchronize()` 在 D2H 后的阻塞语义（§3.2 平台陷阱），host 侧完成保证不能只依赖单事件同步。
3. `activation_offload_cpu_gpu_bw` 应校准为实测 19–27GB/s（负载相关，建议运行时标定），否则 prefetch 前移量不足。
4. 卸载粒度 ≥64MB（带宽饱和下限），避免 <16MB 的小块（延迟与同步原语主导）。

## 附：产物

- `verify_act_offload/test_e2e.py` / `probe_min.py`：激活卸载功能验证（ correctness + 峰值）
- `verify_act_offload/bench_bw_sync.py` / `bench_bw_sync.json`：带宽与同步开销
- `verify_act_offload/bench_critical.py` / `bench_critical.json`：临界卸载量扫描（两条 sink_wait 曲线）
- 本文档：`verify_act_offload/OFFLOAD_ANALYSIS.md`
- MindSpeed 对照源码：激活卸载 `MindSpeed-LLM/mindspeed_llm/fsdp2/features/async_offload.py`（接线示例 `fsdp2/models/qwen3/qwen3_moe.py:302`，收益宣称 `docs/zh/pytorch/features/mcore/async_activation_offload.md:43-44`）；优化器 swap `MindSpeed/mindspeed/core/optimizer/swap_optimizer/swap_optimizer.py`（`SwapDistributedOptimizer` + `swap_adamw_step`，经 `features_manager/optimizer/swap_optimizer_feature.py` 启用）

## 附二：测量陷阱记录（11-03 发现）

1. **`Event.synchronize()` 语义不可靠**：D2H（256MB，空闲流）后紧接 ev.sync 实测 0.15–0.52ms 返回（物理不可能）；同结构在跨流 `wait_event` 链之后则可靠阻塞（21.2ms 等到 matmul+copy，残余 0.2ms）。本环境所有计时一律改用 `torch.npu.synchronize()`。
2. **同 pinned dst 重复 D2H + ev.sync 计时**出现亚毫秒假象（`bench_bw_sync.py` Part C 的 staging 0.29–0.31ms 即此假象，已弃用）；设备级同步下同 dst 拷贝是真实的（13ms/256MB）。
3. **带宽数字随机器负载波动**：同一脚本两次运行 D2H 大块 18.9→26.6 GB/s。共享主机上引用绝对带宽数字必须注明测量时刻的负载状态；临界卸载量经验曲线（设备级同步）两次运行逐点偏差 <0.5pct，可信。

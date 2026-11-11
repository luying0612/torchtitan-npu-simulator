# Multi-Head-MoE（SLA2）接入 torchtitan 0.3.0 仿真栈 — 交接文档

> 交接对象：后续接手的 AI/工程师。本文记录开发过程、最原始需求与全部设计决策，供审查。
> 日期：2026-11-05。作者：AI 会话（opencode / GLM）。
> 位置说明：本文件位于仓库之外（`/home/l30079348/titan030-qwen3.5/`），遵守 simulator 仓库
> "不再增加阶段迁移／自审文档，历史证据由 Git 保留" 的文档纪律。

## 0. 最原始开发需求（按会话记录归档，尽量逐字）

1. 参照 qwen3.5 的接入先例，把新模型 multi-head-moe（来自
   `https://github.com/ZhanluModelSim/inference-models-simulator`，路径
   `external_models/multi_head_moe/`）接入 torchtitan 0.3.0 仿真栈，跑通流程。
2. 不需要做实测对比；**需要计算算子建模（通信除外）**；部分融合算子可以对照 CANN 包实现。
3. 需要生成一份交接文档给 AI 审查，记录开发过程和最原始的需求原文，让其他 AI 能准确理解原始信息。
4. 对之前的参考文档保持批判性（可能存在错误），不要照搬。
5. （后续补充）完成之后先不用提交，暂存即可；仿真硬件配置用库上的样例
   （`sim-platform/hardware/chip/example.json`），不要沿用 qwen3.5 的
   （qwen3.5 用的是 simulator 内置标定档 `npu_910b3`）。
   *（后续两次更新：算子建模只认 opm-dev 仓；硬件最终采用
   `/home/l30079348/hardware` 的 DV120 真实规格，样例仅作过中途验证，见第 7 节。）*

## 1. 环境与仓库

- 容器 `titan030-qwen35`；宿主 `/home/l30079348/titan030-qwen3.5/` 挂载为容器 `/workspace`。
- 容器内必须先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 且在 import torch 前
  设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。torch 2.14.0+cpu，torch_npu 2.14.0.dev，CANN 9.2.0，
  torchtitan 0.3.0（site-packages），ruff 0.17.3，pyrefly（容器内需 `pip install pyrefly`）。
- 主仓：`torchtitan-npu-simulator030/`（分支 `titan030`，接入起点 commit `fca58c5`
  "Model the GDN and mhc_post fused operators"）。主包 `torchtitan_npu/`，
  仿真 distribution `experiments/simulator/`（独立 pyproject，pytest/pythonpath=["."]）。
- 上游 0.3.0 精简 checkout：`torchtitan030/torchtitan/`（容器内另有完整 site-packages 版本，
  含 `models/common/decoder_sharding.py`、`models/common/moe_sharding.py` 等）。
- `torchtitan_npu` / `torchtitan_simulator` / `operator_performance_models` / `sim_platform`
  均为 editable 安装，MetaPathFinder 优先于 PYTHONPATH —— 做干净基线只能 `git stash`，不能用 worktree。

## 2. 参考实现（以及它为什么不能直接信）

- 参考代码：`/tmp/opencode/inference-models-simulator/external_models/multi_head_moe/multi-head_moe.py`
  （纯 PyTorch 朴素实现，MAGI-2 preview 风格：SLA2Attention + Multi-Head-MoE）。
- **参考实现从未运行成功，含 3 个真实 bug**（移植时全部修正，修正版与最小前向/反向
  验证脚本在 `/home/l30079348/titan030-qwen3.5/_mhoe_scratch/`，scratch 仅作证据，不入仓）：
  1. RoPE 广播错误：`freqs_cis.unsqueeze(2)` 与它自己的 `[B,S,H,D]` 布局矛盾，
     正确为 `unsqueeze(1)`（freqs_cis 为 `[S, Dh/2]`）。
  2. 线性分支 einsum `"bhqk,bhkd->bhqd"` 非法（5 维操作数配 4 个下标）；
     修正为 `"bhqk,bhkld->bhqld"`，kv state/phi_k_sum 需 `repeat_interleave(bs, dim=2)`，
     norm 用 `matmul(phi_q.unsqueeze(-2), phi_k_sum.unsqueeze(-1)).squeeze(-1)`。
  3. 布局错位：`SLA2Attention` 产出 `[B,S,H,D]`（`unflatten(2,...)`），但
     `InnerBlockSLA2`（其自身 docstring）期望 `[B,H,T,D]` —— 在边界处 transpose 修正。
- 其他参考坑：router/gate 权重用 `torch.empty` 且从不初始化；alpha 固定 0.5
  （sigmoid≈0.62 分支混合）。移植版给 router/gate 标准 small-std normal 初始化
  （标准库 `trunc_normal_`/`normal_`），alpha 保持 0.5 构造值不动。
- 生产几何（用于文档对照）：dim=8192, vocab=102400, 80 层, 64 heads/128 hd,
  MoE 32 头 × 896 专家, K=16, shared 4096, expert mult 4, block 64,
  topk 比例 0.005/0.001（≤1M/超长）。
- 数值边界提示（忠实继承参考实现，真实 kernel 数值对齐前须知）：参考实现
  未加因果 mask；padding 位置的 softmax(0) 均匀分布会泄入 linear 分支的
  norm 分母（分子不受影响，因 padding 位置 phi_v=0）。

## 3. 融合算子边界决策（对照 CANN）

先用最小探针在 Meta 上测过 CANN 算子可用性（结果重要，勿凭版本号臆断）：

| 算子 | Meta 支持 |
| --- | --- |
| `npu_rms_norm` / `npu_swiglu` / `torch._grouped_mm` | OK |
| `npu_rotary_mul` | 无 Meta kernel（已有 `compat/rotary.py` 桥接，meta_runtime 无条件安装） |
| `npu_moe_gating_top_k_softmax` | 无 Meta kernel |
| `npu_block_sparse_attention` | 有 Meta kernel（能过 dispatch；卡在参数校验 "block_shape[1] (blockShapeY) must be a multiple of 128, got 64"） |

决策：**SLA2 核心仍按两个显式虚拟融合算子建模**，锚点是 CANN 契约文档中的
`aclnnBlockSparseAttention`（稀疏分支；该 kernel 只算选中的 block）：
- `multi_head_moe_sim::sla2_sparse_attention(+_grad)`：块稀疏 softmax 分支。
- `multi_head_moe_sim::sla2_linear_attention(+_grad)`：分块线性注意力分支（约定生产名，
  无公开 CANN 锚点）。
- 稀疏分支的 1/sqrt(Dh) 缩放在 seam 之外用原生乘法表达（常数乘留在融合边界外，
  seam 只覆盖注意力主体）。虚拟算子签名除捕获张量外可收 schema 标量：
  `selected_blocks` 以标量参数进入捕获节点 `node.attrs`（flex_attention 同机制），
  不作为捕获 tensor 声明语义角色；backward 为标量返回 `None` 梯度。
- mask（训练态软 mask）是可微真实输入，两个融合的 backward 都返回 q/k/v/mask 四个梯度。
- 块池化、block router（真实参数投影）、gumbel 软 top-k mask、`M_not = 1 - M_c`、
  alpha 分支混合全部留在原生 dispatch，不做虚拟化。
- 路由/gate 不用 `npu_moe_gating_top_k_softmax`（无 Meta kernel），保持 ATen；
  专家计算复用 common `RoutedExperts`（token dispatcher + grouped experts +
  `torch._grouped_mm`，Meta OK）。

## 4. 结构映射与关键设计

- **Multi-Head-MoE 展平**：把 (head, expert) 展平为专家 id `head*Es+expert`，
  复用 common `RoutedExperts`：`GroupedExperts(num_experts=S*Es, dim=hs, hidden_dim=hs*mult)`；
  子空间视作 `[T, S_h, hs]`（B=T, L=S_h, D=hs 喂 dispatcher）。gate router 参数
  `[S, hs, Es]` 声明为 Replicate（专家分片只作用于 grouped experts）。
- **Debug 负载均衡**：`MultiHeadGate` 自带 `_debug_force_load_balance` 字段
  （round-robin ids，对齐 common router 的 debug 模式）；上游
  `Decoder.Config.update_from_config` 会把 `debug.moe_force_load_balance` 传播到
  `layer_cfg.moe.router._debug_force_load_balance` —— 字段名必须叫 `router`。
- **拒绝项（fail fast，理由写进异常文本）**：TP（SLA2 head 轴与 per-head gate 需要
  head 分片布局）、CP（块池化与逐 block 线性状态跨全序列）、MTP（preset 拒绝）。
- **EP>1**：`Config.update_from_config` 把 dispatcher 换成上游 `AllToAllTokenDispatcher.Config`。
- **sharding 注解（spmd_types 后端必需）**：FSDP2 在提供 `dp_mesh_dims` 时拒绝 plain
  tensor 参数。上游模式是 `Config.update_from_config` 里调用 `set_*_sharding_config`
  （先于 build），参考 `gpt_oss/sharding.py`。本项目特有两处 plain 参数
  （`InnerBlockSLA2.alpha`、`MultiHeadGate.router`）在各自 Config 的 `__post_init__`
  里给 Replicate 的 `ShardingConfig`（`dense_param_placement(tp=spmd.R)`）。
  新增 `models/multi_head_moe/sharding.py`：根状态用 `set_decoder_sharding_config`，
  层内线性/norm 用 Replicate，MoE 复用 common `moe_sharding` 的
  `_routed_experts_sharding_configs` / `_shared_experts_sharding_configs`。
  注意：**不能直接套 `set_moe_sharding_config`** —— 它会给我的 MoE wrapper 注
  `expert_bias_E`/`tokens_per_expert_E` 状态声明（我的 wrapper 没有这些参数），
  且它假设 router 有 `.gate` 子模块。
- **RoPE/norm**：复用 common `ComplexRoPE` + override `rope.asc_complex`；
  RMSNorm 走 `_COMMON_OVERRIDES` 的 `rms_norm.asc`。GroupedExperts 需要显式
  `param_init`（patch 后的 `_ClampGroupedExperts` 无默认 init）—— 用上游惯例
  `depth_scaled_std(0.02, layer_id)` 的 `trunc_normal_`（参考 gpt_oss）。
- **参数公式（debug 几何）**：D=256, V=102400, 8 层（0,1,6,7 dense；2–5 MoE）,
  H=4, hd=64, block=64, router_proj=64, inter=512; MoE S=4, hs=64, Es=16, Ks=4,
  expert mult 2（inter=128）, shared 128, eps=1e-6, rope theta=10000。
  `TOTAL = 2·V·D + D + 4×664,196 + 4×2,077,316 = 63,395,104`；
  参数张量数 `3 + 4×14 + 4×20 = 139`。

## 5. 交付物清单（全部已暂存，未提交）

新增：
- `torchtitan_npu/models/multi_head_moe/{__init__.py, model.py, config_registry.py, parallelize.py, sharding.py}`
- `experiments/simulator/torchtitan_simulator/compat/multi_head_moe.py`
- `experiments/simulator/tests/test_multi_head_moe_capture.py`

修改（纯增量）：
- `experiments/simulator/torchtitan_simulator/presets.py`（`multi_head_moe_debugmodel`，拒绝 MTP）
- `experiments/simulator/torchtitan_simulator/config.py`（factories 字典、家族白名单、`_MULTI_HEAD_MOE_OVERRIDES`）
- `experiments/simulator/torchtitan_simulator/runtime.py`（`configure_model` 分支安装虚拟融合）
- `experiments/simulator/torchtitan_simulator/capture/op_mapping.py`（4 个 SLA2 虚拟算子 raw→op_type 映射）
- `experiments/simulator/torchtitan_simulator/cost/op_cost_model.py`（SLA2 库路由 handler、attrs 透传、
  规范名/`_canonical_dtype` 映射、handler 表宽化；`_library_matmul_duration` 对齐库新 API）
- `experiments/simulator/torchtitan_simulator/cost/hardware.py`（新增 `dv120_hardware()`：
  解析 `/home/l30079348/hardware/dv120-chip.txt` 为库 `ChipCapabilities`，只做格式翻译不加标定；
  `npu_910b3()` 保留不动）
- `experiments/simulator/docs/capability-matrix.md`（新行 + 虚拟融合计数更新）

opm-dev 仓（需求方指定 `/home/l30079348/opm-dev`，git 管理；改动**未提交**，文件即交付）：
- `semantics/npu_fused.py` / `semantics/npu_gradients.py`（4 个 SLA2 语义 dataclass；
  linear 族标量字段名 `complement_blocks`，与 sparse 的 `selected_blocks` 区分源契约）
- `adaptors/npu_fused.py` / `adaptors/npu_gradients.py`（4 个 SourceAdaptor + `_sla2_grid`/
  `_sla2_selected` 共享校验；grad 规范名 `Sla2*Backward`、输入顺序 dy 末尾）
- `models/roofline/fused.py`（`_Sla2Roofline` 基类：FP32 cube + 声明式 vec、
  `max(cube+vec, memory)+head_tail` 组成（MHC 投影同款）+ 4 个 Roofline 模型，`_MODELS` 注册）
- `roofline_config.py`（PrimitiveVecConfig family 白名单 +4）
- `tests/test_sla2_roofline.py`（6 个用例：adaptor 往返、4 模型精确工作量与 duration、
  配置门控、网格/越界拒绝）
- `docs/sla2-roofline-models.md`（新）、`docs/cost-model-status.md`（66 adaptor/94 名/69 族）、
  `AGENTS.md`（覆盖声明）
- 容器接线：`docker cp opm-dev → /workspace/opm-dev` + `pip install -e /workspace/opm-dev`
  （host 编辑后需重新 docker cp 同步）

仓库内没有为本次接入新增任何硬件标定常数；`dv120_hardware` 仅解析需求方提供的规格文件。

## 6. 验证结果（2026-11-05，容器内实测）

- 端到端：`python -m torchtitan_simulator --config multi_head_moe_debugmodel` 静默成功（PYEXIT=0）。
- pytest `tests/test_multi_head_moe_capture.py`：**15 passed**（评审优化轮后）。覆盖：
  参数公式（63,395,104 / 139）；捕获矩阵 single / FSDP rank0 / rank1 / HSDP(4,2) /
  full-AC / selective-AC（8 层 × 2 融合：fwd 8，full/selective AC 重算后 16，bwd 8）；
  虚拟融合计数与语义声明（仅 4 个 sim 算子带 tensor_semantics，原生算子零泄漏）；
  EP2 all-to-all（`all_to_all_single` 精确 20 个）；TP / CP `NotImplementedError`；
  MTP `ValueError`；直发 dispatch 的 saved-inputs/backward-identity 检查；
  SLA2 成本注记（见下）；seq_len=128 真实 block 选择路径；DV120 真实硬件档加载。
- SLA2 成本注记（`annotate_workload_costs`）：SLA2 建模**整体迁入 opm-dev 仓**
  （需求方指定 `/home/l30079348/opm-dev`，git 管理，容器内以
  `/workspace/opm-dev` editable 安装消费）：4 个语义族（sparse/linear ×
  fwd/gradient，闭式 cube GEMM 公式 + 声明式 softmax/normalize vec 段，见该仓
  `docs/sla2-roofline-models.md`）；simulator `op_cost_model.py` 只做库路由
  （OpFeatures + `selected_blocks` 标量 + 规范名映射 + 显式声明 VecOpConfig）。
  迁移中确立的三个源契约（adaptor 层翻译）：grad 规范名是 `Sla2*Backward`
  （非 `...Gradient`）、grad 输入顺序 `(q, k, v, mask, dy)`（dy 末尾）、
  **linear 融合的标量按源契约是补集块数**（语义字段名 `complement_blocks`，
  模型按 `(complement+1)` 遍历计；sparse 的标量才是真选中数）。捕获 dtype
  需经 `_canonical_dtype` 规范化（`bf16`→`bfloat16` 等）。无硬件时 SLA2 节点
  `cost_unknown`（库 roofline 需要 chip capabilities，不做部分注记）；带硬件
  （DV120，见第 7 节）时获得 cube+vec flops、逻辑流量与 duration。
  seq_len=16（补齐 T=64）实测精确值：sparse fwd 4,292,608 / sparse bwd
  10,633,220 / linear fwd 2,129,920（complement=0 合法退化）/ linear bwd
  4,276,228。`selected_blocks` 经融合算子 schema 标量进入 `node.attrs`
  （flex_attention 同机制），bwd 为标量返回 None 梯度。
  seq_len=128（nb=2，topk=1）覆盖非退化选择：selected=1、complement=1，
  两融合 fwd flops 8,585,216 / 8,454,144。
- 内存字节预期：`persistent_param_bytes = 本地参数×4`（rank1 减半验证）；
  `optimizer_state_bytes = 本地参数×8 + 139×4`（fused AdamW 两个 FP32 矩 + 步长标量）。
- 通信字节注记（第三轮审视项 2）：EP2 的 20 个 all_to_all + 11 个 reduce_scatter
  节点获得 `comm_bytes` = 逻辑跨卡体量（in+out 各一次，按 dtype 位宽），测试从
  捕获张量事实独立重算并逐节点精确断言；duration 需通信链路模型
  （CollectiveRooflineConfig + 拓扑），本轮不消费（与 dv120-topo.txt 同一后续项）。
  unknown 判定相应计入 comm_bytes（带字节的通信节点不再是 unknown）。
- 耗时包络不变量（第三轮审视项 1 的可落地部分）：SLA2 路由测试对每个融合节点
  断言 `duration_us ≥ max(flops/cube峰值, traffic/HBM带宽)`——声明组成
  `max(cube+vec, memory)+overhead` 的下界，防路由回归。真机
  `aclnnBlockSparseAttention` microbench 锚点校准超出需求边界（需求第 2 条），未做。
- lint/类型：ruff check + format 干净（仅 4 个本次未触碰文件的存量告警
  I001/B905×2/RUF036）；pyrefly 主包 0 错误，`op_cost_model.py` 0 错误
  （库 API 对齐后 `_library_matmul_duration` 的存量 `bad-argument-type` 一并消除）。
- 存量失败基线（已核对）：全量 suite 有 47 个**存量**失败（629 passed），分布在
  `test_kimi_quantization / test_memory_delivery / test_model_overrides / test_precision /
  test_routing / test_s4_mxfp8 / test_storage_lifetime_capture`。
  干净 checkout（`git stash` 后）重跑同一 47 个用例：**失败集合与带改动时逐条一致**
  （`/tmp/mhm_failed_with.txt` vs `/tmp/mhm_failed_clean.txt` diff 为空）—— 本次改动零回归。
  例证：`test_storage_lifetime_capture.py::test_native_capture_releases_match_full_scan_at_every_boundary[hsdp-selective-mxfp8]`
  在干净 checkout 上同样失败。

### 6.1 无基准数学等价校准（2026-11-06，黄金参照法）

模型只有结构代码、没有可运行的原训练框架，因此引入"修正版朴素参考实现 = 数学黄金参照"
的校准（脚本在 `_mhoe_scratch/calib_golden_equivalence.py` / `calib_dispatch_and_memory.py`）：

- **发现并修复一个真实架构 bug（`TransformerBlock.forward` 残差丢失）**：第二残差
  错误接回块输入 `x_BLD` 而非 post-attention 隐状态 `h`，attention 输出只留在 FFN
  分支内部，等价于丢掉 `h = x + attn` 的残差项（逐层对账定位：layer0 attention/FFN
  输出均与参考零差，而 block 输出差恰为一个 attention 幅值）。15 个捕获测试全部无法
  发现它（形状/计数/内存自洽）——捕获保真不证明数学正确。已修复为
  `h + moe(ffn_norm(h))`，与参考 `out = h + FFN(Norm(x+attn))` 一致。
- 修复后黄金测试（同权重、eval 硬掩码、fp32、seq=128）：logits max_abs=1.7e-6，
  loss 逐位一致，123 组映射参数梯度 max_rel ≤ 7.6e-5，0 个超 1e-3（fp32 噪声级）。
  覆盖：MoE (head,expert) 展平 vs 参考逐头循环、gate/门控 top-k、SLA2 双分支与
  alpha 融合、RoPE（ComplexRoPE 与参考 freqs_cis 约定逐字一致）、块池化/路由。
- 校准过程排除了两个 harness 陷阱（均非移植错误，记录防复发）：参考实现的
  `InnerBlockSLA2` 惰性创建于首次 forward，`eval()` 后物化的子模块 training=True
  （需物化后重新 eval）；同理其 BlockRouter bias 逃过构建期置零（需物化后再置零）。
- **CPU dispatch trace vs Meta 捕获对照**（同几何 world=1）：`aten._grouped_mm`
  fwd 12/bwd 24 两侧精确一致；`npu_rms_norm` 33（头 norm 16+块 norm 16+final 1）、
  `npu_rotary_mul` 16（8 层×q/k）精确对应层结构；SLA2 虚拟融合 fwd16/bwd16 恰好
  替换 CPU eager 分支算子组（masked_fill/softmax/bmm）。差异项全部可归因：
  CE loss（sim）vs pow2（harness）、训练态 gumbel `rand_like`×8（sim train 模式）、
  override 替换（rms_norm/rope/token_dispatcher）。
- **内存 oracle**：真实 CPU 模型 persistent 参数字节 253,580,416 = 公式精确一致；
  139 张量精确一致；AdamW 矩量 nparams×8 吻合（CPU torch AdamW 与 NPU fused
  kernel 的 step 标量记账约定不同，差 556B 量级，公式按 fused 契约）。

## 7. 仿真硬件配置决策（按需求第 5 条）

- qwen3.5 接入时的耗时注记用 simulator 内置标定档 `npu_910b3()`
  （`cost/hardware.py`，标定常数来自生产 trace）。本次**不沿用**。
- 最终采用**真实 DV120 规格**（需求方指定 `/home/l30079348/hardware/`：
  `dv120-chip.txt` + `dv120-topo.txt`）。sim-platform 的 chip 加载器只认
  schema v1（`chip_config` 形态），不识别 DV120 原生形态，故在 simulator
  `cost/hardware.py` 新增 `dv120_hardware(path)`：把 DV120 JSON（compute
  峰值 TFLOPS 按 floatN 位宽、log/constant 利用率模型、HBM 容量/带宽、
  双 head-tail 常数）映射为 operator-performance-models 的 `ChipCapabilities`
  （log_a/log_b → LogUtilization 的 k/b），**只做格式翻译、不加标定**。
  链路：`dv120_hardware()` → `annotate_workload_costs(workload, hardware=...)`。
  L2cache 记录与 `dv120-topo.txt`（通信拓扑，对应 CollectiveRooflineConfig
  领域）本轮不消费，属后续通信建模任务。
- 对照实测（2026-11-05 晚，seq_len=128 + DV120，opm-dev 库）：
  - multi_head_moe：非 SLA2 部分 **1518 duration / 1867 unknown**，SLA2 融合
    **32/32 全部获得 duration**（DV120 roofline）；unknown 为 view/t/detach/
    transpose/slice 等零计算布局算子与 stream_wait/profiler 事件，设计如此非缺口。
    qwen3.5 同配置 1831 / 1655，构成相同、无回归。
  - SLA2 的 vec 段显式声明 FP32 计算（等价 `VecOpConfig(32, 1.0, float32)`，
    结构单位数 ×1.0 字面量，非标定系数）；DV120 含 32-bit cube（216 TFLOPS）
    与 vec（13.5 TFLOPS）条目，`npu_910b3()`（仅 16-bit vec）下 SLA2 会
    `cost_unknown`。
  - 结论边界：simulator 仓不做模型专属耗时标定；耗时预测仍是显式 opt-in。
    SLA2 融合的耗时来自 opm-dev 的声明式粗粒度 roofline（与 mhc_post 同状态：
    结构公式无系数、无真实负载 trace 校验），非实测标定。
- 历史记录：中途曾用 `sim-platform/hardware/chip/example.json`（虚构样例）验证
  全链路（`_mhoe_scratch/verify_hardware_sample.py`：324 个 mm 类节点获得
  duration）；后被需求方指示替换为 DV120 真实规格。

## 8. 遗留事项与复验入口

- [x] 评审优化轮（`MULTI_HEAD_MOE_REVIEW.md` P1–P4）：
  P1 成本缺口（selected_blocks 标量 + SLA2 flops/流量 handler + attrs 透传）已实现并验证；
  P2 文档精度（本文件与汇报口径修正：标量可入虚拟算子 attrs、ruff/pyrefly 存量计数限定）；
  P3 参考实现出处（`_mhoe_scratch/multi-head_moe_ORIGINAL.py` 为 pristine 原版存档，
  `_mhoe_scratch/multi-head_moe.py` 为 3 处 bug 修正版，修正注释在文件头；
  回归证据 `_mhoe_scratch/regression_failed_*.txt`）；
  P4 测试补强（seq_len=128 真实选择、CP 拒绝、EP2 精确计数、偶数 numel 前提注释、成本精确值断言）。
- [x] 架构迁移（需求方指示：算子建模只认 opm-dev 仓 `/home/l30079348/opm-dev`，硬件用
  `/home/l30079348/hardware` 的 DV120 规格）：SLA2 的 4 个语义族（闭式 cube 公式 +
  声明式 vec 段，结构单位无系数）整体迁入 opm-dev 并按其规范重写（MHC 投影同款
  组成、is_fused、`requires_vector_config` 门控），simulator 侧 `_sla2_*` 公式
  handler 删除、改为库路由；`dv120_hardware()` 解析真实规格文件；库消费路径
  `_library_matmul_duration` 对齐库新 API（构造期 semantics/hardware）。
  opm-dev 侧 pytest 1793 passed 全绿（ruff 全绿，无既有失败）；simulator 侧
  15 passed（成本精确值断言全部为库口径：4,292,608 / 10,633,220 / 2,129,920 /
  4,276,228；seq128 8,585,216 / 8,454,144），ruff/pyrefly 0 错误。遗留：simulator
  其余算子族 handler（vec/cube/GDN）仍为 simulator 自带实现，按"只认模型库"方向
  属后续全量迁移任务；`dv120-topo.txt`（通信拓扑）与 L2cache 未消费。
- [x] 第三轮审视意见（`MULTI_HEAD_MOE_REVIEW.md`）处置：
  项 1 的 SLA2 耗时建模已收敛（15 passed、DV120 32/32 duration），包络不变量已入测试，
  真机 microbench 超需求边界未做；项 2 通信字节对账已实现（comm_bytes + 精确断言）；
  项 3 黄金测试入仓**按需求方指示忽略**（涉及 CPU 结果对比，不予执行，暂不建仓；
  校准证据保留在 `_mhoe_scratch/calib_golden_equivalence.py`）；项 4 真机终局校验
  超需求边界；项 5 数值边界提示已加入本文件 §2；项 6 并行在途工作即本轮迁移，
  已收敛全绿。附录所言"§3 A 项措辞仍待修正"系审查者读取旧版文档，当前版本已修正。
- [x] 多模态核对：需求方提示模型可能多模态；参考仓 README 明确 multi_head_moe 为
  **text-only** 分类，`HybridMultiHeadMoEModel.forward(tokens)` 仅接受 tokens，
  无视觉分支/专用 loss。模拟器捕获链按 token 级透传 `extra_inputs`，若后续参考实现
  增加多模态输入，接入点在 `model.py` 前向入口，虚拟融合边界不受影响。
- [x] 忠实继承的参考实现数学瑕疵（接手真实 kernel 数值对齐前须知）：参考实现
      未加因果 mask；padding 位置的 softmax(0) 均匀分布会泄入 linear 分支的
      norm 分母（分子不受影响，因 padding 位置 phi_v=0）。为与参考实现保持
      逐位一致，本接入未修正该行为，生产化时需与真实 kernel 行为对齐。
- [x] 全量回归新旧失败清单对比：**完成**，失败集合逐条一致，零回归（见第 6 节）。
- [ ] 真实 NPU 数值/性能验证：超出本次需求范围（需求第 2 条明确不做实测对比）。
- [ ] `npu_block_sparse_attention` 的 block_shape 校验（128 对齐）与参考 block=64
      的关系值得在真实 kernel 接入时复核；当前虚拟算子不受影响。
- 复验命令：
  - `docker exec -w /workspace/torchtitan-npu-simulator030/experiments/simulator
    -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 titan030-qwen35 bash -c
    "source /usr/local/Ascend/ascend-toolkit/set_env.sh && python -m pytest tests/test_multi_head_moe_capture.py -q"`
  - 端到端：`python -m torchtitan_simulator --config multi_head_moe_debugmodel`
  - 样例硬件校验：`python -m sim_platform.hardware example`（容器内 /workspace/sim-platform）
- 暂存说明：按需求第 5 条，全部改动 `git add` 暂存、**未提交**。

## 9. 源码保真整改与融合开关（2026-11-10，owner 指令）

Owner 裁定：模型实现必须与给定源码一致，禁止自行以融合/虚拟算子替换实现；已整改并复验。

- **恢复源码小算子实现**（默认路径）：RoPE = 参考 `apply_rotary_emb`（复数乘法，unsqueeze(1) 修复）、RMSNorm = 模型内置小算子 norm、缩放恢复 matmul 后除法、SLA2 两分支恢复为 module-level seam 函数 `run_sla2_sparse_attention` / `run_sla2_linear_attention`（model.py），eager 下逐字执行参考算子序列。
- **融合边界与范围**（唯一允许替换的两处）：上述两个 seam 函数。sparse = 掩码展开+全分数 matmul+1/sqrt(Dh)+masked_fill+softmax；linear = phi softmax+分块 einsum+KV 状态 matmul+repeat_interleave+归一化。边界之外（块池化/BlockRouter/soft top-k mask/M_not/alpha 混合/pad/crop/QKV 投影）两种模式均为原生 dispatch。
- **开关**：`SimulationConfig.enable_sla2_fusion`（默认 False=eager）。开启时 runtime 装 `compat/multi_head_moe.py` 虚拟融合算子（4 个：sparse/linear × fwd/grad），成本经 opm-dev SLA2 族路由；关闭时成本走标准 aten handler。
- **MoE 专家计算（owner 决策）**：两种模式均为 grouped-GEMM（参考的 bincount/where 循环无法被 Meta 捕获——输出形状依赖张量数值，实测报错；bincount 可给精确 meta kernel 但 `.tolist()` 数值分支与 where 紧凑索引无法静态化）。
- **保留的已批准修复**：3 个参考 bug（RoPE 广播/einsum 下标/布局转置）+ 初始化（router/gate std0.02）+ topk 每调用传入（参考懒缓存会过期）。
- **验证**：`tests/test_multi_head_moe_capture.py` 14/14（eager 默认 6 参数化 + EP2/TP/CP/MTP + eager 成本 + seq128 分块签名 + 融合开关开启断言）；dsv4/kimi/config/semantics/memory 抽样全绿；全量套件 48 失败为 5747770 基线预存（mxfp8/CLI geometry，临时基线 worktree 复现证实与本改动无关）。ruff check+format / mypy 干净。
- **环境**：容器 /workspace/opm-dev 曾被回退到 650b214（丢失 SLA2），已从宿主机 c59ba69+未提交 SLA2 重同步；容器未提交的 fused_recurrent 增量已包含于宿主机状态，无损失。

# 02 · 模型迁移实现（multi_head_moe → titan030 仿真栈）

> 归档正本：`archive/MULTI_HEAD_MOE_ADAPT.md` §3–§5。本文为梳理版，反映**最终状态**（含保真整改后的双版本结构）。

## 2.1 结构映射与关键设计

- **Multi-Head-MoE 展平**：把 (head, expert) 展平为专家 id `head*Es+expert`，复用 common
  `RoutedExperts`：`GroupedExperts(num_experts=S*Es, dim=hs, hidden_dim=hs*mult)`；
  子空间视作 `[T, S_h, hs]` 喂 dispatcher。gate router 参数 `[S, hs, Es]` 声明 Replicate。
- **Debug 负载均衡**：`MultiHeadGate._debug_force_load_balance` 字段（round-robin ids，对齐
  common router debug 模式）；上游 `Decoder.Config.update_from_config` 传播要求字段名必须叫 `router`。
- **拒绝项（fail fast，理由写进异常文本）**：TP（SLA2 head 轴与 per-head gate 需要 head 分片）、
  CP（块池化与逐 block 线性状态跨全序列）、MTP（preset 拒绝）。
- **EP>1**：`Config.update_from_config` 把 dispatcher 换成上游 `AllToAllTokenDispatcher.Config`。
- **sharding 注解（spmd_types 后端必需）**：FSDP2 在提供 `dp_mesh_dims` 时拒绝 plain tensor 参数。
  两处 plain 参数（`InnerBlockSLA2.alpha`、`MultiHeadGate.router`）在各自 Config `__post_init__`
  给 Replicate 的 `ShardingConfig`。`sharding.py`：根状态 `set_decoder_sharding_config`，层内
  Replicate，MoE 复用 common `moe_sharding` 的 routed/shared 配置。
  **不能直接套 `set_moe_sharding_config`**（会给 wrapper 注入不存在的状态声明，且假设 router 有 `.gate`）。
- **RoPE/norm**：复用 common `ComplexRoPE` + override `rope.asc_complex`；RMSNorm 走
  `_COMMON_OVERRIDES` 的 `rms_norm.asc`。GroupedExperts 显式 `param_init`（`depth_scaled_std` trunc_normal_）。
- **框架契约适配**：`Decoder.Config` 要求 `attention.rope.max_seq_len` → 新增 `RopeGeometry`
  元数据 dataclass 承载配置；`RMSNorm` 补 `reset_parameters`；Meta 下 autocast 不兼容 →
  `x.device.type == "meta"` 守卫。

## 2.2 融合边界演进（最终形态，owner 裁定）

- 最初：SLA2 核心按两个显式虚拟融合算子建模（锚点 CANN `aclnnBlockSparseAttention`），
  seam 函数携带 `selected_blocks` schema 标量进 `node.attrs`（flex_attention 同机制）。
  **Meta 探针结论（勿凭版本号臆断）**：`npu_rms_norm`/`npu_swiglu`/`torch._grouped_mm` OK；
  `npu_rotary_mul` 无 Meta kernel（已有 compat 桥）；`npu_moe_gating_top_k_softmax` 无；
  `npu_block_sparse_attention` 有但卡参数校验（blockShapeY 须 128 对齐，参考 block=64）。
- **保真整改后（当前形态）**：
  - **默认 eager**：`run_sla2_sparse_attention` / `run_sla2_linear_attention`（model.py:150/175，
    调用点 InnerBlockSLA2.forward:321）逐字执行参考算子序列。
  - **融合 opt-in**：`SimulationConfig.enable_sla2_fusion`（默认 **False**）。开启时 runtime 装
    `compat/multi_head_moe.py` 虚拟融合（4 个：sparse/linear × fwd/grad），成本经 opm-dev SLA2 族路由。
  - 融合范围仅限两 seam：sparse = 掩码展开 + 全分数 matmul + 1/sqrt(Dh) + masked_fill + softmax；
    linear = phi softmax + 分块 einsum + KV 状态 matmul + repeat_interleave + 归一化。
    边界外（块池化/BlockRouter/soft top-k mask/M_not/alpha 混合/pad/crop/QKV 投影）两模式均原生 dispatch。
- **MoE 专家计算（owner 决策）**：两种模式均为 grouped-GEMM。参考的 bincount/where 逐专家循环
  **无法被 Meta 捕获**（实测：bincount 无 fake impl；`.tolist()` 数值分支与 where 紧凑索引
  数据依赖输出形状；topk 可捕获）。

## 2.3 交付物（最终状态）

simulator worktree `mhm-wip-wt/`（分支 `multi-head-moe-wip`）：

- 新增：`torchtitan_npu/models/multi_head_moe/{__init__.py, model.py, config_registry.py, parallelize.py, sharding.py}`；
  `experiments/simulator/torchtitan_simulator/compat/multi_head_moe.py`；
  `experiments/simulator/tests/test_multi_head_moe_capture.py`。
- 修改（纯增量）：`presets.py`（`multi_head_moe_debugmodel`，拒绝 MTP）、`config.py`
  （factories/family 白名单/`_MULTI_HEAD_MOE_OVERRIDES`）、`runtime.py`（开关门控安装虚拟融合）、
  `capture/op_mapping.py`（SLA2 raw→op_type 映射）、`cost/op_cost_model.py`（SLA2 库路由 +
  attrs 透传 + `_canonical_dtype`）、`cost/hardware.py`（`dv120_hardware()`：解析
  `/home/l30079348/hardware/dv120-chip.txt` 为库 `ChipCapabilities`，只翻译不标定）、
  `docs/capability-matrix.md`。

### Git 状态（如实记录）

| 内容 | 位置 | 状态 |
| --- | --- | --- |
| 基线（SLA2 库路由 + 模型包 + 捕获） | `multi-head-moe-wip` @ `5747770` | 已推送 fork `luying0612/torchtitan-npu-simulator` 分支 `030-multiheadmoe-1109` |
| 保真整改 + 融合开关（+348/−306，9 文件：model.py/config_registry/sharding/compat/runtime/op_mapping/op_cost_model/config/测试） | `mhm-wip-wt` 工作区 | **未提交未推送**（待指示） |

### 硬件档

`dv120_hardware()` 只做格式翻译不加标定：DV120 JSON（compute 峰值按 floatN 位宽、log/constant
利用率模型、HBM 容量/带宽、双 head-tail）→ 库 `ChipCapabilities`（log_a/log_b → LogUtilization k/b）。
`npu_910b3()` 保留不动。`dv120-topo.txt`（通信拓扑）与 L2cache 未消费，属后续通信建模任务。

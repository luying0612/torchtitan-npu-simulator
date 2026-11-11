# Multi-Head-MoE（SLA2）接入审查报告 — 第三轮：无基准校准执行结果

> 审查日期：2026-11-06。本轮执行"无基准校验方案"的 P0 项（黄金数学等价 + CPU dispatch 对照 + 内存 oracle），
> 并回答三个问题：从哪些角度验证了什么、哪些已证明准确、哪些仍需开发。
> 第一、二轮见文末摘要。校准脚本佐证：`_mhoe_scratch/calib_golden_equivalence.py`、
> `calib_locate_divergence.py`、`calib_dispatch_and_memory.py`。

## 总体结论

**校准发现并修复了一个 15 个捕获测试都无法发现的真实架构 bug（TransformerBlock 残差丢失）；
修复后模型的数学正确性、捕获保真、静态内存公式三个维度均已实证闭环。
算子耗时建模正在从"无 duration"迁往"operator-performance-models 库 roofline + dv120 档"
（并行会话在途，测试先行、实现未完成，1 个测试红着），这是当前唯一未闭环维度。**

## 一、本轮验证角度与结果

### 角度 1：数学等价（黄金参照法）—— 发现真实 bug ✅（修复后闭环）

方法：以修正版朴素参考实现为数学 oracle，同权重、同几何、CPU fp32 逐阶段对账。

| 步骤 | 结果 |
| --- | --- |
| 权重映射（123 组，含 (head,expert) 展平堆叠） | 拷贝零差 |
| embedding / attention_norm / rope 后 q/k/v | **零差** |
| SLA2 核心（同输入：router score、mask、双分支、alpha 融合） | **零差** |
| block 级对账 | **port block − x − attn − ffn = 0.196（恰一个 attention 幅值）→ 真 bug** |
| 修复后端到端 logits / loss | max_abs 1.7e-6（fp32 噪声级）/ loss 逐位一致 |
| 修复后全参数梯度（123 组） | max_rel ≤ 7.6e-5，0 个超 1e-3 |

**Bug 内容**：`model.py TransformerBlock.forward` 第二残差接回 `x_BLD` 而非 `h`（post-attention 隐状态），
attention 残差被丢弃、只经 FFN 分支间接影响输出。捕获测试全部绿灯（形状/计数/内存自洽）——
**这正是"捕获保真 ≠ 数学正确"的实证**，也证明了黄金参照法对无基准模型的必要性。
修复已入工作区（`h + moe(ffn_norm(h))`），黄金测试、ruff、pyrefly、端到端 PYEXIT=0 复验通过。

排查中排除的两个 harness 陷阱（非移植错误，已记录）：参考实现 `InnerBlockSLA2` 惰性创建于首次
forward，`eval()` 后物化的子模块 training=True、其 router bias 逃过构建期置零。

### 角度 2：捕获保真（CPU dispatch trace vs Meta 捕获图）✅

方法：同一模型代码在 CPU 真实执行（TorchDispatchMode 记录 ATen 算子）与仿真捕获图（world=1）
逐算子族对照。

| 算子族 | CPU 真实执行 | Meta 捕获 | 判定 |
| --- | --- | --- | --- |
| `aten._grouped_mm` | fwd 12 / bwd 24 | fwd 12 / bwd 24 | **精确一致** |
| rms_norm | F.rms_norm 分解（mean/pow/rsqrt） | `npu_rms_norm` 33 = 头16+块16+final1 | override 精确对应层结构 |
| RoPE | view_as_complex ×16 | `npu_rotary_mul` 16（8 层×q/k） | 精确对应 |
| SLA2 分支核心 | eager 数学（masked_fill/softmax/bmm 组） | 虚拟融合 fwd 16 / bwd 16 | 按设计替换，计数吻合 |
| 差异项 | — | nll_loss×8、rand_like×8、`_to_copy` 增量 | 全部可归因：CE loss、训练态 gumbel、override/bf16 生产路径 |

结论：**Meta 捕获图 = 真实执行同一代码的算子集合（在声明的 override/虚拟化边界内）**。

### 角度 3：静态内存 oracle ✅

真实 CPU 模型 + 真实 AdamW 实测：persistent 参数字节 253,580,416 与公式精确一致；139 张量精确一致；
矩量 nparams×8 吻合（step 标量记账 CPU/NPU 约定不同，差 556B 量级，公式按 fused 契约，合理）。

### 角度 4：既有验证（第一、二轮）延续有效

参数公式手工推导、47 存量失败零回归、EP2 精确 20、CP/TP/MTP 拒绝、seq128 非退化块选择、
CANN 探针（block=64 需 128 对齐）、硬件样例链路（324 mm 节点）、原始参考 3 bug 实锤——全部保持成立。

## 二、当前已证明准确的部分

1. **模型数学**：移植版与参考实现逐参数梯度等价（≤7.6e-5），含 MoE 展平、gate、SLA2 双分支、RoPE、alpha。
2. **捕获图/trace/算子**：算子集合与真实执行同构，override 与虚拟融合边界清晰、可全部归因。
3. **静态内存**：persistent/optimizer 公式与真实分配精确一致（139 张量、nparams×4/×8）。
4. **通信结构**：EP2 a2a 精确 20 个、FSDP 梯度清单 139 完整（结构级；字节数未单测，见下）。
5. **并行边界**：TP/CP/MTP fail-fast 有测试；FSDP/HSDP/full-AC/selective-AC 捕获矩阵全绿。
6. **工程纪律**：零回归、lint/type 干净、文档/暂存纪律符合仓库规则。

## 三、仍需开发或改进（按优先级）

1. **SLA2 耗时建模（在途，唯一红灯）**：正迁往 operator-performance-models 库 roofline
   （sparse/linear × fwd/gradient 四语义族 + dv120 档）。当前 `test_...routes_sla2_fusions_through_the_library`
   红着（期望 duration>0，实现未跟上；已验证与本轮 bug 修复无关——还原修复后同样失败）。
   完成后建议在真实 910 卡上用 workload shape 清单 microbench `aclnnBlockSparseAttention` 做锚点校准，
   并把"duration ≥ max(flops/峰值算力, traffic/带宽)"包络不变量加进 validate_workload。
2. **通信字节对账**：a2a/AG/RS 的 bytes 注记尚无解析公式断言（结构计数已有）。半天工作量。
3. **黄金测试入仓**：`calib_golden_equivalence.py` 目前是 scratch 证据；建议裁剪为主包 CPU 测试
   （eval 模式 + 梯度对比），防回归。注意 harness 两个惰性初始化陷阱需随测试固化。
4. **真实 NPU 终局校验**（需求明确不做实测对比，列为后续）：移植模型本身可跑 torchtitan trainer，
   1 step + npu profiler 做层级对照（qwen35-real-bench 方法论）即可补上最后一环。
5. **数值边界提示**：无因果 mask、padding 泄入 linear 分支 norm 分母（忠实继承参考实现）——
   建议在文档 §2 加一句，供真实 kernel 数值对齐时知悉。
6. **并行会话协同**：工作区正被同时修改（dv120/库 roofline/`hardware.py`）。提交前需等该在途
   工作收敛并全绿，避免把红灯测试提交入仓。

## 附录：第一、二轮结论追溯

- 第一轮：交付真实、验证充分；P1 成本缺口 / P2 文档精度 / P3 参考溯源 / P4 覆盖度，共 10 项意见。
- 第二轮：P1–P4 全部落实（成本建模 flops/流量、文档修正、原始参考归档且 3 bug 实锤、测试补强 15 passed）；
  剩 A（§3 标量措辞矛盾）/ B（cost 测试注释强于断言）两个文档级瑕疵。
- 第三轮（本轮）：黄金校准发现并修复残差丢失 bug；数学/捕获/内存三维度闭环；耗时建模为唯一在途项。
  注：第二轮的 A 项（§3"标量进不去"措辞）在当前文档版本中仍待修正；B 项已被库 roofline 新测试取代。

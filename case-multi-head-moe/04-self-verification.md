# 04 · 自我验证准确性（验证矩阵与关键发现）

> 归档正本：`archive/MULTI_HEAD_MOE_ADAPT.md` §6（含 §6.1 黄金校准）、`archive/MULTI_HEAD_MOE_REVIEW.md`。
> 核心结论：**捕获保真 ≠ 数学正确**——黄金参照法发现了 15 个捕获测试都无法发现的真实架构 bug。

## 4.1 验证维度总表（最终状态）

| 维度 | 方法 | 结果 |
| --- | --- | --- |
| 参数公式 | 手工推导 vs 构建实测 | 63,395,104 参数 / 139 张量，精确一致 |
| 捕获矩阵 | pytest 参数化（最终 **14/14**） | single / FSDP2 rank0+rank1 / HSDP(4,2) / full-AC / selective-AC / EP2 / TP·CP·MTP 拒绝 / seq128 真实块选择 / 融合开关 |
| 数学等价（黄金参照法） | 修正版朴素参考实现为 oracle，同权重 CPU fp32 逐阶段对账 | **发现并修复 TransformerBlock 残差丢失 bug**；修复后 logits max_abs=1.7e-6，loss 逐位一致，123 组梯度 max_rel ≤ 7.6e-5 |
| 捕获保真 | CPU dispatch trace vs Meta 捕获图（同几何 world=1） | `aten._grouped_mm` fwd12/bwd24 两侧精确一致；override（rms_norm 33 / rotary_mul 16）精确对应层结构；差异项全部可归因（CE loss / gumbel / override） |
| 静态内存 oracle | 真实 CPU 模型 + 真实 AdamW 实测 | persistent 253,580,416 字节与公式精确一致；139 张量一致；AdamW 矩量 nparams×8 吻合（step 标量 CPU/NPU 记账约定差 556B，按 fused 契约） |
| 通信结构 | EP2 精确计数 + 字节对账 | a2a 精确 20 个 + 11 RS 逐节点 `comm_bytes` 独立重算精确断言 |
| 成本精确值 | 库口径闭式断言 | seq16/seq128 精确 flops（见 03 文档 §3.4）+ 包络不变量 |
| 耗时包络 | 融合节点下界断言 | `duration ≥ max(flops/cube, traffic/HBM)` 防路由回归 |
| lint/类型 | ruff check+format / mypy（simulator 侧曾用 pyrefly） | 干净（先有一次"check 通过≠format 通过"的教训，见 §4.4） |
| 回归基线 | 全量 suite 新旧失败清单逐条 diff | 47–48 个存量失败与干净 checkout 逐条一致，**零回归**（mxfp8/CLI geometry，5747770 基线预存） |
| opm 仓回归 | opm-dev pytest | **1793 passed 全绿** + mypy strict + ruff 全绿 |

## 4.2 黄金校准的关键发现（残差丢失 bug）

- **bug**：`TransformerBlock.forward` 第二残差错误接回块输入 `x_BLD` 而非 post-attention 隐状态
  `h`——attention 输出只留在 FFN 分支内部，等价丢掉 `h = x + attn` 残差项。
  定位手法：逐层对账（layer0 attention/FFN 输出与参考零差，block 输出差恰为一个 attention 幅值 0.196）。
- **为什么 15 个捕获测试发现不了**：形状/计数/内存全部自洽——捕获保真只证明结构，不证明数学。
  这是对"无基准模型必须做黄金等价校准"的实证。
- 修复后黄金测试覆盖：MoE (head,expert) 展平 vs 参考逐头循环、gate/门控 top-k、SLA2 双分支与
  alpha 融合、RoPE（ComplexRoPE 与参考 freqs_cis 约定逐字一致）、块池化/路由。
- 排除的两个 harness 陷阱（非移植错误，记录防复发）：参考实现 `InnerBlockSLA2` 惰性创建于首次
  forward，`eval()` 后物化子模块 training=True（需物化后重新 eval）；其 BlockRouter bias 逃过
  构建期置零（需物化后再置零）。
- 黄金测试入仓：**按 owner 指示忽略**（涉及 CPU 结果对比不予执行）；校准脚本保留于
  `_mhoe_scratch/calib_golden_equivalence.py` / `calib_dispatch_and_memory.py`。

## 4.3 测试计数演进（如实记录）

- 12 → 15（R1 补强：成本精确值 / seq128 真实选择 / CP 拒绝 / EP2 精确 ==20）。
- 保真整改后重写参数化：**14/14**（eager 默认 6 参数化 + EP2/TP/CP/MTP + eager 成本 +
  seq128 分块签名 + `test_multi_head_moe_opt_in_fusion_replaces_only_the_two_branch_seams`）。
  计数断言按实测钉死：masked_fill.Scalar=8×recompute+8、view_as_complex=16×recompute+16、
  topk=8×(2 if full-AC)；einsum 分解为 bmm。

## 4.4 验证纪律教训（防止复犯）

1. **"ruff check 通过"≠"ruff format 通过"**：一次汇报 ruff 干净后复查发现 2 文件需 reformat
   ——自验证汇报必须跑齐 check+format+type 三件套。
2. **全量失败必须对照干净基线**：临时 worktree 在 5747770 复现同一 48 失败集合，
   才能声明"与本改动无关"。
3. **内存断言公式必须可解释**：写死观察值时注释推导（FP32 驻留、奇 numel shard 偏移、
   SparseAdam stub 4496B 等），否则后续维护者无法区分 bug 与偏移。
4. **环境漂移会污染验证**：容器 `/workspace/opm-dev` 曾被并行会话回退到 `650b214`（丢失 SLA2），
   表现为成本路由静默失效；宿主机↔容器需重同步（`docker cp`）并确认 `import operator_performance_models`
   指向正确路径。

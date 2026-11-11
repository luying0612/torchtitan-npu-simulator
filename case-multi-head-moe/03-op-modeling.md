# 03 · 算子建模（SLA2 → 算子建模仓 opm-dev）

> **正本位置**：SLA2 算子建模的全部代码/测试/文档在算子建模仓
> `/home/l30079348/opm-dev`（分支 `gdn-sla2`，提交 **`0dd983e`** "feat(sla2): model SLA2 sparse/linear
> attention branches for library routing"，9 文件 +724，已推送远端 `ZhanluModelSim/operator-performance-models`）。
> 本目录只记录过程与指针，**不复制建模代码**。分支在本次提交后由并行会话继续推进至 `21a35b6`。

## 3.1 演进路线（三轮架构指示）

1. **R1 成本缺口**：SLA2 两个虚拟融合的 32 个捕获节点全 `cost_unknown`，违反需求
   "计算算子建模（通信除外）"。先探针验证 **custom_op 的 schema 标量在捕获时进入 `node.attrs`**
   （flex_attention 实测：scale/enable_gqa 在 attrs）——推翻"标量进不去虚拟算子"的判断；
   随后在 simulator 侧实现 flops/traffic handler（含两次返工，见 `05-review-revisions.md` §2.3）。
2. **指示"算子建模只认 operator-performance-models 仓"**：simulator 侧自建 handler 属待废弃路径；
   范围明确为"库中已有的族不重复迁移，仅迁移本模型用到而库中没有的（SLA2）"。
3. **目标仓更换为 opm-dev + DV120 真实硬件**：按库仓规范（SourceAdaptor/OpSource/BaseOp/
   PrimitiveVecConfig）在 opm-dev 重建全部 SLA2 资产。

## 3.2 最终建模内容（在 opm 仓，此处为摘要）

- **语义 4 族**（`semantics/npu_fused.py` + `npu_gradients.py`）：sparse/linear × fwd/gradient；
  linear 族标量字段名 `complement_blocks`（源契约是补集块数），与 sparse 的 `selected_blocks` 区分。
- **Adaptor 4 个**（`adaptors/npu_fused.py` + `npu_gradients.py`）：4 个 SourceAdaptor +
  `_sla2_grid`/`_sla2_selected` 共享校验；grad 规范名 `Sla2*Backward`、输入顺序 `(q,k,v,mask,dy)`（dy 末尾）。
- **模型 4 个**（`models/roofline/fused.py`）：`_Sla2Roofline` 基类——FP32 cube GEMM 闭式公式 +
  声明式 vec（softmax 6/3、normalize 2/3 结构单位，字面量非标定系数）；组成
  `max(cube+vec, memory)+head_tail`（MHC 投影同款）。公式/流量表详见 opm 仓
  `docs/sla2-roofline-models.md`（副本见本目录 `archive/opm-sla2-roofline-models.md`）。
- **配置**：`roofline_config.py` PrimitiveVecConfig family 白名单 +4。
- **测试**：`tests/test_sla2_roofline.py`（adaptor 往返、4 模型精确工作量与 duration、
  配置门控、网格/越界拒绝）。

## 3.2.1 保真整改后的成本路由形态

- **默认 eager（开关关）**：SLA2 分支是原生 aten 小算子，成本走标准 aten handler；
  op_mapping/op_cost_model 中的 SLA2 虚拟算子条目仅在 `enable_sla2_fusion=True` 时被触达。
- **融合开启**：4 个虚拟算子经 `_sla2_library_cost` 库路由（OpFeatures + selected_blocks 标量 +
  `_canonical_dtype` 别名 + 显式声明 `VecOpConfig(32, 1.0, float32)`）；无硬件 → `cost_unknown`
  （库 roofline 需 chip capabilities，不做部分注记）；DV120 下 32/32 全部获得 duration。

## 3.3 迁移中确立的三个源契约（adaptor 层翻译，排查自"失败→复现→对源码核实"）

1. **grad 规范名是 `Sla2*Backward`**（库 OpSource），simulator 表里的 `...Gradient` 永远匹配不上
   ——表现为 SLA grad 节点静默 `cost_unknown`。
2. **grad 融合的捕获输入顺序是 `(q, k, v, mask, dy)`**（compat 签名即如此），adaptor 假设 dy
   首位导致形状校验错位。
3. **linear 融合的 `selected_blocks` 标量按源契约是"补集块数"**（compat docstring 明示；
   seq16 时 fwd=1 而 grad=0 证实）——语义字段改名 `complement_blocks`，模型按 `(complement+1)` 计。
   另：捕获 dtype 为 `bfloat16` 等规范名，`_canonical_dtype` 容错 `bf16` 简写。

## 3.4 seq_len 实测精确值（库口径，DV120）

- seq_len=16（补齐 T=64）：sparse fwd 4,292,608 / sparse bwd 10,633,220 /
  linear fwd 2,129,920（complement=0 合法退化）/ linear bwd 4,276,228。
- seq_len=128（nb=2，topk=1，非退化选择）：selected=1、complement=1，两融合 fwd flops 8,585,216 / 8,454,144。
- 包络不变量入测试：每融合节点 `duration_us ≥ max(flops/cube峰值, traffic/HBM带宽)`（防路由回归）。
- 结论边界：耗时来自 opm-dev 声明式粗粒度 roofline（与 mhc_post 同状态：结构公式无系数、
  无真实负载 trace 校准）；真机 `aclnnBlockSparseAttention` microbench 锚点校准超需求边界未做。

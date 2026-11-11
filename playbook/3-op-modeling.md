# 3 · 补充算子建模（优先复用算子建模仓已有逻辑）

> **仓内权威规则：`AGENTS.md`"修改归属"节——性能预测、硬件、cube/vec 分类、DES 与最终指标归下游，
> 不在训练仓恢复 cost 或编造耗时**（simulator 侧只做库路由）。建模正本仓：
> `/home/l30079348/opm-dev`（operator-performance-models，远端 ZhanluModelSim/operator-performance-models）。
> 样例：`case-multi-head-moe/03-op-modeling.md`。

## 3.1 复用优先：三步查询

1. **查已注册族**：读 opm 仓 `docs/cost-model-status.md`（adaptor/算子名/族计数与清单）+
   `src/operator_performance_models/models/roofline/*.py` 的 `_MODELS` 注册 + `models/_defaults.py`
   （默认入口发现的族）。
2. **语义匹配**：捕获到的 raw op（ATen 名）按语义族对号——elementwise(binary/scalar/unary)、
   where、clone/copy、reduction、softmax/log_softmax(+grad)、activation(+grad)、fused_adamw、
   rotary(+grad)、rms_norm(+grad)、matmul/bmm/grouped_mm、mhc_post、gdn(gated_delta_rule)、
   sla2 sparse/linear attention、moe token permute/unpermute/rerouting……已有族**只接路由，不写模型**。
3. **路由接线**（simulator 侧仅此而已）：`cost/op_cost_model.py` 把捕获 raw op 映射到库
   （OpFeatures + 规范名 + 显式声明 VecOpConfig），库有硬件档即出 duration。

**判定"实在没有才新写"**：语义族在库中不存在（样例：SLA2 分块稀疏/线性注意力无先例）才新写；
新写走 §3.2 规范，**禁止**在 simulator 侧私建 handler（历史指示：simulator 自建建模属待废弃路径，
只认库）。

## 3.2 新写规范（opm 仓，SLA2 先例）

- 分层：语义 dataclass（`semantics/`）→ SourceAdaptor（`adaptors/`）→ Roofline 模型
  （`models/roofline/`），组成 `max(cube+vec, memory)+head_tail`。
- vec 段声明式结构单位（字面量，**非标定系数**）；所有系数零内置默认、显式配置必填
  （PrimitiveVecConfig 白名单注册）。
- 闭式公式优于标定（如 MHC 投影 2(n+1) 等效算子/元素，直接来自 kernel 源码结构）。
- 延迟模型（如 GDN per-token 延迟）常数必须来自真实负载 trace，空载 DVFS 不可标定要披露。
- 测试：adaptor 往返 + 精确工作量/duration 断言 + 配置门控 + 越界拒绝；全仓 pytest + mypy strict
  + ruff 全绿。
- 文档：`docs/<family>-roofline-models.md` 新增 + `cost-model-status.md` 计数更新 + AGENTS 覆盖声明。

## 3.3 源契约三查（新写 adaptor 必做，失败→复现→对源码核实）

1. **grad 规范名以库 OpSource 为准**（样例：库叫 `Sla2*Backward`，simulator 表叫 `...Gradient`
   ——永不匹配，表现为 grad 节点静默 unknown）。
2. **grad 捕获输入顺序以 compat 签名为准**（样例：dy 在末尾，adaptor 假设首位则形状校验错位）。
3. **标量的源语义与库字段名对齐**（样例：源契约是"补集块数"，库字段 `complement_blocks`；
   sparse 才是"选中数"）。另：dtype 用规范名（`bfloat16`），别名映射容错（`bf16`）。

## 3.4 验证

- 精确值断言：小几何下闭式手算对账（样例：seq16 sparse fwd 4,292,608 等 4 组）。
- 包络不变量：`duration ≥ max(flops/cube峰值, traffic/HBM带宽)` 入测试防路由回归。
- 无硬件 → `cost_unknown`（不做部分注记）；带硬件（DV120）全族获得 duration。
- 与保真路径解耦：默认 eager 走标准 aten handler，库路由仅在融合开关开启时触达。

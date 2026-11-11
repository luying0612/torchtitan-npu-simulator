# Multi-Head-MoE（SLA2）评审意见修改过程记录

> 配套文档：`MULTI_HEAD_MOE_ADAPT.md`（交接）、`MULTI_HEAD_MOE_REVIEW.md`（审查报告，含两轮结论）。
> 本文件按时间顺序记录每一轮"审查 → 修改 → 验证 → 再审查"的完整过程，含中间错误与修正，
> 供后续 reviewer 追溯每处改动的动机与证据。本文件位于仓库之外，遵守 simulator 仓库文档纪律。
> 记录截至 2026-11-05 晚（第二轮审查 A/B/C 处理前）。

## 总览

| 轮次 | 触发 | 产出 |
| --- | --- | --- |
| R0 | —（初版交付） | 模型接入 + 4 个虚拟融合 + 12 个测试，交接文档 v1 |
| R1 审查 | 第一轮评审 | P1–P4 共 9 条意见（见 §1） |
| R1 修改 | 评审优化轮 | P1 成本建模 / P2 文档精度 / P3 出处存档 / P4 测试补强，测试 12→15 |
| R2 审查 | 第二轮评审 | P1–P4 逐项核销 ✓；剩 A/B/C 三个文档级小项（待处理，见 §5） |
| R2 后更正 | 用户质疑 duration 结论 | 澄清 vec/GDN duration 路径，交接文档 §7 重写（无代码改动） |

## 1. 第一轮审查意见（输入）

- **P1**：SLA2 两个虚拟融合算子无计算建模（捕获图中 32 个节点全 `cost_unknown`），
  与基线 GDN/mhc_post 先例不一致，违反需求第 2 条"计算算子建模（通信除外）"。
- **P2**：三处"干净/0 错误"表述强于事实（ruff/pyrefly 存量计数未限定范围）。
- **P3**：原始参考代码不可溯源（/tmp 已清理），3 处 bug 修正无法对照。
- **P4**：块选择仅形状级覆盖（seq_len=16 单 block）/ CP 拒绝无测试 / EP2 断言弱
  （`> 0`）/ 偶数 numel 前提未注明 / padding 泄入 norm 未提示 / "多模态"术语需澄清。

## 2. R1 修改：P1 成本建模（核心改动，含两次返工）

### 2.1 机制探针（先验证再动手）

- 确认 PyTorch custom_op 的 **schema 标量参数在捕获时进入 `node.attrs`**
  （用 flex_attention 探针实测：`scale=0.125`、`enable_gqa=True` 出现在 attrs）。
  这推翻了我此前"标量进不去虚拟算子"的判断——只有*语义声明*必须覆盖捕获张量，
  标量可游离于张量语义之外。

### 2.2 代码改动（文件逐一）

- `torchtitan_npu/models/multi_head_moe/model.py`：两个 seam（`run_sla2_sparse_attention` /
  `run_sla2_linear_attention`）增加 `selected_blocks: int` 形参；调用点分别传
  `topk_blocks` 与 `nb - topk_blocks`（sparse 选中数 / linear 补集数）。
- `experiments/simulator/torchtitan_simulator/compat/multi_head_moe.py`：
  4 个算子（fwd + grad）schema 增加标量；`_sparse_setup`/`_linear_setup` 改为
  `save_for_backward(*inputs[:4])` 存 4 张量、标量挂 `ctx.selected_blocks`；
  3 条 bridge 更新调用。
- `experiments/simulator/torchtitan_simulator/capture/op_mapping.py`：新增 4 条
  raw→op_type 映射（fwd/grad 同名映射，沿用 kimi chunk_kda 先例）。
- `experiments/simulator/torchtitan_simulator/cost/op_cost_model.py`：
  新增 `_sla2_common_shapes`（q/k/v [B,H,T,Dh] + mask [B,H,qb,kb] 校验）、
  `_sla2_sparse_attention_cost`、`_sla2_linear_attention_cost`（只给
  flops + `memory_traffic_bytes`，**不造 duration**）；`compute_cost` 增加
  `attrs` 参数并经 `_accepts_attrs`（按签名分发）透传；`_annotate_node` 补
  `attrs=node.attrs`。

### 2.3 返工记录（错误 → 修正）

1. **反向梯度数量错误**：首跑报 `returned an incorrect number of gradients
   (expected 5, got 4)`——schema 加标量后 backward 契约变为 5 输出。
   修正：两个 `_backward` 末尾追加 `None`（标记标量不可导）。
2. **grad 算子被误判 unknown**：首跑 8 节点有 flops、24 个 unknown。诊断出 3 处：
   - `_sla2_common_shapes` 要求 `len(outputs)==1`，把 4 输出的 grad 算子整体排除
     → 重构为只校验输入张量，outputs 计数下放到各 handler 按 fwd/bwd 分支判断；
   - backward 判定误写 `len(inputs)==6`（以为 dy 使张量数变 6）——实际标量
     **不是捕获张量**，bwd 张量输入是 5（4 + dy）→ 改为 `== 5`；
   - seq_len=16 单 block 时 linear 补集数 = 0 是**合法退化**，被 `selected > 0`
     拒绝 → linear 放宽为 `0 <= selected <= kv_blocks`（sparse 保持 `> 0`）。
3. **探测脚本键名错误**（非代码 bug）：我一度用 `traffic_bytes` 读注记，实际键名
   是 `memory_traffic_bytes`，造成"traffic 为 None"的误判，查常量后澄清。
4. **pyrefly `bad-argument-count`**：`_HANDLERS` 声明为 3 参 `Callable[[...], ...]`
   与 4 参分发调用冲突（`_gdn_cost` 3 参混在联合里）→ handler 表宽化为
   `Callable[..., CostEstimate]` + 注释说明运行期由 `_accepts_attrs` 守卫。

### 2.4 修后验证（实测值）

4 种算子全部获得精确注记、无 `cost_unknown`、无 duration（seq_len=16，补齐 T=64）：

| 算子 | flops | traffic |
| --- | --- | --- |
| sparse fwd | 4,194,304 | 131,080 |
| sparse bwd（2×） | 8,388,608 | 229,392 |
| linear fwd（complement=0 退化） | 2,097,152 | 131,080 |
| linear bwd（2×） | 4,194,304 | 229,392 |

## 3. R1 修改：P3 出处存档 + P4 测试补强 + P2 文档精度

- **P3**：重新克隆参考仓至 `/tmp/opencode/inference-models-simulator/`；pristine
  原版存档 `_mhoe_scratch/multi-head_moe_ORIGINAL.py`；scratch 版文件头改为
  "3 处 bug 修正"注释（替换原"仅本类修改"的不实声明）；回归失败清单归档
  `_mhoe_scratch/regression_failed_{with_changes,clean_checkout}.txt`。
- **P4**（测试 12→15，全过）：
  - 新增 `test_multi_head_moe_cost_model_annotates_sla2_fusions_without_fabricated_duration`：
    4 算子 flops 精确断言（含推导注释）+ traffic > 0 + 无 duration；
  - 新增 `test_multi_head_moe_seq_len_128_exercises_real_block_selection`：
    nb=2 真实选择路径（selected=1 / complement=1，两融合 fwd flops 均 8,388,608）；
  - 新增 `test_multi_head_moe_rejects_context_parallel`（world=2 + CP=2 →
    `NotImplementedError`，先 grep 确认报错文案）；
  - EP2 断言 `> 0` 改精确 `== 20`（先实测确认 20 稳定）；
  - 内存公式前加"world 为 2 的幂且均分参数"前提注释；
  - 既有直发/语义测试的算子调用补标量实参。
- **P2**：交接文档 §5–§8 更新：ruff 存量限定为"4 个未触碰文件的 I001/B905×2/RUF036"；
  pyrefly 表述改为定向检查剩 2 个存量类错误；§8 登记 P1–P4 落实与多模态证据
  （参考仓 README 明确 text-only，`forward(tokens)` 仅收 tokens）。

## 4. 第二轮审查结论（输入）与本轮 session 的更正

- **R2 审查**（`MULTI_HEAD_MOE_REVIEW.md`）：P1–P4 逐项核销 ✓，15 passed 独立复现，
  容器内复算与文档数字逐项吻合；剩 A/B/C（见 §5）。
- **duration 路径更正（本 session，无代码改动）**：我最初向需求方报告"样例硬件
  算不出 vec/GDN 耗时（缺 `VecRooflineConfig` → `MissingCostmodel`）"——**错误**，
  混淆了平台侧消费路径与 simulator 自带路径。经用户质疑后实测澄清：
  - simulator 的 `op_cost_model.py` **自带** `_vec_duration`（max(vec,memory)+overhead，
    族级声明 equivalent-ops）、`_cube_duration`、GDN 延迟 handler（trace 常数，
    qwen3.5 实测 412.672us/grad 节点）；只有 matmul 走 operator-performance-models 库；
  - example.json 与 `npu_910b3()` 对 multi_head_moe 的注记结果**逐项一致**
    （1518 duration / 1867 unknown / 32 个 SLA2 无 duration）——证明 vec 覆盖与
    系数配置无关，样例硬件即可算出 mm/vec 耗时；
  - unknown 的真实构成：view/t/detach/transpose/slice 等**零计算布局算子** +
    stream_wait/profiler 事件，设计如此非缺口；
  - 交接文档 §7 对应段落已按实测事实重写（原错误表述不留档于交接文档，
    过程留档于本节）。

## 5. 第二轮审查 A/B/C 处理（2026-11-05 晚，已落实）

- **A（文档 §3 自相矛盾）**：`MULTI_HEAD_MOE_ADAPT.md` §3 原"虚拟算子签名只收
  tensor……标量进不去"改为："虚拟算子签名除捕获张量外可收 schema 标量，
  `selected_blocks` 以标量参数进入捕获节点 `node.attrs`（flex_attention 同机制），
  不作为捕获 tensor 声明语义角色；backward 为标量返回 `None` 梯度"；
  1/sqrt(Dh) 缩放留在边界外的理由同步改写（常数乘不进融合边界）。
- **B（测试硬件断言）**：成本测试在无硬件断言后追加 `annotate_workload_costs(
  workload, hardware=npu_910b3())` 重注记，断言 SLA2 节点带硬件仍无 duration
  （matmul 族正常获得 duration 作对照语境）；注释同步弱化为"无硬件与带硬件均无
  duration"。选 `npu_910b3()` 而非 example.json 是为避免测试引入 sim_platform
  跨包依赖。
- **C（数学瑕疵提示）**：交接文档 §8 新增条目：参考实现无因果 mask、padding 位置
  softmax(0) 均匀泄入 linear 分支 norm 分母（分子不受影响，phi_v=0），本接入
  忠实继承未修正，生产化数值对齐时需知。
- 验证：pytest 15 passed（B 修改既有用例，未新增用例数）；ruff/pyrefly 增量干净；
  全部重新 `git add` 暂存。

## 5b. 架构指示记录：算子建模只认 operator-performance-models（待执行）

需求方指示（2026-11-05 晚）：**simulator 侧算子建模全部废弃，只能使用
operator-performance-models 仓**。在该前提下"vec 和 GDN 没有耗时"成立，原因是
模型库的显式配置纪律：所有计算系数与延迟常数均为显式配置、无内置标定默认值——
vec 族 17 个模型系数经 `VecRooflineConfig` 传入，None 即不注册（`MissingCostmodel`）；
GDN 为延迟模型，`GatedDeltaLatencyConfig` 常数为显式必填且须来自真实负载
（空载 DVFS 降频 ~3 倍不可标定）；example.json 只有 chip capabilities、不含模型系数。
此前 qwen3.5 实测中 vec/GDN 的 duration 全部来自 simulator 自带 handler
（`_vec_duration`/`_gdn_cost` 硬编码 trace 常数），按此指示属待废弃路径。

待明确的执行范围（影响面从大到小）：
1. 仅迁移本次 SLA2 建模（`_sla2_*` handler 移入 operator-performance-models，
   仿 mhc_post 闭式公式先例，simulator 改为查库）；
2. 全量迁移 simulator cost 层（qwen3.5/kimi 等全部算子族 handler）——独立大任务，
   需单独排期与各模型回归；
3. 暂只在本套文档登记方向，代码不动。

## 5b. 架构迁移执行：SLA2 建模迁入 operator-performance-models（2026-11-05 晚）

需求方明确范围：库中已有的族不重复迁移，**仅迁移本模型用到而库中没有的（SLA2）**，
并按库仓规范重写。执行记录：

- **发现**：库目录在会话中被整体替换（约 17:05）——.git 丢失、AGENTS.md/vec 文档的
  qwen3.5 更新节消失、GDN 模型不存在（早先文档声明已实现）；src 呈 DeepSeek-V4
  基线 + 大量新增族（sparse/mhc/moe 等）。按当前 src 基线继续。
- **库侧（按 SourceAdaptor/OpSource/BaseOp/PrimitiveVecConfig 规范）**：
  语义 4 族（semantics/npu_fused.py + npu_gradients.py）、adaptor 4 个（含
  `_sla2_geometry` 共享校验）、模型 4 个（`_Sla2Roofline` 基类：cube GEMM 闭式
  公式 + 声明式 softmax 6/3、normalize 2/3 结构单位；组成
  `max(cube+vec, memory)+overhead`，与稀疏注意力族一致）、
  PrimitiveVecConfig 白名单 +4、`tests/test_sla2_roofline.py` 7 用例、
  `docs/sla2-roofline-models.md` + cost-model-status + AGENTS 覆盖声明。
  顺手修复存量缺失导入 `OptionalTensorOperands`。
- **simulator 侧**：删除 `_sla2_common_shapes`/两个公式 handler，新增
  `_sla2_library_cost` 库路由（OpFeatures + selected_blocks 标量 + 显式声明
  `VecOpConfig(32, 1.0, float32)`——结构单位字面量，非标定）；无硬件 →
  `cost_unknown`（不做部分注记）；**发现并修复** `_library_matmul_duration`
  对库新 API（构造期 semantics/hardware、predict 无参）的静默失效——修复后
  mm duration 恢复，example.json 下 multi_head_moe duration 节点 1518 → 1550
  （+32 恰为全部 SLA2 融合）。
- **验证**：库侧 pytest 1743 passed（7 个新用例全过；9 个 import 隔隔离失败为
  `__init__.py:33` 顶层导入 models._defaults 的存量问题，替换前即存在）；
  ruff 全绿（6 个 markdown 格式告警为容器 ruff ≥0.16 与 HEAD pin <0.16 的既存
  差异）；mypy 容器未安装。simulator 侧 15 passed（成本精确值断言全部更新为
  库口径：4,292,608 / 10,633,220 / 2,129,920 / 4,276,228；seq128 8,585,216 /
  8,454,144），ruff/pyrefly（op_cost_model.py）0 错误。
- **"无 duration"结论修正**：库建模后 SLA2 **有 duration**（声明式粗粒度
  roofline，与 mhc_post 同状态：无真实负载 trace 校验）；此前"无时延锚点不造
  duration"是 simulator 侧纪律，库的 coarse-v1 声明策略取代之。交接文档
  §3/§5/§6/§7/§8 已同步。

## 5c. 迁移目标更换：opm-dev 仓 + DV120 真实硬件（2026-11-05 深夜）

需求方两项新指示：算子建模仓库改为 **`/home/l30079348/opm-dev`**（git 管理的真仓，
领先旧副本两个修复提交）；硬件改用 **`/home/l30079348/hardware`** 的 DV120 规格。
执行记录：

- **环境再刷新**：旧库目录再度被替换（我先前写入的 SLA2 文件不在盘上）——按交接
  文档记录的规格 + 会话锚值**在 opm-dev 重建**全部 SLA2 资产，并按 opm-dev 规范
  重写：`_FusedRoofline` 的 `is_fused`/`requires_vector_config` 门控、
  `config.compute_dtype` 作 cube_flops 键、组成沿用 MHC 投影同款
  `max(cube+vec, memory)+head_tail`（`_MhcProjectionRoofline` 先例）。
- **接线**：`docker cp /home/l30079348/opm-dev → /workspace/opm-dev`（容器内该路径
  原是不完整副本）+ `pip install -e /workspace/opm-dev`（替换 site-packages 的旧
  普通安装）；此后 host 编辑需重新 docker cp 同步。
- **三个源契约发现**（均由失败→复现→对 capture/compat 源码核实）：
  1. grad 规范名是 `Sla2*Backward`（库 OpSource），simulator 表里的
     `...Gradient` 永远匹配不上——表现为 SLA2 grad 节点静默 `cost_unknown`；
  2. grad 融合的捕获输入顺序是 `(q, k, v, mask, dy)`（dy 末尾，compat 签名即如此），
     adaptor 假设 dy 首位导致形状校验错位；
  3. **linear 融合的 `selected_blocks` 标量按源契约是"补集块数"**（compat
     docstring 明示；seq16 时 fwd=1 而 grad=0 证实）——库 linear 语义字段改名
     `complement_blocks`，adaptor 做源名→语义翻译，模型按 `(complement+1)` 计。
     另：捕获 dtype 是 `bfloat16` 等规范名，但加了 `_canonical_dtype` 别名映射
     以容错 `bf16` 简写。
- **硬件**：`cost/hardware.py` 新增 `dv120_hardware()`——DV120 JSON（compute
  按 floatN 位宽的 TFLOPS 峰值、log_a/log_b/min/max 利用率、HBM、双 head-tail）
  → 库 `ChipCapabilities`（log_a→k、log_b→b）；sim-platform 加载器只认 schema v1
  不识别 DV120 形态，故解析放 simulator 侧（只翻译不标定）；topo 与 L2 不消费。
- **验证**：opm-dev 侧 **1793 passed 全绿**（含 6 个 import 隔离测试，无既有失败；
  ruff check/format 全绿）；simulator 侧 **15 passed**（成本精确值改为 DV120+库
  口径；DV120 下 SLA2 32/32 全部获得 duration；非 SLA2 计数 1518/1867 与基线
  一致，qwen3.5 1831/1655 无回归）；ruff/pyrefly 0 错误（容器 ruff ≥0.16 对
  2 个触碰文件的 format 建议与仓库 pin <0.16 冲突，属既有 artifact 不采纳）。
- **交付状态**：opm-dev 改动未提交（host git 仓，由需求方提交）；simulator 3 个
  触碰文件重新暂存。

## 5d. 第三轮审视意见处置（2026-11-06）

`MULTI_HEAD_MOE_REVIEW.md` 第三轮（无基准校准执行结果）6 项意见的处置记录：

- **项 1（SLA2 耗时建模，当时唯一红灯）**：红灯即本会话 §5c 迁移的在途状态，
  已收敛（15 passed、DV120 下 SLA2 32/32 duration）。可落地部分"包络不变量
  duration ≥ max(flops/cube峰值, traffic/带宽)"已加入 SLA2 路由测试；真机
  `aclnnBlockSparseAttention` microbench 超出需求边界（需求第 2 条不做实测），未做。
- **项 2（通信字节对账，半天）**：已实现。`_communication_cost` handler（a2a/
  reduce_scatter：逻辑跨卡体量 in+out 各一次）+ `comm_bytes` 注记键 +
  unknown 判定计入 comm_bytes（带字节不再是 unknown）；EP2 测试对 20 a2a +
  11 RS 逐节点从捕获张量事实独立重算精确断言。单卡负载无通信节点，基线计数
  （1518/1867）不变；duration 需链路模型（CollectiveRooflineConfig + 拓扑），
  与 dv120-topo.txt 同属后续通信建模任务。
- **项 3（黄金测试入仓）**：**按需求方明确指示忽略**——涉及 CPU 结果对比，
  不予执行；曾已将参考实现与校准逻辑移植为 `tests/test_multi_head_moe_golden.py`
  （实测在仓环境 logits 偏差 9.0e-5 > scratch 报告的 1.7e-6，未达 1e-5 预算，
  未及排查即收到指示），文件已删除不入仓；校准证据保留在
  `_mhoe_scratch/calib_golden_equivalence.py`。
- **项 4（真机 NPU 终局校验）**：需求第 2 条明确不做实测对比，未做。
- **项 5（数值边界提示）**：已加入交接文档 §2（无因果 mask、padding 泄入
  linear norm 分母），与 §8 既有条目呼应。
- **项 6（并行会话协同）**：在途工作即本轮 §5c 迁移，已收敛全绿后统一暂存。
- 附录"§3 A 项措辞仍待修正"系审查者读取旧版文档所致，当前 §3 已是修正后措辞。
- 验证：simulator 15 passed（含新增包络/通信断言），ruff check/pyrefly 0 错误。

## 6. 当前状态快照

- 已暂存未提交 13 个文件（`git add`，模型包 5 + compat 1 + capture 1 + cost 1 +
  测试 1 + simulator 注册 3 + 能力矩阵 1）；A/B/C 处理后重新暂存。
- pytest `tests/test_multi_head_moe_capture.py`：**15 passed**（多次独立运行
  81–84s；第二轮审查复现 83.92s；A/B/C 后含硬件重注记断言）。
- ruff：触碰文件增量干净；pyrefly：主包 0 错误，触碰 simulator 文件剩 2 个存量类错误。
- 端到端 CLI PYEXIT=0；example.json 硬件链路 1518 duration 注记验证通过。
- 开放架构项：§5b 算子建模仓库迁移指示，执行范围待需求方明确。

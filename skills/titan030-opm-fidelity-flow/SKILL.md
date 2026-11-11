---
name: titan030-opm-fidelity-flow
description: 在 torchtitan-npu-simulator 0.3.0（titan030）上接入新模型并完成"算子建模入库 operator-performance-models + 保真验证 + 审视修改 + sim-platform/真机算子对比"的全流程。覆盖源码保真铁律（禁止未经批准的算子融合）、Meta 捕获能力探针、SLA2 式库建模三源契约、黄金参照法校准（能发现捕获测试发现不了的数学 bug）、sim-platform 导入手术、实测算子对比方法论（含 repeat_interleave 类复合算子分解缺口）。当用户要求"迁移新模型并做算子建模入库 / 保真整改 / 自验证 / 仿真与实测算子差异对比"时使用。基础适配步骤（preset/compat/测试骨架）见 titan030-sim-model-adapt skill，本文不重复。
---

# titan030 新模型全流程：保真迁移 + opm 算子建模 + 验证 + 对比

> 多头 MoE（SLA2）实战沉淀。**通用五阶段 playbook**：`/home/l30079348/new-model-doc/playbook/`
> （0 原则 / 1 迁移 / 2 缺失验证 / 3 建模复用优先 / 4 验证审视 / 5 端到端对比）——先读 playbook，
> 本文是其 AI 执行细节版；本次工作完整样例：`/home/l30079348/new-model-doc/case-multi-head-moe/`。

## 0. 保真铁律（最高优先级，违反即返工）

1. **模型实现必须与给定源码一模一样**。禁止自行以融合/虚拟算子替换实现；RoPE、RMSNorm、
   score 缩放位置、分支内算子序列必须逐字还原源码。
2. 融合仅允许出现在**显式 seam 函数**中，且必须经开关（如 `SimulationConfig.enable_sla2_fusion`，
   **默认 False**）opt-in；融合范围先列清单交 owner 确认。
3. 允许的例外必须逐项获 owner 批准并留档：参考实现的真实 bug 修复（附最小复现）、
   Meta 无法捕获的算子（附探针证据，见 §2）、初始化缺失补齐。
4. 汇报验证结论必须跑齐 **ruff check + ruff format + 类型检查** 三件套（"check 通过"≠"format 通过"，
   有过实际返工）。

## 1. 全流程步骤

1. **探针先行**：任何"能不能捕获/能不能融合"的判断先写 Meta 最小探针实测，勿凭版本号臆断。
2. **迁移**（preset/config/compat/parallelize/sharding/测试）：基础步骤见 `titan030-sim-model-adapt`
   skill。本流程新增纪律：sharding 不能套通用 `set_moe_sharding_config`（会给 wrapper 注入不存在的
   状态声明）；模型自有 RMSNorm/RoPE 优先还原源码而非换框架实现。
3. **算子建模入库**（见 §3）：正本只认算子建模仓；simulator 侧只做库路由。
4. **自验证**（见 §4）：捕获矩阵 + 黄金参照 + 内存 oracle + 存量失败基线对照。
5. **审视修改**：每轮意见记录"意见→修改→验证"闭环；中途错误与修正留档（含错误结论本身，
   如 duration 路径误报），防复发。
6. **sim-platform 端到端 + 真机对比**（见 §5）。

## 2. Meta 捕获能力探针结论（multi_head_moe 实测，可直接引用但勿外推）

| 算子 | Meta 捕获 |
| --- | --- |
| `torch._grouped_mm` / `npu_rms_norm` / `npu_swiglu` | OK |
| `npu_rotary_mul` | 无（已有 compat 桥） |
| `npu_moe_gating_top_k_softmax` | 无 |
| `npu_block_sparse_attention` | 有核但参数校验拒绝（blockShapeY 须 128 对齐） |
| `bincount` / `torch.where` 紧凑索引 | **失败**（无 fake impl / 数据依赖输出形状）——MoE 逐专家循环无法捕获，GMM 替换的依据 |
| `topk` | OK |
| **复合隐式算子（如 `repeat_interleave`）** | **被分解后才到达 TorchDispatchMode**（expand+reshape）——真机直连 ACLNN 核则不分解，形成系统性缺口（见 §6） |

## 3. 算子建模入库 opm 仓（SLA2 先例）

- **规范**：语义 dataclass（semantics/）→ SourceAdaptor（adaptors/）→ Roofline 模型（models/roofline/）
  组成 `max(cube+vec, memory)+head_tail`；vec 段声明式结构单位（字面量非标定系数）；
  PrimitiveVecConfig 白名单注册；系数零内置默认，显式配置必填。
- **先写探测脚本确认机制**：custom_op 的 schema 标量在捕获时进入 `node.attrs`（flex_attention 同机制），
  语义声明只需覆盖捕获张量。
- **三个源契约（SLA2 实例，模式可复用）**：① grad 规范名以库 OpSource 为准（`Sla2*Backward`，
  simulator 侧叫法可能永不匹配 → grad 节点静默 unknown）；② grad 捕获输入顺序以 compat 签名为准
  （dy 末尾）；③ 标量的源语义要与库字段名对齐（"补集块数" vs "选中数"）。
- **验证**：库侧 pytest + mypy strict + ruff 全绿；simulator 侧成本精确值闭式断言 + 包络不变量
  `duration ≥ max(flops/cube, traffic/HBM)`；无硬件 → `cost_unknown`（不做部分注记）。
- **保真整改后的形态**：eager 默认路径走标准 aten handler，库路由仅在融合开关开启时触达——
  建模入库与默认保真路径解耦。

## 4. 自验证矩阵（无基准模型）

1. 捕获测试参数化（single/FSDP rank0+1/HSDP/full-AC/selective-AC/EP/拒绝项）+ 精确计数
   （计数按实测钉死并注释推导；einsum 会分解为 bmm）。
2. **黄金参照法**（关键，捕获测试发现不了的 bug 靠它）：修正版朴素参考实现为数学 oracle，
   同权重 CPU fp32 逐阶段对账（权重映射 → embedding/norm/rope → 分支核心 → 块级残差）。
   实战战绩：发现 TransformerBlock 第二残差接回 `x_BLD` 而非 `h` 的架构 bug（逐层对账定位：
   前序零差、block 输出差恰一个 attention 幅值）。陷阱：惰性创建的子模块 `eval()` 后仍 training=True、
   物化后 router bias 逃过构建期置零——物化后需重新设置。
3. CPU dispatch trace vs Meta 捕获图逐算子族对照（差异项必须全部归因）。
4. 内存 oracle：真实 CPU 模型 + 真实 AdamW；公式先实测再写死并注释推导。
5. 回归：全量 suite 失败清单与**干净基线**（临时 worktree 复现）逐条 diff，才能声明零回归。
6. 环境漂移检查：容器内 editable 副本可能被并行会话回退（opm-dev 事故）——成本路由静默失效时
   先查 `import operator_performance_models.__file__` 与提交号。

## 5. sim-platform 端到端（原 simulator 无端到端耗时）

- 入口 `sim-platform train <toml>`（`sim_platform.cli.training:main`）；toml：[training]
  config/world_size/dp_shard/ep/local_batch_size/seq_len/output_dir + [platform] hardware + [prediction] backends=["roofline"]。
- **导入手术**（模型不在平台默认环境时）：剥离 editable finder → `sys.path.insert` 指向目标仓 →
  **预导入** `torchtitan_simulator` + `torchtitan_npu`（缓存 sys.modules）→ 再调 CLI main。
- 产物：`runs/<name>/l1/`——`statistics/operators_by_name.csv`（归一化算子族）、`training.json`
  （end_to_end_us）、`000_F/001_B/002_OPTIMIZER.csv` 原始捕获、trace。

## 6. 实测对比方法论（只比计算算子，不比耗时/通信）

1. 实测容器：宿主 `npu-smi` 全卡无进程 ≠ 可用——**容器级 dcmi/aclInit 持有**会导致 507899；
   解法 = 同镜像同挂载新建 privileged 容器刷新设备会话。scipy 1.18 与 numpy 1.26 不兼容
   （np.long）→ 回退 1.13.1。
2. 运行：`NGPU=2` 脚本 + profiler freq4/warmup2/active1 → `kernel_details.csv`（每 rank 一份）。
3. 聚合：健壮自写脚本（引号变体会击穿模板脚本）；按 active step 过滤。
4. 对比：real ACLNN kernel 族 ↔ sim 归一化算子族，先建映射表再报缺口；已知粒度映射别当缺口
   （fused_adamw 1↔N、cast 逐算子插入、MoE dispatcher 路径差异由虚拟算子覆盖语义）。
5. **先核 dtype 口径**：真机栈默认 bf16 混精度，仿真捕获可能 FP32——不影响算子种类但影响
   cast 计数与 impl 选择，报告必须声明。
6. **复合算子缺口**（repeat_interleave 类）：sim 侧查 trace/原始 csv（0 次）+ 真机侧查 kernel 名；
   修复 = `torch.library.register_fake` 提供直接 Meta 核 → op_mapping 加映射 → opm 加
   bytes 拷贝类 cost handler（read in + write out×repeats），不动模型源码；量化影响
   （kernel 数占比 + 设备时间占比 + 随 batch 的增长）后交 owner 决策。

## 7. 归档纪律

- 每阶段结束把文档归集到固定目录（本例 `/home/l30079348/new-model-doc/`：docs/ 梳理版 +
  docs/archive/ 原始逐字归档 + skills/ 可复用工作流），正本在哪个仓（如 opm 建模文档）要用
  指针明确标注，不复制代码只留记录。
- Git 状态如实记录：哪些已推送、哪些未提交（保真整改增量未提交也是状态）。

# 05 · 审视意见与修改过程（三轮审查 + 保真裁定）

> 归档正本：`archive/MULTI_HEAD_MOE_REVIEW.md`（审查报告）、`archive/MULTI_HEAD_MOE_REVIEW_CHANGES.md`
> （逐轮过程含中间错误与修正）、`archive/MULTI_HEAD_MOE_ADAPT.md` §8–§9。本文为梳理版。

## 5.1 轮次总览

| 轮次 | 触发 | 产出 |
| --- | --- | --- |
| R0 | 初版交付 | 模型接入 + 4 个虚拟融合 + 12 测试 + 交接文档 v1 |
| R1 | 第一轮评审（P1–P4 共 9 条） | P1 成本建模（attrs 标量机制，2 次返工）/ P2 文档精度 / P3 参考溯源存档 / P4 测试补强（12→15） |
| R2 | 第二轮评审 | P1–P4 逐项核销；剩 A（§3 标量措辞矛盾）/ B（硬件断言弱化）/ C（数学瑕疵提示）三个文档级小项，当晚落实 |
| R2+ | owner 质疑 duration 结论 | 澄清 vec/GDN duration 路径（simulator 自带 vs 库路由混淆），文档按实测重写 |
| R3 | 第三轮评审（无基准校准执行） | 黄金校准发现残差 bug；数学/捕获/内存三维度闭环；项 1 耗时建模收敛、项 2 通信字节实现、项 3 按 owner 指示忽略、项 4 超边界、项 5 数值提示入档 |
| 架构指示 | "算子建模只认 opm 仓" + 目标仓换 opm-dev + DV120 | SLA2 建模整体迁入算子建模仓（详见 `03-op-modeling.md`） |
| **保真裁定** | **owner："必须和我给你的源码一模一样的实现。严禁进行这样的操作"** | **恢复源码小算子实现 + 双版本 + 融合开关（默认关）**，见 §5.2 |

## 5.2 保真整改（最高优先级指令的执行记录）

**背景**：owner 指出实现中存在未经批准的算子替换（SLA2 分支小算子被替换为虚拟融合算子、
RoPE/RMSNorm 被替换为框架实现），要求与源码一模一样。

**偏差清单（自查确认）**：① SLA2 两分支核心 → 虚拟融合（compat 层）；② RoPE 复数乘 →
torchtitan ComplexRoPE；③ RMSNorm 小算子 → 框架 RMSNorm（npu_rms_norm）；④ score 缩放
div→mul 位置改变；⑤ MoE 逐专家循环 → grouped-GEMM。

**owner 三项决策**：
1. 保留小算子版本与融合版本，告知融合范围；
2. 设开关选择是否开启融合，**默认关闭**；
3. 小算子情况下逐专家仍替换成 GMM（owner 批准的例外：bincount/where 无法 Meta 捕获）。

**执行**：
- 删除 compat 融合 + runtime 钩子 + op_mapping/op_cost_model SLA2 接线 → 恢复 model.py 为
  源码逐字实现（本地 RMSNorm、`apply_rotary_emb` 复数乘法（unsqueeze(1) 修复）、缩放恢复
  matmul 后除法、`freqs_cis` 经 forward override 只算一次 + `precompute_freqs_cis`）。
- 恢复 seam 函数 `run_sla2_sparse_attention` / `run_sla2_linear_attention`，`InnerBlockSLA2.forward`
  调用它们；新增 `SimulationConfig.enable_sla2_fusion: bool = False`，runtime 钩子按开关门控；
  compat/op_mapping/op_cost_model 的 SLA2 条目自 git HEAD 恢复（仅融合开启时触达）。
- 保留的已批准修复：3 个参考 bug + router/gate 初始化（std 0.02）+ topk 每调用传入。
- 框架契约适配（非算子替换）：`RopeGeometry`（Decoder 配置契约）、RMSNorm `reset_parameters`、
  Meta autocast 守卫。

**验证**：14/14 捕获测试（含新增融合开关 seam 断言测试）；ruff check+format / mypy 干净；
opm-dev 1793 全绿；48 全量失败经临时基线 worktree 证实为 5747770 预存。

## 5.3 过程返工记录（防复发价值）

- **反向梯度数量**：schema 加标量后 backward 契约变 5 输出 → 两个 `_backward` 末尾追加 `None`。
- **grad 被误判 unknown**：`_sla2_common_shapes` 曾要求 outputs==1 排除 grad 算子；backward 判定
  误写 inputs==6（标量不是捕获张量，实际 5）；linear 补集 0 是合法退化被 `>0` 拒绝 → 放宽为
  `0 <= selected <= kv_blocks`。
- **探测脚本键名**：`traffic_bytes` vs 实际键 `memory_traffic_bytes` 造成误判。
- **pyrefly 签名冲突**：handler 表 3 参声明与 4 参分发冲突 → 宽化为 `Callable[..., CostEstimate]`。
- **"无 duration"结论修正**：最初报"样例硬件算不出 vec/GDN 耗时"是错的（混淆平台消费路径与
  simulator 自带路径）；实测澄清后文档按事实重写，原错误表述不留档于交接文档（过程留档于本文件）。

## 5.4 待办与边界（如实）

- 保真整改增量（9 文件 +348/−306）**未提交未推送**，待 owner 指示。合入路径（如要做）：
  提交 9 文件 → `multi-head-moe-wip` rebase 到当前 `titan030`（9c4454e，分支基于旧 `fca58c5`）→
  合入 titan030 → 推送。
- 真机 `aclnnBlockSparseAttention` microbench 锚点校准、`dv120-topo.txt` 通信链路模型、
  L2cache 消费：均超出本次需求边界，列为后续。
- 黄金测试入仓：按 owner 指示忽略。

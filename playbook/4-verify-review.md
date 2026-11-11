# 4 · 自我验证与审视并修改

> 样例：`case-multi-head-moe/04-self-verification.md`（验证矩阵）、`05-review-revisions.md`（逐轮过程）。

## 4.1 自验证矩阵（全绿才算交付）

| # | 验证项 | 方法 | 通过标准 |
| --- | --- | --- | --- |
| 1 | 捕获测试 | pytest 参数化（single/FSDP rank0+1/HSDP/full-AC/selective-AC/EP/拒绝项） | 全过；计数按实测钉死并注释推导（einsum 分解为 bmm、复合算子分解等已知规则计入） |
| 2 | 数学等价（黄金参照法） | 修正版参考实现为 oracle，同权重 CPU fp32 逐阶段对账（权重→embedding/norm/rope→分支核心→**块级残差**） | logits/loss 噪声级一致；全参数梯度 max_rel 达标 |
| 3 | 捕获保真 | CPU dispatch vs Meta 捕获图逐族对照 | 差异全部可归因（playbook 2） |
| 4 | 内存 oracle | 真实 CPU 模型 + 真实 AdamW 实测 | persistent/optimizer 公式精确一致 |
| 5 | 通信结构 | EP 精确计数 + comm_bytes 独立重算 | 逐节点精确断言 |
| 6 | 成本精确值 | 闭式手算 vs 库注记 | 精确一致 + 包络不变量 |
| 7 | 回归基线 | 全量 suite 失败清单 vs **干净基线**（临时 worktree 复现）逐条 diff | 集合一致 = 零回归 |
| 8 | lint/类型 | ruff check + **ruff format** + mypy/pyrefly | 全绿 |
| 9 | opm 仓回归 | opm-dev pytest + mypy strict + ruff | 全绿 |

黄金参照法要点（实战战绩：发现 15 个捕获测试都发现不了的 TransformerBlock 残差丢失 bug）：
逐层对账定位（前序零差、块输出差恰一个 attention 幅值 → 第二残差接错）；陷阱：惰性创建的子模块
`eval()` 后仍 training=True、物化后 bias 逃过构建期置零——**物化后重新设置**。

## 4.2 审视修改协议

1. **轮次闭环**：每轮"意见 → 修改 → 验证 → 再审查"完整记录（意见原文、改动文件、验证数据、
   中间错误与修正），产出独立过程文档。
2. **先核实意见**：审查者可能读的是旧版文档（样例：A 项措辞争议实为版本差）；先复现再改。
3. **错误结论也要留档**：自己报错过的结论（如 duration 路径误报）按实测澄清后，错误表述不留于
   正式文档，过程留档于过程文档。
4. **中间返工记录**（样例）：schema 加标量后 backward 契约变化、outputs 计数误判、合法退化被拒、
   键名读错、类型签名冲突——每条"错误 → 修正"对后续模型可复用。
5. **owner 指令最高优先级**：保真裁定类指令（恢复源码/开关/批准例外）立即执行并全量回归。

## 4.3 交付前检查单

- [ ] 验证矩阵全绿 + 数字与文档一致（文档口径复核：check≠format、存量计数限定范围）
- [ ] 干净基线零回归证明
- [ ] Git 状态如实（已推/未推/未提交分开列）
- [ ] 环境漂移检查（容器内 editable 副本是否被并行会话回退/替换：`import X.__file__` + 提交号核对）
- [ ] 交接/归档文档更新（含原始需求逐字、批准例外清单）

# case-multi-head-moe — 参考样例说明

> 本目录是 2026-11 **multi-head-moe（SLA2）适配的完整工作记录**，作为通用 playbook
> （`../playbook/`）的参考样例。通用流程不要从这里读——按 playbook 执行，遇到具体问题时
> 再来这里查同款问题的解法与返工记录。
> **给人看的总结先读 `SUMMARY.md`**（五阶段脉络、关键决策、能力边界、推送记录、待办、复验命令）。

## 与 playbook 五阶段的对应

| playbook 阶段 | 样例文档 |
| --- | --- |
| 1 迁移模型代码到仿真仓库 | `02-model-migration.md`（+ `01-original-requirements.md` 需求与 3 bugs） |
| 2 验证算子缺失/未捕获 | `04-self-verification.md`（CPU dispatch 对照节 + 探针结论）、`06` §6.4 |
| 3 补充算子建模（复用优先） | `03-op-modeling.md`（SLA2 为"库里没有才新写"的实例；复用实例见 `archive/WORKSUMMARY.md` qwen3.5 节） |
| 4 自我验证与审视修改 | `04-self-verification.md` + `05-review-revisions.md`（三轮审查 + 保真裁定） |
| 5 端到端运行并与实测对比 | `06-sim-vs-real-diff.md`（sim-platform + 实测 + repeat_interleave 缺口） |

## 样例期产物正本位置（不在本目录）

- SLA2 算子建模代码/测试/文档：opm 仓 `/home/l30079348/opm-dev`（gdn-sla2 @ `0dd983e`）。
- simulator 改动：worktree `mhm-wip-wt`（分支 multi-head-moe-wip @ `5747770` 已推 fork；
  保真整改 9 文件未提交）。
- 仿真/实测产物：`titan030-qwen3.5/{mhoe-platform-run, mhoe-real-bench}/`。
- 参考源码 pristine 存档：`titan030-qwen3.5/_mhoe_scratch/multi-head_moe_ORIGINAL.py`。

`archive/` 为当时的原始文档逐字归档（未改动），是 `01–06` 梳理版的信息正本。

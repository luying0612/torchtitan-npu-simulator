# 新模型适配通用文档（torchtitan-npu-simulator 0.3.0）

> 本目录是一份**通用文档**：下次有新模型需要适配时，按 `playbook/` 五阶段流程开发即可。
> 本周期 multi-head-moe（SLA2）适配的全部工作作为**参考样例**收于 `case-multi-head-moe/`。

## 使用方式（新模型适配时）

1. 读 `playbook/0-principles.md`（铁律 + 流程总览 + 环境）。
2. 按顺序执行五阶段，每阶段对照对应 playbook 文档的 checklist：
   `1-migrate`（迁移模型代码到仿真仓库）→ `2-capture-completeness`（验证算子缺失/未捕获）→
   `3-op-modeling`（补充算子建模，**优先复用算子建模仓已有逻辑，实在没有才新写**）→
   `4-verify-review`（自我验证与审视并修改）→ `5-e2e-real-diff`（端到端运行并与实测对比）。
3. AI 执行细节用 `skills/` 两个 skill（基础适配 / 全流程含保真·建模·对比）。
4. 每步卡住时查样例：`case-multi-head-moe/` 有同一问题在真实模型上的完整解法与返工记录。
5. 新样例完成后：把本次文档归档为 `case-<model>/`（playbook 不动，发现问题则迭代 playbook）。

## 目录结构

```
new-model-doc/
├── README.md                          ← 本文（通用入口）
├── playbook/                          ← 通用开发文档（模型无关，下次适配按此执行）
│   ├── 0-principles.md                铁律/五阶段总览/环境速查
│   ├── 1-migrate.md                   迁移模型代码到仿真仓库（来源判定/移植纪律/接线 checklist）
│   ├── 2-capture-completeness.md      算子缺失验证（CPU 对照/探针工具箱/unknown 扫描）
│   ├── 3-op-modeling.md               算子建模（复用优先三步查询/新写规范/源契约三查）
│   ├── 4-verify-review.md             自验证矩阵/审视修改协议/交付检查单
│   └── 5-e2e-real-diff.md             sim-platform 端到端/真机实测/对比口径/复合算子缺口
├── case-multi-head-moe/               ← 参考样例（2026-11 multi-head-moe/SLA2 适配全记录）
│   ├── SUMMARY.md                     给人看的总结（五阶段脉络/关键决策/能力边界/推送记录/复验命令）
│   ├── 01-original-requirements.md    原始需求逐字 + 环境 + 参考 3 bugs
│   ├── 02-model-migration.md          迁移实现/边界演进/交付物/Git 状态
│   ├── 03-op-modeling.md              SLA2 建模（正本在 opm 仓 gdn-sla2@0dd983e）
│   ├── 04-self-verification.md        验证矩阵 + 黄金校准（残差 bug）+ 教训
│   ├── 05-review-revisions.md         三轮审查 + 保真裁定执行记录
│   ├── 06-sim-vs-real-diff.md         sim-platform 运行 + 实测 + repeat_interleave 缺口
│   └── archive/                       原始文档逐字归档（ADAPT/REVIEW/REVIEW_CHANGES/
│                                      WORKSUMMARY/实测 REPORT/opm SLA2 文档副本）
└── skills/                            ← AI 可执行工作流
    ├── titan030-sim-model-adapt/      基础适配（preset/compat/runner seam/测试骨架）
    └── titan030-opm-fidelity-flow/    全流程（保真铁律/探针/opm 建模/黄金校准/对比）
```

## 关键仓库（通用 + 样例期状态）

| 仓 | 地址/路径 | 说明 |
| --- | --- | --- |
| simulator 正仓 | https://github.com/ZhanluModelSim/torchtitan-npu-simulator | 本地 `torchtitan-npu-simulator030/`（titan030 分支）+ worktree `mhm-wip-wt/`（样例分支 multi-head-moe-wip） |
| 算子建模仓（opm） | `/home/l30079348/opm-dev`（opm = **operator-performance-models**，远端 ZhanluModelSim/operator-performance-models，分支 gdn-sla2） | **建模正本只在这里**：各算子的 roofline/延迟成本模型 + 语义 + adaptor；复用优先的查询入口见 playbook 3 |
| 实测仓 | 本地 `torchtitan-npu-real/`（同 simulator 仓的隔离 clone） | 真机 bench 用，容器内 PYTHONPATH 隔离 |
| sim-platform | 本地 `sim-platform/`（main，端到端链路） | 用法见 playbook 5 |

## 样例期的关键结论（对新模型有直接参考价值）

- **保真裁定**：源码一模一样，融合仅限 owner 批准的 seam 且默认关闭（详见 case 05）。
- **捕获缺口两类**：数据依赖算子不可捕获（bincount/where → GMM 决策）；复合算子被分解
  （repeat_interleave → register_fake 修复三件套，0.33% 设备时间影响量化）。
- **黄金参照法**：发现了捕获测试全部漏掉的真实架构 bug（残差丢失）——无基准模型必做。
- **实测对比**：主计算算子双侧语义一一对应；真机是 bf16 混精度（口径必须先核）。

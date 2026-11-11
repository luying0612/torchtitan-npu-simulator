# 0 · 通用原则（任何新模型适配前先读）

> 本目录是**模型无关的通用流程**；本次 multi-head-moe 适配的完整实例见 `../case-multi-head-moe/`。
> AI 可执行的细节版见 `../skills/`（titan030-sim-model-adapt = 基础适配；titan030-opm-fidelity-flow = 全流程含保真/建模/对比）。

## 铁律（违反即返工）

1. **源码保真**：新模型实现必须与给定源码一模一样，严禁任何与源码粒度不一致的算子融合/虚拟化替换。
   融合只允许出现在显式 seam 函数，且必须经开关（默认关闭）opt-in，融合范围清单先交 owner 确认。
2. **例外需批准**：参考实现的真实 bug 修复（附最小复现证据）、Meta 无法捕获的算子替换（附探针证据）、
   初始化缺失补齐——逐项留档。
3. **探针先行**：任何"能不能捕获/能不能融合/库里有没有"的判断先做最小探针实测，勿凭版本号或经验臆断。
4. **复用优先**：算子建模先查算子建模仓已有族，能复用绝不新写（见 playbook 3）。
5. **验证三件套**：任何"验证通过"的汇报必须跑齐 ruff check + ruff format + 类型检查
   （"check 通过"≠"format 通过"，有实际返工教训）。

## 五阶段流程总览（对应 playbook 1–5）

| 阶段 | 目标 | 关键产物 |
| --- | --- | --- |
| 1 迁移 | 模型代码进仿真仓，preset 可跑通 | 模型包 + compat + preset + 捕获测试骨架 |
| 2 缺失验证 | 确认捕获图算子无缺失、无未捕获 | CPU dispatch 对照报告 + 探针记录 |
| 3 算子建模 | 捕获到的计算算子全部有成本模型（复用优先） | opm 仓提交或库路由接线 |
| 4 自验证与审视 | 验证矩阵全绿 + 审视意见闭环 | 测试矩阵 + 修改记录 |
| 5 端到端与实测对比 | sim-platform 跑通 + 真机算子差异报告 | L1 导出 + kernel CSV + 对比报告 |

## 权威文档指引（各阶段对应的仓内规则）

新栈（titan030）的接入规则与最终验收**在 simulator 仓内**；本 playbook 是实操补充。
每阶段动手前先读对应章节，验收以仓内判据为准（文档位于仓 `experiments/simulator/`）：

| 阶段 | 权威文档 | 对齐内容 |
| --- | --- | --- |
| 0 总纲 | `AGENTS.md` | 阅读顺序、修改归属、必须保持的边界（原生 dispatch 优先、虚拟融合四契约、稳态捕获、FSDP 通信视图、内存去重/offload 规则）、验证与文档纪律 |
| 1 迁移 | `docs/model-integration.md` §1–§5 | 固定环境与模型依据、**先选复用路线**（§2）、配置树/ModelSpec/接入检查清单（§3.3）、算子先原生后必要兼容（§4）、并行按顺序使能（§5） |
| 2 缺失验证 | `AGENTS.md` 边界节 + `docs/custom-tensor-semantics.md` | 按 schema 与最小 Meta/梯度探针判断兼容缺口（不按版本号装 patch）、语义声明契约 |
| 4 验收 | `docs/model-integration.md` §6–§7 | **验收关卡 A–F**：A 配置与构建 / B 核心组件 / C 单任务 F/B/O / D 并行与 AC / E 内存与卸载 / F 量化与交付；独立预期值对账、拒绝组合要有测试、**不用 build-only/测试数量/Meta 成功替代验收**；§7 交接项清单 |
| 5 端到端 | `AGENTS.md`（性能预测归下游）+ `docs/capability-matrix.md` | cost/DES/最终指标归下游平台（sim-platform），不在训练仓恢复；能力边界以矩阵为准 |

专题文档按需读：`model-integration-pitfalls.md`（常见问题）、`capture-runtime.md`、
`steady-state-capture.md` / `steady-state-memory.md`、`activation-offload-design.md` /
`optimizer-offload.md`、`developer-guide.md`、`upstream-integration.md`。

## 环境速查（通用）

- 容器 `titan030-qwen35`（仿真/Meta）；宿主 `/home/l30079348/titan030-qwen3.5/` bind-mount 为 `/workspace`。
- 必须先 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`；import torch 前设
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。日志噪声（DrvMng/ascend_hal/W11xx）grep 过滤即可。
- 仓与安装：`torchtitan_npu`/`torchtitan_simulator`/`operator_performance_models`/`sim_platform`
  均为 editable（MetaPathFinder 优先于 PYTHONPATH）——**干净基线只能 git stash，不能用 PYTHONPATH 叠加**；
  新模型分支工作用 worktree 隔离（勿污染共享主 checkout）。
- CLI 成功是静默的：`python -m torchtitan_simulator --config <preset>` 以 PYEXIT=0 判定，别 grep 日志。
- 推送纪律：只本地 commit；用户明确说"推送"才推。

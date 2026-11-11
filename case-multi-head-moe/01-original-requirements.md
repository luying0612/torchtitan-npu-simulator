# 01 · 原始需求、环境与参考实现

> 归档正本：`archive/MULTI_HEAD_MOE_ADAPT.md`（§0–§2）。本文为梳理版。

## 1.1 最原始开发需求（会话记录逐字归档）

1. 参照 qwen3.5 的接入先例，把新模型 **multi-head-moe**（来自
   `https://github.com/ZhanluModelSim/inference-models-simulator`，路径
   `external_models/multi_head_moe/`）接入 torchtitan 0.3.0 仿真栈，跑通流程。
2. 不需要做实测对比；**需要计算算子建模（通信除外）**；部分融合算子可以对照 CANN 包实现。
3. 需要生成一份交接文档给 AI 审查，记录开发过程和最原始的需求原文，让其他 AI 能准确理解原始信息。
4. 对之前的参考文档保持批判性（可能存在错误），不要照搬。
5. （后续补充）完成之后先不用提交，暂存即可；仿真硬件配置用库上的样例
   （`sim-platform/hardware/chip/example.json`），不要沿用 qwen3.5 的标定档 `npu_910b3`。
   *（两次更新：算子建模只认 opm-dev 仓；硬件最终采用 `/home/l30079348/hardware` 的 DV120 真实规格。）*

### 1.1.1 追加的 owner 指令（2026-11-10，保真裁定，最高优先级）

> **"严禁进行这样的操作，必须和我给你的源码一模一样的实现。"**

- 模型实现必须与给定源码（`inference-models-simulator/external_models/multi_head_moe/multi-head_moe.py`）
  **一模一样**；禁止任何与源码粒度不一致的算子融合/虚拟化替换。
- 后续 owner 三项决策（详见 `05-review-revisions.md`）：
  ① 保留小算子版本与融合版本两套（融合为显式 opt-in，**默认关闭**）；
  ② 融合边界仅限两个 seam 函数；③ MoE 逐专家循环→grouped-GEMM 在两种模式下保留（owner 批准）。

## 1.2 环境与仓库

| 项 | 值 |
| --- | --- |
| 仿真容器 | `titan030-qwen35`（宿主 `/home/l30079348/titan030-qwen3.5/` bind-mount 为容器 `/workspace`） |
| 实测容器 | `mhoe-real`（镜像 `ttsim-qwen35-real:v1`，privileged，挂全部 8 卡；由 qwen35-real 配置复制新建，见 `06-sim-vs-real-diff.md`） |
| 版本 | torch 2.14.0+cpu / torch_npu 2.14.0.dev / CANN 9.2.0 / torchtitan 0.3.0 / ruff 0.17.3 |
| 容器环境 | 必须 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`；import torch 前设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| simulator 仓 | `torchtitan-npu-simulator030/`（分支 `titan030`）；本次工作在 worktree `mhm-wip-wt/`（分支 `multi-head-moe-wip`，基点 `fca58c5`） |
| 上游 0.3.0 | `torchtitan030/torchtitan/`（容器内另有完整 site-packages 版本） |
| 算子建模仓 | `/home/l30079348/opm-dev`（分支 `gdn-sla2`，本次提交 `0dd983e`，分支后续由并行会话推进至 `21a35b6`） |
| 实测仓 | `torchtitan-npu-real/`（@70a29ba，容器内 `PYTHONPATH` 隔离） |
| editable 安装注意 | `torchtitan_npu`/`torchtitan_simulator`/`operator_performance_models`/`sim_platform` 均为 editable（MetaPathFinder 优先于 PYTHONPATH）——干净基线只能 `git stash`，不能用 worktree 叠加 |

## 1.3 参考实现及其 3 个真实 bug（移植时全部修正，已获 owner 批准保留修复）

参考代码 `/tmp/opencode/inference-models-simulator/external_models/multi_head_moe/multi-head_moe.py`
（纯 PyTorch 朴素实现，MAGI-2 preview 风格：SLA2Attention + Multi-Head-MoE）。**参考实现从未运行成功**：

1. **RoPE 广播错误**：`freqs_cis.unsqueeze(2)` 与 `[B,S,H,D]` 布局矛盾，正确为 `unsqueeze(1)`（freqs_cis 为 `[S, Dh/2]`）。
2. **线性分支 einsum 非法**：`"bhqk,bhkd->bhqd"` 5 维操作数配 4 个下标；修正为 `"bhqk,bhkld->bhqld"`，
   kv state/phi_k_sum 需 `repeat_interleave(bs, dim=2)`，norm 用 `matmul(phi_q.unsqueeze(-2), phi_k_sum.unsqueeze(-1)).squeeze(-1)`。
3. **布局错位**：`SLA2Attention` 产出 `[B,S,H,D]`，但 `InnerBlockSLA2` 期望 `[B,H,T,D]`——边界处 transpose 修正。

其他参考坑：router/gate 权重 `torch.empty` 从不初始化（移植版给标准 small-std normal 初始化，
owner 批准）；topk 每调用传入（参考懒缓存会过期）；alpha 保持 0.5 构造值不动。

数值边界提示（忠实继承，未修正）：参考实现未加因果 mask；padding 位置的 softmax(0) 均匀分布
会泄入 linear 分支 norm 分母（分子不受影响，phi_v=0）。生产化数值对齐时需知。

修正版与最小前向/反向验证脚本：`/home/l30079348/titan030-qwen3.5/_mhoe_scratch/`
（pristine 原版存档 `multi-head_moe_ORIGINAL.py`，scratch 仅作证据不入仓）。

## 1.4 生产几何对照

dim=8192, vocab=102400, 80 层, 64 heads/128 hd, MoE 32 头 × 896 专家, K=16, shared 4096,
expert mult 4, block 64, topk 比例 0.005/0.001（≤1M/超长）。
debug 几何：D=256, V=102400, 8 层（0,1,6,7 dense；2–5 MoE）, H=4, hd=64, block=64,
router_proj=64, inter=512; MoE S=4, hs=64, Es=16, Ks=4, expert mult 2, shared 128。
参数公式 `TOTAL = 2·V·D + D + 4×664,196 + 4×2,077,316 = 63,395,104`，参数张量 139 个。
多模态核对：参考仓 README 明确 multi_head_moe 为 **text-only**（`forward(tokens)` 仅收 tokens，
无视觉分支）；若后续参考实现增加多模态输入，接入点在模型前向入口，融合边界不受影响。

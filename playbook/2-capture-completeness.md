# 2 · 验证是否有算子缺失没有捕获

> 目标：证明 **Meta 捕获图 = 真实执行同一代码的算子集合**（在声明的 override/虚拟化边界内）。
> 样例：`case-multi-head-moe/04-self-verification.md`（CPU dispatch 对照节）。

## 2.1 主方法：CPU dispatch trace vs Meta 捕获图对照

1. 同一模型代码、同几何，在 CPU 真实执行（TorchDispatchMode 记录 ATen 算子）与仿真 Meta 捕获
   （world=1）各跑一次。
2. 按算子族对照计数：`_grouped_mm`/matmul/逐元素/reduction/override 算子（rms_norm/rope）等。
3. **每个差异必须归因**，可归因项白名单：CE loss 差异、训练态随机（gumbel rand_like）、
   override 替换、AC 重算倍增、dtype/cast 路径。归因不了的 = 疑似缺失，回 Step 1 查接线。

## 2.2 探针工具箱（判定"能不能捕获"）

对可疑算子先在 Meta 上写最小探针，已知结论（勿外推，新算子重测）：

| 类别 | 结论 |
| --- | --- |
| `torch._grouped_mm` / `npu_rms_norm` / `npu_swiglu` | 可捕获 |
| `npu_rotary_mul` / `npu_moe_gating_top_k_softmax` | 无 Meta kernel（需 compat 桥或放弃） |
| `npu_block_sparse_attention` | 有核但参数校验可拒绝（如 blockShape 128 对齐） |
| `bincount` / `torch.where` 紧凑索引 | **不可捕获**（无 fake impl / 数据依赖输出形状）→ 替换方案须 owner 批准 |
| `topk` | 可捕获 |
| **复合隐式算子（repeat_interleave 等）** | **TorchDispatchMode 下被分解为 expand/reshape 等原语后才可见**——真机有直连 ACLNN 核则不分解，形成"仿真侧看不到"的缺口（见 playbook 5 §5.4） |

## 2.3 成本注记扫描（辅助信号）

`annotate_workload_costs` 后扫描 `cost_unknown` 节点：view/t/detach/transpose/slice 等
零计算布局算子与 stream_wait/profiler 事件 = 设计如此非缺口；**真实计算算子 unknown = 缺口**
（无 handler 或路由失败——后者先查算子建模仓适配（playbook 3 §3.3 源契约））。

## 2.4 数学正确性另测（黄金参照法，与缺失验证互补）

捕获保真 ≠ 数学正确——捕获测试发现不了残差接错这类 bug（样例实证）。方法见 playbook 4 §4.2。

## 2.5 产出

- 对照表（算子族 × CPU/Meta 计数 × 归因）入验证文档。
- 探针记录（哪些算子不可捕获、依据、替换决策与 owner 批准）入交接文档。

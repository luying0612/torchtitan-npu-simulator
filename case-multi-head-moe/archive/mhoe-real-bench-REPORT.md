# multi_head_moe 仿真 vs 实测 计算算子对比（EP2/DP2, seq=128, FP32, world=2）

日期：2026-11-10。范围：仅计算算子结构差异（不含耗时、不含通信，通信 hcom_* 已按约定排除）。

## 数据位置
- 仿真（sim-platform, eager 小算子路径, dv120 roofline）：
  `/workspace/mhoe-platform-run/runs/eager/l1/`（`statistics/operators_by_name.csv` 85 个归一化算子族；`000_F/001_B/002_OPTIMIZER.csv` 原始捕获）
- 实测（qwen35-real 镜像新容器 `mhoe-real`，cards 0-7 全空闲，torch 2.14 + CANN 9.2）：
  `/workspace/mhoe-real-bench/traces/ep2/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv`
  （2 rank × 2 窗口，各 5124 行；rank0 step7 聚合：`real_mhoe_rank0_step7.json`，178 个 kernel 名）
- 聚合脚本：`agg_kernels.py`（本目录）

## 配置对齐
| 项 | 仿真 | 实测 |
|---|---|---|
| world/EP/DP_SHARD | 2/2/2 | NGPU=2, EP=2, DP_SHARD=2 |
| MBS/GBS/seq | 1/2/128 | 1/2/128 |
| dtype | FP32 | **bf16 混精度**（真机栈默认 mixed_precision=bf16，实测 matmul 576/596 为 DT_BF16） |
| dispatcher | 默认路径（虚拟 NpuMoe* 算子） | override `token_dispatcher.asc`（npu_moe_token_permute/unpermute/re_routing） |
| profiler | meta 捕获（TorchDispatchMode） | kernel_details.csv（ACLNN） |

## 语义对齐的算子族（主计算全部覆盖）
Mm↔MatmulCommon/MatMulV3、Bmm↔BatchMatMulNd、GroupedMm↔GroupedMatmulV5(48)、Add/Adds、Sub/Rsub、Mul/Muls、Div/Divs、Pow、Sum↔ReduceSum、Mean、Rsqrt、Neg、Log、Sigmoid±Grad、Silu±Grad、Softmax±Grad、LogSoftmax±Grad、NLLLoss±Grad、Topk(+GatherElements 内部)、Index/IndexPut/Scatter、MaskedFill、RandLike↔DSARandomUniform、ViewAsComplex/Polar/Complex(RoPE)、Cat/ConcatD+ChunkCat、Stack↔Pack、Arange、Clamp↔ClipByValueV2、Reciprocal、Embedding±Backward、ZerosLike/OnesLike、FusedAdamw↔ApplyAdamWV2(1 列表算子↔139 核, 已知粒度)、foreach norm/add↔LinalgVectorNorm、NpuMoe{TokenPermute,TokenUnpermute,ReRouting}±Grad 双侧齐 ✓

## 发现的差异（计算算子）

### 1. repeat_interleave：sim 缺失、低估（唯一实质缺口）
- 模型源：`run_sla2_linear_attention` 内 `kv_state.repeat_interleave(bs, dim=2)` 与 `phi_k_masked.sum(-2).repeat_interleave(bs, dim=2)`（model.py:201/203）。
- 实测：`aclnnRepeatInterleaveV2` ×32/step（8 层 × fwd2 + bwd2），真实 vector 核。
- 仿真：捕获产物中 **0 次** repeat_interleave/Tile。原因：复合隐式算子在 TorchDispatchMode 下被分解为 expand+reshape，sim 归入零成本 Expand；真机 torch_npu 有直连融合核。
- 影响：sim 将该算子计为零成本，真机为实际 vector kernel——若做耗时对比会低估。

### 2. Cast/Copy 粒度差异（已知层级差异，非缺失）
实测每算子前后按 dtype 插入 cast 核：InplaceCopy_Cast ×406 + 各算子 _Cast（TopkV2/Sort/Cumsum/ReduceSum/IndexPutImpl/NLLLossBackward 等）；sim 仅记模型显式 ToCopy ×221。层级不同，量级低估。

### 3. MoE dispatcher 路径差异（语义已覆盖）
实测 override 路径额外执行 Sort ×8、Cumsum ×8、Gather ×24（路由内部小算子）；sim 用默认路径，语义由虚拟 NpuMoeReRouting/NpuMoeTokenPermute 覆盖。属实现路径差异，非算子缺失。

### 4. 其他小项
- `fill_` ×10/step 仅实测出现（freqs cache/初始化类小算子，影响极小）。
- dtype 口径：真机跑的是 bf16 混精度（非 FP32）。对算子种类无影响，但部分解释了实测 cast 核偏多（bf16/fp32 转换插入）；如需耗时级对齐，仿真侧应切 bf16 混精度重捕获。
- 反向：sim 的 view/expand/permute/detach/clone/stream_wait 等元数据算子实测无设备核——预期正常。

## 结论
主计算算子（GEMM/GMM/逐元素/RoPE/MoE/优化器/损失）双侧语义一一对应，**唯一实质缺失为 repeat_interleave**（sim 分解为零成本 Expand）。如需耗时对齐，需给该算子补 bytes→duration 建模或修捕获分解；Cast 粒度为已知层级差异。

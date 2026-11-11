# 06 · sim-platform 端到端仿真 vs 真机实测对比（计算算子差异）

> 归档正本：`archive/mhoe-real-bench-REPORT.md`。原始产物：
> 仿真 `/home/l30079348/titan030-qwen3.5/mhoe-platform-run/`，实测 `.../mhoe-real-bench/`。
> 任务口径（owner 指定）：**只关注计算算子缺失，不关注耗时差异与通信**。

## 6.1 背景与口径

- 原始 simulator 无端到端耗时（L2 SchedulePlan 的 `duration_est` 字段从未被填充），
  切换 **sim-platform 平台链路**启动仿真（capture → adapt → predict → DES → L1 导出）。
- 配置对齐：config=multi_head_moe_debugmodel，world=2 / dp_shard=2 / ep=2 / local_batch=1 /
  seq_len=128；走**小算子 eager 路径**（`enable_sla2_fusion` 默认 False）。

## 6.2 仿真侧运行方法

- `mhoe-platform-run/mhoe_eager.toml`：[training] 五项 + [platform] hardware=/workspace/qwen35-platform-run/hardware/dv120.json + [prediction] backends=["roofline"]。
- **run_sim.py 导入手术**（关键技巧）：先剥离 editable finder，`sys.path.insert` 指向
  `/workspace/mhm-wip-wt`（含 `experiments/simulator`），**预导入** `torchtitan_simulator` +
  `torchtitan_npu`（在平台解析前缓存进 sys.modules），再调 `sim_platform.cli.training:main`。
- 产物：`runs/eager/l1/`——`statistics/operators_by_name.csv`（85 个归一化算子族）、
  `training.json`（end_to_end_us=4519 = F 1980 / B 1738 / O 800）、`000_F/001_B/002_OPTIMIZER.csv`
  原始捕获、trace 导出。

## 6.3 实测侧运行方法

- **容器**：原 `qwen35-real` 的设备节点失效（aclInit 507899 / dcmi -8020，被 vllm/vnpu 容器持有；
  宿主 npu-smi 显示 8 卡全部无进程——冲突是容器级持有，非卡占用）。
  解法：同镜像 + 同挂载新建 privileged 容器 `mhoe-real`，设备节点全部可用。完整命令：
  ```bash
  docker run -d --name mhoe-real --privileged --ipc=shareable --shm-size 34359738368 \
    -v /usr/local/dcmi:/usr/local/dcmi -v /home/l30079348/titan030-qwen3.5:/workspace \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -e ATB_CXX_ABI=1 ttsim-qwen35-real:v1 /bin/bash -c \
    'source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
     source /usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh && \
     source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1 && sleep infinity'
  ```
- **环境修复**：scipy 1.18.1 与 numpy 1.26.4 不兼容（np.long 报错）→ 回退 scipy==1.13.1
  （11-03 基线 pin）。
- **运行**：`examples/multi_head_moe/mhoe_debugmodel_moe_2p_128.sh`（NGPU=2, EP=2, DP_SHARD=2,
  MBS=1, GBS=2, seq=128, OVERRIDE_IMPORTS=token_dispatcher.asc）+ profiler 参数
  freq4/warmup2/active1 → `kernel_details.csv` 2 rank × 2 窗口各 5124 行。
- **聚合**：`agg_kernels.py`（健壮版；`parse_real.py` 对该 CSV 的引号变体不健壮）→
  rank0 step7：5123 kernels / 178 distinct names。

## 6.4 对比结论（计算算子）

**语义对齐的算子族**（主计算全部覆盖）：Mm↔MatmulCommon/MatMulV3、Bmm↔BatchMatMulNd、
GroupedMm↔GroupedMatmulV5、Add/Sub/Rsub/Mul/Muls/Div/Divs/Pow、Sum↔ReduceSum、Mean、Rsqrt、
Neg、Log、Sigmoid±Grad、Silu±Grad、Softmax±Grad、LogSoftmax±Grad、NLLLoss±Grad、
Topk(+GatherElements 内部)、Index/IndexPut/Scatter、MaskedFill、RandLike↔DSARandomUniform、
ViewAsComplex/Polar/Complex(RoPE)、Cat/ConcatD+ChunkCat、Stack↔Pack、Arange、Clamp、Reciprocal、
Embedding±Backward、ZerosLike/OnesLike、FusedAdamw↔ApplyAdamWV2（1 list-op ↔ 139 kernel，已知粒度）、
foreach norm/add↔LinalgVectorNorm、NpuMoe{TokenPermute,TokenUnpermute,ReRouting}±Grad 双侧齐。

**差异发现**：

1. **repeat_interleave：sim 缺失、低估（唯一实质缺口）**
   - 实测：`aclnnRepeatInterleaveIntWithDim`（V2 核 + 配对 BroadcastTo）64 kernel/step，437us。
   - 仿真：捕获产物中 0 次。原因：**复合隐式算子在 TorchDispatchMode 下先分解为
     expand+reshape 才到达捕获层**（无独立 Meta 核）；真机 torch_npu 注册了直连 ACLNN 核不分解。
   - 修复方案（不动模型源码）：捕获初始化处 `torch.library.register_fake("aten::repeat_interleave", meta_fn)`
     使 dispatcher 有直接 Meta 核 → 捕获可见；op_mapping 加映射；opm 加 vector 拷贝类 cost
     handler（read in + write out×repeats）。影响：当前 bs=1 约 0.33% 设备时间，随 batch 线性增长
     （bs=8 约 2–3%），拷贝类小核不会主导。
2. **Cast/Copy 粒度差异**（已知层级差异非缺失）：实测每算子前后按 dtype 插入 cast 核
   （InplaceCopy_Cast ×406 + 各算子 _Cast）；sim 仅记模型显式 ToCopy ×221。
3. **MoE dispatcher 路径差异**（语义已覆盖）：实测 override 路径额外 Sort ×8 / Cumsum ×8 /
   Gather ×24（路由内部小算子）；sim 用默认路径，语义由虚拟 NpuMoe* 覆盖。
4. **dtype 对齐缺口（重要口径修正）**：**真机实际是 bf16 混精度**（matmul 576/596 为 DT_BF16，
   真机栈默认 mixed_precision=bf16），仿真捕获是 FP32。对算子种类无影响，但部分解释实测 cast
   偏多；如需耗时级对齐，仿真侧应切 bf16 重捕获。
5. 反向：sim 的 view/expand/permute/detach/clone/stream_wait 等元数据算子实测无设备核——预期正常。

## 6.5 产物与脚本索引

（另注：实测仓 `torchtitan-npu-real` 中移植的模型包与 examples 脚本当时为 **untracked 未版本化**，
是否提交待指示；titan030 主线已前进到 `9c4454e`，若合入 simulator 主线需先 rebase。）

| 产物 | 路径 |
| --- | --- |
| 仿真 toml + 导入手术脚本 | `titan030-qwen3.5/mhoe-platform-run/{mhoe_eager.toml, run_sim.py}` |
| 仿真 L1 导出 | `mhoe-platform-run/runs/eager/l1/` |
| 实测 kernel CSV | `mhoe-real-bench/traces/ep2/*/ASCEND_PROFILER_OUTPUT/kernel_details.csv` |
| 实测聚合 JSON + 脚本 | `mhoe-real-bench/{real_mhoe_rank0_step7.json, agg_kernels.py}` |
| 实测运行脚本 | `torchtitan-npu-real/examples/multi_head_moe/mhoe_debugmodel_moe_2p_128.sh`（容器内） |
| 对比报告 | `mhoe-real-bench/REPORT.md`（副本 `archive/mhoe-real-bench-REPORT.md`） |

# titan030 仿真栈模型适配工作总结

> 周期：2026-10 末 – 2026-11-03。范围：torchtitan-npu-simulator 0.3.0（本地分支 `titan030`）上适配 qwen3_5 与 deepseek_v4_1，贯通 Meta 捕获 + sim-platform 链路，并完成 qwen3.5 真实 NPU 实测对比。
> 仓库根：`/home/l30079348/titan030-qwen3.5/`；容器：`titan030-qwen35`（仿真/Meta）、`qwen35-real`（实测，挂 2/3/5/6 卡）。

## 1. 交付物一览

| 交付物 | 位置 | 状态 |
| --- | --- | --- |
| qwen3.5 仿真适配 | simulator 仓 `a7ce3df`（13 文件：compat/presets/config/runtime/测试） | 完成，12/12 测试通过 |
| 捕获 dump 脚本 | simulator 仓 `1b17642`（scripts_bench/dump_sim_workload.py） | 完成 |
| deepseek_v4.1 仿真适配 | simulator 仓 `a25a543`（compat/deepseek_v4_1.py、测试、文档） | 完成（text 捕获 4/4 测试通过） |
| 真实 NPU 实测 | `qwen35-real-bench/`（REPORT.md、日志、trace、聚合 JSON、脚本） | 完成 |
| 实测 vs 仿真对比报告 | `qwen35-real-bench/REPORT.md` | 完成，含修改计划 P0–P2 |
| 分支状态 | `titan030`（rebase 至远端 70a29bae），本地 3 个提交，**未推送** | 按约定只本地提交 |

## 2. qwen3.5 适配（已完成）

上游 TorchTitan 0.3.0 原生模型，MoE；GDN 与 FlexAttention 是 triton-ascend 融合，Meta 下不可执行，在 `compat/qwen3_5.py` 建显式虚拟融合（`qwen3_5_sim.gated_delta_rule[_grad]`、`qwen3_5_sim.flex_attention[_grad]`），声明 tensor_semantics（角色/轴/梯度角色），保证保存值、反向身份、轴语义与原生一致。

关键点与坑：

1. **导入顺序**：上游 model.py 顶层绑定 fla 符号，必须先 `ensure_qwen3_5_importable()` 再导入上游模型；ruff `--fix` 的导入重排会破坏该顺序（已用显式调用规避，勿依赖导入副作用）。
2. **dispatcher 断言**：EP=2 时断言改为配置树派生（`AscAllToAllTokenDispatcher.Config`），不按运行时实例判断。
3. **tensor_semantics 补齐**：远端 3002aa3 引入新设施后，在 compat 声明 GDN（B/S/H/D，reset optional）与 flex（B/H/S/D）语义；固定长度路径由 cu_seqlens 物化 (1,16) 真实 reset 张量，optional 声明两种情况都覆盖。
4. **测试**：`tests/test_qwen3_5_capture.py`——参数公式（133 张量）、6 组并行/AC 参数化、虚拟融合保存值与反向身份、语义声明完整性。

## 3. 真实 NPU 实测与对比（已完成）

容器 `qwen35-real`（docker commit 自仿真容器 + 挂 `/dev/davinci{2,3,5,6}`），独立仓 `torchtitan-npu-real`（`PYTHONPATH` 隔离）。dense 4P（DP_SHARD=4）与 MoE 4P（EP=4）均 Training completed，trace 落盘（profiler active 步 3/7）。

环境坑：triton-ascend 3.2.1 预编译头 `-std=c++17` 与 torch 2.14（C++20）冲突（容器内 sed 两处）；scipy 1.14+ 与 numpy 1.26 不兼容（降 1.13.1）；MoE 脚本需显式 `--parallelism.data-parallel-replicate-degree 1` 与 `--training.global-batch-size -1`。

对比方法（REPORT.md）：ACLNN 层级 vs ATen 层级，**不做耗时建模**，只比算子个数/shape/dtype。关键结论：

- GDN fwd 1 融合 op ↔ 3 个 triton kernel、bwd 1:1；flex 1↔2/4；fused_adamw 1 个 list-op ↔ 逐参数 kernel（133 vs 206）；embedding 完全一致。
- GMM tokens 均衡路由（仿真 2048）vs 实测非均衡（1710）——路由假设偏差。
- vision conv ~2.6 万 kernel/步为 scope 差异；内存 1.33/1.46GB vs 1.50/1.64GiB。

修改计划（P0：ATen↔ACLNN 映射表固化 + text-only 对照组；P1：路由假设偏差注记 + 融合边界展开系数声明；P2：FSDP 桶粒度核对 + 一键复验脚本）——**尚未实施**，属后续工作。

## 4. deepseek_v4.1 适配（本次完成）

主包自带 `torchtitan_npu/models/deepseek_v4_1/`（上游 0.3.0 无此模型）。仿真侧缺口与解法：

1. **preset 注册**：`presets.py` 加 `_deepseek_v4_1_trainer_config` + `deepseek_v4_1_debugmodel_text/multimodal` 包装（工厂不接受 `num_mtp_layers`，包装函数拦截）；`config.py` 注册 factories、family 集合与 `_DSV4_1_OVERRIDES`（rope.asc_half_rotation + mhc.asc_hc_post）。
2. **Engram 表 Meta 桥**：host-offload 表 `HostEngramTable._init_self_parameters` 拒绝非 CPU 设备——compat 桥放宽为 CPU/meta 均可，种子推导与生产一致（`compat/deepseek_v4_1.py`）。
3. **attention 元数据 map 适配（核心）**：v4.1 弃用模型级 `build_attention_masks`，改用逐后端 `AttentionMetadataMap`（patched decoder 的 `_get_attention_metadata` 构造）；simulator runner 的过渡契约仍强制调 `model.build_attention_masks`。适配器复用主包 patch 的 `_get_attention_metadata`（同一实现，不重写遍历），把 map 塞进 extra kwargs；CP 显式拒绝。
4. **稀疏梯度豁免**：`engram.table.weight` 走 SparseAdam 稀疏路径，永远没有 dense `.grad`——runner 的梯度清单按 `_engram_host_offload` 参数标记豁免（gate 投影等 dense 参数不受影响）。
5. **测试**：`tests/test_deepseek_v4_1_capture.py`——single / FSDP2 rank0/rank1 / full-AC 四组。断言：964 个 dense 梯度（966−2 稀疏表）、稀疏表豁免、唯一自定义语义 op 为共享的 `npu_moe_token_unpermute`（40 次，full-AC 下 80 次重算）、参数驻留公式（FP32；FSDP2 下 80 个奇 numel 参数导致 rank1 少 80 元素偏移；engram 表 EP=1 时全表驻留）、优化器状态公式（dense 2×FP32 moment + 4496B SparseAdam stub）。

调试要点（对后续模型适配有参考价值）：

- 报错 `'FSDPDeepSeekV41Model' object has no attribute 'build_attention_masks'` 的正确读法：主包 patch 的 prepare-input 对无该方法的模型走 `_get_attention_metadata` 自动分支，是 runner 绕过了 patch 逻辑；适配器应**复用 patch 实现**而不是重写。
- multimodal capture 未做（vision encoder 全参数无梯度，需 qwen3.5 同款 vision detach，范围外）；multimodal **build-only 通过**（436,510,384 参数、40 层）。

## 5. matmul 耗时建模（本次完成，用户明确指示）

- **落位**：simulator `cost/op_cost_model.py` + 新 `cost/hardware.py`。`annotate_workload_costs(workload, hardware=npu_910b3())` 显式开启后，matmul 节点额外获得 `duration_us` 注记（默认路径不变，无 hardware 参数时零行为差异）；非 matmul 节点不猜耗时。实现复用工作区 `operator-performance-models` 包（MatmulAdaptor 解析 + MatmulRoofline `max(cube, memory)+overhead`），simulator 侧仅延迟导入并给出缺依赖清晰报错。
- **校准（关键结论）**：910B3 cube 峰值 294.912 TFLOPS（BF16）；利用率曲线 `LogUtilization` **只由空载设备微基准标定**（方形阶梯 256..8192 + K=4096 矩形网格 3×3，cube 主导 n=16；批内计时脚手架——逐次 sync 会给小 GEMM 造成 ~86us 假地板）：k=0.1056, b=-1.8626, min_ratio=0.0404, max_ratio=0.9254；拟合集残差中位 0.92×。**不做任何针对目标 workload 的调整**（曾试过用 trace 拟合并钉 max_ratio 到 workload p98，已按"不符合理论"要求撤销）。hbm 3.07 TB/s×0.9（规格假设，未校准）；overhead 3us。
- **实测 vs 建模耗时差距（matmul，方向统一 model/real，>1=仿真比实测长）**：
  - qwen3.5 dense trace（372 mm）：聚合 **0.81×**（模型 2.99ms vs 实测 3.70ms，偏短 ~19%），逐 kernel 中位 1.22×、p10–p90 0.71–2.08×
  - v4.1 ep4 trace（held-out，1696 mm）：聚合 **0.92×**（12.18 vs 13.21ms），中位 1.00×、p10–p90 0.65–1.50×
  - 结构性偏差（1-D FLOPs 曲线的已知局限，已量化）：deep-K/tiny-M 族空载实测比曲线低 3–6×（model/real 0.18–0.42×，(64,248320,256)=0.42×、(2,256,248320)=0.18×）；>1 的尾部来自小 GEMM 在真实连续执行中吃 L2 复用而 roofline 按冷流量计费
  - 对照：理想 roofline（100% 峰值）聚合 model/real ≈ 0.16×/0.09×（qwen3.5 5.20×反向、v4.1 10.18×反向的倒数关系即 model/ideal）
- **批量建模（`845f5c9`）**：在 matmul 之外新增非通信家族——bmm/grouped_mm（cube 路径）、rms_norm(+bwd)/rope/silu/softmax（声明等效算子的 vec 路径）、cat/split/moe_token_permute（纯流量）、adamw_step；unknown 节点按 raw_op_type 分发（mul/add/div/sum/norm/comparison/clone/_to_copy/gather/slice_backward 等 + zeros/arange 只写类）。流量规则统一"每个逻辑输入读一次、每个输出写一次"；无清晰 workload 语义的（embedding/conv/flex_attention/gated_delta_rule/view/控制/分配）如实保持 unknown。HBM 标定改为空载 256MB/1GB copy 实测 1.23TB/s（规格 3.07 的 0.40 常数）。
- **qwen3.5 全家族校验（66346 kernel，方向 model/real，>1=仿真偏长）**：matmul 聚合 0.75×（中位 1.22×）；adamw 中位 0.85×（sim 侧整步 2.1× 偏长——实测 fused kernel 流速 2.3TB/s 超过 copy 标定）；fused vec 激活（silu 族/rsqrt/exp）1.8–2.4× 偏长（真实 kernel 0.9–1.7us 低于 3us 开销地板）；reduce/log_softmax 0.24–0.50×（2–4× 偏短，多 pass 不达 copy 带宽）；copy/变换类（TransData/Cast/Slice）~0.45×（2.2× 偏短）。捕获侧缺口（cost 模型无法看到）：MemSet 167ms/步（trace 无形状）、PadV3 63ms、Conv2D 族 36ms、其余未建模 Type 107ms。
- **GDN/mhc 补建模（`fca58c5`）**：参考 CANN 包内核源码补充两个融合算子——gated_delta_rule（triton 串行递推 kernel，**延迟模型** per_token×T×AIV 波数：fwd 0.725us/token、bwd 3.224us/token，常数取自生产 trace 两个独立 kernel 群体；空载 DVFS 降频 3× 使微基准不可用，已披露）与 mhc_post(+bwd)（按 CANN kernel 公式 `x_{l+1}[i,:]=Σ_j H_res[j,i]·x_l[j,:]+h_out·H_post[i]` 推闭式等效算子模型，仅 v4.1 有节点、单测覆盖无 trace 校验）。
- 测试 14/14（新增 bmm/grouped_mm/pointwise/copy/raw 分发闭式用例），capture 全套 202 通过；本地提交 `822ceb3`/`5b2f201`/`845f5c9`。

## 6. 遗留与后续

1. **P0–P2 修改计划**（REPORT.md）未实施。
2. v4.1 multimodal capture（vision detach）、EP>1、MTP、CP、量化均未接入（与 qwen3.5 的边界一致，capability-matrix 已如实标注）。
3. 六个本地提交（`a7ce3df`、`1b17642`、`a25a543`、`9c4454e`、`08b0614`、`822ceb3`）**待推送**（推送需明确指示）。
4. 耗时建模仅覆盖 matmul；GMM/rms_norm/silu 等 vec/通信算子未建模（`operator-performance-models` 已有 roofline 实现，可按同路径接入）；hbm 带宽无校准点。
5. 给 AI 的复用指南已沉淀为 skill：`/root/.config/opencode/skills/titan030-sim-model-adapt/SKILL.md`。

## 7. 环境速查

- 仿真：`docker exec -w /workspace/torchtitan-npu-simulator030/experiments/simulator -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 titan030-qwen35 bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh && python -m torchtitan_simulator --config ... [--build-only]"`（CLI 成功是静默的，PYEXIT=0 即通过；要产物用 `simulate()` 返回的 workload）。
- 实测：`qwen35-real` 容器 + `PYTHONPATH=/workspace/torchtitan-npu-real`；tokenizer `/workspace/torchtitan/tests/assets/tokenizer`。
- 容器 `/workspace` bind-mount 宿主机 `/home/l30079348/titan030-qwen3.5/`，本地改动即时生效（无需 docker cp）。

## qwen3.5 算子建模入库 operator-performance-models（`0405457`，2026-11-05）

- 按管理员指示把 simulator 侧的 qwen3.5 算子建模移植到指定算子模型仓，达到该仓入库标准（两层分离/显式配置/版本化/mypy strict/ruff）。
- **新增 17 个 cost model**（消费既有 adaptor 语义，无需改解析层）：binary/scalar/unary elementwise、where、softplus、clone、copy、reduction、softmax/log_softmax(+gradient)、activation_gradient(silu_backward 等)、fused_adamw、rotary(+gradient)、rms_norm_gradient、mhc_post(+gradient)。
- **GDN 全链路新增**：semantics/npu_fused.py 新语义族（q/k 同形、d∈{64,128}、mode fwd/bwd 校验）+ adaptors/npu_fused.py 新 adaptor（raw `qwen3_5_sim.gated_delta_rule.default`）+ **延迟模型**（非 roofline）：duration = per_token_us × T × AIV波数，常数全部经 `GatedDeltaLatencyConfig` 显式必填（fwd/bwd per-token、value_block、aiv_cores），硬件能力不被查询；空载 DVFS 3× 降频不可标定已在文档披露。
- **mhc_post 闭式公式**：每输出元素 2(n+1) 等效算子（n AXPY + 1 缩放）直接来自 CANN kernel 源码结构（与 2·M·N·K 同类、无需标定系数）；backward 声明 2× 前向；combination 缺失明确 UnsupportedOperator。
- **fused_adamw**：读 params+grads+m+v(+max)+steps、写 params+m+v(+max)+steps 逻辑流量；list_sizes 未解析明确失败，不做均衡猜测。
- 系数零内置默认：全部经 `VecRooflineConfig` 显式传入（新增 8 个 VecOpConfig 字段 + MhcPostVecConfig + GatedDeltaLatencyConfig）；clone/copy 纯搬运无需系数（类比 Cat）。
- 测试：新增 tests/test_qwen_vec_models.py 11 项（手算、广播流量、空输出、缺配置 MissingCostmodel、结构公式、adaptor→model 端到端）；全仓 1128 项 pytest + mypy strict + ruff 全绿。
- 文档：AGENTS.md 覆盖范围行更新；docs/vec-roofline-models.md 新增"qwen3.5 负载族扩展模型"节（公式/流量表 + GDN 延迟模型与 DVFS 披露）。
- 与 simulator 实测校验的对应：库模型 = simulator `cost/op_cost_model.py` 的移植与规范化（同公式）；simulator 侧 910B3 标定（cube k=0.1056 曲线、HBM 0.40）仍归平台硬件配置，未进库（硬件 JSON 归平台所有，本仓只提供 ChipCapabilities 消费接口）。
- 待办：simulator 捕获特征回放为库内 fixture 回归（类似六模型的 472 节点重放）；simulator `cost/` 模块后续切换为消费本仓模型。

---
name: titan030-sim-model-adapt
description: 在 torchtitan-npu-simulator 0.3.0（titan030 分支新仿真栈）中适配新 LLM 模型并完成真实 NPU 实测对比的完整工作流程。覆盖新栈架构认知、preset/compat 接线、Meta 捕获验证、ACLNN vs ATen 对比方法论、已知坑（runner 过渡 seam、Engram Meta 桥、triton-ascend C++20 冲突等）。当用户要求"在 titan030/0.3.0 仿真栈适配 xx 模型 / 新增 debugmodel preset / 实测对比仿真"时使用。
---

# torchtitan-npu-simulator 0.3.0（titan030）新模型适配与实测对比

> 本文是基础适配（preset/compat/runner seam/测试骨架）的 AI 执行细节版。
> 通用五阶段 playbook（含保真铁律/缺失验证/建模复用优先/验证审视/端到端对比）：
> `/home/l30079348/new-model-doc/playbook/`；完整样例：`/home/l30079348/new-model-doc/case-multi-head-moe/`。

## 架构认知（与旧栈的根本差异，勿套用旧经验）

- simulator 是**独立 distribution** `experiments/simulator`（`torchtitan-simulator`），不在主包内；主包 `torchtitan_npu/models/` 定义模型结构/parallelize/checkpoint，simulator 只做 Meta 捕获接线。
- 捕获交付三图 `s0_F / s0_B / s0_OPTIMIZER`（WorkloadGraph）+ 静态显存 MemoryPlan；**无耗时建模**，无 kernel_summary CSV。算子是 ATen 层级（`raw_op_type`），非 ACLNN。
- 入口二选一：CLI `python -m torchtitan_simulator --config <preset> [--build-only]`（**成功是静默的，PYEXIT=0 即通过**）或 Python `simulate(SimulationConfig(...))` 返回 workload 对象（要产物必须走这个）。
- runner 的过渡契约：`model.build_attention_masks(inputs, labels, extra_kwargs, *, cp_mesh, load_balancer_type)` 返回 `(tokens, labels, kwargs)`；模型无此方法时主包 patched decoder 走 `_get_attention_metadata` 自动分支（0.3.0 patch 里已有实现）。
- 默认先做一次实际 GA 的无捕获结构预热（warmup 里校验梯度 manifest），再捕获单 microbatch F/B 与一次 O。

## 环境与入口（容器 titan030-qwen35）

```bash
docker exec -w /workspace/torchtitan-npu-simulator030/experiments/simulator \
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 titan030-qwen35 bash -c \
  "source /usr/local/Ascend/ascend-toolkit/set_env.sh && python -m torchtitan_simulator --config <preset> [--build-only]"
```

- **必须在导入 torch 前设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`**；日志噪声（DrvMng/Can't get ascend_hal/stale ninja lock/W1103）grep 过滤即可，不代表失败。
- 容器 `/workspace` bind-mount 宿主机 `/home/l30079348/titan030-qwen3.5/`，本地改动即时生效，无需 docker cp。
- 版本：torch 2.14.0+cpu、torch_npu 2.14.0.dev、CANN 9.2.0。ruff 在容器内（0.17.3），宿主机没有。
- `--build-only` 只证明模型可构建，**不证明图/梯度/内存**；验收以 `simulate()` + `validate_workload`（graph_acyclic、peak_storage_unique）+ gradient_validation + memory_plan 为准。
- **协作纪律：只本地 commit，不推送；用户明确说"推送"才推。**

## 适配工作流

### Step 1 判定模型来源
- **主包已有模型**（如 deepseek_v4_1，上游 0.3.0 没有）：只需 simulator 接线（Step 3）。
- **上游 0.3.0 原生模型**（如 qwen3.5）：主包模型已有 + simulator 需要 compat 层（虚拟融合/tensor_semantics）。

### Step 2 preset 注册（simulator 侧）
1. `torchtitan_simulator/presets.py`：加 `_<model>_trainer_config` 包装 + `_<model>_debugmodel_text`（多模态再一个 `_multimodal`）。包装函数拦截工厂不接受的参数（v4_1 工厂拒绝 `num_mtp_layers`）。若 `ensure_<model>_importable()` 存在，**显式调用**（勿依赖导入副作用，ruff --fix 重排导入会破坏顺序）。
2. `torchtitan_simulator/config.py`：factories 加两个 preset、family 集合加模型名、`_<MODEL>_OVERRIDES` 列表（参照生产脚本 `examples/<model>/debug/*.sh` 的 override 组合）。

### Step 3 compat 层（按需）
`torchtitan_simulator/compat/<model>.py` + `runtime.py` 加分支（capability 记录）。三种典型桥：

1. **Meta 初始化桥**：host-offload/自定义 init 拒绝 meta 设备 → `scope.replace(Class, "_init_self_parameters", ...)` 放宽 device 检查，种子推导与生产保持一致。
2. **build_attention_masks 适配器**（runner 强制调 seam 时）：**复用主包 patch 的实现**（`torchtitan_npu/patches/.../decoder.py` 的 `_get_attention_metadata`），不要重写遍历；把产物塞进 extra kwargs；CP 显式 NotImplementedError。
3. **虚拟融合 + tensor_semantics**（triton/自定义算子 Meta 不可执行时）：参照 `compat/qwen3_5.py` 的 gated_delta_rule/flex_attention——显式 fake/Meta、autograd、真实保存输入、`_register_fusion_semantics` 声明角色/轴（角色取 `meaning.role`）；未知轴为 None。

### Step 4 runner 豁免（稀疏/特殊参数）
参数不产生 dense `.grad`（如 SparseAdam 稀疏表）→ `_gradient_manifest` 按**参数上的生产标记**豁免（v4_1 用 `weight._engram_host_offload`）；不要按参数名 grep 豁免。validation.py 会校验 clip 集合 == optimizer 更新集合 == 梯度清单长度。

### Step 5 测试 `tests/test_<model>_capture.py`（仿 test_qwen3_5_capture.py / test_deepseek_v4_1_capture.py）
- parametrize：single / FSDP2 rank0 / FSDP2 rank1 / full-AC（每 case 约 2 分钟起，控制组合数）。
- 断言清单：`validate_workload` 通过；梯度清单长度（总参数张量 − 无 dense grad 的豁免参数）；豁免名单精确匹配（如 `endswith("engram.table.weight")`——**别用宽子串**，gate 等同名前缀 dense 参数会误伤）；自定义语义 op 集合与次数（full-AC 下 forward 重算 ×2）；memory_plan 公式。
- **内存公式先实测再写死**：跑一次取 persistent/optimizer bytes，推导可解释公式（FP32 参数驻留、FSDP2 逐参数 shard 的奇 numel 偏移落 rank1、SparseAdam state stub 常量），注释写明推导。奇 numel 参数在 shard=2 时不是 ⌈⌉/⌊⌋ 均分——本例是 rank0 精确一半、rank1 少 80 元素。
- ruff 120 列（容器内跑 `ruff check experiments/simulator/`）。

## 实测对比工作流（可选，qwen35-real-bench 为模板）

1. **独立实测仓** `torchtitan-npu-real`（clone 自同 base），容器内 `PYTHONPATH=/workspace/torchtitan-npu-real` 覆盖 editable 安装实现隔离；tokenizer 用 `/workspace/torchtitan/tests/assets/tokenizer`。
2. 实测容器由仿真容器 docker commit + 挂载空闲卡（宿主机 npu-smi 查占用）；等卡脚本 `wait_and_run.sh`。
3. profiler 参数：`--profiler.enable-profiling --profiler.profile-freq 4 --profiler.profiler-warmup 2 --profiler.profiler-active 1 --profiler.save-traces-folder ...`（active 步 = warmup2+freq4 → 第 3、7 步）。
4. 对比口径：**ACLNN（实测 trace）vs ATen（仿真 workload）层级差**，只比算子个数/shape/dtype，不比耗时。聚合脚本 `parse_real.py`（实测）与 `scripts_bench/dump_sim_workload.py`（仿真）输出同构 JSON。
5. 已知层级映射（写进报告，别当缺口）：GDN fwd 1 融合 op↔3 triton kernel、bwd 1:1；flex 1↔2/4；fused_adamw 1 list-op↔逐参数 kernel；GMM tokens 均衡路由（2048）vs 实测非均衡（1710）为路由假设偏差；vision conv scope 差异。

## 已知坑（按踩过顺序）

1. **triton-ascend 3.2.1 预编译头** `-std=c++17` 与 torch 2.14（ATen.h 要求 C++20）冲突：实测容器内 sed 改 `triton/backends/ascend/utils.py` 两处为 c++20。
2. **scipy 1.14+ 与 numpy 1.26 不兼容**：实测容器装 scipy==1.13.1；补装 av/einops/pillow。
3. **MoE 8P 脚本内置 EP=8/DP_SHARD=8**：4P 时显式 append `--parallelism.data-parallel-replicate-degree 1`；GBS 与 DP 不整除时 `--training.global-batch-size -1`。
4. **多模态 debugmodel capture 需要 vision detach**（qwen3.5 做了，v4.1 未做）：text-only 数据下 vision encoder 全参数无梯度，warmup 梯度校验即失败；multimodal 的 VL router bias 同理。先只交付 text 捕获并在能力矩阵标注。
5. **ruff --fix 会重排导入破坏"先 ensure_importable 再导入上游"的顺序**：显式函数调用，不靠导入副作用。
6. **memory_plan 数值"漂移"先查 rank 差异再查非确定**：FSDP2 逐参数 shard 下 rank0/rank1 persistent 不同是正常的（奇 numel 偏移），按 `logical_rank` 参数化断言。
7. **CLI 静默成功**：没有"Training completed"类输出；确认结果用 PYEXIT + workload 对象，别 grep 日志。
8. **memory 断言公式必须可解释**：写死观察值时注释推导来源（engram 表 EP=1 全表驻留、SparseAdam stub 4496B 等），否则后续维护者无法区分 bug 与偏移。
9. 报错 `'XModel' object has no attribute 'build_attention_masks'` 的正确读法：查主包 patched decoder 的 prepare-input 是否已有自动分支（`_get_attention_metadata`），适配器复用而非重写。
10. **能力矩阵/README 同步是交付的一部分**：模型行写清边界（text-only、无 MTP/CP/量化），引用测试文件路径；不新增阶段迁移文档（历史证据归 Git）。

## 先例与产物索引

- qwen3.5：simulator 仓 `a7ce3df`（compat/qwen3_5.py 虚拟融合 + 测试 12/12）；`tests/test_qwen3_5_capture.py`；`docs/qwen3-5-integration.md`（工作流关卡式验收记录）。
- deepseek_v4.1：simulator 仓 `a25a543`（compat/deepseek_v4_1.py + 测试 4/4）；`tests/test_deepseek_v4_1_capture.py`；capability-matrix 有模型行。
- 实测对比：`/home/l30079348/titan030-qwen3.5/qwen35-real-bench/`（REPORT.md 为报告模板，含修改计划 P0–P2）。
- 仓库规则：`experiments/simulator/AGENTS.md`（Meta 捕获边界、内存/语义纪律）+ `docs/model-integration.md`（接入工作流）+ `docs/capability-matrix.md`（当前能力）。
- 总结文档：`/home/l30079348/titan030-qwen3.5/WORKSUMMARY.md`。

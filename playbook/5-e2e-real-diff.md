# 5 · 端到端运行并与实测对比

> **仓内权威规则：`AGENTS.md`（性能预测/DES/最终指标归下游平台，不在训练仓恢复）+
> `docs/capability-matrix.md`（能力边界）**；sidecar 回归与类型检查入口
> `bash scripts/validate.sh`（model-integration.md §7）。
> 样例：`case-multi-head-moe/06-sim-vs-real-diff.md` + `archive/mhoe-real-bench-REPORT.md`（对比报告模板）。

## 5.1 sim-platform 端到端（原 simulator 无端到端耗时）

1. toml：`[training]` config/world_size/dp_shard/ep/local_batch_size/seq_len/output_dir +
   `[platform]` hardware=<chip.json 路径> + `[prediction]` backends=["roofline"]。
2. 入口 `sim-platform train <toml>`（`sim_platform.cli.training:main`）。
3. **导入手术**（模型不在平台默认 editable 环境时，样例 `run_sim.py`）：剥离 editable finder →
   `sys.path.insert` 指向目标仓（含 `experiments/simulator`）→ **预导入** `torchtitan_simulator` +
   `torchtitan_npu`（缓存 sys.modules，抢在平台解析前）→ 再调 CLI main。
4. 产物 `runs/<name>/l1/`：`statistics/operators_by_name.csv`（归一化算子族）、`training.json`
   （end_to_end_us 等）、`000_F/001_B/002_OPTIMIZER.csv`（原始捕获）、trace 导出。

## 5.2 真机实测

1. **卡与容器**：宿主 `npu-smi` 全卡无进程 ≠ 可用——容器级 dcmi/aclInit 持有会报 507899；
   解法 = **同镜像同挂载新建 privileged 容器**刷新设备会话。
2. 环境修复清单（样例）：scipy 1.18 与 numpy 1.26 不兼容（np.long）→ 回退 1.13.1；
   triton-ascend C++20 冲突 → sed 两处；tokenizer 路径 `/workspace/torchtitan/tests/assets/tokenizer`。
3. 运行：`NGPU=2 bash examples/<model>/...sh` + profiler 参数
   `--profiler.enable-profiling --profiler.profile-freq 4 --profiler.profiler-warmup 2
   --profiler.profiler-active 1 --profiler.save-traces-folder <dir>`（active 步 = 3、7）。
4. 聚合：自写健壮脚本解析 `kernel_details.csv`（引号变体会击穿模板脚本），按 active step 过滤。

## 5.3 对比口径（默认只比计算算子，不比耗时/通信；需求方另有口径以需求为准）

1. **先核 dtype**：真机栈默认 bf16 混精度，仿真捕获可能 FP32——不影响算子种类但影响 cast 计数
   与 impl 选择，报告必须声明（必要时仿真切 bf16 重捕获）。
2. real ACLNN kernel 族 ↔ sim 归一化算子族：**先建映射表，再报缺口**。
3. 已知粒度映射（写进报告，别当缺口）：fused_adamw 1 list-op ↔ 逐参数 kernel；cast 逐算子插入；
   MoE dispatcher 路径差异由虚拟算子覆盖语义；GDN/flex 融合 op ↔ 多 kernel。
4. 反向差异：sim 的 view/expand/permute/detach 等元数据算子实测无设备核——预期正常。

## 5.4 复合算子缺口（repeat_interleave 类，方法论）

- 症状：sim 捕获 0 次，真机出现成对 kernel（`aclnnRepeatInterleaveV2` + 配对 BroadcastTo）。
- 根因：复合隐式算子在 TorchDispatchMode 下分解后才可见；真机直连 ACLNN 核不分解。
- 修复三件套（**不动模型源码**）：捕获初始化 `torch.library.register_fake("aten::repeat_interleave", meta_fn)`
  → op_mapping 加映射 → opm 加 bytes 拷贝类 cost handler（read in + write out×repeats）。
- 决策前量化影响：kernel 数占比 + 设备时间占比 + 随 batch 增长（样例：bs=1 时 0.33% 设备时间）。

## 5.5 跨芯片（不同硬件）性能对比约定

"跨芯片口径"= 同一模型在不同芯片上对比性能时的统一约定：

- 对比走 sim-platform 链路：端到端耗时由平台 DES 输出，offload 传输耗时由算子建模仓
  （operator-performance-models）的 host-device transfer model 覆盖——这些能力已具备，无需另行开发。
- **不标定、不 DVFS 校准**——跨芯片是相对对比；各芯片各用其真实规格档（如 DV120），
  "只翻译不标定"。
- 报告中先声明已知偏差：① 复合算子缺口（§5.4）；② cast 粒度差异；③ dtype 口径
  （仿真捕获 vs 真机混精度）。

## 5.6 交付

- 对比报告（模板：`case-multi-head-moe/archive/mhoe-real-bench-REPORT.md`）：配置对齐表、
  语义对齐族清单、差异发现（按"实质缺口/粒度差异/路径差异/口径修正"分类）、结论与修复建议。
- 产物索引：仿真 toml/导入脚本/L1 导出、实测 trace/聚合 JSON/运行脚本路径全部列明。

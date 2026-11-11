# 1 · 迁移模型代码到仿真仓库

> **仓内权威规则：`experiments/simulator/docs/model-integration.md` §1–§5（先读它，本文是实操补充）**。
> AI 可执行细节：`skills/titan030-sim-model-adapt/SKILL.md`。样例：`case-multi-head-moe/02-model-migration.md`。

## 1.1 判定模型来源与复用路线（对齐 model-integration.md §2"先选复用路线再确定文件"）

| 来源 | 需要做什么 |
| --- | --- |
| 主包已有模型（上游 0.3.0 没有，如 deepseek_v4_1） | 仅 simulator 接线（Step 3） |
| 上游 0.3.0 原生模型（如 qwen3.5） | 主包模型已有，simulator 需 compat 层 |
| **全新外部模型（如 multi-head-moe，源码在别的仓）** | **忠实移植**到 `torchtitan_npu/models/<model>/` + simulator 全套接线 |

## 1.2 忠实移植纪律（全新外部模型）

1. 源码逐字移植；发现源码 bug 先做最小复现脚本留证，**修复须报 owner 批准**（样例：3 个 bug——
   RoPE 广播/einsum 下标/布局转置）。
2. 参考实现"从未运行成功"是常态：先在 CPU 上把参考实现跑通（黄金参照的基线，见 playbook 2/4）。
3. 初始化缺失（`torch.empty` 不初始化）→ 补标准初始化并留档；数值边界行为（如无因果 mask）
   忠实继承，在文档标注，不自行修。
4. pristine 原版存档（`*_ORIGINAL.py`），修正注释写进文件头。

## 1.3 simulator 接线（checklist）

1. **preset**：`presets.py` 加 `_<model>_trainer_config` 包装 + `_<model>_debugmodel[_text]`
   （包装函数拦截工厂不接受的参数；`ensure_<model>_importable()` 存在则**显式调用**，勿依赖导入副作用）。
2. **config**：`config.py` 注册 factories + family 白名单 + `_<MODEL>_OVERRIDES`
   （对照生产脚本 `examples/<model>/debug/*.sh` 的 override 组合）。
3. **模型包**：`torchtitan_npu/models/<model>/{__init__, model, config_registry, parallelize, sharding}.py`。
   - sharding 注解（spmd_types 必需）：plain 参数给 Replicate ShardingConfig；
     **勿套通用 `set_moe_sharding_config`**（会给 wrapper 注入不存在的状态声明）。
   - 上游框架契约对齐（如 Decoder.Config 要求的 rope 元数据、`reset_parameters`、Meta autocast 守卫）
     ——这是框架适配，不是算子替换。
4. **compat 层**（按需，见 skill Step 3）：Meta 初始化桥 / build_attention_masks 适配器 /
   虚拟融合+tensor_semantics（**须 owner 批准 + 默认关闭开关**，见铁律）。
5. **拒绝项 fail-fast**：TP/CP/MTP 等不支持的并行方式给出带理由的 `NotImplementedError`/`ValueError`。
6. **能力矩阵**：`docs/capability-matrix.md` 加模型行（边界如实：text-only、无 MTP/CP/量化等）。

## 1.4 参数公式与内存公式（先实测再写死）

- 参数总量/张量数手工推导 + 构建实测对照（样例：63,395,104 / 139）。
- persistent/optimizer bytes 公式跑一次真实模型取值后写死，**注释写推导**（FP32 驻留、奇 numel
  shard 偏移、优化器矩量约定差等），否则后续维护者无法区分 bug 与偏移。

## 1.5 交付与 Git 状态

- 新增/修改文件清单入交接文档；worktree 分支开发，主 checkout 保持干净（共享环境）。
- 如实记录：哪些已提交/推送、哪些未提交（未提交也是状态，防"以为推了"）。

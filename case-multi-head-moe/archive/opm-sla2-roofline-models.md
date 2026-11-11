# SLA2 稀疏/线性注意力 roofline 模型

multi_head_moe 虚拟融合算子（`multi_head_moe_sim.sla2_*`）的成本建模：
sparse 与 linear 两个分支 × 前向/反向共 4 个语义族、4 个 adaptor、4 个 roofline 模型。
与 MHC 投影同策略：显式 FP32 cube GEMM 公式 + 声明式结构 vec 单位，
组成 `max(cube_us + total_vec_time, memory_us) + overhead_us`。

## 语义与几何

- query/key/value/output：`[B, H, T, head_dim]`；mask：`[B, H, q_blocks, kv_blocks]`。
- `selected_blocks` 经融合算子 schema 标量传入 `node.attrs`（flex_attention 同机制）。
  **命名契约**：sparse 融合的标量是每行选中的 KV 块数；linear 融合的标量按源契约
  记录的是**补集块数**（源参数名同为 `selected_blocks`），adaptor 将其翻译为
  语义字段 `complement_blocks`，linear 模型直接按补集数计 `(complement+1)` 遍历。
- 网格校验：`T % q_blocks == 0`（block = T // q_blocks）且 `kv_blocks * block == T`
  （自注意力等长 KV 网格），不对齐报 `UnsupportedOperator`。
- 选择策略：sparse 要求 `1 <= selected_blocks <= kv_blocks`；
  linear 允许 `0 <= selected_blocks <= kv_blocks`（complement=0 是合法退化），
  `selected_blocks > kv_blocks` 报 `ValueError`。

## 工作量公式

记 `streams = B*H`，`scores = streams * T * selected_blocks * block`，
`complement = kv_blocks - selected_blocks`：

| 族 | cube FLOPs | vec 单位 |
|---|---|---|
| sla2_sparse_attention | 4·scores·head_dim（QK^T 与 AV 两个 GEMM） | 6·scores（softmax） |
| sla2_sparse_attention_gradient | 10·scores·head_dim | 9·scores + mask 元素数 |
| sla2_linear_attention | 2·streams·T·head_dim²·(complement+1) | 2·output 元素数（normalize） |
| sla2_linear_attention_gradient | 2× 前向 | 5·upstream 元素数 + mask 元素数 |

- cube 精度固定 32（`cube_bits` 覆盖），vec 由 `VecOpConfig(32, 1.0, float32)` 声明
  （结构单位 ×1.0 字面量，非标定系数）；未显式提供该精度配置即 `UnsupportedOperator`。
- 流量：前向读 q/k/v/mask、写 output；反向读 upstream 与保存张量、写四个梯度
  （dq/dk/dv/dmask），经 `_FusedRoofline` 的 `inputs`/`outputs` 属性统一计。

## 锚值（seq_len=16 补齐 T=64，B=1、H=4、head_dim=64、block=64、qb=kb=1、selected=1）

sparse 前向 cube 4,194,304 / vec 98,304；sparse 反向 cube 10,485,760 / vec 147,460；
linear 前向 cube 2,097,152 / vec 32,768；linear 反向 cube 4,194,304 / vec 81,924。
seq_len=128（qb=kb=2、selected=1、complement=1）时两融合前向总工作量
8,585,216 / 8,454,144。回归见 `tests/test_sla2_roofline.py`。

# Plan: decode stage pipeline (018) — architecture decisions

> Derived from specs/018-decode-pipeline/spec.md · 017 series closed.

## Architecture Decision Record

### D1 依赖 DAG（跨层可并行的边画像）

层内严格链：`gather → qkv → flash → o → add_rms(o) → gu → down → add_rms(down)`
→ 下一层 `gather`。**真实数据边**：下一层 qkv 需要 `add_rms(down)` 输出；除此
之外（如 flash 的 o 投影、gu 的权重流）在同一层的 stage 之间是**单生产者**链。

**候选重叠（018 P1）**：把层体切成两组（A: gather+qkv+flash; B: o+add_rms+
gu+down），B 的 **o 投影与 gu 权重流**与 A 的**下一层 qkv gather 流**无数据
依赖——错开后 B 组可以"抢先"读下一层权重（L2 冷读→热线时序）。

**两栅栏方案**：barA（A 组同步只服务 A 链）与 barB（B 组）——总环节点从
每层 1 个全网格 barrier 变为 2 个 halves + 1 个边 barrier（o→add_rms 消费、
gu 消费 add_rms(o)）；**真正省时=等待器相互独立**——**第一实验必须只改
参与集合，不改任何 stage 算术**。

### D2 实验矩阵（每步独立开关，测量→位级→回退/保留）

| 步骤 | 改动 | 开关 | 预期 |
|---|---|---|---|
| P1a | A/B 双组 barrier 树 + 边同步（组内成员=生产/消费 tile 集） | `REINFER_FUSED_PIPE=1` | 层均值 −5~10% |
| P2a | P1a + 下一层 qkv 权重 L2 预处理（组 B 空闲时 prefetch 下轮权重） | `REINFER_FUSED_PIPE=2` | −5% |
| P3a | 关键 path（bar-only edges）改 producer-count 轮询（非全广播） | `REINFER_FUSED_PIPE=3` | −2~5% |

每步：位级五门（fused_decode 全量 5 项 + engine A/B）→ REINFER_DECODE_PROFILE
段表 → gpu busy。**零收益或位级失败 → 该步回退**（018 是探索波，纪律=
诚实记录）。

### D3 位级红线（复用 017 的手法，无新增数值函数）

- tile 程序序/j-序/k-walk/归约树（4-ILP 分组、(a0+a1)+(a2+a3) 前缀序、
  256-slot head-norm 树、软件 RNE）逐字节不动——**只改"哪些块在哪个
  时刻进入哪个生产/消费等待"，以及 stage 的缓冲区索引**。
- 需双缓冲区（o/gu 行 2MB×2×28 层?——**实际上单步，只加层内 2×2MB
  的临时页即可**（上步结果用完释放——位级无变量）。
- 现有断言口径不变（`layer_fused_bw_ab_bitwise` 等）。

### D4 开关与回退

`REINFER_FUSED_PIPE=0/1/2/3`（缺省 **0** = S1-11 字节一致）；与既有
REINFER_FUSED_BW/WC 正交（任一关→更粗回退）。graph 节点数不变（仍是
31 launches）——graph 桶兼容（REINFER_GRAPH 默认 off 不变）。

## Module/File Plan

| 单元 | 文件 | 内容 |
|---|---|---|
| P1a | `decode_layer_fused_kernels.cu` + `layer_fused.rs` | 双组 barrier 树 + 边 barrier（参与者集合×2 上传） |
| P2a | 同上 | 组 B 轮空期 L2 prefetch（`prefetch.global.L2` 不改变数据） |
| P3a | 同上 | 关键边 producer-count 轮询 |
| 验证 | `crates/cuda/tests/fused_decode.rs` | 5 项位级门 + engine A/B 保持 |
| 记录 | `bench/notes.md` §S1-12 | 每步 A/B 表 + 零收益记 |

## Wave plan

```
Wave alone (serial mini): P1a → P2a → P3a（每步独立判定; 038 全按零收益→回退）
```

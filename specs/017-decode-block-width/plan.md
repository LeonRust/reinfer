# Plan: decode block-width (017) — architecture decisions

> Derived from specs/017-decode-block-width/spec.md · Parent 006-2 ·
> Continue of the S1-10c record (single-block hotspots after the power raise).

## Architecture Decision Record

### D1 块宽化模型（不改列序，只改划分）

layer-fused kernel：grid = min(occ×82, max_tiles) = 82 blocks，512 线程，
层内 8 stage 用 sense-reversal atomic grid barrier 串行化。各 stage 的
计算负载目前由**同一组 block**按自身扫描顺序承担；单块串行段（p2_o、
gather、add_rms、p1_gu 等）的实际并行度取决于 stage 内部的多线程划分，
与 grid 宽度无关。

**本波 = 把 stage 的工作按"列区间连续切块"分给更多 block**：
- block b 承担 `cols = [b*width, (b+1)*width)`（列序/归约/前缀序不变）；
- 每个新增块参与相应 barrier（参与者集合扩大——S1-10b 的
  partial-participant DAG 已按集合产生者建模，扩集合即可）；
- 跨 stage 的交换（x/o/down buffer）按列区间取用——**无新 global 依赖**。

### D2 位级论证

- 每个 stage 的**数值函数**（matmul 分块、rms 树、softmax、swiglu、
  flash scan）不接受块宽——只接受"我负责的列区间与读取范围"；
- 列区间切分保持 ascending-slab 段序与 4-ILP 分组（S1-9d D7 已断言）；
- 新增 block 只增加**执行实体**，不改变**计算结果序**（同一 stage 的
  各部分结果位置不变；聚合 partials 依然由同一最终块序完成——若
  聚合首尾块变化，拷贝/还原仍按同一地址序）。

### D3 候选段与收益表（2026-09-01 profile，µs/step）

| 段 | 时间 ms | 当前语义 | 块宽化后目标 | 预期回收 |
|---|---|---|---|---|
| p2_o | 0.83 | 2 MB 读 + 2 MB 写/层，单块+部分 | ≥2× | -0.35 |
| gather/rms0 | 0.69 | 8 MB 首层/后续 gather | ≥2× | -0.25 |
| p1_gu | 0.74 | 4 MB 读×2 per 层 | ≥2× | -0.25 |
| p2_qkv | 0.40 | 2 MB/层 | ≥2× | -0.15 |
| add_rms(o/down) | 0.55 | 1 KB-2 KB/层串行 | ≥2× | -0.20 |
| flash / p2_gu_d | 0.40+0.26 | 不动 | — | 0 |
| lm_head | 0.55 | floor | — | 0 |
| **合计** | **4.34** | | | **-1.0~1.2** → ≤3.3 |

收益评估为**上限**（DRAM 流受 896 GB/s 约束；块宽化解救的是"处理单元
利用率"）。**gather*p1_qkv 首层较大**——首层 gather 8 MB 读（权重）在
带宽 floor 附近，放宽收益有限——**以实测定夺**（task T1/T2 逐段验证）。

### D4 开关与回退

`REINFER_FUSED_BW`（缺省 on——块宽化随层融合默认生效;off → 现 S1-10
行为）；与既有 REINFER_FUSED/REINFER_LAYER_FUSED 正交（三者任关→较粗
回退,位级各由自身测试）。**graph 桶**：新 kernel 节点数不变（31），
graph.rs 只认节点数——块宽化不破坏 graph rebind（kernel source 变化回退
到 eager——graph 桶由 REINFER_GRAPH 控制,默认关）。

### D5 阶段拆解（两波，短步）

- 017-a：T1 审计 + **gather/p2_qkv/p1_gu 块宽化**（最机械,低位级风险）
- 017-b（依赖 a）：**p2_o/add_rms 块宽化**（写路径,更细）+ 段表/门禁判定

## Module/File Plan

| 单元 | 文件 | 内容 |
|---|---|---|
| T1 | `crates/cuda/src/layer_fused.rs` + `decode_layer_fused_kernels.cu` | 每段块覆盖审计 + 位级探针（clock 段采样） |
| T2 | 同上 + `fused.rs` | 块宽化参数化（grid 宽/bw 开关）+ 验证测试 |
| T3 | `bench/gate-fixture.md`/notes.md/roadmap | 门禁判定与记录 |

# Tasks: decode block-width (specs/017)

> Status: open 2026-09-01 · Wave A: T1 审计 + 机械段块宽化（gather/p2_qkv/
> p1_gu）→ B: p2_o/add_rms → C: 门禁（与功耗抬升并列等待）
> Registration: roadmap S1-11；S1-10c 记录续集。

## T1 — 段覆盖审计（块宽化的出发证据）

- [ ] 读 `decode_layer_fused_kernels.cu` 各 stage 的线程/block 使用：每个
      stage 中以 blockIdx.x 驱动的循环结构、每块行/列划分、单块串行的段
      （p2_o、gather、add_rms 等）——标出"当前 block 数 vs 需要的并行"
- [ ] 用 REINFER_DECODE_PROFILE 分段采样（window 21-40，mean 20 steps）
      给出 017 plan D3 表的实测核对值（每段 µs/层×28 → ms/step 与
      0.83/0.69/0.74/0.40/0.55 对照）
- [ ] 产出：审计表（段/线程模型/理论 floor/实测/可宽化数）

## T2a — 机械段块宽化（gather / p2_qkv / p1_gu）

- [ ] 内核参数化：grid 宽 82 → 82×W（W=2/4，env `REINFER_FUSED_BW` 缺省
      W=2）；列区间连续切块；barrier 参与者集合函数更新（S1-10b
      partial-participant DAG 集合扩展）；交换 buffer 按列区间使用
- [ ] **列序/归约树/聚合前缀序零改动**（D2 位级论证的前提）
- [ ] 位级验证：`layer_fused_li1_bit_exact_vs_split`（三段 q/k/v、attn、
      x、xn_attn、down、7 partials、page-1 KV 写 0-ulp）+ 双跑确定性 +
      D7 聚合序断言保持
- [ ] 段表：W=2 的 p2_qkv/p1_gu/gather 均值下降 ≥25%（或记录"已 floor"）
- [ ] 回退面：REINFER_FUSED_BW=off == S1-10 行为（位级）

## T2b — p2_o / add_rms 块宽化（依赖 T2a 模板）

- [ ] 同样式处理 p2_o（2 MB 读写/层，W≥2）与 add_rms(o)/add_rms(down)
      （单块串行 → W≥2）
- [ ] 位级/段表/回退同 T2a 判据

## T3 — 门禁与记录（功耗抬升后执行）

- [ ] `bench/perf-gate.sh` PASS（≥299.8 tok/s；功耗 110W 前提在
      gate-fixture.md 记录）
- [ ] notes.md S1-11 节（段表前后对照、位级、门禁判定、功耗记录）+ road
  map S1-11 + gate-fixture 手动填写
- [ ] 若 FAIL：记录退化表（后续 multi-stream/warp-spec 一阶）

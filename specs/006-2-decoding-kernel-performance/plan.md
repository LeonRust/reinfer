# Plan: decode-side kernel performance (006-2)

> Derived from specs/006-2-decoding-kernel-performance/spec.md · 决策依据：006 plan D1
> （选择链归属）、014 plan（parity 判据 / Backend 签名）、005 plan D5（采样全链数学定义）、
> benchmark-gap §3 G3-G5。r2（2026-08-29 四代理评审修订）：G1/G5 挂起、T5 不做、
> T0 微基准前置、RNG 口径三层化、风险段补全。

## Architecture Decisions

- **D1 选择链终点扩展（decode 系；条件化前置）**：decode-attn 与融合核组共用 006 D1 链
  `Vendor(cubin/dylib) > Jit(fmha/fused) > Jit(dense/003 naive)`。**T1（vendor）条件化**：
  三条件齐备才执行——① 现成 API 的供应商资产 ② 许可扫描通过（R2）③ 目标架构（sm120a）
  有 cubin；任一不满足 → 走 T2（Jit）。**T0 微基准先决**：003 naive decode-attn 隔离
  与整机 vs llama.cpp CUDA 门禁（1-2 人日实验）；若 ≥0.85× → G3 收口为"记录"，
  T1/T2 全砍——比先立项便宜一个数量级。sampler 双路径共驻：`GPU sampler > CPU
  (llm-samplers 现有)`；CPU 路径常驻为回退 + 差分参考（双路同错风险见 R3）。
- **D2 确定性契约（r2 三层化）**：① temp=0：无 RNG 路径（软硬件 argmax，
  tie-break=取首个最大，与 005 D5 锚一致）——bit-identical 与 RNG 谱系**无关**；
  ② temp>0：仅承诺与 CPU 路径同分布；GPU 实现（SplitMix64 内核或 counter-based
  如 Philox4x32）**必须以 (i,p,v) 纯函数索引推进**（005 D5 数学定义），排除
  "独立 stream 顺序消费"的歧义读法——**sampler 用单流**（避免每步 event/同步，
  且与 006 捕获期"唯一流"规则一致）；③ **CPU 侧迁移衔接**：当前 CPU 路径为
  llm-samplers 0.0.7（rand StdRng=ChaCha12 谱系，非 SplitMix64）；005 纯函数化
  落地后统一至 SplitMix64 谱系——本 spec 承诺的对象是"采样语义/谱系统一后的
  CPU 路径"（同分布契约在其落地期内的验证口径记入 notes）。
  **LogitsView 所有权（L1）**：引擎每步 logits 生命周期收归 GPU；CPU 回退经
  `.to_host()` 惰性触发**承续 014 `Backend::logits() -> Vec<f32>` 签名**（接口缝补记）。
- **D3 融合核组（r2 裁剪；仅当 006-2b 重开）**：按"每层 kernel 数与寄存器往返"排序：
  ② fused MLP-SiLU（每层 3→1 核，权重占比 ~70% 主导算子族，CUTLASS epilogue 优先）
  ① fused norm+add（简单算子，D1 自留地，与 CPU 路径共享语义）
  ③ fused QKV+RoPE ——**条件化**：仅当 Jit attn 路径需要 contiguous qkv 前端时做
  ④ attn-out+add+norm ——**不做**（graph-on 无收益，差分面照付，spec hedge 削除）。
  全组套收益尺：**≥max(5%, 2× 噪声带)**（同 006 协议 5 次中位数 vs 锁定基线）；
  计数口径：per-layer ≤4、per-step = 4L+2（L=层数）+ lm_head + sampler；
  graph 视图 = 桶内稳定态 CUDA kernel 节点数（vendor 内部按捕获节点计）。
- **D4 与 006 graph 交互（不变）**：融合核属 eager 基线；graph replay 与 eager
  差分 ≤1 ulp（006 硬门）；融合改变桶内容 → 006 桶按形状重捕获；不一致 → 桶回退
  eager fusion（不静默）。012 JitKey（内容 hash+flags+capability）自动区分融合变体；
  TuneDb 记录带变体号。
- **D5 warp specialization：不做（决策记录）**——设计报告 §8 D1（自研只攻简单算子）；
  单流 decode 带宽受限（bandwidth-bound）而非 SMEM 吞吐受限，WS 的 producer/consumer
  重叠收益假设不成立；如未来 006-2b profile 显示 attn 内核为 compute-bound 占比显著
  （>40% kernel 时间），可经新增量 spec 重开（本决策不做，非否决）。

## Module Breakdown

| 模块 | 变更 |
|---|---|
| `crates/kernels` | LogitsView 抽象（L1）；sampler 链 trait（GPU/CPU 两实现 + 回退注册）；选择器扩展（decode-attn 条件档）；TuneDb 变体号 |
| `crates/cuda` | GPU sampler 内核（penalty+softmax+topk/argmax，单 launch）；（条件档）decode-attn Jit 档；（006-2b）融合核 |
| `crates/samplers` | 现状：llm-samplers 0.0.7 包装；保持为 CPU 回退实现与差分参考（不改其 RNG） |
| `bin/reinfer` | `--perf` 输出扩展（tps + 计数器）；无其他产品面 |
| `bench/` | notes 四元组、baseline.json、llama.cpp CUDA 参照测量产物（T0/T6） |

## Interface Contracts

- `LogitsView`: GPU 常驻 logits；`fn to_host(&self) -> Vec<f32>`（惰性回退拷）；生命周期
  归引擎（每步复用 buffer）；与 `Backend::logits()`（014 host 签名）语义承续关系注明于实现。
- `SamplerChain`（trait）: `fn sample(&mut self, logits: &LogitsView, params, rng_state) ->
  TokenOut`；GPU/CPU 两实现；参数面与顺序 = 005 D5 全链
  （bias→penalties→bad words→temperature→min_p→top_k→top_p→gumbel→argmax）；
  未覆盖参数 → 显式回退 CPU 并记录。
- 计数器枚举（统一命名，006 风格）：`sampler_gpu` / `eager_fallback` / `padding_ratio`
  （回退 ⊆ eager 比例；5 分钟内 >20% 告警）。

## Risk Assessment

| 风险 | 等级 | 缓解 |
|---|---|---|
| R1 差分矩阵扩增：每核 × {eager, graph} × {arch 梯度} × {回退链} | 高 | core-listing 表集中管理；每核固定 3 shape 差分（禁止全 shape 空间）；runner 时间预算入 008 |
| R2 vendor 许可/布局：flashinfer（Apache-2.0）/CUTLASS epilogue 的 cubin 入库→分发须例外声明；vendor KV 布局与 003/005 页表布局耦合 | 高 | 许可扫描 = T1 前置；结论"不可分发" → 跳过 vendor 档并记录；页布局差异入 notes |
| R3 自参照金标：CPU 路径同时是 bit-identical 参考与回退，双路同错 | 中 | 新增 llama.cpp 采样结果对拍（独立差分 T0 附项）；规格固定 0.0.7 |
| R4 时间盒：measure-gated 任务（T2、006-2b 融合项）静默 fallback | 中 | 每项显式 2 周盒；未达标 → note 关闭并维持 003/006 路径（防"试过了没交付"） |
| R5 GPU/CPU 双实现长期维护面（确定性/分布漂移） | 低 | D2 谱系统一序号；分布记录档（Jaccard/TV null 对照）；变体号入 TuneDb |

## Execution Plan (tasks ordering)

1. **T0** 微基准（前置：003 naive decode-attn 隔离 + llama.cpp CUDA 参照构建/测量 → baseline.json）
2. **T1** decode-attn vendor 档（条件化三条件；许可扫描；无 → T2 直通）
3. **T2** decode-attn Jit FMHA 档（gencode 表锚 003 plan D2——（sm120a ≥12.8 而非旧值 13.0））
4. **T3** GPU sampler 链（D2 三层契约；函数级 bit-identical；单 launch；单流）
5. **T6** 基准回归（阶梯=预期轨道记录；0.85× 判据；CI 红 δ≤0.9×）
6. **006-2b 触发条款**：006 落地后 profile（ncu）评估 → 按 G3/G5 挂起项重开
   依赖：T0 先于全部；T1/T2 并列；T3 独立（最早可并行）；T6 最后。006-2b 不在本轮门禁内。

## Ref

- specs/006-cuda-perf/plan.md（D1 选择链/D4 graph/D7 基准协议与供应链继承）
- specs/014-cuda-l3-single-request/plan.md（D6-D8 判据；Backend::logits() 签名；F16 100% 未达先例）
- specs/005-scheduler-serving/plan.md（D5 采样全链数学定义与参数顺序；D9 前缀接口）
- specs/003-cuda-l0/plan.md（D7 容差表唯一源；D6 RNG 纯函数）
- docs/design/benchmark-gap-2026-08-29.md（§3 G3-G5、§4 阶梯）

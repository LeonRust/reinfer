# Spec: decode-side kernel performance (006-2)

> Status: **approved (r2 · four-review 2026-08-29, owner sign-off)**（评审：契约/门禁/SDD/风险
> 四视角；判定 = 本轮仅保留 G4 契约件，G3/G5 挂起 006-2b）· Parent: specs/006（006 遗留清单"后续增量 spec 006-2"）·
> 证据：docs/design/benchmark-gap-2026-08-29.md §3/G3-G5、§4（Wave-1）· 引用标签为
> benchmark-gap §3 的 G-标签（G3=decode-attn 性能档、G4=采样链 CPU→GPU、G5=融合核组）。

## Problem Statement

解码每步（28 层 · Qwen3-0.6B 基准）仍有三块结构低效（实测 gap 见基线文档 G3/G4/G5）：

1. **decode-attn 无性能档（G3）**：`003/014` 只有正确性档（naive paged GQA；
   014 D7 参考=CPU naive），006 FMHA 仅覆盖 prefill——decode 侧无 flash 风格多级
   批形式内核或 vendor 档。
2. **采样/后处理回 CPU（G4）**：采样链当前全在 CPU 侧（crates/samplers =
   llm-samplers crates.io 0.0.7 薄封装 + rand StdRng），`003` 的 sampler 定义仅
   "纯函数 RNG"（CPU 参考语义）；每步 logits 回拷 + CPU softmax/topk/argmax；
   vLLM 全程 GPU（penalties/softmax+TopK 单 kernel）。
3. **无融合核组（G5）**：每层 ~3-6 核（norm/qkv/attn/proj/ffn1/act/ffn2/add…），
   CUDA graph 已摊平 launch 开销，但无 fused norm+add / fused MLP-SiLU 等
   vLLM 标配融合核——核间数据搬运与寄存器/SMEM 往返仍存在（**decode 侧 HBM
   搬运收益 <5%：每步激活往返仅 ~10MB，权重流 ~1.25GB 为下限；融合收益主要在
   eager 模式 launch 与寄存器往返，graph-on 下需 profile 实证**——见 006-2b 门控）。

## Goals / Non-Goals

**本轮落地（G4 契约件）**：
- G2a GPU sampler 链契约：penalty 家族 + logit_bias + softmax + topk/top_p/min_p/temperature
  + gumbel/argmax **参数面与顺序完全对齐 005 D5 全链**（bias→penalties→bad words→
  temperature→min_p→top_k→top_p→gumbel→argmax）；未覆盖参数显式回退 CPU 并记录。
- G2b logits 所有权：LogitsView 抽象（GPU 常驻，不做每步回拷；CPU 回退经 `.to_host()`
  惰性执行承续 014 `Backend::logits()` 语义）。
- G2c 确定性：temp=0 无 RNG——GPU 与 CPU sampler 在**同 logits 输入**下采样输出
  bit-identical（函数级；端到端 token 一致性归 014 parity 四层管辖）；temp>0 仅
  承诺与 CPU 路径**同分布**（记录档）。
- G2d 前置微基准（T0）：隔离测量 003 naive decode-attn 与整机（vs llama.cpp CUDA
  参照），1-2 人日实验决定 G3/T1/T2 存废。

**挂起（006-2b，006 落地后 profile 门控重开）**：
- G3 decode-attn 性能档（vendor 优先选择链）、G5 融合核组（②① 项 + ③ 条件化）——
  挂起理由：采纳前提依赖 006 后的 profile（ncu）测量，现审即"审数据待定的文档"。
- ~~G6 warp specialization~~：**本轮明确不做并记录理由**——违背设计报告 §8 D1
  "Rust kernel 只攻简单算子"；单流 decode 是带宽受限而非 SMEM 吞吐受限，warp
  specialization 收益假设不成立；另见 006-2b 触发条款。

Non-Goals（不变项）：prefill（006）；CUDA graph 桶/双流（006）；MoE/MLA/FP8、
投机、TP/PP/CP（P3/P4）；CPU 后端（007）；grammar/radix 服务化（005/P3）。
vendor cubin 的仓库/发布/manifest 与许可规则照 006 并在 T1 增加许可扫描（R2）。

## Success Metrics（协议随 006 基准协议锁定；单门禁原则）

- **唯一性能门禁**：decode ≥ **0.85× llama.cpp CUDA**（006 门禁；**参照必须先测量**——
  006-2 T0/T6 前置任务：nvcc 13.2 构建 CUDA 档 referee（f280b2698、`-DGGML_CUDA=ON`、
  `CMAKE_CUDA_ARCHITECTURES=120`）+ llama-bench 按 006 参数表+同模型 sha+5 次中位数 →
  baseline.json）。基准协议六要素逐字继承（KV f16、graph on 双侧、同模型 sha、同 batch、
  预热≥3 中位数、commit+构建标志锁定——**裁判侧 llama.cpp commit/参数锁定照 006 D7**）；
  sm90 回退档（0.7× llama.cpp / 6× CPU）随 006 分档适用（判定机 sm120a 无实际影响，
  但"沿用"写明所适用档位）。benchmark-gap §4 阶梯 = **预期轨道记录，非第二判据**。
- **确定性**：temp=0 函数级 bit-identical（同 logits 输入下 GPU vs CPU sampler；
  argmax tie-break 规则 = **与 CPU 现状一致（LastMax——llm-samplers `max_by` 实测；
  006-2 r2 修订：原"取首个最大"假设被否决）**；005 D5 的 FirstMax 定义留待 005
  纯函数化迁移时统一（届时 GPU/CPU 一起翻）；temp>0 同分布记录档（Jaccard/TV，口径见下）。
- **正确性（硬门）**：014 parity 四层不变（tokenizer 100% / F16 100%（回退档 drift
  ≤1e-4）/ Q8_0 ≥99.9% / logits drift ≤1e-2——端到端不承诺 bit-identical，logits 层
  漂移 1e-4~1e-2 使端到端 argmax 翻转是统计必然，归 014 判据）；新核 vs 003 路径差分
  ≤ 003 D7 容差表；不合 → 强制回退（G2→CPU sampler；G3→003 naive；G5→非融合组合），
  **回退≠错误**（选择链引擎透明），计数口径沿用 006（eager_fallback 等；回退 ⊆ eager 比例）。
- **告警**：5 分钟内回退比例 >20% → 告警（分母 = 当期 eager 执行总数，与 006 口径一致）。

## User Stories

1. 作为引擎作者：对同一 OpConfig 自动选择最优实现档（GPU sampler / CPU 回退；
   未来 decode-attn / 融合核），回退对引擎透明；TuneDb 记录实测量。
2. 作为服务者：`--perf` 输出单流 decode tps 与计数器
   （`sampler_gpu/eager_fallback/padding_ratio` 统一命名）；对比表同 006 协议。
3. 作为维护者：无 GPU CI 仍绿；新计数器的差分矩阵面受控（R1：每核固定 3 shape）。

## Acceptance Criteria

- [ ] T0 微基准完成：003 naive decode-attn 隔离与整机单流 vs llama.cpp CUDA 门禁——
      达标（≥0.85×）→ G3/T1/T2 收口为"记录"；未达标 → 按挂起条款重开 006-2b
- [ ] T1/T2（按条件化执行时）：decode-attn 档差分 ≤ D7 容差（fp16 出）逐 token 一致；
      无相应 arch → 回退 003 直通（不降级假装）；vendor 侧许可扫描通过（R2 前置）
- [ ] GPU sampler 链：参数面/顺序 = 005 D5 全链（CPU 适配器现序为既有序
      repeat→topk→topp→temp，r2 记录：适配而暂不重建，见 T3B 注释）；temp=0 同
      logits 输入下与 CPU 路径 10 prompt × 64 tok **函数级 bit-identical**（tie-break =
      与 CPU 一致的 LastMax）；temp>0 分布记录档（null 对照口径）；
      **单次 decode 步 1 次 CUDA launch（计数规则：不含 lm_head GEMM；含 penalty+softmax+
      采样+TokenOut 统计）**；graph 视图 = 桶内 sampler 节点 ≤1
- [ ] 融合核组（仅当 006-2b 重开）：②fused MLP-SiLU / ①fused norm+add 为先；
      ③仅条件化（Jit attn 需要 contiguous qkv 时）；④ 不计划（理由：graph-on 无收益、
      差分面照付）；全组套收益尺（≥max(5%, 2×噪声)，同 006 协议取中位数）
- [ ] 回归：`bench/notes.md` + `baseline.json`（5 次中位数、commit/构建标志锁定）+
      CI 红判据 δ ≤0.9× 基线；benchmark-gap §4 阶梯作为预期轨道记录并同步校订
- [ ] 无 GPU CI 照 006 三档（lint/build/runner）不变；无新增 SDK 依赖（继承 006 供应链条款）

## Constraints

- 去厂商化（spec 只写 WHAT）：vendor/JIT/微架构细节于 plan/tasks；spec 留 Function 语义。
- 供应链：vendor 资产 manifest 入库 + sha 校验；**许可结论行（R2）为 T1 前置**；
  禁止"脚本下载+同源自校验"（006 条款继承）。
- 确定性为 005/003 系产品不变式：GPU 随机化不得引入与 seed 无关的耗散路径；
  任何 GPU sampler 实现必须以 (i,p,v) 纯函数索引推进（见 plan D2）。
- 双后端节奏：CUDA 为本次目标；CANN/CPU 侧仅需"选择器回退"与"差分参考"存在。

## Risks（简表；明细见 plan Risk Assessment）

R1 差分矩阵扩增；R2 vendor 许可/布局耦合；R3 采样结果为自参照金标（需 llama.cpp
采样对拍做第三方裁决）；R4 时间盒（measure-gated 任务显式 2 周盒）；R5 双轨
（GPU/CPU sampler）长期维护面。缓解措施见 plan.md Risk Assessment。

## Refs

- specs/006-cuda-perf（父；基准协议/供应链/内存记账/容差表**唯一源 = 003 plan D7**）
- specs/014-cuda-l3-single-request（parity 四层 D8 判据；`Backend::logits()` 签名）
- specs/005-scheduler-serving（确定性 D5 全链数学定义与参数顺序；D9 前缀接口）
- specs/003-cuda-l0（容差表 D7；纯函数 RNG 谱系）
- docs/design/benchmark-gap-2026-08-29.md（§3 G3-G5、§4 Wave-1 与阶梯）
- docs/design/feature-list.md（P1-06 登记锚）

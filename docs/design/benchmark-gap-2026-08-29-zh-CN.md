# 与 vLLM 的性能差距——实测基线（2026-08-29）

> 英文主版：`benchmark-gap-2026-08-29.md`（本文为中文并存版）。
> 证据来源：`bench-vs-vllm/` 测试台（`report.md`、`results/{engine}/*`、`probes/`），
> 提交 `cd4620995b6bde104a614aa806cd6ebb7b477b71`；一次成型跑法
> `python run_all.py --engine both`。本文记录**当前二进制交付了什么、缺什么**，
> 作为 specs/005 与 specs/006 的基线引用。

## 1. 测试条件（协议锁定）

| 项 | 值 |
|---|---|
| GPU | RTX 5090 (sm_120a)，JIT 走 driver 侧 CUDA 13.2 |
| 模型 | Qwen/Qwen3-0.6B fp16，双引擎同一目录 `~/.reinfer/models/Qwen/Qwen3-0.6B` |
| 基准（vLLM） | vLLM 0.28.0 官方 wheel；`serve --dtype float16 --max-model-len 4096 --max-num-seqs 1`（F1 公平档）/ `8`（F2） |
| 被测（reinfer） | `reinfer serve` release 二进制，串行引擎队列（V1），fp16，`max-model-len 4096` |
| 语义 | 两侧均 `chat_template_kwargs.enable_thinking=false`（Qwen3 关闭思考） |
| 固定参数 | seed=42；门档 temperature=0；记录档 temperature=1.0 + top_p=1.0；R1 logprobs top-5 |
| 度量 | TTFT / TPOT / ITL / E2EL（客户端单调钟、逐 SSE 帧）；稳态用 pynvml |

## 2. 差距总表（实测）

| 指标 | vLLM（金标） | reinfer（现状） | 差距 |
|---|---|---|---|
| TTFT，c1（并发 1）p50 | 9 ms | 4 926 ms | **≈550×** |
| decode 吞吐（单流，tpot p50） | 363 tok/s（2.7 ms/tok） | 11.2 tok/s（89.6 ms/tok） | **≈33×** |
| 长输入 TTFT（in=2048 词，out=16）p50 | 13 ms | 944 235 ms（≈15.7 分钟） | **≈7 万×** |
| c4（并发 20）TTFT p50 / p95 | 1 822 / 3 594 ms | 106 199 / 207 460 ms | **≈60× 且发散** |
| GPU 显存稳态 | 23 336.9 MB | 3 102.9 MB | KV 池未被使用 |
| T1 greedy token 一致性（temperature=0） | 基准 | **0%**（10/10 首 token 即分歧；EOS 从未触发） | 正确性未闭合 |
| T7 API 兼容 | 8/8 | 7/8（`n>1` 单候选；`stop` 解析但忽略） | 接近对齐 |

补充门事实：`t2/eos_short` —— vLLM `finish=stop`（10 tok 自然停），reinfer
`finish=length`（跑满 64）；R1 分布 Jaccard 0.296 / TV 0.245（仅 3 步对齐——
序列在第 0 token 处即分叉）；R5 稳态两者均 `trend=flat`。

## 3. 差距 → 缺失件映射

每项差距带 G 标签（供 specs/005/006/006-2/014 稳定引用）。

| G 标签 | 实测差距 | 缺失件 | 归属 |
|---|---|---|---|
| G1 | prefill ≈7 万× | FMHA prefill（当前为两段 GEMM + fp32 中间 buffer，即 014 D7 的 naive 结构）；无融合 | specs/006（D2） |
| G2 | decode ≈33×（通用部分） | decode fused Q8_0 dequant-dot 核；CUDA graph decode 桶；双流重叠 | specs/006（D3–D5） |
| G3 | decode ≈33×（attention 结构部分） | decode-attn 性能档（除 003 naive 外无 flash 风格 paged decode 核） | specs/006-2（G1；挂起至 006-2b） |
| G4 | decode ≈33×（采样部分） | 采样/penalty 链在 CPU（llm-samplers crates.io 0.0.7 + rand StdRng）；无 GPU sampler / logits 回拷 | specs/006-2（G2，本轮） |
| G5 | decode ≈33×（launch 部分） | decode 融合核组（fused MLP-SiLU、fused norm+add…；graph 已清零 launch 开销但不清零核内搬运） | specs/006-2（G4；挂起至 006-2b） |
| G6 | 并发 ≈60×、发散 | serve 目前 `engine.lock()` 串行队列；scheduler crate 为 2 行空壳 | specs/005（服务化） |
| G7 | 显存 3.1 GB vs 23.3 GB | token-budget 准入 + KV 池预算（90% 显存）+ `max-num-seqs` 语义 | specs/005（D2）+ P1-02 |
| G8 | （暂无直接度量） | 前缀缓存（RadixCache/vLLM 语义）——见 005 D9 接口承诺 | specs/005（D9）+ P3-01 |
| G9 | T1 0%、EOS 不触发 | EOS 停止语义（§5 O1 待验证）；014 D8 要求「EOS 命中即停」 | specs/014（D8） |
| G10 | t3 断流后服务端仍算完（~100 s 拖尾） | 无客户端断连取消；abort/tombstone 隔离 | specs/005（隔离） |

带宽核算——推导式（不写死模型常量；一切模型量运行时从模型 `config.json` / 测试台
实测取值）：
- 每步权重字节 = 由 config 字段推导：`vocab_size` × `hidden_size` × 2B（embedding；
  `tie_word_embeddings`=true 时与 lm_head 共矩阵、只计一次）+ Σ 各层
  [（qkv：3× hidden 系 + o 投影）+（gate/up/down：3× hidden×intermediate）] × 2B。
  Qwen3-0.6B 实例（取值为该模型 config.json 实际值）：≈1.50 GB/步。
- 机器带宽 = 设备上报值（nvidia-smi 显存频率 × 位宽）；**本机为 RTX 5090 Laptop**
  （256-bit GDDR7 ≈896 GB/s——并非桌面 1.79 TB/s）→ 上述实例天花板 ≈596 tok/s。
- **实测参照**（llama.cpp f280b2698 + nvcc 13.2 + sm120；llama-bench
  `-b 1 -n 512 -fa 1 -ngl 99`，5 次中位数；模型 sha `d04bceb6…` 见
  `bench/baseline-llamacpp.json`）：**352.70 tok/s**（59% 带宽效率，bs=1 小 GEMM
  启动开销所致）→ 0.85× 门禁目标 ≈ **299.8 tok/s**；4K ctx 时 KV 读另加 ≈19%。

## 4. 推进顺序（三波）

1. **波 0 —— 正确性闭合**：EOS（`<|im_end|>`）停止语义；014 parity 四层
   （tokenizer 100% / F16 100% / Q8_0 ≥99.9% / logits drift ≤1e-2）；temp=0 短路
   argmax。**不增益性能**——但在此之前任何数字都不具可比性。
2. **波 1 —— 单请求吞吐（specs/006 + 006-2 G2 契约）**：FMHA prefill（最大杠杆）→
   CUDA graph decode 桶 → fused Q8_0 dequant-dot → 双流重叠 → Vendor 回退链 +
   TuneDb。**006-2（本轮）**：GPU sampler 契约（G4：确定性锚 + LogitsView +
   函数级 bit-identical）。**G3/G5 挂起**：006 落地后 profile 门控重开。
   门禁：decode ≥0.85× llama.cpp CUDA（**先测参照**——见 specs/006-2 T0/T6）。
3. **波 2 —— 服务化并发（specs/005）**：调度状态机
   （Waiting→Prefill→Chunked→Decode→Done/Aborted/Preempted）、连续批、
   chunked prefill + token 预算、token-budget 准入、abort 隔离（tombstone、
   恰一次释放）、抢占=重算（vLLM 语义）、`req_id` 确定性、KV 池预算。
   门禁：P1（decode ≥85% SGLang）。
4. **波 3 —— 成熟度（P3/P4，与「同量级」弱相关、可选）**：RadixCache/前缀缓存
   （共享前缀负载收益明显）、投机解码、grammar、TP/PP/CP、MoE/MLA/FP8。

预期验收阶梯——**预期轨道记录，非第二判据**；波 1 的唯一性能门禁 = decode
≥0.85× llama.cpp CUDA（先测参照，见 specs/006-2 T0）。测量：
`bench-vs-vllm: run_all.py --engine both --suite perf_c1,perf_prefill`：

| 里程碑 | 单流 decode | 长输入 TTFT | c4 TTFT p50 | vs vLLM |
|---|---|---|---|---|
| 现在 | 11 tok/s | 944 s | 106 s | 33× / 7 万× / 60× |
| 波 0 后 | 11（可信） | 944（行为正确） | — | 可比 |
| 波 1 后 | ~150–250* | ~3–8 s | ~50 s | ~1.5–2×* / ~100× / ~25× |
| 波 2 后 | ~150–250 | ~2–4 s | ~2–4 s | ~1.5× / ~20–50× / ~1–2× |

\*注：单流 ~150-250 tok/s 与"~1.5-2× vs vLLM"（=545-726 tok/s）曾**互相矛盾**
——2026-08-29 参照实测后已校订：llama.cpp CUDA = 352.70、0.85× 门禁 = 299.8
≈ **0.83×** vs vLLM 锚（363）→ 阶梯"波 1 后"的 vs-vLLM 列修正为 ~0.83×；
decode 行仍为 006+006-2b 后的预期轨道。
**中间锚点（2026-08-29，006 前 GPU sampler 已落地）**：实测单流 decode
**12.53 tok/s**（tpot 79.8 ms）——G4 契约交付（较 11.16 +12%）；证明 decode 步
主体（79.8 ms）为 dense 层循环 GEMM/attn/launch，即 G3/G5 确认为必做但以 006
profile 为门控（006-2b 触发条款"需测量"前提已由本行满足）。

## 5. 待验证假设（**非结论**，验证前勿当缺陷处理）

- **O1 — EOS id 来源**：`pipeline.rs:144` 比较 `Some(next) == eos_id`，参数来自
  `serve.rs` ← `AppState.eos` ← 模型 `config.json` 的 `eos_token_id`。Qwen3 真实
  EOS（`<|im_end|>` = 151643）在 `generation_config.json` / `tokenizer_config.json`；
  `config.json` 键可能是 `null` → `eos_id=None` → EOS 永不触发。**未验证**：先
  读该值再开 spec 任务。
- **O2 — t3 拖尾**：客户端断连不取消服务端请求（行为已确认：客户端超时后服务端
  仍跑完 ~100 s+ 长 prefill）；serve/pipeline 内根因未追。
- **O3 — SIGTERM**：`serve` 收到 SIGTERM 30 s+ 不退出（`stop_servers.sh` 依赖
  SIGKILL 兜底）；关停路径不完整。

## 6. 相关产物

- `../bench-vs-vllm/report.md` —— 完整报告（门/记录档/性能表）
- `../bench-vs-vllm/results/` —— 每引擎每套件原始 jsonl/csv
- `../bench-vs-vllm/README.md` —— 测试台用法 + 发现清单
- specs/005-scheduler-serving、specs/006-cuda-perf、specs/014-cuda-l3-single-request、
  specs/007-core-inference —— 上文引用的实现锚点

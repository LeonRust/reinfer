# 达到 vLLM 同量级性能路线图——完整功能清单（2026-08-31）

> 英文主版：`roadmap-to-vllm-parity-2026-08-31.md`（本文为中文并存版）。
> 依据：`docs/design/benchmark-gap-2026-08-29.md`（G1-G10）、`bench/notes.md`
> （实测/BLOCKER 记录）、specs/005/006/006-2（approved）。
> 目标锚（实测，RTX 5090 Laptop / Qwen3-0.6B fp16）：
> decode 363 tok/s · long-prefill TTFT 13 ms · TTFT(c1) 9 ms · c4(20 并发) 1.8 s ·
> 显存 23 GB · T1 = 100%。门禁：decode 0.85× llama.cpp CUDA（299.8）/
> prefill 0.7×（238.8）。

## 功能清单（编号 · spec 锚 · 依赖 · 验收）

### Stage 0 —— 数值正确性基座（≈1 周；此前一切性能无可比性）

| ID | 功能 | spec 锚 | 依赖 | 验收 |
|---|---|---|---|---|
| S0-1 | **B2 修复**：2048 词 FMHA 卡死（`FmhaKernels::new` 返回到批量 prefill 循环之间的阻塞） | 006 T1 | — | 2048 词 prompt 走 FMHA <60s |
| S0-2 | **EOS 停止语义**（`<|im_end|>`=151643 取自 generation/tokenizer config，而非 null 的 config 键） | 014 D8 | — | T2/eos_short=stop；T1 prompt 自然 EOS 停 |
| S0-3 | **014 parity 四层对拍**（tokenizer 100% / F16 100% / Q8_0≥99.9% / drift≤1e-2） | 014 D8 | （CPU 档 referee 已就绪） | T1 门 10/10=100% |

### Stage 1 —— 单流吞吐（≈2-4 周；decode→299.8、prefill→238.8）

| ID | 功能 | spec 锚 | 依赖 | 验收 |
|---|---|---|---|---|
| S1-1 | **decode 逐段 profile**（六段 cudaEvent：norm/QKV/attn/o/MLP/lm_head） | 006（记录） | S0 | 归因表入 notes |
| S1-2 | **lm_head 优化**（m=1, n=151936 GEMM + cast + tie-embedding 布局；split 专用核） | 006 增量 | S1-1 | decode ×2-4（实测） |
| S1-3 | **graph 重放接线**（BLOCKER-A 三步：gemm 参数格改造 → cudarc 按 cuda-13020 节点参数读回 → 逐 launch KernelSpec + PtrUpdate 注册表） | 006 T4 | S0 | replay==eager 位级；launch 摊平 |
| S1-4 | **G5 融合核 ①fused MLP-SiLU ②fused norm+add**（006-2b ①②；≥5% 尺） | 006-2 T4 | S1-1/S1-3 | ≤4 kernel/层；≥5% 否则记录跳过 |
| S1-5 | （条件）**decode-attn FMHA 档**（G3；仅 profile 显示 attn>40% 才做） | 006-2 T2 | S1-1 | D7 + 4K 文本 100% |
| S1-6 | **双流模式①**（图内事件节点） | 006 T5 | S1-3 | 两模式一致 |
| S1-7 | **prefill 深度**：QKV 融合核 + FMHA heuristics 调准（+ 条件 vendor 档） | 006 T1 | S0-1 | prefill ≈238.8+ |
| S1-8 | **基准回归门禁**（baseline.json 5 次中位数 + CI 红 δ≤0.9× + 台面差分运行） | 006 T7 | S1-2/3/4/7 | CI 10% 回归即红。**2026-09-01 建档（纯文档/脚本/测试域，未触 crates）：** 协议卡 `bench/gate-fixture.md`（计算式、阈值 0.85×=299.8 / 0.9×=317.4、commit+构建 flags 手动填、每波重跑登记表）+ `bench/perf-gate.sh`（一行式：build → 参照检查 → `run_all.py perf_c1` → tpot 中位数 → tok/s → PASS/FAIL；`--update-baseline`）+ 无 GPU fixture 单测 `gate_fixture_verdict_cases`（三 case + 边界，数值锁于 `bench/gate-fixture.json`）。判定生效待 decode ≥299.8 tok/s（S1-9b ~285 tok/s）；benchmark-gap §4 阶梯=预期轨道（非第二门禁） |

### Stage 2 —— 服务化并发（≈4-8 周；c4→≈2s）

| ID | 功能 | spec 锚 | 依赖 | 验收 |
|---|---|---|---|---|
| S2-1 | **005 Scheduler 状态机**（Waiting→Prefill→Chunked→Decode→Done/Aborted/Preempted；req 双游标；token-budget 准入；abort tombstone；抢占=重算） | 005 | S1 基线 | c4 TTFT ≈2s；bit-identical 重跑 |
| S2-2 | **连续批 + chunked prefill + token 预算**（调度核心） | 005 | S2-1 | 批量 decode GPU 利用 |
| S2-3 | **KV 池预算**（90% 显存）+ `max-num-seqs` 语义 | 005 D2 | S2-1 | 显存常驻 20GB+、趋势平坦 |
| S2-4 | **前缀缓存接口实现**（D9 lookup/refill → P3-01 RadixCache） | 005 D9 / P3-01 | S2-3 | 共享前缀 2-10× 收益；bit-identical |

**2026-09-01**：S2-4 已开工 —— `specs/016-prefix-cache/`（draft，v1 = 页对齐 run 缓存，命中省计算；radix 前端 ‖ 引擎 prefill 偏移 ‖ executor 复制钩子；D1 run 地址/层步长、D2 refill 并入释放守卫、D3 位级论证见 plan.md）。Wave A agent 运行中；验收表在 bench/notes.md §P3-01。

**2026-09-01 记录**：S2-1/S2-2 真机验收全项通过（RTX 5090 Laptop，Qwen3-0.6B，
`REINFER_SCHEDULER=on`、`--max-num-seqs 20`）：c4（20 并发）TTFT p50 **1.17 s**
（vLLM 0.28 参考 1.8 s，**领先 34%**，S2-1 门槛"≈2 s"达成）；同 seed/temp=0 双跑输出
byte-identical；abort 隔离 9/9 幸存者 == 基线；on/off 单请求文本一致且
completion_tokens == max_tokens（48/48）；c1 TTFT p50 68 ms。验收中修复两个阻断
bug（页口径换算 `serve.rs::sched_kv_pages`；spawn 握手死锁 `SchedHandle::spawn`）
——详见 `bench/notes.md` S2-D 节。已知：单请求 TTFT（68 ms vs 9 ms 参考）受 CPU
采样 readback 上限（GPU sampler 波在后）；VRAM 池 20.6 GB 按 0.9 预算精确闭环。
S2-3（steady 20 GB+ 平稳）与 S2-4（前缀缓存）尚未启动。

### Stage 3 —— API/功能对齐与成熟度（长尾；两条关键）

| ID | 功能 | spec 锚 | 依赖 | 验收 |
|---|---|---|---|---|
| S3-1 | **n>1 多候选 + stop 生效**（T7 差项） | 005（服务） | S2 基线 | T7 8/8 |
| S3-2 | penalties/logit_bias 服务面（005 D5 全链接 GPU sampler） | 005 | S0 | API 兼容 + 数值记录 |
| S3-3 | 投机解码 / grammar（llguidance） | P3 | — | 95% 测试 |
| S3-4 | FP8 / KV offload / PD 分离（lightllm 协议） | P4 | — | 记录 |
| S3-5 | 多模型验证（GLM/其他；台面已就绪） | 013/bench | — | 每模型矩阵绿 |

## 依赖图（并行波次）

```
Wave 0（并行×3）      S0-1 ‖ S0-2 ‖ S0-3
Wave 1（并行×2）      A=[S1-1→S1-2] ‖ B=[S1-3 graph 侧（graph.rs+cudarc+kernels）]
Wave 2                 C=[S1-3 引擎接线] ‖（此后 S1-6）
Wave 3（并行×2）      D=[S1-4] ‖ E=[S1-7]（engine.rs 不同区段；hunk 报告）
Wave 4                 S1-6 ‖ S1-8 门禁
Wave 5（串行小件）    S2-1 → S2-2 → S3-1/S3-2（服务面）
Wave 6（并行）        S2-3 ‖ S2-4
Wave 7                 S3-3/S3-4/S3-5（长尾）
Wave 8                 `run_all.py --engine both` 全矩阵验收 + 报告刷新
```

## 完成门禁（复测清单）

T1=100% · T2 eos stop · T7=8/8 · decode ≥299.8 tok/s · prefill ≥238.8 ·
c4 ≈2s · 稳态 20GB+ flat · 多模型绿。每波结束跑一次
`run_all.py --engine both --suite perf_c1,perf_prefill`（5 分钟）增量验证。

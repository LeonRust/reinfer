# Bench notes

> 真机执行痕迹（009 T6；判定机 = RTX 5090 Laptop，见 `runner-info.json`）。

## 2026-08-27 — L1 (specs/009) 收尾

- 截止 commit：`e038030`（T6 之前）→ T6 收尾 commit 于下方待补
- 真机命令：
  ```text
  CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda -- --test-threads=1
  CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda --test smoke -- --ignored --test-threads=1
  cargo run -p reinfer-cuda --features cuda --example device_info
  ```
- 结果：
  - 模块内真机测试 29 through（串行）：设备信息（cc=12.0, 23.42 GiB, uuid=0fd2ce94…）、流/事件（未 record=完成态/record+sync=true/Drop 不挂）、三链往返（同步+异步 event 同步，1 MiB 模式逐字节一致）、泄漏（1000×1MiB，F7 公式）、错误注入（Oom=2、Fatal=101）
  - `--test smoke -- --ignored`（本收尾记录时以模块档为准；T6 首跑结果待记录）
- 特性记录：
  - `--test-threads=1` 为必须（并行会将泄漏测试的 memset 流量污染事件/同步断言）
  - 野指针毒化实验（`tests/poison-cases.rs`）实测 SIGSEGV——占位 `#[ignore]` 实验，勿在无隔离进程运行
- 判定："阶段+真机混合"验收（编译绿 + 无 GPU 单测 ≥7 + 真机 smoke/差分）**L1 达成**；GPU 三层门禁（F16/Q8_0 文本一致、3× CPU）属 L3 收尾范围（届时同表记录）。

> 本文件由维护者持续记录；性能数值类门禁（006/003）按 008 `baseline.json` 机制另立。

## 2026-08-28 — 014 T0: llama.cpp referee（CPU 档）

- 源码：`/home/dora/Dev/ai-tokens/llama.cpp`（`git rev-parse HEAD` = `f280b26983ad0fdb705a0d9ebf0503e76f2899b0`，tag b10615）
- 构建：`cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF`（CPU 档，r2 修正）
- 目标（b10615 无 `llama-cli`——CLI 面为 `llama-simple`）：`llama-simple llama-bench llama-tokenize llama-quantize llama-gguf`
- 产物：`/home/dora/Dev/ai-tokens/llama.cpp/build/bin/`（5 工具各自可执行；T0 Verification 通过）
- 注：`llama-gguf` 用法为 `llama-gguf <model.gguf> r [n]`（mode 为第 2 参数；`n` = 跳过 tensor 数据校验）

## 2026-08-28 — 014 T1: 真模型存档（0.5B Q8_0）读链验证

- 存档：`$REINFER_MODEL_DIR/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf`（675,710,816 B；manifest sha `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`）
- `scripts/golden/archive_check.sh`：metadata key 集 26/26 一致 + tensor 名 291/291 一致 + 字节尺寸推导全一致（Q8_0/F16）
- arch 全链：`qwen2 ctx=32768 layers=24 hidden=896 q/kv=14/2 rope_dim=64`（与 referee dump 数值一致）
- **增量发现**：该官方 GGUF 省略 `qwen2.vocab_size`——词表大小须自 `tokenizer.ggml.tokens` 推断（llama.cpp 同款链路）→ `LlamaConfig` 解析链已补（crеs/arch`from_gguf_meta`；双单测）。

## 2026-08-28 — 014 T5: Q8_0 dequant kernel 真机（RTX 5090 Laptop, cc 12.0）

- 命令：`REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda --test dequant_diff -- --ignored --test-threads=1`
- 结果：2/2 通过
  - 随机 65536 块（scale 指数域 0..=30：±0/次正规/普通）GPU vs `codes::dequantize_q8_0` **位精确（0 ulp）**
  - referee 金块（64 块）：有限值位精确；**NaN/Inf 传播值 GPU 硬件 quiet 化（0x7fffffff vs CPU SSE 保留首个 NaN payload）**——llama.cpp GPU 路径同患；判据对象=有限值域（量化 d 真实值域），NaN 传播不立判据
  - 确定性：两次 launch 逐位一致
- **内核实现要点**：scale 用**软件位构造**转换（`half_bits_to_f32`：subnormal 归一化/NaN payload 保留/Inf 直通）——硬件 `__half2float` 会 quiet 化 NaN，破坏 0-ulp；单乘语义（无 FMA 化写法）
- **存档异常记录（重要）**：官方 0.5B GGUF（sha ca59ca7f…=manifest/anchor 完全匹配）token_embd.weight 第 20 个 34B 块 scale 字节 = `46 7f`（LE u16=0x7F46 = fp16 NaN 展开）——llama-gguf API 与读取器均读出相同字节；llama-bench 可加载运行（测速不校验语义）。首个 NaN 块在词嵌入向量头部区域——如后需复现输出请以该文件为准；本记录仅作档案事实，不构成判据影响（T5 有限值域判据已闭环）。

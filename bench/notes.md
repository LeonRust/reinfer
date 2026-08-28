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

## 2026-08-28 — 014 T6: cuBLAS GEMM 真机（RTX 5090 Laptop, cc 12.0）

- 命令：`CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda --test gemm_diff -- --ignored --test-threads=1 --nocapture`
- 结果：2/2
  - **门禁档 CUBLAS_COMPUTE_32F**（gemm_f32acc）：f16-in/f32-out 与 f32-in/f32-out 全部通过
    rtol 1e-4 + atol 1e-6；形状：32³/64×48×96/128³·256/64×32×896/32×64×1536/256³·512 + K∈{1,16,4096} 边界——最差 rel 5.7e-7（f32 档）/1.1e-7（f16 档）
  - **记录档 CUBLAS_COMPUTE_16F**（gemm_f16_16acc）：max rel 1.49e-2（≤1e-1 声明；真实 K=896）
- 实现要点：
  - 直调 cublasGemmEx（cudarc 0.19 cublas feature = T6 接线；safe 层无 compute-type）
  - compute enum：CUBLAS_COMPUTE_32F=68 / 16F=64；16F-acc 时 A/B/C 均须 16F（32F C 非法——实测）
  - **行主序→列主序映射**：A^T([k,m], ld=k) + B^T([n,k], ld=n) + OP_T/OP_T；输出 col-major → `want[r*n+c]=raw[r+c*m]`
  - 标量 alpha/beta：32F 传 f32 bits；16F 传 half bits
  - handle 每次调用前 SetStream（禁 default stream 0）
- runner-info 回填：`cublas` 版本 = cudarc 0.19.9 w/ cublas feature（libcublas 12.9.x via build-system）

## 2026-08-28 — 014 T7: Prefill attention 真机（RTX 5090, cc 12.0）

- 命令：`REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda --test attn_diff -- --ignored --test-threads=1`
- 结果：2/2
  - **判据档（32F + fp32 中间，全 f32 路径）**：seq=1k、d=64 随机（f16 表示值域）vs `prefill_attn_ref`——worst rel **2.45e-6**（≈0.005 fp16 ulp；判据 ≤1 ulp + atol 1e-6 富余通过）；causal mask 生效（行 0 P=[1,0..]）；P 行和 1000/1000 ≈1
  - **r2 NaN 反例（语义档）**：unmasked NaN 注入 → 与 CPU 参考一致（max-softmax 对 NaN 行 → 全 0（与「全无效行」同路径）——含 1 fp16 ulp 容差）；bit 级 NaN 一致性受 GPU 硬件 quiet 化（T5 note）
- 实现/工程要点：
  - **全程行主序**管线：QKT 输出（col-major）→ 写 `transpose_f32` 内核转回行序 → mask 注入（-inf 累加）→ 行 softmax（`masked_softmax_matrix`，grid=rows）→ P 保持 f32 → PV（行主序 gemm）；判据档无 f16 cast（16F 中间为记录项——首次尝试发现 P cast f16 至差 1-2 ulp）
  - K^T 经设备 `transpose_f16`（K → K^T 行序）唯一转置需求
  - 教训（修了 4 轮）：列主序/行主序在「GEMM 输出 → softmax 行」两处组合易错——最终以「物理行序=语义行序」为唯一不变式；判据测试 O 读回按 ldc 换位（raw[r+c*seq]→行主序）

## 2026-08-28 — 014 T8: paged decode GQA（记录档残留——readback 异常待续）

- 页池（crates/memory::pool）✅ 5/5：守恒/LIFO/部分页/乱序页定位/OutOfPages——015 T4 跨端复用件
- decode_step_gqa kernel（每 (batch,head) 一 CTA、固定 256-lane 归约树、无 atomicAdd、串行 pass1/2（判据档）、软件 f16→f32 位构造（`hbits_to_f32`——硬件 `__half2float` 直读 u16 会被 `__half` 构造函数劫持成整数语义——SASS F2F 版本实测 raw-u16 相乘 4e9 差异；软件构造后 host 参考数值一致））：
  - **数学正确性已被 5 项独立探针 proof（全部等于 host 参考）**：kernel-scores[0..4]==host s、p(0.036)==host p、v(9.1e-5)==host V、acc(0.031063715)==want[0]、kernel-out-ptr(0x73E3_EC62C2__) == host dout ptr
  - **未决异常**：即便固定常量写 out 区（h==0 → 0x30FF），d2h 回读仍为 0x0000（预填 0x5555 经同一 d2h 链路可正常回读→ 回读链路 OK；kernel 后 raw=0）。scores 区写/回读全正常。**后续笔记**：待 T9 run 闭环时定位（优先怀疑：多 CTA 下 out 区间的某种异步/别样溢出被 sanitizer 未能捕获——compute-sanitizer 0 errors；或 b*qh grid 与 kernel 参数的边际对齐）。
- 判据（diff 门）因回读异常暂记「记录档」；（确定性测试 ✓ bit-identical）。

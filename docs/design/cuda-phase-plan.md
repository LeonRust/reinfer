# CUDA Phase 1 plan (aligned 2026-08-27)

> 与所有者对齐后的执行计划。决策记录见文末「Decision log」。预算：L1→L2→L3 连续推进，勿跳级。

## 环境基线（本机 = 判定机）

| 项 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 Laptop GPU, 24463 MiB, **compute 12.0 (sm_120)**, UUID GPU-0fd2ce94-… |
| Driver / CUDA | 595.84 / 13.2（`/dev/nvidia*` 健全，非 WSL/容器） |
| Toolkit (系统) | cuda-12.6/12.8/12.9/13.0/13.2（cudarc = from-build-system → 13.2） |
| 参考 | ai-tokens/llama.cpp（本地源码，构建 `LLAMA_CUDA=ON` 作 referee；commit f280b2698 锁） |
| 权重 | 待生成：Qwen2.5-1.5B-Instruct **从 ModelScope（魔搭）下载**（所有者指定）→ convert → GGUF Q8_0 + F16（~1.2GB/2.6GB，`~/models/`） |

## 阶段与验收（阶段口径 = 编译绿 + 无GPU单测绿 + 真机 smoke/差分；L3 收尾 = 003 三层门禁）

### L1 设备基座 —— 已展开为 `specs/009-cuda-runtime-base`（功能与验收以 009 spec 为准；003 T2 为指针；评审见 docs/design/review-cuda-l1-2026-08-27.md）
> ✅ 完成（2026-08-27）：T0-T6 全部落地；真机 smoke 入库（`tests/smoke.rs`）；记录见 `bench/notes.md`；"阶段+真机混合"验收达成。
功能：`CudaContext`（设备发现/算力/版本）、`Stream`、`Event`、`DeviceBuffer(Send)`、`HostBuffer`、memcpy（D2H/H2D/D2D），错误走 T3 `map_err`。
验收：真机 smoke —— 设备列表/alloc+copy 往返/事件同步；无 GPU 单测（构造/Debug）。
提交：`feat(cuda): add device, stream and buffer wrappers`。

### L2 算子层（003 T4/T5）—— **✅ 已完成（2026-08-27，锚 specs/012。真实提交见 git log，非计划清单）**
功能：JitCache v1（键=嵌入内容+flags 保序+toolchain realpath+triple；temp+rename/meta 提交点；跨进程锁+双检+重建一次；`REINFER_CUDA_ARCH` 预烘焙；实测梯度 sm120a≥12.8）+ 内核 `rms_norm/RoPE/masked-softmax` + vec_add 链路闭环 + sampler host 管线 + KernelProvider 选择链（D0）。
验收：真机 6/6 smoke 绿（差分 D7 容差、bit-exact 确定性、命中 <50ms、跨进程单次编译）——`REINFER_CUDA_NVCC=/usr/local/cuda-12.8/bin/nvcc REINFER_JIT_CACHE=... cargo test -p reinfer-cuda --features cuda --test jit_smoke -- --ignored --test-threads=1`。
注：启动阻塞式 prewarm（003 T4 原文）**延至 L3 引擎启动切片**（012 r1 R3/R8）——本切片为离线预烘焙 + 懒构建。

### L3 单请求闭环（003 T9-T12 + 004 tokenizer）
功能：GGUF/arch(001-003) → F16/Q8_0 dequant → cuBLAS GEMM（16F-acc 对齐 llama.cpp）→ paged decode attention（先正确后优化）→ sampler（greedy+gumbel 纯函数）→ `reinfer cli --backend cuda` 流式输出。
前置：004 tokenizer（该切片完成后再接）；权重转换（装 transformers→convert_hf_to_gguf.py）；llama.cpp CUDA 编译。
验收（003 三层门禁，本机 RTX 5090）：F16 同 compute-type → 20 prompts 逐 token 100%；Q8_0 greedy ≥99.9% + logits ≤1e-2（记录）；decode ≥3× llama.cpp CPU。对照协议：llama-bench 参数、KV f16、graph on；全部写入 `bench/notes.md`。

## 003 spec 修订记录（本计划带来的）

- sm_120 为判定档：plan D2 gencode 表补 `sm120a ≥13.0`（本机 toolkit 13.2 ✅）；rtx5090 作为基准机记录（GPU UUID + driver + cuBLAS 版本 → parity/baseline）。
- 本机即判定机（008 中 "gpu-runner" 概念保持，本机为 local-fixed 档；远程 runner 上线后同协议迁移）。

## Decision log

- 2026-08-27：L1→L2→L3 连续推进；权重 = **从 ModelScope（魔搭）下载** Qwen2.5-1.5B 后转换（模型一律 ModelScope，不用 HuggingFace）；验收 = 阶段+真机混合；目标机型 sm_120。
- 先前误报"本机无 GPU"→ 更正为 RTX 5090 Laptop 24GB（驱动正常）。

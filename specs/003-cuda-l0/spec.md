# Spec: CUDA L0 — GPU inference loop for a single request

> Status: approved (review 2026-08-25, see docs/design/review-2026-08-25.md) · Parent: specs/000-project-mvp (P1 first slice)
> 修订记录：spec 层去厂商化；数值门禁三层化；容差体系统一；GPU sampler 核纳入范围；引用修正（006 为 perf、005 为 serving、004 为 tokenizer）。
> 环境基线（2026-08-27 对齐，见 docs/design/cuda-phase-plan.md）：判定机 = 本机 RTX 5090 Laptop 24GB sm_120（driver 595.84/CUDA 13.2）；llama.cpp 参考本地构建。

## Problem Statement

reinfer 尚无 GPU 路径。P1 需在 NVIDIA 上证明引擎假设：**GGUF → GPU kernels → 流式 token**（单请求），含 paged-KV decode 与三档 KernelProvider 骨架。本切片交付 GPU 基座与 `reinfer cli --backend cuda` 单请求闭环。

## Success Metrics

- **Kernel 数值正确**：每个 kernel 与 CPU 参考逐元素一致，容差按"输入/累积/输出 dtype、绝对/相对"三元组判定（见 plan.md §数值容差表），**并且该一致性不推导为 token 级一致**，两者独立判定。
- **文本对齐（三层门禁）**，参考 llama.cpp（commit f280b2698 固定）：
  - F16：与 llama.cpp 强制同 compute type（其侧 `GGML_CUDA_CUBLAS_COMPUTE_TYPE=16F`）时 → 20 条 golden prompt 逐 token 100%；无法满足时回退 000 口径：累积 logits drift ≤ 1e-4；
  - Q8_0：**贪心 token 一致率 ≥ 99.9%**（20 prompts × ≥200 tokens）+ logits 相对漂移 ≤ 1e-2（记录项；恢复 100% 仅当复刻 mmq 路径，RFC 候选）；
  - 实现与 referee 的算法差异声明在 `bench/notes.md`（F16 累积方式、Q8_0 走 dequant+GEMM vs referee mmq、prefill 两段 GEMM vs referee flash-attn）。
- **性能门禁（CPU 档）**：decode ≥ **3× llama.cpp CPU**（协议锁定见 005/基准协议：`llama-bench -ngl 0 -t <物理核数> -b 1 -n 512`，预热≥3 取中位数；判定机=gpu-runner，开发机仅记录）。
- **内存**：单请求 8B Q8_0 显存 ≤ weights + KV + 0.5×weights；两轮运行零页泄漏（页计数判定，见 005 池基线定义）。
- **工程**：无 GPU/无 toolkit CI 全绿（CUDA 代码 cfg 隔离；GPU 测试 `#[ignore]` + gpu-runner `--include-ignored`，矩阵入 specs/008-ci-infra）。

## User Stories

1. 作为用户：`reinfer cli --backend cuda --model model.gguf "prompt"` 在 NVIDIA GPU 上流式输出。
2. 作为后端作者：我实现并注册 `KernelProvider`s，unsafe 只出现在 `crates/cuda`/`crates/jit`（经 cudarc）。
3. 作为维护者：GPU runner 上一键重跑差分 + 文本对齐，无 GPU CI 覆盖可测部分（错误映射、JitCache 键/锁、选择器回退链）。

## Acceptance Criteria

- [ ] 默认构建零 CUDA 依赖（feature `cuda` 门控；无 toolkit 也能 `cargo check --workspace`）
- [ ] `crates/cuda`：L1 运行时 wrappers（设备/流/事件/缓冲/拷贝）——**功能与验收以 specs/009 为准**（本 spec 不再重复列举）；`cudaError→LaunchError` 白名单（内存→Oom/context 类→Driver/其余→Fatal，fail-closed；锚=002/plan 错误映射表）
- [ ] JitCache v1（细节见 plan/tasks）：源码经运行时编译，缓存键含源码+头闭包+gencode/flags+nvcc 版本，原子写入；nvcc 缺失→专用错误，不静默降级；`REINFER_CUDA_ARCH` 支持无 GPU 预烘焙
- [ ] kernel 集：RMSNorm、RoPE、masked softmax、Q8_0/F16 解量化、GQA paged decode attention、**sampler（softmax+gumbel+argmax，纯函数 RNG）**——每 kernel 有 CPU 参考 + 差分测试
- [ ] GEMM 走厂商加速库；prefill 注意力采用分块/低内存路径（厂商 FMHA 为 006，本切片可为两段 GEMM，须在 notes 声明与 referee 的差异）
- [ ] paged KV 池（策略在 `crates/memory`，CUDA 实现 `MemOps`；块 16/32；refcount；泄漏检测）
- [ ] `reinfer cli --backend cuda` 流式输出；golden/parity 矩阵（specs/000/parity.md）跑通
- [ ] CI：无 GPU 档绿 + GPU 档（差分、parity、3× CPU 门禁）跑通并入 008-ci-infra 矩阵

## Non-Goals

- 多请求并发/HTTP（specs/005）；TP/PP/CP、Radix、投机、grammar（P3）
- vendor FMHA（CUTLASS/cubin）与 CUDA Graph（specs/006）；投机后 renormalize 等高级采样
- W4/W8 编码器（仅 Q8_0/F16 解码）；Mamba/MLA/MoE（P4）；昇腾（specs/002，暂缓）
- 复刻 llama.cpp mmq 路径（RFC 候选）

## Constraints

- 引擎自身内核源码（CUDA C++）；运行时编译器为系统 nvcc（toolkit 仅 GPU 机）；仅 cudarc 绑定（driver/runtime/cublas）
- 禁止 torch（§1.3）——包括"以 torch 作性能对照基线"
- 数值裁判 = llama.cpp（commit/参数锁定，见 parity.md 与基准协议）；每 kernel 的 CPU 参考保持同数学（f32 累积）
- 内核资产路径统一 `crates/cuda/kernels/`（变更经 PR 审查）

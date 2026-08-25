# Spec: CUDA performance upgrade — FA3/CUTLASS vendor tier + CUDA graphs

> Status: proposal · Owner: maintainers · Created: 2026-08-25 · Parent: specs/003 (correctness base)

## Problem Statement

003 交付了正确性（功能关然）——但 GPU 性能未达标：prefill 是两段 GEMM（内存峰值、带宽低效），decode 是手写 naive，缺少 graph 捕获。006 的目标是**基于 vendor 档的两种关键能力**把吞吐对齐到 85% llama.cpp-CUDA：① FMHA prefill（CUTLASS/FA3 cubin 装载）② CUDA Graph 桶化捕获与重放。

## Success Metrics

- **性能**: decode ≥ **85% llama.cpp CUDA**（同 GPU 同模型同 batch）；prefill（4K seq, fp16）≥ 70% llama.cpp CUDA
- **正确性**: 与 003 差分一致（文本 100% 对齐；kernel level ≤1e-5）；图形捕获后单请求/多请求结果与 eager 相同（≥99.9% 灰度——bucket miss 时自动回退 eager）
- **内存**: 复用 003 页池；graph buffer 池化固定（捕获 batch 桶 16/32/64 时峰值 ≤ 基线 + 8%）
- **工程**: 无 GPU CI 不变；加 `--perf` 回归基准记录（`bench/notes.md`）与回归门禁（1 个 decile 掉速即阻）。

## User Stories

1. 作为引擎作者：`KernelProvider` 档1(Vendor cubin) 支持 sm90+ 的 FMHA（CUTLASS gen + FA3 via flashinfer-cubin 下载协议）；装载走 `crates/jit`（同一 JitCache，键=sha256+device）。
2. 作为服务者：`--features cuda` 单卡吞吐对齐 llama.cpp-CUDA 见 `bench/notes.md`。

## Acceptance Criteria

- [ ] FMHA prefill kernel 替换两段 GEMM（`kernels/fmha/*.cu` 经 JitCache nvcc 编译；sm90 warp-mem layout，sm80 fallback 保留 003 路径）
- [ ] 可选档1：FA3 cubin 装载器（cudarc + cubin sha256 + version-check 脚本挂 CI；下载失败自动回退 CUTLASS）
- [ ] CUDA Graph：`capture(bucket)` 池（bs×seq 桶 8/16/32/64；内存复用 `cudaGraph` 管理）；`replay` + event 同步；桶 miss → eager 回退
- [ ] 双流重叠：`attn` 与 `FFN` 流分离（`cudaStream` 双流 + 无界队）
- [ ] `bench/notes.md` 记录：decode/prefill 相对 llama.cpp-CUDA 的比值、图捕获收益（~18+%，对 ferrum 基线）、`--perf` 回归列盘

## Non-Goals

- Flash MoE/MLA（P4）；warp specialization 级优化手工调（留 007 RFC）；torch.compile/IR 编译；多卡 CC 调度

## Constraints

- llama.cpp-CUDA 作为 referee（同 003）；vendor cubin 下载协议复用 flashinfer-cubin 思想（但不做 Python 依赖；脚本式即可）
- 所有 vendor 挂载失败必须有明确 fallback 链（Vendor→CUTLASS→003 code）
- 与原 KernelProvider 三档选择逻辑一致：同一 `OpConfig` 决策

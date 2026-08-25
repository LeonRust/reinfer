# Spec: CUDA L0 — GPU inference loop for a single request

> Status: proposal · Owner: maintainers · Created: 2026-08-25
> Parent: specs/000-project-mvp (P1 first slice) · Upstream facts: CANN track parked per owner decision; NVIDIA first.

## Problem Statement

reinfer has no GPU path yet. P1 needs to prove the full engine hypothesis on NVIDIA: **GGUF → GPU kernels → streamed tokens**, for one request, with paged-KV decode — without Python, without torch, with the three-tier kernel design in place. This slice delivers the GPU base (context, streams, buffers, JIT kernels, cuBLAS GEMM, paged decode attention) and a single-request `reinfer cli --backend cuda` run.

## Success Metrics

- **Numeric parity**: every CUDA kernel matches its CPU reference kernel element-wise (f32 accumulation) within 1e-5, verified by host-side differential tests.
- **Text parity**: Llama-8B Q8_0 and F16, 20 golden prompts — output tokens identical (100%) to llama.cpp (same weights, greedy, same machine, same batch config).
- **Performance gate**: decode tok/s ≥ **3× llama.cpp CPU** on the same box (gate for P1); stretch: ≥85% of llama.cpp CUDA after the P1.5 FA3 upgrade (spec 005).
- **Memory**: 8B Q8_0 single request: VRAM ≤ weights + KV + 2×max workspace; two consecutive runs leak 0 pages (pool parity check).
- **Engineering**: no-GPU CI is still green (CUDA code fully cfg-gated); GPU runner runs the differential + parity gates.

## User Stories

1. As a user, `reinfer cli --backend cuda --model model.gguf "prompt"` streams tokens on an NVIDIA GPU.
2. As a backend author, I implement `KernelProvider`s and register them — I never see `unsafe` outside `crates/cuda` / `crates/jit`.
3. As a maintainer, I can run overnight parity/differential tests on a GPU runner without touching the engine.

## Acceptance Criteria

- [ ] Workspace wires `cudarc` (optional, feature `cuda`); default builds without any CUDA toolkit installed
- [ ] `crates/cuda`: `Context`, `Device`, `Stream`, `Event`, `DeviceBuffer` (+`Send`), `HostBuffer`, memcpy; `cudaError → LaunchError` whitelist mapping (memory→Oom, context-lost class→Driver, unknown→Fatal, fail-closed)
- [ ] `crates/jit`: JitCache v1 (nvcc + source hash + FileLock + on-disk cubin cache under `~/.cache/reinfer/jit`, prewarm on startup)
- [ ] Kernels (CUDA C++ via JitCache): RMSNorm, RoPE, softmax(masked), Q8_0/F16 dequant, GQA paged decode attention; host-side CPU reference counterparts exist and differential-tests pass
- [ ] GEMM via cuBLAS FFI (cudarc::cublas) fp16/bf16/f32; prefill attention as two GEMMs + epilogue split (non-flash, accepted for slice)
- [ ] Paged KV pool (policy in `crates/memory` — `MemOps` trait; CUDA impl in `crates/cuda`) with block size 16/32, page table, alloc/free; leak detection
- [ ] `reinfer cli --backend cuda` streams tokens; 20-prompt parity test harness in `bench/`
- [ ] CI: (a) no-GPU jobs green; (b) GPU job (`gpu-runner` label) runs differential + parity

## Non-Goals

- Multi-request concurrency / HTTP serving (spec 004); TP/PP/CP/DP; radix cache; speculative decode; grammar (P3)
- FA3/CUTLASS vendor cubin adoption (spec 005, P1.5); CUDA graph capture (spec 005)
- W4A16 encoders (dequant Q8_0/F16 only); Mamba/MLA/MoE (P4)
- Ascend track (spec 002 — parked by owner decision until NVIDIA slice lands)

## Constraints

- Only cudarc (driver/runtime/cublas bindings) + system nvcc via JitCache; torch forbidden (constitution §1.3)
- CUDA toolkit ≥ 12.4 present only on GPU machines; all CUDA code `#[cfg(feature = "cuda")]`-gated
- Kernel sources are engine-owned assets (`crates/cuda/kernels/*.cu`, `include_str!`); SIMD/CPU reference keeps exact same math per kernel (f32 accumulate)
- Numeric referee = llama.cpp (same weights, greedy, deterministic) per specs/000

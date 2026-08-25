# Tasks: CUDA L0 — GPU inference loop

> Derived from specs/003-cuda-l0/plan.md · each independently verifiable

## Task 1: CUDA wiring (workspace)

- Add `cudarc = "0.19"` optional workspace dep; `crates/cuda` dep-gated by `[features] cuda`; `bin/reinfer` forwards `cuda` feature; `.gitignore` jit cache
- Verification: `cargo check --workspace` (no GPU/toolkit) green; `cargo check --features cuda` on GPU machine green

## Task 2: Device/Stream/Event/Buffer wrappers (crates/cuda)

- `CudaContext::init` (device count/set per thread), `CudaStream`, `CudaEvent`, `CudaBuffer` (**Send** + as_ptr), `HostBuffer` (pinned), memcpy paths
- Verification: unit test with null-traits on CPU; GPU smoke on runner (alloc/free/copy roundtrip)

## Task 3: Error mapping (whitelist)

- `map_error(cudaError_t) -> LaunchError` consistent with D5; `# unit tests` for each class + unknown→Fatal
- Verification: `cargo test -p reinfer-cuda` (no GPU needed — pure mapping table)

## Task 4: JitCache v1 (crates/jit)

- `find_nvcc` (PATH → CUDA_HOME), `#gencode` per `capability()` (sm80/90/100 family fallback), source-hash key, cross-process `FileLock`, cubin cache in `~/.cache/reinfer/jit`, background prewarm
- Verification: tiny `.cu` (addk) compile-cache-recompile; run twice → second run skips nvcc (logged, timing asserted <50ms)

## Task 5: norm/rope/softmax kernels + CPU references

- RMSNorm (fused scale+eps), RoPE (fp32 acc, per-dtype), masked softmax (chunked smem, online-max)
- Verification: host differential vs `crates/cpu` reference, 1e-5, random shapes incl. head_dim 64/128; GPU runner executes in CI GPU job

## Task 6: dequant kernels (Q8_0 / F16)

- Q8_0 block dequant (port from llama.cpp math, block 256), F16→fp16 pass-through (with fp32 accumulate in consumer)
- Verification: differential vs CPU reference (golden blocks from 001 golden GGUF), ≤1 ulp

## Task 7: cuBLAS GEMM wrapper

- `gemm_f16/f32(cublas)` via cudarc; ndim 2/3 (payloads), handle caching, row-major checks
- Verification: differential vs CPU matmul (100 shapes) 1e-5; perf sanity ≥ cublas 80% of torch baseline (marked, not gated)

## Task 8: Prefill attention (GEMM two-phases)

- QK^T (GEMM) + softmax + PV (GEMM), split-k epilogue v1, KV layout NHD
- Verification: output differential vs CPU attn reference at seq 1k (1e-4 fp16 tol)

## Task 9: Paged decode attention (GQA)

- Kernel: block 16/32 KV paging, GQA group mapping (group = head/gqa), smem staging, causal single-token query; `MemOps` block allocator in `crates/memory` + CUDA impl
- Verification: differential vs dense CPU reference (random page tables, batched seq=1, n=1..64); leak check run (alloc/free 1M pages, pool size stable)

## Task 10: Engine integration + `cli --backend cuda`

- Wire ModelRunner path: GGUF(001) → arch(001) → CUDA kernels → continuous KV → streamed decode; `cli` uses `--backend cuda`
- Verification: `reinfer cli --backend cuda --model Llama-8B-Q8_0.gguf "prompt"` streams; 20 golden prompts vs llama.cpp = 100% token match (F16 + Q8_0); decode tok/s & VRAM recorded in `bench/notes.md`

## Task 11: CI gates

- no-GPU jobs: lint/check/test (with `--no-default-features` CUDA exclusions); GPU job: differential + parity (label `gpu-runner`, nightly + PR-label trigger)
- Verification: both jobs green in CI runs ✓ documented

---

Completion gate: Tasks 1–11 accepted; text parity 100%; 3× CPU gate recorded in `bench/notes.md`; reviewer approval. Next slice spec 004 (scheduler + HTTP serving).

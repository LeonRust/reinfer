# Plan: CUDA L0 — GPU inference loop

> Derived from specs/003-cuda-l0/spec.md

## Architecture Decisions

- **D1 Libs**: `cudarc` for driver/runtime/cublas — the only GPU dependency (feature `cuda`); FFI surface stays inside `crates/cuda` (narrow, one crate).
- **D2 Kernel ownership**: all hand-written kernel sources live in `crates/cuda/kernels/*.cu` as crate assets, compiled at runtime by `crates/jit` (nvcc, source-hash keyed, FileLock, on-disk cache `~/.cache/reinfer/jit`) — mirrors FlashInfer `JitSpecNvcc` protocol (tri_phase try_load/build/load) but with our own hash/locking. No bindgen, no build-time CUDA for non-GPU CI.
- **D3 GEMM**: cuBLAS via `cudarc::cublas` (fp16/bf16/f32). Prefill attention = two GEMMs + split-k epilogue (non-flash acceptable for this slice; FA3 upgrade in 005).
- **D4 Paged KV**: policy lives in `crates/memory` (block allocator, refcount, free list — backend-agnostic) with `MemOps` trait; `crates/cuda` implements `MemOps` (VMM not required yet; `cudaMalloc`-sized slabs, block size 16/32).
- **D5 Error mapping**: `cudaError_t → LaunchError` by whitelist: `cudaErrorMemoryAllocation` → `Oom`; context-lost class (`cudaErrorDeviceUnavailable`, `cudaErrorNoDevice`, `cudaErrorIllegalAddress` bounded set) → `Driver`; everything unknown → `Fatal` (fail-closed, same rule as Ascend contract).
- **D6 Determinism**: every kernel independently numerical — host CPU reference counterpart per kernel; parity test at text level vs llama.cpp.

## Module Breakdown

| Module | Content |
|---|---|
| `crates/cuda/src/ctx.rs / device.rs / stream.rs / event.rs / buffer.rs` | safe wrappers (mirror of cann L0 shapes; `DeviceBuffer: Send`) |
| `crates/cuda/src/error.rs` | cudaError→LaunchError whitelist + tests |
| `crates/jit/src/nvcc.rs` | find nvcc, build cmd, `#gencode` per arch (sm80/90/100 family), hash-key, lock, cubin cache load/store |
| `crates/cuda/src/kernels/mod.rs` | kernel registry (`KernelHandle` → cubin symbol) |
| `crates/cuda/src/kernels/{norm,rope,softmax,quant,attn}.cu` | RMSNorm, RoPE, masked softmax, Q8_0/F16 decode, GQA paged decode kernel |
| `crates/cuda/src/gemm.rs` | cuBLAS wrapper (`f16/f32` matmul, no-op guard when cuda absent) |
| `crates/cuda/src/pool.rs` | `MemOps` impl: slab alloc + block table + page ops (leak counters) |
| `crates/memory/src/block.rs` | backend-agnostic block allocator policy (refcount + free list + epochs) |
| `crates/cpu/src/kernels/*.rs` | CPU reference counterparts for every GPU kernel |
| `bin/reinfer/src/cli.rs` | `--backend {cpu,cuda}` routing; `cli` subcommand streams |

## Interface Contracts (slice-local)

```rust
// crates/cuda
pub struct CudaContext;                      // cudarc CUDA init + device set per-thread
impl CudaContext { pub fn init() -> Result<Self, LaunchError>; }
pub struct CudaStream; pub struct CudaEvent;
pub struct CudaBuffer { /* Send */ }         // cudaMalloc slab / pinned host
pub fn map_error(e: cudaError_t) -> LaunchError;   // whitelist, fail-closed

// crates/kernels
pub fn launch_norm(...) -> Result<(), LaunchError>;   // RMSNorm epilogue fused
pub fn launch_rope(...) -> Result<(), LaunchError>;
pub fn launch_masked_softmax(...) -> Result<(), LaunchError>;
pub fn launch_dequant(...) -> Result<(), LaunchError>;         // Q8_0 / F16 → f16 fp32acc
pub fn launch_paged_attn_decode(...) -> Result<(), LaunchError>; // GQA, block 16/32, smem staging

// crates/memory (backend-agnostic policy)
pub trait MemOps { fn alloc(&mut self, pages: usize) -> Result<PageSpan, ...>; ... }
pub struct BlockPool;                        // refcount, free list, epoch pairs

// crates/jit
pub struct JitCache;                         // nvcc + hash + lock + ~/.cache/reinfer/jit + prewarm
impl JitCache { pub fn get_or_build(&self, key: &JitKey, src: &str) -> Result<JLib, LaunchError>; }
```

## Reference assets (资产清单 → 依据 docs/深入设计补充 §3)

- llama.cpp `ggml-quants.c` — Q8_0 dequant math → port to CUDA kernel + CPU reference
- FlashInfer `jit/core.py` — three-phase try_load/build/load + FileLock protocol → JitCache design
- vLLM block pool (`kv_cache_manager.py`, `block_pool.py`) — refcount/free-list semantics → `crates/memory::BlockPool`
- mistral.rs / ferrum-runtime — cudarc integration patterns (contexts per thread, cublas usage)
- mini-sglang `kernel/index.cu`, `store.cu` — PDL launch hints & warp-copy idioms for decode kernels

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| nvcc JIT first-call latency (seconds) | Medium | prewarm at startup (background thread); hash-cached cubin makes it one-time |
| No GPU in dev environment | Medium | CPU differential is the primary CI gate; GPU parity only on runner |
| Naive paged decode < 3× CPU gate | High | gate set relative to llama.cpp CPU initially; block-16 + smem staging first; FA3 upgrade path (005) is the committed fallback |
| cudaError class drift | Low | fail-closed whitelist + test for `unknown → Fatal` |
| Non-CUDA CI accidentally compiling CUDA path | High | `#[cfg(feature="cuda")]` on every module; no-GPU CI also runs `cargo check -p reinfer-cuda --no-default-features` |
| Device determinism vs llama.cpp (ulps → token flips) | Low | greedy + same seed; parity gate = 100% tokens for Llama family with 1e-5 kernel tolerance |

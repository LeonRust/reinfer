# Plan: CUDA L0 — GPU inference loop

> Derived from specs/003-cuda-l0/spec.md

## Architecture Decisions

- **D1 依赖**：仅 `cudarc`（driver/runtime/cublas，feature `cuda`）；FFI 面收敛于 `crates/cuda`（unsafe 宿主）。
- **D2 Tier 语义（谱系溯源；2026-08-27 r1 裁决改序并回写深入设计 §1.1）**：`Vendor`（预编译 cubin/vendor 库）> `Jit`（引擎自有 CUDA C++ 经 JitCache nvcc 编译——含本切片全部 kernel）> `Native`（直写 Rust/CubeCL 内核——CUDA 侧暂缺，保留档位）> CPU 参考。本切片选择链实为 `Jit(dense)` 单档；006 升级为 `Vendor > Jit(fmha) > Jit(dense)`。`select()` 与 TuneDb 位于 `crates/kernels`（safe 层），Provider 实现（触 FFI）位于 `crates/cuda`。**gencode 梯度（工具链实测基线，锚 specs/012 r1 R6）**：sm90a≥12.3 / sm100a≥12.8 / **sm120a≥12.8**（12.6 不支持 sm_120；12.8 产物在判定机加载+launch 位精确——原"≥13.0"为事实错误）。
- **D3 GEMM**：cuBLAS。F16 路径用 fp16 累积（与 llama.cpp `CUBLAS_COMPUTE_16F` 一致），比对时双侧强制同 compute type；Q8_0 为"dequant→fp16→GEMM"（与 referee mmq 不同的算法，作为记录差异，见 spec 三层门禁）。
- **D4 JitCache**（按评审加固）：键 = sha256(源码 + 头传递闭包(nvcc -M depfile) + gencode/flags + nvcc --version + capability)；写入 temp+rename；`cuModuleLoad` 失败→删除重建一次；prewarm = 启动**阻塞**前滚完成（不在后台与首请求并发）；按 key 粒度文件锁（双检）；锁文件放 /tmp；无 GPU CI 用 `REINFER_CUDA_ARCH` + 预烘焙 cubin 缓存。
- **D5 错误映射**：`cudaError→LaunchError` 白名单 fail-closed（表锚定 002/plan）。
- **D6 确定性**：采样 RNG 为纯函数（契约见 005：`rng(seed_i,pos,v)`）；本切片先实现 greedy + gumbel-max，种子沿用 005 定义。
- **D7 数值容差表**（唯一判据来源）：

| 输出 dtype | 判据 |
|---|---|
| fp16/bf16 | 参考先舍入到同 dtype，再比较：≤1 ulp 视为相等 |
| fp32 | rtol=1e-5, atol=1e-7（numpy allclose 语义，pr 取 max） |
| GEMM（按输入累积） | f16-accurate：rel 1e-4；bf16：rel 1e-2 或舍入后精确；f32：rel 1e-5 |

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/cuda/src/{ctx,device,stream,event,buffer,error}.rs` | 安全包装（DeviceBuffer: Send） |
| `crates/cuda/src/gemm.rs` | cuBLAS 包装（f16/f32，compute type 可选） |
| `crates/cuda/src/kernels/{norm,rope,softmax,quant,attn,sampler}.cu` | 引擎自有内核（Jit 档） |
| `crates/cuda/src/pool.rs` | `MemOps` CUDA 实现（slab+页表+泄漏计数） |
| `crates/kernels/src/tunedb.rs` | tune.json 读写 + `select_*`（safe，供全后端复用） |
| `crates/memory/src/block.rs` | 后端无关块分配器策略 |
| `crates/cpu/src/kernels/*.rs` | 每 kernel 的 CPU 参考 |
| `crates/engine`（本切片末建，005 建） | ModelRunner/Engine 宿主（forbid unsafe） |
| `bin/reinfer/src/cli.rs` | `--backend {cpu,cuda}` 路由 |

## Interface Contracts（标注 crates/cuda，非 kernels）

```rust
// crates/cuda（unsafe 宿主）
pub fn map_error(e: cudaError_t) -> LaunchError;      // 白名单 fail-closed
pub fn launch_norm(...) -> Result<(), LaunchError>;
pub fn launch_rope(...) -> Result<(), LaunchError>;
pub fn launch_masked_softmax(...) -> Result<(), LaunchError>;
pub fn launch_dequant(...) -> Result<(), LaunchError>;
pub fn launch_paged_attn_decode(...) -> Result<(), LaunchError>;   // GQA block16/32
pub fn launch_sampler(...) -> Result<(), LaunchError>;             // masked softmax + gumbel + argmax

// crates/kernels（safe：trait/注册/选择/调优）
pub trait KernelProvider { fn tier(&self) -> ProviderTier; fn launch(...); }
pub struct TuneDb; pub fn select(...);

// crates/memory（后端无关策略）
pub trait MemOps { fn alloc(&mut self, pages: usize) -> Result<PageSpan, ...>; }
```

## Reference assets（只列增量，全量见 docs/深入设计补充 §3）

- llama.cpp `ggml-quants.c` — Q8_0 数学（仅算法）；`mmq.cuh` 结构仅作 RFC 参考，本切片不复刻
- FlashInfer `jit/core.py` — 三段式 + FileLock + 双检范式
- vLLM `kv_cache_manager/block_pool` — refcount/free-list 语义
- mini-sglang `index.cu`/`store.cu` — warp-copy/PDL 惯用法
- ferrum-runtime — cudarc 多上下文每线程用法

## Risk Assessment

| Risk | Mitigation |
|---|---|
| 与 llama.cpp 算法差异导致 token 差（High，评审定级） | 三层门禁（spec）；差异声明 notes；mmq 复刻为 RFC |
| JitCache 缓存失效（head 变更/nvcc 升级） | D4 键含头闭包与 nvcc 版本；原子写 |
| prewarm 与首请求竞争 | D4 阻塞式预热 + 锁内双检 |
| nvcc 缺失/toolkit 不足 | 专用错误；按 arch 检查最低版本（sm90a≥12.3/sm100a≥12.8） |
| 无 GPU CI 空转 | `#[ignore]` 矩阵 + JitCache/TuneDb/选择器单测跑无 GPU 档；008-ci-infra |

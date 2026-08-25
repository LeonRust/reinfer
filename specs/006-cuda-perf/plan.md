# Plan: CUDA performance upgrade — vendor tier + CUDA graphs

> Derived from specs/006-cuda-perf/spec.md

## Architecture Decisions

- **D1 选择链与归属（评审 A-H1/M1）**：链 = `Vendor(cubin/dylib, curated)` > `Jit(fmha)` > `Jit(dense/003)`。**TuneDb、`select()`、`ProviderChoice` 位于 `crates/kernels`**（safe 层；trait+注册+tune.json 读写）；Provider 实现（触 FFI/cub 加载）位于 `crates/cuda`。tier 定义：Vendor=预编译二进制；Jit=引擎自有 CUDA C++ 经 JitCache 编译（003 全核 + 006 FMHA/decode核）；Native(Rust 直写)=CUDA 侧暂缺保留；CPU 参考在下游。
- **D2 FMHA 源码**：FMHA 采用**摘录式** vendored 头 + 自有 `.cu`（`.cu` 经 JitCache nvcc，gencode 按 capability：sm90a/100a；最低 CUDA：sm90a≥12.3、sm100a≥12.8、sm120a≥13.0）；cutlass/warp 相关宏（`CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED` 等）由 build 模板注入；"sm90a 失败退 sm90"**仅对 FA2 级源码成立**——含 wgmma/TMA 的源码不降级，直接回退 Jit(dense)；vendored 头目录带 `version.json`（版本 bump 即 JitCache 键失效——对源头文件 hash 之外的头闭包法补充）。
- **D3 cubin 供应链**：期望 sha256 表以 **`cubins/manifest.json` 入库**（含 arch/来源/版本）；下载仅 HTTPS 固定源；离线或校验失败 → **硬失败回退**（不产静默下载）；优先方案 = vendor cubin 随 release 产物分发（同 flashinfer 包内 cubins 思路）；校验失败 warning 入 tracing。
- **D4 Graph 池（decode-only）**：
  - 桶：bs 8..128 步 8、128..256 步 16（vLLM 实测曲线；禁幂次桶）；**prefill 不捕获**（实验性小桶需 `--cuda-graph-prefill=experimental` 显式开启）；
  - 单一共享 pool（所有桶复用），捕获内存实测记账（profile：首捕获后按图增量累计，计入预算公式）——参考 vLLM `get_global_graph_pool`/`profile_cudagraph_memory` 语义；
  - 捕获串行化：全局锁（llama.cpp 模式）；**捕获期 `--no-overlap`**；
  - ExecUpdate：仅"同形状指针刷新"；失败 → destroy + re-instantiate（cudaErrorGraphExecUpdateFailure 路径）；含事件节点或 cuBLASLt handle 的图禁用 ExecUpdate（直接重捕获）；
  - 运行期计数：graph_replay / eager_fallback / padding_ratio（每桶），告警阈值 5min eager>20%。
- **D5 双流重叠**：两模式——① 事件节点入图（llama.cpp：graph 内 record/wait）；② 非捕获期 `--no-overlap` 降级（vLLM 模式）；捕获窗口内禁止并发发射（T9 decode 主循环与捕获不得共流并发）。
- **D6 失败链**：每个 Provider `launch()` 返回 `Fallback`（选择器层概念：回退下一档，不进 LaunchError 枚举）；nvcc 缺失/toolkit 不足 → `LaunchError::Fatal` 专用消息（不静默）；`REINFER_CUDA_ARCH` 唯一影响选择（离线预烘焙）。
- **D7 门禁与基准**：基准协议 = llama-bench 参数表（decode：`-b 1 -n 512 -fa 1 -ngl 99`；prefill：`-b 2048 -ub 512`），commit f280b2698、构建 flags、GPU UUID/driver/cuBLAS 版本记录；`bench/baseline.json`（5 次中位数，`--perf` 覆写保留历史）；CI 红 = 中位数 ≤0.9× 基线。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/cuda/src/fmha/` | FMHA 源码（engines-owned）+ heuristics |
| `crates/cuda/src/decode/` | fused Q8_0 dequant-dot decode 核 |
| `crates/cuda/src/fa3/` | cubin 加载 + manifest 校验（D3） |
| `crates/cuda/src/graph.rs` | GraphPool（D4） |
| `crates/cuda/src/overlap.rs` | 双流双模式（D5） |
| `crates/kernels/src/tunedb.rs` | TuneDb/select/ProviderChoice（safe） |
| `bench/` | 协议脚本、baseline.json、notes |

## Interface Contracts（选择器在 kernels；loader 在 cuda）

```rust
// crates/kernels (safe)
pub enum ProviderChoice { Vendor(JitKey), JitFmha(JitKey), JitDense(JitKey) }
pub fn select_fmha(cfg: &OpConfig, db: &TuneDb) -> ProviderChoice;   // fallback 链决定
pub struct TuneDb { /* load/save tune.json, atomic write */ }

// crates/cuda
pub struct Fa3Provider;   // manifest 校验 + 回退链
pub struct GraphPool;     // get_or_capture(bucket, tpl) / replay / release + counters
pub fn sample_decode_quant_dot(...) -> Result<(), LaunchError>;   // T6 核心
```

## Risk Assessment

| Risk | Mitigation |
|---|---|
| 摘录头版本漂移 → 静默旧语义 | version.json bump + JitCache 头闭包 hash（D2/D4 键） |
| ExecUpdate 命中性静默错 | 仅 ptr 刷新 + re-instantiate 回退 + 计数 |
| 捕获内存爆预算 | decode-only + 单池 + profile 记账入预算 |
| run 期桶 miss 飚高静默 | 计数器 + 5min eager>20% 告警（notes/监控） |
| CUTLASS 头体积 | 摘录 FMHA 必需头（不 vendored 全量），+体积实测记录 |

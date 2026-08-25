# Plan: CUDA performance upgrade — vendor tier + CUDA graphs

> Derived from specs/006-cuda-perf/spec.md

## Architecture Decisions

- **D1 三档落地顺序**：002 定的 Vendor > Native > Jit 在此落地为：`Native(003 code)` → `Vendor-CUTLASS(FMHA)` → `Vendor-cubin(FA3, optional)`。每一档都是 `KernelProvider` 的一个实现，`select()` 按 `OpConfig`（sm 型号 + 桶形状 + 裁剪标志）选择，调优 score 入 `TuneDb`（`kernels/{device}/{op}/{cfg}/tune.json`）。
- **D2 FMHA 装载**: FMHA 以源码形式随 crate（`crates/cuda/kernels/fmha/*.cu` + `include/flashinfer` 风格的 cutlass 头），经 JitCache nvcc 编译（gcode sm90a/sm100a）；初版用 flashinfer FMHA2 风格（100 行 kernel cuda，sm90 warpspec 懒做——007），存 heuristics：`head_dim 128 → warp 64`、`seq > 512 → 分块 128`。
- **D3 cubin 装载**: `cubins/{device}-{arch}/fa3-{sha256}.cubin` 与 `cudaLaunchKernelEx`；下载脚本（curl % 源 + js 小工具 sha256 校验），手动/CI 交付，失败回退 D2。
- **D4 CUDA Graph**: `GraphPool`（bucket 键 = (bs 桶, seq 桶, dtype)）；捕获模板驱动 `capture()`，复用内存（工作区地址重分配适配）；重放走原 batch 内地址族（`cudaGraphExecUpdate` 静态刷新 8/16/32/64）；miss → eager。
- **D5 双流重叠**: `compute_stream` + `comm_stream`(预留) — attn/FFN 跨流事件对（`cudaEventRecord/StreamWaitEvent`）。
- **D6 失败链**: 每个 Provider `launch()` 返回 `Fallback`（回退 003 code）而非 panic。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/cuda/src/fmha/` | kernel 源码 + `select_heuristics` + Provider (Vendor-CUTLASS) |
| `crates/cuda/src/fa3/` | cubin loader（下载/校验/回退链） |
| `crates/cuda/src/graph.rs` | `GraphPool`/`GraphBucket`, 捕获/重放模板 |
| `crates/cuda/src/overlap.rs` | 双流 wrap（事件对 API） |
| `crates/cuda/src/tunedb.rs` | bench 自动调优 + `tune.json` 读写 + 调度键 |

## Interface Contracts (slice-local)

```rust
pub struct FmhaProvider;                 // KernelProvider impl（Vendor-CUTLASS）
impl FmhaProvider { pub fn launch_attn_prefill(...); }      // sm90+；否则 None
pub struct Fa3CubinProvider;             // 可选：需设备上存在 cubin；否则 None → 自动回退
pub struct GraphPool;                    // get_or_capture(bucket) / replay(bucket, batch) / release(bucket)
impl GraphPool {
  pub fn get_or_capture(&mut self, key: BucketKey, tpl: &GraphTemplate) -> Result<usize, LaunchError>;
  pub fn replay(&self, idx: usize) -> Result<(), LaunchError>;
}
pub struct TuneDb;                       // load/save tune.json；select() 读
pub fn select_fmha(cfg: &OpConfig, db: &TuneDb) -> enum ProviderChoice; // Vendor>CUTLASS>native>eager
```

## Reference assets

- 本地仓库 `flashinfer/`: `fmha_v2/`（heuristics、warp config、TMA/warpspec 配置样例）与 `flashinfer-cubin`（下载 URL/校验语义）——仅借鉴模式，不引入其 Python 生态
- vLLM `cuda_graph` 桶化 + `registry`（buffer 生命周期）→ `GraphPool`
- ferrum-runtime: CUDA Graph +18% 参考数据（notes 中对标基准）
- llama.cpp 后端 2026 年对 sm90/100 的 tiling 调整（参考 `ggml-cuda/fmha-tiled`）

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| cubin/源码与 sm90a/100a 差异导致编译失败 | High | gcode 分级（sm90a 若失败退 sm90）；`select_heuristics` 仅按实测列 |
| graph 捕获下 batch 逃出桶（miss 增长） | Medium | 桶宽设计（8/16/32/64）+ 重放 fallback eage no-graph；桶 miss 余量监控 |
| vendor 回退链风险（静默降级） | Medium | 每条路径打 `tracing::warn`；notes.md 记录当日档位 |
| 源码内嵌 cutlass 头多（体积+SDK 镜像） | Low | fmha 仅用 flashinfer 少量头（`fmha_kernel_head.cuh` 摘录）；依赖仓库体积控制在 +5MB |

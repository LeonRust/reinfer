# Spec: CUDA runtime base (L1 — device/stream/event/buffer wrappers)

> Status: proposal · Owner: maintainers · Created: 2026-08-27 · Parent: specs/003-cuda-l0 (T2 展开) · Alignment: docs/design/cuda-phase-plan.md
> 前置事实（已入主线）：T1 cudarc 接线 ✅、T3 `cudaError→LaunchError` 白名单 ✅（crates/cuda/src/error.rs）。

## Problem Statement

003 T2 只有一行结果描述。L1 的目标是建立**可安全消费的 CUDA 运行时基座**：设备发现与信息、流、事件、设备/主机内存、异步拷贝——全部以 RAII 安全 API 形式交付，错误一律经 T3 白名单（fail-closed），为 L2（JitCache+首批内核）与 L3（推理闭环）提供唯一 FFI 入口。本切片不写任何 kernel。

## Success Metrics

- **接口完整**：L1 提供 `CudaContext`/`Stream`/`Event`/`DeviceBuffer`/`HostBuffer` 五个类型 + memcpy（D2H/H2D/D2D），语义与 002 契约中 ascen 层同构（跨后端对称）
- **真机 smoke（判定机 RTX 5090 sm_120）**：设备列表正确（≥1、可见 UUID）；alloc+copy 往返（D2H→H2D→D2D 全链）逐字节一致；事件 record→device_sync 成功；反复 1000 次 alloc/free 无泄漏（driver 计数稳定）
- **无 GPU 单测**：构造/Debug/错误前置/常量测试全绿（无驱动程序也编得过、测得了）
- **线程语义明确**：per-thread device 绑定（CUDA 运行时语义）在文档与 `debug_assert` 中约束
- **工程**：默认 feature 编译零 CUDA 依赖；`--features cuda` 编译+smoke 通过

## User Stories

1. 作为后端作者：`CudaContext::init(dev)` 后即可 `Stream::new`、`Event::new`、`DeviceBuffer::alloc`、`copy`——全部 `Result<_, LaunchError>`，无需碰 cudarc 细节。
2. 作为引擎调度者：`DeviceBuffer` 可跨 worker 线程传递（`Send`），并能断言"仅归属设备使用"（SAFETY 注释 + 调试钩子）。

## Acceptance Criteria

- [ ] `CudaContext`：`init()`（设备发现）/`device_count()`/`device_info(idx) -> DeviceInfo`（name/算力 major.minor/显存/设备 UUID 字符串）；每线程设备绑定语义有文档；`Drop` 释放次序明确（先销毁子资源）
- [ ] `Stream`：创建/销毁 RAII；Drop 前不隐式全同步（文档明示）；`as_cuda_stream()` 句柄暴露给 Jit/GEMM 层（crates/cuda 内部）
- [ ] `Event`：`record(stream)`/`device_sync()`/`query()`；Drop 前若未同步自动同步（防悬挂）
- [ ] `DeviceBuffer`：`alloc(size)->Result<Self>`（**Send** + SAFETY 注释"仅限归属 device"）；`as_ptr()`/`size()`；Drop → free；debug_assert device 一致性
- [ ] `HostBuffer`：pinned host 内存；Drop → free；`as_ptr()`
- [ ] memcpy：`copy_h2d/d2h/d2d`（异步版本走 stream；同步版本内部 stream+sync），错误经 `map_err`
- [ ] 真机 smoke 用例集 + 无 GPU 单测集（定义于 tasks T6）；`bench/notes.md` 记录设备信息（UUID/驱动/显存）
- [ ] 与 002（ascen）边界契约不冲突：内存**原语**属运行时层，**策略**（页池/引用）归 `crates/memory`——本切片不引入策略

## Non-Goals

- 任何 kernel/JitCache（L2）；内存池/VMM 策略（crates/memory 的 MemOps）；CUDA Graph 捕获（006）；多设备并行拓扑（TP）；WSL/容器内核兼容性说明之外的平台适配

## Constraints

- 仅经 cudarc（driver/runtime）；所有错误收敛 `map_err`（T3 白名单）；本 crate 是 unsafe 宿主（其余 crate 禁反向依赖，宪法 §2.1）
- 不修改 cudarc 类型（只在其上安全包装）；不新增第三方 GPU 依赖
- 数值裁判/参考实现与 L1 无关（无 math），跳过 parity：L1 验收 = 行为 smoke 而非数值

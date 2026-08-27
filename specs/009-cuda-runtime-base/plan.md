# Plan: CUDA runtime base (L1)

> Derived from specs/009-cuda-runtime-base/spec.md

## Architecture Decisions

- **D1 层级与所有权**：五个类型安全包装 → 全部 `Result<_, LaunchError>`；`crates/cuda` 为唯一 unsafe 宿主（cudarc 内部 unsafe），对外零 unsafe；`as_cuda_stream()` 等句柄 `pub(crate)` 内部可见（L2/L3 使用）。
- **D2 线程模型**：CUDA 运行时 = per-thread 设备绑定（`cudaSetDevice` 一次/线程）；`CudaContext` 记录 `dev`；每个 buffer 构造时快照 `dev`，`debug_assert` 校验访问线程的当前设备（通过 `cudaGetDevice` 校验），生产构建放行（无错误）仅文档约束。
- **D3 RAII 次序**：上下文内子资源（stream/event/buffer）析构先于上下文 Drop；Event Drop 自动 `device_sync()`；Stream Drop 前不隐式同步（文档明示，靠上层调度纪律）。
- **D4 memcpy 语义**：同步版 = 自己创建 ephemeral stream（或默认流）+ `cudaMemcpyAsync` + 同步 → `map_err`；异步版 = 显式 stream 内 `cudaMemcpyAsync`，返回 `Result<()>`。D2D 同设备；跨设备 D2D 显式报"not supported yet"（`LaunchError::Fatal` + 文档声明）。
- **D5 设备信息**：`DeviceInfo { index, name, major, minor, total_mem, uuid }` —— 从 cudarc driver/runtime 属性读取（**实现时探测函数名**：cudaGetDeviceProperties / cudaGetDeviceAttribute 组合；uuid 经 `cudaDeviceGetUuid` 或驱动 CUDA 属性，若本 toolkit 无 UUID 属性则退化为 hash(dev 信息)——task 中标注 TODO(probe)）。
- **D6 错误面**：`cudaSetDevice/cudaMallocHost/cudaMemcpyAsync` 等返回值 → `cudarc result` → 转 `map_err`（T3）；**strerror** 信息（`cudarc runtime::RuntimeError::error_string()`）附加到日志 `tracing::warn`（debug 助手）。
- **D7 测试分层**：`#[cfg(test)]` 无 driver 单测（构造/Debug/常量/错误前置）+ `#[cfg(all(test, feature="cuda"))]` 真机 tests（`#[ignore]` + allowlist 映射 `gpu-smoke` job→008 T2）；avoid panics in tests — use `helper::assert_launch`。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/cuda/src/context.rs` | `CudaContext`, `DeviceInfo`, `device_count`, per-thread 绑定 |
| `crates/cuda/src/stream.rs` / `event.rs` | RAII wrappers + 句柄暴露 |
| `crates/cuda/src/buffer.rs` | `DeviceBuffer`(Send)/`HostBuffer` + memcpy 族 |
| `crates/cuda/src/error.rs` | （已有 T3）+ `map_err` 辅助 & `tracing` 日志 |
| `crates/cuda/tests/smoke.rs` | 真机 smoke（`#[ignore]`, `--features cuda`） |
| `crates/cuda/lib.rs` | 模块导出 + ASCII 版文档图（L1-L3 定位） |

## Interface Contracts（L1 安全 API；实现细节（cudarc 映射）在 tasks 探测）

```rust
#[derive(Debug, Clone)]
pub struct DeviceInfo { pub index: u32, pub name: String, pub major: u32, pub minor: u32,
                        pub total_mem: u64, pub uuid: String }

pub struct CudaContext { /* dev: u32 */ }
impl CudaContext {
    pub fn init(default_dev: u32) -> Result<Self, LaunchError>;   // 每线程调用一次（D2）
    pub fn device_count() -> Result<u32, LaunchError>;
    pub fn device_info(index: u32) -> Result<DeviceInfo, LaunchError>;
    pub fn device_id(&self) -> u32;
}
impl Drop for CudaContext { /* 弱校验：若本线程尚有子资源则 warn（不强制） */ }

pub struct Stream { /* inner */ }
impl Stream {
    pub fn new(ctx: &CudaContext) -> Result<Self, LaunchError>;
    pub fn id(&self) -> CloudHandle;              // pub(crate)
    pub fn synchronize(&self) -> Result<(), LaunchError>;
}
pub struct Event { /* inner */ }
impl Event {
    pub fn new(ctx: &CudaContext) -> Result<Self, LaunchError>;
    pub fn record(&self, stream: &Stream) -> Result<(), LaunchError>;
    pub fn device_sync(&self) -> Result<(), LaunchError>;
    pub fn query(&self) -> Result<bool, LaunchError>;      // true=完成
}
impl Drop for Event { /* device_sync 兜底 */ }

pub struct DeviceBuffer { /* ptr, size, dev */ }
impl DeviceBuffer {
    pub fn alloc(dev: u32, size: usize) -> Result<Self, LaunchError>;
    pub fn as_ptr(&self) -> *const u8;
    pub fn size(&self) -> usize;
    pub fn device(&self) -> u32;
}
unsafe impl Send for DeviceBuffer { /* SAFETY: CUDA runtime 设备指针跨线程安全；仅限归属 device */ }
impl Drop for DeviceBuffer { /* cudaFree */ }

pub struct HostBuffer { /* ptr, size */ }   // pinned
impl HostBuffer { pub fn alloc(size: usize) -> Result<Self, LaunchError>; pub fn as_ptr(&self) -> *const u8; }

pub enum MemcpyKind { H2D, D2H, D2D }
pub fn copy(dev: u32, kind: MemcpyKind, dst: *mut u8, src: *const u8, bytes: usize,
            stream: Option<&Stream>) -> Result<(), LaunchError>;
```

## Reference assets（增量；全量见 深入设计补充 §3）

- cudarc 0.19.9 源码（`runtime/sys/mod.rs` 枚举与签名；`runtime/result.rs` error_string）
- 002/cann-rs 0001 L0 契约（跨后端对称性对照——同构边界、Send 决策、错误白名单）
- mini-sglang `engine.py`（每线程设备 + stream 分层用法）

## Risk Assessment

| Risk | Mitigation |
|---|---|
| cudarc 类型/函数名与契约不符（探测误差） | tasks T1-T5 均带 `TODO(probe)` 与编译期断言；spec 契约以真实签名回溯为准（changelog） |
| 线程语义误用（跨线程 buffer 传给另一 device 的 kernel） | D2 debug_assert + 文档（release 无 runtime 校验 → 未来加 device 标记校验） |
| Event/Stream Drop 次序引发悬挂 | D3 顺序 + Drop 兜底同步；smoke 压测 |
| UUID 属性缺 API | 退化方案（hash）记录，`DeviceInfo.uuid` 语义允许多值 |

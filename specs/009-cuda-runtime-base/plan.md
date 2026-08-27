# Plan: CUDA runtime base (L1)

> Derived from specs/009-cuda-runtime-base/spec.md · 事实基线 = cudarc 0.19.9 源码（B 组核查）

## Architecture Decisions

- **D1 所有权与宿主边界（A-H1 裁决）**：CUDA FFI 唯一宿主 = `crates/cuda`。`crates/jit`（L2）只负责编译流水线（nvcc 子进程、哈希、锁、产物缓存）——**不触 FFI**；`cuModuleLoad/Launch` 归 `crates/cuda`。因此 `CudaStream` 句柄 `pub(crate)` 成立；"唯一 FFI 入口"的精确表述 = "CUDA FFI 唯一宿主，jit 经其公开入口消费"。
- **D2 绑定层（B 核查结论）**：统一采用 **cudarc runtime 层窄绑定**（`runtime::sys::*` + `RuntimeError`），保证错误码单一体系（`cudaError_t` ↔ T3 白名单）。cudarc driver 安全层（`CudaContext/CudaStream/CudaEvent/CudaSlice`）仅作 L2/L3 内部便利层；009 内部直调 sys（本 crate 即 unsafe 宿主）。同步 memcpy 使用现成 `runtime::result::memcpy_*_sync`（不自建 ephemeral stream）；异步用 `memcpy_*_async` + 返回 `CudaEvent`。
- **D3 RAII 次序**：Context 无子资源注册表（**删除"弱校验 Drop"**——不可实现且增设计面；A-L1）；`CudaStream` Drop 只 `cudaStreamDestroy`（不隐式同步，纪律 = 上层在回收前 synchronize）；`CudaEvent` 以 `cudaEventCreateWithFlags(..., BlockingSync)` 创建（默认 DISABLE_TIMING 的 event 调 synchronize 不阻塞 CPU —— B#4），Drop 自动 synchronize，语义 ="仅等待本事件"。
- **D4 拷贝 API（A-H2 裁决）**：以 `MemRef<'a> { Device(&DeviceBuffer), Host(&HostBuffer) }` 为入参——拷贝前校验：方向合法（H2D/D2H 仅涉对应端；D2D 设备一致或 peer 可用）、边界（offsets+bytes ≤ 各自 len）、`DeviceBuffer` 归属设备与本线程当前设备（debug 构建断言）。**对外字段全私有，无裸指针暴露**。跨设备 D2D：`cudaDeviceCanAccessPeer` 探测 → 通过走 `cudaMemcpyPeerAsync`；不通过/执行失败 → 白名单分类（不得硬编码 Fatal-不支持——B#8）。
- **D5 设备信息**：`DeviceInfo { index, name, major, minor, total_mem, uuid }`；来源 `cudaDeviceGetProperties`（`cudaDeviceProp` 含全部字段，**uuid 直接读取，无退化方案**——B#2b）；uuid 格式化 8-4-4-4-12 hex；`Debug/Clone` 无 feature 依赖。
- **D6 错误面**：单轨 `RuntimeError` → `map_err`（T3；`error_string()` 返回 `&CStr` → `to_string_lossy()` 记 `tracing::warn`）。函数名勘误（已核）：`cudaGetDeviceAttribute`→**`cudaDeviceGetAttribute`**；`cudaGetMemInfo`→**`cudaMemGetInfo`**；**`cudaSetDeviceAsync` 不存在**（debug_assert 用 `cudaGetDevice`；线程绑定语义 = `cudaSetDevice`/driver `bind_to_thread` 类）。`cudaErrorNotReady`=600（CUDA 12 新值）。
- **D7 测试分层**（C 组措辞）：真机 = 集成测试 `tests/smoke.rs`（`#[cfg(feature="cuda")]` + `#[ignore]`，allowlist 预登记，映射 008 接线表 `smoke` 行）；无 GPU = 具名单测（计数闸 ≥7）；**毒化隔离**：涉及 `IllegalAddress` 的用例放独立测试二进制末尾运行；**独占前提**：`CUDA_VISIBLE_DEVICES=0` + `--test-threads=1`；泄漏口径 = 绑定分配体积公式（F7），非 ±1% 总量。
- **D8 依赖收窄（F10）**：workspace `cudarc = { version="0.19", default-features=false, features=["driver","runtime","dynamic-linking"] }`（已落地）。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/cuda/src/context.rs` | `CudaContext`/`DeviceInfo`/`DeviceId`（core 类型） |
| `crates/cuda/src/stream.rs` / `event.rs` | RAII + `pub(crate)` 句柄/事件查询 |
| `crates/cuda/src/buffer.rs` | `DeviceBuffer`(Send)/`HostBuffer`/`MemRef` + `copy/copy_async` |
| `crates/cuda/src/error.rs` | 已有 T3 白名单 + query 特判 600 + `map_err`（错误面辅助） |
| `crates/core/src/device.rs` | `DeviceId`（纯类型，新建） |
| `crates/cuda/tests/smoke.rs` | 真机 smoke（`#[ignore]` → 008 `smoke` job） |
| `bench/runner-info.json` | 设备四元组（UUID/驱动/sm/cuBLAS；008 D2 工件） |

## Interface Contracts（按 cudarc 实测改写；实现细节可微调但语义锁定）

```rust
// crates/core
pub struct DeviceId(u32);   // 新建；core 无 feature 依赖

// crates/cuda（unsafe 宿主；内部 = cudarc runtime::sys 窄绑定）
pub struct CudaContext { /* dev */ }
impl CudaContext {
    pub fn init(dev: DeviceId) -> Result<Self, LaunchError>;   // 每线程一次（D 文档）
    pub fn device_count() -> Result<u32, LaunchError>;
    pub fn device_info(index: u32) -> Result<DeviceInfo, LaunchError>;  // cudaDeviceGetProperties
    pub fn current_device() -> Result<DeviceId, LaunchError>;           // cudaGetDevice（调试校验）
}
pub struct DeviceInfo { pub index: u32, pub name: String, pub major: u32, pub minor: u32,
                        pub total_mem: u64, pub uuid: String }          // Clone+Debug

pub struct CudaStream { /* dev, raw */ }
impl CudaStream {
    pub fn new(dev: DeviceId) -> Result<Self, LaunchError>;
    pub fn synchronize(&self) -> Result<(), LaunchError>;
    pub(crate) fn handle(&self) -> sys::cudaStream_t;   // 内部（L2/L3 经 cuda 公开入口）
}
pub struct CudaEvent { /* dev, raw */ }
impl CudaEvent {
    pub fn new(dev: DeviceId) -> Result<Self, LaunchError>;     // cudaEventCreateWithFlags(BlockingSync)
    pub fn record(&self, stream: &CudaStream) -> Result<(), LaunchError>;
    pub fn synchronize(&self) -> Result<(), LaunchError>;
    pub fn query(&self) -> Result<bool, LaunchError>;           // 600 → Ok(false)；其余错误分类
}
impl Drop for CudaEvent { /* synchronize 兜底：仅等待本事件 */ }

pub struct DeviceBuffer { /* dev, ptr, size */ }                 // unsafe impl Send（SAFETY：仅限归属 device）
impl DeviceBuffer { pub fn alloc(dev: DeviceId, size: usize) -> Result<Self, LaunchError>;
                    pub fn as_ptr(&self) -> *const u8; pub fn size(&self) -> usize; pub fn device(&self) -> DeviceId; }
pub struct HostBuffer { /* ptr, size */ }                        // cudaMallocHost/cudaFreeHost

pub enum MemRef<'a> { Device(&'a DeviceBuffer), Host(&'a HostBuffer) }  // 无裸指针对外
pub fn copy(dst: &mut MemRef<'_>, src: &MemRef<'_>, bytes: usize,
            stream: Option<&CudaStream>) -> Result<(), LaunchError>;     // None=同步（内部 sync 包装）
pub fn copy_async(dst: &mut MemRef<'_>, src: &MemRef<'_>, bytes: usize,
                  stream: &CudaStream) -> Result<CudaEvent, LaunchError>; // Event=同步凭证
// 跨设备 D2D：cudaDeviceCanAccessPeer 探测 → cudaMemcpyPeerAsync；失败按白名单分类
```

## Reference assets（增量；全量见 深入设计补充 §3）

- cudarc 0.19.9：`runtime/sys/mod.rs`（绑定与码值）、`runtime/result.rs`（sync 包装/error_string）、driver 层 `CudaContext::uuid()`（备用）
- 002/plan（跨端对称）；cann-rs 0001 L0（Send 决策来源）；mini-sglang `engine.py`（每线程设备/流分层）

## Risk Assessment

| Risk | Mitigation |
|---|---|
| sys 绑定签名/码值与契约漂移 | 编译期断言 + 实现回填 changelog（R3 契约先行） |
| 事件阻塞语义误用 | BlockingSync 显式 + 语义文档（仅等待本事件） |
| 跨设备 D2D 探测失败路径 | 白名单分类 + 真机 smoke 覆盖 peer 场景（若双卡不可用：跳过记录） |
| 上下文毒化（IllegalAddress） | D7 独立进程/末尾 + allowlist 注释 |
| 无 GPU 空转 | 计数闸 ≥7 + 具名清单（拒绝"空跑绿"） |

//! CUDA 事件（L1 T2；feature `cuda` 门控）。
//!
//! 事件以 `cudaEventBlockingSync` 创建：`cudaEventSynchronize` 阻塞 CPU 至
//! 本事件完成；**未设** `cudaEventDisableTiming` → 事件记录计时，`elapsed_ms`
//! 可用（S1-2 decode 步 profile 探针的基础原语——event.rs 常量此前误用
//! 运行时 API 值 0x2（= `cudaEventDisableTiming`），导致计时不可用、
//! 阻塞语义从未生效；按 13.2 头文件 `driver_types.h` 修正为 0x1）。
//! `Drop` 兜底 = 先 `synchronize`（**仅等待本事件**，非设备级全同步）再 `destroy`。

use cudarc::runtime::sys;
use reinfer_core::DeviceId;

use crate::error::{LaunchError, event_query_status, from_runtime_error};
use crate::stream::CudaStream;

/// `cudaEventBlockingSync = 0x1`（cudarc sys 未导出 CUDA 头常量，本地定义；
/// 值源 `driver_types.h`——0x2 为 `cudaEventDisableTiming`，勿混用）。
pub const CU_EVENT_BLOCKING_SYNC: u32 = 0x1;

/// 事件（记录在流上、轮询/阻塞其完成）。
#[derive(Debug)]
pub struct CudaEvent {
    dev: DeviceId,
    raw: sys::cudaEvent_t,
}

impl CudaEvent {
    /// 创建阻塞同步事件（显式 BlockingSync 标志）。
    pub fn new(dev: DeviceId) -> Result<Self, LaunchError> {
        let mut raw = core::ptr::null_mut();
        unsafe { sys::cudaEventCreateWithFlags(&mut raw, CU_EVENT_BLOCKING_SYNC) }
            .result()
            .map_err(from_runtime_error)?;
        Ok(Self { dev, raw })
    }

    /// 在指定流上记录本事件。
    pub fn record(&self, stream: &CudaStream) -> Result<(), LaunchError> {
        unsafe { sys::cudaEventRecord(self.raw, stream.handle()) }
            .result()
            .map_err(from_runtime_error)
    }

    /// 阻塞等待本事件完成。
    pub fn synchronize(&self) -> Result<(), LaunchError> {
        unsafe { sys::cudaEventSynchronize(self.raw) }.result().map_err(from_runtime_error)
    }

    /// 非阻塞完成态：`Ok(true)` 已完成 / `Ok(false)` 未完成（600 特判）/ `Err` 真错误（白名单）。
    pub fn query(&self) -> Result<bool, LaunchError> {
        event_query_status(unsafe { sys::cudaEventQuery(self.raw) } as i32)
    }

    /// 本事件到 `end` 的流上耗时（毫秒）。两个事件必须已 record 到同一流
    /// （或同一流依赖链）且均已完成；`end` 必须晚于 `self`。计时可用性：
    /// 事件以 `CU_EVENT_BLOCKING_SYNC` 创建（未设 `CU_EVENT_DISABLE_TIMING`），
    /// 故 elapsed 有效——S1-2 decode 步 profile 探针的基础原语。
    pub fn elapsed_ms(&self, end: &CudaEvent) -> Result<f32, LaunchError> {
        let mut ms: f32 = 0.0;
        // SAFETY: 两个事件句柄有效且已 record（驱动侧校验先后关系）。
        unsafe { sys::cudaEventElapsedTime(&mut ms, self.raw, end.raw) }
            .result()
            .map_err(from_runtime_error)?;
        Ok(ms)
    }

    /// 所属设备。
    #[inline]
    pub fn device(&self) -> DeviceId {
        self.dev
    }
}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        // BlockingSync 事件：先阻塞直至完成（仅本事件），再销毁；错误显式忽略
        let _ = unsafe { sys::cudaEventSynchronize(self.raw) }.result();
        let _ = unsafe { sys::cudaEventDestroy(self.raw) }.result();
    }
}

#[cfg(all(test, feature = "cuda"))]
mod ffi_tests {
    use super::*;
    use crate::CudaContext;

    fn setup() -> (crate::CudaContext, CudaStream) {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init");
        let dev = ctx.device_id();
        let stream = CudaStream::new(dev).expect("stream");
        (ctx, stream)
    }

    #[test]
    fn unrecorded_event_query_is_completed() {
        let (_ctx, _stream) = setup();
        let evt = CudaEvent::new(DeviceId::new(0)).expect("event");
        // 实测（RTX 5090 / driver 595.84 / cudarc 0.19.9，2026-08-27）：从未 record 的
        // 事件即"完成态"；`cudaErrorNotReady(600)` 仅在"已 record 且未完成"时返回。
        // 009 评审 C-F3 的"未 record → false"预设按实测回填（probe 划线）。
        // 真机不做确定性 false 断言（异步时序 flaky）；600 分支由 error.rs 纯函数单测覆盖。
        assert!(evt.query().expect("query"));
    }

    #[test]
    fn record_sync_then_query_true() {
        let (_ctx, stream) = setup();
        let evt = CudaEvent::new(DeviceId::new(0)).expect("event");
        evt.record(&stream).expect("record");
        stream.synchronize().expect("sync stream");
        assert!(evt.query().expect("query"));
    }

    #[test]
    fn drop_after_completed_does_not_hang() {
        let (_ctx, stream) = setup();
        let evt = CudaEvent::new(DeviceId::new(0)).expect("event");
        evt.record(&stream).expect("record");
        stream.synchronize().expect("sync stream");
        drop(evt); // BlockingSync 兜底同步 + 销毁，不挂起
    }

    #[test]
    fn elapsed_between_two_recorded_events() {
        let (_ctx, stream) = setup();
        let a = CudaEvent::new(DeviceId::new(0)).expect("event a");
        let b = CudaEvent::new(DeviceId::new(0)).expect("event b");
        a.record(&stream).expect("record a");
        b.record(&stream).expect("record b");
        stream.synchronize().expect("sync stream");
        let ms = a.elapsed_ms(&b).expect("elapsed");
        assert!(ms >= 0.0, "elapsed must be non-negative, got {ms}");
    }
}

//! CUDA 事件（L1 T2；feature `cuda` 门控）。
//!
//! 事件以 `CU_EVENT_BLOCKING_SYNC` 创建：默认 `CU_EVENT_DISABLE_TIMING`
//! 事件调用 `cudaEventSynchronize` 不会阻塞 CPU——见 009 评审 B#4。
//! `Drop` 兜底 = 先 `synchronize`（**仅等待本事件**，非设备级全同步）再 `destroy`。

use cudarc::runtime::sys;
use reinfer_core::DeviceId;

use crate::error::{LaunchError, event_query_status, from_runtime_error};
use crate::stream::CudaStream;

/// `CU_EVENT_BLOCKING_SYNC = 0x2`（cudarc sys 未导出 CUDA 头常量，本地定义；源自 cudaEventCreateWithFlags 文档）。
pub const CU_EVENT_BLOCKING_SYNC: u32 = 0x2;

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
}

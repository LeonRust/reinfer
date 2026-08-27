//! CUDA 流（L1 T2；feature `cuda` 门控）。
//!
//! RAII 包装 `cudaStreamCreate/Destroy`；`Drop` 只销毁、不隐式同步——
//! 回收纪律由上层负责（specs/009 plan D3）。

use cudarc::runtime::sys;
use reinfer_core::DeviceId;

use crate::error::{LaunchError, from_runtime_error};

/// 计算流（创建即持有；句柄仅 crate 内部使用）。
#[derive(Debug)]
pub struct CudaStream {
    dev: DeviceId,
    raw: sys::cudaStream_t,
}

impl CudaStream {
    /// 在指定设备上下文（本线程已绑定）上创建流。
    pub fn new(dev: DeviceId) -> Result<Self, LaunchError> {
        let mut raw = core::ptr::null_mut();
        unsafe { sys::cudaStreamCreate(&mut raw) }.result().map_err(from_runtime_error)?;
        Ok(Self { dev, raw })
    }

    /// 阻塞等待本流上的全部工作完成。
    pub fn synchronize(&self) -> Result<(), LaunchError> {
        unsafe { sys::cudaStreamSynchronize(self.raw) }.result().map_err(from_runtime_error)
    }

    /// 原始句柄（crate 内部：注册表/后续 kernel 启动经公开入口消费）。
    pub(crate) fn handle(&self) -> sys::cudaStream_t {
        self.raw
    }

    /// 所属设备。
    #[inline]
    pub fn device(&self) -> DeviceId {
        self.dev
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        // 显式忽略错误：销毁发生在用户可见结果之后
        let _ = unsafe { sys::cudaStreamDestroy(self.raw) }.result();
    }
}

#[cfg(all(test, feature = "cuda"))]
mod ffi_tests {
    use super::*;
    use crate::CudaContext;

    #[test]
    fn stream_create_sync_and_drop() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init");
        let dev = ctx.device_id();
        let stream = CudaStream::new(dev).expect("stream");
        stream.synchronize().expect("empty stream sync"); // 空流同步应立即返回
        drop(stream); // Drop 不 hang
    }
}

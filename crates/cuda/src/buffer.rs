//! 设备/主机内存缓冲（L1 T3；feature `cuda` 门控）。
//!
//! RAII 包装 `cudaMalloc/cudaFree`（设备侧）与 `cudaMallocHost/cudaFreeHost`
//! （pinned 主机侧）。`DeviceBuffer` 实现 `Send`——SAFETY 语义
//! "仅限归属设备使用"（决策锚：specs/002 plan 契约行 ← cann-rs 0001）。

use cudarc::runtime::sys;
use reinfer_core::DeviceId;

use crate::error::{LaunchError, from_runtime_error};

/// 设备侧内存（`cudaMalloc`），跨线程可传（Send），仅限归属设备使用。
#[derive(Debug)]
pub struct DeviceBuffer {
    dev: DeviceId,
    ptr: *mut core::ffi::c_void,
    len: usize,
}

/// 安全：CUDA 运行时设备指针可跨线程持有（每线程自行绑定设备后使用）；
/// 语义约束"仅限归属设备使用"（见 crate 文档与 specs/009 plan D2）。
///
/// # Safety
/// `len` 与指针生命周期一致（由本结构持有并负责 `cudaFree`）。
unsafe impl Send for DeviceBuffer {}

impl DeviceBuffer {
    /// 在指定设备上分配 `size` 字节。
    pub fn alloc(dev: DeviceId, size: usize) -> Result<Self, LaunchError> {
        let mut ptr = core::ptr::null_mut();
        unsafe { sys::cudaMalloc(&mut ptr, size) }.result().map_err(from_runtime_error)?;
        Ok(Self { dev, ptr, len: size })
    }

    /// 设备侧指针（只读句柄）。
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.cast::<u8>()
    }

    /// 分配字节数。
    #[inline]
    pub fn size(&self) -> usize {
        self.len
    }

    /// 归属设备。
    #[inline]
    pub fn device(&self) -> DeviceId {
        self.dev
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        let _ = unsafe { sys::cudaFree(self.ptr) }.result();
    }
}

/// pinned 主机侧内存（`cudaMallocHost`），用于 D2H/H2D 与 offload 路径。
#[derive(Debug)]
pub struct HostBuffer {
    ptr: *mut core::ffi::c_void,
    len: usize,
}

impl HostBuffer {
    /// 分配 pinned 主机内存。
    pub fn alloc(size: usize) -> Result<Self, LaunchError> {
        let mut ptr = core::ptr::null_mut();
        unsafe { sys::cudaMallocHost(&mut ptr, size) }.result().map_err(from_runtime_error)?;
        Ok(Self { ptr, len: size })
    }

    /// 主机侧指针（只读句柄）。
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.cast::<u8>()
    }

    /// 分配字节数。
    #[inline]
    pub fn size(&self) -> usize {
        self.len
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        let _ = unsafe { sys::cudaFreeHost(self.ptr) }.result();
    }
}

#[cfg(all(test, feature = "cuda"))]
mod ffi_tests {
    use super::*;
    use crate::CudaContext;
    use cudarc::runtime::sys;

    /// 独占设备 + 单线程前提下的小块泄漏检测（009 F7 公式）：
    /// 快照 `(free, total)` → 1000 × alloc(1 MiB)+memset 写回 + free →
    /// `free_after >= free_before - (1 MiB*1000*1%) - 8 MiB(slack)` 且 `total` 不变。
    #[test]
    fn alloc_free_1000_no_leak() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init");
        let dev = ctx.device_id();
        let (mut free_before, mut total) = (0usize, 0usize);
        let snapshot = |f: &mut usize, t: &mut usize| {
            unsafe { sys::cudaMemGetInfo(f, t) }.result().expect("mem info");
        };
        snapshot(&mut free_before, &mut total);
        assert!(total > 0);
        const ONE: usize = 1 << 20; // 1 MiB
        for _ in 0..1000 {
            let buf = DeviceBuffer::alloc(dev, ONE).expect("alloc");
            // 写回校验（非空读写工作负载；不影响泄漏判定）
            unsafe { sys::cudaMemsetAsync(buf.ptr, 0xAB, ONE, core::ptr::null_mut()) }
                .result()
                .expect("memset");
            unsafe { sys::cudaStreamSynchronize(core::ptr::null_mut()) }.result().expect("sync");
        }
        let (mut free_after, mut total_after) = (0usize, 0usize);
        snapshot(&mut free_after, &mut total_after);
        assert_eq!(total_after, total, "total mem changed");
        let allowance = ONE * 1000 / 100 + 8 * ONE; // 1% 总泄漏 + 8 MiB slack
        assert!(
            free_after >= free_before.saturating_sub(allowance),
            "free {} < before {} - allowance {}",
            free_after,
            free_before,
            allowance
        );
    }
}

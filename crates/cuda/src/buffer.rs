//! 设备/主机内存缓冲（L1 T3；feature `cuda` 门控）。
//!
//! RAII 包装 `cudaMalloc/cudaFree`（设备侧）与 `cudaMallocHost/cudaFreeHost`
//! （pinned 主机侧）。`DeviceBuffer` 实现 `Send`——SAFETY 语义
//! "仅限归属设备使用"（决策锚：specs/002 plan 契约行 ← cann-rs 0001）。

use cudarc::runtime::sys;
use reinfer_core::DeviceId;

use crate::error::{LaunchError, from_runtime_error};
use crate::{CudaContext, CudaEvent, CudaStream};

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

/// 拷贝视图：对一端（Device/Host）的整体引用；用于 `copy/copy_async`。
#[derive(Debug, Clone, Copy)]
pub enum MemRef<'a> {
    /// 设备侧缓冲。
    Device(&'a DeviceBuffer),
    /// pinned 主机侧缓冲。
    Host(&'a HostBuffer),
}

impl MemRef<'_> {
    /// 起始指针。
    #[inline]
    pub fn ptr(&self) -> *const u8 {
        match self {
            Self::Device(b) => b.as_ptr(),
            Self::Host(b) => b.as_ptr(),
        }
    }

    /// 长度（字节）。
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Device(b) => b.size(),
            Self::Host(b) => b.size(),
        }
    }

    /// 是否为空视图（len == 0）。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 设备索引（`None` = Host 侧）。
    #[inline]
    pub fn device(&self) -> Option<u32> {
        match self {
            Self::Device(b) => Some(b.device().index()),
            Self::Host(_) => None,
        }
    }

    pub(crate) fn end(&self) -> reinfer_kernels::mem_check::MemRefEnd {
        reinfer_kernels::mem_check::MemRefEnd { offset: 0, len: self.len(), dev: self.device() }
    }
}

fn kind_of(
    dst: Option<u32>,
    src: Option<u32>,
) -> Result<reinfer_kernels::mem_check::MemcpyKind, LaunchError> {
    use reinfer_kernels::mem_check::MemcpyKind::*;
    match (dst, src) {
        (Some(_), None) => Ok(H2D),
        (None, Some(_)) => Ok(D2H),
        (Some(_), Some(_)) => Ok(D2D),
        (None, None) => Err(LaunchError::Fatal), // host→host 不支持
    }
}

/// 同步/异步拷贝实现：同设备走 `cudaMemcpyAsync`，跨设备 D2D 先
/// `cudaDeviceCanAccessPeer` 探测再 `cudaMemcpyPeerAsync`（009 评审 B#8）。
fn memcpy_launch(
    kind: reinfer_kernels::mem_check::MemcpyKind,
    dst: *mut core::ffi::c_void,
    src: *const core::ffi::c_void,
    bytes: usize,
    dst_dev: Option<u32>,
    src_dev: Option<u32>,
    stream: cudarc::runtime::sys::cudaStream_t,
) -> Result<(), LaunchError> {
    use cudarc::runtime::sys;
    match (kind, dst_dev, src_dev) {
        (_, Some(d), Some(s)) if d != s => {
            // 跨设备：能力探测（dst 设备可否访问 src 设备）
            let mut can: core::ffi::c_int = 0;
            unsafe { sys::cudaDeviceCanAccessPeer(&mut can, d as i32, s as i32) }
                .result()
                .map_err(from_runtime_error)?;
            if can == 0 {
                return Err(LaunchError::Fatal); // 无 peer 能力（fail-closed）
            }
            unsafe { sys::cudaMemcpyPeerAsync(dst, d as i32, src, s as i32, bytes, stream) }
                .result()
                .map_err(from_runtime_error)
        }
        _ => {
            use cudarc::runtime::sys::cudaMemcpyKind::*;
            let k = match kind {
                reinfer_kernels::mem_check::MemcpyKind::H2D => cudaMemcpyHostToDevice,
                reinfer_kernels::mem_check::MemcpyKind::D2H => cudaMemcpyDeviceToHost,
                reinfer_kernels::mem_check::MemcpyKind::D2D => cudaMemcpyDeviceToDevice,
            };
            unsafe { sys::cudaMemcpyAsync(dst, src, bytes, k, stream) }
                .result()
                .map_err(from_runtime_error)
        }
    }
}

/// 阻塞拷贝：`stream` 为 `None` 时使用默认流并立即同步返回。
///
/// 拷贝前经 [`reinfer_kernels::mem_check::validate_memref`]（方向/边界/归属），
/// 失败返回 `LaunchError::Fatal`（参数类）。
pub fn copy(
    dst: &mut MemRef<'_>,
    src: &MemRef<'_>,
    bytes: usize,
    stream: Option<&CudaStream>,
) -> Result<(), LaunchError> {
    let current = CudaContext::current_device()?.index();
    let kind = kind_of(dst.device(), src.device())?;
    reinfer_kernels::mem_check::validate_memref(
        kind,
        &dst.end(),
        &src.end(),
        bytes,
        &reinfer_kernels::mem_check::PeerPolicy { current_dev: current, allow_peer: true },
    )?;
    let raw_stream = stream.map(CudaStream::handle).unwrap_or(core::ptr::null_mut());
    let dst_ptr = dst.ptr() as *mut core::ffi::c_void;
    let src_ptr = src.ptr() as *const core::ffi::c_void;
    memcpy_launch(kind, dst_ptr, src_ptr, bytes, dst.device(), src.device(), raw_stream)?;
    // None（阻塞语义）：默认流同步（CUDA 12 的 legacy 顺序语义下等价于阻塞）
    if stream.is_none() {
        unsafe { cudarc::runtime::sys::cudaStreamSynchronize(core::ptr::null_mut()) }
            .result()
            .map_err(from_runtime_error)?;
    }
    Ok(())
}

/// 异步拷贝：返回 `CudaEvent` 作为同步凭证（记录于给定流；909 plan A-L8）。
pub fn copy_async(
    dst: &mut MemRef<'_>,
    src: &MemRef<'_>,
    bytes: usize,
    stream: &CudaStream,
) -> Result<CudaEvent, LaunchError> {
    let current = CudaContext::current_device()?.index();
    let kind = kind_of(dst.device(), src.device())?;
    reinfer_kernels::mem_check::validate_memref(
        kind,
        &dst.end(),
        &src.end(),
        bytes,
        &reinfer_kernels::mem_check::PeerPolicy { current_dev: current, allow_peer: true },
    )?;
    let dst_ptr = dst.ptr() as *mut core::ffi::c_void;
    let src_ptr = src.ptr() as *const core::ffi::c_void;
    memcpy_launch(kind, dst_ptr, src_ptr, bytes, dst.device(), src.device(), stream.handle())?;
    let evt = CudaEvent::new(stream.device())?;
    evt.record(stream)?;
    Ok(evt)
}

#[cfg(all(test, feature = "cuda"))]
mod ffi_tests {
    use super::*;
    use crate::CudaContext;
    use cudarc::runtime::sys;

    const ONE_MIB: usize = 1 << 20;

    fn fill_host(buf: &HostBuffer, seed: u8) {
        // SAFETY：宿主 pinned 内存由本结构持有，长度固定
        unsafe {
            let s = core::slice::from_raw_parts_mut(buf.as_ptr() as *mut u8, buf.size());
            for (i, b) in s.iter_mut().enumerate() {
                *b = seed.wrapping_add(i as u8);
            }
            // 同步：pinned host 写对设备侧可见无需额外屏障（D2H/H2D 拷贝隐式定义次序）
        }
    }

    fn host_snapshot(buf: &HostBuffer) -> Vec<u8> {
        // SAFETY：同上；读取以比对
        unsafe {
            let s = core::slice::from_raw_parts(buf.as_ptr(), buf.size());
            s.to_vec()
        }
    }

    /// H2D → D2D → D2H 三链同步往返：与 1 MiB 确定性填充源逐字节一致。
    #[test]
    fn memcpy_roundtrip_sync() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init");
        let dev = ctx.device_id();
        let src = HostBuffer::alloc(ONE_MIB).expect("host src");
        fill_host(&src, 0xA5);
        let d1 = DeviceBuffer::alloc(dev, ONE_MIB).expect("d1");
        let d2 = DeviceBuffer::alloc(dev, ONE_MIB).expect("d2");
        let out = HostBuffer::alloc(ONE_MIB).expect("host out");

        copy(&mut MemRef::Device(&d1), &MemRef::Host(&src), ONE_MIB, None).expect("h2d");
        copy(&mut MemRef::Device(&d2), &MemRef::Device(&d1), ONE_MIB, None).expect("d2d");
        copy(&mut MemRef::Host(&out), &MemRef::Device(&d2), ONE_MIB, None).expect("d2h");

        assert_eq!(host_snapshot(&out), host_snapshot(&src), "sync roundtrip mismatch");
    }

    /// 异步三链 + 事件同步。
    #[test]
    fn memcpy_roundtrip_async() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init");
        let dev = ctx.device_id();
        let stream = CudaStream::new(dev).expect("stream");
        let src = HostBuffer::alloc(ONE_MIB).expect("host src");
        fill_host(&src, 0x5A);
        let d1 = DeviceBuffer::alloc(dev, ONE_MIB).expect("d1");
        let d2 = DeviceBuffer::alloc(dev, ONE_MIB).expect("d2");
        let out = HostBuffer::alloc(ONE_MIB).expect("host out");

        let e1 = copy_async(&mut MemRef::Device(&d1), &MemRef::Host(&src), ONE_MIB, &stream)
            .expect("h2d");
        let e2 = copy_async(&mut MemRef::Device(&d2), &MemRef::Device(&d1), ONE_MIB, &stream)
            .expect("d2d");
        let e3 = copy_async(&mut MemRef::Host(&out), &MemRef::Device(&d2), ONE_MIB, &stream)
            .expect("d2h");
        for e in [&e1, &e2, &e3] {
            e.synchronize().expect("event sync");
        }
        assert_eq!(host_snapshot(&out), host_snapshot(&src), "async roundtrip mismatch");
    }

    /// 跨设备 peer：本机单卡时跳过（空跑合法）；双卡环境启用时构造 dev0/dev1 往返。
    #[test]
    fn memcpy_peer_d2d_when_available() {
        let count = CudaContext::device_count().expect("count");
        if count >= 2 {
            // 双卡：dst=dev1、src=dev0，经 peer 探测后 cudaMemcpyPeerAsync
            let ctx1 = CudaContext::init(DeviceId::new(1)).expect("init dev1");
            let d0 = DeviceBuffer::alloc(DeviceId::new(0), ONE_MIB).expect("d0");
            let d1 = DeviceBuffer::alloc(DeviceId::new(1), ONE_MIB).expect("d1");
            let _ = ctx1;
            // 归属校验需在 dev1 线程执行 copy（当前线程已 set dev1）
            copy(&mut MemRef::Device(&d1), &MemRef::Device(&d0), ONE_MIB, None).expect("peer d2d");
        }
    }

    /// 注入 (a)：`alloc(total_mem + 1)` → 精确 `Err(Oom)`（内存白名单 2）。
    #[test]
    fn alloc_overflow_is_oom() {
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init");
        let dev = ctx.device_id();
        let total = CudaContext::device_info(0).expect("info").total_mem;
        let err = DeviceBuffer::alloc(dev, total as usize + 1).expect_err("must fail");
        assert_eq!(err, LaunchError::Oom, "expected Oom for over-size alloc");
    }

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

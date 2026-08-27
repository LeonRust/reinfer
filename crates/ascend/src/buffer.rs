//! 昇腾缓冲与拷贝（011 T3；安全面：校验先行，`unsafe` 仅收敛于 cann memcpy 两次调用）。

use cann::buffer::MemcpyKind as CannKind;
use reinfer_core::DeviceId;

use crate::error::from_cann_error;
use crate::stream::{AscendEvent, AscendStream};
use reinfer_kernels::LaunchError;
use reinfer_kernels::mem_check::{self, MemRefEnd, MemcpyKind, PeerPolicy};

/// 昇腾设备缓冲（含归属设备记录——cann 原语不携带，本层补上）。
#[derive(Debug)]
pub struct AscendDeviceBuffer {
    _inner: cann::buffer::DeviceBuffer,
    dev: DeviceId,
}

impl AscendDeviceBuffer {
    /// 在当前线程绑定设备上分配（`MemFlags::NormalOnly` 起步，flags 可后续透传）。
    pub fn alloc(dev: DeviceId, size: usize) -> Result<Self, LaunchError> {
        cann::buffer::DeviceBuffer::alloc(size, cann::buffer::MemFlags::NormalOnly)
            .map(|inner| Self { _inner: inner, dev })
            .map_err(|e| from_cann_error(&e))
    }

    /// 设备侧指针（只读句柄）。
    pub fn as_ptr(&self) -> *const u8 {
        self._inner.as_ptr()
    }

    /// 分配字节数。
    pub fn size(&self) -> usize {
        self._inner.len()
    }

    /// 归属设备。
    pub fn device(&self) -> DeviceId {
        self.dev
    }
}

/// 昇腾 pinned 主机缓冲。
#[derive(Debug)]
pub struct AscendHostBuffer {
    _inner: cann::buffer::HostBuffer,
}

impl AscendHostBuffer {
    /// 分配 pinned 主机内存。
    pub fn alloc(size: usize) -> Result<Self, LaunchError> {
        cann::buffer::HostBuffer::alloc(size)
            .map(|inner| Self { _inner: inner })
            .map_err(|e| from_cann_error(&e))
    }

    /// 主机侧指针（只读句柄）。
    pub fn as_ptr(&self) -> *const u8 {
        self._inner.as_ptr()
    }

    /// 分配字节数。
    pub fn size(&self) -> usize {
        self._inner.len()
    }
}

/// 拷贝视图：设备/主机两端（同 011 镜像；公开面零裸指针）。
#[derive(Debug, Clone, Copy)]
pub enum AscendMemRef<'a> {
    /// 设备侧缓冲。
    Device(&'a AscendDeviceBuffer),
    /// pinned 主机侧缓冲。
    Host(&'a AscendHostBuffer),
}

impl AscendMemRef<'_> {
    /// 起始指针。
    pub fn ptr(&self) -> *const u8 {
        match self {
            Self::Device(b) => b.as_ptr(),
            Self::Host(b) => b.as_ptr(),
        }
    }

    /// 长度（字节）。
    pub fn len(&self) -> usize {
        match self {
            Self::Device(b) => b.size(),
            Self::Host(b) => b.size(),
        }
    }

    /// 设备索引（`None` = Host 侧）。
    pub fn device(&self) -> Option<u32> {
        match self {
            Self::Device(b) => Some(b.device().index()),
            Self::Host(_) => None,
        }
    }

    /// 是否为空视图（len == 0）。
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn end(&self) -> MemRefEnd {
        MemRefEnd { offset: 0, len: self.len(), dev: self.device() }
    }
}

fn kind_of(dst: Option<u32>, src: Option<u32>) -> Result<MemcpyKind, LaunchError> {
    use MemcpyKind::*;
    match (dst, src) {
        (Some(_), None) => Ok(H2D),
        (None, Some(_)) => Ok(D2H),
        (Some(_), Some(_)) => Ok(D2D),
        (None, None) => Err(LaunchError::Fatal),
    }
}

fn to_cann(kind: MemcpyKind) -> CannKind {
    match kind {
        MemcpyKind::H2D => CannKind::HostToDevice,
        MemcpyKind::D2H => CannKind::DeviceToHost,
        MemcpyKind::D2D => CannKind::DeviceToDevice,
    }
}

/// 阻塞拷贝（stream=None 用默认流同步路径——cann 同步原语即同步）。
///
/// 跨设备 D2D：**本切片暂时返回 `Fatal`**（差异注记：ACL 8.5 用 kind 6/7
/// INNER/INTER 或 `aclrtMemcpyPeer`；无能力探测 API，留 011 T5 前实测扩展）。
pub fn copy(
    dst: &mut AscendMemRef<'_>,
    src: &AscendMemRef<'_>,
    bytes: usize,
    _stream: Option<&AscendStream>,
) -> Result<(), LaunchError> {
    let kind = kind_of(dst.device(), src.device())?;
    // CANN 侧当前不实现跨设备 D2D（无探测原语）；严格校验拦在 FFI 前
    if kind == MemcpyKind::D2D && dst.device() != src.device() && dst.device().is_some() {
        return Err(LaunchError::Fatal);
    }
    let policy = PeerPolicy {
        current_dev: 0, // 归属校验由 CANN per-thread set_device 语义保证（无 GetDevice API），
        // 本层以 allow_peer=true 跳过"当前设备"比对，跨设备 D2D 由上方显式拦截（011 差异注记）
        allow_peer: true,
    };
    mem_check::validate_memref(kind, &dst.end(), &src.end(), bytes, &policy)?;
    // SAFETY：指针来自对应端缓冲且经 mem_check（方向/边界）校验；绑定线程上下文由 cann 原语保证。
    unsafe {
        cann::buffer::memcpy(
            to_cann(kind),
            dst.ptr() as *mut std::ffi::c_void,
            src.ptr() as *const std::ffi::c_void,
            bytes,
        )
    }
    .map_err(|e| from_cann_error(&e))
}

/// 异步拷贝：返回事件作为同步凭证（011 镜像；事件 record 于给定流）。
pub fn copy_async(
    dst: &mut AscendMemRef<'_>,
    src: &AscendMemRef<'_>,
    bytes: usize,
    stream: &AscendStream,
) -> Result<AscendEvent, LaunchError> {
    let kind = kind_of(dst.device(), src.device())?;
    if kind == MemcpyKind::D2D && dst.device() != src.device() && dst.device().is_some() {
        return Err(LaunchError::Fatal);
    }
    let policy = PeerPolicy {
        current_dev: 0, // 归属校验由 CANN per-thread set_device 语义保证（无 GetDevice API），
        // 本层以 allow_peer=true 跳过"当前设备"比对，跨设备 D2D 由上方显式拦截（011 差异注记）
        allow_peer: true,
    };
    mem_check::validate_memref(kind, &dst.end(), &src.end(), bytes, &policy)?;
    // SAFETY：同 copy；额外要求流句柄有效（AscendStream::inner 保证）。
    unsafe {
        cann::buffer::memcpy_async(
            to_cann(kind),
            dst.ptr() as *mut std::ffi::c_void,
            src.ptr() as *const std::ffi::c_void,
            bytes,
            stream.inner(),
        )
    }
    .map_err(|e| from_cann_error(&e))?;
    // CANN 无轮询 query——同步凭证语义 = 事件 record + 使用者 synchronize
    let evt = AscendEvent::new()?;
    evt.record(Some(stream))?;
    Ok(evt)
}

#[cfg(all(test, not(feature = "ascend")))]
mod tests {
    use super::*;

    #[test]
    fn stub_alloc_returns_unavailable() {
        assert!(matches!(
            AscendDeviceBuffer::alloc(DeviceId::new(0), 1024),
            Err(LaunchError::Fatal)
        ));
        assert!(matches!(AscendHostBuffer::alloc(1024), Err(LaunchError::Fatal)));
    }
}

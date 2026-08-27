//! 设备上下文与设备信息（L1 T1；feature `cuda` 门控）。
//!
//! 实现基质：cudarc `runtime::result::device` 窄绑定（单一 `cudaError_t` 码体系）；
//! 错误一律经 [`crate::error::from_runtime_error`] 白名单分类（fail-closed）。
//!
//! 线程语义（CUDA 运行时）：`cudaSetDevice` 为 **per-thread** 绑定，
//! 每个需要设备访问的线程应在其起始处调用一次 [`CudaContext::init`]。

use cudarc::runtime::result::device;
use reinfer_core::DeviceId;

use crate::device_info::{DeviceInfo, dev_name_to_string, format_uuid};
use crate::error::{LaunchError, from_runtime_error};

/// 逻辑设备索引 → 上下文。持有 per-thread 绑定；本切片无子资源注册表，
/// 子资源（流/事件/缓冲）的释放次序由上层纪律保证（specs/009 plan D3）。
#[derive(Debug)]
pub struct CudaContext {
    dev: DeviceId,
}

impl CudaContext {
    /// 绑定当前线程到指定设备（每线程一次；文档约束线程亲和性）。
    pub fn init(dev: DeviceId) -> Result<Self, LaunchError> {
        device::set(dev.index() as i32).map_err(from_runtime_error)?;
        Ok(Self { dev })
    }

    /// 设备数量（CUDA runtime 报告）。
    pub fn device_count() -> Result<u32, LaunchError> {
        device::get_count().map(|n| n as u32).map_err(from_runtime_error)
    }

    /// 当前线程已绑定的设备（调试/校验用）。
    pub fn current_device() -> Result<DeviceId, LaunchError> {
        device::get().map(|n| DeviceId::new(n as u32)).map_err(from_runtime_error)
    }

    /// 设备信息（name / 算力 / 显存 / UUID）。
    ///
    /// `index` 越界 → 驱动返回 `cudaErrorInvalidDevice(101)`，
    /// 不在白名单 → `LaunchError::Fatal`（fail-closed）。
    pub fn device_info(index: u32) -> Result<DeviceInfo, LaunchError> {
        let p = device::get_device_prop(index as i32).map_err(from_runtime_error)?;
        Ok(DeviceInfo {
            index,
            name: dev_name_to_string(&p.name),
            major: p.major as u32,
            minor: p.minor as u32,
            total_mem: p.totalGlobalMem as u64,
            uuid: format_uuid(&p.uuid.bytes),
        })
    }

    /// 所在设备索引。
    #[inline]
    pub const fn device_id(&self) -> DeviceId {
        self.dev
    }
}

#[cfg(all(test, feature = "cuda"))]
mod ffi_tests {
    use super::*;

    /// 真机冒烟（RTX 5090 判定档；无 GPU 机器不启用 feature 即不编译）。
    #[test]
    fn device_info_smoke() {
        let count = CudaContext::device_count().expect("device_count");
        assert!(count >= 1);
        let ctx = CudaContext::init(DeviceId::new(0)).expect("init dev 0");
        assert_eq!(ctx.device_id(), DeviceId::new(0));
        let info = CudaContext::device_info(0).expect("device_info(0)");
        assert!(!info.name.is_empty());
        assert!(info.major >= 10, "compute capability {}.{}", info.major, info.minor);
        assert!(info.total_mem > 0);
        // uuid: 8-4-4-4-12 小写 hex
        let mut parts = info.uuid.splitn(6, '-');
        let lens: Vec<usize> = parts.by_ref().map(|p| p.len()).collect();
        assert_eq!(lens, vec![8, 4, 4, 4, 12]);
        assert!(info.uuid.chars().all(|c| c == '-' || c.is_ascii_hexdigit()));
        assert_eq!(CudaContext::current_device().expect("current_device"), DeviceId::new(0));
    }

    /// 越界 index → fail-closed Fatal（101 不在白名单）。
    #[test]
    fn device_info_out_of_range_is_fatal() {
        let count = CudaContext::device_count().expect("device_count");
        if count > 0 {
            let err = CudaContext::device_info(count).expect_err("must fail");
            assert_eq!(err, LaunchError::Fatal);
        }
    }
}

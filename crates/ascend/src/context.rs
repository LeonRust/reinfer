//! 昇腾设备上下文与基本信息（011 T3/M2）。
//!
//! 线程亲和性（ACL）：`aclrtSetDevice` 为 per-thread 绑定；用法与 CUDA 侧一致。
//! 设备属性（显存/算力/UUID）等 `DeviceProps` 归属 cann-rs（011 缺口 T2）——
//! 在补齐前本层只提供 SoC 名称（`aclrtGetSocName`，cann-rs 0001 已核实）。

use reinfer_core::DeviceId;

use crate::error::from_cann_error;
use reinfer_kernels::LaunchError;

/// CANN 上下文（`aclInit` RAII）。创建后本线程即可使用设备原语。
#[derive(Debug)]
pub struct AscendContext {
    _inner: cann::Context,
}

impl AscendContext {
    /// 初始化 CANN 运行环境（对应 `aclInit`；每进程一次，RAII 析构 `aclFinalize`）。
    pub fn new() -> Result<Self, LaunchError> {
        cann::Context::new().map(|inner| Self { _inner: inner }).map_err(|e| from_cann_error(&e))
    }

    /// 设备数量。
    pub fn device_count() -> Result<u32, LaunchError> {
        cann::device::device_count().map_err(|e| from_cann_error(&e))
    }

    /// 绑定当前线程到指定设备（每线程一次；文档约束线程亲和性）。
    pub fn set_device(dev: DeviceId) -> Result<(), LaunchError> {
        cann::device::set_device(dev.index()).map_err(|e| from_cann_error(&e))
    }

    /// 解绑当前线程设备（`aclrtResetDevice`）。
    pub fn reset_device(dev: DeviceId) -> Result<(), LaunchError> {
        cann::device::reset_device(dev.index()).map_err(|e| from_cann_error(&e))
    }

    /// 设备信息（011 M2 范围的 SoC 起步档；设备属性在 cann-rs `DeviceProps` 落地后扩展）。
    pub fn device_info(_index: u32) -> Result<DeviceInfo, LaunchError> {
        let soc_name = cann::device::soc_name().map_err(|e| from_cann_error(&e))?;
        Ok(DeviceInfo { soc_name })
    }
}

/// 昇腾设备信息（当前 = SoC 名称；字段随 `DeviceProps` 扩展）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// SoC 型号（如 `Ascend910B`）。
    pub soc_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_mode_returns_unavailable() {
        // 无 ffi（桩）：全部运行路径返回错误 → 分类 Fatal（fail-closed）
        assert!(matches!(AscendContext::new(), Err(LaunchError::Fatal)));
        assert!(matches!(AscendContext::device_count(), Err(LaunchError::Fatal)));
        assert!(matches!(AscendContext::set_device(DeviceId::new(0)), Err(LaunchError::Fatal)));
        assert!(matches!(AscendContext::device_info(0), Err(LaunchError::Fatal)));
    }
}

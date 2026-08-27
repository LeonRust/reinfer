//! CANN 错误 → 引擎 `LaunchError` 映射（011 L0 镜像）。
//!
//! 码段语义在 cann-rs（`cann::Error::is_oom/is_recoverable`，白名单 fail-closed，
//! 见 cann-rs 0001 Task 7/0002）；本层只做三分类归并，无 unsafe、无 feature 依赖。

use cann::Error;
use reinfer_kernels::LaunchError;

/// CANN 错误 → 引擎分类（fail-closed：未知码经 cann 分类为"不恢复"→ Fatal）。
#[inline]
pub fn from_cann_error(e: &Error) -> LaunchError {
    if e.is_oom() {
        LaunchError::Oom
    } else if e.is_recoverable() {
        LaunchError::Driver
    } else {
        LaunchError::Fatal
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(code: i32, msg: &str) -> Error {
        Error { code, message: msg.to_string() }
    }

    #[test]
    fn oom_maps_to_oom() {
        // 207001 = ACL_ERROR_RT_MEMORY_ALLOCATION 段（cann 白名单内）
        assert_eq!(from_cann_error(&err(207001, "alloc")), LaunchError::Oom);
    }

    #[test]
    fn recoverable_maps_to_driver() {
        // 真值（cann-rs acl_error_code.rs）：507000 INTERNAL_ERROR / 507033 DEV_SETUP_ERROR —— cann is_recoverable 白名单
        assert_eq!(from_cann_error(&err(507000, "internal")), LaunchError::Driver);
        assert_eq!(from_cann_error(&err(507033, "dev_setup")), LaunchError::Driver);
        // 非白名单的 507xxx 段码 → fail-closed Fatal
        assert_eq!(from_cann_error(&err(507999, "other")), LaunchError::Fatal);
    }

    #[test]
    fn unknown_maps_to_fatal() {
        // 桩模式 unavailable code=-1 与任何未分类码 → Fatal（fail-closed）
        assert_eq!(from_cann_error(&err(-1, "unavailable")), LaunchError::Fatal);
        assert_eq!(from_cann_error(&err(42, "unknown")), LaunchError::Fatal);
    }
}

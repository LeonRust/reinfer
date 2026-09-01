//! `cudaError_t → LaunchError` 白名单映射（003 T3）。
//!
//! 锚定：`specs/002-ascend-backend/plan.md` §Error mapping（fail-closed：
//! 未列入白名单的码一律 `Fatal`，与昇腾契约同构）。
//!
//! 码值与名称来源：cudarc 0.19.9 `runtime/sys/mod.rs`（`cudaError_t` 权威枚举）。
//! 注意：CUDA 12.x 将 `cudaErrorDeviceUnavailable` 更名为
//! **`cudaErrorDevicesUnavailable = 46`**（官方 ABI 值稳定，名称演进）。

// 经模块聚合导出（crate::error::LaunchError），并作为本地可见名使用
pub use reinfer_kernels::LaunchError;

/// `cudaError_t` 原始值（C ABI 稳定，此类型避免过度绑定 cudarc）。
pub type CudaErrorCode = i32;

/// 白名单：成功（仅供调用方前置检查，非错误分类对象）。
pub const CUDA_SUCCESS: CudaErrorCode = 0;
/// 白名单：内存分配失败 → `Oom`。
pub const CUDA_ERROR_MEMORY_ALLOCATION: CudaErrorCode = 2;
/// 白名单：无设备（驱动类，重建上下文可重试）。
pub const CUDA_ERROR_NO_DEVICE: CudaErrorCode = 100;
/// 白名单：设备不可用（驱动类；`cudaErrorDeviceUnavailable` 在 CUDA 12.x 更名为此，值 46 稳定）。
pub const CUDA_ERROR_DEVICES_UNAVAILABLE: CudaErrorCode = 46;
/// 白名单：非法地址（上下文类，重建后可重试）。
pub const CUDA_ERROR_ILLEGAL_ADDRESS: CudaErrorCode = 700;
/// 事件查询专用：本事件未完成（`cudaErrorNotReady`，CUDA 12 新值 600）。
///
/// **不在** `Oom/Driver` 白名单内——仅用于 [`event_query_status`] 特判（009 评审 A-M1/C-F3），
/// fail-closed 语义下绝不允许它被分类为 `Fatal`。
pub const CUDA_ERROR_NOT_READY: CudaErrorCode = 600;

/// 事件查询纯逻辑：`cudaEventQuery` 返回码 → 完成态。
///
/// - `SUCCESS(0)` → `Ok(true)`（已完成）；
/// - `NOT_READY(600)` → `Ok(false)`（未完成，非错误）；
/// - 其余 → 白名单分类 fail-closed。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))] // 无 feature 下由单测覆盖，真机路径 feature 下引用
pub(crate) fn event_query_status(rc: CudaErrorCode) -> Result<bool, LaunchError> {
    match rc {
        CUDA_SUCCESS => Ok(true),
        CUDA_ERROR_NOT_READY => Ok(false),
        other => Err(classify(other).unwrap_or(LaunchError::Fatal)),
    }
}

/// 对错误码做白名单分类；成功码返回 `None`（调用方契约：只对失败码调用）。
#[inline]
pub fn classify(code: CudaErrorCode) -> Option<LaunchError> {
    match code {
        CUDA_SUCCESS => None,
        CUDA_ERROR_MEMORY_ALLOCATION => Some(LaunchError::Oom),
        CUDA_ERROR_NO_DEVICE | CUDA_ERROR_DEVICES_UNAVAILABLE | CUDA_ERROR_ILLEGAL_ADDRESS => {
            Some(LaunchError::Driver)
        }
        // fail-closed：参数/未知/未来新码一律 Fatal（须显式加入白名单才放宽）
        other => {
            // TEMP-DIAG: 保留原始错误码便于部署侧诊断（BLOCKER-B 排查）；
            // 定位后删除本行或提升为 debug 日志。
            eprintln!("reinfer-cuda: classify fallback code={other:?}");
            Some(LaunchError::Fatal)
        }
    }
}

/// 以 `Result` 形式映射（成功码 → `Ok(())`）。
#[inline]
pub fn map_err(code: CudaErrorCode) -> Result<(), LaunchError> {
    match classify(code) {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// cudarc 运行时错误 → `LaunchError`（feature `cuda` 可用）。
///
/// 以（非 trait 实现）函数形式暴露：`LaunchError` 定义于 `reinfer-kernels`，
/// 无法在本地为外部类型实现 `From`（孤儿规则）。
#[cfg(feature = "cuda")]
#[inline]
pub fn from_runtime_error(e: cudarc::runtime::result::RuntimeError) -> LaunchError {
    // RuntimeError 包裹的 cudaError_t 为 C 枚举，直接取数值。
    classify(e.0 as CudaErrorCode).unwrap_or(LaunchError::Fatal)
}

#[cfg(all(test, feature = "cuda"))]
mod ffi_tests {
    use super::*;
    use cudarc::runtime::result::RuntimeError;
    use cudarc::runtime::sys::cudaError_t;

    #[test]
    fn runtime_error_conversion_roundtrip() {
        assert_eq!(
            from_runtime_error(RuntimeError(cudaError_t::cudaErrorMemoryAllocation)),
            LaunchError::Oom
        );
        assert_eq!(
            from_runtime_error(RuntimeError(cudaError_t::cudaErrorNoDevice)),
            LaunchError::Driver
        );
        assert_eq!(
            from_runtime_error(RuntimeError(cudaError_t::cudaErrorInvalidValue)),
            LaunchError::Fatal // 非白名单，fail-closed
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn success_is_not_an_error() {
        assert_eq!(classify(CUDA_SUCCESS), None);
        assert!(map_err(CUDA_SUCCESS).is_ok());
    }

    #[test]
    fn oom_whitelist() {
        assert_eq!(classify(CUDA_ERROR_MEMORY_ALLOCATION), Some(LaunchError::Oom));
    }

    #[test]
    fn driver_whitelist() {
        for code in
            [CUDA_ERROR_NO_DEVICE, CUDA_ERROR_DEVICES_UNAVAILABLE, CUDA_ERROR_ILLEGAL_ADDRESS]
        {
            assert_eq!(classify(code), Some(LaunchError::Driver), "code={code}");
        }
    }

    #[test]
    fn fail_closed_default_fatal() {
        // 非白名单：包含 CUDA_ERROR_INVALID_VALUE(1) 及未来新码。
        for code in [1, 3, 12345, i32::MAX] {
            assert_eq!(classify(code), Some(LaunchError::Fatal), "code={code}");
        }
    }

    #[test]
    fn map_err_roundtrip() {
        assert_eq!(map_err(CUDA_ERROR_MEMORY_ALLOCATION), Err(LaunchError::Oom));
        assert_eq!(map_err(99), Err(LaunchError::Fatal));
    }

    #[test]
    fn event_query_status_completed_is_true() {
        assert_eq!(event_query_status(CUDA_SUCCESS), Ok(true));
    }

    #[test]
    fn event_query_status_not_ready_is_false() {
        // 未完成（600）绝不落入分类器（否则变 Fatal）——T2 关键守卫
        assert_eq!(event_query_status(CUDA_ERROR_NOT_READY), Ok(false));
    }

    #[test]
    fn event_query_status_other_hits_whitelist() {
        // 真实错误（如 InvalidValue=1）→ 非白名单 → Fatal
        assert_eq!(event_query_status(1), Err(LaunchError::Fatal));
        // 白名单错误（如 2）也会命中分类
        assert_eq!(event_query_status(CUDA_ERROR_MEMORY_ALLOCATION), Err(LaunchError::Oom));
    }
}

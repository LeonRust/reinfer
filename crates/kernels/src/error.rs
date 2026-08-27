//! 内核执行失败分类（跨后端统一语义）。
//!
//! 锚定：`specs/002-ascend-backend/plan.md` §Error mapping —— 后端将厂商错误
//! 映射为本枚举；分类采用**白名单 fail-closed**：未显式列出的错误一律 `Fatal`。

use core::fmt;

/// 后端无关的启动/执行错误分类。
///
/// 各后端（CUDA/昇腾）对厂商错误码做白名单映射，引擎与调度层只消费本类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LaunchError {
    /// 内存/资源不足：上层走「驱逐 → 抢占 → 重试」链（宪法 §2.5）。
    Oom,
    /// 驱动/上下文类：上层应重建上下文并重试请求。
    Driver,
    /// 参数/实现/未知类：放弃该请求，进程保持存活（fail-closed 默认）。
    Fatal,
}

impl LaunchError {
    /// 是否可安全重试（Oom 驱逐后、Driver 重建后）。
    #[inline]
    pub const fn retryable(self) -> bool {
        matches!(self, Self::Oom | Self::Driver)
    }
}

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Oom => "kernel launch failed: out of resources",
            Self::Driver => "kernel launch failed: device/driver error",
            Self::Fatal => "kernel launch failed: invalid or unknown error",
        })
    }
}

impl std::error::Error for LaunchError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_only_oom_and_driver() {
        assert!(LaunchError::Oom.retryable());
        assert!(LaunchError::Driver.retryable());
        assert!(!LaunchError::Fatal.retryable());
    }

    #[test]
    fn display_is_compact() {
        for e in [LaunchError::Oom, LaunchError::Driver, LaunchError::Fatal] {
            assert!(e.to_string().starts_with("kernel launch failed"));
        }
    }
}

//! 磁盘/IO 错误 → `LaunchError` 分类（fail-closed）。
//!
//! 契约（012 spec 错误面行）：磁盘 IO/权限/锁超时 → `Fatal`；磁盘满 → `Oom`
//! （可重试语义）；编译失败 → `Fatal` 附带 nvcc stderr 尾部（`compile.rs` 侧）。

use reinfer_kernels::LaunchError;

/// io::Error → 分类。仅 ENOSPC（28, Linux errno）判 `Oom`，其余 fail-closed `Fatal`。
#[inline]
pub(crate) fn fs_err(e: &std::io::Error) -> LaunchError {
    if e.raw_os_error() == Some(28) { LaunchError::Oom } else { LaunchError::Fatal }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enospc_maps_to_oom() {
        let e = std::io::Error::from_raw_os_error(28);
        assert_eq!(fs_err(&e), LaunchError::Oom);
    }

    #[test]
    fn other_io_maps_to_fatal() {
        let e = std::io::Error::from_raw_os_error(13); // EACCES
        assert_eq!(fs_err(&e), LaunchError::Fatal);
    }
}

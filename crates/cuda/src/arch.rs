//! 目标架构解析（012 通用性修正：**无静默默认**）。
//!
//! 优先级：显式 `REINFER_CUDA_ARCH`（用户覆盖，如 `sm_120a`/`sm_86`）
//! → 设备实测 `sm_{major}{minor}`（无 `a` 后缀——base 档是向下兼容的
//! 通用目标；本引擎内核不使用 arch-specific 指令，无需 `-a`）。
//! 不允许默认值指向任何特定硬件（评审 C-F 修复：不得为开发机写特判）。

use crate::context::CudaContext;
use reinfer_kernels::LaunchError;

/// 架构解析环境变量。
pub const ARCH_ENV: &str = "REINFER_CUDA_ARCH";

/// 由设备算力构造规范串（`cc 8.6 → sm_86`、`cc 12.0 → sm_120`）。
pub fn arch_from_cc(major: u32, minor: u32) -> String {
    format!("sm_{major}{minor}")
}

/// 解析目标架构：env 优先，否则读设备 0 的算力。
pub fn resolve_arch() -> Result<String, LaunchError> {
    if let Some(a) = std::env::var(ARCH_ENV).ok().filter(|s| !s.is_empty()) {
        return Ok(a);
    }
    let info = CudaContext::device_info(0)?;
    Ok(arch_from_cc(info.major, info.minor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cc_to_arch_forms() {
        assert_eq!(arch_from_cc(8, 6), "sm_86");
        assert_eq!(arch_from_cc(12, 0), "sm_120");
        assert_eq!(arch_from_cc(9, 0), "sm_90");
    }
}

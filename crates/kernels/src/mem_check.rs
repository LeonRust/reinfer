//! 内存拷贝校验的共享纯逻辑（后端无关：CUDA / 昇腾 CANN 共用）。
//!
//! 边界条约判定："换 CUDA 仍成立 → reinfer 共享资产"——方向匹配 / 边界（含溢出防护）/
//! 设备归属与跨设备策略属于引擎语义，各后端 PEERER 原语不同但校验一致。
//! 无 cudarc / ACL 依赖、无 feature 依赖、无 unsafe。

use crate::error::LaunchError;

/// 拷贝方向（对应 `cudaMemcpyKind` / ACL `aclrtMemcpyKind` 语义的受控子集）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyKind {
    /// Host → Device
    H2D,
    /// Device → Host
    D2H,
    /// Device → Device（同设备；跨设备由各后端决定：peer 探测或运行时分类）
    D2D,
}

/// 拷贝一端的目标描述（dst/src 各一）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemRefEnd {
    /// 起始偏移（字节）。
    pub offset: usize,
    /// 缓冲区长度（字节）。
    pub len: usize,
    /// `None` = Host 侧；`Some(idx)` = Device 侧。
    pub dev: Option<u32>,
}

/// 当前线程设备与跨设备策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerPolicy {
    /// 本线程绑定的设备。
    pub current_dev: u32,
    /// `false`（严格模式）时跨设备 D2D 直接拒绝；`true`（peer 模式）时交由后端能力探测，
    /// 本层只负责方向与边界。
    pub allow_peer: bool,
}

/// 校验规则（全部返回 `LaunchError::Fatal`——参数/编程错误类，fail-closed）：
///
/// 1. **方向匹配**：`H2D` 要求 dst=Device 且 src=Host；`D2H` 反之；`D2D` 要求两端均 Device；
/// 2. **边界**：`offset + bytes <= len`（逐端，溢出视为非法）；
/// 3. **设备归属**：任一 Device 端必须等于 `policy.current_dev`（本线程绑定设备）——
///    仅在非 peer 模式检查；
/// 4. **跨设备 D2D**：`policy.allow_peer=false` 时直接 `Err(Fatal)`。
pub fn validate_memref(
    kind: MemcpyKind,
    dst: &MemRefEnd,
    src: &MemRefEnd,
    bytes: usize,
    policy: &PeerPolicy,
) -> Result<(), LaunchError> {
    use MemcpyKind::*;
    match kind {
        H2D if dst.dev.is_none() || src.dev.is_some() => return Err(LaunchError::Fatal),
        D2H if dst.dev.is_some() || src.dev.is_none() => return Err(LaunchError::Fatal),
        D2D if dst.dev.is_none() || src.dev.is_none() => return Err(LaunchError::Fatal),
        _ => {}
    }

    // 边界（含溢出防护）
    for end in [dst, src] {
        let Some(oend) = end.offset.checked_add(bytes) else {
            return Err(LaunchError::Fatal);
        };
        if oend > end.len {
            return Err(LaunchError::Fatal);
        }
    }

    // 设备归属与跨设备策略
    if kind == D2D && policy.allow_peer {
        return Ok(());
    }
    if let Some(d) = dst.dev
        && d != policy.current_dev
    {
        return Err(LaunchError::Fatal);
    }
    if let Some(s) = src.dev
        && s != policy.current_dev
    {
        return Err(LaunchError::Fatal);
    }
    if kind == D2D && !policy.allow_peer && dst.dev != src.dev {
        return Err(LaunchError::Fatal);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CUR: u32 = 0;
    const OTHER: u32 = 1;

    fn end(off: usize, len: usize, dev: Option<u32>) -> MemRefEnd {
        MemRefEnd { offset: off, len, dev }
    }

    fn policy(allow_peer: bool) -> PeerPolicy {
        PeerPolicy { current_dev: CUR, allow_peer }
    }

    fn ok(kind: MemcpyKind, dst: Option<u32>, src: Option<u32>) -> bool {
        validate_memref(kind, &end(0, 16, dst), &end(0, 16, src), 16, &policy(false)).is_ok()
    }

    #[test]
    fn direction_matching() {
        assert!(ok(MemcpyKind::H2D, Some(CUR), None));
        assert!(ok(MemcpyKind::D2H, None, Some(CUR)));
        assert!(ok(MemcpyKind::D2D, Some(CUR), Some(CUR)));
        assert!(!ok(MemcpyKind::H2D, None, None));
        assert!(!ok(MemcpyKind::H2D, Some(CUR), Some(CUR)));
        assert!(!ok(MemcpyKind::D2H, Some(CUR), None));
        assert!(!ok(MemcpyKind::D2D, None, Some(CUR)));
    }

    #[test]
    fn bounds_and_overflow() {
        assert!(
            validate_memref(
                MemcpyKind::D2D,
                &end(8, 16, Some(CUR)),
                &end(8, 16, Some(CUR)),
                8,
                &policy(false)
            )
            .is_ok()
        );
        assert!(
            validate_memref(
                MemcpyKind::D2D,
                &end(9, 16, Some(CUR)),
                &end(0, 16, Some(CUR)),
                8,
                &policy(false)
            )
            .is_err()
        );
        assert!(
            validate_memref(
                MemcpyKind::D2D,
                &end(usize::MAX, 16, Some(CUR)),
                &end(0, 16, Some(CUR)),
                16,
                &policy(false)
            )
            .is_err()
        );
    }

    #[test]
    fn device_ownership_and_cross_device() {
        assert!(
            validate_memref(
                MemcpyKind::D2D,
                &end(0, 16, Some(OTHER)),
                &end(0, 16, Some(OTHER)),
                1,
                &policy(false)
            )
            .is_err()
        );
        assert!(
            validate_memref(
                MemcpyKind::D2D,
                &end(0, 16, Some(CUR)),
                &end(0, 16, Some(OTHER)),
                1,
                &policy(false)
            )
            .is_err()
        );
        // peer 模式：跨设备在 validate 层放行（能力探测在后端 copy 内）
        assert!(
            validate_memref(
                MemcpyKind::D2D,
                &end(0, 16, Some(CUR)),
                &end(0, 16, Some(OTHER)),
                1,
                &policy(true)
            )
            .is_ok()
        );
    }

    #[test]
    fn zero_length_is_legal() {
        assert!(
            validate_memref(
                MemcpyKind::D2D,
                &end(16, 16, Some(CUR)),
                &end(16, 16, Some(CUR)),
                0,
                &policy(false)
            )
            .is_ok()
        );
    }
}

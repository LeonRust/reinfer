//! MemRef 拷贝校验的纯逻辑（无 cudarc 依赖、无 feature 依赖、无 unsafe）。
//!
//! 供 [`crate::buffer`] 的 `copy`/`copy_async` 在 FFI 前调用，保证
//! "方向 / 边界 / 设备归属"全部显式判定（009 plan D4；无 GPU 单测载体）。
//!
//! 说明：校验入口当前仅被单测使用（T4 的 copy 接入后由调用方派生），
//! 因此若干 item 标注 `#[allow(dead_code)]`；T4 接入后此项自动失效，人工移除即可。

use crate::error::LaunchError;

/// 拷贝方向（对应 `cudaMemcpyKind` 语义的受控子集）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemcpyKind {
    /// Host → Device
    H2D,
    /// Device → Host
    D2H,
    /// Device → Device（同设备；跨设备见 T4 的 peer 探测）
    D2D,
}

/// 拷贝一端的目标描述（dst/src 各一）。
#[allow(dead_code)]
pub(crate) struct MemRefEnd {
    pub offset: usize,
    pub len: usize,
    /// `None` = Host 侧；`Some(idx)` = Device 侧。
    pub dev: Option<u32>,
}

/// 当前线程设备与跨设备策略。
#[allow(dead_code)]
pub(crate) struct PeerPolicy {
    /// 本线程绑定的设备。
    pub current_dev: u32,
    /// `false`（本切片）时跨设备 D2D 直接拒绝；T4 引入
    /// `cudaDeviceCanAccessPeer` 探测后放开（009 评审 B#8）。
    pub allow_peer: bool,
}

/// 校验规则（全部返回 `LaunchError::Fatal`——参数/编程错误类，fail-closed）：
///
/// 1. **方向匹配**：`H2D` 要求 dst=Device 且 src=Host；`D2H` 反之；`D2D` 要求两端均 Device；
/// 2. **边界**：`offset + bytes <= len`（逐端，溢出视为非法）；
/// 3. **设备归属**：任一 Device 端必须等于 `policy.current_dev`（本线程绑定设备）；
/// 4. **跨设备 D2D**：`policy.allow_peer=false`（本切片）时直接 `Err(Fatal)`。
#[allow(dead_code)]
pub(crate) fn validate_memref(
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
        // 同设备 D2D 保持快路径；跨设备在 T4 交由 peer 探测决定
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
        assert!(!ok(MemcpyKind::H2D, None, None)); // dst 必须是 device
        assert!(!ok(MemcpyKind::H2D, Some(CUR), Some(CUR))); // src 必须是 host
        assert!(!ok(MemcpyKind::D2H, Some(CUR), None)); // dst 必须是 host
        assert!(!ok(MemcpyKind::D2D, None, Some(CUR))); // 两端必须 device
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
        // 越界
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
        // 溢出（offset+bytes 超过 usize 上限）
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
        // 另一设备（≠ current）→ 归属不符 → Fatal
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
        // 伪造两个不同 DeviceId（009 T3 验收）：跨设备 D2D → Err(Fatal)
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
        // allow_peer 目前只对同设备组合有实际路径（T4 引入探测后跨设备放宽）
        assert!(
            validate_memref(
                MemcpyKind::D2D,
                &end(0, 16, Some(CUR)),
                &end(0, 16, Some(CUR)),
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

//! 设备标识（后端无关）。
//!
//! `DeviceId` 是逻辑设备编号（0 基），跨 CUDA / 昇腾 / CPU 后端统一使用，
//! 避免各后端以裸 `u32` 互传导致归属混淆（见 specs/009-cuda-runtime-base，A-L2 裁决）。

use core::fmt;

/// 逻辑设备编号（0 基；与具体硬件映射由各后端负责）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceId(u32);

impl DeviceId {
    /// 从索引构造（0 基逻辑编号）。
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// 取回 0 基索引。
    #[inline]
    pub const fn index(self) -> u32 {
        self.0
    }
}

impl From<u32> for DeviceId {
    #[inline]
    fn from(index: u32) -> Self {
        Self::new(index)
    }
}

impl From<DeviceId> for u32 {
    #[inline]
    fn from(dev: DeviceId) -> Self {
        dev.index()
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "device#{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_index_roundtrip() {
        for idx in [0u32, 1, 7, u32::MAX] {
            let d = DeviceId::new(idx);
            assert_eq!(d.index(), idx);
            assert_eq!(d.to_string(), format!("device#{idx}"));
        }
    }

    #[test]
    fn from_conversions_roundtrip() {
        let d: DeviceId = 42u32.into();
        assert_eq!(d, DeviceId::new(42));
        let back: u32 = d.into();
        assert_eq!(back, 42);
    }

    #[test]
    fn order_and_eq() {
        let (a, b) = (DeviceId::new(1), DeviceId::new(2));
        assert_eq!(a, DeviceId::new(1));
        assert!(a < b);
        assert_ne!(a, b);
    }

    #[test]
    fn debug_is_stable() {
        let d = DeviceId::new(3);
        assert_eq!(format!("{d:?}"), "DeviceId(3)");
    }
}

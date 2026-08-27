//! 最小 dtype 枚举（001 的 P0-01 子集——仅 OpConfig/差分矩阵所需；量化族延后）。

/// 张量/内核数值类型（后端无关；映射规则在平台 crate）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 16-bit half (IEEE 754 半精度)。
    F16,
    /// 8-bit float (E4M3 起语义)。
    F8,
    /// 16-bit bfloat16。
    BF16,
    /// 32-bit float。
    F32,
}

impl DType {
    /// 稳定显示名（tune/诊断/差分矩阵键）。
    pub const fn name(self) -> &'static str {
        match self {
            DType::F16 => "f16",
            DType::F8 => "f8",
            DType::BF16 => "bf16",
            DType::F32 => "f32",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_stable() {
        assert_eq!(DType::F32.name(), "f32");
        assert_eq!(DType::F16.name(), "f16");
        assert_eq!(DType::BF16.name(), "bf16");
        assert_eq!(DType::F8.name(), "f8");
    }
}

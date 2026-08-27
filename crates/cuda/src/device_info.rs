//! 设备信息纯数据与格式化（无 cudarc 依赖、无 feature 依赖、无 unsafe）。
//!
//! 该模块保持「无 CUDA 依赖」，以便无 GPU 环境也可单测
//! （009 无 GPU 具名测试集：`format_uuid` 全模式 + `DeviceInfo` Debug/Clone）。

use core::ffi::c_char;

/// 设备信息（纯数据，无底层依赖；可 Clone/Debug 无 GPU 单测）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// 设备索引（0 基）。
    pub index: u32,
    /// 设备名称（驱动报告，UTF-8）。
    pub name: String,
    /// 计算能力主版本。
    pub major: u32,
    /// 计算能力次版本。
    pub minor: u32,
    /// 总显存（字节）。
    pub total_mem: u64,
    /// 设备 UUID（8-4-4-4-12 hex，小写）。
    pub uuid: String,
}

/// `[c_char; 256]`（NUL 结尾）→ UTF-8 字符串（截断于首个 NUL；纯字节处理，无 unsafe）。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))] // 无 feature 构建下仅作单测载体，真机路径由 feature 门控的 context.rs 引用
pub(crate) fn dev_name_to_string(name: &[c_char; 256]) -> String {
    let len = name.iter().position(|&c| c == 0).unwrap_or(name.len());
    let bytes: Vec<u8> = name[..len].iter().map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// UUID 字节（cudarc `CUuuid_st.bytes`，`c_char` 即 i8）→ 8-4-4-4-12 小写 hex。
#[cfg_attr(not(feature = "cuda"), allow(dead_code))] // 同上：无 feature 时由单测覆盖、真机路径 feature 下引用
pub(crate) fn format_uuid(bytes: &[c_char; 16]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(36);
    for (i, b) in bytes.iter().enumerate() {
        let b = *b as u8; // c_char 位模式 → u8（含 i8 负值，如 0x80）
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
        if matches!(i, 3 | 5 | 7 | 9) {
            out.push('-');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes16(v: [u8; 16]) -> [c_char; 16] {
        let mut out = [0i8; 16];
        for (i, b) in v.iter().enumerate() {
            out[i] = *b as i8;
        }
        out
    }

    #[test]
    fn uuid_format_zero() {
        assert_eq!(format_uuid(&bytes16([0; 16])), "00000000-0000-0000-0000-000000000000");
    }

    #[test]
    fn uuid_format_pattern_and_dashes() {
        let v: [u8; 16] = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(format_uuid(&bytes16(v)), "12345678-9abc-def0-0102-030405060708");
    }

    #[test]
    fn uuid_format_high_bit_byte() {
        // 0x80/0xff 作为 i8 为负值，位模式转回必须保持
        let v: [u8; 16] = [0x80, 0x7f, 0x00, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(format_uuid(&bytes16(v)), "807f00ff-0000-0000-0000-000000000000");
    }

    #[test]
    fn device_info_debug_and_clone() {
        let info = DeviceInfo {
            index: 0,
            name: "NVIDIA GeForce RTX 5090 Laptop GPU".into(),
            major: 12,
            minor: 0,
            total_mem: 24 * 1024 * 1024 * 1024,
            uuid: "807f00ff-0000-0000-0000-000000000000".into(),
        };
        let dbg = format!("{info:?}");
        assert!(dbg.contains("5090"));
        assert!(dbg.contains("807f00ff"));
        assert_eq!(info.clone(), info);
    }
}

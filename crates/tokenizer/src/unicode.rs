//! Byte-level BPE 的 byte ↔ unicode 映射（GPT-2 `bytes_to_unicode` 原表）。
//!
//! 锚点：llama.cpp f280b2698 `src/unicode.cpp` `unicode_byte_to_utf8_map` /
//! `unicode_utf8_to_byte_map`（行 148 / 172）：
//! - 0x21..=0x7E、0xA1..=0xAC、0xAE..=0xFF 的 byte → 自身 code point；
//! - 其余 68 个 byte（0x00..=0x20、0x7F..=0x80、0x81..=0xA0、0xAD）→ U+0100+n，
//!   n 按 byte 值升序（0x20 → 'Ġ' U+0120）。

use std::collections::HashMap;

/// 缺失 byte（0x00..=0x20、0x7F..=0x80、0x81..=0xA0、0xAD）在缺失集合中
/// 按 byte 值升序的序号 n（GPT-2 映射 = U+0100 + n）。
fn missing_rank(b: u8) -> u32 {
    match b {
        0x00..=0x20 => u32::from(b),
        0x7F..=0x80 => 33 + u32::from(b - 0x7F),
        0x81..=0xA0 => 35 + u32::from(b - 0x81),
        0xAD => 67,
        _ => unreachable!("非缺失 byte"),
    }
}

/// byte → unicode char（长度 256，下标即 byte 值）。
pub(crate) fn byte_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    for (b, slot) in map.iter_mut().enumerate() {
        let b = b as u8;
        *slot = match b {
            0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF => b as char,
            _ => char::from_u32(0x100 + missing_rank(b)).expect("0x100+n 恒为合法码点"),
        };
    }
    map
}

/// unicode char → byte（`byte_to_unicode` 的逆映射，全 256 项）。
pub(crate) fn unicode_to_byte() -> HashMap<char, u8> {
    byte_to_unicode().into_iter().enumerate().map(|(b, c)| (c, b as u8)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_matches_gpt2_reference() {
        let map = byte_to_unicode();
        // 直接映射区
        assert_eq!(map[b'!' as usize], '!'); // 0x21
        assert_eq!(map[b'~' as usize], '~'); // 0x7E
        assert_eq!(map[0xA1], '¡');
        assert_eq!(map[0xAC], '¬');
        assert_eq!(map[0xAE], '®');
        assert_eq!(map[0xFF], 'ÿ');
        // 间接映射区：0x20 → 'Ġ' U+0120，0x00 → U+0100，0xAD → U+0143
        assert_eq!(map[0x20], 'Ġ');
        assert_eq!(map[0x00], '\u{0100}');
        assert_eq!(map[0x7F], '\u{0121}'); // n=33（0x00..0x20 已占 33 个）
        assert_eq!(map[0x80], '\u{0122}'); // n=34
        assert_eq!(map[0xAD], '\u{0143}'); // n=67
        // 逆映射完备
        let inv = unicode_to_byte();
        assert_eq!(inv.len(), 256);
        for (b, c) in map.into_iter().enumerate() {
            assert_eq!(inv[&c], b as u8);
        }
    }
}

//! 增量 UTF-8 解码器（分块自洽：逐块 take 拼接 ≡ 一次性解码）。
//!
//! 语义对齐 `std::str::from_utf8_lossy`：每个最大无效序列输出一个 U+FFFD。
//! - `push` 接收任意字节（token piece 字节、跨 chunk 的截断序列等）；
//! - `take` 只发出**完整** UTF-8 序列：完整无效序列立即换 U+FFFD 发出；
//!   不完整前缀（有效前缀 + 悬尾）留在缓冲，等后续字节；
//! - `flush` 把缓冲尾部的不完整序列折算为一个 U+FFFD 并清空；
//!   连续 flush 幂等（空缓冲 → 空输出）。

/// 流式 UTF-8 增量解码器：逐 chunk 压入字节，可取走所有完整序列。
#[derive(Debug, Default)]
pub struct IncrementalDecoder {
    pending: Vec<u8>,
}

impl IncrementalDecoder {
    /// 新建空解码器。
    pub fn new() -> Self {
        Self::default()
    }

    /// 压入一段字节；完整序列可立即经 [`Self::take`] 取走。
    pub fn push(&mut self, bytes: &[u8]) {
        self.pending.extend_from_slice(bytes);
    }

    /// 取出当前所有**完整**序列的字符串；不完整尾部保留在缓冲内。
    pub fn take(&mut self) -> String {
        let mut out = String::new();
        let mut start = 0usize;
        loop {
            match std::str::from_utf8(&self.pending[start..]) {
                Ok(rest) => {
                    out.push_str(rest);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    match std::str::from_utf8(&self.pending[start..start + valid]) {
                        Ok(part) => out.push_str(part),
                        Err(_) => unreachable!("prefix of a valid-prefix is valid"),
                    }
                    match e.error_len() {
                        Some(n) => {
                            // 完整无效序列：一个 U+FFFD，跳过 n 字节后继续。
                            out.push('\u{FFFD}');
                            start += valid + n;
                        }
                        None => {
                            // 不完整序列：已发出的有效前缀清掉，悬尾保留。
                            self.pending.drain(..start + valid);
                            return out;
                        }
                    }
                }
            }
        }
    }

    /// 冲刷：尾部不完整序列 → 一个 U+FFFD，缓冲清空。
    pub fn flush(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let mut out = self.take();
        if !self.pending.is_empty() {
            out.push('\u{FFFD}');
            self.pending.clear();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_str(d: &mut IncrementalDecoder, s: &str) {
        d.push(s.as_bytes());
    }

    #[test]
    fn complete_sequences_pass_through() {
        let mut d = IncrementalDecoder::new();
        push_str(&mut d, "你");
        push_str(&mut d, "好");
        assert_eq!(d.take(), "你好");
        assert_eq!(d.take(), "");
    }

    #[test]
    fn split_at_chunk_boundary() {
        // '你' = E4 BD A0，'好' = E5 A5 BD：每个 chunk 都在序列中间切断。
        let mut d = IncrementalDecoder::new();
        d.push(&[0xE4]);
        assert_eq!(d.take(), ""); // 悬尾
        d.push(&[0xBD, 0xA0, 0xE5]);
        assert_eq!(d.take(), "你"); // 完整序列发出，0xE5 仍悬尾
        d.push(&[0xA5, 0xBD]);
        assert_eq!(d.take(), "好");
    }

    #[test]
    fn invalid_bytes_yield_one_replacement_each() {
        let mut d = IncrementalDecoder::new();
        d.push(&[b'a', 0xFF, b'b', 0x80, 0x80]);
        // 0xFF 单独无效 → 一个 U+FFFD；0x80 0x80 连续无效 → 每个各一个 U+FFFD
        assert_eq!(d.take(), "a\u{FFFD}b\u{FFFD}\u{FFFD}");
    }

    #[test]
    fn incomplete_tail_flushes_to_one_replacement() {
        let mut d = IncrementalDecoder::new();
        d.push(&[b'x', 0xF0, 0x9F]);
        assert_eq!(d.take(), "x"); // 0xF0 0x9F 悬尾
        assert_eq!(d.flush(), "\u{FFFD}");
        assert_eq!(d.flush(), ""); // 幂等
        // 冲刷后可继续使用
        push_str(&mut d, "ok");
        assert_eq!(d.take(), "ok");
    }

    #[test]
    fn lossy_equivalence_across_arbitrary_splits() {
        // 任意 chunk 切分下，逐块 take+flush 拼接 ≡ 一次性 lossy 解码。
        let corpus = "Hello, 世界！🙂 \u{1F600}a\u{FFFD}b\u{FF}\u{80} 中文 mixed!";
        let bytes = corpus.as_bytes();
        let reference = String::from_utf8_lossy(bytes).into_owned();
        for split in 0..=bytes.len() {
            let mut d = IncrementalDecoder::new();
            let mut out = String::new();
            d.push(&bytes[..split]);
            out.push_str(&d.take());
            d.push(&bytes[split..]);
            out.push_str(&d.take());
            out.push_str(&d.flush());
            assert_eq!(out, reference, "split at {split}");
        }
    }
}

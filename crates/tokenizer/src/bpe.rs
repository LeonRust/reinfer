//! Byte-level BPE tokenizer（锚点：llama.cpp f280b2698 `llama-vocab.cpp`）。
//!
//! 语义对齐（逐项核对源文件）：
//! - special 分段：`tokenizer_st_partition`（行 3208）——special 集合 =
//!   类型 CONTROL|USER_DEFINED|UNKNOWN（行 2996），按文本长度降序处理
//!   （行 3000）；USER_DEFINED 恒 pre-tokenize，CONTROL/UNKNOWN 仅
//!   `parse_special` 时（行 3215-3221）。等长 special 的次序 llama.cpp 用
//!   `std::sort`（不稳定，未定义行为），本实现取 id 升序（仅影响互相重叠
//!   的等长 special；Qwen2 系 vocab 不存在这种条目）。
//! - 符号切分：逐 unicode 字符（`unicode_len_utf8`，`src/unicode.cpp` 行 16）。
//! - 合并：优先队列取「rank 最小、left 最左」（`llm_bigram_bpe` 比较器
//!   行 263-268）；旧 bigram 以位置快照不匹配跳过（等价
//!   `left+right != bigram.text` staleness 检查，行 660-663）；合并后仅
//!   更新 (prev,left) 与 (left,next) 两个邻接 bigram（行 671-672）。
//! - 终化：符号文本命中 vocab 直接取 id；未命中时 byte-level 模型按
//!   **每个 byte** 做 1 字节字符串查找（行 684-689），找不到静默跳过
//!   （BPE encode 不插入 UNK，与 llama.cpp 一致）。
//!
//! 已记录 deviation（014 plan 事实修正）：
//! - 不做 regex 预分段（本 slice directive：跳过；整段文本视为一个 word，
//!   `unicode_byte_encoding_process` 对整段逐 byte 编码）。
//! - QWEN2 无「▁→空格」：`escape_whitespaces` 对 BPE 默认 false
//!   （行 2131），QWEN2 pre 不重开（行 2229-2234）；Qwen2.5 真实 vocab
//!   以 'Ġ'（U+0120）表示空格、无任何 ▁ 条目。
//! - merges 两侧均非 vocab piece 的条目（HF 格式实际不会出现）不进 id 键
//!   索引，退化文本查找（语义等价）。

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

use reinfer_gguf::{ArrayValue, MetaValue, ModelMeta};

use crate::decode::IncrementalDecoder;
use crate::unicode::{byte_to_unicode, unicode_to_byte};
use crate::{TokenizerError, meta_bool, meta_special_id};

/// GGUF `tokenizer.ggml.token_type` 取值（1 基，与 llama.cpp
/// `llama_token_type` 一致：UNDEFINED=0, NORMAL=1, UNKNOWN=2, CONTROL=3,
/// USER_DEFINED=4, UNUSED=5, BYTE=6；SentencePiece type 同构）。
const TYPE_UNDEFINED: u32 = 0;
const TYPE_NORMAL: u32 = 1;
const TYPE_UNKNOWN: u32 = 2;
const TYPE_CONTROL: u32 = 3;
const TYPE_USER_DEFINED: u32 = 4;
const TYPE_UNUSED: u32 = 5;
const TYPE_BYTE: u32 = 6;

/// 合并链上的一个符号（编码文本的字符区间 + 链指针）。
struct Sym {
    start: usize,
    len: usize,
    prev: isize,
    next: isize,
}

/// 一条合并规则的预计算项（id 键索引只需 rank；终化时以文本查 id）。
#[derive(Debug)]
struct MergeEntry {
    rank: u32,
}

/// 合并优先队列元素（min rank，leftmost tie-break）。
#[derive(PartialEq, Eq)]
struct Bigram {
    rank: u32,
    left: usize,
    right: usize,
    /// push 时的位置快照，用于 staleness 检查。
    l_start: usize,
    l_len: usize,
    r_start: usize,
    r_len: usize,
}

impl Ord for Bigram {
    /// 反转比较实现最小堆：rank 升序，其次 left 升序。
    fn cmp(&self, other: &Self) -> Ordering {
        other.rank.cmp(&self.rank).then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// 分段结果：原始文本片段或特殊 token。
enum Fragment<'a> {
    Raw(&'a str),
    Token(u32),
}

/// BPE tokenizer（byte-level；自 GGUF 元数据构造）。
#[derive(Debug)]
pub struct BpeTokenizer {
    pieces: Vec<String>,
    types: Vec<u32>,
    text_to_id: HashMap<String, u32>,
    unicode_to_byte: HashMap<char, u8>,
    /// 字节编码字符 → id（byte 兜底查找；vocab 缺该字符则不入表）。
    byte_to_id: HashMap<u8, u32>,
    /// 两侧均为 vocab piece 的 merge：id 键快速索引。
    merges_by_id: HashMap<(u32, u32), MergeEntry>,
    /// 其余 merge：外层 left 文本 → 内层 right 文本 → rank。
    merges_by_text: HashMap<String, HashMap<String, u32>>,
    /// special ids（文本长度降序、id 升序）。
    specials: Vec<u32>,
    unk: Option<u32>,
    bos: Option<u32>,
    eos: Option<u32>,
    add_bos: bool,
    add_eos: bool,
}

impl BpeTokenizer {
    /// 从 GGUF 元数据构造（`tokenizer.ggml.*` 键）。
    pub fn from_meta(meta: &ModelMeta) -> Result<Self, TokenizerError> {
        let model = meta.meta_str("tokenizer.ggml.model")?.ok_or_else(|| {
            TokenizerError::MissingMetadata { key: "tokenizer.ggml.model".into() }
        })?;
        if model != "gpt2" {
            return Err(TokenizerError::UnsupportedModel { model: model.to_string() });
        }

        let pieces: Vec<String> = meta
            .meta_array_str("tokenizer.ggml.tokens")?
            .ok_or_else(|| TokenizerError::MissingMetadata { key: "tokenizer.ggml.tokens".into() })?
            .to_vec();

        let types = match meta.get("tokenizer.ggml.token_type") {
            None => vec![TYPE_NORMAL; pieces.len()],
            Some(MetaValue::Array(ArrayValue::U32(xs))) => {
                if xs.len() < pieces.len() {
                    return Err(TokenizerError::InvalidMetadata {
                        key: "tokenizer.ggml.token_type".into(),
                        why: format!("len {} < tokens len {}", xs.len(), pieces.len()),
                    });
                }
                xs[..pieces.len()].to_vec()
            }
            Some(_) => {
                return Err(TokenizerError::InvalidMetadata {
                    key: "tokenizer.ggml.token_type".into(),
                    why: "expected u32 array".into(),
                });
            }
        };

        let merges: Vec<String> = meta
            .meta_array_str("tokenizer.ggml.merges")?
            .ok_or_else(|| TokenizerError::MissingMetadata { key: "tokenizer.ggml.merges".into() })?
            .to_vec();

        let bos = meta_special_id(meta, "tokenizer.ggml.bos_token_id", "<s>", &pieces)?;
        let eos = meta_special_id(meta, "tokenizer.ggml.eos_token_id", "</s>", &pieces)?;
        let unk = meta_special_id(meta, "tokenizer.ggml.unknown_token_id", "<unk>", &pieces)?;

        let add_bos = meta_bool(meta, "tokenizer.ggml.add_bos_token", false)?;
        let add_eos = meta_bool(meta, "tokenizer.ggml.add_eos_token", false)?;

        Self::from_parts(pieces, types, merges, bos, eos, unk, add_bos, add_eos)
    }

    /// 从三表 + special 配置构造（fixture 与真实 GGUF 共用）。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        pieces: Vec<String>,
        types: Vec<u32>,
        merges: Vec<String>,
        bos: Option<u32>,
        eos: Option<u32>,
        unk: Option<u32>,
        add_bos: bool,
        add_eos: bool,
    ) -> Result<Self, TokenizerError> {
        let text_to_id: HashMap<String, u32> =
            pieces.iter().enumerate().map(|(i, p)| (p.clone(), i as u32)).collect();

        // byte → id：字节编码字符存在即入表（= llama.cpp `text_to_token(byte_str)`）。
        let byte_map = byte_to_unicode();
        let mut byte_to_id = HashMap::with_capacity(256);
        for (b, c) in byte_map.into_iter().enumerate() {
            if let Some(&id) = text_to_id.get(c.to_string().as_str()) {
                byte_to_id.insert(b as u8, id);
            }
        }

        let mut merges_by_id: HashMap<(u32, u32), MergeEntry> =
            HashMap::with_capacity(merges.len());
        let mut merges_by_text: HashMap<String, HashMap<String, u32>> = HashMap::new();
        for (i, m) in merges.iter().enumerate() {
            // 首空格分割（llama.cpp 行 2013-2019：`word.find(' ', 1)`）。
            let Some(sp) = m.find(' ') else {
                return Err(TokenizerError::CorruptMerges {
                    why: format!("merge #{i} lacks a space separator: {m:?}"),
                });
            };
            let (left, right) = (m[..sp].to_string(), m[sp + 1..].to_string());
            let entry = MergeEntry { rank: i as u32 };
            match (text_to_id.get(&left), text_to_id.get(&right)) {
                (Some(&l), Some(&r)) => {
                    merges_by_id.insert((l, r), entry);
                }
                _ => {
                    merges_by_text.entry(left).or_default().insert(right, i as u32);
                }
            }
        }

        let mut specials: Vec<u32> = types
            .iter()
            .enumerate()
            .filter(|&(_, t)| matches!(*t, TYPE_CONTROL | TYPE_USER_DEFINED | TYPE_UNKNOWN))
            .map(|(i, _)| i as u32)
            .collect();
        specials.sort_by(|&a, &b| {
            pieces[b as usize].len().cmp(&pieces[a as usize].len()).then(a.cmp(&b))
        });

        Ok(Self {
            pieces,
            types,
            text_to_id,
            unicode_to_byte: unicode_to_byte(),
            byte_to_id,
            merges_by_id,
            merges_by_text,
            specials,
            unk,
            bos,
            eos,
            add_bos,
            add_eos,
        })
    }

    /// 编码（`add_special` 恒开：按元数据前置 BOS / 后置 EOS）。
    pub fn encode(&self, text: &str, parse_special: bool) -> Result<Vec<u32>, TokenizerError> {
        let mut out = Vec::new();
        if self.add_bos {
            let bos = self.bos.ok_or(TokenizerError::MissingSpecial { name: "bos" })?;
            out.push(bos);
        }

        if !text.is_empty() {
            let active: Vec<u32> = if parse_special {
                self.specials.clone()
            } else {
                self.specials
                    .iter()
                    .copied()
                    .filter(|&id| self.types[id as usize] == TYPE_USER_DEFINED)
                    .collect()
            };
            for frag in partition(text, &active, &self.pieces) {
                match frag {
                    Fragment::Raw(seg) => self.encode_word(seg, &mut out),
                    Fragment::Token(id) => out.push(id),
                }
            }
        }

        if self.add_eos {
            let eos = self.eos.ok_or(TokenizerError::MissingSpecial { name: "eos" })?;
            out.push(eos);
        }
        Ok(out)
    }

    /// 对单个 word（byte-encode 后的整段文本）执行 BPE 合并。
    fn encode_word(&self, word: &str, out: &mut Vec<u32>) {
        // 1. byte encoding：逐 byte → unicode char（`unicode_byte_encoding_process`）。
        let mut encoded = String::with_capacity(word.len());
        let btou = byte_to_unicode();
        for &b in word.as_bytes() {
            encoded.push(btou[b as usize]);
        }
        if encoded.is_empty() {
            return;
        }

        // 2. 符号链：每个 unicode 字符一个 symbol（`unicode_len_utf8` 切分）。
        let mut symbols: Vec<Sym> = Vec::with_capacity(encoded.chars().count());
        let mut prev = -1isize;
        for (idx, ch) in encoded.char_indices() {
            let i = symbols.len() as isize;
            if prev >= 0 {
                symbols[prev as usize].next = i;
            }
            symbols.push(Sym { start: idx, len: ch.len_utf8(), prev, next: -1 });
            prev = i;
        }
        let n = symbols.len();

        let mut queue = BinaryHeap::new();
        for i in 1..n {
            self.add_new_bigram(i as isize - 1, i as isize, &encoded, &symbols, &mut queue);
        }

        // 3. 合并主循环。
        while let Some(b) = queue.pop() {
            let left = &symbols[b.left];
            let right = &symbols[b.right];
            if left.len == 0 || right.len == 0 {
                continue;
            }
            // staleness：位置快照与当前一致才有效（≡ `left+right == bigram.text`）。
            if left.start != b.l_start
                || left.len != b.l_len
                || right.start != b.r_start
                || right.len != b.r_len
            {
                continue;
            }
            // merge right into left；right 出链（先拷贝再写，避免重叠借用）。
            let right_len = right.len;
            let right_next = right.next;
            symbols[b.left].len += right_len;
            symbols[b.left].next = right_next;
            symbols[b.right].len = 0;
            if right_next >= 0 {
                symbols[right_next as usize].prev = b.left as isize;
            }
            // 更新两个邻接 bigram（行 671-672）。
            self.add_new_bigram(
                symbols[b.left].prev,
                b.left as isize,
                &encoded,
                &symbols,
                &mut queue,
            );
            self.add_new_bigram(b.left as isize, right_next, &encoded, &symbols, &mut queue);
        }

        // 4. 终化：文本 → token；未命中 → byte 兜底（静默跳过）。
        for sym in &symbols {
            if sym.len == 0 {
                continue;
            }
            let text = &encoded[sym.start..sym.start + sym.len];
            if let Some(&id) = self.text_to_id.get(text) {
                out.push(id);
                continue;
            }
            for &b in text.as_bytes() {
                if let Some(&id) = self.byte_to_id.get(&b) {
                    out.push(id);
                }
            }
        }
    }

    /// 尝试登记邻接 bigram（无对应 merge 规则则跳过）。
    #[allow(clippy::too_many_arguments)]
    fn add_new_bigram(
        &self,
        left: isize,
        right: isize,
        encoded: &str,
        symbols: &[Sym],
        queue: &mut BinaryHeap<Bigram>,
    ) {
        if left < 0 || right < 0 {
            return;
        }
        let (l, r) = (left as usize, right as usize);
        let Some(ls) = symbols.get(l) else { return };
        let Some(rs) = symbols.get(r) else { return };
        if ls.len == 0 || rs.len == 0 {
            return;
        }
        let l_text = &encoded[ls.start..ls.start + ls.len];
        let r_text = &encoded[rs.start..rs.start + rs.len];

        let rank = match (self.text_to_id.get(l_text), self.text_to_id.get(r_text)) {
            (Some(&li), Some(&ri)) => self.merges_by_id.get(&(li, ri)).map(|e| e.rank),
            _ => self.merges_by_text.get(l_text).and_then(|inner| inner.get(r_text)).copied(),
        };
        let Some(rank) = rank else { return };

        queue.push(Bigram {
            rank,
            left: l,
            right: r,
            l_start: ls.start,
            l_len: ls.len,
            r_start: rs.start,
            r_len: rs.len,
        });
    }

    /// 单 token 的 piece 字节（decode 用；`special=false` 时 CONTROL/UNKNOWN
    /// 输出空——llama.cpp `token_to_piece` 返回 0 字节）。
    pub(crate) fn piece_bytes(&self, id: u32, special: bool) -> Option<Vec<u8>> {
        let piece = self.pieces.get(id as usize)?;
        let t = self.types.get(id as usize).copied().unwrap_or(TYPE_NORMAL);
        match t {
            TYPE_CONTROL | TYPE_UNKNOWN => {
                if special {
                    Some(piece.as_bytes().to_vec())
                } else {
                    Some(Vec::new())
                }
            }
            TYPE_USER_DEFINED => Some(piece.as_bytes().to_vec()),
            TYPE_NORMAL => Some(decode_normal_piece(piece, &self.unicode_to_byte)),
            TYPE_BYTE => Some(vec![byte_from_hex_piece(piece)]),
            // UNUSED / UNDEFINED：llama.cpp 非 special 分支无输出。
            TYPE_UNUSED | TYPE_UNDEFINED => Some(Vec::new()),
            _ => Some(Vec::new()),
        }
    }

    /// 全量解码（坏 id → "[UNK]"）。
    pub(crate) fn decode_all(&self, ids: &[u32]) -> String {
        let mut dec = IncrementalDecoder::new();
        for &id in ids {
            match self.piece_bytes(id, false) {
                Some(bytes) => dec.push(&bytes),
                None => dec.push(b"[UNK]"),
            }
        }
        dec.flush()
    }

    pub(crate) fn unmatched(&self) -> Option<u32> {
        self.unk
    }

    pub(crate) fn bos(&self) -> Option<u32> {
        self.bos
    }

    pub(crate) fn eos(&self) -> Option<u32> {
        self.eos
    }

    pub(crate) fn add_bos(&self) -> bool {
        self.add_bos
    }

    pub(crate) fn vocab_size(&self) -> usize {
        self.pieces.len()
    }
}

/// special token 分段（等价 llama.cpp 逐 special 迭代；最左优先、同起点
/// 取最长、平局取最小 id）。
fn partition<'a>(text: &'a str, specials: &[u32], pieces: &[String]) -> Vec<Fragment<'a>> {
    let mut frags = Vec::new();
    let mut pos = 0usize;
    let bytes = text.as_bytes();
    while pos < bytes.len() {
        let mut best: Option<(usize, usize, u32)> = None;
        for &sid in specials {
            let stext = pieces[sid as usize].as_bytes();
            if stext.is_empty() {
                continue;
            }
            let Some(rel) = memmem(&bytes[pos..], stext) else { continue };
            let start = pos + rel;
            match best {
                None => best = Some((start, stext.len(), sid)),
                Some((bs, bl, bid)) => {
                    if start < bs
                        || (start == bs && (stext.len() > bl || (stext.len() == bl && sid < bid)))
                    {
                        best = Some((start, stext.len(), sid));
                    }
                }
            }
        }
        let Some((start, len, sid)) = best else {
            frags.push(Fragment::Raw(&text[pos..]));
            break;
        };
        if start > pos {
            frags.push(Fragment::Raw(&text[pos..start]));
        }
        frags.push(Fragment::Token(sid));
        pos = start + len;
    }
    frags
}

/// 朴素子串查找（special 文本均很短）。
fn memmem(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() > haystack.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// NORMAL piece → 原始字节（char 逐字节回映射；未映射 → llama.cpp 的
/// `[UNK_BYTE_0x<hex><原 piece>]` 格式，行 3351-3360）。
fn decode_normal_piece(piece: &str, unicode_to_byte: &HashMap<char, u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(piece.len());
    for ch in piece.chars() {
        match unicode_to_byte.get(&ch) {
            Some(&b) => out.push(b),
            None => {
                let mut hex = String::new();
                for b in ch.to_string().as_bytes() {
                    use std::fmt::Write as _;
                    let _ = write!(hex, "{b:02x}");
                }
                out.extend_from_slice(format!("[UNK_BYTE_0x{hex}{piece}]").as_bytes());
            }
        }
    }
    out
}

/// BYTE 类型 piece（`<0xXX>`）→ 单字节；格式不符 → 0（`strtol` 语义）。
fn byte_from_hex_piece(piece: &str) -> u8 {
    let bytes = piece.as_bytes();
    if bytes.len() >= 5 && &bytes[..3] == b"<0x" && bytes[bytes.len() - 1] == b'>' {
        let hex = &bytes[3..bytes.len() - 1];
        if hex.len() == 2 {
            if let Ok(v) = u8::from_str_radix(std::str::from_utf8(hex).unwrap_or(""), 16) {
                return v;
            }
        }
    }
    0
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// 迷你 Qwen2 风格 vocab：字符 piece + 字节编码字符（'Ġ' 空格、
    /// 'ä'/'½'/U+0142 即 '你' 的 E4/BD/A0 字节）+ special。
    const TOKENS: &[&str] = &[
        "<unk>", "<s>", "</s>", // 0-2 specials
        "a", "b", "c", "ab", "bc", "abc", // 3-8
        "h", "e", "l", "o", "he", "hel", "hell", "hello", // 9-16
        "Ġ", "Ġh", "Ġhe", "Ġhel", "Ġhell", "Ġhello", // 17-22
        "w", "r", "d", "world", // 23-26
        "ä", "½", "\u{0142}", "你", // 27-30
        "<0x0a>", "<|sep|>", // 31-32
    ];
    const TYPES: &[u32] = &[
        2, 3, 3, // <unk> <s> </s>
        1, 1, 1, 1, 1, 1, // a b c ab bc abc
        1, 1, 1, 1, 1, 1, 1, 1, // h e l o he hel hell hello
        1, 1, 1, 1, 1, 1, // Ġ Ġh Ġhe Ġhel Ġhell Ġhello
        1, 1, 1, 1, // w r d world
        1, 1, 1, 1, // ä ½ U+0142 你
        6, 4, // <0x0a> <|sep|>
    ];
    const MERGES: &[&str] = &[
        "a b",     // 0 → ab
        "b c",     // 1 → bc
        "ab c",    // 2 → abc
        "Ġ h",     // 3 → Ġh
        "h e",     // 4 → he
        "he l",    // 5 → hel
        "hel l",   // 6 → hell
        "hell o",  // 7 → hello
        "Ġ hello", // 8 → Ġhello
        "w o",     // 9 → wo
        "wo r",    // 10 → wor
        "wor l",   // 11 → worl
        "worl d",  // 12 → world
    ];

    /// fixture 元数据（供 `Tokenizer::from_meta` 端到端测试复用）。
    pub(crate) fn fixture_kvs() -> Vec<(String, MetaValue)> {
        let array_str = |xs: &[&str]| {
            MetaValue::Array(ArrayValue::Str(xs.iter().map(|s| s.to_string()).collect()))
        };
        vec![
            ("tokenizer.ggml.model".to_string(), MetaValue::Str("gpt2".into())),
            ("tokenizer.ggml.tokens".to_string(), array_str(TOKENS)),
            (
                "tokenizer.ggml.token_type".to_string(),
                MetaValue::Array(ArrayValue::U32(TYPES.to_vec())),
            ),
            ("tokenizer.ggml.merges".to_string(), array_str(MERGES)),
            ("tokenizer.ggml.bos_token_id".to_string(), MetaValue::U32(1)),
            ("tokenizer.ggml.eos_token_id".to_string(), MetaValue::U32(2)),
            ("tokenizer.ggml.unknown_token_id".to_string(), MetaValue::U32(0)),
        ]
    }

    fn fixture(add_bos: bool, add_eos: bool) -> BpeTokenizer {
        BpeTokenizer::from_parts(
            TOKENS.iter().map(|s| s.to_string()).collect(),
            TYPES.to_vec(),
            MERGES.iter().map(|s| s.to_string()).collect(),
            Some(1),
            Some(2),
            Some(0),
            add_bos,
            add_eos,
        )
        .expect("valid fixture")
    }

    #[test]
    fn encode_deterministic_merges() {
        let tok = fixture(false, false);
        // "abc"：a+b(rank 0) → ab；ab+c(rank 2) → abc（b+c rank 1 过期跳过）
        assert_eq!(tok.encode("abc", false).unwrap(), [8]);
        // "hello"：h e → he → hel → hell → hello 全链合并
        assert_eq!(tok.encode("hello", false).unwrap(), [16]);
        // 幂等：同文本两次编码一致
        let t = "hello world abc";
        assert_eq!(tok.encode(t, false).unwrap(), tok.encode(t, false).unwrap());
    }

    #[test]
    fn encode_space_uses_byte_token() {
        let tok = fixture(false, false);
        // ' ' → byte 0x20 → 'Ġ'(17)；w o → wo → wor → worl → world 全链合并
        assert_eq!(tok.encode(" world", false).unwrap(), [17, 26]);
    }

    #[test]
    fn encode_cjk_falls_back_to_byte_tokens() {
        let tok = fixture(false, false);
        // '你' = E4 BD A0 → 字节编码字符 'ä'(27) '½'(28) U+0142(29)
        assert_eq!(tok.encode("你", false).unwrap(), [27, 28, 29]);
    }

    #[test]
    fn encode_uncovered_bytes_are_skipped() {
        let tok = fixture(false, false);
        // '\u{0}' → U+0100 不在 vocab，byte 兜底也找不到 → 静默跳过
        assert_eq!(tok.encode("\u{0}", false).unwrap(), []);
        // emoji 字节全未覆盖 → 空序列
        assert_eq!(tok.encode("🙂", false).unwrap(), []);
    }

    #[test]
    fn encode_special_partition_requires_parse_special() {
        let tok = fixture(false, false);
        // CONTROL：parse_special=false 时按普通文本处理（'<' 's' '>' 未覆盖 → 跳过）
        assert_eq!(tok.encode("a<s>b", false).unwrap(), [3, 4]);
        assert_eq!(tok.encode("a<s>b", true).unwrap(), [3, 1, 4]);
        // 最左优先：'<s>' 在 '<|sep|>' 起点之前 → 取 '<s>'（分段贪心）
        assert_eq!(tok.encode("<s><|sep|>", true).unwrap(), [1, 32]);
    }

    #[test]
    fn encode_user_defined_split_regardless_of_parse_special() {
        let tok = fixture(false, false);
        assert_eq!(tok.encode("hello<|sep|>world", false).unwrap(), [16, 32, 26]);
        assert_eq!(tok.encode("hello<|sep|>world", true).unwrap(), [16, 32, 26]);
    }

    #[test]
    fn encode_appends_bos_eos_from_metadata() {
        assert_eq!(fixture(true, true).encode("hello", false).unwrap(), [1, 16, 2]);
        assert_eq!(fixture(true, false).encode("hello", false).unwrap(), [1, 16]);
        assert_eq!(fixture(false, true).encode("hello", false).unwrap(), [16, 2]);
        // 空文本同样加 BOS/EOS（llama.cpp 行为）
        assert_eq!(fixture(true, true).encode("", false).unwrap(), [1, 2]);
    }

    #[test]
    fn decode_roundtrips_vocab_text() {
        let tok = fixture(false, false);
        let text = "hello world abc 你";
        let ids = tok.encode(text, false).unwrap();
        assert_eq!(tok.decode_all(&ids), text);
        // USER_DEFINED 恒输出原文
        assert_eq!(tok.decode_all(&[32]), "<|sep|>");
    }

    #[test]
    fn decode_skips_control_and_unknown() {
        let tok = fixture(false, false);
        assert_eq!(tok.decode_all(&[0]), ""); // <unk>
        assert_eq!(tok.decode_all(&[1]), ""); // <s>
        assert_eq!(tok.decode_all(&[2]), ""); // </s>
    }

    #[test]
    fn decode_byte_pieces_and_cjk() {
        let tok = fixture(false, false);
        assert_eq!(tok.decode_all(&[27, 28, 29]), "你");
        // 未映射字符（'你' 非字节编码字符）→ llama.cpp `[UNK_BYTE_0x<hex>]` +
        // 整个 piece 文本（llama-vocab.cpp 行 3360-3364）
        assert_eq!(tok.decode_all(&[30]), "[UNK_BYTE_0xe4bda0你]");
        assert_eq!(tok.decode_all(&[31]), "\n"); // <0x0a> → 单字节
        assert_eq!(tok.decode_all(&[17, 26]), " world");
    }

    #[test]
    fn decode_bad_ids_yield_unknown() {
        let tok = fixture(false, false);
        assert_eq!(tok.decode_all(&[999]), "[UNK]");
        assert_eq!(tok.decode_all(&[0, 999, 16]), "[UNK]hello");
    }

    #[test]
    fn chunked_decode_matches_bulk() {
        let tok = fixture(false, false);
        let text = "Hello, 世界！🙂\n\u{0301} 你";
        let ids = tok.encode(text, false).unwrap();
        let bulk = tok.decode_all(&ids);

        let mut dec = IncrementalDecoder::new();
        let mut chunked = String::new();
        for &id in &ids {
            dec.push(&tok.piece_bytes(id, false).unwrap_or_else(|| b"[UNK]".to_vec()));
            chunked.push_str(&dec.take());
        }
        chunked.push_str(&dec.flush());
        assert_eq!(chunked, bulk);
    }

    #[test]
    fn fuzz_encode_decode_never_panics() {
        let tok = fixture(false, false);
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as usize
        };
        let pool: Vec<char> = "abc hello世界🙂\n\u{0301}\u{FFFD}Ġ ".chars().collect();
        for _ in 0..200 {
            let len = next() % 64;
            let s: String = (0..len).map(|_| pool[next() % pool.len()]).collect();
            // encode 不 panic 不报错；任意 id（含坏 id）decode 不 panic
            let ids = tok.encode(&s, next() % 2 == 0).expect("encode never errors");
            let _ = tok.decode_all(&ids);
            let _ = tok.decode_all(&[next() as u32, next() as u32]);
        }
    }

    #[test]
    fn accessors_reflect_fixture() {
        let tok = fixture(false, false);
        assert_eq!(tok.vocab_size(), 33);
        assert_eq!(tok.bos(), Some(1));
        assert_eq!(tok.eos(), Some(2));
        assert_eq!(tok.unmatched(), Some(0));
        assert!(!tok.add_bos());
        assert!(fixture(true, false).add_bos());
    }
}

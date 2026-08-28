//! SentencePiece 容器读入（`spiece.model` proto 手工 wire 扫描）。
//!
//! 只提取 encode 需要的字段，不依赖 protobuf 运行时：
//! - `ModelProto`（顶层）：field 1 = repeated `SentencePiece`；其余字段
//!   （normalizer_spec/trainer_spec/denormalizer_spec/self_test_data）跳过。
//! - `SentencePiece`：field 1 = piece（string）；field 2 = score
//!   （wire-5，f32）；field 3 = type（varint，1 NORMAL … 6 BYTE，
//!   与 GGUF `tokenizer.ggml.token_type` 同构，可直接复用）。
//!
//! 非法 wire type / 截断 → `CorruptSpm`（可读错误，绝不 panic）。

use reinfer_gguf::{ArrayValue, MetaValue, ModelMeta};

use crate::decode::IncrementalDecoder;
use crate::{TokenizerError, meta_bool, meta_special_id};

/// SentencePiece type（与 GGUF `tokenizer.ggml.token_type` 同构）。
const TYPE_UNDEFINED: u32 = 0;
const TYPE_NORMAL: u32 = 1;
const TYPE_UNKNOWN: u32 = 2;
const TYPE_CONTROL: u32 = 3;
const TYPE_USER_DEFINED: u32 = 4;
const TYPE_UNUSED: u32 = 5;
const TYPE_BYTE: u32 = 6;

/// SentencePiece tokenizer（encode 暂未实现；decode 对齐 llama.cpp SPM 分支：
/// NORMAL 做 ▁→空格，BYTE 做 `<0xXX>`→单字节）。`add_eos` 元数据随 encode
/// 一并延后消费（等 004 M4 golden）。
#[derive(Debug)]
pub struct SpmTokenizer {
    pieces: Vec<String>,
    types: Vec<u32>,
    unk: Option<u32>,
    bos: Option<u32>,
    eos: Option<u32>,
    add_bos: bool,
}

impl SpmTokenizer {
    /// 自 GGUF 元数据构造（`tokenizer.ggml.model = "sentencepiece"`）。
    pub fn from_meta(meta: &ModelMeta) -> Result<Self, TokenizerError> {
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
            // 014 T4：真实发布 GGUF 的 token_type 为 i32 数组（转换器 v4 起）——
            // 与 u32 同值，符号扩展安全（token_type 取值域 0..=6）。
            Some(MetaValue::Array(ArrayValue::I32(xs))) => {
                if xs.len() < pieces.len() {
                    return Err(TokenizerError::InvalidMetadata {
                        key: "tokenizer.ggml.token_type".into(),
                        why: format!("len {} < tokens len {}", xs.len(), pieces.len()),
                    });
                }
                xs[..pieces.len()].iter().map(|v| *v as u32).collect()
            }
            Some(_) => {
                return Err(TokenizerError::InvalidMetadata {
                    key: "tokenizer.ggml.token_type".into(),
                    why: "expected u32/i32 array".into(),
                });
            }
        };

        let bos = meta_special_id(meta, "tokenizer.ggml.bos_token_id", "<s>", &pieces)?;
        let eos = meta_special_id(meta, "tokenizer.ggml.eos_token_id", "</s>", &pieces)?;
        let unk = meta_special_id(meta, "tokenizer.ggml.unknown_token_id", "<unk>", &pieces)?;

        // llama.cpp：SPM 默认 add_bos=true（行 2391-2392）；add_eos 见结构体注释。
        let add_bos = meta_bool(meta, "tokenizer.ggml.add_bos_token", true)?;

        Ok(Self { pieces, types, unk, bos, eos, add_bos })
    }

    /// 自 `spiece.model` 容器构造。
    pub fn from_container(container: SpmContainer) -> Self {
        let pieces = container.pieces.iter().map(|(p, _)| p.clone()).collect::<Vec<_>>();
        let types = container.pieces.iter().map(|(_, t)| *t).collect::<Vec<_>>();
        let bos = pieces.iter().position(|p| p == "<s>").map(|i| i as u32);
        let eos = pieces.iter().position(|p| p == "</s>").map(|i| i as u32);
        let unk = pieces.iter().position(|p| p == "<unk>").map(|i| i as u32);
        Self { pieces, types, unk, bos, eos, add_bos: true }
    }

    /// SPM encode 未实现（等 004 M4 golden）。
    pub fn encode(&self) -> Result<Vec<u32>, TokenizerError> {
        Err(TokenizerError::SpmEncodeUnimplemented)
    }

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

    /// 单 token 的 piece 字节（llama.cpp `token_to_piece` SPM/WPM 分支；
    /// `special=false` 时 UNKNOWN/CONTROL 无输出）。
    fn piece_bytes(&self, id: u32, special: bool) -> Option<Vec<u8>> {
        let piece = self.pieces.get(id as usize)?;
        let t = self.types.get(id as usize).copied().unwrap_or(TYPE_NORMAL);
        match t {
            TYPE_UNKNOWN | TYPE_CONTROL => {
                if special {
                    Some(piece.as_bytes().to_vec())
                } else {
                    Some(Vec::new())
                }
            }
            TYPE_USER_DEFINED => Some(piece.as_bytes().to_vec()),
            TYPE_NORMAL => Some(piece.replace('\u{2581}', " ").into_bytes()),
            TYPE_BYTE => Some(vec![byte_from_hex_piece(piece)]),
            // UNUSED / UNDEFINED：llama.cpp 非 special 分支无输出。
            TYPE_UNUSED | TYPE_UNDEFINED => Some(Vec::new()),
            _ => Some(Vec::new()),
        }
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

/// 解析后的 piece 表：(文本, gguf token_type)。
#[derive(Debug, Clone)]
pub struct SpmContainer {
    /// 解析出的 (piece 文本, SentencePiece type) 列表。
    pub pieces: Vec<(String, u32)>,
}

impl SpmContainer {
    /// 从原始 `spiece.model` 字节解析。
    pub fn parse_model_proto(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let mut pieces = Vec::new();
        let mut pos = 0usize;
        while pos < bytes.len() {
            let (field, wire) = read_key(bytes, &mut pos)?;
            match (field, wire) {
                (1, 2) => {
                    let len = read_varint(bytes, &mut pos)? as usize;
                    let end = pos
                        .checked_add(len)
                        .filter(|&e| e <= bytes.len())
                        .ok_or_else(|| corrupt("SentencePiece 截断"))?;
                    let sp = parse_sentence_piece(&bytes[pos..end])?;
                    pieces.push(sp);
                    pos = end;
                }
                // 其余顶层字段：按 wire type 跳过。
                (_, 0) => {
                    read_varint(bytes, &mut pos)?;
                }
                (_, 1) => {
                    skip_fixed::<8>(bytes, &mut pos)?;
                }
                (_, 2) => {
                    let len = read_varint(bytes, &mut pos)? as usize;
                    skip(bytes, &mut pos, len)?;
                }
                (_, 5) => {
                    skip_fixed::<4>(bytes, &mut pos)?;
                }
                (_, w) => {
                    return Err(corrupt(&format!("未知 wire type {w}")));
                }
            }
        }
        if pieces.is_empty() {
            return Err(corrupt("空 vocab"));
        }
        Ok(Self { pieces })
    }
}

fn corrupt(why: &str) -> TokenizerError {
    TokenizerError::CorruptSpm { why: why.to_string() }
}

/// (field_number, wire_type)。
fn read_key(bytes: &[u8], pos: &mut usize) -> Result<(u32, u32), TokenizerError> {
    let key = read_varint(bytes, pos)?;
    Ok(((key >> 3) as u32, (key & 7) as u32))
}

/// LEB128 varint（≤ 10 字节，溢出 → CorruptSpm）。
fn read_varint(bytes: &[u8], pos: &mut usize) -> Result<u64, TokenizerError> {
    let mut value = 0u64;
    for i in 0..10 {
        let b = *bytes.get(*pos).ok_or_else(|| corrupt("varint 截断"))?;
        *pos += 1;
        if i == 9 && b & 0xFE != 0 {
            return Err(corrupt("varint 溢出"));
        }
        value |= u64::from(b & 0x7F) << (7 * i);
        if b & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(corrupt("varint 超长"))
}

/// 定长跳过（wire type 1 / 5）。
fn skip_fixed<const N: usize>(bytes: &[u8], pos: &mut usize) -> Result<(), TokenizerError> {
    skip(bytes, pos, N)
}

/// 跳过 `len` 字节。
fn skip(bytes: &[u8], pos: &mut usize, len: usize) -> Result<(), TokenizerError> {
    let end =
        pos.checked_add(len).filter(|&e| e <= bytes.len()).ok_or_else(|| corrupt("字段截断"))?;
    *pos = end;
    Ok(())
}

fn parse_sentence_piece(bytes: &[u8]) -> Result<(String, u32), TokenizerError> {
    let mut piece = String::new();
    let mut kind = 1u32; // SentencePiece 默认 NORMAL
    let mut pos = 0usize;
    while pos < bytes.len() {
        let (field, wire) = read_key(bytes, &mut pos)?;
        match (field, wire) {
            (1, 2) => {
                let len = read_varint(bytes, &mut pos)? as usize;
                let end = pos
                    .checked_add(len)
                    .filter(|&e| e <= bytes.len())
                    .ok_or_else(|| corrupt("piece 字符串截断"))?;
                let raw = &bytes[pos..end];
                piece = String::from_utf8(raw.to_vec()).map_err(|_| corrupt("piece 非 UTF-8"))?;
                pos = end;
            }
            (2, 5) => {
                skip_fixed::<4>(bytes, &mut pos)?;
            }
            (3, 0) => {
                let v = read_varint(bytes, &mut pos)?;
                kind = v as u32;
            }
            // 未知字段按 wire type 跳过（向后兼容）。
            (_, 0) => {
                read_varint(bytes, &mut pos)?;
            }
            (_, 1) => {
                skip_fixed::<8>(bytes, &mut pos)?;
            }
            (_, 2) => {
                let len = read_varint(bytes, &mut pos)? as usize;
                skip(bytes, &mut pos, len)?;
            }
            (_, 5) => {
                skip_fixed::<4>(bytes, &mut pos)?;
            }
            (_, w) => {
                return Err(corrupt(&format!("未知 wire type {w}")));
            }
        }
    }
    Ok((piece, kind))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 手工编码一个 SentencePiece 字段。
    fn sp_field(field: u64, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![((field << 3) | 2) as u8];
        let mut len = payload.len() as u64;
        while len >= 0x80 {
            out.push((len as u8 & 0x7F) | 0x80);
            len >>= 7;
        }
        out.push(len as u8);
        out.extend_from_slice(payload);
        out
    }

    fn varint(v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        let mut v = v;
        while v >= 0x80 {
            out.push((v as u8 & 0x7F) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
        out
    }

    /// 编码 SentencePiece type 字段（field 3，wire type 0 = varint）。
    fn type_field(v: u64) -> Vec<u8> {
        let mut out = vec![3 << 3];
        out.extend(varint(v));
        out
    }

    #[test]
    fn parses_minimal_proto() {
        // 三个 SentencePiece：hello(NORMAL)、<unk>(UNKNOWN=2)、<0x00>(BYTE=6)
        let hello = sp_field(1, b"hello");
        let hello_sp = [&hello[..], &type_field(1)[..]].concat();
        let unk = sp_field(1, b"<unk>");
        let unk_sp = [&unk[..], &type_field(2)[..]].concat();
        let byte = sp_field(1, b"<0x00>");
        let byte_sp = [&byte[..], &type_field(6)[..]].concat();
        // 顶层：field 1 = repeated SentencePiece；field 3 = trainer_spec（跳过）
        let mut proto = Vec::new();
        proto.extend(sp_field(1, &hello_sp));
        proto.extend(sp_field(3, &[]));
        proto.extend(sp_field(1, &unk_sp));
        proto.extend(sp_field(1, &byte_sp));

        let parsed = SpmContainer::parse_model_proto(&proto).expect("可解析");
        assert_eq!(
            parsed.pieces,
            vec![("hello".to_string(), 1), ("<unk>".to_string(), 2), ("<0x00>".to_string(), 6),]
        );
    }

    #[test]
    fn rejects_corrupt_wire_data() {
        // 未知 wire type 3
        assert!(SpmContainer::parse_model_proto(&[0x0B]).is_err());
        // 截断的 length-delimited
        assert!(SpmContainer::parse_model_proto(&[0x0A, 0x05, b'a']).is_err());
        // 空 vocab
        assert!(SpmContainer::parse_model_proto(&[]).is_err());
        // 非 UTF-8 piece
        let bad = sp_field(1, b"\xFF\xFE");
        let bad_sp = [&bad[..], &sp_field(3, &varint(1))[..]].concat();
        assert!(SpmContainer::parse_model_proto(&sp_field(1, &bad_sp)).is_err());
    }

    #[test]
    fn spm_tokenizer_decodes() {
        // ▁hello(NORMAL) / <unk>(UNKNOWN) / <0x0A>(BYTE) / <s>(CONTROL)
        let hello_sp = [&sp_field(1, "▁hello".as_bytes())[..], &type_field(1)[..]].concat();
        let unk_sp = [&sp_field(1, b"<unk>")[..], &type_field(2)[..]].concat();
        let byte_sp = [&sp_field(1, b"<0x0A>")[..], &type_field(6)[..]].concat();
        let ctrl_sp = [&sp_field(1, b"<s>")[..], &type_field(3)[..]].concat();
        let mut proto = Vec::new();
        for sp in [&hello_sp, &unk_sp, &byte_sp, &ctrl_sp] {
            proto.extend(sp_field(1, sp));
        }

        let container = SpmContainer::parse_model_proto(&proto).expect("parse");
        let tok = SpmTokenizer::from_container(container);
        // ▁→空格；UNKNOWN/CONTROL 跳过；<0x0A> → 0x0A
        assert_eq!(tok.decode_all(&[0, 1, 2, 3]), " hello\n");
        // SPM 默认 add_bos=true（llama.cpp 行 2391）
        assert!(tok.add_bos());
        assert_eq!(tok.unmatched(), Some(1));
        assert_eq!(tok.vocab_size(), 4);
    }
}

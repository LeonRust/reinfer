//! Tokenizer（014 T4 / 004 规格实现：byte-level BPE + SPM 容器读入）。
//!
//! - `Tokenizer::from_meta`：自 GGUF `ModelMeta`（`tokenizer.ggml.*`）构造；
//! - `encode`：special 分段 + byte-level BPE 合并（无 regex 预分段，
//!   见 [`crate::bpe`] 的 deviation 记录）；SPM encode 暂返回
//!   [`TokenizerError::SpmEncodeUnimplemented`]（等 004 M4 golden）；
//! - `decode_all`：增量 UTF-8 解码（坏 id → "[UNK]"）；
//! - [`IncrementalDecoder`]：分块自洽的流式解码器。
//!
//! 锚点：llama.cpp commit f280b2698（`llama-vocab.cpp` / `src/unicode.cpp`）。

mod bpe;
mod decode;
mod spm;
mod unicode;

pub use decode::IncrementalDecoder;
pub use spm::SpmContainer;

use std::fmt;

use reinfer_gguf::{GgufReader, MetaValue, ModelMeta};

/// Tokenizer 错误（004 规格：bad id 不 panic，坏数据给出可读错误）。
#[derive(Debug)]
pub enum TokenizerError {
    /// 必需元数据缺失。
    MissingMetadata {
        /// 缺失的 GGUF 元数据键。
        key: String,
    },
    /// 元数据存在但类型不对。
    InvalidMetadata {
        /// 出错的 GGUF 元数据键。
        key: String,
        /// 类型/取值不符的具体原因。
        why: String,
    },
    /// tokenizer 模型不识别。
    UnsupportedModel {
        /// `tokenizer.ggml.model` 的取值。
        model: String,
    },
    /// merges 表损坏。
    CorruptMerges {
        /// 损坏原因（条目序号 + 原文）。
        why: String,
    },
    /// add_bos/add_eos 开启但对应 token id 缺失。
    MissingSpecial {
        /// 缺失的 special 名称（"bos"/"eos"）。
        name: &'static str,
    },
    /// SentencePiece proto 损坏。
    CorruptSpm {
        /// 损坏原因。
        why: String,
    },
    /// SPM encode 尚未实现（等 004 M4 golden 锚定）。
    SpmEncodeUnimplemented,
    /// 底层 GGUF 元数据解析错误。
    Gguf(reinfer_gguf::GgufError),
}

impl From<reinfer_gguf::GgufError> for TokenizerError {
    fn from(e: reinfer_gguf::GgufError) -> Self {
        TokenizerError::Gguf(e)
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::MissingMetadata { key } => {
                write!(f, "tokenizer metadata missing: {key}")
            }
            TokenizerError::InvalidMetadata { key, why } => {
                write!(f, "tokenizer metadata key {key}: {why}")
            }
            TokenizerError::UnsupportedModel { model } => {
                write!(f, "unsupported tokenizer model: {model}")
            }
            TokenizerError::CorruptMerges { why } => write!(f, "corrupt merges: {why}"),
            TokenizerError::MissingSpecial { name } => {
                write!(f, "missing special token: {name}")
            }
            TokenizerError::CorruptSpm { why } => write!(f, "corrupt spiece model: {why}"),
            TokenizerError::SpmEncodeUnimplemented => {
                write!(f, "sentencepiece encode not implemented yet")
            }
            TokenizerError::Gguf(e) => write!(f, "gguf metadata: {e}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// Tokenizer（BPE / SentencePiece 两模型）。
#[derive(Debug)]
pub enum Tokenizer {
    /// Byte-level BPE（`tokenizer.ggml.model = "gpt2"`）。Box 隔离两张
    /// HashMap 表带来的 ~340B 变体差异（构造一次、随后只读）。
    Bpe(Box<bpe::BpeTokenizer>),
    /// SentencePiece（`"sentencepiece"`；encode 未实现，等 004 M4 golden）。
    Spm(spm::SpmTokenizer),
}

impl Tokenizer {
    /// 自 GGUF 文件读取器构造。
    pub fn from_gguf(reader: &GgufReader) -> Result<Self, TokenizerError> {
        Self::from_meta(reader.metadata())
    }

    /// 自 GGUF 元数据构造（`tokenizer.ggml.model` 决定模型类型）。
    pub fn from_meta(meta: &ModelMeta) -> Result<Self, TokenizerError> {
        let model = meta.meta_str("tokenizer.ggml.model")?.ok_or_else(|| {
            TokenizerError::MissingMetadata { key: "tokenizer.ggml.model".into() }
        })?;
        match model {
            "gpt2" => Ok(Tokenizer::Bpe(Box::new(bpe::BpeTokenizer::from_meta(meta)?))),
            "sentencepiece" => Ok(Tokenizer::Spm(spm::SpmTokenizer::from_meta(meta)?)),
            other => Err(TokenizerError::UnsupportedModel { model: other.to_string() }),
        }
    }

    /// 自原始 `spiece.model` 字节构造 SPM tokenizer。
    pub fn from_spm_proto(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let container = SpmContainer::parse_model_proto(bytes)?;
        Ok(Tokenizer::Spm(spm::SpmTokenizer::from_container(container)))
    }

    /// 自 HF `tokenizer.json` + `tokenizer_config.json` 构造
    /// （模型文件统一对象——非 GGUF 输入面的 tokenizer 路径）。
    pub fn from_hf_json(
        tok: &serde_json::Value,
        cfg: &serde_json::Value,
    ) -> Result<Self, TokenizerError> {
        let model =
            tok.get("model").and_then(|m| m.get("type")).and_then(|t| t.as_str()).ok_or_else(
                || TokenizerError::InvalidMetadata {
                    key: "tokenizer.json".into(),
                    why: "missing model.type".into(),
                },
            )?;
        match model {
            "BPE" => Ok(Tokenizer::Bpe(Box::new(bpe::BpeTokenizer::from_hf_json(tok, cfg)?))),
            other => Err(TokenizerError::UnsupportedModel { model: other.to_string() }),
        }
    }

    /// 编码文本（special 分段 + BPE 合并；按元数据加 BOS/EOS）。
    pub fn encode(&self, text: &str, parse_special: bool) -> Result<Vec<u32>, TokenizerError> {
        match self {
            Tokenizer::Bpe(b) => b.encode(text, parse_special),
            Tokenizer::Spm(_) => Err(TokenizerError::SpmEncodeUnimplemented),
        }
    }

    /// 全量解码（坏 id → "[UNK]"；CONTROL/UNKNOWN 按 llama.cpp 默认跳过）。
    pub fn decode_all(&self, ids: &[u32]) -> String {
        match self {
            Tokenizer::Bpe(b) => b.decode_all(ids),
            Tokenizer::Spm(s) => s.decode_all(ids),
        }
    }

    /// 未知 token id（`<unk>`；缺失 → None）。
    pub fn unmatched_token(&self) -> Option<u32> {
        match self {
            Tokenizer::Bpe(b) => b.unmatched(),
            Tokenizer::Spm(s) => s.unmatched(),
        }
    }

    /// BOS token id（缺失 → None）。
    pub fn bos_token(&self) -> Option<u32> {
        match self {
            Tokenizer::Bpe(b) => b.bos(),
            Tokenizer::Spm(s) => s.bos(),
        }
    }

    /// EOS token id（缺失 → None）。
    pub fn eos_token(&self) -> Option<u32> {
        match self {
            Tokenizer::Bpe(b) => b.eos(),
            Tokenizer::Spm(s) => s.eos(),
        }
    }

    /// 编码时是否前置 BOS（元数据 `tokenizer.ggml.add_bos_token`）。
    pub fn add_bos(&self) -> bool {
        match self {
            Tokenizer::Bpe(b) => b.add_bos(),
            Tokenizer::Spm(s) => s.add_bos(),
        }
    }

    /// vocab 大小。
    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::Bpe(b) => b.vocab_size(),
            Tokenizer::Spm(s) => s.vocab_size(),
        }
    }
}

/// 读布尔元数据（缺键 → 默认）。
fn meta_bool(meta: &ModelMeta, key: &str, default: bool) -> Result<bool, TokenizerError> {
    match meta.get(key) {
        None => Ok(default),
        Some(MetaValue::Bool(b)) => Ok(*b),
        Some(_) => {
            Err(TokenizerError::InvalidMetadata { key: key.into(), why: "expected bool".into() })
        }
    }
}

/// 读特殊 token id（缺键 → 名字回退）。
fn meta_special_id(
    meta: &ModelMeta,
    key: &str,
    name: &str,
    pieces: &[String],
) -> Result<Option<u32>, TokenizerError> {
    match meta.get(key) {
        None => Ok(pieces.iter().position(|p| p == name).map(|i| i as u32)),
        Some(MetaValue::U32(v)) => Ok(Some(*v)),
        Some(_) => {
            Err(TokenizerError::InvalidMetadata { key: key.into(), why: "expected u32".into() })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bpe::tests::fixture_kvs;
    use reinfer_gguf::ArrayValue;

    fn meta(kvs: Vec<(String, MetaValue)>) -> ModelMeta {
        ModelMeta::from_kvs(kvs)
    }

    #[test]
    fn from_meta_bpe_encodes() {
        let tok = Tokenizer::from_meta(&meta(fixture_kvs())).expect("bpe");
        assert!(matches!(tok, Tokenizer::Bpe(_)));
        assert_eq!(tok.encode("abc", false).unwrap(), [8]);
        assert_eq!(tok.decode_all(&[16, 17, 23, 12, 24, 11, 25]), "hello world");
        assert_eq!(tok.vocab_size(), 33);
        assert_eq!(tok.bos_token(), Some(1));
        assert_eq!(tok.eos_token(), Some(2));
        assert_eq!(tok.unmatched_token(), Some(0));
        assert!(!tok.add_bos());
    }

    #[test]
    fn from_meta_spm_selects_sentencepiece() {
        let kvs = vec![
            ("tokenizer.ggml.model".to_string(), MetaValue::Str("sentencepiece".into())),
            (
                "tokenizer.ggml.tokens".to_string(),
                MetaValue::Array(ArrayValue::Str(
                    vec!["▁hello", "<unk>", "<s>", "</s>"].into_iter().map(String::from).collect(),
                )),
            ),
            (
                "tokenizer.ggml.token_type".to_string(),
                MetaValue::Array(ArrayValue::U32(vec![1, 2, 3, 3])),
            ),
        ];
        let tok = Tokenizer::from_meta(&meta(kvs)).expect("spm");
        assert!(matches!(tok, Tokenizer::Spm(_)));
        assert!(matches!(tok.encode("hello", false), Err(TokenizerError::SpmEncodeUnimplemented)));
        assert_eq!(tok.decode_all(&[0]), " hello"); // ▁ → 空格
        assert_eq!(tok.decode_all(&[1]), ""); // <unk> 跳过
        assert!(tok.add_bos()); // SPM 默认 add_bos=true
        assert_eq!(tok.vocab_size(), 4);
    }

    #[test]
    fn rejects_unsupported_model() {
        let mut kvs = fixture_kvs();
        kvs[0] = ("tokenizer.ggml.model".to_string(), MetaValue::Str("wordpiece".into()));
        let err = Tokenizer::from_meta(&meta(kvs)).unwrap_err();
        assert!(
            matches!(err, TokenizerError::UnsupportedModel { ref model } if model == "wordpiece")
        );
    }

    #[test]
    fn rejects_missing_metadata() {
        let err = Tokenizer::from_meta(&meta(vec![])).unwrap_err();
        assert!(matches!(
            err,
            TokenizerError::MissingMetadata { ref key } if key == "tokenizer.ggml.model"
        ));

        let kvs = vec![("tokenizer.ggml.model".to_string(), MetaValue::Str("gpt2".into()))];
        let err = Tokenizer::from_meta(&meta(kvs)).unwrap_err();
        assert!(matches!(
            err,
            TokenizerError::MissingMetadata { ref key } if key == "tokenizer.ggml.tokens"
        ));

        let mut kvs = fixture_kvs();
        kvs.retain(|(k, _)| k != "tokenizer.ggml.merges");
        let err = Tokenizer::from_meta(&meta(kvs)).unwrap_err();
        assert!(matches!(
            err,
            TokenizerError::MissingMetadata { ref key } if key == "tokenizer.ggml.merges"
        ));
    }

    #[test]
    fn rejects_corrupt_merges() {
        let mut kvs = fixture_kvs();
        for (k, v) in &mut kvs {
            if k == "tokenizer.ggml.merges" {
                *v = MetaValue::Array(ArrayValue::Str(vec!["no-space-merge".into()]));
            }
        }
        let err = Tokenizer::from_meta(&meta(kvs)).unwrap_err();
        assert!(matches!(err, TokenizerError::CorruptMerges { .. }));
    }

    #[test]
    fn rejects_wrong_metadata_types() {
        let mut kvs = fixture_kvs();
        for (k, v) in &mut kvs {
            if k == "tokenizer.ggml.token_type" {
                *v = MetaValue::Array(ArrayValue::Str(vec!["1".into()]));
            }
        }
        let err = Tokenizer::from_meta(&meta(kvs)).unwrap_err();
        assert!(
            matches!(err, TokenizerError::InvalidMetadata { ref key, .. } if key == "tokenizer.ggml.token_type")
        );

        let mut kvs = fixture_kvs();
        for (k, v) in &mut kvs {
            if k == "tokenizer.ggml.bos_token_id" {
                *v = MetaValue::Str("1".into());
            }
        }
        let err = Tokenizer::from_meta(&meta(kvs)).unwrap_err();
        assert!(
            matches!(err, TokenizerError::InvalidMetadata { ref key, .. } if key == "tokenizer.ggml.bos_token_id")
        );
    }

    #[test]
    fn special_ids_fall_back_to_names() {
        let mut kvs = fixture_kvs();
        kvs.retain(|(k, _)| {
            !matches!(
                k.as_str(),
                "tokenizer.ggml.bos_token_id"
                    | "tokenizer.ggml.eos_token_id"
                    | "tokenizer.ggml.unknown_token_id"
            )
        });
        let tok = Tokenizer::from_meta(&meta(kvs)).expect("fallback");
        assert_eq!(tok.bos_token(), Some(1)); // "<s>" 在 vocab 中的位置
        assert_eq!(tok.eos_token(), Some(2));
        assert_eq!(tok.unmatched_token(), Some(0));
    }

    #[test]
    fn spm_proto_end_to_end() {
        // 最小 spiece.model：▁hello(NORMAL) / <unk>(UNKNOWN) / <0x0A>(BYTE)
        let piece = |p: &[u8]| {
            let mut f = vec![0x0A]; // field 1, wire 2
            f.push(p.len() as u8);
            f.extend_from_slice(p);
            f
        };
        let type_field = |v: u64| {
            let mut f = vec![0x18]; // field 3, wire 0
            let mut v = v;
            while v >= 0x80 {
                f.push((v as u8 & 0x7F) | 0x80);
                v >>= 7;
            }
            f.push(v as u8);
            f
        };
        let mut proto = Vec::new();
        for (p, t) in [("▁hello", 1u64), ("<unk>", 2), ("<0x0A>", 6)] {
            let mut sp = piece(p.as_bytes());
            sp.extend(type_field(t));
            let mut top = vec![0x0A]; // 顶层 field 1, wire 2
            top.push(sp.len() as u8);
            top.extend(sp);
            proto.extend(top);
        }

        let tok = Tokenizer::from_spm_proto(&proto).expect("spm proto");
        assert_eq!(tok.decode_all(&[0, 1, 2]), " hello\n");
        assert!(matches!(tok.encode("x", false), Err(TokenizerError::SpmEncodeUnimplemented)));
    }
}

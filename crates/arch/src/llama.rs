//! llama/qwen2 架构元数据 → [`LlamaConfig`]（014 T3）。
//!
//! 键名与默认值对齐 llama.cpp `llm_arch_table` / `llm_load_hparams`：
//! - 键族 `{arch}.{...}`，`arch` 取 `general.architecture` 小写值（llama / qwen2 白名单）；
//! - 必需键缺失 → [`ArchError::MissingKey`]（消息含键名）；
//! - 可选键缺失 → 架构默认值（各字段注释标注）；
//! - `head_dim` 解析链：`attention.key_length` → `rope.dimension_count` →
//!   `embedding_length / head_count`（整除失败 → 错误）；
//! - GQA 映射（`kv_head = q_head / kv_ratio`，014 D3）由装配层消费，本层只校验
//!   `0 < kv_heads <= q_heads`（非整除方向在 T9 实现处固定并核验 14/2、12/2、5/2 三例）。
//!
//! 零模型标识：测试使用虚构架构形状，真实模型名只在文档/env 示例范畴（013 铁律）。

use std::fmt;

use reinfer_gguf::{GgufError, ModelMeta};

/// 支持的架构标识（`general.architecture` 白名单）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    /// LLaMA 系（含 Llama 3.x；rope type = NORM）。
    Llama,
    /// Qwen2 系（Qwen2 / 2.5；rope type = NEOX，freq base 1e6）。
    Qwen2,
    /// Qwen3（与 Qwen2 同 rope 语义，另具 q/k head norm）。
    Qwen3,
}

impl Architecture {
    /// GGUF 元数据键前缀（`{arch}.`）。
    pub fn as_str(self) -> &'static str {
        match self {
            Architecture::Llama => "llama",
            Architecture::Qwen2 => "qwen2",
            Architecture::Qwen3 => "qwen3",
        }
    }

    /// 从 `general.architecture` 小写值解析；未知值 → `None`。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "llama" => Some(Architecture::Llama),
            "qwen2" => Some(Architecture::Qwen2),
            "qwen3" => Some(Architecture::Qwen3),
            _ => None,
        }
    }

    /// 架构默认 rope freq base（llama.cpp `llm_load_hparams` 默认值）。
    fn default_rope_freq_base(self) -> f32 {
        match self {
            Architecture::Llama => 10_000.0,
            Architecture::Qwen2 | Architecture::Qwen3 => 1_000_000.0,
        }
    }

    /// llama.cpp 架构 → rope type 映射（`llm_arch_table`）。
    fn rope_type(self) -> RopeType {
        match self {
            Architecture::Llama => RopeType::Norm,
            Architecture::Qwen2 | Architecture::Qwen3 => RopeType::Neox,
        }
    }

    /// 是否带 q/k head norm（Qwen3 特有——`k_norm`/`q_norm` 张量）。
    fn has_head_norm(self) -> bool {
        matches!(self, Architecture::Qwen3)
    }
}

/// 旋转位置编码类型（llama.cpp `llama_rope_type`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeType {
    /// 原始 LLaMA 旋转（两半拼接）。
    Norm,
    /// Neox 风格旋转（交替维度）。
    Neox,
}

/// 类型化模型配置（数值核/装配层消费；字段语义对齐 llama.cpp `llama_hparams`）。
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    /// 架构标识（决定键前缀与 rope type 等默认）。
    pub architecture: Architecture,
    /// 最大上下文长度（`{arch}.context_length`）。
    pub ctx_len: usize,
    /// 层数（`{arch}.block_count`）。
    pub n_layer: usize,
    /// 隐藏维（`{arch}.embedding_length`）。
    pub hidden_size: usize,
    /// 查询头数（`{arch}.attention.head_count`）。
    pub q_heads: usize,
    /// KV 头数（`{arch}.attention.head_count_kv`；缺省 = 查询头数，MHA 降级）。
    pub kv_heads: usize,
    /// 每头键维（`{arch}.attention.key_length`；缺省按解析链推导）。
    pub head_dim: usize,
    /// 每头值维（`{arch}.attention.value_length`；缺省 = head_dim）。
    pub value_dim: usize,
    /// RoPE 维数（`{arch}.rope.dimension_count`；缺省 = head_dim）。
    pub rope_dim: usize,
    /// RoPE 频率基数（`{arch}.attention.rope_freq_base`；缺省见 [`Architecture::default_rope_freq_base`]）。
    pub rope_theta: f32,
    /// RoPE 类型（架构派生）。
    pub rope_type: RopeType,
    /// RMSNorm 层 epsilon（`{arch}.attention.layer_norm_rms_epsilon`；缺省 1e-5）。
    pub rms_eps: f32,
    /// FFN 中间维（`{arch}.feed_forward_length`）。
    pub ffn_hidden: usize,
    /// 词表大小（`{arch}.vocab_size`）。
    pub vocab_size: usize,
    /// BOS token id（`tokenizer.ggml.bos_token_id`；缺省无）。
    pub bos_id: Option<u32>,
    /// EOS token id（`tokenizer.ggml.eos_token_id`；缺省无）。
    pub eos_id: Option<u32>,
    /// UNK token id（`tokenizer.ggml.unk_token_id`；缺省无）。
    pub unk_id: Option<u32>,
    /// q/k head norm（Qwen3 系：RoPE 前对 q/k 逐头 RMSNorm）。
    pub head_norm: bool,
}

/// 配置解析错误（消息含键名，便于定位）。
#[derive(Debug)]
pub enum ArchError {
    /// 底层 GGUF 元数据访问错误（类型不匹配等）。
    Gguf(GgufError),
    /// `general.architecture` 不在白名单。
    UnknownArchitecture(
        /// `general.architecture` 实际字符串值。
        String,
    ),
    /// 必需键缺失。
    MissingKey {
        /// 缺失的元数据键名。
        key: String,
    },
    /// 值非法（NaN/越界/推导失败等）。
    InvalidValue {
        /// 非法值所在的元数据键名。
        key: String,
        /// 非法原因。
        why: &'static str,
    },
}

impl fmt::Display for ArchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArchError::Gguf(e) => write!(f, "gguf metadata error: {e}"),
            ArchError::UnknownArchitecture(arch) => {
                write!(f, "unknown architecture `{arch}` (supported: llama, qwen2)")
            }
            ArchError::MissingKey { key } => write!(f, "missing required metadata key `{key}`"),
            ArchError::InvalidValue { key, why } => {
                write!(f, "invalid value for metadata key `{key}`: {why}")
            }
        }
    }
}

impl std::error::Error for ArchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ArchError::Gguf(e) => Some(e),
            _ => None,
        }
    }
}

impl From<GgufError> for ArchError {
    fn from(e: GgufError) -> Self {
        ArchError::Gguf(e)
    }
}

/// 由元数据表构造类型化配置（fail-closed：缺键/非法值 → 错误，消息含键名）。
pub fn from_gguf_meta(meta: &ModelMeta) -> Result<LlamaConfig, ArchError> {
    let arch_str = required_str(meta, "general.architecture")?;
    let architecture = Architecture::parse(arch_str)
        .ok_or_else(|| ArchError::UnknownArchitecture(arch_str.to_string()))?;

    let ctx_len = required_u32(meta, &key(architecture, "context_length"))? as usize;
    let n_layer = required_u32(meta, &key(architecture, "block_count"))? as usize;
    let hidden_size = required_u32(meta, &key(architecture, "embedding_length"))? as usize;
    let ffn_hidden = required_u32(meta, &key(architecture, "feed_forward_length"))? as usize;
    // vocab_size 解析链（014 T1 增量——Qwen 官方 GGUF 省略 {arch}.vocab_size）：
    // {arch}.vocab_size → tokenizer.ggml.tokens 数组长度 → MissingKey
    // （llama.cpp 同款链路；真实档案无该键，见 specs/014 T1）
    let vocab_size = match meta.meta_u32(&key(architecture, "vocab_size"))? {
        Some(v) => v as usize,
        None => meta
            .meta_array_str("tokenizer.ggml.tokens")?
            .map(|t| t.len())
            .ok_or_else(|| ArchError::MissingKey { key: key(architecture, "vocab_size") })?,
    };

    let q_heads = required_u32(meta, &key(architecture, "attention.head_count"))? as usize;
    if q_heads == 0 {
        return Err(ArchError::InvalidValue {
            key: key(architecture, "attention.head_count"),
            why: "query head count must be > 0",
        });
    }
    let kv_heads = optional_u32(meta, &key(architecture, "attention.head_count_kv"))?
        .map_or(q_heads, |v| v as usize);
    if kv_heads == 0 || kv_heads > q_heads {
        return Err(ArchError::InvalidValue {
            key: key(architecture, "attention.head_count_kv"),
            why: "kv heads must be in 1..=query heads",
        });
    }

    let rms_eps =
        optional_f32(meta, &key(architecture, "attention.layer_norm_rms_epsilon"))?.unwrap_or(1e-5);

    // head_dim 解析链：key_length → rope.dimension_count → embedding/heads（整除）
    let head_dim = match optional_u32(meta, &key(architecture, "attention.key_length"))? {
        Some(v) => v as usize,
        None => match optional_u32(meta, &key(architecture, "rope.dimension_count"))? {
            Some(v) => v as usize,
            None => {
                // MSRV 1.85（workspace rust-version）；is_multiple_of 需 1.87+
                #[allow(clippy::manual_is_multiple_of)]
                if hidden_size % q_heads != 0 {
                    return Err(ArchError::InvalidValue {
                        key: key(architecture, "attention.key_length"),
                        why: "cannot derive head_dim: embedding_length not divisible by head_count \
                              (and neither attention.key_length nor rope.dimension_count present)",
                    });
                }
                hidden_size / q_heads
            }
        },
    };
    if head_dim == 0 {
        return Err(ArchError::InvalidValue {
            key: key(architecture, "attention.key_length"),
            why: "head_dim must be > 0",
        });
    }

    let value_dim = optional_u32(meta, &key(architecture, "attention.value_length"))?
        .map_or(head_dim, |v| v as usize);
    if value_dim == 0 {
        return Err(ArchError::InvalidValue {
            key: key(architecture, "attention.value_length"),
            why: "value dim must be > 0",
        });
    }

    let rope_dim = optional_u32(meta, &key(architecture, "rope.dimension_count"))?
        .map_or(head_dim, |v| v as usize);
    if rope_dim == 0 {
        return Err(ArchError::InvalidValue {
            key: key(architecture, "rope.dimension_count"),
            why: "rope dim must be > 0",
        });
    }

    let rope_theta = optional_f32(meta, &key(architecture, "attention.rope_freq_base"))?
        .unwrap_or_else(|| architecture.default_rope_freq_base());
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(ArchError::InvalidValue {
            key: key(architecture, "attention.rope_freq_base"),
            why: "rope freq base must be finite and > 0",
        });
    }

    Ok(LlamaConfig {
        architecture,
        ctx_len,
        n_layer,
        hidden_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        rope_dim,
        rope_theta,
        rope_type: architecture.rope_type(),
        rms_eps,
        ffn_hidden,
        vocab_size,
        bos_id: optional_u32(meta, "tokenizer.ggml.bos_token_id")?,
        eos_id: optional_u32(meta, "tokenizer.ggml.eos_token_id")?,
        unk_id: optional_u32(meta, "tokenizer.ggml.unk_token_id")?,
        head_norm: architecture.has_head_norm(),
    })
}

/// 由 HF `config.json` 构造（模型文件统一对象——非 GGUF 输入面）。
///
/// 键映射（transformers 4.51 Qwen3/Qwen2/LLaMA 形态，fail-closed）：
/// - 架构：`architectures[0]`（"Qwen3ForCausalLM" 等）或 `model_type`
///   （"qwen3" / "qwen2" / "llama"；未知 → 错误）；
/// - `max_position_embeddings` / `num_hidden_layers` / `hidden_size` /
///   `intermediate_size` / `vocab_size`；
/// - 头数：`num_attention_heads` / `num_key_value_heads`（缺省 = q 头，MHA 降级）、
///   `head_dim`（缺省 = hidden / q_heads 推导）；
/// - `rope_theta`（缺省架构默认）、`rms_norm_eps`（缺省 1e-5）；
/// - `bos_token_id` / `eos_token_id`（数字直取）；
/// - Qwen3 → `head_norm = true`。
///
/// 零模型标识：仅架构类型字符串白名单（llama.cpp arch 表同款），无仓库/文件名。
pub fn from_hf_config(value: &serde_json::Value) -> Result<LlamaConfig, ArchError> {
    let model_type = value
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|a| a.as_str())
        .or_else(|| value.get("model_type").and_then(|m| m.as_str()))
        .ok_or_else(|| ArchError::MissingKey { key: "architectures".into() })?;
    let (architecture, head_norm) = match model_type {
        "LlamaForCausalLM" | "llama" => (Architecture::Llama, false),
        "Qwen2ForCausalLM" | "qwen2" => (Architecture::Qwen2, false),
        "Qwen3ForCausalLM" | "qwen3" => (Architecture::Qwen3, true),
        other => return Err(ArchError::UnknownArchitecture(other.to_string())),
    };

    let u = |key: &str| -> Result<usize, ArchError> {
        value
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| ArchError::MissingKey { key: key.into() })
    };
    let ctx_len = u("max_position_embeddings")?;
    let n_layer = u("num_hidden_layers")?;
    let hidden_size = u("hidden_size")?;
    let ffn_hidden = u("intermediate_size")?;
    let vocab_size = u("vocab_size")?;
    let q_heads = u("num_attention_heads")?;
    if q_heads == 0 {
        return Err(ArchError::InvalidValue {
            key: "num_attention_heads".into(),
            why: "query head count must be > 0",
        });
    }
    let kv_heads = value
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(q_heads);
    if kv_heads == 0 || kv_heads > q_heads {
        return Err(ArchError::InvalidValue {
            key: "num_key_value_heads".into(),
            why: "kv heads must be in 1..=query heads",
        });
    }
    let head_dim = value
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or_else(|| if hidden_size % q_heads != 0 { 0 } else { hidden_size / q_heads });
    if head_dim == 0 {
        return Err(ArchError::InvalidValue {
            key: "head_dim".into(),
            why: "cannot derive head_dim from hidden_size/head_count",
        });
    }
    let rope_theta = value
        .get("rope_theta")
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or_else(|| architecture.default_rope_freq_base());
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(ArchError::InvalidValue {
            key: "rope_theta".into(),
            why: "rope theta must be finite and > 0",
        });
    }
    let rms_eps =
        value.get("rms_norm_eps").and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(1e-5);
    let id =
        |key: &str| -> Option<u32> { value.get(key).and_then(|v| v.as_u64()).map(|v| v as u32) };

    Ok(LlamaConfig {
        architecture,
        ctx_len,
        n_layer,
        hidden_size,
        q_heads,
        kv_heads,
        head_dim,
        value_dim: head_dim,
        rope_dim: head_dim,
        rope_theta,
        rope_type: architecture.rope_type(),
        rms_eps,
        ffn_hidden,
        vocab_size,
        bos_id: id("bos_token_id"),
        eos_id: id("eos_token_id"),
        unk_id: id("unk_token_id"),
        head_norm,
    })
}

/// EOS 解析优先级（014 D8；vLLM 对齐）：
/// `generation_config.json` 的 `eos_token_id`（列表取首）> `config.json`
/// 的 `eos_token_id` > tokenizer 的 eos（`<|im_end|>` 类特殊 token）>
/// 模型族兜底（qwen3 → 151645，其余无）。
///
/// `eos_token_id` 兼收标量与数组（`generation_config.json` 常为
/// `[151645, 151643]` 形式）。调用于 serve/run 载入 tokenizer 之后，以便
/// 传入 `tokenizer.eos_token()`。
pub fn resolve_eos(
    cfg: &serde_json::Value,
    gen_cfg: Option<&serde_json::Value>,
    tokenizer_eos: Option<u32>,
) -> Option<u32> {
    let first_eos = |v: &serde_json::Value| -> Option<u32> {
        v.get("eos_token_id").and_then(|e| {
            e.as_u64().map(|i| i as u32).or_else(|| {
                e.as_array().and_then(|a| a.first()).and_then(|x| x.as_u64()).map(|i| i as u32)
            })
        })
    };
    if let Some(v) = gen_cfg.and_then(first_eos) {
        return Some(v);
    }
    if let Some(v) = first_eos(cfg) {
        return Some(v);
    }
    if let Some(v) = tokenizer_eos {
        return Some(v);
    }
    let mt = cfg.get("model_type").and_then(|m| m.as_str()).unwrap_or_default();
    if mt.starts_with("qwen3") {
        return Some(151_645); // `<|im_end|>`
    }
    None
}

/// `{arch}.{suffix}` 完整键名。
fn key(architecture: Architecture, suffix: &str) -> String {
    format!("{}.{suffix}", architecture.as_str())
}

fn required_str<'a>(meta: &'a ModelMeta, key: &str) -> Result<&'a str, ArchError> {
    meta.meta_str(key)?.ok_or_else(|| ArchError::MissingKey { key: key.to_string() })
}

fn required_u32(meta: &ModelMeta, key: &str) -> Result<u32, ArchError> {
    meta.meta_u32(key)?.ok_or_else(|| ArchError::MissingKey { key: key.to_string() })
}

fn optional_u32(meta: &ModelMeta, key: &str) -> Result<Option<u32>, ArchError> {
    Ok(meta.meta_u32(key)?)
}

fn optional_f32(meta: &ModelMeta, key: &str) -> Result<Option<f32>, ArchError> {
    Ok(meta.meta_f32(key)?)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败

    use super::*;
    use reinfer_gguf::MetaValue;

    /// Qwen2 形状元数据 fixture（虚构名，013 零模型标识铁律）。
    /// 0.5B 形状：hidden 896 / 24 层 / 14:2 heads / head_dim 64 / rope 1e6 / ffn 4864。
    fn qwen2_kvs() -> Vec<(String, MetaValue)> {
        vec![
            ("general.architecture".into(), MetaValue::Str("qwen2".into())),
            ("qwen2.context_length".into(), MetaValue::U32(32_768)),
            ("qwen2.embedding_length".into(), MetaValue::U32(896)),
            ("qwen2.block_count".into(), MetaValue::U32(24)),
            ("qwen2.attention.head_count".into(), MetaValue::U32(14)),
            ("qwen2.attention.head_count_kv".into(), MetaValue::U32(2)),
            ("qwen2.attention.layer_norm_rms_epsilon".into(), MetaValue::F32(1e-6)),
            ("qwen2.attention.key_length".into(), MetaValue::U32(64)),
            ("qwen2.attention.value_length".into(), MetaValue::U32(64)),
            ("qwen2.attention.rope_freq_base".into(), MetaValue::F32(1_000_000.0)),
            ("qwen2.rope.dimension_count".into(), MetaValue::U32(64)),
            ("qwen2.feed_forward_length".into(), MetaValue::U32(4_864)),
            ("qwen2.vocab_size".into(), MetaValue::U32(151_936)),
            ("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(0)),
            ("tokenizer.ggml.eos_token_id".into(), MetaValue::U32(151_645)),
            ("tokenizer.ggml.unk_token_id".into(), MetaValue::U32(151_643)),
        ]
    }

    fn meta(kvs: Vec<(String, MetaValue)>) -> ModelMeta {
        ModelMeta::from_kvs(kvs)
    }

    /// 覆盖：Qwen2 0.5B 形状全字段断言。
    #[test]
    fn qwen2_shapes_map_correctly() {
        let cfg = from_gguf_meta(&meta(qwen2_kvs())).expect("qwen2 配置可解析");
        assert_eq!(cfg.architecture, Architecture::Qwen2);
        assert_eq!(cfg.ctx_len, 32_768);
        assert_eq!(cfg.n_layer, 24);
        assert_eq!(cfg.hidden_size, 896);
        assert_eq!(cfg.q_heads, 14);
        assert_eq!(cfg.kv_heads, 2);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.value_dim, 64);
        assert_eq!(cfg.rope_dim, 64);
        assert_eq!(cfg.rope_theta, 1_000_000.0);
        assert_eq!(cfg.rope_type, RopeType::Neox);
        assert_eq!(cfg.rms_eps, 1e-6);
        assert_eq!(cfg.ffn_hidden, 4_864);
        assert_eq!(cfg.vocab_size, 151_936);
        assert_eq!(cfg.bos_id, Some(0));
        assert_eq!(cfg.eos_id, Some(151_645));
        assert_eq!(cfg.unk_id, Some(151_643));
    }

    /// 覆盖：Llama 形状 + 全部可选键缺省路径
    /// （head_dim 推导、MHA 降级、rope freq base 默认、eps 默认、特殊 token 缺省）。
    #[test]
    fn llama_defaults_and_derivation() {
        let kvs = vec![
            ("general.architecture".into(), MetaValue::Str("llama".into())),
            ("llama.context_length".into(), MetaValue::U32(4_096)),
            ("llama.embedding_length".into(), MetaValue::U32(4_096)),
            ("llama.block_count".into(), MetaValue::U32(32)),
            ("llama.attention.head_count".into(), MetaValue::U32(32)),
            ("llama.feed_forward_length".into(), MetaValue::U32(11_008)),
            ("llama.vocab_size".into(), MetaValue::U32(32_000)),
        ];
        let cfg = from_gguf_meta(&meta(kvs)).expect("llama 配置可解析");
        assert_eq!(cfg.architecture, Architecture::Llama);
        assert_eq!(cfg.kv_heads, 32, "缺 head_count_kv → MHA 降级");
        assert_eq!(cfg.head_dim, 128, "head_dim = 4096/32 推导");
        assert_eq!(cfg.value_dim, 128);
        assert_eq!(cfg.rope_dim, 128);
        assert_eq!(cfg.rope_theta, 10_000.0, "缺 freq_base → llama 默认");
        assert_eq!(cfg.rope_type, RopeType::Norm);
        assert_eq!(cfg.rms_eps, 1e-5, "缺 eps → llama.cpp 默认");
        assert_eq!(cfg.bos_id, None);
    }

    /// 用例①：缺 `{arch}.vocab_size` → 错误消息含键名。
    #[test]
    fn missing_vocab_size_derives_from_tokens_array() {
        // 014 T1：Qwen 官方 GGUF 省略 {arch}.vocab_size——须从 tokenizer.ggml.tokens
        // 数组长度推断（llama.cpp 同款链路）。
        let mut kvs: Vec<(String, MetaValue)> =
            qwen2_kvs().into_iter().filter(|(k, _)| k != "qwen2.vocab_size").collect();
        kvs.push((
            "tokenizer.ggml.tokens".into(),
            MetaValue::Array(reinfer_gguf::ArrayValue::Str(vec![
                "a".into(),
                "b".into(),
                "c".into(),
            ])),
        ));
        let cfg = from_gguf_meta(&meta(kvs)).unwrap();
        assert_eq!(cfg.vocab_size, 3);
    }

    #[test]
    fn missing_vocab_size_and_tokens_errors_with_key_name() {
        let kvs: Vec<(String, MetaValue)> =
            qwen2_kvs().into_iter().filter(|(k, _)| k != "qwen2.vocab_size").collect();
        let err = from_gguf_meta(&meta(kvs)).unwrap_err();
        assert!(matches!(err, ArchError::MissingKey { .. }));
        assert!(err.to_string().contains("qwen2.vocab_size"));
    }

    /// 用例②：缺 `{arch}.attention.head_count` → 错误消息含键名。
    #[test]
    fn missing_head_count_errors_with_key_name() {
        let kvs: Vec<(String, MetaValue)> =
            qwen2_kvs().into_iter().filter(|(k, _)| k != "qwen2.attention.head_count").collect();
        let err = from_gguf_meta(&meta(kvs)).unwrap_err();
        assert!(matches!(err, ArchError::MissingKey { .. }));
        assert!(err.to_string().contains("qwen2.attention.head_count"));
    }

    /// 用例③：head_dim 推导失败（key_length/rope.dimension_count 均缺且整除失败）→ 错误。
    #[test]
    fn missing_head_dim_derivation_failure_errors() {
        let kvs: Vec<(String, MetaValue)> = qwen2_kvs()
            .into_iter()
            .filter(|(k, _)| k != "qwen2.attention.key_length" && k != "qwen2.rope.dimension_count")
            .map(|(k, v)| {
                // 896 % 14 == 0，需要破坏整除 → 改 embedding_length
                if k == "qwen2.embedding_length" { (k, MetaValue::U32(900)) } else { (k, v) }
            })
            .collect();
        let err = from_gguf_meta(&meta(kvs)).unwrap_err();
        assert!(matches!(err, ArchError::InvalidValue { .. }));
        assert!(err.to_string().contains("attention.key_length"));
    }

    /// 用例④：非法 rope freq base（NaN/0/负/+inf）→ 错误。
    #[test]
    fn invalid_rope_freq_base_errors() {
        for bad in [f32::NAN, 0.0, -1.0, f32::INFINITY] {
            let kvs: Vec<(String, MetaValue)> = qwen2_kvs()
                .into_iter()
                .map(|(k, v)| {
                    if k == "qwen2.attention.rope_freq_base" {
                        (k, MetaValue::F32(bad))
                    } else {
                        (k, v)
                    }
                })
                .collect();
            let err = from_gguf_meta(&meta(kvs)).unwrap_err();
            assert!(matches!(err, ArchError::InvalidValue { .. }), "rope_freq_base {bad} 必须报错");
            assert!(err.to_string().contains("qwen2.attention.rope_freq_base"));
        }
    }

    /// 用例⑤：kv_heads 越界（0 或 > 查询头数）→ 错误。
    #[test]
    fn invalid_kv_heads_errors() {
        for bad in [0u32, 15, 100] {
            let kvs: Vec<(String, MetaValue)> = qwen2_kvs()
                .into_iter()
                .map(|(k, v)| {
                    if k == "qwen2.attention.head_count_kv" {
                        (k, MetaValue::U32(bad))
                    } else {
                        (k, v)
                    }
                })
                .collect();
            let err = from_gguf_meta(&meta(kvs)).unwrap_err();
            assert!(matches!(err, ArchError::InvalidValue { .. }), "kv_heads={bad} 必须报错");
            assert!(err.to_string().contains("attention.head_count_kv"));
        }
    }

    /// 用例⑥：未知 architecture → 错误（消息含架构值）。
    #[test]
    fn unknown_architecture_errors() {
        let kvs: Vec<(String, MetaValue)> = qwen2_kvs()
            .into_iter()
            .map(|(k, v)| {
                if k == "general.architecture" {
                    (k, MetaValue::Str("gemma3".into()))
                } else {
                    (k, v)
                }
            })
            .collect();
        let err = from_gguf_meta(&meta(kvs)).unwrap_err();
        assert!(matches!(err, ArchError::UnknownArchitecture(ref arch) if arch == "gemma3"));
        assert!(err.to_string().contains("gemma3"));
    }

    /// 值类型不匹配（vocab_size 写成字符串）→ 底层 GGUF 错误传播。
    #[test]
    fn wrong_value_type_propagates_gguf_error() {
        let kvs: Vec<(String, MetaValue)> = qwen2_kvs()
            .into_iter()
            .map(|(k, v)| {
                if k == "qwen2.vocab_size" { (k, MetaValue::Str("151936".into())) } else { (k, v) }
            })
            .collect();
        let err = from_gguf_meta(&meta(kvs)).unwrap_err();
        assert!(matches!(err, ArchError::Gguf(GgufError::InvalidMetadata { .. })));
        assert!(err.to_string().contains("qwen2.vocab_size"));
    }

    /// 空元数据表 → 第一个必需键报错（general.architecture）。
    #[test]
    fn empty_meta_reports_architecture_key() {
        let err = from_gguf_meta(&ModelMeta::default()).unwrap_err();
        assert!(matches!(err, ArchError::MissingKey { .. }));
        assert!(err.to_string().contains("general.architecture"));
    }

    /// 5/2 非整除 GQA（014 D3 核验例）：合法通过（映射方向由装配层固定）。
    #[test]
    fn non_divisible_gqa_accepted() {
        let kvs: Vec<(String, MetaValue)> = qwen2_kvs()
            .into_iter()
            .map(|(k, v)| match k.as_str() {
                "qwen2.attention.head_count" => (k, MetaValue::U32(5)),
                "qwen2.attention.head_count_kv" => (k, MetaValue::U32(2)),
                _ => (k, v),
            })
            .collect();
        let cfg = from_gguf_meta(&meta(kvs)).expect("5/2 非整除 GQA 应解析通过");
        assert_eq!((cfg.q_heads, cfg.kv_heads), (5, 2));
    }
}

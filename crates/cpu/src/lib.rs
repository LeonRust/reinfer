//! reinfer-cpu：「无加速卡也能推理」的 CPU 全链路后端（007 spec C1/C2）。
//!
//! 数值：fp32 累加 naive（与 `kernels::refs` 同源语义——CPU 后端 = 参考
//! 实现本身的完整化）；单线程顺序、无锁无设备 → 同机跨机位级确定。
//! 生成语义（014 T9 必备块）：EOS 停 / `-n` 硬限 / logits 全 NaN 显式错误 /
//! embedding OOV → `RunError` / `-t 0` 短路 argmax（tie-break 首个最大）。

#![forbid(unsafe_code)]

pub mod model;
pub mod ops;

pub use model::Model;

use reinfer_arch::llama::ArchError;
use reinfer_gguf::schema::GgufError;

/// CPU 后端错误面。
#[derive(Debug)]
pub enum RunError {
    /// GGUF 数据错误。
    Gguf(GgufError),
    /// 架构配置错误。
    Arch(ArchError),
    /// 缺失张量。
    MissingTensor(String),
    /// embedding 越界 token。
    EmbeddingOov(u32),
    /// 权重形状不一致。
    WeightShape(String),
    /// 不支持 dtype。
    UnsupportedDtype(String),
    /// logits 全量 NaN（生成语义：显式错误——不 argmax 走号）。
    NaNLogits,
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Gguf(e) => write!(f, "gguf: {e}"),
            RunError::Arch(e) => write!(f, "arch: {e}"),
            RunError::MissingTensor(t) => write!(f, "missing tensor: {t}"),
            RunError::EmbeddingOov(t) => write!(f, "embedding OOV token: {t}"),
            RunError::WeightShape(s) => write!(f, "weight shape: {s}"),
            RunError::UnsupportedDtype(d) => write!(f, "unsupported dtype: {d}"),
            RunError::NaNLogits => write!(f, "logits contain only NaN — refuse to sample"),
        }
    }
}

impl std::error::Error for RunError {}

impl From<GgufError> for RunError {
    fn from(e: GgufError) -> Self {
        RunError::Gguf(e)
    }
}

impl From<ArchError> for RunError {
    fn from(e: ArchError) -> Self {
        RunError::Arch(e)
    }
}

/// 生成结果。
#[derive(Debug)]
pub struct Generation {
    /// 生成的 token 序列（不含 prompt；不含 EOS）。
    pub tokens: Vec<u32>,
    /// 是否因 EOS 终止。
    pub ended_by_eos: bool,
}

/// 单请求生成（已在 prompt ids 上 prefill 后连续采样）。
///
/// - `temperature == 0.0`：argmax（tie-break 首个最大——012 语义）；
/// - `temperature > 0.0`：本实现暂以 argmax 仅支持 temp=0（记录：temp>0 需
///   采样实现——012 host sampler 管线后续接线；当前返回错误避免静默）。
/// - `n_max`：硬限（超限即停，`ended_by_eos == false`）。
/// - EOS（config.eos_id）命中即停。
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &mut Model,
    prompt_ids: &[u32],
    n_max: u32,
    temperature: f32,
    eos_id: Option<u32>,
) -> Result<Generation, RunError> {
    if temperature != 0.0 {
        return Err(RunError::UnsupportedDtype(
            "temperature > 0 sampling not wired (record: 012 sampler host pipeline lies ahead)"
                .into(),
        ));
    }

    // prefill：逐 token 前向（KV 写入在 decode_step 内）
    for (pos, &tok) in prompt_ids.iter().enumerate() {
        let emb = model.embed_vec(tok)?;
        let _ = ops::decode_step(model, &emb, pos, pos + 1)?;
    }

    let mut tokens = Vec::new();
    let mut pos = prompt_ids.len();
    let mut cur = if prompt_ids.is_empty() { 0 } else { prompt_ids[prompt_ids.len() - 1] };
    while pos < pos.saturating_add(n_max as usize) {
        let emb = model.embed_vec(cur)?;
        let logits = ops::decode_step(model, &emb, pos, pos + 1)?;
        // NaN 全量 → 显式错误（014 T9）
        if logits.iter().all(|l| l.is_nan()) {
            return Err(RunError::NaNLogits);
        }
        // argmax（tie-break 首个最大）
        let next = argmax_first(&logits);
        if Some(next) == eos_id {
            return Ok(Generation { tokens, ended_by_eos: true });
        }
        tokens.push(next);
        cur = next;
        pos += 1;
    }
    Ok(Generation { tokens, ended_by_eos: false })
}

/// argmax（首个最大 tie-break——llama.cpp temp=0 语义一致）。
pub fn argmax_first(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, l) in logits.iter().enumerate().skip(1) {
        if l > &logits[best] {
            best = i;
        }
    }
    best as u32
}

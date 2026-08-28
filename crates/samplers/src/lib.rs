//! reinfer-samplers：采样链（成熟第三方 `llm-samplers` 的薄封装）。
//!
//! 生成语义分层：本 crate 只从 logits 产 token——温度/核采样/重复惩罚的
//! 组合与顺序对齐 llama.cpp 建议链（repeat → top-k → top-p → temperature →
//! 分布抽样；temp≤0 → greedy）。EOS / `-n` 硬限 / NaN 显式错误 / OOV 由
//! 编码/引擎层负责（014 T9）。
//!
//! `SamplingParams` 与 CLI（run）和 OpenAI 请求体（serve）共用同一结构。

#![forbid(unsafe_code)]

use llm_samplers::prelude::*;
use llm_samplers::types::Sampler as _; // sample_token/sampled_token_id trait
use rand::rngs::StdRng;
use rand::SeedableRng;

/// 采样参数（CLI/API 通用；缺省 = OpenAI 工程缺省：temp 1.0 之外默认无约束）。
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// 采样温度（≤0 → greedy）。
    pub temperature: f32,
    /// top-k 截断（None = 关）。
    pub top_k: Option<usize>,
    /// top-p 核采样（None = 关）。
    pub top_p: Option<f32>,
    /// 重复惩罚（须 >1.0 才启用；llama.cpp 语义：logits>0 除法、<0 乘法）。
    pub repeat_penalty: Option<f32>,
    /// 重复惩罚作用于最近 N token。
    pub repeat_last_n: usize,
    /// 随机种子（Some → 全链确定）。
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repeat_penalty: None,
            repeat_last_n: 64,
            seed: None,
        }
    }
}

/// 采样器（链上持有 RNG 与最近 token 历史）。
#[derive(Debug)]
pub struct Sampler {
    chain: SamplerChain,
    res: SimpleSamplerResources,
}

impl Sampler {
    /// 按参数构建链（顺序 = llama.cpp 建议：重复惩罚 → top-k → top-p →
    /// temperature → 分布抽样；temp≤0 → greedy）。
    pub fn new(params: &SamplingParams) -> Result<Self, SamplerError> {
        let rng: Box<dyn rand::RngCore + Send + Sync> = match params.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(StdRng::from_entropy()),
        };
        let mut chain = SamplerChain::new();
        if let Some(pen) = params.repeat_penalty.filter(|p| *p > 1.0) {
            chain.push_sampler(SampleRepetition::new(pen, params.repeat_last_n));
        }
        if let Some(k) = params.top_k.filter(|k| *k > 0) {
            chain.push_sampler(SampleTopK::new(k, 1));
        }
        if let Some(p) = params.top_p.filter(|p| *p > 0.0 && *p < 1.0) {
            chain.push_sampler(SampleTopP::new(p, 1));
        }
        if params.temperature > 0.0 {
            chain.push_sampler(SampleTemperature::new(params.temperature));
            chain.push_sampler(SampleRandDistrib::new());
        } else {
            chain.push_sampler(SampleGreedy::new());
        }
        Ok(Self { chain, res: SimpleSamplerResources::new(Some(rng), Some(Vec::new())) })
    }

    /// 采样一个 token（logits 行主序 [vocab]；NaN 由调用方先行拒绝）。
    pub fn sample(&mut self, logits: &[f32]) -> Result<u32, SamplerError> {
        let mut lg = Logits::try_from_iter(logits.iter().copied())
            .map_err(|e| SamplerError::Logits(e.to_string()))?;
        let tok = self
            .chain
            .sample_token(&mut self.res, &mut lg)
            .map_err(|e| SamplerError::Chain(e.to_string()))?;
        tok.ok_or(SamplerError::NoToken)
    }

    /// 记录最近生成 token（重复惩罚的状态喂给）。
    pub fn feed(&mut self, token: u32) {
        let mut push = |v: &mut Vec<u32>| v.push(token);
        self.res.with_last_tokens_mut(&mut push)
            .expect("last_tokens resource present when constructing Sampler");
    }

}

/// 采样错误面。
#[derive(Debug)]
pub enum SamplerError {
    /// logits 转换/形状错误。
    Logits(String),
    /// 采样链错误。
    Chain(String),
    /// 链未产出 token（空分布等）。
    NoToken,
}

impl std::fmt::Display for SamplerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SamplerError::Logits(e) => write!(f, "logits: {e}"),
            SamplerError::Chain(e) => write!(f, "sampler: {e}"),
            SamplerError::NoToken => write!(f, "sampler produced no token"),
        }
    }
}

impl std::error::Error for SamplerError {}

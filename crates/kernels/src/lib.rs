//! reinfer-kernels：KernelProvider trait + Registry + 选择器 + TuneDb
//!
//! safe 层：trait/注册/选择/调优/错误分类；后端实现（FFI）位于 `crates/cuda` 等。
#![forbid(unsafe_code)]

pub mod error;
pub mod logits;
pub mod provider;
pub mod refs;
pub mod sampler;
pub mod sampler_chain;
pub mod tune;

pub use error::LaunchError;
pub use logits::{DeviceBuffer, LogitsView};
pub use provider::{
    KernelProvider, LaunchArgs, OpConfig, ProviderChoice, ProviderSet, ProviderTier,
    SelectionCache, TuneEntry, select, select_attn, select_fmha,
};
pub use sampler_chain::{
    CpuSamplerChain, FallbackSamplerChain, RngState, SampleError, SamplerChain, SamplerCounters,
    SamplerImpl, SamplerParams, TieBreak, TokenOut, UnsupportedParam, select_sampler,
};
pub use tune::TuneDb;

/// 内存拷贝校验共享纯逻辑（CUDA / 昇腾共用）。
pub mod mem_check;

pub use mem_check::MemcpyKind;

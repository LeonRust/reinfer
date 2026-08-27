//! reinfer-arch：模型架构元数据 → 类型化运行配置。
//!
//! 消费 GGUF 元数据表（[`reinfer_gguf::ModelMeta`]），将 llama.cpp 格式的
//! 架构键族（`{arch}.{...}`）映射为类型化配置（[`llama::LlamaConfig`]），
//! 供 014 L3 装配层（tokenizer/张量加载/数值核）消费。
//! 键名与默认值对齐 llama.cpp `llm_arch_table` / `llm_load_hparams`；
//! 缺键/非法值 fail-closed（错误消息含键名）。
//! 锚：`specs/014-cuda-l3-single-request` T3。
//!
//! 纯 Rust、`#![forbid(unsafe_code)]`（与数据管道零 unsafe 约束一致）。

#![forbid(unsafe_code)]

pub mod llama;

pub use llama::{ArchError, Architecture, LlamaConfig, RopeType};

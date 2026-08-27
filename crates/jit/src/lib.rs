//! reinfer-jit：JitCache 共享层（specs/012；仿 FlashInfer 三段式 + FileLock）。
//!
//! 平台无关纯 Rust：零 unsafe、零 CUDA 依赖——CUDA（nvcc）与昇腾（bisheng）
//! 两个编译后端共用（002 边界条约：AscendC 编译流水线 = reinfer jit）。
//! 加载/launch 属平台 crate（`crates/cuda`）；本层只做 键/缓存/锁/meta/编译子进程。
#![forbid(unsafe_code)]

pub mod cache;
pub mod error;
pub mod key;
pub mod lock;
pub mod meta;
pub mod types;

pub use cache::JitCache;

pub use key::JitKey;
pub use meta::JLibMeta;
pub use types::{HeaderFile, KernelSource, ToolchainId};

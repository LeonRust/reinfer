//! reinfer-ascend：昇腾后端消费层（specs/011 mirror of CUDA L1）。
//!
//! 分层：SDK 表面原语在 cann-rs（0001/本 PR 的 memcpy）；本 crate 只做：
//! - 同构安全面（Context/Stream/Event/Buffer/MemRef/copy——与 crates/cuda 的 009 形状一致）；
//! - 错误归并（`LaunchError`：cann 码段白名单 → Oom/Driver/Fatal）；
//! - 共享校验（`reinfer_kernels::mem_check`，与 CUDA 侧同一套单测用例）。
//!
//! 线程亲和性：ACL 的 `aclrtSetDevice` 为 per-thread 绑定（同 CUDA 语义）。
//! `unsafe` 仅出现在 `buffer.rs` 的两次 cann memcpy 调用（AFER 校验先行）。

pub mod buffer;
pub mod context;
pub mod error;
pub mod stream;

pub use buffer::{AscendDeviceBuffer, AscendHostBuffer, AscendMemRef, copy, copy_async};
pub use context::{AscendContext, DeviceInfo};
pub use stream::{AscendEvent, AscendStream};

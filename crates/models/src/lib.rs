//! reinfer-models：纯 Rust 模型获取（013——ModelScope 优先，auto 回退 HuggingFace）。
//!
//! - 零 Python/外部 CLI；`ureq+rustls`（纯 Rust TLS、无 OpenSSL、无 tokio）；
//! - 爬虫/契约见 `specs/013-model-fetch/spec.md`（端点经代理实测钉死）；
//! - **模型标识零硬编码**：repo/文件名全部来自调用方/CLI/env（MODEL_RESOLVER 层）；
//! - 错误面：`LaunchError`（网络/校验/解析 → Fatal；ENOSPC → Oom）+ stderr 详情。

pub mod api;
pub mod download;
pub mod hf;
pub mod resolver;

pub use api::FileEntry;
pub use download::{ManifestEntry, Verify, download_file};
pub use reinfer_kernels::LaunchError;
pub use resolver::{ModelResolver, ModelSource, ModelSpec};

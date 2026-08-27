//! reinfer-gguf：GGUF 模型文件读取器（安全子集）
//!
//! - 头部/元数据/张量表一次性解析，权重数据惰性按需读取（`GgufReader::tensor_data`）。
//! - 纯 Rust、`#![forbid(unsafe_code)]`：不采用 mmap——memmap2 的 `map` 自 0.7 起为
//!   `unsafe`（映射期间文件被并发截断即 UB），与数据管道的零 unsafe 约束冲突；
//!   改用 `FileExt::read_at` 等价的惰性语义（见 `reader` 模块说明）。
//! - 锚：`specs/001-gguf-loader` T2/T3（数据格式约定）、`specs/014-cuda-l3-single-request` T1。
//! - 与 llama.cpp 的元数据层对拍属真机验证（014 T10）；本 crate 以格式自洽性测试兜底
//!   （字节级 golden fixture + proptest）。真实模型文件不进测试（013 模型标识零硬编码铁律）。

#![forbid(unsafe_code)]

pub mod reader;
pub mod schema;

#[cfg(test)]
mod fixture;

pub use reader::{GgufReader, ModelMeta};
pub use schema::{ArrayValue, GgufDtype, GgufError, GgufTensor, MetaValue};

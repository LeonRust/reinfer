//! reinfer-kernels：KernelProvider trait + Registry + 选择器 + TuneDb
//!
//! safe 层：trait/注册/选择/调优/错误分类；后端实现（FFI）位于 `crates/cuda` 等。
#![forbid(unsafe_code)]

pub mod error;

pub use error::LaunchError;

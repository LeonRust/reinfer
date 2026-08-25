//! reinfer-cuda：NVIDIA 窄 FFI —— cudarc、vendor cubin、CUDA Graph、VMM
//!
//! 本 crate 是 unsafe 宿主（FFI 经 cudarc）；engine 侧禁止反向依赖（见宪法 §2.1）。
#![allow(unsafe_code)] // 窄 FFI 宿主：unsafe 只允许出现在这里

/// 接线探针：feature `cuda` 开启时依赖链路必须可编译（003 T1）。
#[cfg(feature = "cuda")]
pub use cudarc as _cudarc;

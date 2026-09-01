//! reinfer-cuda：NVIDIA 窄 FFI —— cudarc、vendor cubin、CUDA Graph、VMM
//!
//! 本 crate 是 unsafe 宿主（FFI 经 cudarc）；engine 侧禁止反向依赖（见宪法 §2.1）。
#![allow(unsafe_code)] // 窄 FFI 宿主：unsafe 只允许出现在这里

/// 接线探针：feature `cuda` 开启时依赖链路必须可编译（003 T1）。
#[cfg(feature = "cuda")]
pub use cudarc as _cudarc;

/// `cudaError → LaunchError` 白名单映射（003 T3；纯 CPU 可测）。
pub mod error;

/// 设备信息纯数据与格式化（无 feature 依赖）。
pub mod device_info;

/// 设备上下文（feature `cuda`）。
#[cfg(feature = "cuda")]
pub mod context;

#[cfg(feature = "cuda")]
pub use context::CudaContext;
pub use device_info::DeviceInfo;

/// CUDA 流（feature `cuda`）。
#[cfg(feature = "cuda")]
pub mod stream;

/// CUDA 事件（feature `cuda`）。
#[cfg(feature = "cuda")]
pub mod event;

/// Jit 产物加载与 launch（012 C1；feature `cuda`）。
#[cfg(feature = "cuda")]
pub mod jit;

/// 目标架构解析（012；feature `cuda`；env 优先/设备实测兜底，无默认特判）。
#[cfg(feature = "cuda")]
pub mod arch;

/// vec_add Jit provider（012 C2；feature `cuda`）。
#[cfg(feature = "cuda")]
pub mod jit_provider;

pub mod attention;
pub mod decode;
pub mod dequant;
/// diff 内核（rms_norm/rope/masked_softmax；012 D2；feature `cuda`）。
#[cfg(feature = "cuda")]
pub mod diff;
pub mod engine;
pub mod fmha;
/// S1-9: fused decode-step kernels（融合组装载/plan 表/发射；feature `cuda`）。
#[cfg(feature = "cuda")]
pub mod fused;
pub mod gemm;
pub mod graph;
pub mod layer_fused;
mod send_sync;

#[cfg(feature = "cuda")]
pub use event::CudaEvent;
#[cfg(feature = "cuda")]
pub use stream::CudaStream;

/// 设备/主机内存缓冲（feature `cuda`）。
#[cfg(feature = "cuda")]
pub mod buffer;

#[cfg(feature = "cuda")]
pub use buffer::{DeviceBuffer, HostBuffer, MemRef, copy, copy_async};

/// GPU sampler 链（006-2 T3C；feature `cuda`）：单 launch 内核 +
/// `GpuSamplerChain`（penalty+softmax+topk/argmax，单流，D2 三层契约）。
#[cfg(feature = "cuda")]
pub mod sampler;

#[cfg(feature = "cuda")]
pub use sampler::GpuSamplerChain;
// MemcpyKind 共享于 kernels（昇腾后端复用同一校验逻辑）
pub use reinfer_kernels::MemcpyKind;

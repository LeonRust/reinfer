//! 窄 FFI 宿主惯例：CUDA 句柄的跨线程共享标注。
//!
//! CUDA 驱动/运行时 API 的 handle 语义均为线程安全（cudaStream/cublas
//! handle/CUlibrary 可在任意线程使用——注意每线程 CUDA *context* 需
//! `cuCtxSetCurrent` 绑定，launch 路径已由 `jit::CtxGuard` 每次设置）；
//! 此处显式标注的对象均附带 RAII 生命周期与所有权（Pin 到进程单实例），
//! 跨线程的**并发使用**由宿主串行化（serve 的 `Mutex<Engine>` / 单流）。

#![allow(unsafe_code)]

use crate::buffer::HostBuffer;
use crate::event::CudaEvent;
use crate::gemm::Gemm;
use crate::jit::JLib;
use crate::jit::KernelFn;
use crate::stream::CudaStream;

// SAFETY: CUDA 句柄线程安全（驱动文档）；生命周期由 RAII 保证；host 侧并发
// 使用按模块契约串行（stream/engine Mutex），launch 前 per-thread context 绑定。
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}
unsafe impl Send for Gemm {}
unsafe impl Sync for Gemm {}
unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}
unsafe impl Send for JLib {}
unsafe impl Sync for JLib {}
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}
unsafe impl Send for KernelFn {}
unsafe impl Sync for KernelFn {}

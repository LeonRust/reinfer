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

// SAFETY: CUDA graph handle 线程安全（驱动文档）；所有权 RAII（Drop 时销毁
// graph/exec）；宿主并发使用按既有契约串行（engine Mutex + capture 全局锁）。
unsafe impl Send for crate::graph::GraphPool {}
unsafe impl Sync for crate::graph::GraphPool {}
unsafe impl Send for crate::graph::GraphExec {}
unsafe impl Sync for crate::graph::GraphExec {}

// SAFETY: GemmPlan 稳定 host cell（pointer-stable staging）——cell 生命周期与
// engine 同构（RAII），内容仅 host 侧读写；launch 时驱动读 cell（C3 纪律）。
unsafe impl Send for crate::gemm::GemmPlan {}
unsafe impl Sync for crate::gemm::GemmPlan {}

// SAFETY: S1-9 融合内核单元：FusedGeom 的原生指针（partials 段/plan table）
// 均指向 `FusedDecodeKernels` 自身持有的 DeviceBuffer 与 engine 持有的稳定
// 缓冲（RAII 生命周期与 engine 同构，build_plans 后固定不再移动）；跨线程
// 并发使用由宿主串行化（engine Mutex + 单流），launch 前 per-thread context
// 绑定（CtxGuard）——与 GemmPlan 同一契约。
unsafe impl Send for crate::fused::FusedGeom {}
unsafe impl Sync for crate::fused::FusedGeom {}
unsafe impl Send for crate::fused::FusedDecodeKernels {}
unsafe impl Sync for crate::fused::FusedDecodeKernels {}

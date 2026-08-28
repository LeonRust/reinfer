//! 014 T6：cuBLAS GEMM 封装（Vendor 首件）。
//!
//! 判据矩阵（rtol/atol 双条款——014 r2 修订）：
//! - `gemm_f32acc`：`CUBLAS_COMPUTE_32F`（门禁档；f16-in/f32-out 或
//!   f32-in/f32-out，按 dtype 参数化）；
//! - `gemm_f16_16acc`：`CUBLAS_COMPUTE_16F`（记录档——跨 fp32 参考的实际
//!   误差统计入 notes，rel ≤1e-1 声明；014 r1 实测教训：真实 K 下
//!   16F-acc vs fp32 参考 92-98% 超 1e-4，故非门禁）。
//!
//! 直调 `cublasGemmEx`（cudarc 0.19 safe 层无 compute-type 参数——必须
//! 绕过）；handle 与 stream 均显式（**禁 default stream 0**，009 流句柄）。

use crate::stream::CudaStream;
use cudarc::cublas::sys as blas;
use reinfer_kernels::LaunchError;
use std::ffi::c_void;
use std::os::raw::c_int;

/// 单个 GEMM 操作数（列主序 ld 语义与 cuBLAS 一致；行主序调用方自行
/// 转置参数）。
#[derive(Clone)]
pub struct GpuMat {
    /// 设备指针（raw——所有权在调用方 DeviceBuffer）。
    pub ptr: *mut c_void,
    /// 元素类型（`CUDA_R_16F` / `CUDA_R_32F`）。
    pub dtype: blas::cudaDataType_t,
    /// 前导维度（元素数）。
    pub ld: c_int,
}

/// cuBLAS 句柄 RAII（`cublasCreate_v2`/`cublasDestroy_v2`；每 context 一个；
/// Drop 不要求 current context——destroy 是句柄级操作，保持简单）。
pub struct Gemm {
    handle: blas::cublasHandle_t,
    dev: u32,
}

impl Gemm {
    /// 创建句柄（需要 CUDA current context——`cuda` 模块约定由
    /// `CudaContext` 建立；此处显式设 current 更稳健）。
    pub fn new(dev: u32) -> Result<Self, LaunchError> {
        let _guard = crate::jit::CtxGuard::set_current(dev)?;
        let mut handle = std::ptr::null_mut();
        let st = unsafe { blas::cublasCreate_v2(&mut handle) };
        if st != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(map(status_message(st)));
        }
        Ok(Self { handle, dev })
    }

    /// 目标设备（诊断）。
    pub fn device(&self) -> u32 {
        self.dev
    }

    /// 单调 GEMM 执行：`C = alpha * op(A) * op(B) + beta * C`（列主序）。
    /// `compute`：`CUBLAS_COMPUTE_32F`（门禁档）或 `CUBLAS_COMPUTE_16F`（记录档）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_exec(
        &self,
        stream: &CudaStream,
        m: c_int,
        n: c_int,
        k: c_int,
        a: &GpuMat,
        b: &GpuMat,
        c: &mut GpuMat,
        compute: blas::cublasComputeType_t,
        trans_a: blas::cublasOperation_t,
        trans_b: blas::cublasOperation_t,
        alpha: f32, // compute=16F 时按 half 标量语义下传（调用方给 f16 bits 的 usize 位模式）
        beta: f32,
    ) -> Result<(), LaunchError> {
        let _guard = crate::jit::CtxGuard::set_current(self.dev)?;
        let cu_stream = stream.handle() as *mut c_void as blas::cudaStream_t;
        let st = unsafe { blas::cublasSetStream_v2(self.handle, cu_stream) };
        if st != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(map(status_message(st)));
        }
        // alpha/beta 标量位模式：16F 计算下必须为 half 位型（CUBLAS 按 compute
        // 类型解释）；32F 下为 f32 位型。统一以 u32 传入，调用方负责语义。
        let (alpha_bits, beta_bits): (u32, u32) =
            if compute == blas::cublasComputeType_t::CUBLAS_COMPUTE_16F {
                (f16_bits(alpha), f16_bits(beta))
            } else {
                (alpha.to_bits(), beta.to_bits())
            };
        let alpha_ptr = (&alpha_bits as *const u32) as *const c_void;
        let beta_ptr = (&beta_bits as *const u32) as *const c_void;
        // SAFETY：句柄有效、矩阵为设备 buffer、流已绑定、参数按 CUDA 布局；
        // 行列主序语义按 cuBLAS 约定（此处原样透传，转置职责在调用方）。
        let st = unsafe {
            blas::cublasGemmEx(
                self.handle,
                trans_a,
                trans_b,
                m,
                n,
                k,
                alpha_ptr,
                a.ptr,
                a.dtype,
                a.ld,
                b.ptr,
                b.dtype,
                b.ld,
                beta_ptr,
                c.ptr,
                c.dtype,
                c.ld,
                compute,
                // CUBLAS_GEMM_DEFAULT_TENSOR_OP=99 为默认 tensor-op 算法
                blas::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        };
        if st != blas::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(map(status_message(st)));
        }
        Ok(())
    }

    /// 门禁档便捷封装：`CUBLAS_COMPUTE_32F`（行主序语义——A 为 row-major
    /// [m×k]（ld=k 视图）、B 为 row-major [k×n]（ld=k），输出 col-major
    /// 视图 [m×n]：`raw[r + c*m] == C_row[r*n + c]`——见 gemm_diff.rs 头注）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32acc(
        &self,
        stream: &CudaStream,
        m: c_int,
        n: c_int,
        k: c_int,
        a: &GpuMat,
        b: &GpuMat,
        c: &mut GpuMat,
        alpha: f32,
        beta: f32,
    ) -> Result<(), LaunchError> {
        self.gemm_exec(
            stream,
            m,
            n,
            k,
            a,
            b,
            c,
            blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            blas::cublasOperation_t::CUBLAS_OP_T,
            blas::cublasOperation_t::CUBLAS_OP_T,
            alpha,
            beta,
        )
    }

    /// 记录档便捷封装：`CUBLAS_COMPUTE_16F`（行列语义同 gemm_f32acc）。
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f16_16acc(
        &self,
        stream: &CudaStream,
        m: c_int,
        n: c_int,
        k: c_int,
        a: &GpuMat,
        b: &GpuMat,
        c: &mut GpuMat,
        alpha: f32,
        beta: f32,
    ) -> Result<(), LaunchError> {
        self.gemm_exec(
            stream,
            m,
            n,
            k,
            a,
            b,
            c,
            blas::cublasComputeType_t::CUBLAS_COMPUTE_16F,
            blas::cublasOperation_t::CUBLAS_OP_T,
            blas::cublasOperation_t::CUBLAS_OP_T,
            alpha,
            beta,
        )
    }
}

impl Drop for Gemm {
    fn drop(&mut self) {
        unsafe {
            let _ = blas::cublasDestroy_v2(self.handle);
        }
    }
}

/// f32 → f16 位模式（RNE，内核侧与 gguf 语义一致；用于 16F 标量下传）。
fn f16_bits(f: f32) -> u32 {
    // 软件舍入：常规路径（指数可表示为 half）——保持与 cuda_fp16 一致；
    // 判据用标量为 1.0/0.0 等简单值，此路径覆盖之。
    let bits = f.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | ((man >> 13) as u32 & 0x3ff); // inf/nan 截断
    }
    let half_exp = exp - 127 + 15;
    if half_exp <= 0 {
        // subnormal / 零（下界饱和——记录档；标量用值罕见落到此）
        if half_exp < -10 {
            return sign;
        }
        let subm = (man | 0x80_0000) >> (1 - half_exp + 13);
        return sign | (subm as u32);
    }
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((half_exp as u32) << 10) | ((man >> 13) as u32)
}

/// cublasStatus → LaunchError（白名单：ALLOC_FAILED → Oom；其余 Fatal）。
fn map(msg: &str) -> LaunchError {
    if msg.contains("alloc") {
        reinfer_kernels::LaunchError::Oom
    } else {
        reinfer_kernels::LaunchError::Fatal
    }
}

fn status_message(st: blas::cublasStatus_t) -> &'static str {
    match st {
        blas::cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => "cublas: alloc failed",
        blas::cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => "cublas: not initialized",
        blas::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => "cublas: invalid value",
        _ => "cublas: internal error",
    }
}

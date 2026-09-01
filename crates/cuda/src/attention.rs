//! 014 T7：prefill attention 组装（两段 GEMM 32F + fp32 中间 + fp32 softmax）。
//!
//! 判据档：`CUBLAS_COMPUTE_32F` 两段 GEMM（QK^T、PV）+ fp32 中间 buffer
//! + fp32 softmax → f32 输出（调用方按「fp16 舍入后 ≤1 ulp」比较）。
//!
//! 参考：`kernels::refs::prefill_attn_ref`。
//!
//!
//!
//! 编排（单头 seq×d；**全程行主序**——S/P 物理行序 = 语义行序）：
//! 0) K^T = transpose(K)（设备 f16 转置 kernel）；
//! 1) S = Q·K^T（行主序 gemm 语义：OP_T/OP_T 组合——见 gemm_diff 头注）；
//! 2. mask 行主序注入（-inf 位累加）；
//! 3. 行 softmax（`masked_softmax_matrix`：S 已含 -inf 位；全 -inf 行 → 全 0）；
//!    输出（f32）cast f16 → P；
//! 4. O = P·V（行主序；P [seq×seq]、V [seq×d]）。

use crate::buffer::DeviceBuffer;
use crate::diff::DiffKernels;
use crate::gemm::{Gemm, GpuMat};
use crate::stream::CudaStream;
use cudarc::cublas::sys as blas;
use reinfer_core::DeviceId;
use reinfer_kernels::LaunchError;
use std::ffi::c_void;
use std::os::raw::c_int;

/// prefill 中间缓冲（S、P；按最大 seq 一次性分配）。
#[derive(Debug)]
pub struct PrefillScratch {
    /// S = QK^T（f32，seq×seq，行主序）。
    pub s: DeviceBuffer,
    /// P = softmax(S)（**f16**，seq×seq（行主序）——PV dtype 一致；014 T7）。
    pub p: DeviceBuffer,
    /// K^T 临时（f32 [d×seq] 行主序——判据档全 f32 路径）。
    pub kt: DeviceBuffer,
    /// S 行主序临时（f32 seq×seq——QKT 输出转置后）。
    pub sr: DeviceBuffer,
}

impl PrefillScratch {
    /// 分配 scratch（seq²×4 + seq²×2 + seq×d×2 字节）。
    pub fn alloc(dev: DeviceId, seq: usize, d: usize) -> Result<Self, LaunchError> {
        Ok(Self {
            s: DeviceBuffer::alloc(dev, seq * seq * 4)?,
            p: DeviceBuffer::alloc(dev, seq * seq * 2)?,
            kt: DeviceBuffer::alloc(dev, seq * d * 4)?,
            sr: DeviceBuffer::alloc(dev, seq * seq * 4)?,
        })
    }
}

/// 单头 prefill：q/k/v 行主序设备 buffer（[seq×d]、f16 或 f32 按 dtype）；
/// `mask` 已融合进 s 的 -inf 位（调用方上传含掩码的 QK^T 中间——见
/// `prefill_s_with_mask`）；out f32 [seq×d]。
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention(
    dev: u32,
    blas: &Gemm,
    sm: &DiffKernels,
    stream: &CudaStream,
    scratch: &mut PrefillScratch,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    mask: &DeviceBuffer,
    seq: usize,
    d: usize,
    out: &mut DeviceBuffer,
) -> Result<(), LaunchError> {
    let (seq_i, d_i) = (seq as c_int, d as c_int);

    // 0) K^T = transpose(K)（f32 [seq×d] → [d×seq] 行主序；判据档 32F 路径）。
    sm.launch_transpose_f32(
        dev,
        stream,
        k.as_ptr() as *const f32,
        scratch.kt.as_ptr() as *mut f32,
        seq as u32,
        d as u32,
    )?;

    // 1) S = Q·K^T（行主序 gemm：f32×f32、compute 32F——判据档）。
    let a =
        GpuMat { ptr: q.as_ptr() as *mut c_void, dtype: blas::cudaDataType_t::CUDA_R_32F, ld: d_i };
    let b = GpuMat {
        ptr: scratch.kt.as_ptr() as *mut c_void,
        dtype: blas::cudaDataType_t::CUDA_R_32F,
        ld: seq_i,
    };
    let mut c = GpuMat {
        ptr: scratch.s.as_ptr() as *mut c_void,
        dtype: blas::cudaDataType_t::CUDA_R_32F,
        ld: seq_i,
    };
    blas.gemm_exec(
        stream,
        seq_i,
        seq_i,
        d_i,
        &a,
        &b,
        &mut c,
        blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        blas::cublasOperation_t::CUBLAS_OP_T,
        blas::cublasOperation_t::CUBLAS_OP_T,
        1.0,
        0.0,
    )?;

    // 1a) S col-major（gemm 输出）→ 转置行主序 sr。
    sm.launch_transpose_f32(
        dev,
        stream,
        scratch.s.as_ptr() as *const f32,
        scratch.sr.as_ptr() as *mut f32,
        seq as u32,
        seq as u32,
    )?;

    // 1b) 掩码注入（行主序：-inf 位累加）。
    sm.launch_add_mask(
        dev,
        stream,
        scratch.sr.as_ptr() as *mut f32,
        mask.as_ptr() as *const f32,
        (seq * seq) as u32,
    )?;

    // 2) 行 softmax（sr 行主序 → 原地；P 保持 f32——fp32 中间判据档）。
    sm.launch_masked_softmax_matrix(
        dev,
        stream,
        scratch.sr.as_ptr() as *const f32,
        scratch.sr.as_ptr() as *mut f32,
        seq as u32,
        seq as u32,
    )?;

    // 3) O = P·V（行主序全 f32；P 即 sr、V [seq×d]）。
    let pa = GpuMat {
        ptr: scratch.sr.as_ptr() as *mut c_void,
        dtype: blas::cudaDataType_t::CUDA_R_32F,
        ld: seq_i,
    };
    let vb =
        GpuMat { ptr: v.as_ptr() as *mut c_void, dtype: blas::cudaDataType_t::CUDA_R_32F, ld: d_i };
    let mut oc = GpuMat {
        ptr: out.as_ptr() as *mut c_void,
        dtype: blas::cudaDataType_t::CUDA_R_32F,
        ld: seq_i,
    };
    blas.gemm_exec(
        stream,
        seq_i,
        d_i,
        seq_i,
        &pa,
        &vb,
        &mut oc,
        blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        blas::cublasOperation_t::CUBLAS_OP_T,
        blas::cublasOperation_t::CUBLAS_OP_T,
        1.0,
        0.0,
    )?;
    stream.synchronize()?;
    Ok(())
}

/// 生成因果下三角掩码：把 `s`（行主序 [seq×collen]）中 j > i 的位置改写为
/// -inf（f32 重填）。host 侧调用（构造设备值）。
pub fn mask_causal_inplace(s: &mut [f32], seq: usize, collen: usize) {
    for i in 0..seq {
        for j in (i + 1)..collen {
            s[i * collen + j] = f32::NEG_INFINITY;
        }
    }
}

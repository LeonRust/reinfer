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

use crate::buffer::DeviceBuffer;
use crate::graph::{KernelSpec, NodeRole, ParamLayout, PtrRole};
use crate::jit::{CtxGuard, JLib, KernelFn, cu_kernel_of, launch_rows};
use crate::stream::CudaStream;
use cudarc::cublas::sys as blas;
use cudarc::runtime::sys;
use reinfer_core::DeviceId;
use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::{JitCache, JitKey, KernelSource, probe_toolchain_for_arch};
use reinfer_kernels::LaunchError;
use std::ffi::c_void;
use std::os::raw::c_int;
use std::path::PathBuf;
use std::sync::Mutex;

/// 单个 GEMM 操作数（列主序 ld 语义与 cuBLAS 一致；行主序调用方自行
/// 转置参数）。
#[derive(Clone, Debug)]
pub struct GpuMat {
    /// 设备指针（raw——所有权在调用方 DeviceBuffer）。
    pub ptr: *mut c_void,
    /// 元素类型（`CUDA_R_16F` / `CUDA_R_32F`）。
    pub dtype: blas::cudaDataType_t,
    /// 前导维度（元素数）。
    pub ld: c_int,
}

/// 单次 GEMM 调用的完整参数格（S1-2 "graph prelude"：gemm1/gemm1r 调用面
/// 平整为固定 m/n/k + 固定 buffer 指针布局的稳定 cell 集合——CUDA-graph 波
/// （graph.rs `KernelSpec`/`ParamLayout::Gemm`/`PtrRole` + staging 种子模式）
/// 需要的 ≥第一版：每个 cell 有稳定地址（引擎持有 `GemmPlan` 的固定存储——
/// 如 `Vec<LayerGemmPlans>` 装载后不再移动），S1-3 波用 `PtrUpdate` 指向
/// 这些 cell 即可刷新 cublas 节点，无需重新推导参数。
///
/// 数值语义与旧 gemm1/gemm1r 完全一致：同样的 cublasGemmEx 参数
/// （handle/stream/compute/transposes/lds/alpha/beta/指针）逐位透传。
#[derive(Debug, Clone, Copy)]
pub struct GemmPlan {
    /// 输出行数（cublas m）。
    pub m: c_int,
    /// 输出列数（cublas n）。
    pub n: c_int,
    /// 归约深度（cublas k）。
    pub k: c_int,
    /// 操作数 A 指针（cell）。
    pub a: *mut c_void,
    /// 操作数 B 指针（cell）。
    pub b: *mut c_void,
    /// 输出 C 指针（cell）。
    pub c: *mut c_void,
    /// A 元素类型。
    pub a_dt: blas::cudaDataType_t,
    /// B 元素类型。
    pub b_dt: blas::cudaDataType_t,
    /// C 元素类型。
    pub c_dt: blas::cudaDataType_t,
    /// A 前导维度（列主序 ld 语义）。
    pub ld_a: c_int,
    /// B 前导维度。
    pub ld_b: c_int,
    /// C 前导维度。
    pub ld_c: c_int,
    /// 标量 alpha（按 compute 类型解释——32F 下 f32 位型）。
    pub alpha: f32,
    /// 标量 beta。
    pub beta: f32,
    /// 计算类型（`CUBLAS_COMPUTE_32F` 门禁档 / `CUBLAS_COMPUTE_16F` 记录档）。
    pub compute: blas::cublasComputeType_t,
    /// A 转置。
    pub trans_a: blas::cublasOperation_t,
    /// B 转置。
    pub trans_b: blas::cublasOperation_t,
}

impl GemmPlan {
    /// 行主序 f16-in / f32-out GEMM（gemm1 布局：`C = A[m×k] · B[k×n]`，
    /// A 行主序 ld=k、B 行主序 ld=n、输出列主序视图 ld=m；OP_T/OP_T）。
    pub fn row_major_f16(
        a: *const u16,
        b: *const u16,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Self {
        Self {
            m: m as c_int,
            n: n as c_int,
            k: k as c_int,
            a: a as *mut c_void,
            b: b as *mut c_void,
            c: c as *mut c_void,
            a_dt: blas::cudaDataType_t::CUDA_R_16F,
            b_dt: blas::cudaDataType_t::CUDA_R_16F,
            c_dt: blas::cudaDataType_t::CUDA_R_32F,
            ld_a: k as c_int,
            ld_b: n as c_int,
            ld_c: m as c_int,
            alpha: 1.0,
            beta: 0.0,
            compute: blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            trans_a: blas::cublasOperation_t::CUBLAS_OP_T,
            trans_b: blas::cublasOperation_t::CUBLAS_OP_T,
        }
    }

    /// 行主序 f32-in / f32-out GEMM（parity-f32 判据档；布局同
    /// `row_major_f16`——权重为 f16 值的 f32 展开）。
    pub fn row_major_f32(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Self {
        let mut p = Self::row_major_f16(a as *const u16, b as *const u16, c, m, n, k);
        p.a_dt = blas::cudaDataType_t::CUDA_R_32F;
        p.b_dt = blas::cudaDataType_t::CUDA_R_32F;
        p.c_dt = blas::cudaDataType_t::CUDA_R_32F;
        p
    }

    /// 行主序输出 GEMM（gemm1r 布局——prefill 批路径专用）：`C = A·B` 以
    /// OP_N/OP_N 交换操作数调用（cublas A = B^T（ld n）、cublas B = A^T
    /// （ld k）、m'=n、n'=m、ldc=n）——推导见 gemm1r 头注。
    pub fn col_major_swap_f16(
        a: *const u16,
        b: *const u16,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Self {
        Self {
            m: n as c_int,
            n: m as c_int,
            k: k as c_int,
            a: b as *mut c_void,
            b: a as *mut c_void,
            c: c as *mut c_void,
            a_dt: blas::cudaDataType_t::CUDA_R_16F,
            b_dt: blas::cudaDataType_t::CUDA_R_16F,
            c_dt: blas::cudaDataType_t::CUDA_R_32F,
            ld_a: n as c_int,
            ld_b: k as c_int,
            ld_c: n as c_int,
            alpha: 1.0,
            beta: 0.0,
            compute: blas::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            trans_a: blas::cublasOperation_t::CUBLAS_OP_N,
            trans_b: blas::cublasOperation_t::CUBLAS_OP_N,
        }
    }

    /// 本计划的 KernelSpec 声明（S1-3 波接线用：`GraphPool::declare_specs`
    /// 或 `capture` 的 specs 切片按 launch 序提供）。角色 = cublas 节点
    /// （handle/几何由 graph.rs 捕获后读回）；布局按本机实测 cublas
    /// kernelParams 面（graph.rs 读回注释）：m/n/k 落槽 0/1/2，
    /// `MAX_READBACK_SLOTS = 16` 之上限以内。`ptr_slots` 由调用方按
    /// 读回核对的 A/B/C 槽位声明——本波不执行 capture（BLOCKER-A 待 S1-3）。
    pub fn kernel_spec(&self, ptr_slots: Vec<(usize, PtrRole)>) -> KernelSpec {
        KernelSpec {
            role: NodeRole::CublasGemm,
            layout: ParamLayout::Gemm { slots: GEMM_PARAM_SLOTS, m: 0, n: 1, k: 2 },
            ptr_slots,
            handle: std::ptr::null_mut(),
            grid: sys::dim3 { x: 1, y: 1, z: 1 },
            block: sys::dim3 { x: 256, y: 1, z: 1 },
            shared: 0,
        }
    }
}

/// 本机 cublas GEMM 内核（OP_T/OP_T f16-in/32F-out 族）kernelParams 槽数：
/// graph.rs 读回实测 22 槽（槽 23 越界；`MAX_READBACK_SLOTS = 16` 上限内
/// 声明即可——本波仅作 spec 骨架）。
pub const GEMM_PARAM_SLOTS: usize = 22;

/// cuBLAS 句柄 RAII（`cublasCreate_v2`/`cublasDestroy_v2`；每 context 一个；
/// Drop 不要求 current context——destroy 是句柄级操作，保持简单）。
pub struct Gemm {
    handle: blas::cublasHandle_t,
    dev: u32,
}

impl std::fmt::Debug for Gemm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemm").field("dev", &self.dev).finish()
    }
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

    /// 按 `GemmPlan`（稳定参数格）执行一次 GEMM：把 plan 的 cell 展开为
    /// GpuMat 后透传 `gemm_exec`——与旧 gemm1/gemm1r 的 cublasGemmEx 调用
    /// 参数逐位一致（数值不变；S1-2 平整后的唯一执行路径）。
    pub fn execute(&self, stream: &CudaStream, plan: &GemmPlan) -> Result<(), LaunchError> {
        let amat = GpuMat { ptr: plan.a, dtype: plan.a_dt, ld: plan.ld_a };
        let bmat = GpuMat { ptr: plan.b, dtype: plan.b_dt, ld: plan.ld_b };
        let mut cmat = GpuMat { ptr: plan.c, dtype: plan.c_dt, ld: plan.ld_c };
        self.gemm_exec(
            stream,
            plan.m,
            plan.n,
            plan.k,
            &amat,
            &bmat,
            &mut cmat,
            plan.compute,
            plan.trans_a,
            plan.trans_b,
            plan.alpha,
            plan.beta,
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
        return sign | 0x7c00 | ((man >> 13) & 0x3ff); // inf/nan 截断
    }
    let half_exp = exp - 127 + 15;
    if half_exp <= 0 {
        // subnormal / 零（下界饱和——记录档；标量用值罕见落到此）
        if half_exp < -10 {
            return sign;
        }
        let subm = (man | 0x80_0000) >> (1 - half_exp + 13);
        return sign | subm;
    }
    if half_exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((half_exp as u32) << 10) | (man >> 13)
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

// ---------------------------------------------------------------------------
// JIT m=1 GEMM (decode-path gemv) — replaces cublas on the skinny m=1
// projections the decode step issues every step (q/k/v/o/gate/up/down per
// layer + lm_head). cuBLAS COMPUTE_32F gets ~15% SM efficiency on these
// shapes; a plain per-column dot-product kernel is bandwidth-bound instead
// (see kernels/gemm_m1.cu).
//
// Numeric semantics: f16 in / f32 out with fp32 accumulation — the same
// criterion tier as `Gemm::gemm_f32acc` (CUBLAS_COMPUTE_32F). The reduction
// ORDER differs from cublas' blocked reduction, so outputs drift at the
// D7 level (~1e-6..1e-5 rel; recorded, never bit-identical). Deterministic:
// fixed per-thread order, no atomics (bit-identical across repeated
// launches). The engine's parity-f32 tier (row_major_f32 plans) is not
// matched — it stays on cublas.
//
// Fallback discipline (engine side): env `REINFER_JGEMM=off` keeps the
// original cublas path; a launch failure falls back to cublas per call with
// the `jgemm_fallbacks` counter incremented (observable via the engine
// getter). Load failure (toolchain/compile) fails open to cublas with a
// note.
// ---------------------------------------------------------------------------

/// JIT m=1 GEMM unit: the `gemv_m1_f16f32` / `gemv_m1_f16f32_reduce`
/// kernel pair loaded via the standard JitCache pipeline (probe -> key ->
/// build_once -> load -> kernel), plus the S2-B+ batched pair
/// (`gemv_mb_f16f32` / `gemv_mb_f16f32_reduce`, same cubin — the batch
/// decode step's m=B projections; per-row bit-identical to the m=1 pair).
#[derive(Debug)]
pub struct Jgemm {
    lib: JLib,
    kernel: KernelFn,
    kernel_reduce: KernelFn,
    /// S2-B+: batched (m=B) phase-1 kernel — one launch over B rows.
    kernel_batch: KernelFn,
    /// S2-B+: batched (m=B) phase-2 kernel.
    kernel_batch_reduce: KernelFn,
    dev: u32,
    /// Slab-partials scratch (phase 1 output / phase 2 input), sized
    /// n*nslabs*4 on demand. Mutex: `launch` takes `&self` (the engine
    /// holds `Jgemm` behind `&self`); the decode thread is serial, so the
    /// lock never contends.
    partials: Mutex<Option<(usize, DeviceBuffer)>>,
}

impl Jgemm {
    /// Load the m=1 gemv kernel pair (JitCache pipeline; same shape as
    /// `DenseKernels::new`). `cache_dir` None -> REINFER_JIT_CACHE/XDG.
    pub fn new(dev: u32, arch: &str, cache_dir: Option<PathBuf>) -> Result<Self, LaunchError> {
        let tc = probe_toolchain_for_arch(arch)?;
        let src = KernelSource {
            name: "gemm_m1",
            src: include_str!("../kernels/gemm_m1.cu"),
            headers: vec![],
            flags: gencode_flags(arch)?,
            arch: arch.to_string(),
            toolchain_ver: tc.ver_line.clone(),
        };
        let cache = JitCache::open(cache_dir)?;
        let key = JitKey::new(&src, &tc);
        let (_, cubin_path) = cache.build_once(&key, &src, || compile_cubin(&src, &tc))?;
        let bytes = std::fs::read(&cubin_path).map_err(|_| LaunchError::Fatal)?;
        let lib = JLib::from_bytes(bytes)?;
        let kernel = lib.kernel("gemv_m1_f16f32")?;
        let kernel_reduce = lib.kernel("gemv_m1_f16f32_reduce")?;
        let kernel_batch = lib.kernel("gemv_mb_f16f32")?;
        let kernel_batch_reduce = lib.kernel("gemv_mb_f16f32_reduce")?;
        Ok(Self {
            lib,
            kernel,
            kernel_reduce,
            kernel_batch,
            kernel_batch_reduce,
            dev,
            partials: Mutex::new(None),
        })
    }

    /// Raw cubin library handle — the S1-3 graph declaration path takes the
    /// `CUkernel` for this kernel via `cu_kernel_of(lib, "gemv_m1_f16f32")`
    /// and declares it as `NodeRole::CustomKernel` with full pointer-slot
    /// coverage (graph wiring is the next wave; the KernelFn load surface
    /// is the readiness anchor: same CUkernel form capture records).
    pub fn raw_lib(&self) -> cudarc::driver::sys::CUlibrary {
        self.lib.raw()
    }

    /// Phase-2 kernel (`gemv_m1_f16f32_reduce`) launch handle — the S1-9
    /// fused decode path reuses it for the lm_head reduction (the fused
    /// unit owns the fused kernels; the shared reduce kernel stays here).
    #[must_use]
    pub fn kernel_reduce_fn(&self) -> KernelFn {
        self.kernel_reduce
    }

    /// Grid shape of the two-phase launch for an (n, k) plan: (ncols,
    /// nslabs) — the exact arithmetic `launch` performs. The graph
    /// declaration needs it to mirror the capture-time launch geometry
    /// (refresh rewrites the node's grid from the declared spec).
    ///
    /// S1-9b tuning (2026-08-31): the block target is 192 for very tall
    /// plans (2n < k — the down projection, k=3072) and 96 otherwise.
    /// Measured (engine profile, window 2): the down plan at 48 slabs
    /// (grid 192) cut ffn_d 0.848 -> 0.520 ms (DRAM-resident, more
    /// blocks hide the longer-k latency); doubling slabs on the
    /// short-k plans (g/u 8->16) slightly regressed them, and the o
    /// plan at 48 slabs regressed its segment (the p2_add_rms partial
    /// sums doubled) — hence the 2n < k rule isolates the down plan.
    /// Accumulation order changed vs the pre-tuning target (more
    /// partials per column) — D7 note, expected |Δ| <= 1e-6.
    #[must_use]
    pub fn shape(&self, n: c_int, k: c_int) -> (u32, u32) {
        let ncols = (n as u32).div_ceil(256).max(1);
        let target = if 2 * n < k { 192u32 } else { 96u32 };
        let nslabs = target.div_ceil(ncols).clamp(1, (k as u32 / 32).max(1));
        (ncols, nslabs)
    }

    /// Ensure the shared slab-partials scratch covers `n * nslabs * 4`
    /// bytes (grows monotonically; one buffer serves every plan) and return
    /// the stable device pointer. The pointer is stable once the maximum
    /// size is ensured, so the graph declaration can record it in its
    /// argument cells BEFORE the capture window opens — the eager launch
    /// then never allocates inside the window (cudaMalloc during capture is
    /// illegal).
    pub fn ensure_partials(&self, n: usize, nslabs: u32) -> Result<*mut f32, LaunchError> {
        let bytes = n * nslabs as usize * 4;
        let mut slot = self.partials.lock().expect("jgemm partials lock");
        let grow = match &*slot {
            Some((cap, _)) => *cap < bytes,
            None => true,
        };
        if grow {
            // DeviceBuffer::alloc needs a current context (launch sets the
            // same guard before allocating).
            let _guard = CtxGuard::set_current(self.dev)?;
            let buf = DeviceBuffer::alloc(DeviceId::new(self.dev), bytes)
                .map_err(|_| LaunchError::Fatal)?;
            *slot = Some((bytes, buf));
        }
        Ok(slot.as_ref().expect("just ensured").1.as_ptr() as *mut f32)
    }

    /// `CUkernel` handles of the two kernels — the handle form capture
    /// records for `cuLibraryLoadData` kernels (`cudaKernelFunctionTypeKernel`,
    /// graph.rs `FN_TYPE`); the graph declaration must use it for the
    /// refresh path (`KernelSpec.handle`).
    pub fn kernel_handles(&self) -> Result<(*mut c_void, *mut c_void), LaunchError> {
        Ok((
            cu_kernel_of(self.raw_lib(), "gemv_m1_f16f32")?,
            cu_kernel_of(self.raw_lib(), "gemv_m1_f16f32_reduce")?,
        ))
    }

    /// Whether `plan` is served by this kernel: m == 1, f16 in / f32 out,
    /// the row-major [k x n] B layout (ld_a = k, ld_b = n, ld_c = 1) with
    /// OP_T/OP_T and alpha=1/beta=0 — exactly `GemmPlan::row_major_f16`
    /// with m == 1 (the decode step's production channel).
    pub fn matches(&self, plan: &GemmPlan) -> bool {
        plan.m == 1
            && plan.a_dt == blas::cudaDataType_t::CUDA_R_16F
            && plan.b_dt == blas::cudaDataType_t::CUDA_R_16F
            && plan.c_dt == blas::cudaDataType_t::CUDA_R_32F
            && plan.trans_a == blas::cublasOperation_t::CUBLAS_OP_T
            && plan.trans_b == blas::cublasOperation_t::CUBLAS_OP_T
            && plan.alpha == 1.0
            && plan.beta == 0.0
            && plan.ld_a == plan.k
            && plan.ld_b == plan.n
            && plan.ld_c == 1
    }

    /// Launch the m=1 gemv for `plan` (must pass `matches`): C[1 x n] =
    /// A[1 x k] x B[k x n], block = 256, one thread per output column.
    /// Two-phase: phase 1 (`gemv_m1_f16f32`) splits k into nslabs slabs —
    /// grid = ncols * nslabs (the decode shapes' ncols alone is 4..12
    /// blocks, too few threads in flight to cover DRAM latency) — then
    /// phase 2 (`gemv_m1_f16f32_reduce`) sums the per-column partials in
    /// fixed ascending-slab order. Deterministic — see the kernel header.
    ///
    /// # Safety
    /// Same contract as every engine kernel launch: `plan` cells are valid
    /// device pointers of the declared dtype/sizes for this context; the
    /// stream is valid.
    pub fn launch(&self, stream: &CudaStream, plan: &GemmPlan) -> Result<(), LaunchError> {
        debug_assert!(self.matches(plan));
        let _guard = CtxGuard::set_current(self.dev)?;
        // Target ~96 grid blocks (>= 4/SM on 16 SMs); cap by k/32 so every
        // slab holds at least 32 k positions (per-position guards make any
        // tail exact anyway).
        let (ncols, nslabs) = self.shape(plan.n, plan.k);
        // Phase-1 partial scratch (n * nslabs f32), cached in `self` so it
        // outlives the async launches; realloc only on growth (the stream
        // serializes the kernels, so an old buffer is idle when replaced).
        // The graph path pre-ensures the maximum size at declaration build
        // so no allocation can happen inside a capture window.
        let partials_v = self.ensure_partials(plan.n as usize, nslabs)?;
        // C3 discipline (jit.rs header): kernelParams entries must be
        // addresses of LOCAL variables — no inline conversion chains.
        let a_v: *const u16 = plan.a as *const u16;
        let b_v: *const u16 = plan.b as *const u16;
        let n_v: c_int = plan.n;
        let k_v: c_int = plan.k;
        let nslabs_v: c_int = nslabs as c_int;
        let mut args1: [*mut c_void; 6] = [
            (&a_v as *const *const u16) as *mut c_void,
            (&b_v as *const *const u16) as *mut c_void,
            (&partials_v as *const *mut f32) as *mut c_void,
            (&n_v as *const c_int) as *mut c_void,
            (&k_v as *const c_int) as *mut c_void,
            (&nslabs_v as *const c_int) as *mut c_void,
        ];
        let grid1 = ncols * nslabs;
        // SAFETY: `plan` pointers valid (caller contract); `partials_v`
        // valid for n*nslabs*4 bytes (just ensured); locals for params.
        unsafe { launch_rows(self.kernel, stream, self.dev, grid1, 256, args1.as_mut_ptr())? };
        let c_v2: *mut f32 = plan.c as *mut f32;
        let n_v2: c_int = plan.n;
        let nslabs_v2: c_int = nslabs as c_int;
        let mut args2: [*mut c_void; 4] = [
            (&partials_v as *const *mut f32) as *mut c_void,
            (&c_v2 as *const *mut f32) as *mut c_void,
            (&n_v2 as *const c_int) as *mut c_void,
            (&nslabs_v2 as *const c_int) as *mut c_void,
        ];
        // SAFETY: same pointer contracts as above.
        unsafe { launch_rows(self.kernel_reduce, stream, self.dev, ncols, 256, args2.as_mut_ptr()) }
    }

    /// S2-B+: batched m-row launch — C[m x n] = A[m x k] x B[k x n] in ONE
    /// two-phase launch (the batch decode step's projections; the engine
    /// routes its m=B GEMMs here when this unit is loaded, else cublas).
    ///
    /// Contract (fixed, not a `GemmPlan` — the batch path has no plans):
    /// `a` is [m x k] f16 row-major (request rows contiguous, ld = k), `b`
    /// is the shared [k x n] f16 row-major weight matrix (ld = n), `c` is
    /// [m x n] f32 row-major — the engine's batch scratch layout, mirroring
    /// `gemm1r`'s cublas call with OP_N/OP_N. `m` rows with the SAME (n, k)
    /// geometry and weight matrix.
    ///
    /// Numerics: per-row arithmetic is byte-for-byte `gemm_m1_f16f32`'s
    /// (same slab split via `shape`, same stride-4 k walk, same fixed
    /// reduction trees) — row r is bit-identical to the m=1 path given the
    /// same A row and B matrix, so the batch step's GEMM surface matches
    /// the single-request path bitwise. Deterministic.
    ///
    /// The shared slab-partials scratch grows to m*n*nslabs (layout
    /// [m][nslabs][n] — row r's chunk at offset r*nslabs*n; the m=1 kernels
    /// use the same buffer from offset 0, so both paths coexist).
    ///
    /// # Safety
    /// Same contract as every engine kernel launch: `a`/`b`/`c` are valid
    /// device pointers of the declared dtype/sizes for this context; the
    /// stream is valid.
    pub fn launch_batch(
        &self,
        stream: &CudaStream,
        a: *const u16,
        b: *const u16,
        c: *mut f32,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(self.dev)?;
        let (ncols, nslabs) = self.shape(n as c_int, k as c_int);
        // Phase-1 partial scratch [m][nslabs][n] — same slot as the m=1
        // path (grows monotonically; the stream serializes the launches).
        let partials_v = self.ensure_partials(n * m, nslabs)?;
        // C3 discipline (jit.rs header): kernelParams entries must be
        // addresses of LOCAL variables — no inline conversion chains.
        let a_v: *const u16 = a;
        let b_v: *const u16 = b;
        let m_v: c_int = m as c_int;
        let n_v: c_int = n as c_int;
        let k_v: c_int = k as c_int;
        let nslabs_v: c_int = nslabs as c_int;
        let mut args1: [*mut c_void; 7] = [
            (&a_v as *const *const u16) as *mut c_void,
            (&b_v as *const *const u16) as *mut c_void,
            (&partials_v as *const *mut f32) as *mut c_void,
            (&m_v as *const c_int) as *mut c_void,
            (&n_v as *const c_int) as *mut c_void,
            (&k_v as *const c_int) as *mut c_void,
            (&nslabs_v as *const c_int) as *mut c_void,
        ];
        let grid1 = m as u32 * ncols * nslabs;
        // SAFETY: `a`/`b` valid (caller contract); `partials_v` valid for
        // m*n*nslabs*4 bytes (just ensured); locals for params.
        unsafe { launch_rows(self.kernel_batch, stream, self.dev, grid1, 256, args1.as_mut_ptr())? };
        let c_v2: *mut f32 = c;
        let m_v2: c_int = m as c_int;
        let n_v2: c_int = n as c_int;
        let nslabs_v2: c_int = nslabs as c_int;
        let mut args2: [*mut c_void; 5] = [
            (&partials_v as *const *mut f32) as *mut c_void,
            (&c_v2 as *const *mut f32) as *mut c_void,
            (&m_v2 as *const c_int) as *mut c_void,
            (&n_v2 as *const c_int) as *mut c_void,
            (&nslabs_v2 as *const c_int) as *mut c_void,
        ];
        // SAFETY: same pointer contracts as above.
        unsafe {
            launch_rows(
                self.kernel_batch_reduce,
                stream,
                self.dev,
                m as u32 * ncols,
                256,
                args2.as_mut_ptr(),
            )
        }
    }
}

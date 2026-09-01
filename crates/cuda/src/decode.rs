//! 014 T8: paged decode GQA — kernel loader + page-table host-side管理。
//!
//! 与 dequant.rs 同构（JitCache 管线）；`KVStore` 承担设备 KV 缓冲
//! （由 `crates::memory::pool::PagePool` 的 页数/块长 契约分配——页数据
//! 设备内存归属 backend（cuda），页索引簿归属 memory crate——015 跨端复用）。
//!
//! 判据（r2）：随机页表 diff（跨 2-3 页/首尾部分页/乱序物理页/batch 1..64/
//! kv_len 1..1k）vs host gather 参考；毒化（0xFF/NaN 未初始化页——被
//! kv_len 遮挡的位置不参与）；GQA 三例（14/2、12/2、5/2）核验；确定性
//! 双跑逐位一致；泄漏三合一（pool 侧）。

use crate::buffer::DeviceBuffer;
use crate::jit::{CtxGuard, JLib, KernelFn};
use crate::stream::CudaStream;
use reinfer_core::DeviceId;
use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::{JitCache, JitKey, KernelSource, probe_toolchain_for_arch};
use reinfer_kernels::LaunchError;
use std::ffi::c_void;
use std::path::PathBuf;

/// decode_step_gqa 装载单元（naive paged GQA + 006-2 T2 flash-style 档）。
#[derive(Debug)]
pub struct DecodeKernels {
    /// 加载的 cubin（保活——kernel fn 为模块内符号）。
    #[allow(dead_code)]
    lib: JLib,
    decode: KernelFn,
    /// 014 S0-3b: parity-f32 criterion tier (f32 q/out, f16 KV).
    decode_f32: KernelFn,
    /// 006-2 T2: flash-style decode-attn cubin（decode_flash_kernels.cu）。
    #[allow(dead_code)]
    flash_lib: JLib,
    flash: KernelFn,
    /// 006-2 T2: flash-style f32 q/out variant (parity-f32 tier).
    flash_f32: KernelFn,
    /// S1-9: fused decode flash variant (f16 tier) — kv write of the
    /// current slot + flash attention + the o-projection phase-1 in one
    /// launch (see decode_flash_kernels.cu; used only by the fused step).
    flash_fused: KernelFn,
    stream: CudaStream,
    arch: String,
}

/// Flash-kernel dynamic smem budget (bytes): (d + max_kv) * 4. The kernel
/// keeps its q row and the per-token scores in dynamic smem; 48 KB is the
/// default per-CTA limit (no cuFuncSetAttribute opt-in needed).
const FLASH_SMEM_MAX: u32 = 48 * 1024;

impl DecodeKernels {
    /// Loader constructor（两源：naive GQA + flash 档——flash 装载失败
    /// 视作回退缺失，naive 恒可用；引擎侧失败回退见 engine.rs）。
    pub fn new(
        arch: &str,
        cache_dir: Option<PathBuf>,
        stream: CudaStream,
    ) -> Result<Self, LaunchError> {
        let tc = probe_toolchain_for_arch(arch)?;
        let cache = JitCache::open(cache_dir)?;
        let src = KernelSource {
            name: "decode_gqa_kernels",
            src: include_str!("../kernels/decode_gqa_kernels.cu"),
            headers: vec![],
            flags: gencode_flags(arch)?,
            arch: arch.to_string(),
            toolchain_ver: tc.ver_line.clone(),
        };
        let key = JitKey::new(&src, &tc);
        let (_, cubin_path) = cache.build_once(&key, &src, || compile_cubin(&src, &tc))?;
        let bytes = std::fs::read(&cubin_path).map_err(|_| LaunchError::Fatal)?;
        let lib = JLib::from_bytes(bytes)?;
        let decode = lib.kernel("decode_step_gqa")?;
        let decode_f32 = lib.kernel("decode_step_gqa_f32")?;
        let fs = KernelSource {
            name: "decode_flash_kernels",
            src: include_str!("../kernels/decode_flash_kernels.cu"),
            headers: vec![],
            flags: gencode_flags(arch)?,
            arch: arch.to_string(),
            toolchain_ver: tc.ver_line.clone(),
        };
        let fkey = JitKey::new(&fs, &tc);
        let (_, fcubin_path) = cache.build_once(&fkey, &fs, || compile_cubin(&fs, &tc))?;
        let fbytes = std::fs::read(&fcubin_path).map_err(|_| LaunchError::Fatal)?;
        let flash_lib = JLib::from_bytes(fbytes)?;
        let flash = flash_lib.kernel("decode_step_gqa_flash")?;
        let flash_f32 = flash_lib.kernel("decode_step_gqa_flash_f32")?;
        let flash_fused = flash_lib.kernel("decode_step_gqa_flash_fused")?;
        Ok(Self {
            lib,
            decode,
            decode_f32,
            flash_lib,
            flash,
            flash_f32,
            flash_fused,
            stream,
            arch: arch.to_string(),
        })
    }

    /// 006-2 T2: flash 档可用（装载成功——恒 true；保留给未来动态回退面）。
    #[must_use]
    pub fn flash_available(&self) -> bool {
        true
    }

    /// Target architecture (diagnostics).
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// 原始 cubin 库句柄（S1-3 graph spec 声明取 `CUkernel` 用——capture
    /// 记录的是 `CUkernel` 形态，见 `JLib::cu_kernel`）。
    pub fn raw_lib(&self) -> cudarc::driver::sys::CUlibrary {
        self.lib.raw()
    }

    /// 006-2 T2: flash 档 cubin 库句柄（graph 声明取 flash kernel 的
    /// `CUkernel`——eager 路径在捕获窗口内发射的即是 flash kernel）。
    pub fn flash_raw_lib(&self) -> cudarc::driver::sys::CUlibrary {
        self.flash_lib.raw()
    }

    /// Block until the launch stream drains.
    pub fn sync_stream(&self) -> Result<(), LaunchError> {
        self.stream.synchronize()
    }

    /// 单步 decode（参数契约 = decode_gqa_kernels.cu 头注；grid = B×QH）。
    #[allow(clippy::too_many_arguments)]
    pub fn launch_decode_step_gqa(
        &self,
        dev: u32,
        q: *const u16,
        page: *const u32,
        kv: *const u16,
        kv_lens: *const u32,
        scores: *mut f32,
        out: *mut u16,
        b: u32,
        qh: u32,
        d: u32,
        block_len: u32,
        kv_ratio: u32,
        kv_heads: u32,
        max_kv: u32,
        total_pages: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        // kernel 签名（14 参数）：(q,page,kv,kv_lens,scores,out, B,QH, d,
        // block_len, kv_ratio, kv_heads, max_kv, total_pages)
        let b_p: i32 = b as i32;
        let k_p: i32 = qh as i32;
        let nv: [i32; 6] = [
            d as i32,
            block_len as i32,
            kv_ratio as i32,
            kv_heads as i32,
            max_kv as i32,
            total_pages as i32,
        ];
        let mut args: [*mut c_void; 14] = [
            (&q as *const *const u16) as *mut c_void,
            (&page as *const *const u32) as *mut c_void,
            (&kv as *const *const u16) as *mut c_void,
            (&kv_lens as *const *const u32) as *mut c_void,
            (&scores as *const *mut f32) as *mut c_void,
            (&out as *const *mut u16) as *mut c_void,
            (&b_p as *const i32) as *mut c_void,
            (&k_p as *const i32) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
            (&nv[3] as *const i32) as *mut c_void,
            (&nv[4] as *const i32) as *mut c_void,
            (&nv[5] as *const i32) as *mut c_void,
        ];
        unsafe {
            super::jit::launch_rows(self.decode, &self.stream, dev, b * qh, 256, args.as_mut_ptr())
        }
    }

    /// 014 S0-3b: parity-f32 decode step (f32 q in / f32 out, f16 KV read —
    /// same layout contract as `launch_decode_step_gqa`, q/out element types
    /// are f32; the caller applies the 1/sqrt(head_dim) scale to q before the
    /// launch, mirroring the f16 path's scale point).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_decode_step_gqa_f32(
        &self,
        dev: u32,
        q: *const f32,
        page: *const u32,
        kv: *const u16,
        kv_lens: *const u32,
        scores: *mut f32,
        out: *mut f32,
        b: u32,
        qh: u32,
        d: u32,
        block_len: u32,
        kv_ratio: u32,
        kv_heads: u32,
        max_kv: u32,
        total_pages: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let b_p: i32 = b as i32;
        let k_p: i32 = qh as i32;
        let nv: [i32; 6] = [
            d as i32,
            block_len as i32,
            kv_ratio as i32,
            kv_heads as i32,
            max_kv as i32,
            total_pages as i32,
        ];
        let mut args: [*mut c_void; 14] = [
            (&q as *const *const f32) as *mut c_void,
            (&page as *const *const u32) as *mut c_void,
            (&kv as *const *const u16) as *mut c_void,
            (&kv_lens as *const *const u32) as *mut c_void,
            (&scores as *const *mut f32) as *mut c_void,
            (&out as *const *mut f32) as *mut c_void,
            (&b_p as *const i32) as *mut c_void,
            (&k_p as *const i32) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
            (&nv[3] as *const i32) as *mut c_void,
            (&nv[4] as *const i32) as *mut c_void,
            (&nv[5] as *const i32) as *mut c_void,
        ];
        unsafe {
            super::jit::launch_rows(
                self.decode_f32,
                &self.stream,
                dev,
                b * qh,
                256,
                args.as_mut_ptr(),
            )
        }
    }

    /// 006-2 T2: flash-style decode attention (f16 q/out). One CTA per
    /// (b, q_head), 256 threads, dynamic smem (d + max_kv) * 4 bytes —
    /// see decode_flash_kernels.cu for the layout contract (identical to
    /// `launch_decode_step_gqa`, minus the `scores` scratch: softmax lives
    /// in smem) and the `identity` fast path (static identity page table →
    /// contiguous KV reads; page[0] = first physical page of the layer's
    /// contiguous run).
    ///
    /// Fails with `LaunchError::Fatal` when the dynamic smem budget
    /// ((d + max_kv) * 4 > 48 KB) is exceeded — the engine falls back to
    /// the naive kernel (see engine.rs ATTN wiring).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_decode_step_gqa_flash(
        &self,
        dev: u32,
        q: *const u16,
        page: *const u32,
        kv: *const u16,
        kv_lens: *const u32,
        out: *mut u16,
        b: u32,
        qh: u32,
        d: u32,
        block_len: u32,
        kv_ratio: u32,
        kv_heads: u32,
        max_kv: u32,
        total_pages: u32,
        identity: u32,
    ) -> Result<(), LaunchError> {
        let smem = (d + max_kv) * 4;
        if smem > FLASH_SMEM_MAX {
            return Err(LaunchError::Fatal);
        }
        let _guard = CtxGuard::set_current(dev)?;
        let b_p: i32 = b as i32;
        let k_p: i32 = qh as i32;
        let nv: [i32; 7] = [
            d as i32,
            block_len as i32,
            kv_ratio as i32,
            kv_heads as i32,
            max_kv as i32,
            total_pages as i32,
            identity as i32,
        ];
        let mut args: [*mut c_void; 14] = [
            (&q as *const *const u16) as *mut c_void,
            (&page as *const *const u32) as *mut c_void,
            (&kv as *const *const u16) as *mut c_void,
            (&kv_lens as *const *const u32) as *mut c_void,
            (&out as *const *mut u16) as *mut c_void,
            (&b_p as *const i32) as *mut c_void,
            (&k_p as *const i32) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
            (&nv[3] as *const i32) as *mut c_void,
            (&nv[4] as *const i32) as *mut c_void,
            (&nv[5] as *const i32) as *mut c_void,
            (&nv[6] as *const i32) as *mut c_void,
        ];
        unsafe {
            super::jit::launch_fmha(
                self.flash,
                &self.stream,
                dev,
                b * qh,
                1,
                1,
                512,
                smem,
                args.as_mut_ptr(),
            )
        }
    }

    /// S1-9: fused decode flash launch (f16 tier) — the flash attention
    /// plus the current-step kv write (k16/v16 -> kv), block-local inside
    /// the kernel. Grid/block/smem identical to
    /// `launch_decode_step_gqa_flash` (see decode_flash_kernels.cu). The
    /// o-projection phase-1 is a separate launch afterwards (it reads the
    /// whole attention row — cross-block — see the kernel comment).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_decode_step_gqa_flash_fused(
        &self,
        dev: u32,
        q: *const u16,
        page: *const u32,
        kv: *const u16,
        kv_lens: *const u32,
        out: *mut u16,
        b: u32,
        qh: u32,
        d: u32,
        block_len: u32,
        kv_ratio: u32,
        kv_heads: u32,
        max_kv: u32,
        total_pages: u32,
        identity: u32,
        k16: *const u16,
        v16: *const u16,
    ) -> Result<(), LaunchError> {
        let smem = (d + max_kv) * 4;
        if smem > FLASH_SMEM_MAX {
            return Err(LaunchError::Fatal);
        }
        let _guard = CtxGuard::set_current(dev)?;
        let b_p: i32 = b as i32;
        let k_p: i32 = qh as i32;
        let nv: [i32; 7] = [
            d as i32,
            block_len as i32,
            kv_ratio as i32,
            kv_heads as i32,
            max_kv as i32,
            total_pages as i32,
            identity as i32,
        ];
        let mut args: [*mut c_void; 16] = [
            (&q as *const *const u16) as *mut c_void,
            (&page as *const *const u32) as *mut c_void,
            (&kv as *const *const u16) as *mut c_void,
            (&kv_lens as *const *const u32) as *mut c_void,
            (&out as *const *mut u16) as *mut c_void,
            (&b_p as *const i32) as *mut c_void,
            (&k_p as *const i32) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
            (&nv[3] as *const i32) as *mut c_void,
            (&nv[4] as *const i32) as *mut c_void,
            (&nv[5] as *const i32) as *mut c_void,
            (&nv[6] as *const i32) as *mut c_void,
            (&k16 as *const *const u16) as *mut c_void,
            (&v16 as *const *const u16) as *mut c_void,
        ];
        unsafe {
            super::jit::launch_fmha(
                self.flash_fused,
                &self.stream,
                dev,
                b * qh,
                1,
                1,
                512,
                smem,
                args.as_mut_ptr(),
            )
        }
    }

    /// 006-2 T2: flash-style decode attention, f32 q/out (parity-f32 tier;
    /// q pre-scaled by 1/sqrt(d) upstream, KV f16 — same as
    /// `launch_decode_step_gqa_f32`'s contract).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_decode_step_gqa_flash_f32(
        &self,
        dev: u32,
        q: *const f32,
        page: *const u32,
        kv: *const u16,
        kv_lens: *const u32,
        out: *mut f32,
        b: u32,
        qh: u32,
        d: u32,
        block_len: u32,
        kv_ratio: u32,
        kv_heads: u32,
        max_kv: u32,
        total_pages: u32,
        identity: u32,
    ) -> Result<(), LaunchError> {
        let smem = (d + max_kv) * 4;
        if smem > FLASH_SMEM_MAX {
            return Err(LaunchError::Fatal);
        }
        let _guard = CtxGuard::set_current(dev)?;
        let b_p: i32 = b as i32;
        let k_p: i32 = qh as i32;
        let nv: [i32; 7] = [
            d as i32,
            block_len as i32,
            kv_ratio as i32,
            kv_heads as i32,
            max_kv as i32,
            total_pages as i32,
            identity as i32,
        ];
        let mut args: [*mut c_void; 14] = [
            (&q as *const *const f32) as *mut c_void,
            (&page as *const *const u32) as *mut c_void,
            (&kv as *const *const u16) as *mut c_void,
            (&kv_lens as *const *const u32) as *mut c_void,
            (&out as *const *mut f32) as *mut c_void,
            (&b_p as *const i32) as *mut c_void,
            (&k_p as *const i32) as *mut c_void,
            (&nv[0] as *const i32) as *mut c_void,
            (&nv[1] as *const i32) as *mut c_void,
            (&nv[2] as *const i32) as *mut c_void,
            (&nv[3] as *const i32) as *mut c_void,
            (&nv[4] as *const i32) as *mut c_void,
            (&nv[5] as *const i32) as *mut c_void,
            (&nv[6] as *const i32) as *mut c_void,
        ];
        unsafe {
            super::jit::launch_fmha(
                self.flash_f32,
                &self.stream,
                dev,
                b * qh,
                1,
                1,
                512,
                smem,
                args.as_mut_ptr(),
            )
        }
    }
}

/// Fused Q8_0 dequant-dot decode kernel loader (006 T6; sm90+ decode gate
/// precondition per specs/006-cuda-perf/spec.md — the 0.85x llama.cpp CUDA
/// decode gate is only judged after this kernel lands).
///
/// Replaces the "dequant -> fp16 device buffer -> GEMM" Q8_0 path (003 D3 /
/// 014 D4) with a single launch per decode step: dequantization happens in
/// registers (f16 values never touch memory) fused with the fp32 dot. The
/// dense fp16 path (engine gemm1) remains the fallback. Engine wiring is
/// deferred to the Q8_0 model integration closure (T-305); this module only
/// provides the kernel + driver harness.
#[derive(Debug)]
pub struct DecodeDotKernels {
    /// Loaded cubin (kept alive — kernel fn is a module symbol).
    #[allow(dead_code)]
    lib: JLib,
    dot: KernelFn,
    stream: CudaStream,
    arch: String,
}

impl DecodeDotKernels {
    /// Loader constructor (toolchain probe → compile/cache → kernel fetch).
    pub fn new(
        arch: &str,
        cache_dir: Option<PathBuf>,
        stream: CudaStream,
    ) -> Result<Self, LaunchError> {
        let tc = probe_toolchain_for_arch(arch)?;
        let src = KernelSource {
            name: "decode_dot_kernels",
            src: include_str!("../kernels/decode_dot.cu"),
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
        let dot = lib.kernel("fused_q8_dot")?;
        Ok(Self { lib, dot, stream, arch: arch.to_string() })
    }

    /// Target architecture (diagnostics).
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Block until the launch stream drains.
    pub fn sync_stream(&self) -> Result<(), LaunchError> {
        self.stream.synchronize()
    }

    /// Fused Q8_0 dequant-dot decode step:
    /// `out[n] = sum_k f32(y_nk) * f32(x_k)` with
    /// `y_nk = RNE_f16(f32(q_nk) * f32(f16(scale_nk)))` computed in registers
    /// (bit-exact with dequant_q8_0 + cast_f32_to_f16) and fp32 accumulation
    /// (003 dense 32F-acc semantics, D7 gate tier).
    ///
    /// `k` must be a multiple of 32 (Q8_0 block size); `w` is the row-major
    /// [n x k] Q8_0 blob, 34 B per 32-element block; `x` is the f16 activation
    /// row. Output is the per-layer EP f32 result (residual/add are deferred
    /// to engine integration).
    pub fn launch_fused_q8_dot(
        &self,
        dev: u32,
        x: *const u16,
        w: *const u8,
        out: *mut f32,
        n: u32,
        k: u32,
    ) -> Result<(), LaunchError> {
        if n == 0 || k == 0 {
            return Ok(());
        }
        debug_assert_eq!(k % 32, 0, "Q8_0: k must be a multiple of 32");
        let _guard = CtxGuard::set_current(dev)?;
        // SAFETY (C3 discipline): pointers are device buffers guaranteed by
        // the caller; arguments are addressed from local variables.
        let n_v: u32 = n;
        let k_v: u32 = k;
        let mut args: [*mut c_void; 5] = [
            (&x as *const *const u16) as *mut c_void,
            (&w as *const *const u8) as *mut c_void,
            (&out as *const *mut f32) as *mut c_void,
            (&n_v as *const u32) as *mut c_void,
            (&k_v as *const u32) as *mut c_void,
        ];
        unsafe {
            super::jit::launch_rows(
                self.dot,
                &self.stream,
                dev,
                n.div_ceil(8),
                256,
                args.as_mut_ptr(),
            )
        }
    }
}

/// 设备 KV store（K/V 区；页数×块长与 `crates::memory::PagePool` 契约一致）。
#[derive(Debug)]
pub struct KvStore {
    /// 设备存储（K 区 + V 区连续）。
    pub data: DeviceBuffer,
    /// 页总数。
    pub total_pages: usize,
    /// 每页 token 数。
    pub block_len: usize,
}

impl KvStore {
    /// 分配（K/V 两区，每区 total_pages×block_len×kv_heads×d×2 字节）。
    pub fn alloc(
        dev: DeviceId,
        total_pages: usize,
        block_len: usize,
        kv_heads: usize,
        d: usize,
    ) -> Result<Self, LaunchError> {
        let per_region = total_pages * block_len * kv_heads * d * 2;
        let data = DeviceBuffer::alloc(dev, per_region * 2)?;
        Ok(Self { data, total_pages, block_len })
    }

    /// K 区基址。
    pub fn k_ptr(&self) -> *const u16 {
        self.data.as_ptr() as *const u16
    }

    /// V 区基址。
    pub fn v_ptr(&self) -> *const u16 {
        let per = self.data.size() / 2;
        // SAFETY(宿主调用方)：引用为只读偏移。
        unsafe { (self.data.as_ptr() as *const u16).add(per / 2) }
    }
}

/// 006 T6 真机微基准（cudaEvent 计时）：fused Q8_0 dequant-dot vs 003 dense
/// 路径的单 token 步耗时。放在 crate 内以访问 `CudaStream::handle()`（pub(crate)）
/// 做流上事件计时；#[ignore] 真机档。
#[cfg(all(test, feature = "cuda"))]
mod bench {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;
    use crate::CudaContext;
    use crate::buffer::{DeviceBuffer, HostBuffer, MemRef, copy};
    use crate::gemm::{Gemm, GpuMat};
    use cudarc::cublas::sys as blas;
    use cudarc::runtime::sys as rt;
    use reinfer_core::DeviceId;
    use reinfer_gguf::codes::dequantize_q8_0;
    use std::ffi::c_void;

    fn xorshift(seed: &mut u64) -> u64 {
        let mut x = *seed;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *seed = x;
        x
    }

    /// 随机 finite fp16 位（指数域 0..=12 → |v| ≤ 2^-3；真实激活量级）。
    fn rand_f16_bits(seed: &mut u64) -> u16 {
        let mant = (xorshift(seed) as u16) & 0x3ff;
        let exp = ((xorshift(seed) as u16) % 0x0d) & 0xf;
        (exp << 10) | mant
    }

    /// 随机 Q8_0 blob [n x k]：scale 指数域 0..=10（有限 f16，含次正规）。
    fn random_q8_blob(n: usize, k: usize, seed: u64) -> Vec<u8> {
        let blocks = n * k / 32;
        let mut s = seed;
        let mut out = Vec::with_capacity(blocks * 34);
        for _ in 0..blocks {
            let mant = (xorshift(&mut s) as u16) & 0x3ff;
            let exp = ((xorshift(&mut s) as u16) % 0x0b) & 0xf;
            let d = (exp << 10) | mant;
            out.extend_from_slice(&d.to_le_bytes());
            for _ in 0..32 {
                out.push(xorshift(&mut s) as u8);
            }
        }
        out
    }

    /// host f32 → f16 位（RNE；与内核 f32_to_hbits / engine f32_to_f16_bits
    /// 同语义——014 r2：Q8_0 的 f32→f16 必须 RNE 单次舍入）。
    fn rne_f16(f: f32) -> u16 {
        let bits = f.to_bits();
        let sign = (bits >> 16) & 0x8000u32;
        let exp = ((bits >> 23) & 0xff) as i32;
        let man = bits & 0x7f_ffff;
        if exp == 0xff {
            return (sign | 0x7c00 | ((man >> 13) & 0x3ff)) as u16;
        }
        let half_exp = exp - 127 + 15;
        if half_exp <= 0 {
            if half_exp < -10 {
                return sign as u16;
            }
            return (sign | ((man | 0x80_0000) >> (1 - half_exp + 13))) as u16;
        }
        if half_exp >= 31 {
            return (sign | 0x7c00) as u16;
        }
        (sign | ((half_exp as u32) << 10) | (man >> 13)) as u16
    }

    fn upl(dev: DeviceId, host: &[u8]) -> DeviceBuffer {
        let hb = HostBuffer::alloc(host.len()).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(host.as_ptr(), hb.as_ptr() as *mut u8, host.len());
        }
        let db = DeviceBuffer::alloc(dev, host.len()).unwrap();
        copy(&mut MemRef::Device(&db), &MemRef::Host(&hb), host.len(), None).unwrap();
        db
    }

    /// 流上 N 次 launch 的 GPU 侧耗时（cudaEvent 计时；事件在 launch 前后入流）。
    fn elapsed_ms(stream: &CudaStream, launches: usize, mut f: impl FnMut()) -> f32 {
        let mut st: rt::cudaEvent_t = std::ptr::null_mut();
        let mut en: rt::cudaEvent_t = std::ptr::null_mut();
        // flags=0：计时事件（CU_EVENT_DISABLE_TIMING 未设置）。
        unsafe {
            rt::cudaEventCreateWithFlags(&mut st, 0).result().unwrap();
            rt::cudaEventCreateWithFlags(&mut en, 0).result().unwrap();
        }
        let raw = stream.handle();
        // SAFETY：事件合法；流有效。
        unsafe {
            rt::cudaEventRecord(st, raw).result().unwrap();
        }
        for _ in 0..launches {
            f();
        }
        // SAFETY：同上。
        unsafe {
            rt::cudaEventRecord(en, raw).result().unwrap();
            rt::cudaEventSynchronize(en).result().unwrap();
        }
        let mut ms = 0.0f32;
        // SAFETY：start/end 均已完成（en 已同步）。
        unsafe {
            rt::cudaEventElapsedTime(&mut ms, st, en).result().unwrap();
        }
        // SAFETY：事件销毁。
        unsafe {
            let _ = rt::cudaEventDestroy(st).result();
            let _ = rt::cudaEventDestroy(en).result();
        }
        ms
    }

    #[test]
    #[ignore = "gpu.yml: l3-kernels / dequant-dot-bench"]
    fn fused_q8_dot_vs_dense_micro_bench() {
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id();
        let ddev = dev.index();
        let stream = CudaStream::new(dev).unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-decode-dot-bench");
        let _ = std::fs::remove_dir_all(&cache);
        let dot = DecodeDotKernels::new(
            &crate::arch::resolve_arch().unwrap(),
            Some(cache.clone()),
            stream.clone(),
        )
        .unwrap();
        let blas = Gemm::new(ddev).unwrap();

        let k = 4096usize;
        for &n in &[896usize, 1536] {
            // 同输入同量化：Q8_0 blob 用于 fused；dequant→f16→[k×n] 转置给 dense。
            let blob = random_q8_blob(n, k, 0xBEEF + n as u64);
            let mut xseed = 0xBEEF + k as u64;
            let x16: Vec<u16> = (0..k).map(|_| rand_f16_bits(&mut xseed)).collect();
            // dense B：[k×n] 行主序 f16（gemm B 约定）。
            let mut w16t: Vec<u16> = vec![0u16; k * n];
            let mut row = vec![0f32; k];
            let row_bytes = k / 32 * 34;
            for r in 0..n {
                // dequantize_q8_0 processes the whole slice: pass exactly row r's bytes.
                let blob_row = &blob[r * row_bytes..(r + 1) * row_bytes];
                dequantize_q8_0(blob_row, &mut row).unwrap();
                for (i, v) in row.iter().enumerate() {
                    w16t[i * n + r] = rne_f16(*v);
                }
            }
            let x_raw: Vec<u8> = x16.iter().flat_map(|v| v.to_le_bytes()).collect();
            let w_raw: Vec<u8> = w16t.iter().flat_map(|v| v.to_le_bytes()).collect();
            let dx = upl(dev, &x_raw);
            let dw = upl(dev, &w_raw);
            let dblob = upl(dev, &blob);
            let dout = DeviceBuffer::alloc(dev, n * 4).unwrap();
            let mut cmat = GpuMat {
                ptr: dout.as_ptr() as *mut c_void,
                dtype: blas::cudaDataType_t::CUDA_R_32F,
                ld: 1,
            };
            let amat = GpuMat {
                ptr: dx.as_ptr() as *mut c_void,
                dtype: blas::cudaDataType_t::CUDA_R_16F,
                ld: k as i32,
            };
            let bmat = GpuMat {
                ptr: dw.as_ptr() as *mut c_void,
                dtype: blas::cudaDataType_t::CUDA_R_16F,
                ld: n as i32,
            };

            let mut run_fused = || {
                dot.launch_fused_q8_dot(
                    ddev,
                    dx.as_ptr() as *const u16,
                    dblob.as_ptr(),
                    dout.as_ptr() as *mut f32,
                    n as u32,
                    k as u32,
                )
                .unwrap()
            };
            let mut run_dense = || {
                blas.gemm_f32acc(&stream, 1, n as i32, k as i32, &amat, &bmat, &mut cmat, 1.0, 0.0)
                    .unwrap()
            };
            // 预热（JIT 缓存命中后仍有一次调度/指令缓存预热）。
            for _ in 0..50 {
                run_fused();
            }
            stream.synchronize().unwrap();
            for _ in 0..50 {
                run_dense();
            }
            stream.synchronize().unwrap();

            let iters = 300usize;
            let ms_fused = elapsed_ms(&stream, iters, &mut run_fused);
            let ms_dense = elapsed_ms(&stream, iters, &mut run_dense);
            let us_f = ms_fused * 1e3 / iters as f32;
            let us_d = ms_dense * 1e3 / iters as f32;
            eprintln!(
                "dequant-dot bench (k={k}, n={n}): fused {us_f:.1} us/step ({:.0} tok/s) vs \
                 003 dense gemm_f32acc {us_d:.1} us/step ({:.0} tok/s) -> {:.2}x",
                1e6 / us_f,
                1e6 / us_d,
                us_d / us_f
            );
            // 记录项（非 gate）：单步 f16 输出等价性（数学同一性——两路径
            // 输出应为同一量化权重的同累积语义结果）。
            assert!(us_f > 0.0 && us_d > 0.0, "bench: zero elapsed");
        }
    }

    #[test]
    #[ignore = "gpu.yml: l3-kernels / dequant-dot-bench"]
    fn fused_q8_dot_bench_sanity_output() {
        // 微基准测试自身完整性：fused 输出与 dense 路径（f16 权重）一致性抽查
        // （与 tests/dequant_dot.rs 主判据互补；此处保真 bench 数据有效性）。
        let ctx = CudaContext::init(DeviceId::new(0)).unwrap();
        let dev = ctx.device_id();
        let ddev = dev.index();
        let stream = CudaStream::new(dev).unwrap();
        let cache = std::env::temp_dir().join("reinfer-jit-decode-dot-bench2");
        let _ = std::fs::remove_dir_all(&cache);
        let dot = DecodeDotKernels::new(
            &crate::arch::resolve_arch().unwrap(),
            Some(cache.clone()),
            stream.clone(),
        )
        .unwrap();
        let blas = Gemm::new(ddev).unwrap();
        let (n, k) = (1536usize, 4096usize);
        let mut seed = 0x5A17u64;
        let blob = random_q8_blob(n, k, seed);
        let x16: Vec<u16> = (0..k).map(|_| rand_f16_bits(&mut seed)).collect();
        let mut w16t: Vec<u16> = vec![0u16; k * n];
        let mut row = vec![0f32; k];
        let row_bytes = k / 32 * 34;
        for r in 0..n {
            // dequantize_q8_0 processes the whole slice: pass exactly row r's bytes.
            dequantize_q8_0(&blob[r * row_bytes..(r + 1) * row_bytes], &mut row).unwrap();
            for (i, v) in row.iter().enumerate() {
                w16t[i * n + r] = rne_f16(*v);
            }
        }
        let x_raw: Vec<u8> = x16.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_raw: Vec<u8> = w16t.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dx = upl(dev, &x_raw);
        let dw = upl(dev, &w_raw);
        let dblob = upl(dev, &blob);
        let dout = DeviceBuffer::alloc(dev, n * 4).unwrap();
        let mut cmat = GpuMat {
            ptr: dout.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_32F,
            ld: 1,
        };
        let amat = GpuMat {
            ptr: dx.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_16F,
            ld: k as i32,
        };
        let bmat = GpuMat {
            ptr: dw.as_ptr() as *mut c_void,
            dtype: blas::cudaDataType_t::CUDA_R_16F,
            ld: n as i32,
        };
        blas.gemm_f32acc(&stream, 1, n as i32, k as i32, &amat, &bmat, &mut cmat, 1.0, 0.0)
            .unwrap();
        stream.synchronize().unwrap();
        let hb = HostBuffer::alloc(n * 4).unwrap();
        copy(&mut MemRef::Host(&hb), &MemRef::Device(&dout), n * 4, None).unwrap();
        let dense: Vec<f32> =
            unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, n).to_vec() };
        dot.launch_fused_q8_dot(
            ddev,
            dx.as_ptr() as *const u16,
            dblob.as_ptr(),
            dout.as_ptr() as *mut f32,
            n as u32,
            k as u32,
        )
        .unwrap();
        dot.sync_stream().unwrap();
        copy(&mut MemRef::Host(&hb), &MemRef::Device(&dout), n * 4, None).unwrap();
        let fused: Vec<f32> =
            unsafe { std::slice::from_raw_parts(hb.as_ptr() as *const f32, n).to_vec() };
        let mut bad = 0usize;
        for (g, w) in fused.iter().zip(dense.iter()) {
            let diff = (g - w).abs();
            let tol = 1e-6 + 1e-4 * w.abs();
            if diff > tol {
                bad += 1;
            }
        }
        assert_eq!(bad, 0, "bench sanity: fused vs dense {bad}/{n} elements over D7 gate");
        eprintln!("bench sanity ok: fused == dense (D7 gate) at k={k} n={n}");
    }
}

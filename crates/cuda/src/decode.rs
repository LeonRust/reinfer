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

/// decode_step_gqa 装载单元。
#[derive(Debug)]
pub struct DecodeKernels {
    lib: JLib,
    decode: KernelFn,
    stream: CudaStream,
    arch: String,
}

impl DecodeKernels {
    /// Loader constructor。
    pub fn new(arch: &str, cache_dir: Option<PathBuf>, stream: CudaStream) -> Result<Self, LaunchError> {
        let tc = probe_toolchain_for_arch(arch)?;
        let src = KernelSource {
            name: "decode_gqa_kernels",
            src: include_str!("../kernels/decode_gqa_kernels.cu"),
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
        let decode = lib.kernel("decode_step_gqa")?;
        Ok(Self { lib, decode, stream, arch: arch.to_string() })
    }

    /// Target architecture (diagnostics).
    pub fn arch(&self) -> &str {
        &self.arch
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
        unsafe { super::jit::launch_rows(self.decode, &self.stream, dev, b * qh, 256, args.as_mut_ptr()) }
    }
}

/// 设备 KV store（K/V 区；页数×块长与 `crates::memory::PagePool` 契约一致）。
pub struct KvStore {
    pub data: DeviceBuffer,
    pub total_pages: usize,
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

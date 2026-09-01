//! 014 T5：Q8_0 解量化内核装载（与 `diff.rs` 同构：JitCache 管线复用）。
//!
//! 判据：真机 diff vs `crates::gguf::codes::dequantize_q8_0` 位精确
//! （0 ulp——单乘语义；见 kernels/dequant_kernels.cu）。

use crate::jit::{CtxGuard, JLib, KernelFn};
use crate::stream::CudaStream;
use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::{JitCache, JitKey, KernelSource, probe_toolchain_for_arch};
use reinfer_kernels::LaunchError;
use std::ffi::c_void;
use std::path::PathBuf;

/// Q8_0 解量化装载单元。
#[derive(Debug)]
pub struct DequantKernels {
    lib: JLib,
    dequant: KernelFn,
    stream: CudaStream,
    arch: String,
}

impl DequantKernels {
    /// Loader constructor (toolchain probe → compile/cache → kernel fetch).
    pub fn new(
        arch: &str,
        cache_dir: Option<PathBuf>,
        stream: CudaStream,
    ) -> Result<Self, LaunchError> {
        let tc = probe_toolchain_for_arch(arch)?;
        let src = KernelSource {
            name: "dequant_kernels",
            src: include_str!("../kernels/dequant_kernels.cu"),
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
        let dequant = lib.kernel("dequant_q8_0")?;
        Ok(Self { lib, dequant, stream, arch: arch.to_string() })
    }

    /// Target architecture (diagnostics).
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// Block until the launch stream drains.
    pub fn sync_stream(&self) -> Result<(), LaunchError> {
        self.stream.synchronize()
    }

    /// 解量化 `nblocks` 个 Q8_0 块（blob = 34B×nblocks）→ `out`（32×nblocks
    /// f32，设备内存）。
    pub fn launch_dequant_q8_0(
        &self,
        dev: u32,
        blob: *const u8,
        out: *mut f32,
        nblocks: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        // SAFETY（C3 纪律）：指针由调用方保证为设备缓冲；参数局部变量取址。
        let mut args: [*mut c_void; 3] = [
            (&blob as *const *const u8) as *mut c_void,
            (&out as *const *mut f32) as *mut c_void,
            (&nblocks as *const u32) as *mut c_void,
        ];
        unsafe {
            super::jit::launch_rows(self.dequant, &self.stream, dev, nblocks, 32, args.as_mut_ptr())
        }
    }

    /// Raw library handle (diagnostics/upstream orchestration).
    pub fn raw_lib(&self) -> cudarc::driver::sys::CUlibrary {
        self.lib.raw()
    }
}

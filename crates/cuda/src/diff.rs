//! 012 D2：diff 内核（rms_norm/rope/masked_softmax）装载与行式 launch。
//!
//! 与 `VecAddProvider` 同构：单文件 `diff_kernels.cu` 一次编译/缓存，
//! 一个库装载三个 `extern "C"` 内核。参数打包遵循 C3 实测纪律
//! （局部变量取址；见 `jit.rs` 头注释）。

use crate::jit::{CtxGuard, JLib, KernelFn};
use crate::stream::CudaStream;
use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::{JitCache, JitKey, KernelSource, probe_toolchain_for_arch};
use reinfer_kernels::LaunchError;
use std::ffi::c_void;
use std::path::PathBuf;

/// 三个行式 diff 内核的装载单元。
#[derive(Debug)]
pub struct DiffKernels {
    lib: JLib,
    rms: KernelFn,
    rope: KernelFn,
    softmax: KernelFn,
    softmax_matrix: KernelFn,
    cast_f16: KernelFn,
    add_mask: KernelFn,
    transpose16: KernelFn,
    transpose32: KernelFn,
    stream: CudaStream,
    arch: String,
}

impl DiffKernels {
    /// 完整构造（工具链 → 编译/缓存 → 加载 → 取核 ×3）。
    pub fn new(
        arch: &str,
        cache_dir: Option<PathBuf>,
        stream: CudaStream,
    ) -> Result<Self, LaunchError> {
        // 工具链自适配：按目标 arch 在所有候选里选最优（env 仍可显式覆盖）；
        // 无支持工具链 → Fatal（不静默）
        let tc = probe_toolchain_for_arch(arch)?;
        let src = KernelSource {
            name: "diff_kernels",
            src: include_str!("../kernels/diff_kernels.cu"),
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
        let rms = lib.kernel("rms_norm_row")?;
        let rope = lib.kernel("rope_row")?;
        let softmax = lib.kernel("masked_softmax_row")?;
        let softmax_matrix = lib.kernel("masked_softmax_matrix")?;
        let cast_f16 = lib.kernel("cast_f32_to_f16")?;
        let add_mask = lib.kernel("add_f32_f32_inplace")?;
        let transpose16 = lib.kernel("transpose_f16")?;
        let transpose32 = lib.kernel("transpose_f32")?;
        Ok(Self {
            lib,
            rms,
            rope,
            softmax,
            softmax_matrix,
            cast_f16,
            add_mask,
            transpose16,
            transpose32,
            stream,
            arch: arch.to_string(),
        })
    }

    /// 目标架构（诊断）。
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// 阻塞等待 launch 流排空。
    pub fn sync_stream(&self) -> Result<(), LaunchError> {
        self.stream.synchronize()
    }

    /// 单行 RMSNorm（`x[i] / sqrt(mean(x²)+eps) * w[i]`；n ≤ 8192）。
    pub fn launch_rms_norm(
        &self,
        dev: u32,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        n: u32,
        eps: f32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        // SAFETY（C3 纪律）：指针由调用方保证为设备缓冲；参数局部变量取址。
        let mut args: [*mut c_void; 5] = [
            (&x as *const *const f32) as *mut c_void,
            (&w as *const *const f32) as *mut c_void,
            (&out as *const *mut f32) as *mut c_void,
            (&n as *const u32) as *mut c_void,
            (&eps as *const f32) as *mut c_void,
        ];
        unsafe { super::jit::launch_row(self.rms, &self.stream, dev, 256, args.as_mut_ptr()) }
    }

    /// 单头单位置 RoPE（Neox 半旋转；half ≤ 1024）。
    pub fn launch_rope(
        &self,
        dev: u32,
        x: *const f32,
        out: *mut f32,
        half: u32,
        pos: u32,
        eta: f32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        // SAFETY：同上；线程数 = half（每线程一对）。
        let mut args: [*mut c_void; 5] = [
            (&x as *const *const f32) as *mut c_void,
            (&out as *const *mut f32) as *mut c_void,
            (&half as *const u32) as *mut c_void,
            (&pos as *const u32) as *mut c_void,
            (&eta as *const f32) as *mut c_void,
        ];
        unsafe { super::jit::launch_row(self.rope, &self.stream, dev, 1024, args.as_mut_ptr()) }
    }

    /// 批量行 softmax（grid=rows；`-inf` 掩码位预置在 x；全无效行 → 全 0）。
    pub fn launch_masked_softmax_matrix(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *const f32,
        out: *mut f32,
        rows: u32,
        rowlen: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let rowlen_v: u32 = rowlen;
        // SAFETY：调用方保证设备指针；args 局部变量取址（C3 纪律）。
        let mut args: [*mut c_void; 4] = [
            (&x as *const *const f32) as *mut c_void,
            (&out as *const *mut f32) as *mut c_void,
            (&rows as *const u32) as *mut c_void,
            (&rowlen_v as *const u32) as *mut c_void,
        ];
        unsafe {
            super::jit::launch_rows(self.softmax_matrix, stream, dev, rows, 256, args.as_mut_ptr())
        }
    }

    /// 单行 softmax（输入已含 -inf 掩码位；全无效行输出全 -inf）。
    pub fn launch_masked_softmax(
        &self,
        dev: u32,
        x: *const f32,
        out: *mut f32,
        n: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        // SAFETY：同上。
        let mut args: [*mut c_void; 3] = [
            (&x as *const *const f32) as *mut c_void,
            (&out as *const *mut f32) as *mut c_void,
            (&n as *const u32) as *mut c_void,
        ];
        unsafe { super::jit::launch_row(self.softmax, &self.stream, dev, 256, args.as_mut_ptr()) }
    }

    /// 直接访问底层（诊断/上层编排）。
    pub fn raw_lib(&self) -> cudarc::driver::sys::CUlibrary {
        self.lib.raw()
    }
}

impl DiffKernels {
    /// f32 → f16 元素转换（RNE；PV 前 dtype 一致化——014 T7）。
    pub fn launch_cast_f32_f16(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *const f32,
        out: *mut u16,
        n: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let n_v: u32 = n;
        // SAFETY：调用方保证设备指针；C3 纪律。
        let mut args: [*mut c_void; 3] = [
            (&x as *const *const f32) as *mut c_void,
            (&out as *const *mut u16) as *mut c_void,
            (&n_v as *const u32) as *mut c_void,
        ];
        // SAFETY：同矩阵 launch。
        unsafe {
            super::jit::launch_rows(
                self.cast_f16,
                stream,
                dev,
                n.div_ceil(256),
                256,
                args.as_mut_ptr(),
            )
        }
    }
}

impl DiffKernels {
    /// 就地加掩码（s += mask；mask 含 -inf 位）。
    pub fn launch_add_mask(
        &self,
        dev: u32,
        stream: &CudaStream,
        s: *mut f32,
        mask: *const f32,
        n: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let n_v: u32 = n;
        // SAFETY：同矩阵 launch。
        let mut args: [*mut c_void; 3] = [
            (&s as *const *mut f32) as *mut c_void,
            (&mask as *const *const f32) as *mut c_void,
            (&n_v as *const u32) as *mut c_void,
        ];
        unsafe {
            super::jit::launch_rows(
                self.add_mask,
                stream,
                dev,
                n.div_ceil(256),
                256,
                args.as_mut_ptr(),
            )
        }
    }
}

impl DiffKernels {
    /// f16 转置行主序 [rows×cols] → [cols×rows]。
    pub fn launch_transpose_f16(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *const u16,
        out: *mut u16,
        rows: u32,
        cols: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let rows_v: u32 = rows;
        let cols_v: u32 = cols;
        // SAFETY：同矩阵 launch。
        let mut args: [*mut c_void; 4] = [
            (&x as *const *const u16) as *mut c_void,
            (&out as *const *mut u16) as *mut c_void,
            (&rows_v as *const u32) as *mut c_void,
            (&cols_v as *const u32) as *mut c_void,
        ];
        // grid (cols/16, rows/16)；block 16×16（transpose 常规形态）。
        let gx = cols.div_ceil(16);
        let gy = rows.div_ceil(16);
        unsafe {
            super::jit::launch_grid(
                self.transpose16,
                stream,
                dev,
                gx,
                gy,
                16,
                16,
                args.as_mut_ptr(),
            )
        }
    }
}

impl DiffKernels {
    /// f32 转置行主序 [rows×cols] → [cols×rows]。
    pub fn launch_transpose_f32(
        &self,
        dev: u32,
        stream: &CudaStream,
        x: *const f32,
        out: *mut f32,
        rows: u32,
        cols: u32,
    ) -> Result<(), LaunchError> {
        let _guard = CtxGuard::set_current(dev)?;
        let rows_v: u32 = rows;
        let cols_v: u32 = cols;
        // SAFETY：同矩阵 launch。
        let mut args: [*mut c_void; 4] = [
            (&x as *const *const f32) as *mut c_void,
            (&out as *const *mut f32) as *mut c_void,
            (&rows_v as *const u32) as *mut c_void,
            (&cols_v as *const u32) as *mut c_void,
        ];
        unsafe {
            super::jit::launch_grid(
                self.transpose32,
                stream,
                dev,
                cols.div_ceil(16),
                rows.div_ceil(16),
                16,
                16,
                args.as_mut_ptr(),
            )
        }
    }
}

//! Jit 产物加载与 launch（012 C1；unsafe 收敛于此——FFI 宿主 crate）。
//!
//! `JLib` = `cuLibraryLoadData` 句柄（RAII，持字节至 unload）；`KernelFn`
//! = `cuLibraryGetKernel` 取到的内核（cuda.h 明言 CUkernel 可直接 cast
//! 为 CUfunction 交给 cuLaunchKernel——工具链实测通过）。
//!
//! 生命周期契约（012 plan r1）：JLib 仅在所属 `CudaContext` 存活期内有效；
//! launch 前调用方负责将 context 置为 current（009 per-thread 纪律）；
//! 禁止在 provider 内新建 safety-layer context（会混用非 primary 上下文）。

use crate::error::{CudaErrorCode, classify};
use crate::stream::CudaStream;
use reinfer_kernels::LaunchError;
use std::ffi::{CString, c_void};

use cudarc::driver::sys;

/// 已加载库中的内核句柄。
#[derive(Debug, Clone, Copy)]
pub struct KernelFn(sys::CUkernel);

/// 裸 cubin 库句柄（RAII：Drop → cuLibraryUnload）。
#[derive(Debug)]
pub struct JLib {
    lib: sys::CUlibrary,
    /// 持有代码字节至 unload（驱动卸载前保持合法引用）。
    _code: Vec<u8>,
}

/// sys 返回值 → 白名单分类（成功=0）。
fn rc(rc: sys::CUresult) -> Result<(), LaunchError> {
    if rc == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(classify(rc as CudaErrorCode).unwrap_or(LaunchError::Fatal))
    }
}

impl JLib {
    /// 从 ELF cubin 字节加载库。
    pub fn from_bytes(code: Vec<u8>) -> Result<Self, LaunchError> {
        if code.len() < 4 || &code[..4] != b"\x7fELF" {
            return Err(LaunchError::Fatal);
        }
        let mut lib: sys::CUlibrary = std::ptr::null_mut();
        // SAFETY: code 为有效 ELF；输出槽位有效；选项组全空（0 个选项）。
        let r = unsafe {
            sys::cuLibraryLoadData(
                &mut lib,
                code.as_ptr().cast::<c_void>(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        rc(r)?;
        if lib.is_null() {
            return Err(LaunchError::Fatal);
        }
        Ok(Self { lib, _code: code })
    }

    /// 取元内核（名字 = KernelSource.name = `extern "C"` 导出符号）。
    pub fn kernel(&self, name: &str) -> Result<KernelFn, LaunchError> {
        let cname = CString::new(name).map_err(|_| LaunchError::Fatal)?;
        let mut k: sys::CUkernel = std::ptr::null_mut();
        // SAFETY: 输出槽位有效；名字为 NUL 结尾。
        let r = unsafe { sys::cuLibraryGetKernel(&mut k, self.lib, cname.as_ptr()) };
        rc(r)?;
        if k.is_null() {
            return Err(LaunchError::Fatal);
        }
        Ok(KernelFn(k))
    }

    /// 原始句柄（诊断/未来 vendor 面）。
    pub fn raw(&self) -> sys::CUlibrary {
        self.lib
    }
}

impl Drop for JLib {
    fn drop(&mut self) {
        // SAFETY: 句柄唯一所有权；unload 失败忽略（进程退出兜底）。
        let _ = unsafe { sys::cuLibraryUnload(self.lib) };
    }
}

/// vec_add launch（block 256 网格）——012 C1 静态 launch（真正 launch 编排归
/// provider 的 `fn launch` 面，见 `provider.rs` C2）。
///
/// # Safety
/// - `a`/`b`/`out` 均为本 context 有效的设备指针，`n` 个元素；
/// - 当前线程已置 primary context；`stream` 有效。
pub unsafe fn launch_vec_add(
    kernel: KernelFn,
    stream: &CudaStream,
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    n: u32,
) -> Result<(), LaunchError> {
    if n == 0 {
        return Ok(());
    }
    let mut args: [*mut c_void; 4] = [
        a.cast::<c_void>() as *mut c_void,
        b.cast::<c_void>() as *mut c_void,
        out.cast::<c_void>(),
        (&n as *const u32).cast::<c_void>() as *mut c_void,
    ];
    let grid = n.div_ceil(256);
    // cudarc runtime/driver 两套 sys 的流类型为同指针不同 Rust 类型：裸指针层级转换。
    let cu_stream: sys::CUstream = stream.handle() as *mut c_void as sys::CUstream;
    // SAFETY: kernel 为当前库内取出的合法函数；args 与内核签名一致（4 参）；
    // stream 有效；context 已 current。
    let r = unsafe {
        sys::cuLaunchKernel(
            kernel.0 as sys::CUfunction,
            grid,
            1,
            1,
            256,
            1,
            1,
            0,
            cu_stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    rc(r)
}

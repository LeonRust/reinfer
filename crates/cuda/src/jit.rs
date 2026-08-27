//! Jit 产物加载与 launch（012 C1；unsafe 收敛于此——FFI 宿主 crate）。
//!
//! `JLib` = `cuLibraryLoadData` 句柄（RAII，持字节至 unload）；`KernelFn`
//! = `cuKernelGetFunction` 的转换结果。
//!
//! 实测纪律（C3，判定机 RTX 5090 / 驱动 595.84 / nvcc 12.8，全部经
//! `examples/kernel_probe.rs` 二分定位）：
//! - **context**：driver launch 需要线程 current primary context
//!   （`CtxGuard`；仅 runtime cudaSetDevice 不够——直 launch SIGSEGV）；
//! - **内核句柄**：`CUkernel` 直 cast 为 `CUfunction` 后 launch SIGSEGV，
//!   必须经 `cuKernelGetFunction`（loadtest4/官方路径）；
//! - **kernelParams 打包**：参数必须以**局部变量取址**写入数组（s5 写法）；
//!   内联转换链值会使 595.84 驱动内部 SIGSEGV（值相同、打包不同）。
//!
//! 生命周期契约：JLib 仅在所属 context 存活期内有效；launch 时由内部
//! guard 设置 current（同线程）；禁止新建专属 context。

use crate::error::{CudaErrorCode, classify};
use crate::stream::CudaStream;
use reinfer_kernels::LaunchError;
use std::ffi::{CString, c_void};

use cudarc::driver::sys;

/// 驱动 API 线程上下文守卫（cuDevicePrimaryCtxRetain + cuCtxSetCurrent）。
///
/// 为什么需要（C3 实测）：`cuLaunchKernel` 需要线程的 **driver current context**；
/// 仅 runtime 面（cudaSetDevice）在 driver API 视角下 current 未必然被设置——
/// 直接 launch 实测 SIGSEGV。guard 在 launch 线程创建（current 是线程局部），
/// Drop 释放 primary 引用（runtime 侧的引用保持存活，上下文不销毁）。
pub struct CtxGuard {
    dev: sys::CUdevice,
    ctx: sys::CUcontext,
}

impl CtxGuard {
    /// 在**本线程** retain primary context 并置为 current。
    pub fn set_current(dev: u32) -> Result<Self, LaunchError> {
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        // SAFETY: 输出槽位有效；dev 为合法设备索引（调用方保证存在）。
        let r = unsafe { sys::cuDevicePrimaryCtxRetain(&mut ctx, dev as sys::CUdevice) };
        rc(r)?;
        if ctx.is_null() {
            return Err(LaunchError::Fatal);
        }
        // SAFETY: ctx 为刚 retain 的合法 primary context。
        let r = unsafe { sys::cuCtxSetCurrent(ctx) };
        rc(r)?;
        Ok(Self { dev: dev as sys::CUdevice, ctx })
    }
}

impl Drop for CtxGuard {
    fn drop(&mut self) {
        // SAFETY: 引用来自 retain 且未释放；runtime 侧引用保持 primary 存活。
        let _ = unsafe { sys::cuDevicePrimaryCtxRelease_v2(self.dev) };
    }
}

/// 已加载库中的内核句柄（转换后的 `CUfunction`——防 CUkernel 直 cast 风险，
/// C3 实测：cast 后 launch 段错误，官方 `cuKernelGetFunction` 转换经
/// loadtest4（真机成功探针）与社区惯例验证）。
#[derive(Debug, Clone, Copy)]
pub struct KernelFn(sys::CUfunction);

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
        // 官方转换 CUkernel -> CUfunction（C3 实测：直接 cast 致 launch SIGSEGV）
        let mut f: sys::CUfunction = std::ptr::null_mut();
        // SAFETY: 输出槽位有效；k 来自当前库。
        let r = unsafe { sys::cuKernelGetFunction(&mut f, k) };
        rc(r)?;
        if f.is_null() {
            return Err(LaunchError::Fatal);
        }
        Ok(KernelFn(f))
    }

    /// 原始句柄（诊断/未来 vendor 面）。
    pub fn raw(&self) -> sys::CUlibrary {
        self.lib
    }
}

impl KernelFn {
    /// 原始 CUfunction 句柄（probe/诊断；交 cuLaunchKernel）。
    pub fn raw(&self) -> sys::CUfunction {
        self.0
    }
}

impl JLib {}

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
/// - `stream` 有效（本函数内会保证 driver current context 已设置）。
pub unsafe fn launch_vec_add(
    kernel: KernelFn,
    stream: &CudaStream,
    dev: u32,
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    n: u32,
) -> Result<(), LaunchError> {
    if n == 0 {
        return Ok(());
    }
    // driver launch 需要线程 current context（C3 实测约束）
    let guard = CtxGuard::set_current(dev)?;
    // C3 实测（595.84 驱动）：参数必须以**局部变量取址**打包进 kernelParams
    // （与 loadtest4/s2/s5 已验证写法一致）；直接以转换链内联值（s3 写法）
    // 会在 driver 内部 SIGSEGV——值相同但打包方式被驱动敏感。
    let a_v: *const f32 = a;
    let b_v: *const f32 = b;
    let out_v: *mut f32 = out;
    let n_v: u32 = n;
    let mut args: [*mut c_void; 4] = [
        (&a_v as *const *const f32) as *mut c_void,
        (&b_v as *const *const f32) as *mut c_void,
        (&out_v as *const *mut f32) as *mut c_void,
        (&n_v as *const u32) as *mut c_void,
    ];
    let grid = n.div_ceil(256);
    // cudarc runtime/driver 两套 sys 的流类型为同指针不同 Rust 类型：裸指针层级转换。
    let cu_stream: sys::CUstream = stream.handle() as *mut c_void as sys::CUstream;
    // SAFETY: kernel 为当前库内取出的合法函数；args 与内核签名一致（4 参）；
    // stream 有效；context 已 current。
    let r = unsafe {
        sys::cuLaunchKernel(
            kernel.0,
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

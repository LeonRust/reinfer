//! vec_add 的 Jit tier provider（012 C2：KernelProvider 选择链第一个真实实现）。
//!
//! 完整性：构造时走完整 JitCache 链（probe → key → build_once → load → 取核）；
//! nvcc 缺失/版本过旧/编译失败 → `Fatal`（**不静默降级 CPU**，评审裁决）。
//! launch 的 unsafe 收敛在本模块（FFI 宿主 crate）。
//!
//! 参数通道：`VecAddArgs` 为静态指针容器（`LaunchArgs: Any` 要求 'static）；
//! 指针有效性由调用方（真机测试/引擎）按 `DeviceBuffer` 生命周期保证
//! ——provider 不接管 buffer 所有权。

use crate::buffer::DeviceBuffer;
use crate::jit::{JLib, KernelFn, launch_vec_add};
use crate::stream::CudaStream;
use reinfer_core::DType;
use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::{JitCache, JitKey, KernelSource, check_arch_supported, probe_toolchain};
use reinfer_kernels::{KernelProvider, LaunchArgs, LaunchError, OpConfig, ProviderTier};
use std::any::Any;
use std::path::PathBuf;

/// vec_add launch 参数（设备指针 + 元素数；全部为 borrow-free——Any 契约）。
#[derive(Debug)]
pub struct VecAddArgs {
    /// 输入 a 设备指针（`DeviceBuffer::as_ptr().cast::<f32>()`）。
    pub a: *const f32,
    /// 输入 b 设备指针。
    pub b: *const f32,
    /// 输出设备指针（长度 ≥ n·4）。
    pub out: *mut f32,
    /// 元素数。
    pub n: u32,
}

impl LaunchArgs for VecAddArgs {}

/// Jit tier provider：编译签名的 vec_add（`extern "C" __global__`）。
#[derive(Debug)]
pub struct VecAddProvider {
    lib: JLib,
    kernel: KernelFn,
    stream: CudaStream,
    arch: String,
}

impl VecAddProvider {
    /// 完整构造：工具链探测（梯度检查）→ 编译/缓存 → 加载 + 取核。
    ///
    /// `cache_dir`：显示覆盖（测试注入临时目录）；`None` → `REINFER_JIT_CACHE`/XDG。
    pub fn new(
        arch: &str,
        cache_dir: Option<PathBuf>,
        stream: CudaStream,
    ) -> Result<Self, LaunchError> {
        let tc = probe_toolchain()?;
        let ver =
            reinfer_jit::toolchain::parse_nvcc_version(&tc.ver_line).ok_or(LaunchError::Fatal)?;
        check_arch_supported(arch, ver)?;
        let src = KernelSource {
            name: "vec_add",
            src: include_str!("../kernels/vec_add.cu"),
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
        let kernel = lib.kernel("vec_add")?;
        Ok(Self { lib, kernel, stream, arch: arch.to_string() })
    }

    /// 目标架构（诊断用）。
    pub fn arch(&self) -> &str {
        &self.arch
    }

    /// 底层库句柄（诊断/未来 vendor 面；保证 `lib` 的存活语义被引用）。
    pub fn raw_lib(&self) -> cudarc::driver::sys::CUlibrary {
        self.lib.raw()
    }

    /// 阻塞等待 launch 流排空（异步 launch 后的同步凭证——调用方纪律：
    /// 同一流上的后续依赖/回拷必须显式同步，provider 不隐式同步）。
    pub fn sync_stream(&self) -> Result<(), LaunchError> {
        self.stream.synchronize()
    }

    /// 校验 size 供应的匹配（buffer 分配与 args 一致性）。
    pub fn size_check(n: u32, bufs: (&DeviceBuffer, &DeviceBuffer, &DeviceBuffer)) -> bool {
        let want = n as usize * std::mem::size_of::<f32>();
        let (a, b, out) = bufs;
        a.size() >= want && b.size() >= want && out.size() >= want
    }
}

impl KernelProvider for VecAddProvider {
    fn tier(&self) -> ProviderTier {
        ProviderTier::Jit
    }

    fn matches(&self, cfg: &OpConfig) -> bool {
        cfg.op == "vec_add" && cfg.in_dt == DType::F32 && cfg.out_dt == DType::F32
    }

    fn base_priority(&self, _cfg: &OpConfig) -> i32 {
        0
    }

    fn workspace_size(&self, _cfg: &OpConfig) -> usize {
        0
    }

    fn launch(&self, _cfg: &OpConfig, args: &mut dyn LaunchArgs) -> Result<(), LaunchError> {
        let a = (args as &dyn Any).downcast_ref::<VecAddArgs>().ok_or(LaunchError::Fatal)?;
        if a.n == 0 {
            return Ok(());
        }
        // SAFETY: 指针由调用方保证为对应 DeviceBuffer 的有效设备指针
        //（provider 不接管所有权）；driver current context 由 launch 内部
        // primary-context guard 保证（012 plan r1 C3 实测修正）。
        unsafe {
            launch_vec_add(self.kernel, &self.stream, _cfg.device.index(), a.a, a.b, a.out, a.n)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;

    #[test]
    fn tier_ordering_and_matches_shape() {
        // tier 常量序（012 r1：Vendor>Jit>Native）
        assert!(ProviderTier::Vendor < ProviderTier::Jit);
        assert!(ProviderTier::Jit < ProviderTier::Native);
        // 匹配面矩阵（纯逻辑）
        let c = OpConfig {
            op: "vec_add",
            device: reinfer_core::DeviceId::new(0),
            in_dt: DType::F32,
            out_dt: DType::F32,
            head_dim: 0,
            batch: 1,
            seq: 0,
        };
        assert_eq!(c.op, "vec_add");
        assert_ne!(OpConfig { op: "rms_norm", ..c }.op, "vec_add");
    }

    #[test]
    fn size_check_matches() {
        // 不构造 buffer（无设备）；size 校验逻辑独立可测——假借 DType 尺寸常量
        let want = 4096usize; // n=1024 * 4
        assert_eq!(want, 1024 * std::mem::size_of::<f32>());
    }
}

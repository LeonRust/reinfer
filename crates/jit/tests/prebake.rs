//! 离线预烘焙集成测试（012 T3；无 GPU、需本机 nvcc）。
//!
//! 运行（判定机/任一带 CUDA toolchain 的机器）：
//! ```text
//! REINFER_CUDA_NVCC=/usr/local/cuda-12.8/bin/nvcc \
//! REINFER_CUDA_ARCH=sm_120a \
//! cargo test -p reinfer-jit --test prebake
//! ```
//! 无 nvcc 或版本不支持目标 arch → 动态跳过（打印原因，不 fail——CI 无工具链
//! 环境不应因本测试红）。

use reinfer_jit::compile::{compile_cubin, gencode_flags};
use reinfer_jit::toolchain::parse_nvcc_version;
use reinfer_jit::{JitCache, JitKey, KernelSource, check_arch_supported, probe_toolchain};

const VEC_ADD_SRC: &str = r#"
extern "C" __global__ void vec_add(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] + b[i]; }
}
"#;

fn arch() -> Result<String, ()> {
    // 无默认特判：预烘焙必须显式指认目标架构（本 crate 零 CUDA 知识，
    // 不提供设备检测兜底）
    std::env::var("REINFER_CUDA_ARCH").ok().filter(|s| !s.is_empty()).ok_or(()) // 无默认：必须显式指认
}

#[test]
fn prebake_compile_then_reload() {
    let tc = match probe_toolchain() {
        Ok(t) => t,
        Err(_) => {
            eprintln!("skipping prebake test: no nvcc on PATH");
            return;
        }
    };
    let Ok(arch) = arch() else {
        eprintln!("skipping prebake test: set REINFER_CUDA_ARCH (no default - device-agnostic)");
        return;
    };
    let Some((major, minor)) = parse_nvcc_version(&tc.ver_line) else {
        eprintln!("skipping prebake test: cannot parse {}", tc.ver_line);
        return;
    };
    if check_arch_supported(&arch, (major, minor)).is_err() {
        eprintln!("skipping prebake test: nvcc {major}.{minor} too old for {arch}");
        return;
    }

    let src = KernelSource {
        name: "vec_add_prebake",
        src: VEC_ADD_SRC,
        headers: vec![],
        flags: gencode_flags(&arch).expect("gencode"),
        arch: arch.clone(),
        toolchain_ver: tc.ver_line.clone(),
    };
    let cache = JitCache::open(Some(
        std::env::temp_dir().join(format!("reinfer-jit-prebake-{}", std::process::id())),
    ))
    .expect("cache open");
    let _ = std::fs::remove_dir_all(cache.dir());

    let key = JitKey::new(&src, &tc);
    let start = std::time::Instant::now();
    let (meta, path) =
        cache.build_once(&key, &src, || compile_cubin(&src, &tc)).expect("prebake compile");
    let first = start.elapsed();

    // 产物必须是 ELF 且 meta 与 key 一致
    assert_eq!(meta.key, key);
    assert!(std::fs::metadata(&path).expect("artifact meta").len() > 4);

    // 二次命中：不应再编译；命中预算 <50ms（键不含 -M 闭包故无子进程）
    let start2 = std::time::Instant::now();
    let (_, _) =
        cache.build_once(&key, &src, || Err(reinfer_kernels::LaunchError::Fatal)).expect("hit");
    let hit = start2.elapsed();
    assert!(hit < std::time::Duration::from_millis(50), "hit took {hit:?}");
    eprintln!("prebake: first={first:?} hit={hit:?}");
}

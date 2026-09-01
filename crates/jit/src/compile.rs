//! 编译子进程：`nvcc -cubin`（012 r1 R5：禁用 `-shared`）+ `-M` 漂移校验。
//!
//! 产物形态（工具链评审实测）：`nvcc -cubin -gencode arch=compute_XXX,code=sm_XXX`
//! 裸 cubin 经 `cuLibraryLoadData`/`cuModuleLoadData` 加载均可；`-shared -fPIC`
//! 的主机链接产物在 sm_120 判定机上 `cuLibraryGetKernel` 恒报 200，故禁用。
//!
//! `-M`（头闭包）在**构建期**执行（键为嵌入内容而非闭包——r1 R4）：
//! 与 `KernelSource.headers` 按**相对路径**比对；命中"已列头"之外的已知
//! 闭包含新增文件 → 报错（防漏列头引发陈旧命中）。系统头（cuda_runtime
//! 等）忽略。

use crate::toolchain::resolve_nvcc;
use crate::types::{KernelSource, ToolchainId};
use reinfer_kernels::LaunchError;
use std::path::Path;
use std::process::Command;

/// 由规范串生成 `-gencode` 参数（`sm_120a` → `arch=compute_120,code=sm_120a`）。
/// 支持 `sm_\d+` / `sm_\d+a`；其它（如平台特定名）→ 原样 `-gencode arch={arch}`。
pub fn gencode_flags(arch: &str) -> Result<Vec<String>, LaunchError> {
    let core = arch.strip_prefix("sm_").ok_or(LaunchError::Fatal)?;
    let digit_end = core.find(|c: char| !c.is_ascii_digit()).unwrap_or(core.len());
    let num = &core[..digit_end];
    let suffix = &core[digit_end..];
    if num.is_empty() || !suffix.is_empty() && suffix != "a" {
        return Err(LaunchError::Fatal);
    }
    // -gencode 与值为分离参数；`-cubin` 下单 gencode（多实例被 nvcc 拒绝——
    // 工具链实测："Option '--cubin' is not allowed ... multiple GPU code instances"）
    Ok(vec!["-gencode".to_string(), format!("arch=compute_{num},code={arch}")])
}

/// 编译 flags：固定项（-cubin / -std=c++17；**不加** -use_fast_math）+ 调用方 flags。
pub fn build_flags(src: &KernelSource) -> Vec<String> {
    let mut f = vec!["-cubin".to_string(), "-std=c++17".to_string()];
    f.extend(src.flags.iter().cloned());
    f
}

/// 编译：临时目录布局 `<tmp>/<name>.cu` + `headers/`（按声明相对路径
/// 落盘——嵌套树如 cute/cutlass 与同名不同目录的头），
/// 运行 `nvcc <flags>` → 收集 stdout 字节。
///
/// 工具链一致性：使用调用方给定的 `ToolchainId.realpath`（探测/自动选择的
/// 那个）——若不可用再回退解析链；绝不允许"探测选 A、编译用 B"。
pub fn compile_cubin(src: &KernelSource, tc: &ToolchainId) -> Result<Vec<u8>, LaunchError> {
    let nvcc = if tc.realpath.is_file() { tc.realpath.clone() } else { resolve_nvcc()? };
    let tmp = std::env::temp_dir().join(format!(
        "reinfer-jit-compile-{}-{}",
        src.name,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| crate::error::fs_err(&e))?;

    let cu = tmp.join(format!("{}.cu", src.name));
    std::fs::write(&cu, src.src).map_err(|e| crate::error::fs_err(&e))?;
    let headers_dir = tmp.join("headers");
    if !src.headers.is_empty() {
        std::fs::create_dir_all(&headers_dir).map_err(|e| crate::error::fs_err(&e))?;
        for h in &src.headers {
            // 按相对路径落盘（保留目录结构；头内的 #include 相对解析与
            // 键内容哈希都依赖该布局——basename 展开会破坏嵌套树）。
            let dst = headers_dir.join(&h.path);
            if let Some(parent) = dst.parent() {
                std::fs::create_dir_all(parent).map_err(|e| crate::error::fs_err(&e))?;
            }
            std::fs::write(&dst, &h.content).map_err(|e| crate::error::fs_err(&e))?;
        }
    }

    // 构建期 -M 漂移校验：编译吃到的头（临时目录内）必须**已被声明**
    // （consumed ⊆ declared；多声明的 opt-in/守护头 inert——只增大键）。
    if !src.headers.is_empty() {
        let dep = run_depfile(&nvcc, &cu, &headers_dir, &src.arch)?;
        verify_closure_files(&dep, src, &headers_dir)?;
    }

    let mut cmd = Command::new(&nvcc);
    for f in build_flags(src) {
        cmd.arg(f);
    }
    if !src.headers.is_empty() {
        cmd.arg(format!("-I{}", headers_dir.display()));
    }
    let out =
        cmd.arg("--output-file").arg(tmp.join(format!("{}.cubin", src.name))).arg(&cu).output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            eprintln!("reinfer-jit: nvcc exec failed: {e}");
            return Err(LaunchError::Fatal);
        }
    };
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        eprintln!("reinfer-jit: nvcc compile failed ({}):\n{}", out.status, tail_of(&stderr));
        return Err(LaunchError::Fatal); // 消息经 stderr 尾部（契约）
    }
    // 防御：编译产物必须存在且为 ELF
    let cubin = tmp.join(format!("{}.cubin", src.name));
    let bytes = std::fs::read(&cubin).map_err(|e| {
        eprintln!("reinfer-jit: nvcc left no cubin: {e}");
        crate::error::fs_err(&e)
    })?;
    if bytes.len() < 4 || &bytes[..4] != b"\x7fELF" {
        eprintln!("reinfer-jit: unexpected output (not an ELF): {} bytes", bytes.len());
        return Err(LaunchError::Fatal);
    }
    Ok(bytes)
}

/// 运行 `nvcc -M` 输出闭包（.d 文本）。
///
/// 传 `-D__CUDA_ARCH__=<archnum>`：cute/cutlass 的 `#if __CUDA_ARCH__ >= 900`
/// 守护 include（sm90/sm100 MMA、TMA 等）在无 arch 的 -M 预处理中不可见，
/// 导致 depfile 漏列真实 device pass 会吃到的头。archnum 由 `sm_120a` → 1200
/// 解析（与 gencode 的目标一致）。
fn run_depfile(
    nvcc: &Path,
    cu: &Path,
    headers_dir: &Path,
    arch: &str,
) -> Result<String, LaunchError> {
    let mut cmd = Command::new(nvcc);
    cmd.arg("-M");
    if !arch.is_empty() {
        cmd.arg(format!("-D__CUDA_ARCH__={}", arch_number(arch)));
    }
    cmd.arg(format!("-I{}", headers_dir.display())).arg("-c").arg(cu);
    let out = cmd.output().map_err(|e| {
        eprintln!("reinfer-jit: nvcc -M exec failed: {e}");
        LaunchError::Fatal
    })?;
    if !out.status.success() {
        eprintln!("reinfer-jit: nvcc -M failed: {}", String::from_utf8_lossy(&out.stderr));
        return Err(LaunchError::Fatal);
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// `sm_120a` → 1200（device pass 的 `__CUDA_ARCH__` 值）。
fn arch_number(arch: &str) -> u32 {
    let digits: String = arch
        .strip_prefix("sm_")
        .unwrap_or(arch)
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    let v: u32 = digits.parse().unwrap_or(0);
    (v / 10) * 100 + (v % 10)
}

/// 校验：`-M` 闭包内、落在 headers 目录下的文件**相对路径**集合 ⊆ 声明的
/// headers 集合。只对"编译吃到了未声明的头"报错（键必须覆盖真实消耗；
/// 路径级比较，避免 cute/cutlass 的同名头误判）。
///
/// 声明多于消耗是**容忍**的：vendor 抽取闭包不识别宏守护（如
/// `CUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED` 下的扩展 MMA 头、RTC 专用头），
/// 这些头从未被编译打开——多声明只增大键哈希，无正确性影响。
fn verify_closure_files(
    dep: &str,
    src: &KernelSource,
    headers_dir: &Path,
) -> Result<(), LaunchError> {
    let joined = dep.replace('\\', "\n");
    let declared: std::collections::BTreeSet<&str> =
        src.headers.iter().map(|h| h.path.as_str()).collect();
    let mut consumed: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let dir_marker = headers_dir.to_string_lossy().replace('\\', "/");
    for token in joined.split_whitespace() {
        if token.ends_with(':') {
            continue; // depfile 目标行（"foo.cu:"）
        }
        if let Some(rel) = token.strip_prefix(&dir_marker) {
            let rel = rel.trim_start_matches(['/', '\\']);
            if !rel.is_empty() {
                consumed.insert(rel);
            }
        }
    }
    if !consumed.is_subset(&declared) {
        let undeclared: Vec<&str> = consumed.difference(&declared).copied().collect();
        eprintln!(
            "reinfer-jit: header closure mismatch - consumed-but-undeclared={undeclared:?} \
             (compiled headers must be part of the JIT key)"
        );
        return Err(LaunchError::Fatal);
    }
    Ok(())
}

fn tail_of(s: &str) -> String {
    s.lines().skip(s.lines().count().saturating_sub(8)).collect::<Vec<_>>().join("\n")
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;

    #[test]
    fn gencode_from_sm() {
        assert_eq!(
            gencode_flags("sm_120a").unwrap(),
            vec!["-gencode".to_string(), "arch=compute_120,code=sm_120a".to_string()]
        );
        assert_eq!(gencode_flags("sm_90").unwrap().len(), 2);
        assert!(gencode_flags("bm_120").is_err());
        assert!(gencode_flags("sm_12a0").is_err());
    }

    #[test]
    fn build_flags_skip_fast_math_by_default() {
        let s = vec![];
        let src = KernelSource {
            name: "k",
            src: "src",
            headers: vec![],
            flags: s,
            arch: "sm_120a".into(),
            toolchain_ver: "rel 12.8".into(),
        };
        let f = build_flags(&src);
        assert_eq!(f[0], "-cubin");
        assert!(!f.iter().any(|x| x.contains("fast_math")));
    }

    #[test]
    fn closure_mismatch_uses_rel_paths() {
        // 同名不同目录的头必须按相对路径区分（cute/cutlass 场景）。
        let src = KernelSource {
            name: "k",
            src: "#include <cutlass/x/y.h>\n",
            headers: vec![
                crate::types::HeaderFile { path: "cutlass/x/y.h".into(), content: "// a".into() },
                crate::types::HeaderFile { path: "cutlass/z/y.h".into(), content: "// b".into() },
            ],
            flags: vec![],
            arch: "sm_120a".into(),
            toolchain_ver: "rel 12.8".into(),
        };
        let headers_dir = Path::new("/tmp/reinfer-jit-test/headers");
        // 两个同名头都进了 -M 闭包（绝对路径）→ 通过。
        let dep = format!(
            "k.cu: \\\n {}/cutlass/x/y.h \\\n {}/cutlass/z/y.h\n",
            headers_dir.display(),
            headers_dir.display()
        );
        assert!(verify_closure_files(&dep, &src, headers_dir).is_ok());
        // 只消耗一个（另一个是宏守护下的 opt-in 头，从未被编译打开）
        // → 声明超集被容忍（consumed ⊆ declared）。
        let dep = format!("k.cu: {}/cutlass/x/y.h\n", headers_dir.display());
        assert!(verify_closure_files(&dep, &src, headers_dir).is_ok());
        // 消耗了未声明的头 → 硬错误（键必须覆盖编译真实吃到的头）。
        let dep = format!("k.cu: {}/cutlass/w/un.h\n", headers_dir.display());
        assert!(verify_closure_files(&dep, &src, headers_dir).is_err());
    }

    #[test]
    fn arch_number_maps_sm_to_cuda_arch() {
        assert_eq!(arch_number("sm_120a"), 1200);
        assert_eq!(arch_number("sm_90a"), 900);
        assert_eq!(arch_number("sm_80"), 800);
        assert_eq!(arch_number("sm_100a"), 1000);
        assert_eq!(arch_number("sm_110"), 1100);
        assert_eq!(arch_number(""), 0);
    }
}

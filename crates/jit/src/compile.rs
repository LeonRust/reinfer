//! 编译子进程：`nvcc -cubin`（012 r1 R5：禁用 `-shared`）+ `-M` 漂移校验。
//!
//! 产物形态（工具链评审实测）：`nvcc -cubin -gencode arch=compute_XXX,code=sm_XXX`
//! 裸 cubin 经 `cuLibraryLoadData`/`cuModuleLoadData` 加载均可；`-shared -fPIC`
//! 的主机链接产物在 sm_120 判定机上 `cuLibraryGetKernel` 恒报 200，故禁用。
//!
//! `-M`（头闭包）在**构建期**执行（键为嵌入内容而非闭包——r1 R4）：
//! 与 `KernelSource.headers` 按 basename 比对；命中"已列头"之外的已知闭包含
//! 新增文件 → 报错（防漏列头引发陈旧命中）。系统头（cuda_runtime 等）忽略。

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

/// 编译：临时目录布局 `<tmp>/<name>.cu` + `headers/`（按 basename），
/// 运行 `nvcc <flags>` → 收集 stdout 字节。
pub fn compile_cubin(src: &KernelSource, _tc: &ToolchainId) -> Result<Vec<u8>, LaunchError> {
    let nvcc = resolve_nvcc()?;
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
            let base = header_basename(&h.path);
            std::fs::write(headers_dir.join(base), &h.content)
                .map_err(|e| crate::error::fs_err(&e))?;
        }
    }

    // 构建期 -M 漂移校验：编译吃到的头（临时目录内）必须与声明一致
    if !src.headers.is_empty() {
        let dep = run_depfile(&nvcc, &cu, &headers_dir)?;
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
fn run_depfile(nvcc: &Path, cu: &Path, headers_dir: &Path) -> Result<String, LaunchError> {
    let out = Command::new(nvcc)
        .arg("-M")
        .arg(format!("-I{}", headers_dir.display()))
        .arg("-c")
        .arg(cu)
        .output()
        .map_err(|e| {
            eprintln!("reinfer-jit: nvcc -M exec failed: {e}");
            LaunchError::Fatal
        })?;
    if !out.status.success() {
        eprintln!("reinfer-jit: nvcc -M failed: {}", String::from_utf8_lossy(&out.stderr));
        return Err(LaunchError::Fatal);
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// 校验：`-M` 闭包内、落在 headers 目录下的文件 basename 集合 == 声明的
/// headers 集合（漏列/多列都报错——防"编译吃到的头不在键内"）。
fn verify_closure_files(
    dep: &str,
    src: &KernelSource,
    headers_dir: &Path,
) -> Result<(), LaunchError> {
    let joined = dep.replace('\\', "\n");
    let declared: std::collections::BTreeSet<&str> =
        src.headers.iter().map(|h| header_basename(&h.path)).collect();
    let mut consumed: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let dir_marker = headers_dir.to_string_lossy().replace('\\', "/");
    for token in joined.split_whitespace() {
        if let Some(base) = token.rsplit(['/', '\\']).next()
            && (token.starts_with(&dir_marker) || token.contains("headers/"))
        {
            consumed.insert(base);
        }
    }
    if consumed != declared {
        let missing: Vec<&str> = declared.difference(&consumed).copied().collect();
        let extra: Vec<&str> = consumed.difference(&declared).copied().collect();
        eprintln!("reinfer-jit: header closure mismatch - missing={missing:?} extra={extra:?}");
        return Err(LaunchError::Fatal);
    }
    Ok(())
}

fn tail_of(s: &str) -> String {
    s.lines().skip(s.lines().count().saturating_sub(8)).collect::<Vec<_>>().join("\n")
}

/// 头文件名 basename（诊断与临时目录布局）。
fn header_basename(path: &str) -> &str {
    path.rsplit(['/', '\\']).next().unwrap_or(path)
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
    fn header_basename_extraction() {
        assert_eq!(header_basename("/a/b/foo.h"), "foo.h");
        assert_eq!(header_basename("b/foo.h"), "foo.h");
        assert_eq!(header_basename("foo.h"), "foo.h");
    }
}

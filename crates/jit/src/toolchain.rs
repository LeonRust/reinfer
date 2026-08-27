//! nvcc 工具链探测（012 r1：解析链 + 梯度检查 + 版本归一）。
//!
//! 解析链：`REINFER_CUDA_NVCC` → `CUDA_HOME/bin/nvcc` → `CUDA_PATH/bin/nvcc`
//! → `PATH` 上的 `nvcc`。梯度表为**实测基线**：sm_90a ≥12.3 / sm_100a ≥12.8 /
//! sm_120a ≥12.8（工具链评审 r1：12.6 不支持 sm_120；12.8 起支持且实测可用）。
//! 未知 `sm_*` 与平台特定架构放行（交由编译后端裁决）。
//!
//! 错误面：`LaunchError` 无 payload（跨后端契约）——人读消息经
//! [`messages`] 常量 + stderr 诊断输出（fail-closed 分类不变）。

use crate::types::ToolchainId;
use reinfer_kernels::LaunchError;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 显式指定 nvcc 的环境变量。
pub const NVCC_ENV: &str = "REINFER_CUDA_NVCC";

/// 人读诊断消息（stderr 输出 + 单测锚点；分类语义见 `LaunchError`）。
pub mod messages {
    /// nvcc 缺失提示。
    pub const NVCC_MISSING: &str =
        "jit: nvcc not found - set REINFER_CUDA_NVCC (or CUDA_HOME/CUDA_PATH/PATH)";
    /// 版本过旧提示（判据实测基线）。
    pub const NVCC_TOO_OLD: &str =
        "jit: nvcc too old for the target arch (sm_120a >= 12.8, sm_100a >= 12.8, sm_90a >= 12.3)";
}

/// 由显式候选解析 nvcc（无 env/无 PATH；测试与复用入口）。
pub fn resolve_at(pick: Option<&str>) -> Result<PathBuf, LaunchError> {
    match pick.map(PathBuf::from) {
        Some(b) if b.is_file() => Ok(b),
        _ => Err(LaunchError::Fatal),
    }
}

/// 解析链 nvcc（env 优先）。
pub fn resolve_nvcc() -> Result<PathBuf, LaunchError> {
    if let Some(p) = std::env::var(NVCC_ENV).ok().filter(|p| !p.is_empty()) {
        let b = PathBuf::from(p);
        if b.is_file() {
            return Ok(b);
        }
        eprintln!("reinfer-jit: {} not found", std::env::var(NVCC_ENV).unwrap_or_default());
        return Err(LaunchError::Fatal);
    }
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(home) = std::env::var(var).ok().filter(|h| !h.is_empty()) {
            let cand = PathBuf::from(home).join("bin/nvcc");
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }
    if let Some(p) = find_on_path("nvcc") {
        return Ok(p);
    }
    eprintln!("reinfer-jit: {}", messages::NVCC_MISSING);
    Err(LaunchError::Fatal)
}

/// 在 PATH 上查找可执行文件。
fn find_on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(name)).find(|p| p.is_file())
}

/// 解析 nvcc `--version` 首行 → (major, minor)。
///
/// 示例行：`Cuda compilation tools, release 12.8, V12.8.61`
pub fn parse_nvcc_version(line: &str) -> Option<(u32, u32)> {
    let idx = line.find("release ")?;
    let rest = line[idx + "release ".len()..].split(',').next()?.trim();
    let mut it = rest.split('.');
    let major: u32 = it.next()?.trim().parse().ok()?;
    let minor: u32 = it.next().map(|s| s.trim().parse().ok()).unwrap_or(Some(0))?;
    Some((major, minor))
}

/// 梯度检查：已知 sm 档的最小 nvcc 版本（实测基线 012 r1 R6）。
pub fn check_arch_supported(arch: &str, ver: (u32, u32)) -> Result<(), LaunchError> {
    let min: Option<(u32, u32)> = match arch {
        "sm_90" | "sm_90a" => Some((12, 3)),
        "sm_100" | "sm_100a" => Some((12, 8)),
        "sm_120" | "sm_120a" => Some((12, 8)),
        _ => None, // 其它 sm_* / 平台特定架构：未知判据，交由编译后端裁决
    };
    if let Some((mn, mnr)) = min
        && ver < (mn, mnr)
    {
        eprintln!("reinfer-jit: {} ({arch} needs {mn}.{mnr})", messages::NVCC_TOO_OLD);
        return Err(LaunchError::Fatal);
    }
    Ok(())
}

/// 运行 nvcc 取版本首行（stdout/stderr 任一）。
fn nvcc_ver_line(nvcc: &Path) -> Result<String, LaunchError> {
    let out = Command::new(nvcc).arg("--version").output();
    let out = match out {
        Ok(o) => o,
        Err(e) => {
            eprintln!("reinfer-jit: nvcc --version failed: {e}");
            return Err(LaunchError::Fatal);
        }
    };
    for text in [String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr)] {
        for line in text.lines() {
            if line.contains("release ") {
                return Ok(line.trim().to_string());
            }
        }
    }
    eprintln!("reinfer-jit: nvcc --version gave no release line");
    Err(LaunchError::Fatal)
}

/// 探测宿主编译器（`-ccbin` 默认；g++ 优先，次 clang++；缺失 → 未知占位）。
fn probe_ccbin() -> (PathBuf, String) {
    for cand in ["g++", "clang++"] {
        if let Some(p) = find_on_path(cand)
            && let Some(v) = Command::new(&p).arg("--version").output().ok()
        {
            let line =
                String::from_utf8_lossy(&v.stdout).lines().next().unwrap_or("").trim().to_string();
            if !line.is_empty() {
                return (p, line);
            }
        }
    }
    (PathBuf::from("unknown"), String::new())
}

/// 完整工具链探测：nvcc 解析链 → 版本行 → `ToolchainId`。
/// （梯度检查按目标 arch 由 Jit provider 在构造时用
/// [`check_arch_supported`] 执行——probe 阶段尚不知 arch。）
pub fn probe_toolchain() -> Result<ToolchainId, LaunchError> {
    let nvcc = resolve_nvcc()?;
    let realpath = std::fs::canonicalize(&nvcc).unwrap_or(nvcc);
    let ver_line = nvcc_ver_line(&realpath)?;
    let _ = parse_nvcc_version(&ver_line).ok_or_else(|| {
        eprintln!("reinfer-jit: cannot parse nvcc version line: {ver_line}");
        LaunchError::Fatal
    })?;
    Ok(ToolchainId { ver_line, realpath, ccbin: probe_ccbin() })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;

    #[test]
    fn parse_release_line() {
        assert_eq!(
            parse_nvcc_version("Cuda compilation tools, release 12.8, V12.8.61"),
            Some((12, 8))
        );
        assert_eq!(parse_nvcc_version("release 13.2, V13.2.0"), Some((13, 2)));
        assert_eq!(parse_nvcc_version("no release here"), None);
    }

    #[test]
    fn gradient_baseline() {
        assert!(check_arch_supported("sm_120a", (12, 8)).is_ok());
        assert!(matches!(check_arch_supported("sm_120a", (12, 6)), Err(LaunchError::Fatal)));
        assert!(check_arch_supported("sm_90a", (12, 6)).is_ok());
        assert!(check_arch_supported("sm_100a", (12, 7)).is_err());
        assert!(check_arch_supported("sm_99b", (11, 0)).is_ok()); // 未知放行
        assert!(check_arch_supported("ms_120", (12, 8)).is_ok()); // 非 CUDA 名放行
    }

    #[test]
    fn resolve_at_rejects_empty_and_missing() {
        assert!(resolve_at(Some("")).is_err());
        assert!(resolve_at(None).is_err());
        assert!(resolve_at(Some("/nonexistent/nvcc")).is_err());
    }
}

//! 平台无关的内核源与工具链标识（契约锚：specs/012 plan Interface Contracts）。
//!
//! `crates/jit` 零 unsafe、零 CUDA 依赖：AscendC（bisheng）经同一
//! `KernelSource`/`ToolchainId` 复用本缓存层。字段命名为平台无关语义
//!（`toolchain_ver` 而非 `nvcc_ver`——011/002 评审 R1 裁决）。

use std::path::PathBuf;

/// 头文件项。路径仅作诊断展示；**键计算只取内容**（路径不入键，换机可命中）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HeaderFile {
    /// 相对/绝对路径（诊断用）。
    pub path: String,
    /// 文件内容（入键哈希来源）。
    pub content: Vec<u8>,
}

/// 内核源：一个可编译产物的全部确定性输入。
///
/// - `src` 为编译期常量（`include_str!` 自 `crates/cuda/kernels/`，003 约束）；
/// - `flags` 保持**调用方原始顺序**（`-I`/`-include`/`-Xcompiler` 顺序敏感，禁排序）；
/// - `arch` 规范串：`"sm_120"` / `"sm_120a"` / `"ascend910b3"`。
#[derive(Debug, Clone)]
pub struct KernelSource {
    /// 导出符号名（内核一律 `extern "C" __global__` 导出，012 r1 裁决）。
    pub name: &'static str,
    /// `.cu`/内核源码文本。
    pub src: &'static str,
    /// 头文件集（内容哈希入键；构建期与 `-M` 闭包做漂移校验）。
    pub headers: Vec<HeaderFile>,
    /// 编译参数（最终展开后的实际参数，原始顺序）。
    pub flags: Vec<String>,
    /// 目标架构规范串。
    pub arch: String,
    /// 编译器版本行（`nvcc --version` 首行 / bisheng 对应行）。
    pub toolchain_ver: String,
}

/// 工具链身份（键与 meta 均记录；防"同版本行不同工具链"误命中）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolchainId {
    /// 编译器版本行（规范化首行）。
    pub ver_line: String,
    /// 编译器可执行文件 realpath。
    pub realpath: PathBuf,
    /// 宿主编译器：（realpath，版本首行）。
    pub ccbin: (PathBuf, String),
}

impl std::fmt::Display for ToolchainId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} @{}{}", self.ver_line, self.realpath.display(), self.ccbin.0.display())
    }
}

//! JitKey：编译产物缓存键（012 r1 契约；零子进程——嵌入内容哈希）。
//!
//! 编码（元素序 = 长度前缀连接，防混叠）：
//! 1. `source` 字节；
//! 2. headers **内容哈希列表**：先按路径排序获得稳定次序，再取每条内容
//!    sha256，最后对哈希列表排序——**路径字符串不入键**（同内容不同
//!    路径 → 同键，换机可命中）；
//! 3. flags **原始顺序**（`-I`/`-include`/`-Xcompiler` 顺序敏感，禁排序）；
//! 4. toolchain 版本行 + nvcc realpath + `-ccbin`（realpath+版本首行）；
//! 5. arch 规范串；
//! 6. host triple（`std::env::consts::TARGET`，防交叉编译误命中）。
//!
//! `-M` 头闭包不参与键（单次 ~90ms 与 <50ms 命中预算冲突），降级为
//! 构建期漂移校验（见 `compile.rs`）。

use crate::types::{HeaderFile, KernelSource, ToolchainId};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// 编译产物缓存键（hex 用于文件名与显示；前 2 字符为目录分片）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JitKey([u8; 32]);

impl JitKey {
    /// 由原始字节重建（meta 回读/测试）。
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// 由确定性输入计算（全内存操作，无子进程）。
    pub fn new(src: &KernelSource, tc: &ToolchainId) -> Self {
        let mut b: Vec<u8> = Vec::new();
        push(&mut b, src.src.as_bytes());
        // headers：按路径排序（稳定次序）→ 内容哈希列表 → 排序（路径不入键）
        let mut by_path: Vec<&HeaderFile> = src.headers.iter().collect();
        by_path.sort_by(|a, b| a.path.cmp(&b.path));
        let mut content_hashes: Vec<[u8; 32]> =
            by_path.iter().map(|h| Sha256::digest(&h.content).into()).collect();
        content_hashes.sort();
        for h in &content_hashes {
            push(&mut b, h);
        }
        for f in &src.flags {
            push(&mut b, f.as_bytes());
        }
        push(&mut b, src.arch.as_bytes());
        push(&mut b, tc.ver_line.as_bytes());
        push(&mut b, tc.realpath.to_string_lossy().as_bytes());
        push(&mut b, tc.ccbin.0.to_string_lossy().as_bytes());
        push(&mut b, tc.ccbin.1.as_bytes());
        // 目标三元组：cargo 注入（部分工具链未设）→ 平台常量拼接兜底
        let host = option_env!("TARGET").map(str::to_string).unwrap_or_else(|| {
            format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
        });
        push(&mut b, host.as_bytes());
        Self(Sha256::digest(&b).into())
    }

    /// 原始字节（meta 回读校验）。
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// 完整 hex（文件名/显示）。
    pub fn hex(&self) -> String {
        self.0.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// 目录分片：hex 前 2 字符。
    pub fn dir_prefix(&self) -> String {
        self.hex()[..2].to_string()
    }
}

impl std::fmt::Display for JitKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.hex())
    }
}

/// 长度前缀写入：8 字节 LE 长度 + 数据（防拼接混叠）。
fn push(buf: &mut Vec<u8>, data: &[u8]) {
    buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
    buf.extend_from_slice(data);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn header(path: &str, content: &[u8]) -> HeaderFile {
        HeaderFile { path: path.into(), content: content.to_vec() }
    }

    fn toolchain() -> ToolchainId {
        ToolchainId {
            ver_line: "Cuda compilation tools, release 12.8".into(),
            realpath: PathBuf::from("/usr/local/cuda-12.8/bin/nvcc"),
            ccbin: (PathBuf::from("/usr/bin/g++-12"), "g++ 12.3.0".into()),
        }
    }

    fn src(
        name: &'static str,
        src: &'static str,
        headers: Vec<HeaderFile>,
        flags: Vec<String>,
    ) -> KernelSource {
        KernelSource {
            name,
            src,
            headers,
            flags,
            arch: "sm_120a".into(),
            toolchain_ver: "Cuda compilation tools, release 12.8".into(),
        }
    }

    #[test]
    fn any_input_change_rekeys() {
        let base = src("k", "src", vec![header("a.h", b"A")], vec!["-DNDEBUG".into()]);
        let k0 = JitKey::new(&base, &toolchain());
        let k1 = JitKey::new(
            &src("k", "srcv2", vec![header("a.h", b"A")], vec!["-DNDEBUG".into()]),
            &toolchain(),
        );
        let k2 = JitKey::new(
            &base,
            &ToolchainId { ver_line: "release 13.0".into(), ..toolchain() },
        );
        assert_ne!(k0, k1); // 源码变
        assert_ne!(k0, k2); // 工具链变
        assert_eq!(JitKey::new(&base, &toolchain()), k0); // 稳定
    }

    #[test]
    fn path_free_same_content_same_key() {
        let a = src("k", "src", vec![header("/a/foo.h", b"ABC"), header("/b/bar.h", b"DEF")], vec![]);
        let b = src("k", "src", vec![header("/x/bar.h", b"DEF"), header("/y/foo.h", b"ABC")], vec![]);
        assert_eq!(JitKey::new(&a, &toolchain()), JitKey::new(&b, &toolchain()));
    }

    #[test]
    fn flags_order_sensitive() {
        // 交换两个 -I：顺序敏感，键必须不同（防头文件遮蔽反转静默命中）
        let a = src("k", "src", vec![], vec!["-IincA".into(), "-IincB".into()]);
        let b = src("k", "src", vec![], vec!["-IincB".into(), "-IincA".into()]);
        assert_ne!(JitKey::new(&a, &toolchain()), JitKey::new(&b, &toolchain()));
    }

    #[test]
    fn length_prefix_prevents_concatenation_aliasing() {
        // 无长度前缀时 src"ab"+header"c" 与 src"a"+header"bc" 字节串相同；有前缀必须不同
        let a = src("k", "ab", vec![header("h.h", b"c")], vec![]);
        let b = src("k", "a", vec![header("h.h", b"bc")], vec![]);
        assert_ne!(JitKey::new(&a, &toolchain()), JitKey::new(&b, &toolchain()));
    }
}

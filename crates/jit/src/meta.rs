//! meta.json 读写与完整性校验（012 r1：meta 为提交点，含产物 sha256）。
//!
//! 配套约定（见 `cache.rs`）：`.cubin` 先 temp+rename 落盘，`write_meta` 的
//! rename 作为提交点——`try_load` 只在两者都可见且哈希一致时判定命中。

use crate::error::fs_err;
use crate::key::JitKey;
use reinfer_kernels::LaunchError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// 产物元数据（序列化到 `<key>.meta.json`）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JLibMeta {
    /// 键（hex 形态经 serde；`JitKey::new` 可重建）。
    pub key: JitKey,
    /// 架构规范串。
    pub arch: String,
    /// 编译器版本行。
    pub toolchain_ver: String,
    /// `.cubin` 内容 sha256（try_load 校验，防坏字节静默命中）。
    pub sha256: String,
    /// `.cubin` 字节数。
    pub size: u64,
    /// gencode 全量数组（诊断/跨设备核对）。
    pub gencode: Vec<String>,
    /// 创建时间戳（分钟精度即可；仅诊断）。
    pub created_at: u64,
}

/// `<dir>/<key>.cubin` 路径。
pub(crate) fn cubin_path(dir: &Path, key: &JitKey) -> PathBuf {
    dir.join(format!("{}.cubin", key.hex()))
}

/// `<dir>/<key>.meta.json` 路径。
pub(crate) fn meta_path(dir: &Path, key: &JitKey) -> PathBuf {
    dir.join(format!("{}.meta.json", key.hex()))
}

/// 读取 + 完整性校验。不存在/JSON 损坏/键不匹配 → `Ok(None)`（miss 语义；
/// 清理与重建由 `build_once` 持锁路径负责）。
pub(crate) fn read_meta(dir: &Path, key: &JitKey) -> Result<Option<JLibMeta>, LaunchError> {
    let p = meta_path(dir, key);
    let bytes = match std::fs::read(&p) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(fs_err(&e)),
    };
    let meta: JLibMeta = match serde_json::from_slice(&bytes) {
        Ok(m) => m,
        // 损坏 → miss（宁重建不猜测）
        Err(_) => return Ok(None),
    };
    if meta.key != *key {
        return Ok(None);
    }
    Ok(Some(meta))
}

/// 原子写：同目录 temp + rename（提交点）。调用方须持锁（`cache.rs` 签名保证）。
pub(crate) fn write_meta(dir: &Path, key: &JitKey, meta: &JLibMeta) -> Result<(), LaunchError> {
    let tmp = dir.join(format!(".{}.meta.tmp.{}", key.hex(), std::process::id()));
    let bytes = serde_json::to_vec(meta).map_err(|e| {
        eprintln!("reinfer-jit: meta serialize error: {e}");
        LaunchError::Fatal
    })?;
    std::fs::write(&tmp, bytes).map_err(|e| fs_err(&e))?;
    std::fs::rename(&tmp, meta_path(dir, key)).map_err(|e| fs_err(&e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_key() -> JitKey {
        JitKey::from_bytes([0xAA; 32])
    }

    fn dummy_meta(key: &JitKey) -> JLibMeta {
        JLibMeta {
            key: *key,
            arch: "sm_120a".into(),
            toolchain_ver: "release 12.8".into(),
            sha256: "deadbeef".into(),
            size: 1,
            gencode: vec!["arch=compute_120,code=sm_120a".into()],
            created_at: 1,
        }
    }

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("reinfer-jit-meta-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let key = dummy_key();
        write_meta(&dir, &key, &dummy_meta(&key)).unwrap();
        let got = read_meta(&dir, &key).unwrap().expect("present");
        assert_eq!(got, dummy_meta(&key));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_is_none() {
        let dir = std::env::temp_dir().join(format!(
            "reinfer-jit-meta-none-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let key = dummy_key();
        assert!(read_meta(&dir, &key).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_json_is_miss() {
        let dir = std::env::temp_dir().join(format!(
            "reinfer-jit-meta-corrupt-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let key = dummy_key();
        std::fs::write(meta_path(&dir, &key), b"{not json").unwrap();
        assert!(read_meta(&dir, &key).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn key_mismatch_is_miss() {
        let dir = std::env::temp_dir().join(format!(
            "reinfer-jit-meta-mismatch-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let key = dummy_key();
        write_meta(&dir, &key, &dummy_meta(&key)).unwrap();
        let other = JitKey::from_bytes([0xBB; 32]);
        assert!(read_meta(&dir, &other).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

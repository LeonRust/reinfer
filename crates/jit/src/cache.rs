//! JitCache：跨进程编译产物缓存（012 r1 契约）。
//!
//! 布局：`REINFER_JIT_CACHE` 或 `<XDG_CACHE_HOME|~/.cache>/reinfer/jit/`；
//! `<key[..2]>/<key>.cubin + <key>.meta.json`。
//! 原子性：`.cubin` 与 meta 均同目录 temp+rename；**meta 为提交点**；
//! `try_load` 校验产物存在、大小与 sha256 与 meta 一致，否则 miss。

use crate::JitKey;
use crate::error::fs_err;
use crate::lock::{self, JitLockGuard};
use crate::meta::{JLibMeta, cubin_path, meta_path, read_meta, write_meta};
use crate::types::KernelSource;
use reinfer_kernels::LaunchError;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// 缓存根环境变量（显式覆盖）。
pub const CACHE_DIR_ENV: &str = "REINFER_JIT_CACHE";

/// 残留 temp 清理阈值（open 时清扫）。
const STALE_TEMP_SECS: u64 = 3600;

/// Jit 编译产物缓存。
#[derive(Debug)]
pub struct JitCache {
    dir: PathBuf,
}

impl JitCache {
    /// 打开缓存：创建目录、清扫过期残留 temp。
    pub fn open(dir: Option<PathBuf>) -> Result<Self, LaunchError> {
        let dir = match dir.or_else(|| std::env::var(CACHE_DIR_ENV).ok().map(PathBuf::from)) {
            Some(d) => d,
            None => default_dir(),
        };
        std::fs::create_dir_all(&dir).map_err(|e| fs_err(&e))?;
        clean_stale_temps(&dir)?;
        Ok(Self { dir })
    }

    /// 缓存根目录。
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// 命中检查：meta 存在且完整、产物存在且 sha256 一致 → `(meta, cubin 路径)`；
    /// 其余（缺失/损坏/错配）一律 `Ok(None)`（miss 语义）。
    pub fn try_load(&self, key: &JitKey) -> Result<Option<(JLibMeta, PathBuf)>, LaunchError> {
        let sub = self.dir.join(key.dir_prefix());
        let Some(meta) = read_meta(&sub, key)? else {
            return Ok(None);
        };
        let path = cubin_path(&sub, key);
        let data = match std::fs::read(&path) {
            Ok(d) => d,
            Err(_) => return Ok(None),
        };
        if data.len() != meta.size as usize || hex_sha256(&data) != meta.sha256 {
            return Ok(None);
        }
        Ok(Some((meta, path)))
    }

    /// 写产物（持锁约定：`&JitLockGuard` 签名强制；.cubin 先、meta>提交点）。
    pub fn store(
        &self,
        key: &JitKey,
        _guard: &JitLockGuard,
        bytes: &[u8],
        meta: &JLibMeta,
    ) -> Result<(), LaunchError> {
        // 防御：meta 哈希必须与内容一致（防止调用方错配静默损坏）
        let expected = hex_sha256(bytes);
        if meta.sha256 != expected || meta.size != bytes.len() as u64 {
            return Err(LaunchError::Fatal);
        }
        if meta.key != *key {
            return Err(LaunchError::Fatal);
        }
        let sub = self.dir.join(key.dir_prefix());
        std::fs::create_dir_all(&sub).map_err(|e| fs_err(&e))?;
        // 1) .cubin：同目录 temp + rename
        let tmp_path = sub.join(format!(".{}.tmp.{}", key.hex(), std::process::id()));
        std::fs::write(&tmp_path, bytes).map_err(|e| fs_err(&e))?;
        std::fs::rename(&tmp_path, cubin_path(&sub, key)).map_err(|e| fs_err(&e))?;
        // 2) meta：提交点
        write_meta(&sub, key, meta)
    }

    /// 删除产物与 meta（持锁；build_once 的"清理-重建"路径）。
    pub fn remove(&self, key: &JitKey, _guard: &JitLockGuard) -> Result<(), LaunchError> {
        let sub = self.dir.join(key.dir_prefix());
        let _ = std::fs::remove_file(cubin_path(&sub, key));
        let _ = std::fs::remove_file(meta_path(&sub, key));
        Ok(())
    }

    /// 获取或构建产物：无锁快速路径 → 持锁 + 双检 → 清理损坏残留 →
    /// `compile` 一次 → 原子提交。编译失败或提交失败一律上抛（不循环重试；
    /// "重建一次"= 本调用内至多编译一次）。
    pub fn build_once(
        &self,
        key: &JitKey,
        src: &KernelSource,
        compile: impl FnOnce() -> Result<Vec<u8>, LaunchError>,
    ) -> Result<(JLibMeta, PathBuf), LaunchError> {
        if let Some(hit) = self.try_load(key)? {
            return Ok(hit);
        }
        let guard = lock::lock(self.dir(), key)?;
        if let Some(hit) = self.try_load(key)? {
            return Ok(hit); // 双检：他人已构建
        }
        // 清理上次失败/损坏残留（失败不阻断编译——store 将覆盖）
        let _ = self.remove(key, &guard);
        let bytes = compile()?; // 失败 → 携带编译错误上抛（B2 附 nvcc stderr 尾）
        let meta = JLibMeta {
            key: *key,
            arch: src.arch.clone(),
            toolchain_ver: src.toolchain_ver.clone(),
            sha256: hex_sha256(&bytes),
            size: bytes.len() as u64,
            gencode: extract_gencode(&src.flags),
            created_at: now_secs(),
        };
        self.store(key, &guard, &bytes, &meta)?;
        // 同一锁内复核（防御；正常路径必然命中）
        self.try_load(key)?.ok_or(LaunchError::Fatal)
    }
}

/// 默认缓存根（XDG）。
fn default_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("reinfer/jit");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".cache/reinfer/jit")
}

/// 内容 sha256（hex）。
pub(crate) fn hex_sha256(data: &[u8]) -> String {
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

/// 清扫过期残留 temp（open 时；递归一层分片目录）。
fn clean_stale_temps(dir: &Path) -> Result<(), LaunchError> {
    let cutoff = SystemTime::now()
        .checked_sub(std::time::Duration::from_secs(STALE_TEMP_SECS))
        .unwrap_or(UNIX_EPOCH);
    let mut dirs: Vec<PathBuf> = vec![dir.to_path_buf()];
    if let Ok(entries) = std::fs::read_dir(dir) {
        for e in entries.flatten() {
            if e.path().is_dir() {
                dirs.push(e.path());
            }
        }
    }
    for d in dirs {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if !name.starts_with('.') || !name.contains(".tmp.") {
                continue;
            }
            let stale = e.metadata().and_then(|m| m.modified()).map(|t| t < cutoff).unwrap_or(true);
            if stale {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    Ok(())
}

/// 时间戳（秒；仅诊断）。
pub(crate) fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// 从 flags 提取 gencode 段（meta 记录，诊断/跨设备核对）。
fn extract_gencode(flags: &[String]) -> Vec<String> {
    flags.iter().filter(|f| f.starts_with("-gencode") || f.contains("arch=")).cloned().collect()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败；仓库惯例 expect/是否弃用
    use super::*;
    use crate::key::JitKey;

    fn cache(tag: &str) -> JitCache {
        let d =
            std::env::temp_dir().join(format!("reinfer-jit-cache-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        JitCache::open(Some(d)).unwrap()
    }

    fn key(n: u8) -> JitKey {
        JitKey::from_bytes([n; 32])
    }

    fn guard(c: &JitCache, k: &JitKey) -> JitLockGuard {
        lock::lock(c.dir(), k).unwrap()
    }

    fn meta(k: &JitKey, bytes: &[u8]) -> JLibMeta {
        JLibMeta {
            key: *k,
            arch: "sm_120a".into(),
            toolchain_ver: "release 12.8".into(),
            sha256: hex_sha256(bytes),
            size: bytes.len() as u64,
            gencode: vec![],
            created_at: now_secs(),
        }
    }

    #[test]
    fn store_then_load_hits() {
        let c = cache("a");
        let k = key(1);
        let bytes = b"cubin-bytes";
        c.store(&k, &guard(&c, &k), bytes, &meta(&k, bytes)).unwrap();
        let (m, p) = c.try_load(&k).unwrap().expect("hit");
        assert_eq!(m.size, bytes.len() as u64);
        assert_eq!(std::fs::read(p).unwrap(), bytes);
    }

    #[test]
    fn meta_without_cubin_is_miss() {
        let c = cache("b");
        let k = key(2);
        let sub = c.dir().join(k.dir_prefix());
        std::fs::create_dir_all(&sub).unwrap();
        let b = b"x";
        write_meta(&sub, &k, &meta(&k, b)).unwrap();
        assert!(c.try_load(&k).unwrap().is_none());
    }

    #[test]
    fn corrupt_cubin_is_miss() {
        let c = cache("c");
        let k = key(3);
        let bytes = b"good";
        c.store(&k, &guard(&c, &k), bytes, &meta(&k, bytes)).unwrap();
        let sub = c.dir().join(k.dir_prefix());
        std::fs::write(cubin_path(&sub, &k), b"corrupt-byte").unwrap();
        assert!(c.try_load(&k).unwrap().is_none());
    }

    #[test]
    fn stale_temp_cleaned_on_open() {
        let d =
            std::env::temp_dir().join(format!("reinfer-jit-cache-stale-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("aa")).unwrap();
        let p = d.join("aa/..hidden.tmp.999");
        std::fs::write(&p, b"junk").unwrap();
        // 回拨 mtime 至 2 小时前（trigger stale 分支）
        filetime::set_file_mtime(
            &p,
            filetime::FileTime::from_unix_time(now_secs() as i64 - 7200, 0),
        )
        .unwrap();
        let c = JitCache::open(Some(d.clone())).unwrap();
        assert!(!c.dir().join("aa/..hidden.tmp.999").exists());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn store_rejects_hash_mismatch() {
        let c = cache("e");
        let k = key(4);
        let bytes = b"abc";
        let mut m = meta(&k, bytes);
        m.size = 999;
        assert!(matches!(c.store(&k, &guard(&c, &k), bytes, &m), Err(LaunchError::Fatal)));
    }

    fn kern_source() -> crate::KernelSource {
        crate::KernelSource {
            name: "k",
            src: "kernel-src",
            headers: vec![],
            flags: vec!["-gencode arch=compute_120,code=sm_120a".into()],
            arch: "sm_120a".into(),
            toolchain_ver: "release 12.8".into(),
        }
    }

    fn tc() -> crate::ToolchainId {
        crate::ToolchainId {
            ver_line: "release 12.8".into(),
            realpath: "/usr/local/cuda-12.8".into(),
            ccbin: ("/usr/bin/g++".into(), "g++ 12".into()),
        }
    }

    #[test]
    fn build_once_miss_compiles_then_hits() {
        let c = cache("f");
        let k = JitKey::new(&kern_source(), &tc());
        let calls = std::cell::Cell::new(0);
        let compiled = b"cubin-v1";
        let (_, _) = c
            .build_once(&k, &kern_source(), || {
                calls.set(calls.get() + 1);
                Ok(compiled.to_vec())
            })
            .unwrap();
        assert_eq!(calls.get(), 1);
        // 再要 → 命中，不编译
        let (_, _) = c
            .build_once(&k, &kern_source(), || {
                calls.set(calls.get() + 1);
                Ok(vec![0])
            })
            .unwrap();
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn build_once_cleans_corrupt_and_rebuilds_once() {
        let c = cache("g");
        let k = JitKey::new(&kern_source(), &tc());
        // 预置：meta 有效 + .cubin 坏字节（try_load = miss 状态）
        let sub = c.dir().join(k.dir_prefix());
        std::fs::create_dir_all(&sub).unwrap();
        write_meta(&sub, &k, &meta(&k, b"ok")).unwrap();
        std::fs::write(cubin_path(&sub, &k), b"corrupt").unwrap();

        let calls = std::cell::Cell::new(0);
        let (_, _) = c
            .build_once(&k, &kern_source(), || {
                calls.set(calls.get() + 1);
                Ok(b"cubin-v2".to_vec())
            })
            .unwrap();
        assert_eq!(calls.get(), 1, "损坏态只编译一次");
        assert!(c.try_load(&k).unwrap().is_some());
    }

    #[test]
    fn build_once_compile_error_propagates_no_residue() {
        let c = cache("h");
        let k = JitKey::new(&kern_source(), &tc());
        let err = c.build_once(&k, &kern_source(), || Err(LaunchError::Fatal));
        assert!(matches!(err, Err(LaunchError::Fatal)));
        assert!(c.try_load(&k).unwrap().is_none());
    }

    #[test]
    fn build_once_concurrent_threads_compile_once() {
        use std::sync::Arc;
        let c = Arc::new(cache("i"));
        let k = JitKey::new(&kern_source(), &tc());
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let c = c.clone();
                let calls = calls.clone();
                std::thread::spawn(move || {
                    let _ = c
                        .build_once(&k, &kern_source(), || {
                            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                            Ok(b"cubin-v3".to_vec())
                        })
                        .unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}

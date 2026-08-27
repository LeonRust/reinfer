//! 按 key 的跨进程文件锁（flock；012 r1：NB 轮询 + 可配超时）。
//!
//! 锁目录默认 `<cache>/locks/`（与缓存同挂载、同命名空间——容器/PrivateTmp
//! 下仍互斥），`REINFER_JIT_LOCK_DIR` 可覆盖。flock 在进程死亡时由内核
//! 自动释放，无陈旧锁；持锁者挂死不释放 → 由超时逃生（NB + 轮询）。

use crate::error::fs_err;
use fs2::FileExt;
use reinfer_kernels::LaunchError;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// 默认等待上限（nvcc 编译同量级）。
pub const LOCK_TIMEOUT_DEFAULT_SECS: u64 = 300;

/// 锁超时环境变量。
pub const LOCK_TIMEOUT_ENV: &str = "REINFER_JIT_LOCK_TIMEOUT";
/// 锁目录环境变量。
pub const LOCK_DIR_ENV: &str = "REINFER_JIT_LOCK_DIR";

/// 持锁句柄（Drop 释放；flock 非递归——同进程双锁需分开 key）。
#[derive(Debug)]
pub struct JitLockGuard {
    file: File,
}

impl Drop for JitLockGuard {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}

/// 等待上限（env 可覆，默认 300s；非法值 → 默认）。
fn timeout() -> Duration {
    let secs = std::env::var(LOCK_TIMEOUT_ENV)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(LOCK_TIMEOUT_DEFAULT_SECS);
    Duration::from_secs(secs)
}

/// 锁目录：env 覆盖 > `<cache>/locks`。
fn lock_dir(cache_dir: &Path) -> PathBuf {
    if let Ok(p) = std::env::var(LOCK_DIR_ENV) {
        return PathBuf::from(p);
    }
    cache_dir.join("locks")
}

fn lock_path(cache_dir: &Path, key: &crate::JitKey) -> Result<PathBuf, LaunchError> {
    let dir = lock_dir(cache_dir);
    std::fs::create_dir_all(&dir).map_err(|e| fs_err(&e))?;
    Ok(dir.join(format!("{}.lock", key.hex())))
}

/// 获取 key 的排他锁：NB 轮询直至超时。
pub(crate) fn lock(cache_dir: &Path, key: &crate::JitKey) -> Result<JitLockGuard, LaunchError> {
    lock_with(cache_dir, key, timeout())
}

/// 超时注入版（测试/特殊路径；`timeout` 显式传入）。
pub(crate) fn lock_with(
    cache_dir: &Path,
    key: &crate::JitKey,
    timeout: Duration,
) -> Result<JitLockGuard, LaunchError> {
    let path = lock_path(cache_dir, key)?;
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .open(&path)
        .map_err(|e| fs_err(&e))?;
    let deadline = Instant::now() + timeout;
    loop {
        match file.try_lock_exclusive() {
            Ok(()) => return Ok(JitLockGuard { file }),
            Err(_) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(25));
            }
            Err(_) => {
                return Err(LaunchError::Fatal); // 超时：含 key 的错误（Fatal；L3 前 fail-closed）
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("reinfer-jit-lock-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn exclusive_lock_blocks_second_holder() {
        let dir = cache_dir("a");
        let key = crate::JitKey::from_bytes([1; 32]);
        let g1 = lock(&dir, &key).unwrap();
        // 短超时（注入）：第二次获取必须失败（Fatal），证明互斥生效
        let r = lock_with(&dir, &key, Duration::from_millis(80));
        assert!(matches!(r, Err(LaunchError::Fatal)));
        drop(g1);
        // 释放后可再次获取
        assert!(lock(&dir, &key).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn distinct_keys_do_not_block() {
        let dir = cache_dir("b");
        let _g1 = lock(&dir, &crate::JitKey::from_bytes([1; 32])).unwrap();
        assert!(lock(&dir, &crate::JitKey::from_bytes([2; 32])).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }
}

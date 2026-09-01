//! TuneDb：实测性能数据库（006 T2；原子写 + 写锁 + 损坏容错）。
//!
//! 持久化文件 `tune.json` 默认落在**用户缓存、仓库之外**（与 git 分离，
//! 本机调优数据不随仓库分发）：
//! `REINFER_TUNE_DB` 覆盖 > `$XDG_CACHE_HOME/reinfer/tune.json` >
//! `$HOME/.cache/reinfer/tune.json` > 系统临时目录兜底。
//!
//! 可靠性：
//! - **原子写**：同目录 tmp（pid+序号命名）→ `sync_all` → `rename`（同文件系统
//!   原子替换；失败路径清理 tmp）；
//! - **写锁**：`<path>.lock` flock 排他锁（fs2，同 `crates/jit` 惯例；进程死亡
//!   由内核自动释放，无陈旧锁）；
//! - **损坏容错**：JSON 解析失败/不可读 → 空库（`was_corrupt()` 可诊断），下次
//!   `save()` 整体覆盖重建；选择器对未知 provider 名静默忽略（foreign 记录不
//!   污染选择，见 `crate::provider::select_chain`）。
//!
//! 记录 = `(op, shape_key, provider, score)`；`score` = 实测耗时 µs，**越低越好**
//! （ms/tok/s 由记录方在写入前归一化为 µs；与 012 骨架 `TuneEntry.us` 同约定）。

use crate::provider::OpConfig;
use core::fmt;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// tune.json 路径环境变量覆盖。
pub const TUNE_DB_ENV: &str = "REINFER_TUNE_DB";
/// vendor 档位名（tune.json 内稳定串）。
pub const PROVIDER_VENDOR: &str = "vendor";
/// jit fmha 档位名。
pub const PROVIDER_JIT_FMHA: &str = "jit_fmha";
/// jit dense/003 档位名。
pub const PROVIDER_JIT_DENSE: &str = "jit_dense";

/// 磁盘格式版本（本版写 1；读取不 gate——结构由 entries 自描述，容忍未来扩展）。
const SCHEMA: u32 = 1;

/// 单条调优记录（持久化单位）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TuneRecord {
    /// 算子名（`fmha`/`attn`/...；与选择器调优命名空间一致）。
    pub op: String,
    /// 参数化形状坐标（`s{seq}_b{batch}_h{head_dim}_{in}->{out}`，见 `shape_key`）。
    pub shape_key: String,
    /// 档位名（`PROVIDER_VENDOR`/`PROVIDER_JIT_FMHA`/`PROVIDER_JIT_DENSE`）。
    pub provider: String,
    /// 实测 score（µs，越低越好）。
    pub score: f64,
}

/// tune.json 磁盘格式（宽松解析：未知字段/未知条目忽略）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TuneFile {
    /// 格式版本。
    schema: u32,
    /// 记录列表（保存时按 (op, shape_key, provider) 排序，输出确定）。
    entries: Vec<TuneRecord>,
}

/// 记录复合键（BTreeMap 有序 → 保存输出确定性）。
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct TuneKey {
    op: String,
    shape_key: String,
    provider: String,
}

/// tune.json 读写错误（io + 序列化统一）。
///
/// 反序列化失败**不是**错误：损坏容错语义为空库（见 `TuneDb::open_at`）。
#[derive(Debug)]
pub enum TuneError {
    /// 文件系统操作失败。
    Io(io::Error),
    /// JSON 序列化失败。
    Json(serde_json::Error),
}

impl fmt::Display for TuneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TuneError::Io(e) => write!(f, "tune.json io error: {e}"),
            TuneError::Json(e) => write!(f, "tune.json serialize error: {e}"),
        }
    }
}

impl std::error::Error for TuneError {}

impl From<io::Error> for TuneError {
    fn from(e: io::Error) -> Self {
        TuneError::Io(e)
    }
}

impl From<serde_json::Error> for TuneError {
    fn from(e: serde_json::Error) -> Self {
        TuneError::Json(e)
    }
}

/// tmp 文件序号（同进程多次保存不碰撞；跨进程靠 pid 隔离）。
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// 写锁句柄（flock 排他；Drop 释放；进程死亡由内核自动释放）。
///
/// `save()` 内部自行持锁；本句柄供"读-改-写"多步临界区显式持锁使用。
#[derive(Debug)]
pub struct TuneLock {
    file: File,
}

impl TuneLock {
    /// 获取排他写锁（阻塞直至可得）。
    pub fn acquire(path: &Path) -> io::Result<TuneLock> {
        ensure_parent(path)?;
        let file =
            OpenOptions::new().create(true).truncate(false).write(true).open(lock_path(path))?;
        file.lock_exclusive()?;
        Ok(TuneLock { file })
    }

    /// 非阻塞获取（不可得 → `Ok(None)`；测试/诊断用）。
    pub fn try_acquire(path: &Path) -> io::Result<Option<TuneLock>> {
        ensure_parent(path)?;
        let file =
            OpenOptions::new().create(true).truncate(false).write(true).open(lock_path(path))?;
        match file.try_lock_exclusive() {
            Ok(()) => Ok(Some(TuneLock { file })),
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
            Err(e) => Err(e),
        }
    }
}

impl Drop for TuneLock {
    fn drop(&mut self) {
        let _ = fs2::FileExt::unlock(&self.file);
    }
}

/// TuneDb：进程内记录 + tune.json 持久化（safe 层；006 T2）。
#[derive(Debug)]
pub struct TuneDb {
    path: PathBuf,
    entries: BTreeMap<TuneKey, f64>,
    /// 最近一次加载是否遭遇损坏（诊断；save 覆盖后自然消失）。
    corrupt: bool,
}

impl TuneDb {
    /// 默认位置打开（env/XDG/HOME 解析；文件缺失或损坏 → 空库，不报错）。
    pub fn open() -> Self {
        Self::open_at(default_path())
    }

    /// 指定路径打开（测试注入/显式覆盖）。
    pub fn open_at(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref().to_path_buf();
        let (entries, corrupt) = load_from(&path);
        Self { path, entries, corrupt }
    }

    /// 目标文件路径（诊断）。
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// 本次打开是否遇到损坏/不可读的 tune.json（下次 save 覆盖恢复）。
    pub fn was_corrupt(&self) -> bool {
        self.corrupt
    }

    /// 记录一条实测（同 (op, shape_key, provider) 覆盖旧值）；返回被覆盖的旧
    /// score（无则 `None`）。`score` 必须有限（µs，越低越好）。
    pub fn record(&mut self, op: &str, shape_key: &str, provider: &str, score: f64) -> Option<f64> {
        debug_assert!(score.is_finite(), "TuneDb::record: score 必须有限");
        let key = TuneKey {
            op: op.to_string(),
            shape_key: shape_key.to_string(),
            provider: provider.to_string(),
        };
        self.entries.insert(key, score)
    }

    /// (op, shape_key) 下 score 最优（最低）的记录；无 → `None`。
    pub fn best(&self, op: &str, shape_key: &str) -> Option<TuneRecord> {
        self.best_in(op, shape_key, &[])
    }

    /// 限定 provider 集合下 score 最优的记录；`providers` 为空 = 不限定。
    /// 未知 provider 名天然不参与（调用方按已知档位名过滤）。
    pub fn best_in(&self, op: &str, shape_key: &str, providers: &[&str]) -> Option<TuneRecord> {
        self.entries
            .iter()
            .filter(|(k, _)| k.op == op && k.shape_key == shape_key)
            .filter(|(k, _)| providers.is_empty() || providers.contains(&k.provider.as_str()))
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(k, s)| TuneRecord {
                op: k.op.clone(),
                shape_key: k.shape_key.clone(),
                provider: k.provider.clone(),
                score: *s,
            })
    }

    /// 记录条数（诊断/测试）。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否无记录。
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 原子保存：flock 写锁 → 同目录 tmp → `sync_all` → `rename`（目录 fsync 尽力）。
    pub fn save(&self) -> Result<(), TuneError> {
        let dir = parent_dir(&self.path);
        fs::create_dir_all(dir)?;
        let _lock = TuneLock::acquire(&self.path)?;
        let file = TuneFile {
            schema: SCHEMA,
            entries: self
                .entries
                .iter()
                .map(|(k, s)| TuneRecord {
                    op: k.op.clone(),
                    shape_key: k.shape_key.clone(),
                    provider: k.provider.clone(),
                    score: *s,
                })
                .collect(),
        };
        let json = serde_json::to_vec_pretty(&file)?;
        let tmp = dir.join(format!(
            ".tune.{}.{}.tmp",
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let res = (|| -> io::Result<()> {
            let mut f = File::create(&tmp)?;
            f.write_all(&json)?;
            f.sync_all()?;
            fs::rename(&tmp, &self.path)
        })();
        if res.is_err() {
            let _ = fs::remove_file(&tmp); // 失败路径不留 tmp
        }
        res?;
        // 目录 fsync（尽力）：让 rename 本身落盘
        if let Ok(d) = File::open(dir) {
            let _ = d.sync_all();
        }
        Ok(())
    }
}

/// 加载：缺失 → 空；解析失败/不可读 → 空 + corrupt 标记（损坏容错）。
fn load_from(path: &Path) -> (BTreeMap<TuneKey, f64>, bool) {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return (BTreeMap::new(), false),
        Err(_) => return (BTreeMap::new(), true),
    };
    match serde_json::from_slice::<TuneFile>(&bytes) {
        Ok(file) => (index(file.entries), false),
        Err(_) => (BTreeMap::new(), true),
    }
}

/// 记录列表 → 索引（文件内重复行取首条）。
fn index(entries: Vec<TuneRecord>) -> BTreeMap<TuneKey, f64> {
    let mut m = BTreeMap::new();
    for r in entries {
        m.entry(TuneKey { op: r.op, shape_key: r.shape_key, provider: r.provider })
            .or_insert(r.score);
    }
    m
}

/// 参数化形状坐标（确定性：同 cfg 必同 key）。
///
/// 编码：`s{seq}_b{batch}_h{head_dim}_{in_dt}->{out_dt}`（seq/batch/head_dim/dtype 全量参与）。
pub fn shape_key(cfg: &OpConfig) -> String {
    format!(
        "s{}_b{}_h{}_{}->{}",
        cfg.seq,
        cfg.batch,
        cfg.head_dim,
        cfg.in_dt.name(),
        cfg.out_dt.name()
    )
}

/// 默认 tune.json 路径：env 覆盖 > XDG > HOME/.cache > 临时目录兜底。
fn default_path() -> PathBuf {
    resolve_path(
        std::env::var(TUNE_DB_ENV).ok(),
        std::env::var_os("XDG_CACHE_HOME").map(PathBuf::from),
        std::env::var_os("HOME").map(PathBuf::from),
    )
}

/// 路径解析（纯函数；env 读取在 `default_path`——测试注入无需进程全局 env）。
fn resolve_path(tune_env: Option<String>, xdg: Option<PathBuf>, home: Option<PathBuf>) -> PathBuf {
    if let Some(p) = tune_env.filter(|p| !p.is_empty()) {
        return PathBuf::from(p);
    }
    let base = xdg
        .or_else(|| home.map(|h| h.join(".cache")))
        .unwrap_or_else(|| std::env::temp_dir().join("reinfer-cache"));
    base.join("reinfer").join("tune.json")
}

/// 锁文件路径（`<path>.lock`，与目标同目录同挂载）。
fn lock_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".lock");
    PathBuf::from(s)
}

/// 父目录（相对路径如 `tune.json` 的 parent 为空串 → 当前目录）。
fn parent_dir(path: &Path) -> &Path {
    path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."))
}

/// 确保目标父目录存在（相对路径无父目录时为空操作）。
fn ensure_parent(path: &Path) -> io::Result<()> {
    fs::create_dir_all(parent_dir(path))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;
    use crate::provider::OpConfig;
    use reinfer_core::{DType, DeviceId};

    fn tmp_dir(tag: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("reinfer-kernels-tune-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn cfg() -> OpConfig {
        OpConfig {
            op: "fmha",
            device: DeviceId::new(0),
            in_dt: DType::F16,
            out_dt: DType::F16,
            head_dim: 128,
            batch: 1,
            seq: 2048,
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tmp_dir("rt");
        let path = dir.join("tune.json");
        let mut db = TuneDb::open_at(&path);
        assert!(db.is_empty());
        assert_eq!(db.record("fmha", "s2048_b1_h128_f16->f16", PROVIDER_JIT_FMHA, 123.5), None);
        db.record("attn", "s1_b8_h128_f16->f16", PROVIDER_VENDOR, 9.25);
        db.save().unwrap();

        let db2 = TuneDb::open_at(&path);
        assert!(!db2.was_corrupt());
        assert_eq!(db2.len(), 2);
        let b = db2.best("fmha", "s2048_b1_h128_f16->f16").unwrap();
        assert_eq!(b.provider, PROVIDER_JIT_FMHA);
        assert_eq!(b.score, 123.5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn atomic_write_leaves_valid_file_and_cleans_tmp() {
        let dir = tmp_dir("atomic");
        let path = dir.join("tune.json");
        let mut db = TuneDb::open_at(&path);
        db.record("fmha", "k", PROVIDER_JIT_DENSE, 1.0);
        db.save().unwrap();
        // 目标即完整 JSON（可再解析），同目录无 tmp 残留
        let parsed: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(parsed["entries"][0]["provider"], "jit_dense");
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "tmp 文件必须随 rename 消失");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn corrupt_json_recovers_by_overwrite() {
        let dir = tmp_dir("corrupt");
        let path = dir.join("tune.json");
        fs::write(&path, b"{ not valid json !!! ").unwrap();
        let mut db = TuneDb::open_at(&path);
        assert!(db.was_corrupt());
        assert!(db.is_empty());
        // 重 bench 覆盖：新记录直接写回（原子替换损坏文件）
        db.record("fmha", "k", PROVIDER_JIT_FMHA, 42.0);
        db.save().unwrap();
        let db2 = TuneDb::open_at(&path);
        assert!(!db2.was_corrupt());
        assert_eq!(db2.best("fmha", "k").unwrap().score, 42.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_file_is_empty_and_not_corrupt() {
        let db = TuneDb::open_at(tmp_dir("missing").join("tune.json"));
        assert!(db.is_empty());
        assert!(!db.was_corrupt());
        let _ = std::fs::remove_dir_all(tmp_dir("missing"));
    }

    #[test]
    fn record_overwrites_same_key_and_reports_old_score() {
        let mut db = TuneDb::open_at(tmp_dir("ow").join("tune.json"));
        assert_eq!(db.record("fmha", "k", PROVIDER_VENDOR, 10.0), None);
        let old = db.record("fmha", "k", PROVIDER_VENDOR, 5.0);
        assert_eq!(old, Some(10.0));
        assert_eq!(db.len(), 1);
        assert_eq!(db.best("fmha", "k").unwrap().score, 5.0);
        let _ = std::fs::remove_dir_all(tmp_dir("ow"));
    }

    #[test]
    fn best_in_filters_providers() {
        let mut db = TuneDb::open_at(tmp_dir("bi").join("tune.json"));
        db.record("fmha", "k", PROVIDER_VENDOR, 100.0);
        db.record("fmha", "k", PROVIDER_JIT_FMHA, 200.0);
        db.record("fmha", "k", PROVIDER_JIT_DENSE, 50.0);
        // 不限定 → 全局最优 dense(50)
        assert_eq!(db.best("fmha", "k").unwrap().provider, PROVIDER_JIT_DENSE);
        // 限定 vendor/fmha → vendor(100)
        let b = db.best_in("fmha", "k", &[PROVIDER_VENDOR, PROVIDER_JIT_FMHA]).unwrap();
        assert_eq!(b.provider, PROVIDER_VENDOR);
        // 未知 provider 名（foreign 记录）不参与已知档位的最优
        db.record("fmha", "k", "vendor_evil", 0.001);
        let known = [PROVIDER_VENDOR, PROVIDER_JIT_FMHA, PROVIDER_JIT_DENSE];
        assert_eq!(db.best_in("fmha", "k", &known).unwrap().provider, PROVIDER_JIT_DENSE);
        let _ = std::fs::remove_dir_all(tmp_dir("bi"));
    }

    #[test]
    fn shape_key_parameterizes_coordinates() {
        let a = cfg();
        let mut b = a;
        b.seq = 4096;
        let mut c = a;
        c.batch = 8;
        let mut d = a;
        d.head_dim = 64;
        let mut e = a;
        e.in_dt = DType::F32;
        let sk = shape_key(&a);
        assert_eq!(sk, "s2048_b1_h128_f16->f16");
        assert_ne!(sk, shape_key(&b));
        assert_ne!(sk, shape_key(&c));
        assert_ne!(sk, shape_key(&d));
        assert_ne!(sk, shape_key(&e));
        assert_eq!(shape_key(&a), sk, "同 cfg 必须同 key");
    }

    #[test]
    fn write_lock_excludes_second_holder() {
        let dir = tmp_dir("lock");
        let path = dir.join("tune.json");
        let g1 = TuneLock::acquire(&path).unwrap();
        assert!(TuneLock::try_acquire(&path).unwrap().is_none(), "第二持锁者必须被排除");
        drop(g1);
        assert!(TuneLock::try_acquire(&path).unwrap().is_some(), "释放后立即可再获锁");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn path_resolution_precedence() {
        // env 覆盖（空串视为未设）
        assert_eq!(
            resolve_path(Some("/x/tune.json".into()), None, None),
            PathBuf::from("/x/tune.json")
        );
        assert_eq!(
            resolve_path(Some(String::new()), Some("/x".into()), None),
            PathBuf::from("/x/reinfer/tune.json")
        );
        // XDG > HOME/.cache
        assert_eq!(
            resolve_path(None, Some("/x".into()), Some("/h".into())),
            PathBuf::from("/x/reinfer/tune.json")
        );
        assert_eq!(
            resolve_path(None, None, Some("/h".into())),
            PathBuf::from("/h/.cache/reinfer/tune.json")
        );
        // 全缺 → 临时目录兜底
        assert!(resolve_path(None, None, None).ends_with("reinfer/tune.json"));
    }
}

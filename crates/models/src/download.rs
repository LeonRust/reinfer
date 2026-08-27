//! 下载/校验/原子落盘/manifest（013 T2）。
//!
//! 纪律沿用 jit：temp+rename 原子写；sha256 校验不匹配 → 删除重试一次 → Fatal；
//! manifest 追加式（提交点=rename 后）。

use crate::api::{self, FileEntry};
use reinfer_kernels::LaunchError;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// manifest 文件名（下载目录内）。
pub const MANIFEST: &str = "manifest.json";

/// manifest 条目（追加式）。
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestEntry {
    /// 文件名。
    pub name: String,
    /// 字节大小。
    pub size: u64,
    /// 内容 sha256（HF 降级时可为 None）。
    pub sha256: Option<String>,
    /// 源仓库（owner/model）。
    pub repo: String,
    /// 分支/Revision。
    pub branch: String,
    /// 拉取时间（unix 秒）。
    pub fetched_at: u64,
}

/// 摘要字节 → hex（不重哈希）。
fn digest_hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// 内容 sha256（hex）。
pub fn sha256_hex(bytes: &[u8]) -> String {
    digest_hex(&Sha256::digest(bytes))
}

/// 校验强度（013 env 面）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verify {
    /// 官方 sha256（ModelScope）；HF 源自动降级 size（源侧标记）。
    Sha256,
    /// 仅大小。
    Size,
    /// 仅存在性（内网信任环境）。
    None,
}

impl Verify {
    /// 解析（大小写敏感；未知 → None）。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sha256" => Some(Verify::Sha256),
            "size" => Some(Verify::Size),
            "none" => Some(Verify::None),
            _ => None,
        }
    }
}

/// repo 内容目录 = 模型根/{owner}/{model}（二级目录；hf/modelscope 惯例——
/// 用户 2026-08-27 定：按 repo 组织，避免不同 repo 同名文件互相污染）。
pub fn repo_dir(root: &Path, repo: &str) -> PathBuf {
    root.join(repo)
}

/// 目标文件路径 = 模型根/{repo}/{name}。
pub fn target_path(root: &Path, repo: &str, name: &str) -> PathBuf {
    root.join(repo).join(name)
}

/// 本地命中判定：存在 + 按 verify 深度校验。
pub fn local_hit(path: &Path, entry: &FileEntry, verify: Verify) -> bool {
    let Ok(meta) = std::fs::metadata(path) else { return false };
    if meta.len() == 0 {
        return false;
    }
    if verify != Verify::None && entry.size != 0 && meta.len() != entry.size {
        return false;
    }
    if verify == Verify::Sha256 {
        match &entry.sha256 {
            Some(expected) => {
                let got = std::fs::read(path).map(|b| sha256_hex(&b)).unwrap_or_default();
                if got != *expected {
                    return false;
                }
            }
            // Sha256 深度但条目无 sha 字段 → 无法验证 → 不判命中（宁可重下）。
            None => return false,
        }
    }
    true
}

/// 读 manifest（缺/损坏 → 空清单；损坏不阻断——重写式追加）。
pub fn read_manifest(dir: &Path) -> Vec<ManifestEntry> {
    let p = dir.join(MANIFEST);
    std::fs::read(&p).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

/// 进程内 manifest 写锁（read→write 事务互斥；rename 仍是原子提交点）。
/// 多线程并发下载时防止 read-modify-write 丢失更新（OneShot 初始化，零成本无竞争路径）。
static MANIFEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn append_manifest(dir: &Path, entry: ManifestEntry) {
    // 毒锁恢复：panic 线程未持锁完成写 → 内容可能已写盘，read_manifest 容错兜底即可。
    let _guard =
        MANIFEST_LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner());
    let p = dir.join(MANIFEST);
    let mut all = read_manifest(dir);
    all.retain(|e| e.name != entry.name);
    all.push(entry);
    if let Ok(bytes) = serde_json::to_vec_pretty(&all) {
        let tmp = dir.join(format!(".{MANIFEST}.tmp-{}", std::process::id()));
        if std::fs::write(&tmp, bytes).is_ok() {
            let _ = std::fs::rename(tmp, p);
        }
    }
}

/// 下载一个文件（302 跟随、流式写 temp、sha/len 校验、失败重试一次、rename+manifest）。
///
/// 幂等：已完成且校验通过 → 返回路径（hit）；`verify` 深度见 [`Verify`]。
/// `revision`：None → master（`--revision` 同语义，manifest branch 记录实际值）。
/// `progress`：回调 (已读字节, 预期总字节)；本地命中时以 (len, len) 调用一次告知 hit；
/// 下载中在每块 read 后调用（`entry.size == 0` → 总字节传 0，表示未知）。
pub fn download_file(
    repo: &str,
    entry: &FileEntry,
    root: &Path,
    verify: Verify,
    revision: Option<&str>,
    progress: Option<&dyn Fn(u64, u64)>,
) -> Result<PathBuf, LaunchError> {
    let dir = repo_dir(root, repo);
    std::fs::create_dir_all(&dir).map_err(io_err)?;
    let path = target_path(root, repo, &entry.name);
    if local_hit(&path, entry, verify) {
        if let (Some(cb), Ok(meta)) = (progress, std::fs::metadata(&path)) {
            cb(meta.len(), meta.len());
        }
        return Ok(path);
    }
    let branch = revision.unwrap_or("master");
    let url = api::ms_download_url_rev(repo, &entry.name, revision);
    download_with_url(&url, entry, &dir, verify, repo, branch, None, progress)
}

/// GET 响应头（最终跳转后）ETag 规范化（strip 引号/W- 前缀不剥——W 前缀是强弱标记，需要恒等比较两端）。
pub(crate) fn normalize_etag(v: &str) -> &str {
    let s = v.trim();
    s.strip_prefix("W/").unwrap_or(s).trim_matches('"')
}

/// 共享下载入口（ModelScope 模板 URL 与 HF resolve URL 共用）。
///
/// `expected_etag`：指定时若最终响应头含 ETag/X-Linked-Etag 且不同 → 校验失败；
/// 响应头无 ETag 字段 → 降级（仅 size/sha 档），打警告（r2：ETag+size 为 HF 上限）。
/// `progress`：原样透传给 [`fetch_to_temp`]（重试时同一回调；语义见 [`download_file`]）。
#[allow(clippy::too_many_arguments)] // 冻结接口（bin 侧按此签名并行开发）；8 参为契约
pub(crate) fn download_with_url(
    url: &str,
    entry: &FileEntry,
    to_dir: &Path,
    verify: Verify,
    repo: &str,
    branch: &str,
    expected_etag: Option<&str>,
    progress: Option<&dyn Fn(u64, u64)>,
) -> Result<PathBuf, LaunchError> {
    let mut last_err: Option<LaunchError> = None;
    let nm = &entry.name;
    for attempt in 1..=2 {
        match fetch_to_temp(url, to_dir, &entry.name, verify, entry, expected_etag, progress) {
            Ok(p) => {
                append_manifest(
                    to_dir,
                    ManifestEntry {
                        name: entry.name.clone(),
                        size: entry.size,
                        sha256: entry.sha256.clone(),
                        repo: repo.to_string(),
                        branch: branch.to_string(),
                        fetched_at: now_secs(),
                    },
                );
                return Ok(p);
            }
            Err(e) if attempt == 1 => {
                last_err = Some(e);
                eprintln!("reinfer-models: attempt {attempt} failed for {nm} ({url}), retrying");
            }
            Err(e) => {
                last_err = Some(e);
                break;
            }
        }
    }
    eprintln!("reinfer-models: download failed for {nm}: {last_err:?}");
    Err(LaunchError::Fatal)
}

/// GET（跟随重定向）→ 流式写 `<to>/.<name>.tmp-<pid>` → 校验 → rename。
///
/// `progress`：每块 read 后调用 (累计已读, 预期总字节)；`entry.size == 0` → 总字节传 0。
fn fetch_to_temp(
    url: &str,
    to_dir: &Path,
    name: &str,
    verify: Verify,
    entry: &FileEntry,
    expected_etag: Option<&str>,
    progress: Option<&dyn Fn(u64, u64)>,
) -> Result<PathBuf, LaunchError> {
    let resp = ureq::get(url).call().map_err(|e| {
        eprintln!("reinfer-models: GET {url} failed: {e}");
        LaunchError::Fatal
    })?;
    if let Some(expect) = expected_etag {
        let got = resp.header("X-Linked-Etag").or_else(|| resp.header("ETag")).map(normalize_etag);
        match got {
            Some(g) if g != normalize_etag(expect) => {
                eprintln!("reinfer-models: etag mismatch (got={g} expected={expect}) for {url}");
                return Err(LaunchError::Fatal);
            }
            Some(_) => {}
            None => eprintln!(
                "reinfer-models: no ETag in response for {url} - falling back to size/sha only"
            ),
        }
    }
    let tmp = to_dir.join(format!(".{name}.tmp-{}", std::process::id()));
    let mut file = std::fs::File::create(&tmp).map_err(io_err)?;
    let mut reader = resp.into_reader();
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    let mut total: u64 = 0;
    #[cfg(test)]
    let mut debug_capture: Vec<u8> = Vec::new();
    loop {
        let n = reader.read(&mut buf).map_err(io_err)?;
        if n == 0 {
            break;
        }
        #[cfg(test)]
        eprintln!("read n={n}");
        hasher.update(&buf[..n]);
        total += n as u64;
        #[cfg(test)]
        debug_capture.extend_from_slice(&buf[..n]);
        file.write_all(&buf[..n]).map_err(io_err)?;
        if let Some(cb) = progress {
            cb(total, if entry.size == 0 { 0 } else { entry.size });
        }
    }
    drop(file);
    if verify != Verify::None && entry.size != 0 && total != entry.size {
        let _ = std::fs::remove_file(&tmp);
        return Err(LaunchError::Fatal);
    }
    if verify == Verify::Sha256
        && let Some(expected) = &entry.sha256
    {
        let got = digest_hex(&hasher.finalize());
        if &got != expected {
            let _ = std::fs::remove_file(&tmp);
            #[cfg(test)]
            eprintln!(
                "reinfer-models: sha256 mismatch (got={got} expected={expected} total={total}\n  bytes={:?}\n  sha(debug_capture)={})",
                debug_capture,
                sha256_hex(&debug_capture)
            );
            return Err(LaunchError::Fatal);
        }
    }
    let path = to_dir.join(name); // 落地目录（=repo 目录）内 rename
    std::fs::rename(&tmp, &path).map_err(io_err)?;
    Ok(path)
}

pub(crate) fn io_err(e: std::io::Error) -> LaunchError {
    // ENOSPC → Oom（与 jit 的 fs_err 语义一致）
    if e.raw_os_error() == Some(28) { LaunchError::Oom } else { LaunchError::Fatal }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------- stub 测试（本地 TcpListener，无真实网络） ----------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;
    use std::net::TcpListener;

    /// stub 响应：状态码 + 体 + 附加响应头（如 ETag）。
    type StubResp = (u16, String, Vec<(String, String)>);

    /// 本地 stub：`handle` 按请求计数返回响应（支持 200/302/404、重试序列、自定义头）。
    struct Stub {
        port: u16,
    }

    impl Stub {
        fn spawn<F>(mut handle: F) -> (Self, std::thread::JoinHandle<()>)
        where
            F: FnMut(&str) -> StubResp + Send + 'static,
        {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = l.local_addr().unwrap().port();
            l.set_nonblocking(true).unwrap();
            let j = std::thread::spawn(move || {
                // 非阻塞 accept + 超时退出：既容纳多请求/重试，又保证 join 不悬死
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
                let mut served = 0;
                while served < 6 && std::time::Instant::now() < deadline {
                    match l.accept() {
                        Ok((mut s, _)) => {
                            let mut buf = [0u8; 4096];
                            let n = s.read(&mut buf).unwrap_or(0);
                            let req = String::from_utf8_lossy(&buf[..n]).to_string();
                            let (code, body, extra) = handle(&req);
                            // 手拼头部：extra 为空时不得产生多余空行（否则解析器把
                            // Connection 行当 body——Content-Length 定的 body 之前空行即结束 header）
                            let mut head = format!(
                                "HTTP/1.1 {code} {}\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\n",
                                if code == 200 { "OK" } else { "Found" },
                                body.len()
                            );
                            for (k, v) in &extra {
                                head.push_str(&format!("{k}: {v}\r\n"));
                            }
                            head.push_str("Connection: close\r\n\r\n");
                            let _ = s.write_all(head.as_bytes());
                            let _ = s.write_all(body.as_bytes());
                            served += 1;
                        }
                        Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
                    }
                }
            });
            (Self { port }, j)
        }
        fn url(&self, path: &str) -> String {
            format!("http://127.0.0.1:{}{}", self.port, path)
        }
    }

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("reinfer-models-dl-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn download_ok_and_manifest() {
        let body = b"hello-models";
        let (stub, j) = Stub::spawn(|_| (200, "hello-models".to_string(), vec![]));
        let dir = tmpdir("ok");
        let entry = FileEntry {
            name: "m.gguf".into(),
            size: body.len() as u64,
            sha256: Some(sha256_hex(body)),
            is_lfs: false,
        };
        // 共享入口（download_with_url）→ 下载+校验+rename+manifest 记录
        let got = download_with_url(
            &stub.url("/x"),
            &entry,
            &dir,
            Verify::Sha256,
            "org/repo",
            "master",
            None,
            None,
        )
        .unwrap();
        assert_eq!(std::fs::read(&got).unwrap(), body);
        let man = read_manifest(&dir);
        assert_eq!(man.len(), 1);
        assert_eq!(man[0].name, "m.gguf");
        assert_eq!(man[0].repo, "org/repo");
        assert_eq!(man[0].branch, "master");
        assert_eq!(man[0].size, body.len() as u64);
        let expected_hex = sha256_hex(body);
        assert_eq!(man[0].sha256.as_deref(), Some(expected_hex.as_str()));
        // 幂等命中由调用方 local_hit 负责（见 download_file）；此处只验证 下载+校验+rename+manifest。
        let _ = std::fs::remove_dir_all(&dir);
        j.join().unwrap();
    }

    #[test]
    fn sha_mismatch_removes_temp() {
        let (stub, j) = Stub::spawn(|_| (200, "wrong-bytes!".to_string(), vec![]));
        let dir = tmpdir("sha");
        let entry = FileEntry {
            name: "m.gguf".into(),
            size: 12,
            sha256: Some(sha256_hex(b"expected")),
            is_lfs: false,
        };
        assert!(
            fetch_to_temp(&stub.url("/x"), &dir, "m.gguf", Verify::Sha256, &entry, None, None)
                .is_err()
        );
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains(".tmp-"))
            .collect();
        assert!(leftovers.is_empty(), "temp must be cleaned: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
        j.join().unwrap();
    }

    /// ETag 坏值 → 校验失败（重试）；第二次与预期一致 → 成功（T7 验收）。
    #[test]
    fn etag_bad_then_ok_retry_once() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let n = Arc::new(AtomicUsize::new(0));
        let seq = Arc::clone(&n);
        let (stub, j) = Stub::spawn(move |_| {
            let i = seq.fetch_add(1, Ordering::SeqCst) + 1;
            let etag = if i == 1 { "bad" } else { "\"good\"" }; // W/ 引号都归一化
            (200, "data".to_string(), vec![("ETag".into(), etag.into())])
        });
        let dir = tmpdir("etag");
        let entry = FileEntry { name: "m.gguf".into(), size: 4, sha256: None, is_lfs: true };
        let got = download_with_url(
            &stub.url("/x"),
            &entry,
            &dir,
            Verify::Size,
            "org/repo",
            "main",
            Some("good"),
            None,
        )
        .unwrap();
        assert_eq!(std::fs::read(&got).unwrap(), b"data");
        let hits = n.load(Ordering::SeqCst);
        assert!(hits >= 2, "bad etag must have triggered a retry (n={hits})");
        let _ = std::fs::remove_dir_all(&dir);
        j.join().unwrap();
    }

    /// 响应无 ETag 头 → 降级为仅 size 校验（r2：ETag+size 为 HF 上限），不失败。
    #[test]
    fn etag_missing_degrades_to_size() {
        let (stub, j) = Stub::spawn(|_| (200, "data".to_string(), vec![]));
        let dir = tmpdir("etagdeg");
        let entry = FileEntry { name: "m.gguf".into(), size: 4, sha256: None, is_lfs: true };
        let got = download_with_url(
            &stub.url("/x"),
            &entry,
            &dir,
            Verify::Size,
            "org/repo",
            "main",
            Some("good"),
            None,
        )
        .unwrap();
        assert_eq!(std::fs::read(&got).unwrap(), b"data");
        let _ = std::fs::remove_dir_all(&dir);
        j.join().unwrap();
    }

    /// progress 回调：下载中每块 read 后调用 (累计, 预期)；size==0 → 总字节 0；hit → (len, len)。
    #[test]
    fn progress_download_unknown_size_and_hit() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Mutex};

        let body = b"hello-models";
        let (stub, j) = Stub::spawn(|_| (200, "hello-models".to_string(), vec![]));
        let dir = tmpdir("prog");

        // 已知大小：每块 read 后 (累计, 预期)，最后一块 = (len, len)
        let entry = FileEntry {
            name: "m.gguf".into(),
            size: body.len() as u64,
            sha256: Some(sha256_hex(body)),
            is_lfs: false,
        };
        let calls = Arc::new(Mutex::new(Vec::new()));
        let cb = {
            let calls = Arc::clone(&calls);
            move |read: u64, total: u64| calls.lock().unwrap().push((read, total))
        };
        let got = download_with_url(
            &stub.url("/x"),
            &entry,
            &dir,
            Verify::Sha256,
            "org/repo",
            "master",
            None,
            Some(&cb),
        )
        .unwrap();
        assert_eq!(std::fs::read(&got).unwrap(), body);
        let calls = calls.lock().unwrap();
        assert!(!calls.is_empty(), "progress must be reported");
        assert_eq!(*calls.last().unwrap(), (body.len() as u64, body.len() as u64));
        drop(calls);

        // 未知大小（size=0）：总字节恒传 0
        let unknown = Arc::new(AtomicUsize::new(0));
        let cb0 = {
            let unknown = Arc::clone(&unknown);
            move |_read: u64, total: u64| unknown.store(total as usize, Ordering::SeqCst)
        };
        let entry0 = FileEntry { name: "m0.gguf".into(), size: 0, sha256: None, is_lfs: true };
        let got0 = download_with_url(
            &stub.url("/x"),
            &entry0,
            &dir,
            Verify::Size,
            "org/repo",
            "master",
            None,
            Some(&cb0),
        )
        .unwrap();
        assert_eq!(std::fs::read(&got0).unwrap(), body);
        assert_eq!(unknown.load(Ordering::SeqCst), 0);

        // 本地命中：progress(len, len) 恰好一次
        let hit = Arc::new(Mutex::new(Vec::new()));
        let cbh = {
            let hit = Arc::clone(&hit);
            move |read: u64, total: u64| hit.lock().unwrap().push((read, total))
        };
        let entry1 = FileEntry {
            name: "m.gguf".into(),
            size: body.len() as u64,
            sha256: Some(sha256_hex(body)),
            is_lfs: false,
        };
        // 按 repo 组织：命中判定在 root/org/repo/ 下（先摆放文件）
        let hit_path = dir.join("org/repo/m.gguf");
        std::fs::create_dir_all(hit_path.parent().unwrap()).unwrap();
        std::fs::write(&hit_path, body).unwrap();
        let p = download_file("org/repo", &entry1, &dir, Verify::Sha256, None, Some(&cbh)).unwrap();
        assert_eq!(p, dir.join("org/repo/m.gguf")); // 按 repo 组织
        assert_eq!(*hit.lock().unwrap(), vec![(body.len() as u64, body.len() as u64)]);

        let _ = std::fs::remove_dir_all(&dir);
        j.join().unwrap();
    }

    #[test]
    fn size_only_verify_and_hit() {
        let dir = tmpdir("size");
        let path = dir.join("m.gguf");
        std::fs::write(&path, [0xAAu8; 100]).unwrap();
        let entry = FileEntry { name: "m.gguf".into(), size: 100, sha256: None, is_lfs: false };
        assert!(local_hit(&path, &entry, Verify::Size));
        assert!(local_hit(&path, &entry, Verify::None)); // none 档只查存在性
        assert!(!local_hit(&path, &entry, Verify::Sha256)); // 无 sha 字段 → 不判命中
        assert!(!local_hit(&dir.join("absent.gguf"), &entry, Verify::None));
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod sha_probe {
    #![allow(clippy::unwrap_used)]
    use super::sha256_hex;

    #[test]
    fn probe() {
        let a = sha256_hex(b"hello-models");
        let data: Vec<u8> = vec![104, 101, 108, 108, 111, 45, 109, 111, 100, 101, 108, 115];
        let b = sha256_hex(&data);
        eprintln!("sha(b\"hello-models\") = {a}");
        eprintln!("sha(vec bytes data)   = {b}");
        assert_eq!(a, b);
    }
}

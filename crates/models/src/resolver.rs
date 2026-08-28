//! 运行时 Resolver（013 r2）：本地命中 → 按策略下载；`AUTODOWNLOAD=off` 绝不联网。
//!
//! env 面（全部可缺省；规范见 specs/013-model-fetch/plan.md D6.1）：
//! `REINFER_MODEL_SOURCE`(modelscope|huggingface|auto) · `DIR` · `VERIFY`(sha256|size|none) ·
//! `AUTODOWNLOAD`(on|off) · `REINFER_MODEL_REPO/QUANT/FILE`（便捷注入——真名来自用户配置，
//! 代码零硬编码不变）。

use crate::api::{self, FileEntry};
use crate::download::{MANIFEST, ManifestEntry, Verify, download_file, local_hit, target_path};
use crate::hf::{hf_download_file, hf_list_files};
use reinfer_kernels::LaunchError;
use std::path::{Path, PathBuf};

/// **旧扁平布局自动迁移**（2026-08-28；旧版 `root/{file}` + 根 `manifest.json` → 新版
/// `root/{repo}/{file}` + per-repo manifest）。幂等：无旧根 manifest = 无操作。
/// 旧 manifest 备份为 `.manifest.json.bak-{pid}`（不删，人工可回退）。
pub fn migrate_legacy_flat(root: &Path) {
    let old = root.join(MANIFEST);
    let entries = crate::download::read_manifest(root);
    if entries.is_empty() || !old.exists() {
        return; // 已是新布局/无旧清单
    }
    let mut dst: std::collections::BTreeMap<String, Vec<ManifestEntry>> = Default::default();
    for e in &entries {
        let src = root.join(&e.name);
        if src.is_file() {
            let dir = root.join(&e.repo);
            if std::fs::create_dir_all(&dir).is_ok() {
                let dst_path = dir.join(&e.name);
                if !dst_path.exists() {
                    let _ = std::fs::rename(&src, &dst_path);
                } else {
                    let _ = std::fs::remove_file(&src); // 新旧同目标 → 弃旧（新已就位）
                }
            }
        }
        dst.entry(e.repo.clone()).or_default().push(e.clone());
    }
    // 拆写 per-repo manifest
    for (repo, items) in dst {
        let dir = root.join(&repo);
        let mut all = crate::download::read_manifest(&dir);
        all.retain(|e| !items.iter().any(|i| i.name == e.name));
        all.extend(items);
        if let Ok(bytes) = serde_json::to_vec_pretty(&all) {
            let tmp = dir.join(format!(".{MANIFEST}.tmp-{}", std::process::id()));
            if std::fs::write(&tmp, bytes).is_ok() {
                let _ = std::fs::rename(tmp, dir.join(MANIFEST));
            }
        }
    }
    // 旧清单备份（保留人工可回退）
    let _ = std::fs::rename(&old, root.join(format!(".{MANIFEST}.bak-{}", std::process::id())));
}

/// 用户主目录 env（平台惯例：Windows=USERPROFILE，Linux/macOS=HOME）。
/// 用户主目录 env（平台惯例：Windows=USERPROFILE，Linux/macOS=HOME）。
fn home_var() -> &'static str {
    if cfg!(windows) { "USERPROFILE" } else { "HOME" }
}

/// 缺省模型目录：`~/.reinfer/models`（Linux/macOS；Windows 为 `%USERPROFILE%\.reinfer\models`）。
pub fn default_dir() -> PathBuf {
    std::env::var(home_var())
        .ok()
        .map(|h| PathBuf::from(h).join(".reinfer").join("models"))
        .unwrap_or_else(|| PathBuf::from("models"))
}

/// `~` 前缀展开为家目录（env/CLI 值可写 `~/…`；家目录不可得 → 原样）。
fn expand_tilde(p: &str) -> PathBuf {
    if let (Some(rest), Ok(h)) = (p.strip_prefix("~/"), std::env::var(home_var())) {
        return PathBuf::from(h).join(rest);
    }
    PathBuf::from(p)
}

/// 主源策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelSource {
    /// 只打 ModelScope。
    Modelscope,
    /// 只打 HuggingFace。
    Huggingface,
    /// ModelScope 优先；缺（404/文件缺失）→ HuggingFace 回退。
    Auto,
}

impl ModelSource {
    /// 解析（大小写敏感；未知 → None）。
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "modelscope" => Some(ModelSource::Modelscope),
            "huggingface" => Some(ModelSource::Huggingface),
            "auto" => Some(ModelSource::Auto),
            _ => None,
        }
    }
}

/// 模型规格（全部显式；无默认模型常量/便捷函数——013 铁律）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    /// 源仓库（owner/model）。
    pub repo: String,
    /// 精确文件名（优先于 quant）。
    pub file: Option<String>,
    /// 量化段（如 `q8_0`；文件名后缀匹配）。
    pub quant: Option<String>,
    /// 分支/Revision；None=源缺省（ModelScope `master`；HuggingFace `main`——D6 r2）。
    pub branch: Option<String>,
}

impl ModelSpec {
    /// 显式构造——无默认 repo/模型常量（013 铁律）。
    pub fn new(repo: impl Into<String>) -> Self {
        Self { repo: repo.into(), file: None, quant: None, branch: None }
    }
    /// 指定精确文件名。
    pub fn with_file(mut self, f: impl Into<String>) -> Self {
        self.file = Some(f.into());
        self
    }
    /// 指定量化段。
    pub fn with_quant(mut self, q: impl Into<String>) -> Self {
        self.quant = Some(q.into());
        self
    }
    /// 指定分支。
    pub fn with_branch(mut self, b: impl Into<String>) -> Self {
        self.branch = Some(b.into());
        self
    }
}

/// 运行时模型解析器（env 策略面）。
#[derive(Debug, Clone)]
pub struct ModelResolver {
    /// 主源策略。
    pub source: ModelSource,
    /// 模型目录（REINFER_MODEL_DIR；缺省 `~/models/reinfer`）。
    pub dir: PathBuf,
    /// 校验强度。
    pub verify: Verify,
    /// 允许联网下载（AUTODOWNLOAD）。
    pub autodownload: bool,
}

impl ModelResolver {
    /// 从环境变量构造（全部 `REINFER_MODEL_*` + `HOME`；非法值 → Fatal 带变量名）。
    pub fn from_env() -> Result<Self, LaunchError> {
        let source = match std::env::var("REINFER_MODEL_SOURCE").ok() {
            None => ModelSource::Auto,
            Some(s) => ModelSource::parse(&s).ok_or_else(|| {
                eprintln!(
                    "reinfer-models: invalid REINFER_MODEL_SOURCE={s} (modelscope|huggingface|auto)"
                );
                LaunchError::Fatal
            })?,
        };
        let dir = std::env::var("REINFER_MODEL_DIR")
            .ok()
            .map(|v| expand_tilde(&v))
            .unwrap_or_else(default_dir);
        let verify = match std::env::var("REINFER_MODEL_VERIFY").ok() {
            None => Verify::Sha256,
            Some(v) => Verify::parse(&v).ok_or_else(|| {
                eprintln!("reinfer-models: invalid REINFER_MODEL_VERIFY={v} (sha256|size|none)");
                LaunchError::Fatal
            })?,
        };
        let autodownload = match std::env::var("REINFER_MODEL_AUTODOWNLOAD").as_deref() {
            Err(_) => true,
            Ok("on") | Ok("1") => true,
            Ok("off") | Ok("0") => false,
            Ok(other) => {
                eprintln!("reinfer-models: invalid REINFER_MODEL_AUTODOWNLOAD={other} (on|off)");
                return Err(LaunchError::Fatal);
            }
        };
        Ok(Self { source, dir, verify, autodownload })
    }

    /// env 便捷注入（REPO/QUANT/FILE；与显式参数冲突时显式优先——由调用方合并）。
    pub fn spec_from_env(explicit: Option<ModelSpec>) -> Result<ModelSpec, LaunchError> {
        if let Some(s) = explicit {
            return Ok(s);
        }
        let repo = std::env::var("REINFER_MODEL_REPO").map_err(|_| {
            eprintln!("reinfer-models: no repo (pass CLI arg or set REINFER_MODEL_REPO)");
            LaunchError::Fatal
        })?;
        let mut spec = ModelSpec::new(repo);
        if let Some(q) = std::env::var("REINFER_MODEL_QUANT").ok().filter(|s| !s.is_empty()) {
            spec = spec.with_quant(q);
        }
        if let Some(f) = std::env::var("REINFER_MODEL_FILE").ok().filter(|s| !s.is_empty()) {
            spec = spec.with_file(f);
        }
        Ok(spec)
    }

    /// 解析期望文件名（本地 glob，**绝不联网**）→ 否则远端列表（组合式；`off` 调用方保证不触网）。
    /// `quant` 段匹配：文件名含 `-{quant}.` 段（任意扩展——用户铁律，无 .gguf 特判）；
    /// 精确 file 直用；多命中 → 列候选。
    pub fn resolve_file_name(&self, spec: &ModelSpec, dir: &Path) -> Result<String, LaunchError> {
        if let Some(n) = self.resolve_local_name(spec, dir)? {
            return Ok(n);
        }
        self.resolve_remote_name(spec)
    }

    /// 本地 quant glob：文件名含 `-{q}.` 段即命中（任意扩展；无 .gguf 特判——用户铁律）。
    /// 按 repo 组织：glob 作用于 `root/{repo}`（2026-08-27 用户定）。
    fn resolve_local_name(
        &self,
        spec: &ModelSpec,
        dir: &Path,
    ) -> Result<Option<String>, LaunchError> {
        if let Some(f) = &spec.file {
            return Ok(Some(f.clone()));
        }
        let Some(q) = &spec.quant else { return Ok(None) };
        if let Ok(entries) = std::fs::read_dir(crate::download::repo_dir(dir, &spec.repo)) {
            let mut hits: Vec<String> = entries
                .flatten()
                .map(|e| e.file_name().to_string_lossy().to_string())
                .filter(|n| n.contains(&format!("-{q}.")))
                .collect();
            hits.sort();
            match hits.len() {
                0 => Ok(None),
                1 => Ok(Some(hits.remove(0))),
                _ => {
                    eprintln!("reinfer-models: --quant {q} ambiguous: {hits:?}");
                    Err(LaunchError::Fatal)
                }
            }
        } else {
            Ok(None)
        }
    }

    fn resolve_remote_name(&self, spec: &ModelSpec) -> Result<String, LaunchError> {
        // Auto 源：MS 列表缺失 → HF siblings 回退（与 fetch 回退对称——名字解析也回退）
        let entries = match self.source {
            ModelSource::Auto => ms_list(spec).or_else(|_| hf_list_entries(spec)),
            ModelSource::Modelscope => ms_list(spec),
            ModelSource::Huggingface => hf_list_entries(spec),
        }?;
        select_name(entries, spec)
    }

    /// 保证模型可用：本地命中 → 返回路径；否则下载（按 source 策略）；off → 明确错误。
    pub fn ensure(&self, spec: &ModelSpec) -> Result<PathBuf, LaunchError> {
        self.ensure_in(spec, &self.dir)
    }

    /// 同 [`ensure`]，指定目录（缺失时落下）——plan D6 公开面。
    pub fn ensure_to(&self, spec: &ModelSpec, dir: &Path) -> Result<PathBuf, LaunchError> {
        self.ensure_in(spec, dir)
    }

    fn ensure_in(&self, spec: &ModelSpec, dir: &Path) -> Result<PathBuf, LaunchError> {
        // 旧扁平布局自动迁移（升级前下载过的机器首跑即归位，无需重下）
        migrate_legacy_flat(dir);
        // off 语义：只本地 glob（绝不联网）；unresolvable → 明确错误
        let name = match self.resolve_local_name(spec, dir)? {
            Some(n) => n,
            None if self.autodownload => self.resolve_remote_name(spec)?,
            None => return Err(offline_err(&spec.repo)),
        };
        let path = target_path(dir, &spec.repo, &name);
        // 本地命中（glob 文件名 → 存在即相当 hit；verify 深度由条目判定）
        let entry = FileEntry { name: name.clone(), size: 0, sha256: None, is_lfs: false };
        if local_hit(&path, &entry, self.size_first(spec))
            && std::fs::metadata(&path).map(|m| m.len() > 0).unwrap_or(false)
        {
            return Ok(path);
        }
        if !self.autodownload {
            return Err(offline_err(&name));
        }
        self.fetch(spec, &name, dir)
    }

    /// 下载（MS → 失败/缺失 → HF（auto 语义）；HF 校验按降级链）。
    pub fn fetch(&self, spec: &ModelSpec, name: &str, dir: &Path) -> Result<PathBuf, LaunchError> {
        match self.source {
            ModelSource::Modelscope => ms_fetch(spec, name, dir, self.verify),
            ModelSource::Huggingface => hf_fetch(spec, name, dir, self.verify),
            ModelSource::Auto => ms_fetch(spec, name, dir, self.verify).or_else(|_| {
                eprintln!("reinfer-models: ModelScope miss, falling back to HuggingFace");
                hf_fetch(spec, name, dir, self.verify)
            }),
        }
    }

    /// Verify::Sha256 下未取到条目（本地文件名 glob 命中、无条目元数据）→ 按 size 兜底
    /// （该文件已存在且 >0 即视为命中；sha 级校验在远端有条目时回归）。
    fn size_first(&self, _spec: &ModelSpec) -> Verify {
        Verify::Size
    }
}

fn offline_err(what: &str) -> LaunchError {
    eprintln!(
        "reinfer-models: {what} not found locally and REINFER_MODEL_AUTODOWNLOAD=off - refusing to dial out"
    );
    LaunchError::Fatal
}

fn ms_list(spec: &ModelSpec) -> Result<Vec<FileEntry>, LaunchError> {
    let url = api::ms_list_url_rev(&spec.repo, spec.branch.as_deref());
    let body = api::http_get(&url)?;
    api::parse_ms_files(&body, &url)
}

/// quant 段文件名匹配（远端列表）：`-{q}.` 段（任意扩展——无 .gguf 特判）；多命中列候选。
pub fn select_name(entries: Vec<FileEntry>, spec: &ModelSpec) -> Result<String, LaunchError> {
    if let Some(f) = &spec.file {
        if entries.iter().any(|e| &e.name == f) {
            return Ok(f.clone());
        }
        eprintln!("reinfer-models: file {f} not in repo {}", spec.repo);
        return Err(LaunchError::Fatal);
    }
    if let Some(q) = &spec.quant {
        let qq = format!("-{q}.");
        let mut hits: Vec<String> =
            entries.iter().filter(|e| e.name.contains(&qq)).map(|e| e.name.clone()).collect();
        hits.sort();
        match hits.len() {
            0 => {
                eprintln!(
                    "reinfer-models: no model file matches --quant {q} in repo {}",
                    spec.repo
                );
                Err(LaunchError::Fatal)
            }
            1 => Ok(hits.remove(0)),
            _ => {
                eprintln!("reinfer-models: --quant {q} ambiguous: {hits:?}");
                Err(LaunchError::Fatal)
            }
        }
    } else {
        eprintln!("reinfer-models: need --quant/--file (repo {} has no default)", spec.repo);
        Err(LaunchError::Fatal)
    }
}

fn hf_list_entries(spec: &ModelSpec) -> Result<Vec<FileEntry>, LaunchError> {
    let files = hf_list_files(&spec.repo)?;
    Ok(files
        .into_iter()
        .map(|name| FileEntry { name, size: 0, sha256: None, is_lfs: true })
        .collect())
}

fn ms_fetch(
    spec: &ModelSpec,
    name: &str,
    dir: &Path,
    verify: Verify,
) -> Result<PathBuf, LaunchError> {
    let entries = ms_list(spec)?;
    let entry = entries.into_iter().find(|e| e.name == name).ok_or_else(|| {
        eprintln!("reinfer-models: {name} not in {} files", spec.repo);
        LaunchError::Fatal
    })?;
    // 进度回调：resolver 层不暴露（bin 侧直接走 download_file 新签名接入）
    download_file(&spec.repo, &entry, dir, verify, spec.branch.as_deref(), None)
}

fn hf_fetch(
    spec: &ModelSpec,
    name: &str,
    dir: &Path,
    verify: Verify,
) -> Result<PathBuf, LaunchError> {
    let files = hf_list_files(&spec.repo)?;
    if !files.iter().any(|f| f == name) {
        eprintln!("reinfer-models: {name} not found in {}", spec.repo);
        return Err(LaunchError::Fatal);
    }
    // HF 校验降级：Sha256 → Size（013 r2）
    let v = if verify == Verify::Sha256 { Verify::Size } else { verify };
    // 分支缺省 main（HF 默认分支；ModelSpec.branch=None 的语义，见 D6 r2）
    let branch = spec.branch.as_deref().unwrap_or("main");
    hf_download_file(&spec.repo, name, branch, dir, v)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;

    #[test]
    fn select_by_quant_and_file() {
        let entries = vec![
            FileEntry { name: "m-fp16.gguf".into(), size: 1, sha256: None, is_lfs: false },
            FileEntry { name: "m-q8_0.gguf".into(), size: 2, sha256: None, is_lfs: false },
            FileEntry { name: "m-q4_0.gguf".into(), size: 3, sha256: None, is_lfs: false },
        ];
        assert_eq!(
            select_name(entries.clone(), &ModelSpec::new("r/e").with_quant("q8_0")).unwrap(),
            "m-q8_0.gguf"
        );
        assert_eq!(
            select_name(entries.clone(), &ModelSpec::new("r/e").with_file("m-fp16.gguf")).unwrap(),
            "m-fp16.gguf"
        );
        // 歧义：无匹配两次（q3 没有）
        assert!(select_name(entries.clone(), &ModelSpec::new("r/e").with_quant("q3_0")).is_err());
        // 缺省无 quant/file → 错误（无默认模型）
        assert!(select_name(entries, &ModelSpec::new("r/e")).is_err());
    }

    /// quant 段匹配与扩展名无关（无 .gguf 特判）：同段不同扩展命中；并列 → 歧义列候选。
    #[test]
    fn select_quant_any_extension() {
        let safe = vec![FileEntry {
            name: "m-q8_0.safetensors".into(),
            size: 1,
            sha256: None,
            is_lfs: false,
        }];
        assert_eq!(
            select_name(safe, &ModelSpec::new("r/e").with_quant("q8_0")).unwrap(),
            "m-q8_0.safetensors"
        );
        // 同 quant 段两种扩展并列 → 歧义（列候选错误）
        let both = vec![
            FileEntry { name: "m-q8_0.gguf".into(), size: 1, sha256: None, is_lfs: false },
            FileEntry { name: "m-q8_0.safetensors".into(), size: 2, sha256: None, is_lfs: false },
        ];
        assert!(select_name(both, &ModelSpec::new("r/e").with_quant("q8_0")).is_err());
        // 非本 quant 段的任意扩展仍不命中（q4_0 缺）
        assert!(
            select_name(
                vec![FileEntry {
                    name: "m-q8_0.safetensors".into(),
                    size: 1,
                    sha256: None,
                    is_lfs: false
                }],
                &ModelSpec::new("r/e").with_quant("q4_0")
            )
            .is_err()
        );
    }

    /// 本地 glob 同样任意扩展：safetensors 命中；与 .gguf 并列 → 歧义。
    #[test]
    fn local_glob_any_extension() {
        let dir = std::env::temp_dir().join(format!("reinfer-models-glob-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let r = ModelResolver {
            source: ModelSource::Modelscope,
            dir: dir.clone(),
            verify: Verify::Size,
            autodownload: false,
        };
        let spec = ModelSpec::new("stub/models").with_quant("q8_0");
        assert_eq!(r.resolve_local_name(&spec, &dir).unwrap(), None);
        std::fs::create_dir_all(dir.join("stub/models")).unwrap();
        std::fs::write(dir.join("stub/models/m-q8_0.safetensors"), [0xCCu8; 32]).unwrap();
        assert_eq!(
            r.resolve_local_name(&spec, &dir).unwrap().as_deref(),
            Some("m-q8_0.safetensors")
        );
        std::fs::write(dir.join("stub/models/m-q8_0.gguf"), [0xDDu8; 32]).unwrap();
        assert!(r.resolve_local_name(&spec, &dir).is_err(), "并列同段 → 歧义");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_and_source_parsing() {
        assert_eq!(Verify::parse("sha256"), Some(Verify::Sha256));
        assert_eq!(Verify::parse("size"), Some(Verify::Size));
        assert_eq!(Verify::parse("none"), Some(Verify::None));
        assert_eq!(Verify::parse("x"), None);
        assert_eq!(ModelSource::parse("auto"), Some(ModelSource::Auto));
        assert_eq!(ModelSource::parse("modelscope"), Some(ModelSource::Modelscope));
        assert_eq!(ModelSource::parse("huggingface"), Some(ModelSource::Huggingface));
        assert_eq!(ModelSource::parse("bogus"), None);
    }

    /// 旧扁平布局：根 manifest + 根下文件 → 迁移到 root/{repo}/（幂等、旧清单备份）。
    #[test]
    fn migrate_legacy_flat_layout() {
        let root = std::env::temp_dir().join(format!("reinfer-mig-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("m.gguf"), b"x").unwrap();
        std::fs::write(root.join("t.json"), b"y").unwrap();
        let old: Vec<serde_json::Value> = vec![
            serde_json::json!({"name":"m.gguf","size":1,"sha256":null,"repo":"a/b","branch":"master","fetched_at":0}),
            serde_json::json!({"name":"t.json","size":1,"sha256":null,"repo":"a/b","branch":"master","fetched_at":0}),
        ];
        std::fs::write(root.join("manifest.json"), serde_json::to_vec(&old).unwrap()).unwrap();
        migrate_legacy_flat(&root);
        assert!(root.join("a/b/m.gguf").exists());
        assert!(root.join("a/b/t.json").exists());
        assert!(!root.join("m.gguf").exists(), "旧根文件须迁移");
        let man = crate::download::read_manifest(&root.join("a/b"));
        assert_eq!(man.len(), 2);
        // 旧清单备份 + 幂等
        let backups: Vec<_> = std::fs::read_dir(&root)
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().contains(".manifest.json.bak-"))
            .collect();
        assert_eq!(backups.len(), 1);
        migrate_legacy_flat(&root); // 再跑无操作不炸
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn tilde_expansion_keeps_plain_paths() {
        // 非 ~/ 前缀 → 原样（HOME 有无都不改）
        assert_eq!(expand_tilde("/a/b"), PathBuf::from("/a/b"));
        assert_eq!(expand_tilde("relative/dir"), PathBuf::from("relative/dir"));
    }

    #[test]
    fn spec_has_no_defaults() {
        let s = ModelSpec::new("x/y");
        assert!(s.file.is_none() && s.quant.is_none());
        assert_eq!(s.repo, "x/y");
        assert!(s.branch.is_none(), "branch=None（源缺省：MS master / HF main）");
    }

    // ---- stub 网络测试（本地 TcpListener 双源；T7 验收：auto 的 MS-404 → HF 回退） ----

    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn spawn_stub<F>(handle: F) -> (u16, std::thread::JoinHandle<()>)
    where
        F: Fn(&str) -> (u16, String) + Send + 'static,
    {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        l.set_nonblocking(true).unwrap();
        let j = std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
            let mut served = 0;
            while served < 8 && std::time::Instant::now() < deadline {
                match l.accept() {
                    Ok((mut s, _)) => {
                        let mut buf = [0u8; 8192];
                        let n = s.read(&mut buf).unwrap_or(0);
                        let req = String::from_utf8_lossy(&buf[..n]).to_string();
                        let (code, body) = handle(&req);
                        let head = format!(
                            "HTTP/1.1 {code} {}\r\nContent-Length: {}\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n",
                            if code == 200 { "OK" } else { "Found" },
                            body.len()
                        );
                        let _ = s.write_all(head.as_bytes());
                        let _ = s.write_all(body.as_bytes());
                        served += 1;
                    }
                    Err(_) => std::thread::sleep(std::time::Duration::from_millis(10)),
                }
            }
        });
        (port, j)
    }

    /// auto：MS files 404 → HF siblings hit → 下载成功（名称解析与 fetch 双级回退）。
    #[test]
    fn auto_falls_back_to_hf() {
        let counter = Arc::new(AtomicUsize::new(0));
        // HF stub：siblings 列表 + HEAD/GET resolve
        let had_head = Arc::clone(&counter);
        let (hport, hj) = spawn_stub(move |req| {
            if req.starts_with("HEAD ") {
                had_head.fetch_add(1, Ordering::SeqCst);
                (200, String::new())
            } else if req.starts_with("GET ") && req.contains("/api/models/") {
                (200, r#"{"siblings":[{"rfilename":"m-q8_0.gguf"}]}"#.to_string())
            } else {
                (200, "data".to_string())
            }
        });
        // MS stub：一律 404（Code!=200 → parse 失败 → 触发 HF 回退）
        let (mport, mj) = spawn_stub(|_| (404, r#"{"Code":404,"Data":{"Files":[]}}"#.to_string()));

        api::test_override::set_ms(&format!("http://127.0.0.1:{mport}"));
        api::test_override::set_hf(&format!("http://127.0.0.1:{hport}"));

        let dir = std::env::temp_dir().join(format!("reinfer-models-rt-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let r = ModelResolver {
            source: ModelSource::Auto,
            dir: dir.clone(),
            verify: Verify::Sha256, // 对 HF 自动降级 size
            autodownload: true,
        };
        let p = r.ensure(&ModelSpec::new("stub/models").with_quant("q8_0")).unwrap();
        assert_eq!(p, dir.join("stub/models/m-q8_0.gguf")); // repo 组织
        assert_eq!(std::fs::read(&p).unwrap(), b"data");
        assert!(counter.load(Ordering::SeqCst) >= 1, "HF HEAD must have run");
        // 二次 ensure：本地命中（不触网也可过——glob 层）
        assert_eq!(r.ensure(&ModelSpec::new("stub/models").with_quant("q8_0")).unwrap(), p);

        let _ = std::fs::remove_dir_all(&dir);
        api::test_override::clear();
        mj.join().unwrap();
        hj.join().unwrap();
    }

    /// off：本地 glob 命中即回（无网络）；缺文件 → 明确错误。
    #[test]
    fn offline_local_hit_or_err() {
        let dir = std::env::temp_dir().join(format!("reinfer-models-off-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // 按 repo 组织（root/{repo}）：写入 root/stub/models/
        std::fs::create_dir_all(dir.join("stub/models")).unwrap();
        std::fs::write(dir.join("stub/models/m-q8_0.gguf"), [0xBBu8; 64]).unwrap();
        let r = ModelResolver {
            source: ModelSource::Huggingface, // 即使 HF 源，off 也绝不联网
            dir: dir.clone(),
            verify: Verify::Size,
            autodownload: false,
        };
        let spec = ModelSpec::new("stub/models").with_quant("q8_0");
        assert_eq!(r.ensure(&spec).unwrap(), dir.join("stub/models/m-q8_0.gguf"));
        // 空目录 + off → 错误
        let dir2 = std::env::temp_dir().join(format!("reinfer-models-off2-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir2);
        std::fs::create_dir_all(&dir2).unwrap();
        let r2 = ModelResolver { dir: dir2.clone(), ..r };
        assert!(r2.ensure(&spec).is_err());
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir2);
    }
}

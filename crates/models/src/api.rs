//! ModelScope / HuggingFace 公开 REST 客户端（013 实测契约）。
//!
//! 端点（2026-08-27 经代理实测钉死——见 specs/013-model-fetch/spec.md）：
//! - 列文件：`/api/v1/models/{owner}/{model}/repo/files?Revision=master` ——
//!   `{Code, Data.Files[{Name, Path, Size, Sha256, IsLFS}]}`（Sha256 可直接用于校验）；
//! - 下载：`/api/v1/models/{owner}/{model}/repo?Revision=master&FilePath={name}`
//!   → 302 到 CDN（含瞬时 auth_key；客户端跟随重定向即可，不自行拼 CDN）。
//!
//! 模型标识零硬编码（013 铁律）：repo/文件名一律来自调用方（CLI/env/ModelSpec）。

use reinfer_kernels::LaunchError;
use serde::Deserialize;

/// ModelScope 列文件 API 路径段。
pub const MS_API_BASE: &str = "https://modelscope.cn/api/v1/models";
/// HuggingFace 根。
pub const HF_BASE: &str = "https://huggingface.co";

/// 仓库文件条目（两源统一形态；sha256 为 ModelScope 提供，HF 为 None）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    /// 仓库内文件名。
    pub name: String,
    /// 字节大小。
    pub size: u64,
    /// sha256（ModelScope 提供；HF 为 None）。
    pub sha256: Option<String>,
    /// 是否为 LFS 大文件（源标记；本实现按普通流下载）。
    pub is_lfs: bool,
}

/// 解析 owner/model（`foo/bar` 两段；非法 → Fatal）。
pub fn split_repo(id: &str) -> Result<(&str, &str), LaunchError> {
    let mut it = id.split('/');
    let (owner, model) = (it.next(), it.next());
    match (owner, model) {
        (Some(o), Some(m)) if !o.is_empty() && !m.is_empty() && it.next().is_none() => Ok((o, m)),
        _ => {
            eprintln!("reinfer-models: invalid repo '{id}' (expected owner/model)");
            Err(LaunchError::Fatal)
        }
    }
}

/// ModelScope 列文件 URL。
pub fn ms_list_url(repo: &str) -> String {
    #[cfg(test)]
    if let Some(base) = test_override::ms() {
        return format!("{base}/{repo}/repo/files?Revision=master");
    }
    format!("{MS_API_BASE}/{repo}/repo/files?Revision=master")
}

/// ModelScope 下载入口 URL（302 → CDN）。
pub fn ms_download_url(repo: &str, path: &str) -> String {
    #[cfg(test)]
    if let Some(base) = test_override::ms() {
        return format!("{base}/{repo}/repo?Revision=master&FilePath={path}");
    }
    format!("{MS_API_BASE}/{repo}/repo?Revision=master&FilePath={path}")
}

/// HuggingFace 仓库文件列表 URL（siblings）。
pub fn hf_list_url(repo: &str) -> String {
    #[cfg(test)]
    if let Some(base) = test_override::hf() {
        return format!("{base}/api/models/{repo}");
    }
    format!("{HF_BASE}/api/models/{repo}")
}

/// HuggingFace 下载 URL（302 → LFS/CDN）。
pub fn hf_download_url(repo: &str, path: &str, branch: &str) -> String {
    #[cfg(test)]
    if let Some(base) = test_override::hf() {
        return format!("{base}/{repo}/resolve/{branch}/{path}");
    }
    format!("{HF_BASE}/{repo}/resolve/{branch}/{path}")
}

/// 测试基址覆盖（cfg(test)）：把两源根重定向到本地 stub host。
/// 缺省（未设置）= 真实常量，快照断言不受影响。
#[cfg(test)]
pub(crate) mod test_override {
    // thread_local：并行测试互不污染（urls 快照断言与 auto-回退 stub 可同时运行）。
    use std::cell::RefCell;

    type Pairs = (Option<String>, Option<String>);
    thread_local! {
        static SLOT: RefCell<Pairs> = const { RefCell::new((None, None)) };
    }
    pub fn set_ms(base: &str) {
        SLOT.with(|s| s.borrow_mut().0 = Some(base.to_string()));
    }
    pub fn set_hf(base: &str) {
        SLOT.with(|s| s.borrow_mut().1 = Some(base.to_string()));
    }
    pub fn ms() -> Option<String> {
        SLOT.with(|s| s.borrow().0.clone())
    }
    pub fn hf() -> Option<String> {
        SLOT.with(|s| s.borrow().1.clone())
    }
    pub fn clear() {
        SLOT.with(|s| *s.borrow_mut() = (None, None));
    }
}

/// GET 文本（跟随重定向；标准代理 env 由 ureq 默认读取）。
pub fn http_get(url: &str) -> Result<String, LaunchError> {
    let resp = ureq::get(url).call().map_err(|e| {
        eprintln!("reinfer-models: GET {url} failed: {e}");
        LaunchError::Fatal
    })?;
    resp.into_string().map_err(|e| {
        eprintln!("reinfer-models: read {url} failed: {e}");
        LaunchError::Fatal
    })
}

// ---- ModelScope files 响应解析 ----

#[derive(Deserialize)]
#[allow(non_snake_case)] // serde 直映 API 大写字段
struct MsResp {
    Code: u32,
    Data: MsData,
}

#[derive(Deserialize)]
#[allow(non_snake_case)]
struct MsData {
    Files: Vec<MsFile>,
}

#[derive(Deserialize)]
#[allow(non_snake_case)]
struct MsFile {
    Name: String,
    Size: u64,
    #[serde(rename = "Sha256")]
    Sha256: Option<String>,
    #[serde(rename = "IsLFS")]
    IsLFS: Option<bool>,
}

/// 解析 files API 响应（bad shape/非 200 → Fatal 带信息）。
pub fn parse_ms_files(body: &str, url: &str) -> Result<Vec<FileEntry>, LaunchError> {
    let parsed: MsResp = serde_json::from_str(body).map_err(|e| {
        eprintln!("reinfer-models: bad JSON from files API ({url}): {e}");
        LaunchError::Fatal
    })?;
    if parsed.Code != 200 {
        eprintln!("reinfer-models: files API code {}", parsed.Code);
        return Err(LaunchError::Fatal);
    }
    Ok(parsed
        .Data
        .Files
        .into_iter()
        .map(|f| FileEntry {
            name: f.Name,
            size: f.Size,
            sha256: f.Sha256.filter(|s| !s.is_empty()),
            is_lfs: f.IsLFS.unwrap_or(false),
        })
        .collect())
}

/// HuggingFace siblings 响应。
#[derive(Deserialize)]
struct HfResp {
    siblings: Vec<HfSibling>,
}

#[derive(Deserialize)]
struct HfSibling {
    rfilename: String,
}

/// 解析 HF `/api/models/{repo}` 的 siblings（size 不可得——HEAD 后补）。
pub fn parse_hf_siblings(body: &str) -> Result<Vec<String>, LaunchError> {
    let parsed: HfResp = serde_json::from_str(body).map_err(|e| {
        eprintln!("reinfer-models: bad HF repo JSON: {e}");
        LaunchError::Fatal
    })?;
    Ok(parsed.siblings.into_iter().map(|s| s.rfilename).collect())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;

    const FIXTURE: &str = r#"{"Code":200,"Data":{"Files":[
        {"Name":"qwen2.5-0.5b-instruct-q8_0.gguf","Path":"qwen2.5-0.5b-instruct-q8_0.gguf","Size":675710816,"Sha256":"ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e","IsLFS":false},
        {"Name":"configure.json","Size":123,"Sha256":null,"IsLFS":true}
    ]}}"#;

    /// fixture 摘录：取 2026-08-27 实测 files API 真实摘录（字段形状锚）。
    #[test]
    fn parse_real_shape() {
        let out = parse_ms_files(FIXTURE, "fixture").unwrap();
        assert_eq!(out.len(), 2);
        let q = &out[0];
        assert_eq!(q.name, "qwen2.5-0.5b-instruct-q8_0.gguf");
        assert_eq!(q.size, 675710816);
        assert_eq!(
            q.sha256.as_deref(),
            Some("ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e")
        );
        assert!(!q.is_lfs);
        assert!(out[1].sha256.is_none());
        assert!(out[1].is_lfs);
    }

    #[test]
    fn bad_code_and_bad_json_are_errors() {
        assert!(parse_ms_files(r#"{"Code":404,"Data":{"Files":[]}}"#, "u").is_err());
        assert!(parse_ms_files("not json", "u").is_err());
    }

    #[test]
    fn repo_parse() {
        assert_eq!(
            split_repo("Qwen/Qwen2.5-0.5B-Instruct-GGUF").expect("ok"),
            ("Qwen", "Qwen2.5-0.5B-Instruct-GGUF")
        );
        assert!(split_repo("Qwen").is_err());
        assert!(split_repo("a/b/c").is_err());
        assert!(split_repo("").is_err());
        assert!(split_repo("/").is_err());
    }

    #[test]
    fn urls_match_measured_templates() {
        assert_eq!(
            ms_list_url("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
            "https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/repo/files?Revision=master"
        );
        assert_eq!(
            ms_download_url("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf"),
            "https://modelscope.cn/api/v1/models/Qwen/Qwen2.5-0.5B-Instruct-GGUF/repo?Revision=master&FilePath=qwen2.5-0.5b-instruct-q8_0.gguf"
        );
        assert_eq!(
            hf_download_url("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "x.gguf", "main"),
            "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/x.gguf"
        );
    }

    #[test]
    fn hf_siblings_shapes() {
        let out =
            parse_hf_siblings(r#"{"siblings":[{"rfilename":"a.gguf"},{"rfilename":"b.gguf"}]}"#)
                .unwrap();
        assert_eq!(out, vec!["a.gguf".to_string(), "b.gguf".to_string()]);
    }
}

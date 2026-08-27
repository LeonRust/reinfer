//! HuggingFace 源（013 r2：ModelScope 缺失时的回退）。
//!
//! 校验强度上限 = ETag+size（HF 公开 API 无 sha256 字段——013 r2 降级链：
//! `VERIFY=sha256` 对 HF 源自动降级为 size；manifest 记录 etag）。

use crate::api::{self, FileEntry};
use crate::download::Verify;
use reinfer_kernels::LaunchError;
use std::path::{Path, PathBuf};

/// 列仓库 siblings（仅文件名；size 由 HEAD 补）。
pub fn hf_list_files(repo: &str) -> Result<Vec<String>, LaunchError> {
    let url = api::hf_list_url(repo);
    let body = api::http_get(&url)?;
    api::parse_hf_siblings(&body)
}

/// HEAD 探测（content-length + etag）。
#[derive(Debug, Clone)]
pub struct HfHead {
    /// 字节大小（Content-Length；缺失 → 0）。
    pub size: u64,
    /// 弱 ETag（X-Linked-Etag → ETag）。
    pub etag: Option<String>,
}

/// HEAD 探测 size/etag（跟随重定向前的 resolve URL）。
pub fn hf_head(repo: &str, path: &str, branch: &str) -> Result<HfHead, LaunchError> {
    let url = api::hf_download_url(repo, path, branch);
    let resp = ureq::head(&url).call().map_err(|e| {
        eprintln!("reinfer-models: HF HEAD {url} failed: {e}");
        LaunchError::Fatal
    })?;
    let len = resp.header("Content-Length").and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    let etag = resp.header("X-Linked-Etag").or_else(|| resp.header("ETag")).map(str::to_string);
    Ok(HfHead { size: len, etag })
}

/// HF 下载（302 → LFS/CDN；校验按 verify 降级：Sha256→size；ETag 由 HEAD 预期值验证）。
pub fn hf_download_file(
    repo: &str,
    path: &str,
    branch: &str,
    to_dir: &Path,
    verify: Verify,
) -> Result<PathBuf, LaunchError> {
    std::fs::create_dir_all(to_dir).map_err(crate::download::io_err)?;
    let url = api::hf_download_url(repo, path, branch);
    let head = hf_head(repo, path, branch)?;
    let entry = FileEntry {
        name: path.to_string(),
        size: head.size,
        sha256: None, // HF 无 sha——校验降级
        is_lfs: true,
    };
    crate::download::download_with_url(
        &url,
        &entry,
        to_dir,
        verify,
        repo,
        branch,
        head.etag.as_deref(),
        None, // HF 源暂不暴露进度（bin 侧按需接入）
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;

    #[test]
    fn parse_hf_siblings_ok_and_bad() {
        let body = r#"{"siblings":[{"rfilename":"a.gguf"},{"rfilename":"b.gguf"}]}"#;
        let out = api::parse_hf_siblings(body).unwrap();
        assert_eq!(out, vec!["a.gguf".to_string(), "b.gguf".to_string()]);
        assert!(api::parse_hf_siblings("[]").is_err());
    }
}

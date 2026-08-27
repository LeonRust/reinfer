//! reinfer —— 支持 CUDA / 昇腾 CANN 的 Rust 推理引擎（server | cli | bench）
//!
//! CLI 契约：docs/design/cli-contract-2026-08-27.md（用户 2026-08-27 定稿；先例对照见契约 §5）。
//! 当前范围：`download` 顶层（hf download / modelscope download 先例）+ `model` 组唯一命令
//! `list`（本地清单：docker/ollama/modelscope-ng 惯例，零参）。
//! 解析用 std（无 clap）；用法错误 → exit 2 + stderr 提示。

use reinfer_models::api::FileEntry;
use reinfer_models::{LaunchError, ModelResolver, ModelSpec};
use std::path::{Path, PathBuf};

/// 用法错误退出码。
const EXIT_USAGE: i32 = 2;

fn main() {
    // 开发测试环境：本目录 .env（gitignored；模板 .env.example）；无文件静默跳过
    let _ = dotenvy::dotenv();
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(run(&args));
}

fn run(args: &[String]) -> i32 {
    match args.first().map(String::as_str) {
        None => {
            print_usage();
            EXIT_USAGE
        }
        Some("help") | Some("-h") | Some("--help") => {
            print_usage();
            0
        }
        Some("download") => run_download(&args[1..]),
        Some("model") => run_model(&args[1..]),
        Some(other) => {
            eprintln!("reinfer: unknown command '{other}' (try `reinfer help`)");
            EXIT_USAGE
        }
    }
}

fn print_usage() {
    println!(
        "reinfer {}\n\
         \n\
         USAGE:\n\
         \x20   reinfer <command> [args]\n\
         \n\
         COMMANDS:\n\
         \x20   download <repo> [file...]     Download model files (hf download semantics)\n\
         \x20   model list                    List locally downloaded GGUF files\n\
         \n\
         DOWNLOAD:\n\
         \x20   reinfer download <repo>                          # entire repo (hf default)\n\
         \x20   reinfer download <repo> <file...>                # exact files\n\
         \x20   reinfer download <repo> -q <qtag>                # quant tag (e.g. q8_0)\n\
         \x20   reinfer download <repo> --include <glob> [--exclude <glob>]\n\
         \x20   reinfer download <repo> ... [--revision <ref>] [--local-dir <dir>]\n\
         \n\
         ENV (source policy): REINFER_MODEL_SOURCE=modelscope|huggingface|auto (default auto\n\
         \x20   = ModelScope first, falls back to HuggingFace); REINFER_MODEL_VERIFY=sha256|size|none;\n\
         \x20   REINFER_MODEL_AUTODOWNLOAD=on|off (off never dials out); REINFER_MODEL_DIR override;\n\
         \x20   standard HTTP_PROXY/HTTPS_PROXY/NO_PROXY (e.g. http://192.168.0.1:7890).\n\
         \n\
         EXAMPLES:\n\
         \x20   reinfer model list\n\
         \x20   reinfer download Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q8_0\n\
         \x20   reinfer download Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q8_0 --local-dir ~/models",
        env!("CARGO_PKG_VERSION")
    );
}

/// download 解析结果（可单测）。
#[derive(Debug, PartialEq, Eq)]
enum DownloadCmd {
    /// 量化档（引擎领域词；互斥于 file.../--include）。
    Quant { repo: String, quant: String, local_dir: Option<PathBuf>, revision: Option<String> },
    /// 精确文件列表（hf 位置 file... 先例；空表只在全量分支出现）。
    Files { repo: String, files: Vec<String>, local_dir: Option<PathBuf>, revision: Option<String> },
    /// 模式过滤（hf --include/--exclude 先例）。
    Patterns {
        repo: String,
        include: String,
        exclude: Option<String>,
        local_dir: Option<PathBuf>,
        revision: Option<String>,
    },
    /// 无显式选择 = 整个仓库（hf 默认语义）。
    All { repo: String, local_dir: Option<PathBuf>, revision: Option<String> },
}

/// model 解析结果（可单测）。
#[derive(Debug, PartialEq, Eq)]
enum ModelCmd {
    Help,
    /// 本地清单。
    List,
}

/// 切分 `--flag=value`（git/gh 风格）。
fn split_flag(arg: &str) -> (&str, Option<&str>) {
    match arg.split_once('=') {
        Some((f, v)) => (f, Some(v)),
        None => (arg, None),
    }
}

/// 取旗子值（`=` 内联或紧跟一项）。
fn flag_value<'a>(args: &'a [String], i: usize, names: &[&str], short: &str) -> Result<&'a str, String> {
    let raw = &args[i];
    let (flag, inline) = split_flag(raw);
    if !names.contains(&flag) && flag != short {
        return Err(format!("unknown option '{raw}'"));
    }
    if let Some(v) = inline {
        return Ok(v);
    }
    args.get(i + 1).map(|s| s.as_str()).ok_or(format!("{flag} needs a value"))
}

fn parse_download(args: &[String]) -> Result<DownloadCmd, String> {
    let repo = args.get(0).ok_or("download needs <repo>")?.to_string();
    let mut quant = None;
    let mut include = None;
    let mut exclude = None;
    let mut local_dir = None;
    let mut revision = None;
    let mut files: Vec<String> = Vec::new();
    let mut i = 1;
    while i < args.len() {
        let (flag, _) = split_flag(&args[i]);
        match flag {
            "-q" | "--quant" => {
                quant = Some(flag_value(args, i, &["--quant"], "-q")?.to_string());
                i += if args[i].contains('=') { 1 } else { 2 };
            }
            "--include" => {
                include = Some(flag_value(args, i, &["--include"], "")?.to_string());
                i += if args[i].contains('=') { 1 } else { 2 };
            }
            "--exclude" => {
                exclude = Some(flag_value(args, i, &["--exclude"], "")?.to_string());
                i += if args[i].contains('=') { 1 } else { 2 };
            }
            "--local-dir" => {
                local_dir = Some(PathBuf::from(flag_value(args, i, &["--local-dir"], "")?.to_string()));
                i += if args[i].contains('=') { 1 } else { 2 };
            }
            "--revision" => {
                revision = Some(flag_value(args, i, &["--revision"], "")?.to_string());
                i += if args[i].contains('=') { 1 } else { 2 };
            }
            _ if flag.starts_with('-') => return Err(format!("unknown option '{}'", args[i])),
            _ => {
                files.push(args[i].clone());
                i += 1;
            }
        }
    }
    if quant.is_some() && (include.is_some() || exclude.is_some() || !files.is_empty()) {
        return Err("-q is exclusive with file.../--include".into());
    }
    if exclude.is_some() && include.is_none() {
        return Err("--exclude requires --include".into());
    }
    let dir_spec = (local_dir, revision);
    let (local_dir, revision) = dir_spec;
    if let Some(q) = quant {
        return Ok(DownloadCmd::Quant { repo, quant: q, local_dir, revision });
    }
    if let Some(inc) = include {
        return Ok(DownloadCmd::Patterns { repo, include: inc, exclude, local_dir, revision });
    }
    if files.is_empty() {
        // hf 默认：无显式文件选择 = 整个仓库
        return Ok(DownloadCmd::All { repo, local_dir, revision });
    }
    Ok(DownloadCmd::Files { repo, files, local_dir, revision })
}

fn parse_model(args: &[String]) -> Result<ModelCmd, String> {
    let Some(sub) = args.first() else {
        return Err("model needs a subcommand (list|help)".into());
    };
    match sub.as_str() {
        "help" | "-h" | "--help" => Ok(ModelCmd::Help),
        "list" => {
            if args.len() > 1 {
                return Err("list takes no arguments".into());
            }
            Ok(ModelCmd::List)
        }
        other => Err(format!("unknown model subcommand '{other}' (try `reinfer model help`)")),
    }
}

fn run_download(args: &[String]) -> i32 {
    let cmd = match parse_download(args) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("reinfer: error: {msg}");
            return EXIT_USAGE;
        }
    };
    let resolver = match ModelResolver::from_env() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("reinfer: {e:?}");
            return 1;
        }
    };
    match cmd {
        DownloadCmd::Quant { repo, quant, local_dir, revision } => {
            let mut spec = ModelSpec::new(repo);
            spec = spec.with_quant(quant);
            if let Some(r) = revision {
                spec = spec.with_branch(r);
            }
            let dir = local_dir.unwrap_or_else(|| resolver.dir.clone());
            match resolver.ensure_to(&spec, &dir) {
                Ok(path) => {
                    println!("{} ready", path.display());
                    print_manifest_line(&dir, &path);
                    0
                }
                Err(e) => {
                    eprintln!("reinfer: download failed: {e:?}");
                    print_proxy_hint();
                    1
                }
            }
        }
        DownloadCmd::Files { repo, files, local_dir, revision } => {
            let dir = local_dir.unwrap_or_else(|| resolver.dir.clone());
            let want = files.clone();
            download_many(&resolver, &repo, &dir, revision.as_deref(), move |entries| {
                entries.iter().filter(|e| want.contains(&e.name)).cloned().collect()
            });
            0
        }
        DownloadCmd::Patterns { repo, include, exclude, local_dir, revision } => {
            let dir = local_dir.unwrap_or_else(|| resolver.dir.clone());
            let (inc, exc) = (include.clone(), exclude.clone());
            download_many(&resolver, &repo, &dir, revision.as_deref(), move |entries| {
                entries
                    .iter()
                    .filter(|e| {
                        glob_match(&inc, &e.name)
                            && exc.as_deref().map(|x| !glob_match(x, &e.name)).unwrap_or(true)
                    })
                    .cloned()
                    .collect()
            });
            0
        }
        DownloadCmd::All { repo, local_dir, revision } => {
            let dir = local_dir.unwrap_or_else(|| resolver.dir.clone());
            download_many(&resolver, &repo, &dir, revision.as_deref(), |e| e.to_vec());
            0
        }
    }
}

fn run_model(args: &[String]) -> i32 {
    let cmd = match parse_model(args) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("reinfer: error: {msg}");
            return EXIT_USAGE;
        }
    };
    match cmd {
        ModelCmd::Help => {
            print_usage();
            0
        }
        ModelCmd::List => cmd_list_local(),
    }
}

/// 本地已下载 GGUF 清单（`list` 惯例：默认本地，零参）。
fn cmd_list_local() -> i32 {
    let resolver = match ModelResolver::from_env() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("reinfer: {e:?}");
            return 1;
        }
    };
    let dir = resolver.dir.clone();
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("reinfer: model dir not readable: {} ({err})", dir.display());
            return 1;
        }
    };
    let man = reinfer_models::download::read_manifest(&dir);
    let mut rows: Vec<(String, u64, String, String)> = entries
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.ends_with(".gguf") {
                return None;
            }
            let size = e.metadata().map(|m| m.len()).ok()?;
            let m = man.iter().find(|m| m.name == name);
            let sha = m
                .and_then(|m| m.sha256.as_deref())
                .map(|s| s[..s.len().min(16)].to_string())
                .unwrap_or_else(|| "-".to_string());
            let src = m.map(|m| format!("{}@{}", m.repo, m.branch)).unwrap_or_else(|| "-".into());
            Some((name, size, sha, src))
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    if rows.is_empty() {
        println!("no local GGUF files in {}", dir.display());
        return 0;
    }
    println!("{} ({} GGUF file(s)):\n", dir.display(), rows.len());
    let (n, s, h, src) = ("name", "size", "sha256", "source");
    println!("{n:<48} {s:>12}  {h:<16}  {src}");
    for (name, size, sha, src) in &rows {
        println!("{name:<48} {size:>12}  {sha:<16}  {src}");
    }
    0
}

/// 共享下载路径（精确/模式/全量）：revision 感知列表 → pick → 逐个下载。
fn download_many(
    resolver: &ModelResolver,
    repo: &str,
    dir: &Path,
    revision: Option<&str>,
    pick: impl FnOnce(&[FileEntry]) -> Vec<FileEntry>,
) {
    let entries = match list_files(repo, revision) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("reinfer: {e:?}");
            print_proxy_hint();
            return;
        }
    };
    let targets = pick(&entries);
    if targets.is_empty() {
        eprintln!("reinfer: nothing to download in {repo}");
        return;
    }
    let mut ok = 0;
    for e in &targets {
        println!("-> {}", e.name);
        match reinfer_models::download::download_file(repo, e, dir, resolver.verify, revision) {
            Ok(p) => {
                ok += 1;
                println!("   {} (done)", p.display());
            }
            Err(err) => eprintln!("reinfer: download {} failed: {err:?}", e.name),
        }
    }
    println!("{ok}/{} downloaded", targets.len());
}

/// 取远端仓库条目列表（revision 可选）。
fn list_files(repo: &str, revision: Option<&str>) -> Result<Vec<FileEntry>, LaunchError> {
    let url = reinfer_models::api::ms_list_url_rev(repo, revision);
    let body = reinfer_models::api::http_get(&url)?;
    reinfer_models::api::parse_ms_files(&body, &url)
}

/// 极简 glob（`*` `?`；`**` 视作 `*`）——`--include/--exclude` 用。
fn glob_match(pat: &str, name: &str) -> bool {
    let src: Vec<char> = name.chars().collect();
    let pat: Vec<char> = pat.chars().collect();
    let (mut p, mut s, mut star, mut star_s) = (0usize, 0usize, None, 0usize);
    while s < src.len() {
        if p < pat.len() && (pat[p] == '?' || pat[p] == src[s]) {
            p += 1;
            s += 1;
        } else if p < pat.len() && pat[p] == '*' {
            star = Some(p);
            p += 1;
            star_s = s;
        } else if let Some(sp) = star {
            p = sp + 1;
            star_s += 1;
            s = star_s;
        } else {
            return false;
        }
    }
    while p < pat.len() && pat[p] == '*' {
        p += 1;
    }
    p == pat.len()
}

/// manifest 留痕提示（有则打印）。
fn print_manifest_line(dir: &Path, path: &Path) {
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let man = reinfer_models::download::read_manifest(dir);
    if let Some(e) = man.iter().find(|e| e.name == name) {
        let sha = e.sha256.as_deref().unwrap_or("-");
        let tail = &sha[sha.len().min(18)..];
        println!("manifest: repo={} branch={} size={} sha256…{tail}", e.repo, e.branch, e.size);
    }
}

fn print_proxy_hint() {
    if std::env::var("HTTPS_PROXY").is_err() && std::env::var("https_proxy").is_err() {
        eprintln!(
            "hint: no HTTPS_PROXY set; set it if your network needs a proxy (e.g. http://192.168.0.1:7890)"
        );
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;

    #[test]
    fn parse_download_quant() {
        let a: Vec<String> = ["Qwen/Qwen2.5-0.5B-Instruct-GGUF", "-q", "q8_0", "--local-dir", "/tmp/m"]
            .map(String::from)
            .to_vec();
        assert_eq!(
            parse_download(&a).unwrap(),
            DownloadCmd::Quant {
                repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF".into(),
                quant: "q8_0".into(),
                local_dir: Some(PathBuf::from("/tmp/m")),
                revision: None,
            }
        );
        // 等号形式 -q=q8_0 + --revision=v2
        let b: Vec<String> = ["r/e", "-q=q8_0", "--revision=v2"].map(String::from).to_vec();
        assert_eq!(
            parse_download(&b).unwrap(),
            DownloadCmd::Quant { repo: "r/e".into(), quant: "q8_0".into(), local_dir: None, revision: Some("v2".into()) }
        );
    }

    #[test]
    fn parse_download_files_and_all() {
        let a: Vec<String> = ["r/e", "a.gguf", "config.json"].map(String::from).to_vec();
        assert_eq!(
            parse_download(&a).unwrap(),
            DownloadCmd::Files {
                repo: "r/e".into(),
                files: vec!["a.gguf".into(), "config.json".into()],
                local_dir: None,
                revision: None,
            }
        );
        // 无选择 → 全仓库（hf 默认）
        assert_eq!(
            parse_download(&["r/e".into()]).unwrap(),
            DownloadCmd::All { repo: "r/e".into(), local_dir: None, revision: None }
        );
    }

    #[test]
    fn parse_download_patterns_and_errors() {
        let a: Vec<String> = ["r/e", "--include", "*.gguf", "--exclude", "*fp16*"].map(String::from).to_vec();
        assert_eq!(
            parse_download(&a).unwrap(),
            DownloadCmd::Patterns {
                repo: "r/e".into(),
                include: "*.gguf".into(),
                exclude: Some("*fp16*".into()),
                local_dir: None,
                revision: None,
            }
        );
        let b: Vec<String> = ["r/e", "--exclude", "*fp16*"].map(String::from).to_vec();
        assert!(parse_download(&b).is_err()); // exclude 须配 include
        let c: Vec<String> = ["r/e", "-q", "q8_0", "--include", "x"].map(String::from).to_vec();
        assert!(parse_download(&c).is_err()); // -q 与 include/file 互斥
        let d: Vec<String> = ["r/e", "--bogus"].map(String::from).to_vec();
        assert!(parse_download(&d).is_err());
        assert!(parse_download(&[].to_vec()).is_err());
    }

    #[test]
    fn glob_works() {
        assert!(glob_match("*.gguf", "a.gguf"));
        assert!(!glob_match("*.gguf", "a.bin"));
        assert!(glob_match("*q8_0*", "m-q8_0.gguf"));
        assert!(glob_match("m-??_0.gguf", "m-q8_0.gguf"));
        assert!(!glob_match("m-??_0.gguf", "m-q10_0.gguf"));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("sub/*.gguf", "sub/x.gguf"));
        assert!(glob_match("sub/*.gguf", "sub/deep/x.gguf")); // fnmatch 语义：* 跨 /
        assert!(!glob_match("other/*.gguf", "sub/x.gguf"));
    }

    #[test]
    fn parse_model_and_unknown() {
        assert_eq!(parse_model(&["list".into()]).unwrap(), ModelCmd::List);
        assert!(parse_model(&["list".into(), "x".into()]).is_err());
        assert_eq!(parse_model(&["help".into()]).unwrap(), ModelCmd::Help);
        assert!(parse_model(&["frob".into()]).is_err());
        assert_eq!(run(&["frobnicate".to_string()]), EXIT_USAGE);
    }
}

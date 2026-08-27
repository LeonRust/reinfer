//! reinfer —— 支持 CUDA / 昇腾 CANN 的 Rust 推理引擎（server | cli | bench）
//!
//! 013：`model` 子命令（list / ls-remote / get）——纯 Rust 模型获取（ModelScope 优先、auto 回退 HF）。
//! 解析用 std（无 clap——宪法"单二进制/最小依赖"）；参数错误 → exit 2 + 用法提示。
//!
//! CLI 设计对齐成熟工具惯例（非自创范式）：
//! - `list` 本地清单——`docker image ls` / `ollama list` / `pip list`：默认本地、零参；
//! - `ls-remote <repo>` 远端清单——git `ls-remote` 的"列远端"表述；
//! - `get` —— `hf download` 语义（repo 必填位置参数、`--local-dir` 命名、`-q`/`-f` 短旗、
//!   `--flag=value` 形式；`--quant`/`--file`/`--all` 互斥语义见 specs/013 plan D6）。

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
         \x20   reinfer model <subcommand> [args]\n\
         \n\
         SUBCOMMANDS (model):\n\
         \x20   list                               List locally downloaded GGUF files\n\
         \x20   ls-remote <repo>                   List GGUF files in a remote repo\n\
         \x20   get <repo> [-q <qtag> | -f <file>] Download a GGUF (quant tag or exact file)\n\
         \x20                            [--all]      Download every GGUF in the repo\n\
         \x20                            [--local-dir <dir>]  Override model dir (default: $HOME/models/reinfer)\n\
         \n\
         MODEL SOURCE: env REINFER_MODEL_SOURCE=modelscope|huggingface|auto (default auto);\n\
         \x20   auto = ModelScope first, falls back to HuggingFace.\n\
         \x20   REINFER_MODEL_VERIFY=sha256|size|none · REINFER_MODEL_AUTODOWNLOAD=on|off,\n\
         \x20   REINFER_MODEL_DIR override (env) · autodownload off never dials out.\n\
         \n\
         EXAMPLES:\n\
         \x20   reinfer model list\n\
         \x20   reinfer model ls-remote Qwen/Qwen2.5-0.5B-Instruct-GGUF\n\
         \x20   reinfer model get  Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q8_0\n\
         \n\
         PROXY: standard HTTP_PROXY / HTTPS_PROXY / NO_PROXY env (e.g. http://192.168.0.1:7890).",
        env!("CARGO_PKG_VERSION")
    );
}

/// model 子命令解析结果（可单测）。
#[derive(Debug, PartialEq, Eq)]
enum ModelCmd {
    Help,
    /// 本地清单。
    List,
    /// 远端仓库文件清单。
    LsRemote {
        repo: String,
    },
    Get {
        repo: String,
        quant: Option<String>,
        file: Option<String>,
        all: bool,
        local_dir: Option<PathBuf>,
    },
}

/// 切分 `--flag=value`（git/gh 风格）为 (flag, Some(value))；无 `=` → (flag, None)。
fn split_flag(arg: &str) -> (&str, Option<&str>) {
    match arg.split_once('=') {
        Some((f, v)) => (f, Some(v)),
        None => (arg, None),
    }
}

/// 取 `--<long>[-q]`/`-x value` 形式的值（第 i 项为旗子；紧跟一项或 `=` 内联）。
fn flag_value<'a>(
    args: &'a [String],
    i: usize,
    names: &[&str],
    short: &str,
) -> Result<&'a str, String> {
    let raw = args[i].as_str();
    let (flag, inline) = split_flag(raw);
    if inline.is_some() {
        if !names.contains(&flag) && flag != short {
            return Err(format!("unknown option '{raw}'"));
        }
    } else if !names.contains(&flag) && flag != short {
        return Err(format!("unknown option '{raw}'"));
    }
    if let Some(v) = inline {
        return Ok(v);
    }
    args.get(i + 1).map(|s| s.as_str()).ok_or(format!("{flag} needs a value"))
}

/// 解析 `model <args...>`（`args[0]='model'` 之后的部分）。
fn parse_model(args: &[String]) -> Result<ModelCmd, String> {
    let Some(sub) = args.first() else {
        return Err("model needs a subcommand (list|ls-remote|get|help)".into());
    };
    match sub.as_str() {
        "help" | "-h" | "--help" => Ok(ModelCmd::Help),
        "list" => {
            if args.len() > 1 {
                return Err("list takes no arguments".into());
            }
            Ok(ModelCmd::List)
        }
        "ls-remote" => {
            if args.len() != 2 {
                return Err("ls-remote takes exactly one <repo>".into());
            }
            Ok(ModelCmd::LsRemote { repo: args[1].clone() })
        }
        "get" => {
            let repo = args.get(1).ok_or("get needs <repo>")?.to_string();
            let mut quant = None;
            let mut file = None;
            let mut all = false;
            let mut local_dir = None;
            let mut i = 2;
            while i < args.len() {
                let (flag, _) = split_flag(&args[i]);
                match flag {
                    "--quant" | "-q" => {
                        quant = Some(flag_value(args, i, &["--quant"], "-q")?.to_string());
                        i += if args[i].contains('=') { 1 } else { 2 };
                    }
                    "--file" | "-f" => {
                        file = Some(flag_value(args, i, &["--file"], "-f")?.to_string());
                        i += if args[i].contains('=') { 1 } else { 2 };
                    }
                    "--local-dir" => {
                        local_dir = Some(PathBuf::from(
                            flag_value(args, i, &["--local-dir"], "")?.to_string(),
                        ));
                        i += if args[i].contains('=') { 1 } else { 2 };
                    }
                    "--all" => {
                        all = true;
                        i += 1;
                    }
                    _ => return Err(format!("unknown get option '{}'", args[i])),
                }
            }
            if all && (quant.is_some() || file.is_some()) {
                return Err("--all is exclusive with -q/--file".into());
            }
            if quant.is_some() && file.is_some() {
                return Err("-q and -f are mutually exclusive".into());
            }
            Ok(ModelCmd::Get { repo, quant, file, all, local_dir })
        }
        other => Err(format!("unknown model subcommand '{other}' (try `reinfer model help`)")),
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
        ModelCmd::LsRemote { repo } => cmd_ls_remote(&repo),
        ModelCmd::Get { repo, quant, file, all, local_dir } => {
            cmd_get(&repo, quant, file, all, local_dir)
        }
    }
}

/// 本地已下载 GGUF 清单（`list`——docker/ollama 惯例：默认本地）。
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

/// 远端仓库文件清单（`ls-remote`——git 惯用表述）。
fn cmd_ls_remote(repo: &str) -> i32 {
    match list_files(repo) {
        Ok(entries) => {
            let (n, s, h) = ("name", "size", "sha256");
            println!("{n:<48} {s:>12}  {h}");
            let n = entries
                .iter()
                .filter(|e| e.name.ends_with(".gguf"))
                .inspect(|e| {
                    println!(
                        "{:<48} {:>12}  {}",
                        e.name,
                        e.size,
                        e.sha256.as_deref().map(|s| &s[..16.min(s.len())]).unwrap_or("-")
                    );
                })
                .count();
            println!("\n{n} GGUF file(s) in {repo}");
            0
        }
        Err(e) => {
            eprintln!("reinfer: model ls-remote failed: {e:?}");
            print_proxy_hint();
            1
        }
    }
}

/// 取远端仓库条目列表。
fn list_files(repo: &str) -> Result<Vec<FileEntry>, LaunchError> {
    let url = reinfer_models::api::ms_list_url(repo);
    let body = reinfer_models::api::http_get(&url)?;
    reinfer_models::api::parse_ms_files(&body, &url)
}

fn cmd_get(
    repo: &str,
    quant: Option<String>,
    file: Option<String>,
    all: bool,
    local_dir: Option<PathBuf>,
) -> i32 {
    let resolver = match ModelResolver::from_env() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("reinfer: {e:?}");
            return 1;
        }
    };
    let dir = local_dir.unwrap_or_else(|| resolver.dir.clone());
    if all {
        return cmd_get_all(&resolver, repo, &dir);
    }
    let mut spec = ModelSpec::new(repo);
    if let Some(q) = quant {
        spec = spec.with_quant(q);
    }
    if let Some(f) = file {
        spec = spec.with_file(f);
    }
    match resolver.ensure_to(&spec, &dir) {
        Ok(path) => {
            println!("{} ready", path.display());
            print_manifest_line(&dir, &path);
            0
        }
        Err(e) => {
            eprintln!("reinfer: model get failed: {e:?}");
            print_proxy_hint();
            1
        }
    }
}

/// 下载该 repo 下全部 GGUF。
fn cmd_get_all(resolver: &ModelResolver, repo: &str, dir: &Path) -> i32 {
    let entries = match list_files(repo) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("reinfer: model ls-remote failed: {e:?}");
            print_proxy_hint();
            return 1;
        }
    };
    let ggs: Vec<FileEntry> = entries.into_iter().filter(|e| e.name.ends_with(".gguf")).collect();
    if ggs.is_empty() {
        eprintln!("reinfer: no GGUF files in {repo}");
        return 1;
    }
    let mut ok = 0;
    for e in &ggs {
        println!("-> {}", e.name);
        match reinfer_models::download::download_file(repo, e, dir, resolver.verify) {
            Ok(p) => {
                ok += 1;
                println!("   {} (done)", p.display());
            }
            Err(err) => eprintln!("reinfer: download {} failed: {err:?}", e.name),
        }
    }
    println!("{ok}/{} downloaded", ggs.len());
    if ok == ggs.len() { 0 } else { 1 }
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
    fn parse_list_local() {
        assert_eq!(parse_model(&["list".into()]).unwrap(), ModelCmd::List);
        // 零参：多余参数 → 错
        assert!(parse_model(&["list".into(), "x".into()]).is_err());
    }

    #[test]
    fn parse_ls_remote() {
        let a: Vec<String> =
            ["ls-remote", "Qwen/Qwen2.5-0.5B-Instruct-GGUF"].map(String::from).to_vec();
        assert_eq!(
            parse_model(&a).unwrap(),
            ModelCmd::LsRemote { repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF".into() }
        );
        assert!(parse_model(&["ls-remote".into()]).is_err());
        assert!(parse_model(&["ls-remote".into(), "a/b".into(), "x".into()]).is_err());
    }

    #[test]
    fn parse_get_ok_long_and_short() {
        let a: Vec<String> =
            ["get", "a/b", "--quant", "q8_0", "--local-dir", "/tmp/m"].map(String::from).to_vec();
        assert_eq!(
            parse_model(&a).unwrap(),
            ModelCmd::Get {
                repo: "a/b".into(),
                quant: Some("q8_0".into()),
                file: None,
                all: false,
                local_dir: Some(PathBuf::from("/tmp/m")),
            }
        );
        // 短旗 + 等号形式（`-f=x.gguf`、`--quant=q8_0`——git/gh 惯例）
        let b: Vec<String> = ["get", "a/b", "-f=x.gguf"].map(String::from).to_vec();
        assert_eq!(
            parse_model(&b).unwrap(),
            ModelCmd::Get {
                repo: "a/b".into(),
                quant: None,
                file: Some("x.gguf".into()),
                all: false,
                local_dir: None
            }
        );
        let c: Vec<String> = ["get", "a/b", "-q", "q8_0"].map(String::from).to_vec();
        assert_eq!(
            parse_model(&c).unwrap(),
            ModelCmd::Get {
                repo: "a/b".into(),
                quant: Some("q8_0".into()),
                file: None,
                all: false,
                local_dir: None
            }
        );
        // --all
        let d: Vec<String> = ["get", "a/b", "--all"].map(String::from).to_vec();
        match parse_model(&d).unwrap() {
            ModelCmd::Get { all, .. } => assert!(all),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parse_get_errors() {
        // 互斥
        let args: Vec<String> = ["get", "a/b", "-q", "q8", "-f", "x"].map(String::from).to_vec();
        assert!(parse_model(&args).is_err());
        let args: Vec<String> = ["get", "a/b", "--all", "-q", "q8"].map(String::from).to_vec();
        assert!(parse_model(&args).is_err());
        // 缺值/未知选项
        let args: Vec<String> = ["get", "a/b", "--quant"].map(String::from).to_vec();
        assert!(parse_model(&args).is_err());
        let args: Vec<String> = ["get", "a/b", "--bogus"].map(String::from).to_vec();
        assert!(parse_model(&args).is_err());
        // 缺 repo
        assert!(parse_model(&["get".into()]).is_err());
        // 未知子命令
        assert!(parse_model(&["frob".into()]).is_err());
    }

    #[test]
    fn parse_help() {
        assert_eq!(parse_model(&["help".into()]).unwrap(), ModelCmd::Help);
        assert_eq!(parse_model(&["-h".into()]).unwrap(), ModelCmd::Help);
    }

    #[test]
    fn run_unknown_command_is_usage() {
        let args = vec!["frobnicate".to_string()];
        assert_eq!(run(&args), EXIT_USAGE);
    }
}

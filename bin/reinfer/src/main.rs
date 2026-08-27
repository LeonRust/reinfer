//! reinfer —— 支持 CUDA / 昇腾 CANN 的 Rust 推理引擎（server | cli | bench）
//!
//! 013：`model` 子命令（list/get）——纯 Rust 模型获取（ModelScope 优先、auto 回退 HF）。
//! 解析用 std（无 clap——宪法"单二进制/最小依赖"）；参数错误 → exit 2 + 用法提示。

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
         \x20   list <repo>                        List GGUF files (name/size/sha256)\n\
         \x20   get  <repo> --quant <q>            Download quantized GGUF\n\
         \x20   get  <repo> --file <name>          Download exact file\n\
         \x20   get  <repo> --all                  Download all GGUF files\n\
         \x20   get  <repo> [--quant|--file] --to <dir>\n\
         \"                                  Override model dir (default: $HOME/models/reinfer)\n\
         \n\
         MODEL SOURCE: env REINFER_MODEL_SOURCE=modelscope|huggingface|auto (default auto);\n\
         \x20   auto = ModelScope first, falls back to HuggingFace. VERIFY=sha256|size|none,\n\
         \x20   AUTODOWNLOAD=on|off (default on; off never dials out).\n\
         \n\
         EXAMPLES:\n\
         \x20   reinfer model list Qwen/Qwen2.5-0.5B-Instruct-GGUF\n\
         \x20   reinfer model get  Qwen/Qwen2.5-0.5B-Instruct-GGUF --quant q8_0\n\
         \n\
         PROXY: standard HTTP_PROXY / HTTPS_PROXY / NO_PROXY env (e.g. http://192.168.0.1:7890).",
        env!("CARGO_PKG_VERSION")
    );
}

/// model 子命令解析结果（可单测）。
#[derive(Debug, PartialEq, Eq)]
enum ModelCmd {
    Help,
    List {
        repo: String,
    },
    Get {
        repo: String,
        quant: Option<String>,
        file: Option<String>,
        all: bool,
        to: Option<PathBuf>,
    },
}

/// 解析 `model <args...>`（`args[0]='model'` 之后的部分）。
fn parse_model(args: &[String]) -> Result<ModelCmd, String> {
    let Some(sub) = args.first() else {
        return Err("model needs a subcommand (list|get|help)".into());
    };
    match sub.as_str() {
        "help" | "-h" | "--help" => Ok(ModelCmd::Help),
        "list" => {
            let repo = args.get(1).ok_or("list needs <owner/model>")?.to_string();
            if args.len() > 2 {
                return Err("list takes exactly one argument".into());
            }
            Ok(ModelCmd::List { repo })
        }
        "get" => {
            let repo = args.get(1).ok_or("get needs <owner/model>")?.to_string();
            let mut quant = None;
            let mut file = None;
            let mut all = false;
            let mut to = None;
            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "--quant" => {
                        quant = Some(args.get(i + 1).ok_or("--quant needs a value")?.clone());
                        i += 2;
                    }
                    "--file" => {
                        file = Some(args.get(i + 1).ok_or("--file needs a value")?.clone());
                        i += 2;
                    }
                    "--to" => {
                        let v = args.get(i + 1).ok_or("--to needs a value")?;
                        to = Some(PathBuf::from(v.as_str()));
                        i += 2;
                    }
                    "--all" => {
                        all = true;
                        i += 1;
                    }
                    other => return Err(format!("unknown get option '{other}'")),
                }
            }
            if all && (quant.is_some() || file.is_some()) {
                return Err("--all is exclusive with --quant/--file".into());
            }
            if quant.is_some() && file.is_some() {
                return Err("--quant and --file are mutually exclusive".into());
            }
            Ok(ModelCmd::Get { repo, quant, file, all, to })
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
        ModelCmd::List { repo } => cmd_list(&repo),
        ModelCmd::Get { repo, quant, file, all, to } => cmd_get(&repo, quant, file, all, to),
    }
}

fn cmd_list(repo: &str) -> i32 {
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
            eprintln!("reinfer: model list failed: {e:?}");
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
    to: Option<PathBuf>,
) -> i32 {
    let resolver = match ModelResolver::from_env() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("reinfer: {e:?}");
            return 1;
        }
    };
    let dir = to.unwrap_or_else(|| resolver.dir.clone());
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
            eprintln!("reinfer: model list failed: {e:?}");
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
fn print_manifest_line(dir: &std::path::Path, path: &std::path::Path) {
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
    fn parse_list_ok() {
        let a: Vec<String> = ["list", "Qwen/Qwen2.5-0.5B-Instruct-GGUF"].map(String::from).to_vec();
        assert_eq!(
            parse_model(&a).unwrap(),
            ModelCmd::List { repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF".into() }
        );
        assert!(parse_model(&["list".into(), "a/b".into(), "extra".into()]).is_err());
        assert!(parse_model(&["list".into()]).is_err());
    }

    #[test]
    fn parse_get_ok() {
        let a: Vec<String> =
            ["get", "a/b", "--quant", "q8_0", "--to", "/tmp/m"].map(String::from).to_vec();
        assert_eq!(
            parse_model(&a).unwrap(),
            ModelCmd::Get {
                repo: "a/b".into(),
                quant: Some("q8_0".into()),
                file: None,
                all: false,
                to: Some(PathBuf::from("/tmp/m")),
            }
        );
        // --file 精确名
        let b: Vec<String> = ["get", "a/b", "--file", "x.gguf"].map(String::from).to_vec();
        assert_eq!(
            parse_model(&b).unwrap(),
            ModelCmd::Get {
                repo: "a/b".into(),
                quant: None,
                file: Some("x.gguf".into()),
                all: false,
                to: None
            }
        );
        // --all
        let c: Vec<String> = ["get", "a/b", "--all"].map(String::from).to_vec();
        match parse_model(&c).unwrap() {
            ModelCmd::Get { all, .. } => assert!(all),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn parse_get_errors() {
        // 互斥
        let args: Vec<String> =
            ["get", "a/b", "--quant", "q8", "--file", "x"].map(String::from).to_vec();
        assert!(parse_model(&args).is_err());
        let args: Vec<String> = ["get", "a/b", "--all", "--quant", "q8"].map(String::from).to_vec();
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
        // 直接验证解析层；exit code 路径由 run() 返回
        assert_eq!(run(&args), EXIT_USAGE);
    }
}

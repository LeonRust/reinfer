//! reinfer —— 支持 CUDA / 昇腾 CANN 的 Rust 推理引擎 CLI（bin 侧）。
//!
//! 命令面（契约：docs/design/cli-contract-2026-08-27.md）：
//! - `download <repo> [file...]`：hf download 语义；`-q/--quant`（与 file.../--include 互斥）、
//!   `--include/--exclude`（fnmatch）、`--revision`、`--local-dir`、`--dry-run`、`--format
//!   table|json|quiet`、`--quiet`；固定 4 worker 并发；TTY 下两层进度条（文件条 + GLOBAL 条）。
//! - `model list [--format]`：本地清单（格式无关：非隐藏/非 manifest/非 *.tmp-* 均列出）。
//! - `completions <bash|zsh|fish>`：clap_complete 生成补全脚本（不再手写模板）。
//! - `doctor [--backend auto|cuda|ascend] [--format table|json|quiet] [--net]`：环境体检。
//!
//! 解析 = clap derive（契约 v2.5 定稿）：用法错误 exit 2（clap 默认）；执行失败 exit 1；
//! 全局旗（-v/-vv/--debug/--no-color/-V/-h）仅命令前（gh 惯例，不设 clap global=true）；
//! `-q` 是 quant（quiet 仅 `--format quiet`/`--quiet` 长旗）；`--` 分隔（dash 开头文件名）。

use clap::{ArgAction, Args, CommandFactory, Parser, Subcommand, ValueEnum};
use reinfer_models::api::FileEntry;
use reinfer_models::download::{MANIFEST, Verify, local_hit, read_manifest, target_path};
use reinfer_models::{LaunchError, ModelResolver, ModelSource, ModelSpec};
use std::collections::VecDeque;
use std::fmt;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// 并发 worker 数（modelscope SDK `max_workers=4` 先例；契约 §2 不增 CLI 旗子）。
const MAX_WORKERS: usize = 4;
/// 进度条刷新：每 256KB 或 200ms（契约 §4）。
const DRAW_BYTES: u64 = 256 * 1024;
const DRAW_TICK: Duration = Duration::from_millis(200);

const AFTER_HELP: &str = "\
EXAMPLES:
  reinfer model list
  reinfer download Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q8_0
  reinfer download <repo> --include '*.json' --revision master --local-dir ~/models
  reinfer download <repo> -q q8_0 --dry-run
  reinfer doctor

ENV (source policy):
  REINFER_MODEL_SOURCE=modelscope|huggingface|auto (default auto = ModelScope first)
  REINFER_MODEL_DIR, REINFER_MODEL_VERIFY=sha256|size|none, REINFER_MODEL_AUTODOWNLOAD=on|off
  REINFER_CUDA_ARCH, REINFER_CUDA_NVCC, REINFER_JIT_*; RUST_LOG (when -v/--debug absent)
  standard HTTP_PROXY/HTTPS_PROXY/NO_PROXY (e.g. http://192.168.0.1:7890)";

/// 输出格式（`--format table|json|quiet`；契约 §2/§3）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Format {
    /// 人类表格（缺省）。
    Table,
    /// 机器可读 JSON。
    Json,
    /// 每成功文件一行路径。
    Quiet,
}

impl FromStr for Format {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "table" => Ok(Self::Table),
            "json" => Ok(Self::Json),
            "quiet" => Ok(Self::Quiet),
            other => Err(format!("unknown format '{other}' (table|json|quiet)")),
        }
    }
}

/// doctor 后端栈（`--backend auto|cuda|ascend`；auto = 双栈，缺省）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Backend {
    Auto,
    Cuda,
    Ascend,
}

impl FromStr for Backend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "auto" => Ok(Self::Auto),
            "cuda" => Ok(Self::Cuda),
            "ascend" => Ok(Self::Ascend),
            other => Err(format!("unknown backend '{other}' (auto|cuda|ascend)")),
        }
    }
}

/// 补全目标 shell（契约 v2.4：bash|zsh|fish 三件）。
#[derive(ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
enum ShellKind {
    Bash,
    Zsh,
    Fish,
}

impl ShellKind {
    fn to_complete_shell(self) -> clap_complete::Shell {
        match self {
            Self::Bash => clap_complete::Shell::Bash,
            Self::Zsh => clap_complete::Shell::Zsh,
            Self::Fish => clap_complete::Shell::Fish,
        }
    }
}

// ---------------------------------------------------------------------------
// clap 定义
// ---------------------------------------------------------------------------

/// 根命令。全局旗仅命令前（gh 惯例；契约 §5——不设 clap `global=true`）。
#[derive(Parser, Debug)]
#[command(
    name = "reinfer",
    version,
    about = "reinfer - CUDA/CANN inference engine CLI (download | model list | completions | doctor)",
    override_usage = "reinfer <command> [args]",
    after_help = AFTER_HELP,
    arg_required_else_help = true,
)]
struct Cli {
    /// Increase diagnostic detail on stderr (-v details, -vv per-step trace)
    #[arg(short = 'v', action = ArgAction::Count)]
    verbose: u8,

    /// Full diagnostics including library-level trace (implies -vv)
    #[arg(long = "debug")]
    debug: bool,

    /// Disable colored output (accepted for compatibility; no color is emitted)
    #[arg(long = "no-color")]
    no_color: bool,

    #[command(subcommand)]
    command: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Download model files from a repo (hf download semantics)
    Download(DownloadArgs),
    /// Local model inventory (docker image ls / ollama list style)
    Model {
        #[command(subcommand)]
        sub: ModelCmd,
    },
    /// Generate shell completion scripts (bash | zsh | fish)
    Completions(CompletionsArgs),
    /// Environment doctor (flutter/cargo doctor style)
    Doctor(DoctorArgs),
}

#[derive(Subcommand, Debug)]
enum ModelCmd {
    /// List locally downloaded model files
    List(ListArgs),
}

#[derive(Args, Debug)]
struct DownloadArgs {
    /// Model repository (owner/model)
    repo: String,

    /// Exact files to download (mutually exclusive with -q/--quant and --include)
    #[arg(value_name = "FILE", conflicts_with_all = ["quant", "include"])]
    files: Vec<String>,

    /// Quant tag (e.g. q8_0) -> file matching *-q8_0.* (mutually exclusive with file.../--include)
    #[arg(short = 'q', long = "quant", value_name = "TAG", conflicts_with_all = ["files", "include"])]
    quant: Option<String>,

    /// Include files matching this glob (fnmatch: '*' crosses '/')
    #[arg(long = "include", value_name = "GLOB", conflicts_with_all = ["files", "quant"])]
    include: Option<String>,

    /// Exclude files matching this glob (requires --include)
    #[arg(long = "exclude", value_name = "GLOB", requires = "include")]
    exclude: Option<String>,

    /// Branch/tag/commit ref (default: source default branch)
    #[arg(long = "revision", value_name = "REF")]
    revision: Option<String>,

    /// Target directory (default: $REINFER_MODEL_DIR or ~/.reinfer/models)
    #[arg(long = "local-dir", value_name = "DIR")]
    local_dir: Option<PathBuf>,

    /// List the download plan only; write nothing (dry run)
    #[arg(long = "dry-run")]
    dry_run: bool,

    /// Output format (table|json|quiet)
    #[arg(long = "format", value_name = "FORMAT", value_parser = Format::from_str)]
    format: Option<Format>,

    /// One path line per successful file (long-only; '-q' is quant)
    #[arg(long = "quiet")]
    quiet: bool,
}

impl DownloadArgs {
    /// 有效输出格式（--quiet 与 --format quiet 等价；契约 §2）。
    fn effective_format(&self) -> Format {
        if self.quiet { Format::Quiet } else { self.format.unwrap_or(Format::Table) }
    }

    /// `reinfer download help` 形态（契约 §5：子命令 help；clap 内置 help 子命令不覆盖叶子命令）。
    fn is_help_form(&self) -> bool {
        self.repo == "help"
            && self.files.is_empty()
            && self.quant.is_none()
            && self.include.is_none()
            && self.exclude.is_none()
            && self.revision.is_none()
            && self.local_dir.is_none()
            && !self.dry_run
            && self.format.is_none()
            && !self.quiet
    }
}

#[derive(Args, Debug)]
struct ListArgs {
    /// Output format (table|json|quiet)
    #[arg(long = "format", value_name = "FORMAT", value_parser = Format::from_str)]
    format: Option<Format>,
}

#[derive(Args, Debug)]
struct CompletionsArgs {
    /// Target shell
    #[arg(value_enum)]
    shell: ShellKind,
}

#[derive(Args, Debug)]
struct DoctorArgs {
    /// Backend stack to check (auto = both, default)
    #[arg(long = "backend", value_name = "BACKEND", value_parser = Backend::from_str, default_value = "auto")]
    backend: Backend,

    /// Output format (table|json|quiet)
    #[arg(long = "format", value_name = "FORMAT", value_parser = Format::from_str)]
    format: Option<Format>,

    /// Probe ModelScope/HF connectivity (offline by default)
    #[arg(long = "net")]
    net: bool,

    /// Leaf help sentinel (`doctor help`; hidden, literal-only)
    #[arg(id = "doctor_help", value_name = "help", value_parser = clap::builder::PossibleValuesParser::new(["help"]), hide = true)]
    help: Option<String>,
}

impl DoctorArgs {
    /// `reinfer doctor help` 形态（契约 §5：子命令 help；doctor 无位置参数，故解析层用
    /// 隐藏位置参数接住字面 help——其他值仍按未知参数拒绝）。
    fn is_help_form(&self) -> bool {
        self.help.as_deref() == Some("help")
            && self.backend == Backend::Auto
            && self.format.is_none()
            && !self.net
    }
}

// ---------------------------------------------------------------------------
// 入口 / 诊断级别
// ---------------------------------------------------------------------------

fn main() {
    // 开发测试环境：本目录 .env（gitignored；模板 .env.example）；无文件静默跳过
    let _ = dotenvy::dotenv();
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(e) => {
            let kind = e.kind();
            let code = e.exit_code();
            let _ = e.print();
            // 契约 §5：unknown 命令补 `try reinfer help` 指引（clap 自带 usage 行）
            if matches!(kind, clap::error::ErrorKind::InvalidSubcommand) {
                eprintln!("try `reinfer help` for usage");
            }
            std::process::exit(code);
        }
    };
    std::process::exit(run(cli));
}

/// 诊断级别（契约 v2.6：CLI -v/-vv/--debug 优先，否则读 RUST_LOG；全部走 stderr）。
#[derive(Debug, Clone, Copy)]
struct Verbosity(u8);

impl Verbosity {
    fn new(verbose: u8, debug: bool) -> Self {
        let level = if debug {
            3
        } else if verbose > 0 {
            verbose.min(3)
        } else {
            rust_log_level()
        };
        Self(level)
    }

    /// level 1 = 详情；2 = 每步跟踪；3 = 全量。
    fn at(&self, level: u8) -> bool {
        self.0 >= level
    }

    fn log(&self, level: u8, msg: impl fmt::Display) {
        if self.at(level) {
            eprintln!("reinfer: {msg}");
        }
    }
}

/// RUST_LOG 生态面（v2.6：日志分级缺省来源；`RUST_LOG=info|debug|trace` 等）。
fn rust_log_level() -> u8 {
    let s = std::env::var("RUST_LOG").unwrap_or_default().to_ascii_lowercase();
    if s.contains("trace") {
        3
    } else if s.contains("debug") {
        2
    } else if s.contains("info") {
        1
    } else {
        0
    }
}

fn run(cli: Cli) -> i32 {
    let vlog = Verbosity::new(cli.verbose, cli.debug);
    if vlog.at(2) {
        let color = !cli.no_color && std::env::var_os("NO_COLOR").is_none();
        eprintln!("reinfer: diagnostics level={} color={color}", vlog.0);
    }
    match cli.command {
        Cmd::Download(a) => {
            if a.is_help_form() {
                print_leaf_help("download");
                return 0;
            }
            cmd_download(a, &vlog)
        }
        Cmd::Model { sub } => match sub {
            ModelCmd::List(a) => cmd_list_local(a, &vlog),
        },
        Cmd::Completions(a) => cmd_completions(a),
        Cmd::Doctor(a) => {
            if a.is_help_form() {
                print_leaf_help("doctor");
                return 0;
            }
            cmd_doctor(a, &vlog)
        }
    }
}

/// 叶子命令 help（`reinfer <cmd> help`；契约 §5 子命令 help）。
fn print_leaf_help(cmd_name: &str) {
    if let Some(sub) = Cli::command().find_subcommand_mut(cmd_name) {
        let mut out = Vec::new();
        let _ = sub.write_long_help(&mut out);
        print!("{}", String::from_utf8_lossy(&out));
    }
}

// ---------------------------------------------------------------------------
// completions
// ---------------------------------------------------------------------------

/// completions <bash|zsh|fish>：clap_complete 生成（契约 v2.4；stdout，可 source 进配置）。
fn cmd_completions(a: CompletionsArgs) -> i32 {
    let mut cmd = Cli::command();
    let mut buf = Vec::new();
    clap_complete::generate(a.shell.to_complete_shell(), &mut cmd, "reinfer", &mut buf);
    // 缓冲后一次性写出；管道关闭（EPIPE）静默忽略——`completions fish | head -1` 不应 panic
    let _ = std::io::stdout().write_all(&buf);
    0
}

// ---------------------------------------------------------------------------
// model list
// ---------------------------------------------------------------------------

/// model list：本地已下载清单（docker/ollama 惯例；零参默认本地）。
fn cmd_list_local(a: ListArgs, vlog: &Verbosity) -> i32 {
    let fmt = a.format.unwrap_or(Format::Table);
    let resolver = match ModelResolver::from_env() {
        Ok(r) => r,
        Err(_) => return 1,
    };
    let dir = resolver.dir.clone();
    vlog.log(1, format!("model dir: {}", dir.display()));
    // 按 repo 组织（root/{owner}/{model}/…）：递归收集 (相对路径, 大小, repo 目录串)
    let mut found: Vec<(String, u64, String)> = Vec::new();
    if let Err(err) = collect_model_files(&dir, Path::new(""), &mut found) {
        eprintln!("reinfer: model dir not readable: {} ({err})", dir.display());
        return 1;
    }
    // manifest 关联：per-repo（root/{repo}/manifest.json）；文件名为末段
    let mut rows: Vec<(String, u64, Option<String>, String)> = found
        .iter()
        .map(|(rel, size, repo)| {
            let man =
                if repo.is_empty() { Vec::new() } else { read_manifest(&dir.join(repo)) };
            let fname = rel.rsplit('/').next().unwrap_or(rel);
            let m = man.iter().find(|e| e.name == fname);
            (
                rel.clone(),
                *size,
                m.and_then(|e| e.sha256.clone()),
                m.map(|e| format!("{}@{}", e.repo, e.branch)).unwrap_or_else(|| {
                    if repo.is_empty() { "-".to_string() } else { format!("{repo}@-") }
                }),
            )
        })
        .collect();
    rows.sort_by(|a, b| a.0.cmp(&b.0));
    match fmt {
        Format::Table => {
            if rows.is_empty() {
                println!("no local model files in {}", dir.display());
                return 0;
            }
            println!("{} ({} file(s)):\n", dir.display(), rows.len());
            println!("{:<48} {:>12}  {:<16}  source", "name", "size", "sha256");
            for (rel, size, sha, src) in &rows {
                let sha16 = sha.as_deref().map(|s| &s[..s.len().min(16)]).unwrap_or("-");
                println!("{rel:<48} {:>12}  {sha16:<16}  {src}", human_bytes(*size));
            }
        }
        Format::Json => {
            let arr: Vec<serde_json::Value> = rows
                .iter()
                .map(|(rel, size, sha, src)| {
                    serde_json::json!({
                        "name": rel.rsplit('/').next().unwrap_or(rel),
                        "path": rel,
                        "size": size,
                        "sha256": sha,
                        "source": src
                    })
                })
                .collect();
            println!("{}", serde_json::to_string(&arr).unwrap_or_default());
        }
        Format::Quiet => {
            for (rel, _, _, _) in &rows {
                println!("{}", dir.join(rel).display());
            }
        }
    }
    0
}

/// 清单可见性（格式无关；契约 v2.7：非隐藏、非 manifest、非下载临时文件）。
#[cfg(test)]
fn is_listed_file(name: &str) -> bool {
    !name.starts_with('.') && name != MANIFEST && !name.contains(".tmp-")
}

/// 递归收集模型文件（按 repo 组织；排除 manifest/隐藏/tmp；repo=相对目录串）。
fn collect_model_files(
    root: &Path,
    rel: &Path,
    out: &mut Vec<(String, u64, String)>,
) -> std::io::Result<()> {
    let abs = if rel.as_os_str().is_empty() { root.to_path_buf() } else { root.join(rel) };
    for e in std::fs::read_dir(&abs)? {
        let e = e?;
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || name == MANIFEST || name.contains(".tmp-") {
            continue;
        }
        let m = e.metadata()?;
        if m.is_dir() {
            collect_model_files(root, &rel.join(&name), out)?;
        } else {
            let repo = rel.to_string_lossy().replace('\\', "/");
            out.push((format!("{repo}/{name}"), m.len(), repo));
        }
    }
    if rel.as_os_str().is_empty() {
        out.sort_by(|a, b| a.0.cmp(&b.0));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// download
// ---------------------------------------------------------------------------

fn cmd_download(a: DownloadArgs, vlog: &Verbosity) -> i32 {
    let fmt = a.effective_format();
    let resolver = match ModelResolver::from_env() {
        Ok(r) => r,
        Err(_) => return 1,
    };
    let dir = match &a.local_dir {
        Some(d) => expand_tilde(&d.to_string_lossy()),
        None => resolver.dir.clone(),
    };
    vlog.log(
        1,
        format!(
            "source={:?} dir={} verify={:?} autodownload={}",
            resolver.source,
            dir.display(),
            resolver.verify,
            resolver.autodownload
        ),
    );
    // AUTODOWNLOAD=off：纯本地解析，绝不联网（契约纪律 + 手动测试 6.1）
    if !resolver.autodownload {
        return cmd_download_offline(&a, fmt, &dir);
    }
    vlog.log(
        1,
        format!(
            "listing {}{}",
            a.repo,
            a.revision.as_deref().map(|r| format!("@{r}")).unwrap_or_default()
        ),
    );
    let (entries, from_hf) = match list_entries(resolver.source, &a.repo, a.revision.as_deref()) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("reinfer: {e:?}");
            print_proxy_hint();
            return 1;
        }
    };
    let targets = match pick_targets(&a, entries) {
        Ok(t) => t,
        Err(msg) => {
            if !msg.is_empty() {
                eprintln!("reinfer: {msg}");
            }
            return 1;
        }
    };
    if targets.is_empty() {
        eprintln!("reinfer: nothing to download in {}", a.repo);
        return 1;
    }
    let total: u64 = targets.iter().map(|e| e.size).sum();
    vlog.log(1, format!("plan: {} file(s) ({} bytes) to {}", targets.len(), total, dir.display()));
    if a.dry_run {
        return cmd_dry_run(&a, &resolver, &targets, from_hf, &dir, fmt);
    }
    run_downloads(&a, &resolver, &targets, from_hf, &dir, fmt, vlog)
}

/// 按源策略列远端条目（REINFER_MODEL_SOURCE；auto = MS 优先，缺 → HF 回退）。
fn list_entries(
    source: ModelSource,
    repo: &str,
    revision: Option<&str>,
) -> Result<(Vec<FileEntry>, bool), LaunchError> {
    match source {
        ModelSource::Modelscope => ms_list(repo, revision).map(|e| (e, false)),
        ModelSource::Huggingface => hf_list(repo, revision).map(|e| (e, true)),
        ModelSource::Auto => match ms_list(repo, revision) {
            Ok(e) => Ok((e, false)),
            Err(_) => {
                eprintln!("reinfer: ModelScope list failed, falling back to HuggingFace");
                hf_list(repo, revision).map(|e| (e, true))
            }
        },
    }
}

fn ms_list(repo: &str, revision: Option<&str>) -> Result<Vec<FileEntry>, LaunchError> {
    let url = reinfer_models::api::ms_list_url_rev(repo, revision);
    let body = reinfer_models::api::http_get(&url)?;
    reinfer_models::api::parse_ms_files(&body, &url)
}

fn hf_list(repo: &str, _revision: Option<&str>) -> Result<Vec<FileEntry>, LaunchError> {
    let names = reinfer_models::hf::hf_list_files(repo)?;
    Ok(names
        .into_iter()
        .map(|name| FileEntry { name, size: 0, sha256: None, is_lfs: true })
        .collect())
}

/// 目标选择（契约 §2：-q / file... / --include/--exclude / 无选择=整仓库）。
/// 返回按文件名排序的条目（确定性；契约 §3：输出顺序按文件名排序）。
fn pick_targets(a: &DownloadArgs, entries: Vec<FileEntry>) -> Result<Vec<FileEntry>, String> {
    let mut out = if let Some(q) = &a.quant {
        let spec = ModelSpec::new(&a.repo).with_quant(q);
        // 复用 resolver 的远端 quant 选择（任意扩展名——无 .gguf 特判；歧义列候选）
        let name = reinfer_models::resolver::select_name(entries.clone(), &spec)
            .map_err(|_| format!("no model file matches --quant {q} in repo {}", a.repo))?;
        match entries.into_iter().find(|e| e.name == name) {
            Some(e) => vec![e],
            None => return Err(format!("no model file matches --quant {q} in repo {}", a.repo)),
        }
    } else if !a.files.is_empty() {
        entries.into_iter().filter(|e| a.files.contains(&e.name)).collect()
    } else if let Some(inc) = &a.include {
        entries
            .into_iter()
            .filter(|e| {
                glob_match(inc, &e.name)
                    && a.exclude.as_deref().map(|x| !glob_match(x, &e.name)).unwrap_or(true)
            })
            .collect()
    } else {
        entries
    };
    out.sort_by(|x, y| x.name.cmp(&y.name));
    Ok(out)
}

/// --dry-run：只列计划不写任何文件（契约 §3；已命中按 verify 深度 local_hit 判定）。
fn cmd_dry_run(
    a: &DownloadArgs,
    resolver: &ModelResolver,
    targets: &[FileEntry],
    from_hf: bool,
    dir: &Path,
    fmt: Format,
) -> i32 {
    let mut targets = targets.to_vec();
    if from_hf {
        // HF 列表无 size 字段——HEAD 补全（计划准确性）
        let branch = a.revision.clone().unwrap_or_else(|| "main".to_string());
        for e in &mut targets {
            if let Ok(head) = reinfer_models::hf::hf_head(&a.repo, &e.name, &branch) {
                e.size = head.size;
            }
        }
    }
    // HF 源 sha256 校验自动降级为 size（hf_fetch 同语义）
    let verify =
        if from_hf && resolver.verify == Verify::Sha256 { Verify::Size } else { resolver.verify };
    let total: u64 = targets.iter().map(|e| e.size).sum();
    let already: Vec<bool> = targets
        .iter()
        .map(|e| local_hit(&target_path(dir, &a.repo, &e.name), e, verify))
        .collect();
    match fmt {
        Format::Table => {
            println!(
                "[dry-run] will download {} file(s) ({}) to {}",
                targets.len(),
                human_bytes(total),
                dir.display()
            );
            println!("{:<48} {:>12}  already", "name", "bytes");
            for (e, a) in targets.iter().zip(&already) {
                println!(
                    "{:<48} {:>12}  {}",
                    e.name,
                    human_bytes(e.size),
                    if *a { "yes" } else { "no" }
                );
            }
        }
        Format::Json => {
            let files: Vec<serde_json::Value> = targets
                .iter()
                .zip(&already)
                .map(|(e, a)| {
                    serde_json::json!({
                        "file": dir.join(&e.name).display().to_string(),
                        "bytes": e.size,
                        "sha256": e.sha256,
                        "downloaded": false,
                        "already": a,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({"dry_run": true, "files": files}))
                    .unwrap_or_default()
            );
        }
        Format::Quiet => {
            for (e, a) in targets.iter().zip(&already) {
                let p = dir.join(&e.name);
                if *a {
                    println!("{} (already)", p.display());
                } else {
                    println!("{}", p.display());
                }
            }
        }
    }
    0
}

/// 实际下载：固定 4 worker 并发（文件间并行）+ TTY 两层进度条。
fn run_downloads(
    a: &DownloadArgs,
    resolver: &ModelResolver,
    targets: &[FileEntry],
    from_hf: bool,
    dir: &Path,
    fmt: Format,
    vlog: &Verbosity,
) -> i32 {
    if let Err(e) = std::fs::create_dir_all(dir) {
        eprintln!("reinfer: cannot create {}: {e}", dir.display());
        return 1;
    }
    let n = targets.len();
    let tty = fmt == Format::Table && std::io::stdout().is_terminal();
    let prog = tty.then(|| Arc::new(Progress::new(targets)));
    if prog.is_some() {
        sigint::install();
    }
    let workers = n.min(MAX_WORKERS);
    let next = AtomicUsize::new(0);
    let results: Arc<Mutex<Vec<Option<DownloadOutcome>>>> = Arc::new(Mutex::new(vec![None; n]));
    let ctx = DownloadCtx {
        resolver,
        a,
        from_hf,
        hf_branch: a.revision.clone().unwrap_or_else(|| "main".to_string()),
        hf_verify: if resolver.verify == Verify::Sha256 { Verify::Size } else { resolver.verify },
        dir,
    };
    let (next, results, prog, targets, vlog, ctx) = (&next, &results, &prog, targets, vlog, &ctx);
    std::thread::scope(|s| {
        for w in 0..workers {
            s.spawn(move || {
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= n {
                        break;
                    }
                    let entry = &targets[i];
                    vlog.log(2, format!("-> {}", entry.name));
                    if let Some(p) = prog {
                        p.begin_file(w, entry, i + 1, n);
                    }
                    let res = if let Some(p) = prog {
                        let p = Arc::clone(p);
                        let cb = move |bytes: u64, total: u64| p.update(w, bytes, total);
                        download_one(ctx, entry, Some(&cb))
                    } else {
                        download_one(ctx, entry, None)
                    };
                    if let Some(p) = prog {
                        p.end_file(w, res.is_ok());
                    }
                    vlog.log(
                        2,
                        format!(
                            "   {} ({})",
                            entry.name,
                            if res.is_ok() { "done" } else { "failed" }
                        ),
                    );
                    if let Ok(mut guard) = results.lock() {
                        guard[i] = Some(res);
                    }
                }
            });
        }
    });
    if let Some(p) = &prog {
        p.finish();
    }
    // 汇总（顺序 = 文件名排序；并行完成后统一打印——确定性）
    let taken: Vec<Result<PathBuf, String>> = results
        .lock()
        .map(|mut g| {
            g.drain(..)
                .map(|r| r.unwrap_or_else(|| Err("worker did not report".to_string())))
                .collect()
        })
        .unwrap_or_default();
    let ok = taken.iter().filter(|r| r.is_ok()).count();
    match fmt {
        Format::Table => {
            for (e, r) in targets.iter().zip(&taken) {
                println!("-> {}", e.name);
                if let Ok(p) = r {
                    println!("   {} (done)", p.display());
                }
            }
            println!("{ok}/{} downloaded", targets.len());
        }
        Format::Json => {
            let files: Vec<serde_json::Value> = targets
                .iter()
                .zip(&taken)
                .map(|(e, r)| {
                    serde_json::json!({
                        "file": dir.join(&e.name).display().to_string(),
                        "bytes": e.size,
                        "sha256": e.sha256,
                        "downloaded": r.is_ok(),
                        "error": r.as_ref().err(),
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({"files": files})).unwrap_or_default()
            );
        }
        Format::Quiet => {
            for (e, r) in targets.iter().zip(&taken) {
                if r.is_ok() {
                    println!("{}", dir.join(&e.name).display());
                }
            }
        }
    }
    for (e, r) in targets.iter().zip(&taken) {
        if let Err(msg) = r {
            eprintln!("reinfer: download {} failed: {msg}", e.name);
        }
    }
    if ok == n { 0 } else { 1 }
}

/// 单文件下载结果（成功 → 落盘路径；失败 → 消息）。
type DownloadOutcome = Result<PathBuf, String>;

/// 单文件下载的共享上下文（避免超参堆积）。
struct DownloadCtx<'a> {
    resolver: &'a ModelResolver,
    a: &'a DownloadArgs,
    from_hf: bool,
    hf_branch: String,
    hf_verify: Verify,
    dir: &'a Path,
}

/// 下载一个文件（MS 路径带进度；auto 时 MS 失败 → HF 重试——与 resolver::fetch 同语义）。
fn download_one(
    ctx: &DownloadCtx,
    entry: &FileEntry,
    progress: Option<&dyn Fn(u64, u64)>,
) -> Result<PathBuf, String> {
    if ctx.from_hf {
        // HF 源：hf.rs 尚不暴露进度回调位（bin 侧按需接入）
        return reinfer_models::hf::hf_download_file(
            &ctx.a.repo,
            &entry.name,
            &ctx.hf_branch,
            ctx.dir,
            ctx.hf_verify,
        )
        .map_err(|e| format!("{e:?}"));
    }
    match reinfer_models::download_file(
        &ctx.a.repo,
        entry,
        ctx.dir,
        ctx.resolver.verify,
        ctx.a.revision.as_deref(),
        progress,
    ) {
        Ok(p) => Ok(p),
        Err(_) if ctx.resolver.source == ModelSource::Auto => {
            eprintln!("reinfer: ModelScope miss, falling back to HuggingFace for {}", entry.name);
            reinfer_models::hf::hf_download_file(
                &ctx.a.repo,
                &entry.name,
                &ctx.hf_branch,
                ctx.dir,
                ctx.hf_verify,
            )
            .map_err(|e| format!("{e:?}"))
        }
        Err(e) => Err(format!("{e:?}")),
    }
}

/// AUTODOWNLOAD=off：纯本地解析（契约纪律——off 绝不联网；手动测试 6.1）。
fn cmd_download_offline(a: &DownloadArgs, fmt: Format, dir: &Path) -> i32 {
    let local: Vec<String> = std::fs::read_dir(dir)
        .map(|it| it.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect())
        .unwrap_or_default();
    let mut names: Vec<String> = if let Some(q) = &a.quant {
        local.into_iter().filter(|n| n.contains(&format!("-{q}."))).collect()
    } else if !a.files.is_empty() {
        let missing: Vec<&str> =
            a.files.iter().filter(|f| !local.contains(f)).map(String::as_str).collect();
        if !missing.is_empty() {
            eprintln!(
                "reinfer: {} not found locally and REINFER_MODEL_AUTODOWNLOAD=off - refusing to dial out",
                missing.join(", ")
            );
            return 1;
        }
        a.files.clone()
    } else {
        eprintln!(
            "reinfer: cannot select files from {} without network (REINFER_MODEL_AUTODOWNLOAD=off); use -q <tag> or explicit file names",
            a.repo
        );
        return 1;
    };
    names.sort();
    if names.is_empty() {
        eprintln!("reinfer: nothing to download in {} (REINFER_MODEL_AUTODOWNLOAD=off)", a.repo);
        return 1;
    }
    let targets: Vec<FileEntry> = names
        .iter()
        .filter_map(|n| {
            std::fs::metadata(dir.join(n)).ok().map(|m| FileEntry {
                name: n.clone(),
                size: m.len(),
                sha256: None,
                is_lfs: false,
            })
        })
        .collect();
    if targets.is_empty() {
        eprintln!("reinfer: nothing to download in {} (REINFER_MODEL_AUTODOWNLOAD=off)", a.repo);
        return 1;
    }
    // 全部为本地命中（不存在失败可能）
    match fmt {
        Format::Table => {
            for e in &targets {
                println!("-> {}", e.name);
                println!("   {} (done)", dir.join(&e.name).display());
            }
            println!("{}/{} downloaded", targets.len(), targets.len());
        }
        Format::Json => {
            let files: Vec<serde_json::Value> = targets
                .iter()
                .map(|e| {
                    serde_json::json!({
                        "file": dir.join(&e.name).display().to_string(),
                        "bytes": e.size,
                        "sha256": null,
                        "downloaded": true,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({"files": files})).unwrap_or_default()
            );
        }
        Format::Quiet => {
            for e in &targets {
                println!("{}", dir.join(&e.name).display());
            }
        }
    }
    0
}

// ---------------------------------------------------------------------------
// 进度条（契约 §4：两层；TTY-only；纯 std 绘制 \r+\x1b[2K+\x1b[A）
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum SlotStatus {
    /// 槽已建但还未取到文件（防首帧空名）。
    Waiting,
    Active,
    Done,
    Failed,
}

struct Slot {
    name: String,
    idx: usize,
    bytes: u64,
    total: u64,
    last: u64,
    status: SlotStatus,
    wb: u64,
    wt: Instant,
}

struct Global {
    done: i64,
    total: u64,
}

struct ProgressInner {
    slots: Vec<Slot>,
    files_total: usize,
    global: Global,
    last_draw: Instant,
    last_draw_done: u64,
    samples: VecDeque<(Instant, u64)>,
    /// 已绘制的总行数（含 GLOBAL）；首帧为 0 → 原地画，后续帧回撤重写。
    drawn_rows: usize,
}

/// 两层进度：文件条（每 worker 一行）+ GLOBAL 条（底行固定）。
/// 刷新节流：每 256KB 或 200ms（契约 §4）；校验重试时该文件进度从 0 重计
/// （回调值为本次 download_file 累计——delta 可为负，GLOBAL 相应回退）。
struct Progress {
    inner: Mutex<ProgressInner>,
}

impl Progress {
    fn new(targets: &[FileEntry]) -> Self {
        let workers = targets.len().min(MAX_WORKERS);
        let total: u64 = targets.iter().map(|e| e.size).sum();
        Self {
            inner: Mutex::new(ProgressInner {
                slots: (0..workers)
                    .map(|_| Slot {
                        name: String::new(),
                        idx: 0,
                        bytes: 0,
                        total: 0,
                        last: 0,
                        status: SlotStatus::Waiting,
                        wb: 0,
                        wt: Instant::now(),
                    })
                    .collect(),
                files_total: targets.len(),
                global: Global { done: 0, total },
                last_draw: Instant::now(),
                last_draw_done: 0,
                samples: VecDeque::new(),
                drawn_rows: 0,
            }),
        }
    }

    /// worker 开始下载一个文件（重置该槽进度）。
    fn begin_file(&self, worker: usize, entry: &FileEntry, idx: usize, _m: usize) {
        let Ok(mut g) = self.inner.lock() else { return };
        let s = &mut g.slots[worker];
        s.name = entry.name.clone();
        s.idx = idx;
        s.bytes = 0;
        s.last = 0;
        s.total = entry.size;
        s.status = SlotStatus::Active;
        s.wb = 0;
        s.wt = Instant::now();
        self.draw(&mut g);
    }

    /// 进度回调（models download_file 的 progress 参数进入点）。
    fn update(&self, worker: usize, bytes: u64, total: u64) {
        let Ok(mut g) = self.inner.lock() else { return };
        let s = &mut g.slots[worker];
        let delta = bytes as i64 - s.last as i64;
        s.last = bytes;
        s.bytes = bytes;
        if total > 0 {
            s.total = total;
        }
        g.global.done += delta;
        let done = g.global.done.max(0) as u64;
        let now = Instant::now();
        if now.duration_since(g.last_draw) >= DRAW_TICK
            || done.saturating_sub(g.last_draw_done) >= DRAW_BYTES
        {
            self.draw(&mut g);
            g.last_draw = now;
            g.last_draw_done = done;
        }
    }

    /// worker 结束一个文件（done/failed 标记）。
    fn end_file(&self, worker: usize, ok: bool) {
        let Ok(mut g) = self.inner.lock() else { return };
        let s = &mut g.slots[worker];
        s.status = if ok { SlotStatus::Done } else { SlotStatus::Failed };
        if ok {
            s.bytes = s.total;
        }
        self.draw(&mut g);
        g.last_draw = Instant::now();
        g.last_draw_done = g.global.done.max(0) as u64;
    }

    /// 收尾：最终帧保留 ~1s，然后把光标落到条区外一行（后续摘要行在此打印，不交叠）。
    fn finish(&self) {
        if let Ok(mut g) = self.inner.lock() {
            self.draw(&mut g);
        }
        std::thread::sleep(Duration::from_millis(1000));
        println!(); // 落出条区（保留最后帧在屏幕）
        let _ = std::io::stdout().flush();
    }

    /// 重绘全部行（行数 = slots + GLOBAL）。
    ///
    /// 基线控制（修复"每帧下移漂移"）：首帧在当前位置原地画；此后每帧先向上回撤
    /// 上次绘制的行数（drawn_rows 行），清行重写。结束光标停留在条区下方一行。
    fn draw(&self, g: &mut ProgressInner) {
        let now = Instant::now();
        let done = g.global.done.max(0) as u64;
        // 2s 滑窗速度采样（GLOBAL）
        g.samples.push_back((now, done));
        while let Some((t, _)) = g.samples.front() {
            if now.duration_since(*t) > Duration::from_secs(2) {
                g.samples.pop_front();
            } else {
                break;
            }
        }
        let gspeed = if g.samples.len() >= 2 {
            let (t0, d0) = g.samples.front().copied().expect("len>=2");
            let (t1, d1) = g.samples.back().copied().expect("len>=2");
            let dt = t1.duration_since(t0).as_secs_f64();
            if dt > 0.0 { (d1 as f64 - d0 as f64) / dt } else { 0.0 }
        } else {
            0.0
        };
        let mut lines: Vec<String> =
            g.slots.iter().map(|s| slot_line(s, g.files_total, now)).collect();
        lines.push(global_line(&g.global, gspeed));
        for s in &mut g.slots {
            s.wb = s.bytes;
            s.wt = now;
        }
        let mut out = String::new();
        if g.drawn_rows > 0 {
            // 回撤上次全部行（+1 让光标站到条区首行），重置行首
            out.push_str(&format!("\x1b[{}A\r", g.drawn_rows));
        }
        for (i, line) in lines.iter().enumerate() {
            out.push_str("\x1b[2K");
            out.push_str(line);
            // 行间换行；最后一行留在本行（光标于条区末行，收尾再落出）
            if i + 1 < lines.len() {
                out.push('\n');
            }
        }
        g.drawn_rows = lines.len();
        print!("{out}");
        let _ = std::io::stdout().flush();
    }
}

fn slot_line(s: &Slot, files_total: usize, now: Instant) -> String {
    let name = truncate(&s.name, 44);
    let pct = pct_of(s.bytes, s.total);
    let total_s = if s.total > 0 { human_dec(s.total) } else { "?".to_string() };
    let mut line = format!(
        "[{}/{}] {:<44} {} {:>3}%  {}/{}",
        s.idx,
        files_total,
        name,
        bar(pct),
        pct,
        human_dec(s.bytes),
        total_s
    );
    match s.status {
        SlotStatus::Done => line.push_str("  (done)"),
        SlotStatus::Failed => line.push_str("  (failed)"),
        SlotStatus::Waiting => line.push_str("  waiting…"),
        SlotStatus::Active => {
            if s.total > 0 {
                let dt = now.duration_since(s.wt).as_secs_f64();
                let speed =
                    if dt >= 0.05 && s.bytes >= s.wb { (s.bytes - s.wb) as f64 / dt } else { 0.0 };
                line.push_str(&format!("  {}/s", human_dec(speed as u64)));
                let eta = if speed > 0.0 {
                    format_eta((s.total - s.bytes) as f64 / speed)
                } else {
                    "-".to_string()
                };
                line.push_str(&format!("  ETA {eta}"));
            }
        }
    }
    line
}

fn global_line(g: &Global, speed: f64) -> String {
    let done = g.done.max(0) as u64;
    let pct = pct_of(done, g.total);
    let mut line =
        format!("GLOBAL {} {:>3}%  {}/{}", bar(pct), pct, human_dec(done), human_dec(g.total));
    if g.total > 0 && done >= g.total {
        line.push_str("  (done)");
    } else if g.total > 0 {
        line.push_str(&format!("  {}/s", human_dec(speed as u64)));
        let eta =
            if speed > 0.0 { format_eta((g.total - done) as f64 / speed) } else { "-".to_string() };
        line.push_str(&format!("  ETA {eta}"));
    }
    line
}

/// 20 格进度条（█ 填充 / ░ 空）。
fn bar(pct: u64) -> String {
    let filled = (pct.min(100) as usize * 20) / 100;
    let mut out = String::with_capacity(20);
    for i in 0..20 {
        out.push(if i < filled { '█' } else { '░' });
    }
    out
}

/// 十进制人类大小（进度条用：MB/GB，1-2 位小数；契约 §4 示例风格）。
fn human_dec(b: u64) -> String {
    if b >= 1_000_000_000 {
        format!("{:.2}GB", b as f64 / 1e9)
    } else if b >= 1_000_000 {
        format!("{:.1}MB", b as f64 / 1e6)
    } else if b >= 1_000 {
        format!("{:.0}KB", b as f64 / 1e3)
    } else {
        format!("{b} B")
    }
}

fn format_eta(secs: f64) -> String {
    if secs >= 3600.0 {
        format!("{:.0}h{:.0}m", secs / 3600.0, (secs % 3600.0) / 60.0)
    } else if secs >= 100.0 {
        format!("{:.0}m{:02.0}s", secs / 60.0, secs % 60.0)
    } else {
        format!("{:.1}s", secs)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// 二进制人类大小（清单/计划用：≤1023 B 显示 B；MiB/GiB 两位小数；契约 v2.7）。
fn human_bytes(b: u64) -> String {
    if b <= 1023 {
        format!("{b} B")
    } else if b < (1 << 30) {
        format!("{:.2}MiB", b as f64 / (1 << 20) as f64)
    } else {
        format!("{:.2}GiB", b as f64 / (1 << 30) as f64)
    }
}

// ---------------------------------------------------------------------------
// doctor（契约 §6：CUDA/CANN 设备、nvcc 工具链、模型目录、env 回显、--net 探针）
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Status {
    Ok,
    Warn,
    Fail,
}

struct Check {
    name: String,
    status: Status,
    detail: String,
    fix: Option<String>,
}

impl Check {
    fn ok(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self { name: name.into(), status: Status::Ok, detail: detail.into(), fix: None }
    }
    fn warn(name: impl Into<String>, detail: impl Into<String>, fix: Option<&str>) -> Self {
        Self {
            name: name.into(),
            status: Status::Warn,
            detail: detail.into(),
            fix: fix.map(str::to_string),
        }
    }
    fn fail(name: impl Into<String>, detail: impl Into<String>, fix: Option<&str>) -> Self {
        Self {
            name: name.into(),
            status: Status::Fail,
            detail: detail.into(),
            fix: fix.map(str::to_string),
        }
    }
}

fn cmd_doctor(a: DoctorArgs, vlog: &Verbosity) -> i32 {
    let fmt = a.format.unwrap_or(Format::Table);
    let mut checks: Vec<Check> = Vec::new();
    match a.backend {
        Backend::Auto => {
            cuda_block(&mut checks);
            cann_block(&mut checks);
        }
        Backend::Cuda => cuda_block(&mut checks),
        Backend::Ascend => cann_block(&mut checks),
    }
    model_dir_block(&mut checks);
    env_block(&mut checks, vlog);
    if a.net {
        net_block(&mut checks);
    }
    let blockers = checks.iter().filter(|c| c.status == Status::Fail).count();
    let warns = checks.iter().filter(|c| c.status == Status::Warn).count();
    let backend_name = match a.backend {
        Backend::Auto => "auto",
        Backend::Cuda => "cuda",
        Backend::Ascend => "ascend",
    };
    match fmt {
        Format::Table => {
            for c in &checks {
                let mark = match c.status {
                    Status::Ok => "[✓]",
                    Status::Warn => "[⚠]",
                    Status::Fail => "[✗]",
                };
                println!("{mark} {:<16} {}", c.name, c.detail);
                if let Some(fix) = &c.fix {
                    println!("{:>3} {:<16} fix: {fix}", "", "");
                }
            }
            println!();
            println!("doctor: {blockers} blocker(s), {warns} warning(s)");
        }
        Format::Json => {
            let arr: Vec<serde_json::Value> = checks
                .iter()
                .map(|c| {
                    let status = match c.status {
                        Status::Ok => "ok",
                        Status::Warn => "warn",
                        Status::Fail => "fail",
                    };
                    serde_json::json!({
                        "name": c.name,
                        "status": status,
                        "detail": c.detail,
                        "fix": c.fix,
                    })
                })
                .collect();
            println!(
                "{}",
                serde_json::to_string(&serde_json::json!({
                    "backend": backend_name,
                    "net": a.net,
                    "checks": arr,
                }))
                .unwrap_or_default()
            );
        }
        Format::Quiet => {
            for c in checks.iter().filter(|c| c.status == Status::Fail) {
                println!("[✗] {:<16} {}", c.name, c.detail);
            }
        }
    }
    if blockers > 0 { 1 } else { 0 }
}

/// CUDA 检查块：设备（数量/型号/算力/显存）+ nvcc 工具链（JIT 依赖）。
#[cfg(feature = "cuda")]
fn cuda_block(out: &mut Vec<Check>) {
    use reinfer_cuda::CudaContext;
    match CudaContext::device_count() {
        Ok(n) if n > 0 => {
            for i in 0..n {
                match CudaContext::device_info(i) {
                    Ok(info) => out.push(Check::ok(
                        format!("cuda.device.{i}"),
                        format!(
                            "{} (sm_{}{}, {})",
                            info.name,
                            info.major,
                            info.minor,
                            human_bytes(info.total_mem)
                        ),
                    )),
                    Err(_) => out.push(Check::fail(
                        format!("cuda.device.{i}"),
                        "device info query failed",
                        None,
                    )),
                }
            }
        }
        Ok(_) => out.push(Check::fail(
            "cuda.device",
            "no CUDA device found",
            Some("check nvidia-smi and CUDA_VISIBLE_DEVICES"),
        )),
        Err(_) => out.push(Check::fail(
            "cuda.device",
            "CUDA runtime unavailable (driver?)",
            Some("install the NVIDIA driver / CUDA toolkit"),
        )),
    }
    // 目标架构（REINFER_CUDA_ARCH 或设备 0 实测）→ nvcc 版本梯度
    // （梯度表镜像 jit 的 check_arch_supported；这里内联判定以保持 doctor 被动——库侧
    // check_arch_supported 失败自带 eprintln，医生输出不应被库噪音打断）
    let arch = reinfer_cuda::arch::resolve_arch().ok();
    match arch {
        Some(a) => match reinfer_jit::probe_toolchain() {
            Ok(tc) => {
                let detail = format!("{} at {}", tc.ver_line.trim(), tc.realpath.display());
                let ver = reinfer_jit::toolchain::parse_nvcc_version(&tc.ver_line);
                let min = match a.as_str() {
                    "sm_90" | "sm_90a" => Some((12, 3)),
                    "sm_100" | "sm_100a" | "sm_120" | "sm_120a" => Some((12, 8)),
                    _ => None,
                };
                match (ver, min) {
                    (Some(v), Some((mn, mnr))) if v < (mn, mnr) => out.push(Check::warn(
                        "jit.nvcc",
                        format!("{detail} (too old for {a} - JIT compile would fail)"),
                        Some("install a newer CUDA toolkit (see REINFER_CUDA_NVCC)"),
                    )),
                    _ => out.push(Check::ok("jit.nvcc", detail)),
                }
            }
            Err(_) => out.push(Check::fail(
                "jit.nvcc",
                "nvcc not found (JIT dependency)",
                Some("set REINFER_CUDA_NVCC or CUDA_HOME/CUDA_PATH (or PATH)"),
            )),
        },
        None => match reinfer_jit::resolve_nvcc() {
            Ok(p) => out.push(Check::ok("jit.nvcc", format!("nvcc at {}", p.display()))),
            Err(_) => out.push(Check::fail(
                "jit.nvcc",
                "nvcc not found (JIT dependency)",
                Some("set REINFER_CUDA_NVCC or CUDA_HOME/CUDA_PATH (or PATH)"),
            )),
        },
    }
}

/// 无 cuda feature：警告级（非阻塞——可选后端切换构建品）。
#[cfg(not(feature = "cuda"))]
fn cuda_block(out: &mut Vec<Check>) {
    out.push(Check::warn(
        "cuda.not-built",
        "CUDA feature not compiled in (rebuild with --features cuda)",
        Some("cargo build --features cuda"),
    ));
}

/// CANN 检查块：昇腾设备数 + SoC 名（feature ascend 才编译 C 侧调用）。
#[cfg(feature = "ascend")]
fn cann_block(out: &mut Vec<Check>) {
    match reinfer_ascend::AscendContext::device_count() {
        Ok(n) if n > 0 => {
            let soc = reinfer_ascend::AscendContext::device_info(0)
                .ok()
                .map(|i| i.soc_name)
                .unwrap_or_default();
            let suffix = if soc.is_empty() { String::new() } else { format!(" ({soc})") };
            out.push(Check::ok("cann.device", format!("{n} Ascend device(s){suffix}")));
        }
        Ok(_) => out.push(Check::fail(
            "cann.device",
            "no Ascend (NPU) device found",
            Some("check the NPU driver and DEVICE_ID"),
        )),
        Err(_) => out.push(Check::fail(
            "cann.device",
            "CANN runtime unavailable",
            Some("install CANN (ASCEND_TOOLKIT_HOME) and the NPU driver"),
        )),
    }
}

/// 无 ascend feature：警告级（非阻塞——可选后端切换构建品）。
#[cfg(not(feature = "ascend"))]
fn cann_block(out: &mut Vec<Check>) {
    out.push(Check::warn(
        "cann.not-built",
        "CANN feature not compiled in (rebuild with --features ascend)",
        Some("cargo build --features ascend"),
    ));
}

/// 模型目录：存在/可写/磁盘余量（statvfs 语义经 fs2）。
fn model_dir_block(out: &mut Vec<Check>) {
    let dir = std::env::var("REINFER_MODEL_DIR")
        .ok()
        .map(|d| expand_tilde(&d))
        .unwrap_or_else(reinfer_models::resolver::default_dir);
    if !dir.exists() {
        out.push(Check::warn(
            "model.dir",
            format!("{} does not exist yet", dir.display()),
            Some("it will be created on the first download"),
        ));
        return;
    }
    let probe = dir.join(format!(".doctor-probe-{}", std::process::id()));
    let writable = match std::fs::write(&probe, b"x") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    };
    if !writable {
        out.push(Check::fail(
            "model.dir",
            format!("{} is not writable", dir.display()),
            Some("fix permissions or point REINFER_MODEL_DIR elsewhere"),
        ));
        return;
    }
    let free = fs2::available_space(&dir).ok();
    let suffix = free.map(|f| format!(", {} free", human_bytes(f))).unwrap_or_default();
    out.push(Check::ok("model.dir", format!("{} (writable{suffix})", dir.display())));
    if let Some(f) = free
        && f < (1 << 30)
    {
        out.push(Check::warn(
            "model.space",
            format!("only {} free on {}", human_bytes(f), dir.display()),
            Some("free disk space before large model downloads"),
        ));
    }
}

/// 百分比（total=0 → 0；防溢出 clamped）。
fn pct_of(done: u64, total: u64) -> u64 {
    done.checked_mul(100).and_then(|d| d.checked_div(total)).map_or(0, |p| p.min(100))
}

/// 配置回显：REINFER_* 家族全部当前值 + 来源标注（.env / environment；契约 v2.6）。
fn env_block(out: &mut Vec<Check>, vlog: &Verbosity) {
    let dotenv_keys: std::collections::BTreeSet<String> = dotenvy::dotenv_iter()
        .ok()
        .into_iter()
        .flatten() // Option<Iter> -> Result<(String, String), dotenvy::Error>
        .flatten() // Result -> (String, String)
        .map(|(k, _)| k)
        .collect();
    vlog.log(
        2,
        format!(
            "dotenv keys: {}",
            dotenv_keys.iter().map(String::as_str).collect::<Vec<_>>().join(",")
        ),
    );
    let mut vars: Vec<(String, String)> =
        std::env::vars().filter(|(k, _)| k.starts_with("REINFER_")).collect();
    vars.sort();
    if vars.is_empty() {
        out.push(Check::ok("env.reinfer", "no REINFER_* variables set (all defaults)"));
        return;
    }
    for (k, v) in vars {
        let src = if dotenv_keys.contains(&k) { ".env" } else { "environment" };
        out.push(Check::ok("env.reinfer", format!("{k}={v} ({src})")));
    }
}

/// --net：ModelScope/HF 连通性探针（默认离线；契约 §6）。
fn net_block(out: &mut Vec<Check>) {
    for (name, host) in [("modelscope", "modelscope.cn"), ("huggingface", "huggingface.co")] {
        match tcp_probe(host, 443, Duration::from_secs(5)) {
            Ok(()) => out.push(Check::ok(format!("net.{name}"), format!("{host}:443 reachable"))),
            Err(e) => out.push(Check::fail(
                format!("net.{name}"),
                format!("{host}:443 unreachable ({e})"),
                Some("check network or set HTTPS_PROXY (e.g. http://192.168.0.1:7890)"),
            )),
        }
    }
}

/// 纯 std 连通性探针：有 HTTPS_PROXY/HTTP_PROXY → CONNECT 隧道；否则直连（不做 TLS 握手）。
fn tcp_probe(host: &str, port: u16, timeout: Duration) -> Result<(), String> {
    use std::io::{Read, Write};
    use std::net::{TcpStream, ToSocketAddrs};
    if let Some(proxy) = proxy_url() {
        let (phost, pport) = parse_proxy(&proxy)?;
        let addr = (phost, pport)
            .to_socket_addrs()
            .map_err(|e| e.to_string())?
            .next()
            .ok_or_else(|| "cannot resolve proxy".to_string())?;
        let mut s = TcpStream::connect_timeout(&addr, timeout).map_err(|e| e.to_string())?;
        let _ = s.set_read_timeout(Some(timeout));
        let _ = s.set_write_timeout(Some(timeout));
        write!(s, "CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n\r\n")
            .map_err(|e| e.to_string())?;
        let mut buf = [0u8; 4096];
        let n = s.read(&mut buf).map_err(|e| e.to_string())?;
        let head = String::from_utf8_lossy(&buf[..n]);
        let status = head.lines().next().unwrap_or("");
        if status.starts_with("HTTP/1.1 200") || status.starts_with("HTTP/1.0 200") {
            Ok(())
        } else {
            Err(format!("proxy CONNECT rejected: {status}"))
        }
    } else {
        let mut addrs = (host, port).to_socket_addrs().map_err(|e| e.to_string())?;
        let addr = addrs.next().ok_or_else(|| "cannot resolve host".to_string())?;
        TcpStream::connect_timeout(&addr, timeout).map_err(|e| e.to_string()).map(|_| ())
    }
}

/// 代理 env（HTTPS 优先；http:// 前缀剥离；socks5 不支持）。
fn proxy_url() -> Option<String> {
    for var in ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"] {
        if let Some(v) = std::env::var(var).ok().filter(|v| !v.is_empty()) {
            return Some(v);
        }
    }
    None
}

fn parse_proxy(url: &str) -> Result<(String, u16), String> {
    let rest = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .ok_or_else(|| format!("unsupported proxy scheme in '{url}' (only http://)"))?;
    let (host, port) = match rest.rsplit_once(':') {
        Some((h, p)) => {
            let port: u16 = p.parse().map_err(|_| format!("bad proxy port in '{url}'"))?;
            (h.to_string(), port)
        }
        None => (rest.to_string(), 80),
    };
    Ok((host, port))
}

// ---------------------------------------------------------------------------
// 通用小工具
// ---------------------------------------------------------------------------

/// 极简 glob（`*` `?`；`**` 视作 `*`；`*` 跨 `/`——fnmatch 语义）——--include/--exclude 用。
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

/// `~` 前缀展开（与 crates/models env 语义一致；CLI --local-dir 亦支持）。
fn expand_tilde(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        let home = if cfg!(windows) { "USERPROFILE" } else { "HOME" };
        if let Ok(h) = std::env::var(home) {
            return PathBuf::from(h).join(rest);
        }
    }
    PathBuf::from(p)
}

fn print_proxy_hint() {
    if std::env::var("HTTPS_PROXY").is_err() && std::env::var("https_proxy").is_err() {
        eprintln!(
            "hint: no HTTPS_PROXY set; set it if your network needs a proxy (e.g. http://192.168.0.1:7890)"
        );
    }
}

/// Ctrl-C（仅 TTY 进度模式安装）：保留 temp（不删除——重试纪律），补 \n 留最后一帧后退出。
/// 处理函数只做异步信号安全调用（write + _exit）。
#[cfg(unix)]
mod sigint {
    unsafe extern "C" fn on_sigint(_: libc::c_int) {
        unsafe {
            libc::write(1, b"\n".as_ptr().cast(), 1);
            libc::_exit(130);
        }
    }

    pub fn install() {
        unsafe {
            #[cfg(target_os = "linux")]
            libc::signal(libc::SIGINT, on_sigint as *const () as libc::sighandler_t);
            #[cfg(target_os = "macos")]
            libc::signal(libc::SIGINT, Some(on_sigint));
        }
    }
}

#[cfg(not(unix))]
mod sigint {
    pub fn install() {}
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // 测试断言崩溃即失败
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(args)
    }

    fn dl(args: &[&str]) -> Result<DownloadArgs, clap::Error> {
        parse(&[&["reinfer", "download"], args].concat()).map(|c| match c.command {
            Cmd::Download(a) => a,
            _ => unreachable!(),
        })
    }

    #[test]
    fn parse_download_quant() {
        let a = dl(&["r/e", "-q", "q8_0", "--local-dir", "/tmp/m"]).unwrap();
        assert_eq!(a.repo, "r/e");
        assert_eq!(a.quant.as_deref(), Some("q8_0"));
        assert_eq!(a.local_dir, Some(PathBuf::from("/tmp/m")));
        // 等号形式 -q=q8_0 + --revision=v2 + --local-dir=/tmp
        let b = dl(&["r/e", "-q=q8_0", "--revision=v2", "--local-dir=/tmp"]).unwrap();
        assert_eq!(b.quant.as_deref(), Some("q8_0"));
        assert_eq!(b.revision.as_deref(), Some("v2"));
        assert_eq!(b.local_dir, Some(PathBuf::from("/tmp")));
    }

    #[test]
    fn parse_download_files_and_all() {
        let a = dl(&["r/e", "a.gguf", "config.json"]).unwrap();
        assert_eq!(a.files, vec!["a.gguf".to_string(), "config.json".to_string()]);
        // 无选择 → 全仓库（hf 默认）
        let b = dl(&["r/e"]).unwrap();
        assert!(b.files.is_empty() && b.quant.is_none() && b.include.is_none());
    }

    #[test]
    fn parse_download_patterns_and_mutual_exclusions() {
        let a = dl(&["r/e", "--include", "*.gguf", "--exclude", "*fp16*"]).unwrap();
        assert_eq!(a.include.as_deref(), Some("*.gguf"));
        assert_eq!(a.exclude.as_deref(), Some("*fp16*"));
        // --exclude 须配 --include
        assert!(matches!(
            dl(&["r/e", "--exclude", "*fp16*"]).unwrap_err().kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        ));
        // -q 与 file.../--include 互斥；file... 与 --include 互斥
        assert!(matches!(
            dl(&["r/e", "-q", "q8_0", "--include", "x"]).unwrap_err().kind(),
            clap::error::ErrorKind::ArgumentConflict
        ));
        assert!(matches!(
            dl(&["r/e", "-q", "q8_0", "x.gguf"]).unwrap_err().kind(),
            clap::error::ErrorKind::ArgumentConflict
        ));
        assert!(matches!(
            dl(&["r/e", "x.gguf", "--include", "y"]).unwrap_err().kind(),
            clap::error::ErrorKind::ArgumentConflict
        ));
        // 重复 -q → 冲突错误（契约：不可重复旗）
        assert!(matches!(
            dl(&["r/e", "-q", "q8_0", "-q", "q4_0"]).unwrap_err().kind(),
            clap::error::ErrorKind::ArgumentConflict
        ));
        // 未知旗
        assert!(matches!(
            dl(&["r/e", "--bogus"]).unwrap_err().kind(),
            clap::error::ErrorKind::UnknownArgument
        ));
        // 缺 repo
        assert!(dl(&[]).is_err());
    }

    #[test]
    fn parse_double_dash_filenames() {
        // `--` 后一切为位置参数（dash 开头文件名；POSIX 通则；契约 §5）
        let a = dl(&["r/e", "--", "-dash.gguf"]).unwrap();
        assert_eq!(a.files, vec!["-dash.gguf".to_string()]);
        // 混合：-- 之前仍解析旗
        let b = dl(&["r/e", "--revision", "v2", "--", "-dash.gguf"]).unwrap();
        assert_eq!(b.revision.as_deref(), Some("v2"));
        assert_eq!(b.files, vec!["-dash.gguf".to_string()]);
    }

    #[test]
    fn parse_format_and_quiet() {
        let a = dl(&["r/e", "--format", "json"]).unwrap();
        assert_eq!(a.format, Some(Format::Json));
        let b = dl(&["r/e", "--format=quiet"]).unwrap();
        assert_eq!(b.format, Some(Format::Quiet));
        // --format bogus → 用法错误
        assert!(dl(&["r/e", "--format", "bogus"]).is_err());
        // --quiet 长旗（-q 是 quant，不是 quiet）
        let c = dl(&["r/e", "--quiet"]).unwrap();
        assert!(c.quiet);
        assert_eq!(c.effective_format(), Format::Quiet);
        // --format quiet 与 --quiet 等价
        assert_eq!(b.effective_format(), Format::Quiet);
        // -q q8_0 是 quant（引擎领域词）
        let d = dl(&["r/e", "-q", "q8_0"]).unwrap();
        assert_eq!(d.quant.as_deref(), Some("q8_0"));
        assert!(!d.quiet);
        // --dry-run 解析
        let e = dl(&["r/e", "-q", "q8_0", "--dry-run"]).unwrap();
        assert!(e.dry_run);
    }

    #[test]
    fn parse_global_flags_before_command_only() {
        // 命令前：-v / -vv / --debug / --no-color / -V
        let a = parse(&["reinfer", "-v", "model", "list"]).unwrap();
        assert_eq!(a.verbose, 1);
        let b = parse(&["reinfer", "-vv", "model", "list"]).unwrap();
        assert_eq!(b.verbose, 2);
        assert!(parse(&["reinfer", "--debug", "model", "list"]).unwrap().debug);
        assert!(parse(&["reinfer", "--no-color", "model", "list"]).unwrap().no_color);
        // 命令后 → 未知参数（gh 惯例：全局旗仅命令前）
        assert!(matches!(
            parse(&["reinfer", "model", "list", "-v"]).unwrap_err().kind(),
            clap::error::ErrorKind::UnknownArgument
        ));
        // 根 -q 不存在（1.8 手动测试）
        assert!(matches!(
            parse(&["reinfer", "-q"]).unwrap_err().kind(),
            clap::error::ErrorKind::UnknownArgument
        ));
        // -V / --version：DisplayVersion 且退出码 0
        for args in [&["reinfer", "-V"][..], &["reinfer", "--version"][..]] {
            let e = parse(args).unwrap_err();
            assert_eq!(e.kind(), clap::error::ErrorKind::DisplayVersion);
            assert_eq!(e.exit_code(), 0);
        }
    }

    #[test]
    fn parse_model_and_completions() {
        assert!(parse(&["reinfer", "model", "list"]).is_ok());
        let a = parse(&["reinfer", "model", "list", "--format", "json"]).unwrap();
        match a.command {
            Cmd::Model { sub } => match sub {
                ModelCmd::List(l) => assert_eq!(l.format, Some(Format::Json)),
            },
            _ => unreachable!(),
        }
        // model 缺子命令 → 用法错误
        assert!(parse(&["reinfer", "model"]).is_err());
        for shell in ["bash", "zsh", "fish"] {
            assert!(parse(&["reinfer", "completions", shell]).is_ok());
        }
        // 只支持 bash|zsh|fish（契约 v2.4）
        assert!(matches!(
            parse(&["reinfer", "completions", "powershell"]).unwrap_err().kind(),
            clap::error::ErrorKind::InvalidValue
        ));
    }

    #[test]
    fn parse_doctor_and_unknown_command() {
        assert!(parse(&["reinfer", "doctor"]).is_ok());
        let a = parse(&["reinfer", "doctor", "--backend", "cuda", "--format", "json", "--net"])
            .unwrap();
        match a.command {
            Cmd::Doctor(d) => {
                assert_eq!(d.backend, Backend::Cuda);
                assert_eq!(d.format, Some(Format::Json));
                assert!(d.net);
            }
            _ => unreachable!(),
        }
        assert!(parse(&["reinfer", "doctor", "--backend", "bogus"]).is_err());
        assert!(parse(&["reinfer", "doctor", "--net"]).is_ok());
        // 未知命令 → InvalidSubcommand（exit 2；含 try reinfer help 指引）
        let e = parse(&["reinfer", "frobnicate"]).unwrap_err();
        assert_eq!(e.kind(), clap::error::ErrorKind::InvalidSubcommand);
        assert_eq!(e.exit_code(), 2);
        // 未立项命令不出现（无 ghost）：serve/run/chat/bench 均为未知
        for ghost in ["serve", "run", "chat", "bench"] {
            assert!(matches!(
                parse(&["reinfer", ghost]).unwrap_err().kind(),
                clap::error::ErrorKind::InvalidSubcommand
            ));
        }
    }

    #[test]
    fn parse_leaf_help_forms() {
        // 根与有子命令层级：clap 内置 help 子命令（DisplayHelp 形态，exit 0）
        for args in [
            &["reinfer", "help"][..],
            &["reinfer", "help", "download"][..],
            &["reinfer", "model", "help"][..],
        ] {
            let e = parse(args).unwrap_err();
            assert_eq!(e.kind(), clap::error::ErrorKind::DisplayHelp);
            assert_eq!(e.exit_code(), 0);
        }
        // 叶子 intercept（契约 §5：reinfer <cmd> help）
        let a = dl(&["help"]).unwrap();
        assert!(a.is_help_form());
        let d = match parse(&["reinfer", "doctor", "help"]).unwrap().command {
            Cmd::Doctor(d) => d,
            _ => unreachable!(),
        };
        assert!(d.is_help_form());
        // 带参数时不是 help 形态（"help" 是 repo 名 → 走正常解析）
        let a = dl(&["help", "-q", "q8_0"]).unwrap();
        assert!(!a.is_help_form());
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
    fn is_listed_file_filters() {
        assert!(is_listed_file("foo.gguf"));
        assert!(is_listed_file("foo.safetensors")); // 格式无关（无 .gguf 特判）
        assert!(!is_listed_file(".hidden"));
        assert!(!is_listed_file(MANIFEST));
        assert!(!is_listed_file("x.tmp-1234"));
    }

    #[test]
    fn human_sizes() {
        assert_eq!(human_bytes(1023), "1023 B");
        assert_eq!(human_bytes(1024), "0.00MiB");
        assert_eq!(human_bytes(675_710_816), "644.41MiB");
        assert_eq!(human_bytes(2 << 30), "2.00GiB");
        assert_eq!(human_dec(675_710_816), "675.7MB");
        assert_eq!(human_dec(5_466_559_488), "5.47GB");
        assert_eq!(human_dec(1234), "1KB");
    }

    #[test]
    fn verbosity_and_rust_log() {
        assert_eq!(Verbosity::new(0, false).0, 0);
        assert_eq!(Verbosity::new(1, false).0, 1);
        assert_eq!(Verbosity::new(2, false).0, 2);
        assert_eq!(Verbosity::new(2, true).0, 3); // --debug 高于 -vv
        unsafe {
            std::env::set_var("RUST_LOG", "reinfer=debug");
            assert_eq!(rust_log_level(), 2);
            std::env::set_var("RUST_LOG", "info");
            assert_eq!(rust_log_level(), 1);
            std::env::set_var("RUST_LOG", "trace");
            assert_eq!(rust_log_level(), 3);
            std::env::set_var("RUST_LOG", "warn");
            assert_eq!(rust_log_level(), 0);
            std::env::remove_var("RUST_LOG");
        }
    }

    #[test]
    fn format_eta_and_bar() {
        assert_eq!(format_eta(2.8), "2.8s");
        assert_eq!(format_eta(46.0), "46.0s");
        assert_eq!(format_eta(120.0), "2m00s");
        let b = bar(62);
        assert_eq!(b.chars().count(), 20); // █ 是 3 字节——按字符计数
        assert_eq!(b.chars().filter(|&c| c == '█').count(), 12);
        assert_eq!(bar(100), "████████████████████");
        assert_eq!(bar(0), "░░░░░░░░░░░░░░░░░░░░");
    }

    #[test]
    fn expand_tilde_and_glob() {
        assert_eq!(expand_tilde("/a/b"), PathBuf::from("/a/b"));
        assert_eq!(expand_tilde("rel/dir"), PathBuf::from("rel/dir"));
    }
}

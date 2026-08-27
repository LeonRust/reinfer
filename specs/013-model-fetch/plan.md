# Plan: Model fetch — pure-Rust ModelScope downloader

> Derived from specs/013-model-fetch/spec.md

## Architecture Decisions

- **D1 分层**：
  - `crates/models`：REST 客户端与下载逻辑——`ModelScopeApi`（list_files/get_download_url 纯 URL 构造 + JSON 解析）、`download_to(path, file_meta, dir)`（跟随 302、流式写 temp、sha256 校验、重试一次、rename+manifest）；依赖 `ureq`（rustls）+ `sha2`/`serde_json`；网络面独立于引擎（`Fatal`: 网络/解析/校验；`Oom`: ENOSPC）
  - `bin/reinfer`：子命令 `model list/get`（std 参数解析：前两个 args = 子命令与 repo；`--file`/`--to/--all` 标志；`--help` 文本）
- **D2 URL 模板（实测）**：`list` = `/api/v1/models/{owner}/{model}/repo/files?Revision=master`；`get` = `/api/v1/models/{owner}/{model}/repo?Revision=master&FilePath={path}`（302 跟随；ureq 默认最多 5 跳，写 fetch 前确认 CDN 跳数 ≤3）。
- **D3 manifest**：`<to>/manifest.json`（serde）：`[{name, size, sha256, repo, revision, fetched_at}]` 追加式；读取端（基准/L3）契约 = 以文件 sha256 为准。
- **D4 凭据与代理**：不认 token（公开仓库）；代理完全由标准 env 驱动（ureq 读 `HTTPS_PROXY` 等 `AgentBuilder/Proxy`）；spec/catalog 不硬编码用户代理 IP。
- **D5 测试策略**：URL/JSON 解析纯函数单测（fixture = 本次实测抓取的 files JSON 摘录）；下载器对本地 HTTP stub 验证（重定向+校验失败+temp 清理）；**端到端大文件**作为人工验证步骤（`--ignored` 性质的 manual，见 tasks T4，不占 CI——500MB+ 文件不入日常流水）。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/models/src/{api,download,manifest}.rs` | files 列表/URL 模板/解析；下载+校验+原子写+重试；manifest 追加 |
| `crates/models/src/error.rs` | `ModelError{网络/校验/io}` → 分类（可选，直接 `LaunchError` 复用更简——抉择：模型获取属工具面，复用 `LaunchError`（Oom/Fatal）+详细 stderr，与 jit/cuda 一致） |
| `bin/reinfer/src/main.rs` | 子命令分发（`model list` / `model get` / `help`）+ 参数解析 |
| `crates/models/tests/stub_download.rs` | 本地 HTTP stub（std TcpListener 手动应答）验证重定向+校验失败+原子写 |
| 真机验证 | 0.5B q8_0 下载（675,710,816 B，sha256 已知——spec 锚点）→ 校验 + manifest（人工，写 notes） |

## Interface Contracts

```rust
// crates/models
pub struct FileEntry { pub name: String, pub size: u64, pub sha256: String, pub is_lfs: bool }
pub fn list_files(owner_model: &str) -> Result<Vec<FileEntry>, LaunchError>;
pub fn download_file(owner_model: &str, entry: &FileEntry, to_dir: &Path) -> Result<PathBuf, LaunchError>;
// bin
//   reinfer model list <owner/model>
//   reinfer model get  <owner/model> --file <name> [--to <dir>]   (缺省 --to ~/models/reinfer)
//   reinfer model get  <owner/model> --all [--to <dir>]
```

错误面：网络/超时/解析/校验 → 重试一次 → `Fatal`（stderr 含详情与代理提示）；ENOSPC → `Oom`；目标已存在且 sha256 匹配 → 跳过（幂等，打印 hit）。

## Risk Assessment

| Risk | Mitigation |
|---|---|
| ModelScope API 漂移（字段/端点变化） | 契约已实测钉死 + 解析容错（缺 Sha256 → Fatal 显式消息）；stub 与真机双保险；变更走 spec changelog |
| CDN auth_key 时效（302 后立即失效） | 跟随重定向及时（ureq 默认跟随）；不做预取 URL 复用 |
| 大文件下载中断 | temp 流式写 + rename 原子（无半成品）；重试 1 次 |
| 代理依赖（无代理环境） | 标准 env 语义：`NO_PROXY` 可排除；文档注明用法 |
| 网络受限沙箱 | 本 spec 契约由代理探测钉死；端到端在用户机验证 |

## 里程碑

- M1：`crates/models`（api+download+manifest + 单元/stub 测试）
- M2：`reinfer model` 子命令 + help
- M3：真机 0.5B 端到端（人工验证 + notes 留痕）
- M4：README/CLAUDE/feature-list 同步

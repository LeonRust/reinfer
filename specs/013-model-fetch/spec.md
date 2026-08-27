# Spec: Model fetch — pure-Rust ModelScope downloader (reinfer model get)

> Status: proposal · Owner: maintainers · Created: 2026-08-27
> Parent/锚：phase-plan L3 前置（001/004 数据管道共用）· 铁律"模型一律 ModelScope 下载"
> 契约实测基线：2026-08-27 经代理探测 ModelScope 公开 REST（见本文件"实测契约"）

## Problem Statement

模型获取必须满足：① 只从 ModelScope（魔搭）下载（铁律）；② **不引入 Python/外部 CLI**
（项目 rust 优先——modelscope 官方 CLI 依赖 pip，明确排除）；③ 融入"单二进制"形态。
ModelScope 公开仓库是纯 REST：官方 SDK/CLI 只是 HTTP 封装，故纯 Rust 客户端可行，
落为 `reinfer model get/list` 子命令。

## 实测契约（2026-08-27，代理 http://192.168.0.1:7890 实测 Qwen/Qwen2.5-0.5B-Instruct-GGUF）

| 项 | 契约 |
|---|---|
| 文件清单 | `GET https://modelscope.cn/api/v1/models/{owner}/{model}/repo/files?Revision=master` → JSON `{Code:200, Data:{Files:[{Name, Path, Size, Sha256, IsLFS, Revision}]}}`；**Sha256 字段可直接用于校验** |
| 下载入口 | `GET https://modelscope.cn/api/v1/models/{owner}/{model}/repo?Revision=master&FilePath={Path}`（Revision 可选）→ **302** 到 `https://cdn-lfs-cn-1.modelscope.cn/prod/lfs-objects/{sha[..2]}/{sha[2..4]}/{sha}?filename=...&auth_key=...`（瞬时签名）——**跟随重定向即可**（不自行拼 CDN URL；auth_key 有时效） |
| LFS | 目标 GGUF `IsLFS=false`（真实 blob，直下即得）——契约注明；若未来 IsLFS=true → 列入 Non-Goal（git-lfs 域） |
| 校验 | 下载文件 sha2-256 必须等于 files API 的 `Sha256`；否则删除重试一次 → 仍失败 `Fatal` |
| 已知基准值（端到端测试锚点） | `qwen2.5-0.5b-instruct-q8_0.gguf`：size=675710816，sha256=`ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` |
| 代理 | 尊重标准 `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` 环境变量（与 git/curl 一致；用户机示例 `http://192.168.0.1:7890`）——HTTP 层不自行追加代理参数 |

## User Stories

1. 作为引擎作者：`reinfer model get Qwen/Qwen2.5-0.5B-Instruct-GGUF --file qwen2.5-0.5b-instruct-q8_0.gguf --to ~/models/reinfer/` —— 一个 Rust 二进制完成模型获取（无 pip、无 Python）。
2. 作为维护者：下载结果带 sha256 校验与 manifest 留痕（换机复制/基准确认）。
3. 作为 CI/复现者：无代理直连、有代理也通（标准 env）；校验失败宁可失败不静默。
4. 作为引擎作者：运行时 `ModelResolver.ensure()` 缺模型自动下载——引擎不假设机器预置模型。
5. 作为离线/管控环境：`REINFER_MODEL_AUTODOWNLOAD=off` 保证引擎绝不联网。

## Acceptance Criteria

- [ ] `reinfer model list <repo>`：解析 files API → 打印 GGUF 文件（名/大小/sha256 前 16 位）
- [ ] `reinfer model get <repo> --file <name> [--to <dir>]`：下载+sha256 校验+原子落盘+manifest 追加
- [ ] 校验失败 → 重试一次 → `Fatal`（不静默、不留半成品 temp）
- [ ] 网络/超时错误分类为可重试（一次）后 `Fatal`；磁盘满 → `Oom`
- [ ] 端到端：真机下载 0.5B q8_0（675,710,816 B）校验通过（一次性人工验证，非日常 CI）
- [ ] README/CLAUDE 增"模型获取"段；feature-list 状态更新
- [ ] r2：`ModelResolver::from_env` 解析全部 `REINFER_MODEL_*`（缺省值语义对齐契约表）；`ensure()` 本地命中/自动下载/off-报错三态可测
- [ ] r2：双源——ModelScope 404 → HF 回退（stub 测）；HF 路径 ETag+size 校验；VERIFY=none 只查存在性
- [ ] r2：`AUTODOWNLOAD=off` → 缺模型返回明确错误（无网络动作）

## r2（2026-08-27 探讨增补）——运行时自动下载 + 双源 + 环境变量策略面

- **运行时代码路径**：`ModelResolver::from_env()?.ensure(&ModelSpec)` —— L3 GGUF 加载器检测本地无模型 → 按策略自动下载 → 返回路径（离线可关）。CLI 与运行时走同一套下载机制（仅入口不同）。
- **双源**：默认 ModelScope（铁律主体）；`auto` 语义 = ModelScope 无此模型/文件 → **HuggingFace 回退**（用户 2026-08-27 明确放开——修订"一律 ModelScope"为"ModelScope 优先 + 可回退"）。
- **环境变量面**（`REINFER_MODEL_*`，语义见 plan D6 表）：`SOURCE`(`modelscope|huggingface|auto`)/`DIR`/`VERIFY`(`sha256|size|none`)/`AUTODOWNLOAD`(`on|off`)。
- **校验强度差异**：ModelScope 有官方 Sha256；HuggingFace 无 sha 字段——`VERIFY=sha256` 对 HF 源降级为 ETag+size（docs 声明）。
- 契约前提不变：网络出口经标准 `HTTP(S)_PROXY` 等 env；端到端在用户机验证。

## r3（2026-08-27 CLI 定版——对齐成熟工具惯例，非自创范式）

- `reinfer model list` —— **本地**已下载清单（先例：`docker image ls`/`ollama list`/`pip list`：默认本地、零参）；
- `reinfer model ls-remote <repo>` —— 远端仓库文件清单（先例：git `ls-remote`）；
- `reinfer model get <repo>` —— `hf download` 语义：repo 位置参数；`-q/--quant`、`-f/--file`、`--all` 互斥选其一（无任何缺省行为——无默认模型）；目录参数 `--local-dir <dir>`（hf 命名；缺省 `REINFER_MODEL_DIR`）；支持 `--flag=value`（git/gh 风格）；
- 参数错误 → exit 2 + 用法提示。**不沿用**旧雏形 `list <repo>`（远端）/`--to` 语义——r1 文档里的 bin 契约块整体由此段替代。

## Non-Goals

- 断点续传/分片（重试 + 原子写已覆盖小规模故障；超 1GB 文件后续可加 `Range`）
- 私有/付费仓库（认证、SDK 式 token）与 `IsLFS=true` 的 git-lfs 拉取
- HuggingFace 侧 sha256 等价强校验（官方 API 无该字段——ETag+size 为上限）
- Revision 非 master（SNAPSHOT 语义将来扩展）
- 下载速度/并发镜像（对象存储多 CDN，v1 单 URL）

## Constraints

- **模型标识零硬编码（用户 2026-08-27 铁律）**：仓库名/文件名/量化段一律由调用方或 CLI 参数提供——`ModelSpec::new(repo)` 必填参数、CLI 必填 repo；库代码**不含任何默认模型常量/便捷函数**（如 `ModelSpec::qwen_05b()` 禁止）；单元/stub 测试一律用虚构 repo（如 `stub/models`）与本地 stub 服务；真实仓库名只出现在文档示例与 T4 真机命令（README/notes 属文档、可含示例，不入代码路径）。
- 纯 Rust：HTTP 客户端选轻量同步 `ureq`（默认 rustls，无 OpenSSL 系统依赖、无 tokio）；`sha2`/`serde_json` workspace 已有
- 原子落盘纪律沿用 jit：同目录 temp + rename；manifest 与 jit meta 一样"提交点"式
- 单二进制：解析仍用 std（不引入 clap 等新命令依赖；参数形态见 plan）
- 网络入口在用户机（本沙箱访问不了 modelscope；契约由代理实测钉死，端到端由用户机验证）

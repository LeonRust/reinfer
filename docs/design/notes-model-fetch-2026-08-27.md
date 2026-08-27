# Machine notes — model fetch (013) · 2026-08-27

> 真机验证记录（开发机 dora + 代理环境）。对应 `specs/013-model-fetch/tasks.md` T4 与验收 AC。
> 本表为人工复查留存：命令可复制、锚值一致、已知偏差逐条记录。

## 环境

- 机：dora (Linux x86_64)，Rust 1.98 (2026-08-18 工具链)
- `.env`（开发测试环境，gitignored；模板 `.env.example`）：

```bash
REINFER_MODEL_SOURCE=auto
REINFER_MODEL_DIR=~/models/reinfer
REINFER_MODEL_VERIFY=sha256
REINFER_MODEL_AUTODOWNLOAD=on
REINFER_MODEL_REPO=Qwen/Qwen2.5-0.5B-Instruct-GGUF
REINFER_MODEL_QUANT=q8_0
HTTPS_PROXY=http://192.168.0.1:7890
HTTP_PROXY=http://192.168.0.1:7890
NO_PROXY=localhost,127.0.0.1,modelscope.cn,huggingface.co
```

> 注：`NO_PROXY` 含 `modelscope.cn`/`huggingface.co` → 对这两个域直连（本机可直连时省去代理）。
> 无直连的三方环境：删 `NO_PROXY` 排他项即可走代理。

## 验证 1 — `model list`（files API 契约）

```bash
cargo run -p reinfer -- model ls-remote Qwen/Qwen2.5-0.5B-Instruct-GGUF
```

结果：9 个 GGUF 列出；关键锚值——

| 文件 | size | sha256（前 16） |
|---|---|---|
| `qwen2.5-0.5b-instruct-q8_0.gguf` | **675710816** | **ca59ca7f13d0e15a** |
| `qwen2.5-0.5b-instruct-fp16.gguf` | 1266425696 | 8e0ae26000627ed6 |

与 spec 实测基线完全一致。

## 验证 2 — `model get`（端到端 sha256 校验 + manifest）

```bash
cargo run -p reinfer -- model get Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q8_0
```

输出（摘）：

```
/home/dora/models/reinfer/qwen2.5-0.5b-instruct-q8_0.gguf ready
manifest: repo=Qwen/Qwen2.5-0.5B-Instruct-GGUF branch=master size=675710816 sha256…fa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e
```

- 落盘：`/home/dora/models/reinfer/qwen2.5-0.5b-instruct-q8_0.gguf` **675,710,816 B**（与 files API 一致）
- sha256 校验通过：`ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`
- manifest 留痕 ✓（`manifest.json` 含 name/size/sha256/repo/branch/fetched_at）
- 二跑同命令 → 本地命中提示（glob 命中 + no re-download）

## 验证 3 — 幂等与 off（本机）

- 二次 `model get` 同参数：立即返回路径，无网络动作（日志无 GET）。
- r3 定版后命令面：`model list`（本地）/`model ls-remote`（远端）/`model get -q|-f|--all [--local-dir]`（2026-08-27）。
- `REINFER_MODEL_AUTODOWNLOAD=off` 且目录空：报 "not found locally … refusing to dial out"，exit 1。

## 已知偏差 / 备注

1. `.env` 的 `REINFER_MODEL_DIR=~/models/reinfer` 含 `~` → resolver 已做 `~` 展开（`expand_tilde`，env/CLI 均可写 `~/…`）。
2. HF 源校验强度上限 = ETag+size（上游 API 无 sha256 字段；`VERIFY=sha256` 对 HF 自动降级）；HF 分支缺省 `main`，ModelScope 用 `master`。
3. 本机未跑 `--all`（9 文件共 ~5.4GB）；该支路在 stub 测试覆盖。
4. 下载为单 URL 流式（无 Range/断点续传；1GB+ 文件后续可加——见 spec Non-Goals）。

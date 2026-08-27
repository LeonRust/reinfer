# reinfer CLI 手动测试方案（2026-08-27）

> 按 CLI 契约（docs/design/cli-contract-2026-08-27.md v2.9）人工测试清单。标注
> [真机] 项需网络（.env 已配直连/代理）；其余本机可完成。每项含：命令 → 预期 → 通过标准。
> 环境：仓库根目录运行（bin 通过 `cargo run -p reinfer -- ...` 或 `./target/debug/reinfer`）。

## 0. 准备

```bash
cd ~/Dev/ai-tokens/reinfer
set -a; . ./.env; set +a      # 测试环境 env（REINFER_MODEL_DIR=~/.reinfer/models 等）
cargo build -p reinfer
```

## 1. 通用/全局

| # | 命令 | 预期 | 通过标准 |
|---|---|---|---|
| 1.1 | `reinfer --version` 与 `-V` | `reinfer 0.1.0` | 输出与 Cargo 版本一致，exit 0 |
| 1.2 | `reinfer`（无参） | 用法帮助 | exit 2 |
| 1.3 | `reinfer help` / `-h` / `--help` | 用法帮助 | exit 0 |
| 1.4 | `reinfer bogus` | unknown command | exit 2 |
| 1.5 | `reinfer download`（缺 repo） | 缺参数错误（clap 格式含 usage） | exit 2 |
| 1.6 | `reinfer -v model list` | 详情输出（stderr 多一行 verbose） | 列表正常 + stderr 有 verbose 行 |
| 1.7 | `reinfer --no-color download ... -q q8_0` | 功能同上（无着色差） | 正常 exit 0 |
| 1.8 | `reinfer -q`（无命令） | 未知全局旗 | exit 2 |
| 1.9 | `reinfer download <repo> -- --dash.gguf` | `--` 后为文件名 | ""--dash.gguf not in repo"" → exit 1（非未知选项） |

## 2. model list（本地清单）

| # | 命令 | 预期 | 通过标准 |
|---|---|---|---|
| 2.1 | `reinfer model list` | 列出模型根下**按 repo 组织**的文件（人类可读 size/sha256/source） | 显示 `Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf` 等（repo 前缀） |
| 2.2 | `reinfer model list --format json` | JSON 数组（原始字节/完整 sha/source） | jq 可解析、字段齐全 |
| 2.3 | `reinfer model list --format quiet` | 每文件一行名 | 无表格 |
| 2.4 | 空目录（`REINFER_MODEL_DIR=/tmp/emptydir reinfer model list`） | "no local model files in …" | exit 0 |
| 2.5 | 目录不存在 | 明确错误 | exit 1 |
| 2.6 | 目录含非模型文件（manifest.json、.tmp、隐藏文件） | 不显示；`foo.safetensors`（repo 子目录内）显示——格式无关 | 显示 repo 内文件，不显示 manifest.json |
| 2.7 | 格式无关 | 同 2.6 | — |

## 3. download —— 远程（真机）

| # | 命令 | 预期 | 通过标准 |
|---|---|---|---|
| 3.1 | `reinfer download Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q8_0` | 已存在 → `-> 名` + `路径 (done)`（契约 §3 格式） | exit 0，无重下（`-v` 可见无 GET） |
| 3.2 | `--local-dir /tmp/m1` 重复 3.1 | 下载完成（~644 MiB）至 /tmp/m1 | 文件存在/size 对/manifest.json 写入 |
| 3.3 | `reinfer download <repo> -q q8_0 --dry-run` | 列计划表（文件名/字节/already） | **不产生任何文件**；already 对已存在文件为 yes |
| 3.4 | `reinfer download <repo> --include '*.json' --local-dir /tmp/m2 --revision master` | 下载 configuration.json | 一个文件即可（/tmp/m2） |
| 3.5 | `reinfer download <repo> qwen2.5-0.5b-instruct-q8_0.gguf --local-dir /tmp/m3` | 精确文件下载 | exit 0 |
| 3.6 | `reinfer download <repo> nonexist.gguf` | "nothing to download"/file not in repo | **exit 1** |
| 3.7 | `reinfer download <repo>`（无选择） | 全仓库（hf 语义）——**大下载确认后单独执行** | exit 0（谨慎触发） |
| 3.8 | `reinfer download <repo> -q q8_0 --format json` | 结果 JSON schema（file/bytes/sha256/downloaded） | jq 字段齐全 |
| 3.9 | `reinfer download <repo> -q q8_0 --quiet` | 仅最终路径一行 | 无表格/进度 |
| 3.10 | 进度条（TTY） | 紧凑视图：`n/11 files` 计数行 + 活跃文件行（≤4）+ GLOBAL 底行（v2.14 方案 A） | 小终端无重复/截断（对照：清单视图在此高度会幽灵双行）；完成即折叠；`model list` 见全清单 |
| 3.11 | `--format json` 下无进度条 | 纯净 JSON | 无 ANSI 残留 |
| 3.12 | 并发 | `--include '*.gguf'` 时多文件并行（4 worker） | 观察行为（完成无错误）——**大下载**可选 |
| 3.13 | `download -q q8_0 -q q4_0` | 重复旗错误 | exit 2 |

## 4. completions

| # | 命令 | 预期 | 通过标准 |
|---|---|---|---|
| 4.1 | `reinfer completions bash` | bash 补全脚本（含 download/model/list 词表） | 输出含 `download` 与 `--quant`；可 source |
| 4.2 | `zsh` / `fish` 同 | 各自语法脚本 | 输出对应头部（#compdef/complete -c） |
| 4.3 | `reinfer completions xyz` | 不支持 | exit 2 |

## 5. doctor（本机）

| # | 命令 | 预期 | 通过标准 |
|---|---|---|---|
| 5.1 | `reinfer doctor`（默认构建，无 cuda/ascend feature） | CUDA/CANN 显示 `[⚠] not compiled in (feature)`；模型目录/工具链/env ✓ | **exit 0**（feature 未编译=warning 非 blocker） |
| 5.1b | `cargo build --features cuda` 后 `reinfer doctor`（本机 GPU） | 真实设备行 `[✓] NVIDIA … (sm_120, 23.42GiB)` + nvcc 状态（`[⚠]` 若低于 sm_120 梯度） | 无 ✗ → exit 0 |
| 5.2 | `reinfer doctor --format json` | 机读 JSON | 字段含 backend 状态与 env 面 |
| 5.3 | `reinfer doctor --backend cuda` | 仅 CUDA 块 | 无 CANN 行（或标注 not built） |
| 5.4 | 阻塞场景（临时把 REINFER_MODEL_DIR 指向不可写路径） | ✗ + exit 1 | 修复建议行 |

## 6. env/策略面

| # | 操作 | 预期 |
|---|---|---|
| 6.1 | `REINFER_MODEL_AUTODOWNLOAD=off download ... 未命中` | 明确报错，**无网络动作**（verbose 或抓包确认），exit 1 |
| 6.2 | `REINFER_MODEL_VERIFY=size` | size 校验档；`none` 仅存在性 |
| 6.3 | `REINFER_MODEL_DIR=~/...`（~ 展开） | 目录正确（已有 .env 场景） |
| 6.4 | 无 .env（清空 env）| 缺省 ~/.reinfer/models 生效 |

## 7. 已知不做（不在本次测试范围）

- serve/run/chat/bench——依赖推理引擎闭环（003/005），未立项，`reinfer help` 不出现（无 ghost）
- `--net` doctor 探针、`--prompt-file`（bench 用，同上）
- CANN（昇腾）支路——本地无 NPU（`--exclude reinfer-ascend`；doctor 会标 not built）

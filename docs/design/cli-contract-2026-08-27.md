# CLI 契约（2026-08-27 定稿）

> 跨功能 CLI 约定。经用户 2026-08-27 多轮决策（三次要点 + 四项分歧点）定稿：
> ① 主命令面 = **动作式**（vllm/gh 系）；② 模型引用 = **位置参数**（vllm 式，路径或 repo 均可）；
> ③ 下载 = **顶层 `download`**（hf download / modelscope download 先例）；④ 无显式文件选择时
> **跟随 hf 语义 = 下载整个仓库**；⑤ 最小集（`ls-remote` 移除，两家均无"列仓库文件"命令）。
> 规则先例对照见 §5；实现落地见 specs/013-model-fetch r5。

## 1. 命令面

```
reinfer <command> [args]

命令（动作式，平级；未立项的命令仅存于本契约，不进 help）：
  download <repo> [file...] [-q <qtag> | --include <glob> --exclude <glob>]
             [--revision <ref>] [--local-dir <dir>]
             模型获取（当前已实现）
  serve   <model> [--host...]     HTTP 推理服务（P1-05 / specs/005；先例 vllm serve）
  chat    <model>                  交互对话（specs/005；先例 vllm chat）
  run     <model> [prompt...]      单次生成（prompt 剩余位置参数拼接）
  bench   <model>                  性能基准（003 / specs/008 gate_throughput）
  diag                              环境诊断（CUDA/昇腾；ASC-03 / specs/002）

名词组（唯一例外——服务数据管道的"查看"，非推理主路径）：
  model list                       本地已下载（先例：docker image ls / ollama list /
                                   modelscope-ng list / pip list——默认本地、零参）
```

## 2. download 语义（013 r5）

- **repo**：位置参数必填（hf 先例）。
- **文件选择**三态（hf 先例的组合）：
  - `<file...>` 位置列表：精确文件（可多个；非 .gguf 也可——hf 语义）；
  - `--include <glob>` / `--exclude <glob>`：模式过滤（支持 `*` `?` 两枚通配；`**` 视作 `*`）；
  - **无任何选择 → 下载整个仓库**（hf 默认；对比 013 旧铁律"无默认模型"——该铁律指**模型标识**
    不硬编码，与"用户显式给 repo 后按 hf 语义全量"不冲突，契约以本条为准）；
  - `-q <qtag>`：引擎专属"量化档选择"（量化段 → 文件名匹配，如 `q8_0`→`-q8_0.gguf`；
    两家 CLI 无此概念，属自有领域词；除该词外其他旗子均有两家先例）。`-q` 与 file.../--include 互斥。
- **`--revision`**：分支/tag/commit（hf、modelscope 同款）；None → ModelScope `master` / HF `main`。
- **`--local-dir`**：落地目录（hf 命名）；缺省 `REINFER_MODEL_DIR`（env 或 `~/models/reinfer`）。
  modelscope 官方同义参数名为 `--local_dir`（snake）——本项目统一 kebab（hf 系 + Rust 生态），
  modelscope 的 snake 形式不兼容吸收。
- 校验/幂等/metadata 不变（013 D6：sha256/ETag+size；manifest 留痕；`AUTODOWNLOAD=off` 拒绝网络）。

## 3. 通用旗子规则（本项目全局）

| 规则 | 值 |
|---|---|
| 短旗 + 长旗 | `-q`/`--quant` 式并存；长旗是规范名 |
| `--flag=value` | 支持（git/gh 风格）与 `--flag value` 两种 |
| 错误退出码 | 用法/参数错误 exit 2 + stderr 提示；执行失败（网络/校验/磁盘）exit 1 + 代理提示 |
| 子命令 help | `reinfer <cmd> help|-h|--help` |
| 机器可读 | 预留 `--format [table|json]`（先例 hf `--format`），未立项不实现 |

## 4. 未来命令契约（未实现；各 spec 立项时按此落位）

- `serve <model> [--host <h>] [--port <p>]`（先例 vllm serve；405/005 定细节）
- `chat <model>`（交互 REPL）
- `run <model> [prompt...]`（单次生成；prompt 位置拼接）
- `bench <model> [--gate ...]`（003/008 gate_throughput 挂接）
- `diag`（无模型参数；ASC-02/03 已定 `reinfer diag` 名）

## 5. 先例对照表

| 规则 | 先例 |
|---|---|
| 动作式主命令 | vllm `serve/bench/chat` · gh `pr list` · uv `run` |
| 下载顶层动词 `download` | `hf download` · `modelscope download`（两家同名） |
| 模型/仓库位置参数 | `hf download gpt2 config.json` · vllm `serve <model>` |
| `file...` 位置文件列表 | `hf download gpt2 config.json model.safetensors` |
| `--include/--exclude` 模式 | hf · modelscope（共同） |
| 无文件 → 全仓库 | `hf download REPO`（默认快照） |
| `--revision` | hf · modelscope（共同） |
| `--local-dir` | `hf download --local-dir`（kebab；modelscope 为 `--local_dir`，不采纳其 snake） |
| 本地清单零参 `list` | docker `image ls` · ollama `list` · modelscope-ng `list` |
| `--flag=value` | git · docker · ffmpeg |
| 量化档领域词 `-q` | 无直接先例（引擎领域）；短旗形式有普适先例；文档注明为自有语义 |

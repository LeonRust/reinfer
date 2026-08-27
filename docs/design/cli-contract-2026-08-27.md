# CLI 契约（2026-08-27 定稿 v2）

> 用户 2026-08-27 多轮决策定稿：① 动作式主命令（vllm/gh 系）；② 模型引用 = 位置参数；
> ③ 下载 = 顶层 `download`（hf/modelscope 同名先例）；④ 无显式文件选择 = 整仓库（hf 默认）；
> ⑤ 最小集；⑥ 默认目录 = `~/.reinfer/models`（Windows `%USERPROFILE%\.reinfer\models`）；
> ⑦ 增强并入：`--dry-run`、`--format/--quiet`、4 并发；⑧ 进度显示两层设计（文件条 + GLOBAL 条）。
> **状态**：v2 已锁定本文件内容的定稿部分（v2.1：serve 章节；v2.2：run/chat/bench 章节；
> v2.3：doctor 章节定稿；v2.4：通用五件套（version/completions/日志分级/no-color/`--`）；v2.5：解析实现=clap；v2.6：env 体系（RUST_LOG 兼容/SERVE_* env/doctor 回显）；v2.7：小件定稿+实现迁移路径）；
> CLI 整体方案仍在与用户继续探讨（后续增减以 r 版本记录在本文件 changelog）；
> 实现动工待整体探讨结束后统一批准。

## 1. 命令面

```
reinfer <command> [args]

download <repo> [file...] [-q <qtag> | --include <glob> --exclude <glob>]
           [--revision <ref>] [--local-dir <dir>]
           [--dry-run] [--format table|json|quiet | --quiet]
           模型获取（已定稿；当前已实现子集见 §8）――注：--dry-run/--format/并发/进度条为实现中

serve   <model> [--host...]      HTTP 推理服务（P1-05 / specs/005；先例 vllm serve）——契约预置，未立项
chat    <model>                   交互对话（specs/005）——契约预置
run     <model> [prompt...]       单次生成（prompt 剩余位置拼接）——契约预置
bench   <model>                   性能基准（003 / specs/008）——契约预置
doctor                              环境体检（flutter/cargo doctor 先例；ASC-03 改名落位）——契约预置

completions <shell>               生成 shell 补全（bash|zsh|fish；gh/kubectl 先例）
model list [--format table|json|quiet]
           本地已下载清单（先例：docker image ls / ollama list / modelscope-ng list——零参默认本地）
```

未立项命令不进 help（无幽灵命令）；各 spec 立项时按本契约落位。

## 2. download 参数全表（v2 定稿）

| 参数 | 语义 | 先例 |
|---|---|---|
| `<repo>` 必填位置 | owner/model（或本地路径——resolver 判定） | hf/vllm |
| `[file...]` | 精确单/多文件（非 .gguf 亦可） | hf 位置文件 |
| `-q/--quant <tag>` | 量化档→文件名（`q8_0`→`-q8_0.gguf`）；与 file.../--include **互斥** | 引擎领域词（短旗惯例） |
| `--include/--exclude <glob>` | 模式过滤：`*` `?`；fnmatch 语义（`*` 跨 `/`）；`--exclude` 须配 `--include` | hf/modelscope 共同 |
| （无任何选择） | 整个仓库（hf 默认快照） | hf |
| `--revision <ref>` | 分支/tag/commit；MS=files/download URL `Revision=`，HF=`resolve/{ref}`；None→MS`master`/HF`main` | hf/modelscope |
| `--local-dir <dir>` | 落地目录；缺省 `REINFER_MODEL_DIR` → `~/.reinfer/models` | hf（kebab） |
| `--dry-run` | 只列计划不下载：文件/大小/`already`（按 verify 深度 local_hit）；不写任何文件 | hf |
| `--format table` | 默认人类表格 | hf |
| `--format json` | 机器可读结果 JSON（§3 schema） | hf |
| `--format quiet` / `--quiet` | 每成功文件仅打印路径行（失败走 stderr）；`-q` 已被 quant 占用，quiet 仅长旗 | hf |
| 并发 | 固定 4 workers（文件间并行）；不增 CLI 旗子 | modelscope SDK `max_workers=4` |

## 3. 输出契约（table/json/quiet + dry-run）

- **table**（默认）：`-> name` / `path (done)` 行 + `n/m downloaded` 汇总；顺序按文件名排序（确定性，并行后稳定）
- **json**：`{"files":[{"file":<path>,"bytes":N,"sha256":"…","downloaded":bool}]}`；`--dry-run` 时顶层加 `"dry_run":true`、每条含 `"already":bool`
- **quiet**：每成功文件一行路径
- **dry-run**：Table 格式：
  ```
  [dry-run] will download N file(s) (X B) to <dir>
  name           bytes               already
  ...            ...                 yes/no
  ```

## 4. 进度显示（v2 定稿）

两层、TTY-only（`std::io::IsTerminal` 检测；纯 std 绘制 `\r`+`\x1b[2K`+`\x1b[A`，不引依赖）：

```
[1/4] qwen2.5-0.5b-instruct-q8_0.gguf      ████████████░░░░░░░░  62%  418.9/675.7MB  91.4MB/s  ETA 2.8s
[2/4] qwen2.5-0.5b-instruct-fp16.gguf      ██████░░░░░░░░░░░░░░░  31%  398.1/1.26GB   88.7MB/s  ETA 9.7s
[3/4] qwen2.5-0.5b-instruct-q2_k.gguf      ████████████████████ 100%  415.2/415.2MB  (done)
GLOBAL ████████████░░░░░░░░░░░░░░░░░░░░░  27%  1.27GB/5.45GB   91.1MB/s  ETA 46s
```

| 规则 | 值 |
|---|---|
| 文件条（活跃 worker 各一行） | 文件名截断 44 列 + 条 + % + 已/总 + 速度 + ETA；`(done)` 保留 ~1s 回收 |
| GLOBAL 条（底行固定） | 条 + % + 落盘字节/总字节（目标合计 size；命中即落盘计入）+ 2s 滑窗速度 + ETA |
| 刷新 | 每 256KB 或 200ms tick；校验重试时该文件进度从 0 重计 |
| 降级矩阵 | `table+TTY`=动态条 · `table+非TTY`=摘要行 · `quiet/json/dry-run`=无条 |
| 单文件 | 文件条=全局，GLOBAL 行仍显示 |
| Ctrl-C | 保留 temp（与重试纪律一致）；结束打印 `\n` 留最后一帧 |

实现落点：`crates/models` 下载层 `fetch_to_temp` 增可选进度回调（`bytes_done,total`）；CLI 聚合 4 worker。

## 5. 通用规则

| 规则 | 值 |
|---|---|
| 短/长旗 | `-q`/`--quant` 并存，长旗规范；`--flag=value` 与 `--flag value` 均支持；`-q` 与 `--quiet` 语义冲突已在 §2 注明 | 
| 用法/参数错误 | exit 2 + stderr（含 `try reinfer help`） |
| 执行失败 | exit 1 + stderr 详情（缺代理打 hint） |
| 子命令 help | `reinfer <cmd> help|-h|--help` |
| env 规则 | **默认值不写进 env 模板**；env 显式设置即生效（通用软件惯例——见"附. 待办"） |

### env 体系（v2.6 定稿——现状家族 + 三项）

**现状（自主面 `REINFER_` 前缀）**：MODEL_（SOURCE/DIR/VERIFY/AUTODOWNLOAD/REPO/QUANT/FILE）·
JIT_（CACHE/LOCK_DIR/LOCK_TIMEOUT）· CUDA_（ARCH/NVCC）——命名空间三节，布尔统一 on/off；
显式 CLI > 便捷注入；默认值不入 env 模板。**生态面**（只消费）：HTTPS_PROXY/HTTP_PROXY/NO_PROXY ·
CUDA_VISIBLE_DEVICES · DEVICE_ID/ASCEND_* · CUDA_HOME/CUDA_PATH/PATH · XDG_CACHE_HOME/HOME · NO_COLOR。

**三项定稿（用户 2026-08-27）**：

1. **日志面兼容 `RUST_LOG`**：日志分级 = CLI `-v/-vv/--debug` 优先，否则读 `RUST_LOG`
   （Rust 生态标准；示例 `RUST_LOG=debug`）；将来 tracing 接入时保持同一入口。
2. **服务面 env **：`REINFER_SERVE_HOST`/`REINFER_SERVE_PORT`/`REINFER_SERVE_DEVICE`
   = serve/run/bench 上述参数的缺省来源（CLI 旗覆盖 env；ollama `OLLAMA_HOST` 先例）；
   env 缺省値不写模板（v2 规则）。
3. **doctor 回显**：检查块含完整 env 表回显（自主面 + RV 现状值＋来源标注：.env/环境）。

### 解析实现（v2.5 定稿：clap，用户 2026-08-27 拍板）

- **采用 `clap`（derive 家族）实现 CLI 解析**——废弃 spec 013 早期"std 手写"决定（该决定为
  r1 时代约束；现以本契约为准；013 spec 侧 r6 决策段同步）。
- **契约规则优先于 clap 默认**（实现时逐条显式配置，不以 clap 默认替代契约语义）：
  1. `reinfer <cmd> help` 子命令 help 形态（clap 默认 `--help` 之外补显式 `help` 子命令）；
  2. `--flag=value` 与 `--flag value` 两种形式（clap 默认即支持，确认行为一致）；
  3. 全局旗（`-v/-vv/--debug/--no-color/-V`）**仅命令前**（gh 惯例；clap 全局旗另有默认，配置对齐）；
  4. `-q/--quant` = quant（**不是** quiet；quiet 仅 `--quiet` 长旗）；
  5. `--format` 枚举（table|json|quiet 等按命令面）；`-v/-vv` 叠用（ArgAction::Count）；
  6. 用法错误 → **exit 2**（clap 默认为 2，验证）；执行失败 exit 1（保持）；
  7. 模型引用三态的 `-q|-f` 互斥/条件（conflicts/requires 声明式表述）；
  8. `--` 分隔、value 以 `-` 开头的显式传参模式。
- **为何 clap**（用户定稿）：声明式类型安全、自动 help/error/`--version`、
  `clap_complete` 五 shell 补全（bash/zsh/fish/powershell/elvish）、冲突/条件校验；规模收益
  （serve/run/bench/doctor 定稿后总旗子 >50，手写解析维护成本已超库成本）。
- **completions 命令契约**（v2.4）改为：`clap_complete` 生成（不再手写模板）；词表覆盖
  已实现命令+旗子（与 help 同则）。
- **成本项记录**：依赖树增长（clap+clap_complete+derive）、编译时间、release 体积（~+0.8MB）、
  契约默认 override 工作——均有评估，定稿接受。

### v2.4 补全（五件套；用户 2026-08-27 定稿）

| 项 | 规则 | 先例 |
|---|---|---|
| `-V` / `--version` | 根旗打印版本（不存在 `version` 子命令——避免冗余面） | git/gh/cargo |
| `completions bash\|zsh\|fish` | 生成 shell 补全脚本（stdout；source 进配置）；覆盖已实现命令+旗子静态词表；不做语境级网络补全 | gh/kubectl/uv |
| `-v` / `-vv` / `--debug` | 全局诊断级别：默认（错误+警告）→ `-v`（详情）→ `-vv`（每步跟踪）→ `--debug`（全量含库内 trace）；作用于 stderr（机器输出面不变）；**须位于命令前**（gh 惯例） | gh/curl |
| `--no-color` | 禁色（TTY 着色仅在 `--no-color` 缺省且 NO_COLOR env 未设时启用） | gh/NO_COLOR 标准 |
| `--` 分隔 | `--` 后一切为位置参数（prompt/文件名以 `-` 开头场景；POSIX 通则） | git/curl |

不做（依据）：`login/logout`（私有仓 Non-Goal）· `-y/--yes`（无破坏性操作）· config 文件（env 体系已定型；P2 需要时再上）· man（help 已覆盖）· 自更新/遥测（未定分发）。

### v2.7 小件定稿（议题 1-5；用户 2026-08-27 批准"按建议"）

- **help 文档面**：英文惯例（repo 通用）；usage 首行 `reinfer <command> [args]`；`-h/--help` 根与
  子命令均可用（clap 自动排版 + 我们一行摘要/示例文案）；版本 banner `reinfer x.y.z`；
  未立项命令不进 help（ghost 禁则延续）。
- **错误消息体系**：两层——① 用法/参数错误：clap 统一格式（含 usage 行）→ exit 2；② 运行时
  失败：单行摘要 + 可选 hint 行（代理/离线提示风格保持）→ exit 1。错误一律 stderr；stdout 只承载
  结果数据（机器面纯净）。
- **进度条**：**不加 `--no-progress`**（quiet/json/dry-run 已覆盖全部抑制场景；降级矩阵即唯一标准）。
- **`model list` 展示**：size 列人类可读（GiB/MiB，>1GiB 显示两位小数+原始 bytes 在 json）；**不加
  --sort/--limit**（无场景）；--format json 保持原始字节/完整 sha/source。
- **别名**：**不加**（最小面；git 式 alias 属于 config 文件面= P2）。

### 实现迁移路径（v2.7；设计→实施交接）

```
阶段 A｜CLI 重写为 clap：bin/reinfer 解析层一次性迁移（derive + clap_complete；命令面
       download/model list/completions/help + 全局旗）；现存解析单测迁移为 try_parse 断言；
       行为契约全保留（§5 8 条适配清单逐条断言）。
阶段 B｜download 增强实现：进度回调（fetch_to_temp）→ 4 并发（manifest 互斥）→ --format/
       --quiet 输出层 → --dry-run。
阶段 C｜其余命令：serve/run/chat/bench/doctor 按各自 specification（005/003/008/002）立项后
       按本契约落位。
随行清理（阶段 A/B 内完成）：① 格式无关摘除（resolver/bin 3 处 .gguf 过滤 + 文案）；
② .env.example 移除 REINFER_MODEL_DIR 默认行；③ REINFER_SERVE_* 接入点；
④ 进度条 TTY 色彩接 --no-color/NO_COLOR。
```

**预留注记（各 spec 立项时细化，不入本契约）**：bench 指标定义（warmup/计时口径、-r/-l 缺省）·
chat REPL 交互细节（历史/编辑、Ctrl-C 语义）· serve 日志风格（结构化 vs 人类双式、访问日志面）
——届时以对应 spec 修约本契约。

## 6. 未来命令契约（未立项）

### serve（v2.1 定稿部分；功能落位见 005/003/P1-05）

```
reinfer serve <model> [-q <qtag> | -f <file>] [--revision <ref>] [--local-dir <dir>]
            [--device <id>] [--host <h>] [--port <p>]
            [--max-model-len <n>] [--max-num-seqs <n>] [--seed <n>]
            [--served-model-name <name>] [--api-key <key>] [--metrics]
```

- **模型引用三态**（ModelRef + loader 注册表候选探测）：① 本地文件/目录 → 直接加载（选择器旗给出 → exit 2）；② repo 单候选（单一权重结构）→ 自动 ensure 下载再运行（无参数）；③ repo 多候选（如量化共存仓）→ 必须 `-q`/`-f`，缺 → exit 2 并列出候选。隐式下载受 `REINFER_MODEL_AUTODOWNLOAD` 纪律（off → 不联网报错）。`run/chat/bench` 的 <model> 同此三态。
- **`--device <id>`**（用户 2026-08-27 定契约）：计算设备视图下标（0 基；CUDA = `CUDA_VISIBLE_DEVICES` 视图内序号；昇腾 = `DEVICE_ID` 语义）；缺省 `auto`（arch 探测自动选）。注：先例多为 env（llama.cpp/vllm 均为 env），本旗为我方扩展（语义与 env 视图完全一致）；文档标注用例。
- **功能面（P1 首期，用户拍板 OpenAI 兼容一次到位）**：`GET /v1/models` · `POST /v1/chat/completions`（含 SSE streaming）· `POST /v1/completions` · OpenAI 错误契约 · **`GET /healthz`**（合入 P1）· `--api-key` 认证 · graceful shutdown（SIGINT/SIGTERM）。采样参数（temperature 等）由 API 请求体承载，不进 CLI。
- 输出面：启动信息 + 运行日志（结构化/人类两式），与 download 的结果面（table/json/quiet）分离；`--metrics`（prometheus）挂 008。
- 不做（P1）：speculative decode、graph bucket（006）、radix cache/grammar（P3-01）、TP/PP/CP、多实例编排。

### run / chat / bench（v2.2 定稿；决策：OpenAI 缺省表 · run/chat 无 json · REPL 核心组现定）

```
run   <model> [-q|-f] [--device] [-n <max-tokens>] [-t <temp>] [--top-p <p>] [--top-k <k>]
             [--seed <s>] [--max-model-len <n>] [prompt...]
chat  <model> [-q|-f] [--device] [-n] [-t] [--top-p] [--top-k] [--seed] [--max-model-len]
bench <model> [-q|-f] [--device] [--max-model-len] [-r/--reps <n>] [-l/--seq-len <n>]
             [--format table|json]
```

- **工程缺省表（用户 2026-08-27 定：OpenAI API 缺省）**：`-t/--temperature=1.0` · `--top-p=1.0` ·
  `--top-k=关` · `-n/--max-tokens=模型上下文上限`。`--seed=<optional>`：给则全链确定（012 sampler host
  管线）；未给 = 随机。**缺省纪律边界**：仅约束"模型身份"选择（不得隐式选定模型/数据源）；
  计算参数允许标准工程缺省。
- **run**：`[prompt...]` 位置拼接；无 prompt → 读 stdin（llama-cli 惯例）；输出=token 流式打印
  （stdout），stats（tps/首 token/延迟）→ **stderr**（管道干净）；**无 --format**（决策②：文本面
  属语义直达；结构化消费属 API/streaming；stat 经 stderr 可取）。
- **chat**：同 run 参数子集（多轮保持 KV）；REPL 内 `/` 命令核心组（决策③现定）：`/help` · `/system <text>` ·
  `/clear` · `/temp <t>` · `/top-p <p>` · `/top-k <k>` · `/seed <n>` · `/quit`；退出 /quit 或 Ctrl-D/Ctrl-C。
- **bench**：`-r/--reps`、`-l/--seq-len`；指标=prefill tps·decode tps·首 token 延迟·峰值显存；
  `--format table|json`（数字面机读；008 gate_throughput.sh 消费 json）；gate 阈值逻辑在脚本层（008），不进 CLI。

### doctor（v2.3 定稿——原预置名 diag，用户 2026-08-27 定版改名）

```
reinfer doctor [--backend auto|cuda|ascend] [--format table|json|quiet] [--net]
```

- **语义**：环境体检（先例：flutter doctor / cargo doctor / npm doctor）——逐项 ✓/⚠/✗ 判定 +
  修复建议 + 结论行；**exit**：出现 ✗（阻塞级：无设备/无工具链/模型目录不可写）→ 1；
  仅 ⚠（建议级：如代理未设）→ 0。
- **检查块**：CUDA（driver/型号/多卡/显存视图）· CUDA 工具链（nvcc 位置与可用性——JIT 依赖）·
  CANN（version/aclnn/昇腾卡数）· 模型目录（存在/可写/余量）· 配置回显（`REINFER_*` 家族全部，
  含来源）· `--net` 追加 ModelScope/HF 连通性探针（**默认离线**，与 AUTODOWNLOAD 纪律同精神）。
- **输出**：flutter 风表格（`[✓]`/`[⚠]`/`[✗]` + 修复建议）；`--format json`（CI 环境门控）；
  `--format quiet`=只打阻塞项。`--backend` 限定栈（auto=双栈，缺省）。
- **命名决策依据**：doctor=体检判定+建议（flutter/cargo/npm 行业标准）；diag=事后采集（无统一
  先例）；本命令语义同候诊体检 → doctor；不设 diag 别名（保持命令面最小）。ASC-03 预置名同步。

### 其他

上述 run/chat/bench 的 <model> 三态同 serve（§6 serve）。

## 7. 先例对照表

| 规则 | 先例 |
|---|---|
| 动作式主命令 | vllm `serve/bench/chat` · gh `pr list` · uv `run` |
| 下载顶层 `download` | `hf download` · `modelscope download` |
| repo/模型位置参数 | `hf download gpt2 config.json` · vllm `serve <model>` |
| `file...` 位置文件 | hf |
| `--include/--exclude` | hf/modelscope 共同 |
| 无文件→全仓库 | `hf download REPO`（默认快照） |
| `--revision` | hf/modelscope |
| `--local-dir` | hf（kebab；modelscope `--local_dir` snake 不采纳） |
| `--dry-run` | hf |
| `--format json|quiet`（`--quiet`） | hf |
| 并发 4 | modelscope SDK `max_workers=4` |
| len 进度：每文件条 + GLOBAL 条 | docker pull / git clone 汇总 + hf/modelscope per-file 条 |
| 本地零参 `list` | docker `image ls` · ollama `list` · modelscope-ng `list` |
| `--flag=value` | git/docker/ffmpeg |
| `-q` 量化档 | 无直接先例（引擎领域词）；短旗形式普适 |

## 8. 实现状态

| 项目 | 状态 |
|---|---|
| download（file.../`-q`/--include/--exclude/--revision/--local-dir/无选择=全量）· model list | ✅ 已实现（67ced3a/494c877） |
| 默认目录 `~/.reinfer/models` 平台化 | ✅ 150122 实现＋本机迁移 |
| `--dry-run` / `--format` / `--quiet` / 4 并发 / 进度两层 | ⏸ **待定稿后实现**（用户已确认设计，实现动工等整体方案探讨完） |
| serve/chat/run/bench/doctor | 契约预置，未立项 |

## 附. 待办（等用户批准）

1. `.env.example` 移除 `REINFER_MODEL_DIR` 默认值行（"默认值不写进 env 模板"规则；本机 .env 由用户自便）。
2. 进度条设计如需对外可见性：README 示例截图更新（实现完成后）。

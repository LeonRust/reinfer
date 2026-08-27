# reinfer

[English](README.md) | [简体中文](README.zh-CN.md)

**reinfer** 是一款用 **Rust** 编写的内存安全、高吞吐 LLM 推理引擎，面向 **NVIDIA CUDA** 与 **昇腾 CANN**（华为 NPU），以单二进制形式交付。

> ⚠️ 状态：早期开发中。**CUDA**：运行时基座（L1）+ JIT 内核流水线（L2）已实现并经真机验证（RTX 5090，6/6 smoke）；L3 单请求流水线推进中。**模型获取**（纯 Rust、ModelScope 优先）已实现并经真实仓库验证。**昇腾**：L0 消费镜像已实现并经 NPU 验证（5/5 smoke）。服务能力（P1）随后落地。

## 亮点

- **内存安全的引擎核心** —— 调度器、radix 缓存、内存管理均 `#![forbid(unsafe_code)]`；所有 `unsafe` 收敛于窄 vendor-FFI crate
- **三档内核架构** —— 厂商预编译内核（FlashInfer cubin / CUTLASS / cuBLAS / CANN ACLNN）、JIT 内核（自有 CUDA C++ / AscendC 现场编译——数值主路径档），以及 Rust 原生内核（cudarc / CubeCL，保留档位）
- **JIT 内核流水线** —— 内核源码 → `nvcc -cubin` → 跨进程磁盘缓存（sha256 内容键、meta 提交点、flock+双检、跨进程单次编译）→ `cuLibraryLoadData` 启动；支持 `REINFER_CUDA_ARCH` 离线预烘焙；**设备自适应**：架构取自实测算力、工具链从已装候选中自动选择（无硬件特判默认）
- **高吞吐服务** —— 连续批处理、chunked prefill、token 预算准入、确定性 decode 批排序
- **Radix 前缀缓存** —— 跨请求 token 级前缀复用（RadixAttention 谱系）
- **结构化生成** —— llguidance 支撑的 grammar / JSON / FSM 约束，零 C FFI
- **量化** —— 兼容 GGUF Q4_0 / Q8_0 / K-quants / IQ 家族，FP8 / NVFP4 路径
- **单二进制** —— `server` / `cli` / `bench` 合于一个可执行文件；GPU 后端按 cargo feature 选择
- **模型获取** —— 纯 Rust ModelScope 客户端（无 Python）；`reinfer model list/get`（sha256 校验、原子落盘 + 运行时自动下载 `ModelResolver`），ModelScope 优先、可选 HuggingFace 回退
- **双硬件** —— NVIDIA 与昇腾（ACLNN + AscendC 内核）；JIT 缓存层为平台无关共享层、零 unsafe

## 模型获取

`reinfer model` 用纯 Rust 下载 GGUF 模型——无 Python、无 pip、无外部 CLI（`crates/models`，规范见 [`specs/013-model-fetch`](specs/013-model-fetch/spec.md)）：

```bash
# 列出本地已下载 GGUF（名/大小/sha256/来源；关联 manifest）
reinfer model list

# 列出远端仓库 GGUF 文件（名 / 大小 / sha256）
reinfer model ls-remote Qwen/Qwen2.5-0.5B-Instruct-GGUF

# 下载量化 GGUF（-q：量化段 → 文件名解析，校验大小 + sha256）
reinfer model get Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q8_0

# 精确文件 / 全部 GGUF / 自定义目录
reinfer model get Qwen/Qwen2.5-0.5B-Instruct-GGUF -f qwen2.5-0.5b-instruct-q8_0.gguf
reinfer model get Qwen/Qwen2.5-0.5B-Instruct-GGUF --all
reinfer model get Qwen/Qwen2.5-0.5B-Instruct-GGUF -q q8_0 --local-dir ~/models/reinfer
```

源优先级与下载策略由 env 控制（CLI 参数优先）：

| 变量 | 取值 | 缺省 | 语义 |
|---|---|---|---|
| `REINFER_MODEL_SOURCE` | `modelscope`/`huggingface`/`auto` | `auto` | `auto` = ModelScope 优先，缺（404/文件缺失）→ HuggingFace 回退 |
| `REINFER_MODEL_DIR` | 路径 | `~/models/reinfer` | 下载/查找根（`~` 自动展开） |
| `REINFER_MODEL_VERIFY` | `sha256`/`size`/`none` | `sha256` | 校验深度；HF 源缺 sha 字段 → 降级 ETag+size |
| `REINFER_MODEL_AUTODOWNLOAD` | `on`/`off` | `on` | `off` = 绝不联网（缺模型即报错） |
| `REINFER_MODEL_REPO`/`QUANT`/`FILE` | 仓库名、量化段、精确文件名 | — | 便捷注入（CLI 参数优先） |
| `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` | 标准 | — | 网络出口，如 `http://192.168.0.1:7890`；`NO_PROXY=...,modelscope.cn,huggingface.co` 可直连 |

下载流式写临时文件、经 ModelScope files API 的 sha256 校验（HF 为 ETag+size）、原子改名并记录
`manifest.json`；校验失败重试一次、失败即报错——不留半成品。`AUTODOWNLOAD=off` 令运行时完全离线。
引擎内**不硬编码任何模型标识**：仓库/文件名一律来自 CLI/env。

## 快速开始

```bash
rustup toolchain install   # stable（见 rust-toolchain.toml）
cargo build --release --features cpu   # 或：--features cuda / --features ascend
cargo run                 # 输出 "reinfer 0.1.0"（骨架阶段）
```

## 仓库结构

```
bin/reinfer     单二进制（server | cli | bench）
crates/         工作区 crate：core, gguf, arch, memory, cache, scheduler,
                kernels, samplers, grammar, ipc, cpu, jit, cuda, ascend, server
docs/design/    设计文档（分析、引擎设计、深入补充、真机留痕）
docs/rfcs/      RFC（变更宪法级规则所需）
specs/          SDD 规格（每功能 spec/plan/tasks；流程见 docs/sdd/README.md）
```

## 文档

- **Specs（SDD）** —— [`specs/`](specs/)（MVP、GGUF 加载器、CUDA 运行时基座、JIT L2、昇腾镜像；流程见 [Spec-Driven Development](docs/sdd/README.md)）
- **功能清单** —— [`docs/design/feature-list.md`](docs/design/feature-list.md)（带追踪关系的实施路线）
- **真机验证** —— [`docs/design/notes-jit-l2-2026-08-27.md`](docs/design/notes-jit-l2-2026-08-27.md)（CUDA L2 人工复查清单；昇腾见 `specs/011-ascend-l0-mirror/npu-test-checklist.md`）
- **项目宪法** —— [`CONSTITUTION.md`](CONSTITUTION.md)（贡献前必读）
- **AI 代理规则** —— [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md)
- **贡献指南** —— [`CONTRIBUTING.md`](CONTRIBUTING.md)

## 许可证

[Apache-2.0](LICENSE)

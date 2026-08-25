# reinfer

[English](README.md) | [简体中文](README.zh-CN.md)

**reinfer** 是一款用 **Rust** 编写的内存安全、高吞吐 LLM 推理引擎，面向 **NVIDIA CUDA** 与 **昇腾 CANN**（华为 NPU），以单二进制形式交付。

> ⚠️ 状态：早期开发中（工作区骨架，P0 阶段）。服务能力在 P1 落地。

## 亮点

- **内存安全的引擎核心** —— 调度器、radix 缓存、内存管理均 `#![forbid(unsafe_code)]`；所有 `unsafe` 收敛于窄 vendor-FFI crate
- **三档内核架构** —— 厂商预编译内核（FlashInfer cubin / CUTLASS / cuBLAS / CANN ACLNN）、Rust 原生内核（cudarc / CubeCL）、JIT/DSL 桥接（Triton / TileLang / AscendC）
- **高吞吐服务** —— 连续批处理、chunked prefill、token 预算准入、确定性 decode 批排序
- **Radix 前缀缓存** —— 跨请求 token 级前缀复用（RadixAttention 谱系）
- **结构化生成** —— llguidance 支撑的 grammar / JSON / FSM 约束，零 C FFI
- **量化** —— 兼容 GGUF Q4_0 / Q8_0 / K-quants / IQ 家族，FP8 / NVFP4 路径
- **单二进制** —— `server` / `cli` / `bench` 合于一个可执行文件；GPU 后端按 cargo feature 选择
- **双硬件** —— NVIDIA（sm90/100+，走 FlashInfer/CUTLASS）与昇腾（ACLNN + AscendC 内核）

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
                kernels, samplers, grammar, ipc, cpu, jit, cuda, can, server
docs/design/    设计文档（分析、引擎设计、深入补充）
docs/rfcs/      RFC（变更宪法级规则所需）
```

## 文档

- **Specs（SDD）** —— [`specs/`](specs/)（项目 MVP、GGUF 加载器；流程见 [Spec-Driven Development](docs/sdd/README.md)）
- **项目宪法** —— [`CONSTITUTION.md`](CONSTITUTION.md)（贡献前必读）
- **AI 代理规则** —— [`AGENTS.md`](AGENTS.md) / [`CLAUDE.md`](CLAUDE.md)
- **贡献指南** —— [`CONTRIBUTING.md`](CONTRIBUTING.md)

## 许可证

[Apache-2.0](LICENSE)

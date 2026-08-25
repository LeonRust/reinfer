# Spec: GGUF tokenizer (SPM/BPE, incremental decode)

> Status: proposal · Owner: maintainers · Created: 2026-08-25
> Dependency: specs/001-gguf-loader (tensor data) · Feeds: specs/003 T10, specs/005

## Problem Statement

`reinfer cli` needs prompt→token ids and streamed token→text. GGUF embeds `tokenizer.ggml.*` data (BPE/SPM/wordpiece models). reinfer must do this in pure Rust — no python, no torch (constitution §1.3) — with byte-accurate UTF-8 incremental decoding (surrogate handling for CJK/emoji per mini-sglang's read/surr offsets).

## Success Metrics

- **Text parity**: for Llama and Qwen tokenizers, `encode(prompt)` and incremental decode match llama.cpp token-by-token on 20 golden prompts × 2 models (100%).
- **Incremental decode**: streaming token→text with partial UTF-8 sequences reconstructs identical strings to batch decode; no replacement chars leaked at split boundaries (fuzz: 10k random byte strings).
- **Performance**: encode ≥ 100k chars/s; decode ≥ 50k chars/s on desktop CPU (log, not gated).
- **Safety**: any invalid tokenizer bytes → readable `TokenizerError`, no panic (fuzz harness).

## User Stories

1. As CLI user, `reinfer cli --model model.gguf "prompt"` returns the same tokens as llama.cpp.
2. As engine author, I build `Tokenizer` from a loaded GGUF file and use it for prefill encode + streaming decode.

## Acceptance Criteria

- [ ] New crate `crates/tokenizer` (reinfer-tokenizer): reads GGUF tokenizer metadata/tensors via 001's `GgufReader`; supports **SPM (sentencepiece-unigram & llama style)** and **BPE** model types
- [ ] `Tokenizer::encode(&str) -> Result<Vec<u32>, TokenizerError>`; special token handling (`<unk>/<bos>/<eos>, BOS/EOS flags in metadata`); tokenizer.ggml 合并顺序与 llama.cpp 一致（分数 + 字符段拆分规则）
- [ ] `IncrementalDecoder`: state machine (read_offset, surrogate buffer) — decode `[t; k]` chunk outputs zero-broken UTF-8; deterministic across any chunking (`decode_all(vec)` == `decode_one_by_one()`)
- [ ] Golden tests: Llama-3-8B user prompt set + Qwen2.5 prompts; goldens generated once by llama.cpp `-n 0` token dump, stored in `tests/golden/` (< 200 KB)
- [ ] Fuzz: `proptest` random unicode strings; roundtrip `decode(encode(s))` byte-identical (mod normalizing special tokens), injectable in CI (no GPU)

## Non-Goals

- Sentencepiece trainer / merges；WordPiece（遇到即报错并提示不支持）；多模态图像 tokenizer（P4）
- 速度优化（SIMD 字节表查找为 stretch）；训练侧工具

## Constraints

- Pure Rust; depends only on `reinfer-core` + `reinfer-gguf`; forbid unsafe (constitution §2.1)
- 张量读取复用 001（tokenizer 数据在 GGUF 中为 string/bool 元数据 + 少量 u32 tensors）
- 与 llama.cpp 行为对齐的语义：SPM 用 byte-level fallback 字节表（`bytes_to_unicode`），BPE 使用 `tokenizer.ggml.split.unicode.whitespace` 预处理

# Spec: GGUF tokenizer (SPM/BPE, incremental decode)

> Status: proposal · Owner: maintainers · Created: 2026-08-25
> Dependency: specs/001-gguf-loader (tensor data) · Feeds: specs/003 T10, specs/005

## Problem Statement

`reinfer cli` needs prompt→token ids and streamed token→text. GGUF embeds `tokenizer.ggml.*` data (BPE/SPM/wordpiece models). reinfer must do this in pure Rust — no python, no torch (constitution §1.3) — with byte-accurate UTF-8 incremental decoding (surrogate handling for CJK/emoji per mini-sglang's read/surr offsets).

## Success Metrics

- **Text parity**: `encode(prompt)` and incremental decode match llama.cpp (anchor: commit f280b2698) **token-by-token (100%)** on the parity matrix in `specs/000-project-mvp/parity.md` — BPE (Llama-8B, Qwen2.5-1.5B) + **SPM (Llama-2-7B)**. SPM is a first-class gate (not best-effort fuzz only).
- **Incremental decode**: streaming token→text with partial UTF-8 sequences reconstructs identical strings to batch decode; no leaked replacement chars at split boundaries (tiered fuzz, see AC).
- **Performance**: encode ≥ 100k chars/s; decode ≥ 50k chars/s on desktop CPU (log, not gated).
- **Safety**: any invalid tokenizer bytes → readable `TokenizerError`, no panic (fuzz harness).

## User Stories

1. As CLI user, `reinfer cli --model model.gguf "prompt"` returns the same tokens as llama.cpp.
2. As engine author, I build `Tokenizer` from a loaded GGUF file and use it for prefill encode + streaming decode.

## Acceptance Criteria

- [ ] New crate `crates/tokenizer` (reinfer-tokenizer): reads GGUF tokenizer metadata/tensors via 001's `GgufReader`; supports **SPM (sentencepiece-unigram & llama style)** and **BPE** model types
- [ ] `Tokenizer::encode(&str) -> Result<Vec<u32>, TokenizerError>`; special token handling (`<unk>/<bos>/<eos>, BOS/EOS flags in metadata`); tokenizer.ggml 合并顺序与 llama.cpp 一致（分数 + 字符段拆分规则）
- [ ] `IncrementalDecoder`: state machine (read_offset, surrogate buffer) — decode `[t; k]` chunk outputs zero-broken UTF-8; deterministic across any chunking (`decode_all(vec)` == `decode_one_by_one()`)
- [ ] Golden tests: parity matrix (specs/000-project-mvp/parity.md) prompt sets; goldens generated from llama.cpp **`llama-tokenize --ids`** (this build's cli client no longer dumps with `-n 0`), stored in `tests/golden/` (< 200 KB) holding **ids AND piece texts**; anchors recorded per golden: llama.cpp commit f280b2698 + GGUF sha256 + convert_hf_to_gguf.py version + `--no-bos/--special` flags; CI job re-renders goldens at pinned build and diffs (drift → manual review)
- [ ] Fuzz (three tiers): curated boundary corpus (~50 entries: whitespace splits U+3000/U+0085/U+2028, SPM byte-fallback rare chars, special tokens, empty string, multi-byte chars crossing token boundary) + random unicode strings + random token-id sequences (incl. out-of-range ids); assertions: any chunking of decode == `decode_all`; bad ids → readable error or `[UNK]`, never panic; injectable in CI (no GPU)

## Non-Goals

- Sentencepiece trainer / merges；WordPiece（遇到即报错并提示不支持）；多模态图像 tokenizer（P4）
- 速度优化（SIMD 字节表查找为 stretch）；训练侧工具

## Constraints

- Pure Rust; depends only on `reinfer-core` + `reinfer-gguf`; forbid unsafe (constitution §2.1)
- 张量读取复用 001（tokenizer 数据在 GGUF 中为 string/bool 元数据 + 少量 u32 tensors）
- 语义对齐目标：llama.cpp commit f280b2698（其 `llama-vocab.cpp` 含 Qwen 特例；SPM byte-fallback/BPE 拆分细节见 plan.md D3 与 Reference assets）

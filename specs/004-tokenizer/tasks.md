# Tasks: GGUF tokenizer

> Derived from specs/004-tokenizer/plan.md

## Task 1: crate scaffold + GGUF 数据读取

- `crates/tokenizer`（`publish = false` + `[lints] workspace = true` + `#![forbid(unsafe_code)]`）；上游数据读取：`tokenizer.ggml.model` 等键集（spm/bpe + 特殊 token flags + merges 张量）
- Verification: 对 001 的 golden GGUF 输出 key 集与 llama.cpp `gguf` 工具一致；`cargo test -p reinfer-tokenizer` 绿

## Task 2: SPM encode

- byte-level unicode 映射 + llama 风格最贪心匹配（unk fallback；`add_bos` 规则）
- Verification: `tests/golden/` 中 Llama 提示词集 encode == golden ids（100%）

## Task 3: BPE encode

- merges 表 → 字符级 rune 表；`split.unicode.whitespace` 预处理；Qwen2 特例（分数比较 + 长字符优先）
- Verification: Qwen2.5 golden 集 == golden ids（100%）

## Task 4: IncrementalDecoder

- read/surr 状态机：任意 `[0..k]` 切分下 `decode_chunk` 串接 == `decode_all`；坏 id → `[UNK]`（对齐 llama.cpp）
- Verification: 单测（ASCII/CJK/emoji/多字节触发边界）+ proptest（随机 unicode 字符串，两路等价性）

## Task 5: 集成与 fuzz

- `bin/reinfer` 接入：`cli` 子命令在 `--backend cpu/cuda` 均先经 Tokenizer；错误路径可读
- Verification: `reinfer cli --backend cpu --model Llama-8B-Q8_0.gguf "prompt"` 首 token 序列 == golden；fuzz 10k 字符串无 panic；`cargo fmt/clippy/test` 全绿

---

Completion gate: Tasks 1–5 accepted; golden 100% for both models; fuzz green. Feeds 003 T10 (parity test) and 005.

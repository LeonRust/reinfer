# Tasks: GGUF loader and typed core layer

> From specs/001-gguf-loader/plan.md · Each task is independently verifiable; verification = acceptance criterion

## Task 1: Core types (reinfer-core)

- Add `DType` enum (F16/BF16/FP32/Q8_0/Q4_0 placeholder for more), `TensorId(u32)`, `Error` (thiserror)
- Verification: `cargo test -p reinfer-core` green; `#![forbid(unsafe_code)]` compiles

## Task 2: GGUF header + metadata parsing (reinfer-gguf)

- Implement magic/version/alignment validation, metadata KV table with typed enum keys, string/array/u32/u64 types
- Verification: unit tests with hand-crafted byte fixtures; `cargo test -p reinfer-gguf`

## Task 3: Tensor table + mmap byte views

- Parse tensor info records (name/shape/offset/size/alignment), build `name → TensorId` map, expose lazy `bytes()`
- Verification: golden-file test (tiny GGUF < 10 MB in `tests/data/`) where names/shapes match llama.cpp output; proptest on aligned offsets

## Task 4: Quant codecs (reference, naive)

- `QuantCodec` trait + `Q8_0` and `F16`/`FP32` codecs (naive scalar); `Q4_0` codec
- Verification: dequantized golden blocks within 1 ULP of precomputed reference; proptest random roundtrip (bytes → f32 sanity)

## Task 5: `reinfer info` CLI subcommand

- Wire reader into `bin/reinfer` `info` subcommand; print architecture/dims/quant table
- Verification: `cargo run -- info tests/data/*.gguf` prints correct table; bad-file path returns a readable error (exit code 1, no panic)

## Task 6: Differential tie-in (slice completion gate)

- Commit a small script/test harness comparing `info` + naive dequant against llama.cpp output on the same golden file
- Verification: CI job `differential` passes; 60%-baseline bench report recorded in docs/bench (decode latency, P0 fallback threshold)

---

Completion gate: all Tasks 1–6 accepted; `cargo check --workspace` + fmt + clippy clean; reviewer approval (spec.md acceptance criteria all checked).

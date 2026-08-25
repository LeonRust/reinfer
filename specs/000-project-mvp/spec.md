# Spec: reinfer Project MVP (P0) — CPU feasibility loop

> Status: proposal · Owner: maintainers · Created: 2026-08-25 · Constitution: CONSTITUTION.md §6.4

## Problem Statement

All core architectural assumptions of reinfer (narrow FFI, three-tier kernels, direct GGUF loading, pure-Rust scheduler for later) have not yet been proven by a running product. The MVP's goal is the cheapest possible proof that **a pure-Rust CPU inference loop is implementable and numerically correct** — from a GGUF file to streaming token output, aligned with llama.cpp as the numeric reference — so that P1 (vendor CUDA kernels + serving) can start from a trusted base.

## Success Metrics

- **Numeric parity**: 20 golden prompts across Llama-8B-Q8_0 and Qwen2.5-1.5B-Q8_0 — `reinfer cli` output must match llama.cpp (same weights, same machine) **token-by-token, 100%**; FP16 weights allow cumulative logits drift ≤ 1e-4
- **Performance baseline**: Q8_0 decode throughput ≥ **60%** of the llama.cpp CPU backend on the same machine/core count
- **Engineering gates**: `cargo check --workspace` + `cargo test` (including differential tests) green without GPU; `cargo fmt --check` and `clippy -D warnings` clean
- **Resource envelope**: loading a 3 GB Q8_0 model → RSS peak ≤ model file size × 1.15 (mmap weights + streaming pages)
- **Distribution**: release single binary ≤ 40 MB (cpu feature only)

## User Stories

1. As a CLI user, I can run `reinfer cli --model model.gguf "prompt"` and receive streamed continuation text.
2. As a kernel author, I can unit-test a kernel against a naive reference and get element-wise-identical results on CPU.
3. As a maintainer, I can catch numeric or performance regressions in GPU-less CI.

## Acceptance Criteria

- [ ] `reinfer info model.gguf` and `reinfer cli` subcommands work with complete `--help`
- [ ] GGUF Q8_0 + F16 weights load and infer for the Llama architecture (GQA, RoPE, KV cache); Qwen2 architecture at parse level only (P0 minimizes arch surface)
- [ ] Differential tests: naive kernel vs SIMD kernel vs llama.cpp reference (three-way, run in CI)
- [ ] crates/gguf, crates/arch, crates/cpu contain no unsafe (`#![forbid(unsafe_code)]`)
- [ ] CLI error paths (bad file, truncated file, unknown architecture) return readable errors, never panic

## Non-Goals

- CUDA / CANN backends (P1+); HTTP serving, radix cache, speculative decoding, structured generation (P3)
- Quantization conversion tooling (reuse llama.cpp `convert_hf_to_gguf.py` to produce test weights)
- Multimodal; training/finetuning; beam/grammar advanced sampling (samplers limited to top-p/top-k/greedy)
- Full coverage of the 149 llama-arch classes (P0 keeps Llama + Qwen2 only)

## Constraints

- Rust edition 2024; torch dependency forbidden (constitution §1.3); GGUF layout compatible with current llama.cpp format (alignment=32)
- Core crates `#![forbid(unsafe_code)]`; mmap read-only (memmap2); numeric reference = llama.cpp with the same parameters
- Apache-2.0; CI gates per constitution §3.4; performance regression >5% is blocking

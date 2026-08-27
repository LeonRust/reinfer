#!/usr/bin/env bash
# Generate the Q8_0 dequantization golden (truth values) for crates/gguf.
#
# Status: SKELETON (014 T2). Not wired into CI yet — it needs two upstream
# deliverables that arrive later:
#   1. llama.cpp pinned build (014 T10, D7): llama-quantize from
#      f280b2698 (CUDA build, -DLLAMA_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<arch>)
#   2. A real-model archive downloaded through the 013 resolver
#      (REINFER_MODEL_REPO=... REINFER_MODEL_QUANT=q8_0, see .env.example)
#
# Planned flow:
#   1. Download the q8_0 GGUF via `reinfer model get --quant q8_0`.
#   2. Use llama-quantize's reference output path (dequantize_row_q8_0) to dump
#      one weight tensor's dequantized f32 bit patterns as golden bytes.
#   3. Compare against crates/gguf dequantize_q8_0 (bit-exact, 0 ulp per 014 r1).
#
# The module itself is already backed by byte-level golden fixtures and an
# all-65536-pattern half conversion cross-check; this script only adds the
# llama.cpp truth anchor.
set -euo pipefail

echo "skeleton: not wired yet (needs llama-quantize build + model archive)"

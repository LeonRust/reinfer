#!/usr/bin/env bash
# Generate the Q8_0 dequantization golden (truth values) — 014 T2 (wired).
#
# Sequence:
#   1. Extract `N_BLOCKS` raw Q8_0 blocks from a real-model archive tensor
#      (REINFER_MODEL_GGUF — env-injected; no hardcoded model identity).
#   2. Decode the same blocks with the referee dequantize_row_q8_0
#      (libggml-cpu of the f280b2698 build = 014 T0).
#   3. Emit tests/golden/q8_0_<n>.json: {"source":"llama.cpp f280b2698
#      dequantize_row_q8_0","input_sha256":..., "blocks":[{"hex":..,"f32_hex":[..]}]}
#   4. crates/gguf `codes::dequantize_q8_0` must reproduce f32_hex bit-exact
#      (checked by tests/golden/q8_0_golden.rs, 014 T2 gate).
#
# Usage: scripts/golden/gen_q8_0_golden.sh [tensor-name] [num-blocks]
# Env:  REINFER_MODEL_GGUF=... (required)
#       REINFER_REFEREE_BIN=<dir> (default ../../llama.cpp/build/bin)
#       REINFER_REPO=<root>       (default: repo root)
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="${REINFER_REPO:-$(cd "$here/../.." && pwd)}"
gguf="${REINFER_MODEL_GGUF:?set REINFER_MODEL_GGUF to the archive path}"
bin="${REINFER_REFEREE_BIN:-$repo/../llama.cpp/build/bin}"
tensor="${1:-token_embd.weight}"
nblocks="${2:-64}"
out="$repo/tests/golden/q8_0_${nblocks}.json"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

gcc -O2 -o "$work/q8_0_refdump" "$here/q8_0_refdump.c" \
  -I"$repo/../llama.cpp/ggml/include" -I"$repo/../llama.cpp/ggml/src" \
  -L"$bin" -lggml-cpu -lggml-base -lggml \
  -Wl,-rpath,"$bin" -lm

# (1) raw blocks hex from the archive (our reader = the unit under test for
#     the dequant path; the extractor itself is format-only and unaffected
#     by quant semantics).
(cd "$repo" && cargo run -q -p reinfer-gguf --example dump_block_bytes -- "$gguf" "$tensor" "$nblocks") > "$work/blocks.hex"

# (2) referee reference f32 bit patterns.
python3 - "$work/blocks.hex" "$work/blocks.bin" <<'PY'
import sys
rows = [l.strip() for l in open(sys.argv[1]) if l.strip()]
open(sys.argv[2], 'wb').write(b''.join(bytes.fromhex(r) for r in rows))
PY
"$work/q8_0_refdump" < "$work/blocks.bin" > "$work/ref.bits"

# (3) golden JSON.
python3 - "$work/blocks.hex" "$work/ref.bits" "$out" "$nblocks" <<'PY'
import hashlib, json, sys
blocks_hex, ref_bits, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
rows = [l.strip() for l in open(blocks_hex) if l.strip()]
bits = [l.strip() for l in open(ref_bits) if l.strip()]
assert len(rows) == n and len(bits) == n*32, f"{len(rows)} blocks, {len(bits)} ref rows (want {n}, {n*32})"
blob = b''.join(bytes.fromhex(r) for r in rows)
entries = [{"hex": rows[i], "f32_hex": bits[i*32:(i+1)*32]} for i in range(n)]
doc = {
    "source": "llama.cpp f280b2698 dequantize_row_q8_0 (via libggml-cpu, 014 T0 referee)",
    "input_sha256": hashlib.sha256(blob).hexdigest(),
    "block_bytes": 34,
    "blocks": entries,
}
json.dump(doc, open(out, "w"), ensure_ascii=False, indent=1)
print(f"golden written: {out} (blocks: {n}, sha256 {doc['input_sha256'][:16]})")
PY

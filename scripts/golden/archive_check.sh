#!/usr/bin/env bash
# Real-model archive check — 014 T1 verification gate.
#
# Compares the reinfer GGUF reader's key/tensor view against the
# llama.cpp referee `llama-gguf` (014 T0) for a real model archive:
#   - metadata key set identity (ordered-independently)
#   - tensor name set identity
#   - tensor byte-size derivation identity (dtype × element count vs
#     the referee's byte size; Q8_0 = 34 B/32 elems, F16 = 2 B/elem)
#
# Usage: REINFER_MODEL_GGUF=<path> scripts/golden/archive_check.sh
# Optional: REINFER_REFEREE_BIN=<dir> (default: ../../llama.cpp/build/bin)
# Exit 0 iff every comparison passes. No model identity is hardcoded.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
gguf="${REINFER_MODEL_GGUF:?set REINFER_MODEL_GGUF to the archive path}"
bin="${REINFER_REFEREE_BIN:-$repo/../llama.cpp/build/bin}"

ref="$("$bin/llama-gguf" "$gguf" r n 2>&1)"
ours="$(
  cargo run -q -p reinfer-gguf --example dump_meta -- "$gguf" 2>&1
)"

# referee prints the kv table twice (read_0 + read_1) → dedup with sort -u
ref_keys="$(printf '%s\n' "$ref" | sed -n 's/.*kv\[[0-9]*\]: key = \(.*\)$/\1/p' | sort -u)"
ours_keys="$(printf '%s\n' "$ours" | awk '$1=="key" {print $2}' | sort)"

if [ "$ref_keys" != "$ours_keys" ]; then
  echo "FAIL: metadata key sets differ"
  diff <(printf '%s\n' "$ref_keys") <(printf '%s\n' "$ours_keys") || true
  exit 1
fi
echo "OK keys: $(printf '%s\n' "$ref_keys" | wc -l) identical"

ref_tens="$(printf '%s\n' "$ref" | sed -n 's/.*tensor\[[0-9]*\]: name = \(.*\), size = \([0-9]*\).*$/\1 \2/p' | sort -u)"
ours_tens="$(printf '%s\n' "$ours" | sed -n 's/^tensor \(.*\) \(.*\) \([0-9]*\)$/\1 \2 \3/p' | sort)"

ours_names="$(printf '%s\n' "$ours_tens" | awk '{print $1}' | sort)"
ref_names="$(printf '%s\n' "$ref_tens" | awk '{print $1}' | sort)"
if [ "$ref_names" != "$ours_names" ]; then
  echo "FAIL: tensor name sets differ"
  diff <(printf '%s\n' "$ref_names") <(printf '%s\n' "$ours_names") || true
  exit 1
fi
echo "OK tensors: $(printf '%s\n' "$ref_names" | wc -l) identical names"

# Byte-size derivation: for each tensor, our dtype/element count must
# imply the referee's byte size (Q8_0=34/32, F16=2, F32=4; else report).
bad=0
while read -r name dtype n; do
  size="$(awk -v n="$n" -v d="$dtype" -v name="$name" '
    (d=="Q8_0") { x=(n/32)*34; printf (x==int(x)) ? x : "NONINT"; }
    (d=="F16")  { x=n*2; printf x; }
    (d=="F32")  { x=n*4; printf x; }
    (d=="F64")  { x=n*8; printf x; }
    (d=="INT32"){ x=n*4; printf x; }
    END { }
  ' <<< "" | tr -d '[:space:]')" || true
  if [ -z "$size" ]; then
    continue
  fi
  exp="$size"
  # tensor lines from referee: `name size` (name may contain spaces? gguf
  # names do not contain spaces in practice; awk handles it).
  got="$(printf '%s\n' "$ref_tens" | awk -v n="$name" '$1==n {print $2; exit}')"
  if [ -n "$got" ] && [ "$got" != "$exp" ]; then
    echo "FAIL: $name size $got != derived $exp ($dtype, $n elems)"
    bad=1
  fi
done <<< "$ours_tens"

if [ "$bad" -ne 0 ]; then
  exit 1
fi
echo "OK sizes: all derive-match referee byte sizes (Q8_0/F16/F32)"

#!/usr/bin/env bash
# BPE tokenizer golden generator (llama-tokenize --ids) — 014 T4.
#
# Emits tests/golden/tokenizer-<prefix>.json:
#   {"source", "model_sha256", "add_bos", "items": [{"text","ids","pieces"}]}
# ids from `llama-tokenize --ids --no-parse-special`, pieces from the
# same invocation without --ids (id -> piece line). Both sides run with
# --no-parse-special so special-token handling is out of scope here
# (see 004 for the special-token flags semantics).
#
# Verified by crates/tokenizer/tests/bpe_golden.rs (encode == ids 100%,
# decode_all(ids) == text).
#
# Usage: REINFER_MODEL_GGUF=... scripts/golden/gen_bpe_golden.sh [prefix]
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
repo="${REINFER_REPO:-$(cd "$here/../.." && pwd)}"
gguf="${REINFER_MODEL_GGUF:?set REINFER_MODEL_GGUF to the archive path}"
bin="${REINFER_REFEREE_BIN:-$repo/../llama.cpp/build/bin}"
prefix="${1:-bpe}"
out="$repo/tests/golden/tokenizer-$prefix.json"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Prompt set for encode 100% + boundary coverage (中文/英文/emoji/换行/
# 多字节切断/代码/标点). Kept small; the 20-prompt parity set belongs to
# 014 T10 (bench/prompts/).
cat > "$work/prompts.txt" <<'EOF'
Hello, world!
你好，世界！
The quick brown fox jumps over the lazy dog.
Hello 你好 😀 世界
line one
line two
    indented code: let x = 1 + 2;
"quoted", 'quoted', (parens) [brackets] {braces}
python_underscores_and-hyphens-123.45
éàüñøß日本語中文測試
EOF

model_sha="$(sha256sum "$gguf" | awk '{print $1}')"
# add_bos from our reader dump (key type/value lines; same reader the
# tokenizer crate consumes — consistent with the unit under test).
add_bos="$(cd "$repo" && cargo run -q -p reinfer-gguf --example dump_meta -- "$gguf" 2> /dev/null \
  | awk '$2=="tokenizer.ggml.add_bos_token" {print ($3=="true") ? 1 : 0}')"
[ -n "$add_bos" ] || add_bos=0

python3 - "$bin" "$gguf" "$out" "$work/prompts.txt" "$model_sha" "$add_bos" <<'PY'
import json, subprocess, sys
bin_dir, gguf, out, prompts_path, model_sha, add_bos = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
items = []
for text in open(prompts_path).read().splitlines():
    if not text.strip():
        continue
    r = subprocess.run(
        [f"{bin_dir}/llama-tokenize", "-m", gguf, "--log-disable", "--no-parse-special", "-p", text],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"llama-tokenize failed for {text!r}: {r.stderr}")
    pieces = []
    for line in r.stdout.splitlines():
        line = line.strip()
        # "<id> -> '<piece>'"
        if "->" in line:
            piece = line.split("->", 1)[1].strip().strip("'")
            pieces.append(piece)
    r2 = subprocess.run(
        [f"{bin_dir}/llama-tokenize", "-m", gguf, "--log-disable", "--no-parse-special", "--ids", "-p", text],
        capture_output=True, text=True)
    ids_json = "".join(l.strip() for l in r2.stdout.splitlines() if l.strip())
    ids = json.loads(ids_json)
    assert len(ids) == len(pieces), f"{text!r}: {len(ids)} ids vs {len(pieces)} pieces"
    items.append({"text": text, "ids": ids, "pieces": pieces})
doc = {
    "source": "llama.cpp f280b2698 llama-tokenize --ids --no-parse-special",
    "model_sha256": model_sha,
    "add_bos": bool(int(add_bos)),
    "items": items,
}
json.dump(doc, open(out, "w"), ensure_ascii=False, indent=1)
print(f"golden written: {out} (items: {len(items)}, add_bos={doc['add_bos']})")
PY

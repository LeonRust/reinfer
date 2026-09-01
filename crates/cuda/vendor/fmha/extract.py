#!/usr/bin/env python3
"""Extract the FMHA kernel header closure from a flash-attn upstream clone.

Usage: python3 extract.py <upstream_dir> <out_dir>

* upstream_dir: flash-attn v2.8.3 clone with submodules checked out
  (csrc/flash_attn/src + third_party/cutlass/include).
* out_dir: destination root (vendor/fmha/headers/). FA2 files land at the
  root, cutlass/cute files keep their nested paths.

Resolution rules mirror nvcc:
  - `#include "X"`  -> first search next to the including file, then cutlass/include
  - `#include <X>`  -> cutlass/include only (system headers stay in the toolkit)
Roots: the FA2 src files that flash_fwd_kernel.h pulls in.
Every copied file keeps its original license header (asserted).
"""

import os
import re
import shutil
import sys

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"].*$', re.M)

FA2_SRC = os.path.join("csrc", "flash_attn", "src")
CUTLASS_INC = os.path.join("csrc", "cutlass", "include")

# Roots: flash_fwd_kernel.h plus the FA2-src files it (and they) include.
ROOTS = [
    "flash_fwd_kernel.h",
    "block_info.h",
    "kernel_traits.h",
    "utils.h",
    "softmax.h",
    "mask.h",
    "dropout.h",
    "rotary.h",
    "namespace_config.h",
    "philox.cuh",
]

# FA2-src files that are *not* in the kernel include chain (kept out on
# purpose; add to ROOTS if the closure misses something).
SKIP = {"static_switch.h", "flash_fwd_launch_template.h", "flash.h", "flash_bwd_kernel.h"}


def collect_roots(upstream):
    fa2 = os.path.join(upstream, FA2_SRC)
    missing = [r for r in ROOTS if not os.path.isfile(os.path.join(fa2, r))]
    if missing:
        sys.exit(f"missing FA2 root files: {missing}")
    return {os.path.join(fa2, r): os.path.basename(r) for r in ROOTS}


def resolve(inc, cur_dir, cutlass_inc, cur_path, found):
    if inc in SKIP:
        return None
    if os.path.isabs(inc):
        return None
    if inc.startswith(("stddef", "cuda", "cooperative_groups", "ATen", "torch")):
        return None  # toolkit/system/torch -> external
    cands = []
    if inc.startswith(("cutlass/", "cute/", "cub/")):
        cands.append(os.path.join(cutlass_inc, inc))
    else:
        cands.append(os.path.join(cur_dir, inc))
        cands.append(os.path.join(cutlass_inc, inc))
    for c in cands:
        if os.path.isfile(c):
            return os.path.normpath(c)
    return None


def main():
    upstream, out = sys.argv[1], sys.argv[2]
    cutlass_inc = os.path.join(upstream, CUTLASS_INC)
    if not os.path.isdir(cutlass_inc):
        sys.exit(f"cutlass include not found under {upstream} (submodules not checked out?)")

    queue = list(collect_roots(upstream).keys())
    copied = {}
    while queue:
        cur = queue.pop()
        if cur in copied:
            continue
        with open(cur, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for m in INCLUDE_RE.finditer(text):
            inc = m.group(1)
            tgt = resolve(inc, os.path.dirname(cur), cutlass_inc, cur, copied)
            if tgt is not None:
                queue.append(tgt)
        if cur.startswith(os.path.join(upstream, FA2_SRC)):
            rel = os.path.basename(cur)
        else:
            rel = os.path.relpath(cur, cutlass_inc)
        copied[cur] = rel

    for src, rel in sorted(copied.items()):
        dst = os.path.join(out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        with open(dst, encoding="utf-8", errors="replace") as f:
            head = f.read(512)
        if "Copyright" not in head and "LICENSE" not in head:
            print(f"WARN: no license header detected in {rel}", file=sys.stderr)

    print(f"copied {len(copied)} files to {out}")
    for rel in sorted(copied.values()):
        print("  " + rel)


if __name__ == "__main__":
    main()

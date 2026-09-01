#!/usr/bin/env bash
# perf-gate.sh — one-line decode regression gate for the reinfer engine (006 T7 / S1-8).
#
# Protocol (bench/gate-fixture.md; machine-readable values = bench/gate-fixture.json):
#   reference = llama.cpp CUDA single-stream decode median (bench/baseline-llamacpp.json)
#   gate      = 0.85 x reference  (wave acceptance; today 299.8 tok/s)
#   ci_red    = 0.90 x reference  (CI red line: median <= 0.9 x baseline => red; today 317.4)
#   benchmark-gap-2026-08-29.md SS4 ladder = expectation track only (record, NOT a gate).
#
# Flow:
#   1) cargo build --release --features cuda          (workspace default-members = bin/reinfer,
#      so no ascend crate is built)
#   2) require bench/baseline-llamacpp.json           (missing => re-run the T0 reference first:
#      llama-bench -b 1 -n 512 -fa 1 -ngl 99 -r 5 on the pinned llama.cpp build)
#   3) measure single-stream decode via the bench-vs-vllm harness:
#      ../bench-vs-vllm/run_all.py --engine reinfer --suite perf_c1
#      (this script only INVOKES harness scripts; it never holds locks or writes into
#      bench-vs-vllm — start/stop/healthz are managed by start_servers.sh inside run_all.py)
#   4) parse results/reinfer/perf_c1_<seed>.jsonl -> median tpot over non-warmup,
#      error-free requests -> tok/s = 1 / median_tpot_s
#   5) verdict vs gate / ci_red -> PASS (exit 0) / FAIL (exit 1)
#
# Usage:
#   bench/perf-gate.sh                          # build + measure + verdict (default)
#   bench/perf-gate.sh --skip-build             # reuse the existing release binary
#   bench/perf-gate.sh --seed 42                # perf seed (must match run_all --seed; default 42)
#   bench/perf-gate.sh --update-baseline <tok/s>  # record a NEW llama.cpp reference baseline
#   bench/perf-gate.sh --dry-run                # parse-only (no build/measure; offline test)
#
# Environment:
#   REINFER_CUDA_NVCC=/usr/local/cuda-13.2/bin/nvcc  (required for sm_120a JIT on this machine)
#   CUDA_VISIBLE_DEVICES=0
#   Both are inherited by the harness (start_servers.sh sources its own CUDA 13.2 env).
#   Optional path overrides (offline tests / alternate results):
#     PERF_GATE_HARNESS  PERF_GATE_BASELINE  PERF_GATE_FIXTURE  PERF_GATE_RESULTS
#
# Recording: per wave, manually fill commit + build flags in bench/gate-fixture.md (SS 4).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
HARNESS_DIR="${PERF_GATE_HARNESS:-$REPO_ROOT/../bench-vs-vllm}"
BASELINE="${PERF_GATE_BASELINE:-$REPO_ROOT/bench/baseline-llamacpp.json}"
GATE_FIXTURE="${PERF_GATE_FIXTURE:-$REPO_ROOT/bench/gate-fixture.json}"
BINARY="$REPO_ROOT/target/release/reinfer"

SKIP_BUILD=0
SEED=42
MODE="gate"

usage() {
  sed -n '2,37p' "$0" | sed 's/^# \{0,1\}//'
  exit 2
}

# ---------------------------------------------------------------- args
NEW_REF=""
while [ $# -gt 0 ]; do
  case "$1" in
    --skip-build) SKIP_BUILD=1 ;;
    --seed) SEED="${2:-42}"; shift ;;
    --update-baseline) MODE="update-baseline"; NEW_REF="${2:-}"; shift ;;
    --dry-run) MODE="dry-run" ;;
    -h|--help) usage ;;
    *) echo "perf-gate.sh: unknown argument '$1'"; usage ;;
  esac
  shift
done
RESULTS_JSONL="${PERF_GATE_RESULTS:-$HARNESS_DIR/results/reinfer/perf_c1_${SEED}.jsonl}"

command -v python3 >/dev/null 2>&1 || { echo "perf-gate.sh: python3 required (json parsing)"; exit 2; }

# ---------------------------------------------------------------- mode: update-baseline
if [ "$MODE" = "update-baseline" ]; then
  [ -n "$NEW_REF" ] || { echo "perf-gate.sh: --update-baseline requires <tok/s>"; exit 2; }
  python3 - "$BASELINE" "$GATE_FIXTURE" "$NEW_REF" <<'PY'
import datetime, json, os, sys, tempfile
baseline_path, fixture_path, new_ref = sys.argv[1], sys.argv[2], float(sys.argv[3])
if not (50.0 <= new_ref <= 2000.0):
    print(f"perf-gate.sh: implausible reference {new_ref} tok/s (expected 50..2000); aborted")
    sys.exit(2)
for p in (baseline_path, fixture_path):
    if not os.path.isfile(p):
        print(f"perf-gate.sh: missing {p}; cannot update baseline")
        sys.exit(2)
with open(baseline_path, encoding="utf-8") as f:
    base = json.load(f)
old_ref = base.get("reference_tok_s") or base.get("result_tg512", {}).get("median_5")
history = base.get("history", [])
history.append({
    "date": base.get("date"),
    "reference_tok_s": old_ref,
    "median_5": base.get("result_tg512", {}).get("median_5"),
    "commit": base.get("commit"),
    "note": "superseded by --update-baseline; commit/build flags re-fill manually in gate-fixture.md",
})
base["history"] = history
base["date"] = datetime.date.today().isoformat()
base["reference_tok_s"] = new_ref
base["commit"] = None  # manual fill: gate-fixture.md keeps the authoritative record
base["result_tg512"]["median_5"] = new_ref  # runs_5x_r1 stays as the historical raw record
gm = base.setdefault("gate_math", {})
gm["reference_tok_s"] = new_ref
gm["engine_target_0_85x"] = round(0.85 * new_ref, 1)
gm["ci_red_0_9x"] = round(0.9 * new_ref, 1)
def atomic_write(path, obj):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)
atomic_write(baseline_path, base)
with open(fixture_path, encoding="utf-8") as f:
    fx = json.load(f)
fx["reference_tok_s"] = new_ref
fx["gate_0_85x"] = round(0.85 * new_ref, 1)
fx["ci_red_0_9x"] = round(0.9 * new_ref, 1)
atomic_write(fixture_path, fx)
print(f"perf-gate.sh: baseline updated: reference {old_ref} -> {new_ref} tok/s "
      f"(gate {gm['engine_target_0_85x']}, ci_red {gm['ci_red_0_9x']})")
print("perf-gate.sh: MANUAL FILL: commit/build flags -> bench/gate-fixture.md; run the gpu-less "
      "fixture test (cargo test -p reinfer gate_fixture_verdict_cases) to re-lock the values")
PY
  exit $?
fi

# ---------------------------------------------------------------- 1) build
if [ "$MODE" = "dry-run" ]; then
  echo "[gate] dry-run: parse-only (no build, no measurement)"
elif [ "$SKIP_BUILD" = "1" ]; then
  [ -x "$BINARY" ] || { echo "perf-gate.sh: --skip-build but $BINARY missing"; exit 2; }
  echo "[gate] reuse $BINARY (--skip-build)"
else
  echo "[gate] build release (--features cuda)..."
  cargo build --release --features cuda || { echo "perf-gate.sh: build failed"; exit 2; }
fi

# ---------------------------------------------------------------- 2) reference check
if [ ! -f "$BASELINE" ]; then
  echo "perf-gate.sh: reference baseline missing: $BASELINE"
  echo "  -> re-run T0 (notes.md 2026-08-29 T0): pinned llama.cpp build +"
  echo "     llama-bench -m Qwen3-0.6B-f16.gguf -b 1 -n 512 -fa 1 -ngl 99 -r 5"
  echo "     then record the 5-run median into bench/baseline-llamacpp.json"
  exit 2
fi

# ---------------------------------------------------------------- 3) measure
if [ "$MODE" != "dry-run" ]; then
  if [ -n "${REINFER_CUDA_NVCC:-}" ]; then
    echo "[gate] REINFER_CUDA_NVCC=$REINFER_CUDA_NVCC"
  else
    echo "[gate] WARN: REINFER_CUDA_NVCC unset (sm_120a JIT needs nvcc 13.2 on this machine)"
  fi
  echo "[gate] measure: python3 run_all.py --engine reinfer --suite perf_c1 (bench-vs-vllm harness)..."
  python3 "$HARNESS_DIR/run_all.py" --engine reinfer --suite perf_c1 --seed "$SEED" \
    || { echo "perf-gate.sh: harness run failed (exit $?)"; exit 2; }
fi

# ---------------------------------------------------------------- 4+5) parse + verdict
python3 - "$RESULTS_JSONL" "$BASELINE" "$GATE_FIXTURE" <<'PY'
import json, os, statistics, sys

results_path, baseline_path, fixture_path = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.isfile(results_path):
    print(f"perf-gate.sh: no results at {results_path} (did perf_c1 finish?); run_all failed lists are fatal")
    sys.exit(2)
tpot = []
for line in open(results_path, encoding="utf-8"):
    row = json.loads(line)
    d = row.get("data", {})
    if d.get("is_warmup"):
        continue
    if d.get("error"):
        continue
    if d.get("tpot") is not None:
        tpot.append(d["tpot"])
if len(tpot) < 3:
    print(f"perf-gate.sh: only {len(tpot)} valid perf_c1 requests (need >= 3); measurement unusable")
    sys.exit(2)

with open(baseline_path, encoding="utf-8") as f:
    base = json.load(f)
ref = base.get("reference_tok_s") or base.get("result_tg512", {}).get("median_5")
gate = 0.85 * ref
ci_red = 0.9 * ref
med_tpot_s = statistics.median(tpot)
tok_s = 1.0 / med_tpot_s
errors = sum(1 for line in open(results_path, encoding="utf-8")
             if json.loads(line).get("data", {}).get("error"))
print(f"[gate] reference (llama.cpp CUDA) = {ref} tok/s")
print(f"[gate] thresholds: gate 0.85x = {round(gate, 1)} tok/s | ci_red 0.9x = {round(ci_red, 1)} tok/s")
print(f"[gate] measured: n={len(tpot)} req, median tpot {med_tpot_s*1000:.3f} ms, "
      f"tok/s = {tok_s:.2f}, errors={errors}")

if tok_s >= ci_red:
    print(f"[gate] PASS (GREEN): {tok_s:.1f} >= {ci_red:.1f} (0.9x) — no CI regression")
    verdict = 0
elif tok_s >= gate:
    print(f"[gate] PASS (CI RED): {tok_s:.1f} in [0.85x, 0.9x) — gate met, but median <= 0.9x "
          f"baseline flags the 10% CI-red criterion (006 T7, coexists with 000's 5% CPU tier)")
    print("[gate] record the value + attribution in bench/gate-fixture.md SS4 (manually fill)")
    verdict = 0
else:
    print(f"[gate] FAIL: {tok_s:.1f} < {gate:.1f} (0.85x gate) — decode gate NOT met")
    print("[gate] benchmark-gap SS4 ladder stays an expectation track (record, not a second gate)")
    verdict = 1
print("[gate] remember: manually fill engine commit + build flags -> bench/gate-fixture.md SS4")
sys.exit(verdict)
PY

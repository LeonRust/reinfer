#!/usr/bin/env bash
# 008-ci-infra D3：每个 `#[ignore]` 测试必须在 allowlist 中（= 有 GPU job 映射承诺）。
# gpu.yml 于 008 T2 落地后，allowlist 每行注释应指向对应 job 名。
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
ALLOWLIST="scripts/ci/ignored-tests.txt"

if ! command -v cargo >/dev/null 2>&1; then
  echo "🔶 无 cargo，跳过（仅 CI 环境强制执行）"
  exit 0
fi

IGNORED="$(
  cargo test --all-features --workspace --no-fail-fast -- --list 2>/dev/null \
    | grep -E ': ignored$' | sed 's/: ignored$//' | sort -u
)" || true # 空匹配时 grep 返回 1；pipefail 下视为正常

if [ -z "$IGNORED" ]; then
  echo "✅ 无 #[ignore] 测试"
  exit 0
fi

MISSING=""
while IFS= read -r t; do
  if ! grep -qxF "$t" "$ALLOWLIST" 2>/dev/null; then
    MISSING="$MISSING\n  - $t"
  fi
done <<< "$IGNORED"

if [ -n "$MISSING" ]; then
  echo "⛔ 以下 #[ignore] 测试不在 allowlist（008 D3 契约：每个 ignore = 必须有 GPU job 映射）：" >&2
  echo -e "$MISSING" >&2
  echo "   请将测试名补入 $ALLOWLIST 并标注对应 job；或取消 #[ignore]。"
  exit 1
fi

echo "✅ 全部 #[ignore] 测试均在 allowlist"

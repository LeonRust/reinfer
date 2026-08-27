#!/usr/bin/env bash
# 008-ci-infra D3：每个 `#[ignore]` 测试必须在 allowlist 中（= 有 GPU job 映射承诺）。
# 2026-08-27 评审修复（C-F1）：
#   - `cargo test -- --list` 对 #[ignore] 与普通测试同样打印 "name: test"，
#     必须用 `--list --ignored` 才能筛出 ignore 集合；
#   - 构建失败（如 cudarc build.rs 无 nvcc panic）必须让脚本整体失败，
#     不得被 `|| true`/stderr 重定向吞掉后输出 "✅ 无"（恒绿 = 假验证）。
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
ALLOWLIST="scripts/ci/ignored-tests.txt"

if ! command -v cargo >/dev/null 2>&1; then
  echo "🔶 无 cargo，跳过（仅 CI 环境强制执行——本地应安装）"
  exit 0
fi

# 1) 先完整跑 --list --ignored：失败（构建错误/无 nvcc）→ set -e 使脚本整体失败
LIST_FILE="$(mktemp)"
trap 'rm -f "$LIST_FILE"' EXIT
cargo test --all-features --workspace --no-fail-fast -- --list --ignored >"$LIST_FILE" 2>&1

# 2) 再解析：grep 空匹配是合法的（无 ignore 测试），用 || true 兜底
IGNORED="$(grep -E ': test$' "$LIST_FILE" | sed 's/: test$//' | sort -u || true)"

if [ -z "$IGNORED" ]; then
  echo "✅ 无 #[ignore] 测试"
  exit 0
fi

MISSING=""
while IFS= read -r t; do
  # 允许 allowlist 行带注释后缀：`name  # gpu.yml: job`
  if ! grep -qE "^${t}([[:space:]]+#.*)?\$" "$ALLOWLIST" 2>/dev/null; then
    MISSING="$MISSING\n  - $t"
  fi
done <<< "$IGNORED"

if [ -n "$MISSING" ]; then
  echo "⛔ 以下 #[ignore] 测试不在 allowlist（008 D3 契约：每个 ignore = 必须有 GPU job 映射）：" >&2
  echo -e "$MISSING" >&2
  echo "   请将测试名补入 $ALLOWLIST 并标注对应 job；或取消 #[ignore]。" >&2
  exit 1
fi

echo "✅ 全部 #[ignore] 测试均在 allowlist"

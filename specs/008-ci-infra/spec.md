# Spec: CI infrastructure (工件与门禁载体)

> Status: proposal · Owner: maintainers · Created: 2026-08-25 · Review demand: docs/design/review-2026-08-25.md §二/12（"CI 工件不存在"）
> 引用方：specs/003（差分/parity/3×CPU 门禁）、005（一致性套件/吞吐记录）、006（perf 门禁/baseline）、004（golden 重建任务）、002（ASC 三档）、**009（L1 真机 smoke，2026-08-27 接线）**

## Problem Statement

仓库目前**没有任何 CI 工件**：所有"GPU job / gpu-runner label / nightly / `#[ignore]` 矩阵"均为纸面描述，特征清单 ASC-05 曾误标 "ready (config only)"。本 spec 定义 CI 基础设施契约——jobs、runner 规范、`#[ignore]` 纪律、工件与缓存、基准门禁接线、触发与超时策略——供所有 spec 单点引用（禁止 spec 自行定义 job 细节）。

## Success Metrics

- **PR 主档**（无 GPU 触发）：`fmt / clippy(-D warnings) / test / cargo-deny` 全绿为合入门禁（A 档，硬）——任何不触及 GPU 的提交不得因 GPU 工件缺失而阻塞
- **GPU 档**（label 触发）：差分 + parity + 性能门禁按 003/005/006 接线表执行，PASS/FAIL 机器可判定（B 档，硬）
- **`#[ignore]` 纪律**：无 GPU 档可运行测试必须全绿；GPU 相关测试统一 `#[ignore]` 且有对应 job 项；"ignore 却无 job" 由检查脚本拦截（防假验证）
- **基准可复现**：`bench/baseline.json`（5 次中位数）nightly 更新；`--perf` 对比输出机器可读；回归门禁（label `perf` 时 ≤0.9× 基线 = 红）判定一致
- **Golden 防漂移**：004 的 golden 重建+diff 任务运行于固定 commit（f280b2698）并随 PR 警示漂移（人工 review 通过才更新 golden）

## User Stories

1. 作为维护者：PR CI 无 GPU 也给出可信信号；GPU 变更才消费 GPU runner（省资源、可排队）。
2. 作为服务/性能作者：`--perf` + CI 给出与 llama-bench 参数一致的对比，月度回归可视化。
3. 作为新增 spec 作者：写法=在本 spec 的接线表加一行，不自定义 job。

## Acceptance Criteria

- [ ] `.github/workflows/` 三个文件落地：`ci.yml`（PR 主档）、`gpu.yml`（label/路径触发：`nvidia-run`）、`bench.yml`（nightly：baseline 更新 + 性能报告）
- [ ] runner 规范文档化：标签（`nvidia-gpu`/`can-gpu`）+ 每 runner 单设备 + 硬件记录（GPU UUID、driver、cuBLAS、sm 型号 → parity.md 引用）
- [ ] `#[ignore]` 契约：测试按 `cuda`/`ascend` 特性分层；`scripts/ci/checked-ignores.sh` 校验"每个 ignore 测试有 job 清单项"（**须用 `--list --ignored`；构建失败必须 exit 1，禁止恒绿**，2026-08-27 评审 C-F1）；无 GPU 档仅 `cargo test --workspace`（不含 `--all-features`——它会因无 nvcc 失败）
- [ ] 缓存/工件：cargo build 缓存（按 lock 哈希）；JitCache 预烘焙（`REINFER_CUDA_ARCH`）入缓存并在 GPU job 恢复；golden/notes/baseline 为工件
- [ ] 门禁接线表（下文）逐行可判定；`perf` label 回归判定（≤0.9× 基线 = 红）有 fixture 测试
- [ ] 运行成本：main PR 主档 < 5 min；GPU 档排队/超时/并发限制（并发=1 runner 自持）声明
- [ ] 触发与放行：commit 到 main 不自动跑 GPU 档；GPU 档仅当 (a) label `gpu-test` 或 (b) paths 命中 `crates/cuda/**` 或 (c) 手动 dispatch

## Non-Goals

- 自托管机器的运维手册（维护者私有供给，仅文档化标签）；Windows/OSX 矩阵；覆盖率阈值；benchmark 跨机型比较
- cann-rs 仓库的 CI（独立仓库自管）；多集群编排；Flaky 自动重试策略（人工处理）

## Constraints

- GitHub Actions 为主；runner 由维护者提供（标签契约）；无 secret 依赖（模型权重/golden 均入库或工件）
- 门禁阈值一致性：CPU 档 5%（000）、GPU 档 10%（006）——本 spec 只接线不覆写
- AI 辅助提交：与宪法一致；CI 工件本身变更须过 spec changelog（本 spec 为锚点）

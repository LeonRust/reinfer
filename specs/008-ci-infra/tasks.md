# Tasks: CI infrastructure

> Derived from specs/008-ci-infra/plan.md

## T1: ci.yml（PR 主档）

- fmt/clippy/test/cargo-deny 四步；`scripts/ci/checked-ignores.sh` 校验挂入；`--seed` 一致性样例（005）纳入 test 集合
- Verification: 无 GPU 本地模拟 `act` 或 CI 首跑全绿；预算 <5min；ignore 校验脚本对故意 ignore 的测试报警

## T2: gpu.yml（标签/路径触发）

- build（--features cuda）→ differential（003 D7 容差）→ parity → throughput（3× CPU 门禁）；串行、45min 超时、并发 1
- Verification: fixture（`REINFER_CUDA_ARCH` + 假 GPU 断言? 不支持）——至少 dry-run 通过 + 触发条件测试（paths/label 匹配）

## T3: bench.yml + baseline.json 机制

- nightly 05:05；`--perf` 5 次中位数 → `bench/summary.json` 工件 + comment；`perf` label PR 复用；`baseline.json` 入库流程（维护者 review 后）说明
- Verification: fixture 模拟回归（构造 0.85× 基准 → 判定红）；注释数字正确

## T4: `#[ignore]` 契约与校验

- 约定文档（plan D3）+ 映射表 == gpu.yml job 名；白名单例外评审留痕
- Verification: 新 ignore 测试无映射 → check 失败；映射正确 → 通过

## T5: 调用方接线（003/004/005/006/002 引用收敛）

- 各 spec 中"job/runner/label"表述改为引用 008（删除自造叙述）；feature-list ASC-05 状态改"✅ specs/008 定义"
- Verification: grep 无残留的自定义 CI 叙述；008 接线表为上（specs 引 008）

## T6: Runner 规范文档 + 缓存恢复

- `docs/dev/runners.md`：标签、单设备、`--test-threads=1`、硬件四元组记录流程；JitCache 预烘焙（REINFER_CUDA_ARCH）入缓存 → GPU job 恢复路径
- Verification: runners.md 与实际 label 一致；缓存恢复 dry-run（恢复后 job 无 nvcc 编译日志）

---

Completion gate：T1–T6 accepted；ci.yml 在仓库 CI 首跑绿；无 GPU 档与 GPU 档分工明确；接线表无悬挂引用。

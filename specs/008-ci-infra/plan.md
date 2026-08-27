# Plan: CI infrastructure

> Derived from specs/008-ci-infra/spec.md

## Architecture Decisions

- **D1 工作流拆分（三个文件，职责单一）**：
  - `ci.yml`（PR/push main）：fmt → clippy → test（无 GPU 全量含 `#[ignore]` 排除集）→ cargo-deny；5 min 预算；
  - `gpu.yml`（manual dispatch + label `gpu-test` + paths `crates/cuda/**`）：cargo build --features cuda → 差分 → parity（000/parity.md）→ 003 3× CPU 门禁 → 006（若命中）perf 记录；串行 runner；
  - `bench.yml`（nightly + manual）：`--perf` 对比 → 「5 次中位数」写入 `bench/baseline.json` 工件 → 报告 comment（openai/opengraph 无倾向，仅数字）；`perf` label 的 PR 触发同一作业（`inputs` 复用）。
- **D2 Runner 规范**：标签 `nvidia-gpu`（附 `gpu-model=sm-XX` 可变 label）；每 runner 单设备（CUDA_VISIBLE_DEVICES 固定）；`--test-threads=1`；硬件四元组写入 `bench/runner-info.json`（UUID/驱动/cuBLAS/sm）→ parity.md 记录档引用；can-gpu 标签预留（002 复活时接线）。
- **D3 `#[ignore]` 契约**：
  - 规则：GPU 依赖测试 → `#[cfg(feature="cuda")]` + `#[ignore]`（无 GPU 不编入二进制）；无 GPU 档的测试（错误映射/JitCache 键/锁/TuneDb/回退链）→ 标配不 ignore；
  - 检查：`scripts/ci/checked-ignores.sh` —— 列出当前全部 `#[ignore]` tests（`cargo test -- --list` 过滤），要求与 `gpu.yml` job 名一一映射（缺映射 → 失败）；
  - 例外白名单（评审留痕）：长跑压测放 `test-labels`。
- **D4 缓存与工件**：`action/cache` 键=cargo lock hash（target 目录）；JitCache `~/.cache/reinfer/jit` 以 `REINFER_CUDA_ARCH` 预烘焙（bench.yml 无 GPU 生成 → gpu.yml 恢复，key=arch+toolkit 版本）；工件：goldens（tests/golden/）+ `bench/summary.json` + `baseline.json`（nightly 后由维护者 review PR 入库）。
- **D5 基准门禁接线（唯一接线表）**：

| 门禁 | spec | job | runner | 判定 | 阈值 |
|---|---|---|---|---|---|
| 差分（每 kernel 容差表 D7） | 003 | gpu.yml `differential` | nvidia-gpu | 测试退出码 | 全过 |
| parity（3 层） | 003+parity.md | gpu.yml `parity` | nvidia-gpu | 脚本断言 | 按 parity.md |
| CPU 档吞吐 | 003 | gpu.yml `throughput` | nvidia-gpu | 脚本断言 | ≥3× llama.cpp CPU |
| 性能回归 | 006 | bench.yml / label `perf` | nvidia-gpu | `--perf` | 中位数 ≤0.9× baseline = 红 |
| 采样/一致性 | 005 | ci.yml `consistency`（cpu 档） | ubuntu-latest | 测试退出码 | 2×bit-identical |
| golden 重建 diff | 004 | ci.yml `goldens` | ubuntu-latest | diff 非空→需人工 | 漂移警示 |
| ASC 三档 | 002 | can-gpu（预留） | can-gpu | — | 002 复活时 |
| L1 运行时 smoke（真机：设备/流/事件/缓冲/拷贝） | specs/009 | gpu.yml `smoke` | nvidia-gpu | `cargo test -p reinfer-cuda --features cuda --test smoke -- --ignored --test-threads=1` | 测试退出码 | 全过 |
| L2 Jit 内核（vec_add/rms/rope/softmax 差分 + 缓存命中 + sampler 组合） | specs/012 | gpu.yml `l2-jit` | nvidia-gpu | `REINFER_CUDA_NVCC=<nvcc≥12.8> cargo test -p reinfer-cuda --features cuda --test jit_smoke -- --ignored --test-threads=1` | 测试退出码 | 全过 |

- **D6 触发与成本**: gpu.yml 仅 manual/label/paths —— main 绿即可合入；超时 45 min；并发 1（runner 单机自持）；nightly 05:00 非整点（与工具冲突）。
- **D7 与宪法衔接**：AI 提交必须跑本契约（ci.yml 的 `#[ignore]` 校验 + fmt/clippy/test）；CI 工件变更走 spec changelog。

## Module Breakdown

| 文件/脚本 | 内容 |
|---|---|
| `.github/workflows/{ci,gpu,bench}.yml` | D1 三工作流 |
| `scripts/ci/checked-ignores.sh` | D3 校验 |
| `scripts/ci/gate_throughput.sh` | D5 CPU 档门禁断言（llama-bench 协议参数） |
| `scripts/ci/gen_goldens.sh` | 004 golden 重建（锚定 commit/flag） |
| `bench/{runner-info.json,baseline.json,summary.json}` | 材料/D2/D5 |

## Risk Assessment

| Risk | Mitigation |
|---|---|
| GPU runner 缺供给 → 门禁长期未跑 | 标注记录档（记录不设闸）+ `gpu-test` label 手动流程 |
| `#[ignore]` 被误用（跳过真测试） | D3 映射校验脚本；diff 审计 |
| bench.yml 噪音（nightly 结果波动） | 5 次中位数；仅注释数字；无自动出图 |
| 缓存毒化（JitCache 旧键） | 键含 arch+toolkit+源码闭包（003 D4）；nightly 重建 |
| 自托管安全（workflow 权限） | `permissions: contents: read`；不可用 secrets |

## Reference assets（增量）

- vLLM/llama.cpp CI 形态（矩阵与缓存策略）；flashinfer `ci/bash.sh`（自托管 GPU 组织范式）

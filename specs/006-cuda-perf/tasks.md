# Tasks: CUDA performance upgrade

> Derived from specs/006-cuda-perf/plan.md

## Task 1: FMHA prefill kernel (Vendor-CUTLASS 档)

- `fmha/` kernel（sm90a warpspec + simple 模板）经 JitCache 编译；`prefill_attn_quad()` 替换 003 两段 GEMM
- Verification: kernel 差分 ≤1e-5（CPU 参考）+ 4K seq 对 003 输出文本一致（≥99.9%）；无 sm90 设备 skip

## Task 2: TuneDb v1

- `tune.json`（device/op/cfg → 时间测量）+ `select_fmha` 选择器；自动 `bench_and_record`
- Verification: 首测（慢）+ 二次（快速读到）+ 设备过滤

## Task 3: FA3 cubin 装载（可选项）

- `fa3/`: 下载 → sha256 校验 → `cuLibraryLoad` → `cudaLaunchKernelEx`（在 sm100a/sm120a 设备）
- Verification: 无 cubin 时 `select() → D2`；有 cubin 时 D3 优先 & 正确性 diff

## Task 4: CUDA Graph pool

- `GraphPool` 捕获（bs 桶 8/16/32/64 × seq 桶 512/1024/2048/4096）；重放/eager 回退；`cudaGraphExecUpdate` 快速断言
- Verification: 各桶捕获/重放与 eager diff ≤1e-5；桶外走 eager 日志可见；内存增量 ≤ 8%

## Task 5: 双流重叠

- attn / ffn 事件对（`compute/comm` reserve）；`overlap::{wrap, sync}`
- Verification: 基准记录（decode/prefill 各自 ratios）; 不稳定时允许 `--no-overlap` 降级开关

## Task 6: 基准与门禁收尾

- `bench/notes.md`：record decode/prefill × llama.cpp-CUDA; `ck --perf` 回归（>10% 掉速阻断）
- CI: GPU job 增加 `perf` 任务（nightly 或 label 触发）；`--features cuda` 无 GPU 时仍全绿

---

Completion gate：Tasks 1–6 完成；decode ≥85% llama.cpp-CUDA、prefill ≥70%；bench 数据 + 回退链文档入库；评审通过。至此 P1/P1.5 结束，下一步 P2（Ascend 完整，依托 cann-rs L1/L2 进度）与 P3 规格。

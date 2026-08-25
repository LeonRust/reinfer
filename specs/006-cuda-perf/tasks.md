# Tasks: CUDA performance upgrade

> Derived from specs/006-cuda-perf/plan.md · 容差用 003 D7 表

## T1: FMHA prefill (Jit 档)

- 摘录头 version.json + `.cu` 编译（sm90a/100a gencode 梯度 + CUTE flag 注入）；heuristics（head_dim/warp/分块，按 flashinfer `fmha_v2` 启发式样例）
- Verification: 差分 ≤D7（fp16 出），4K seq vs 003 dense 文本 100%；无 sm90a 设备 skip；源码含 wgmma 时编译失败→回退 dense（不假装降级）

## T2: TuneDb + select (crates/kernels)

- tune.json（原子写+写锁+损坏容错=重 bench）；`select_fmha` 回退链单测（无 GPU → 恒 dense）
- Verification: 首测慢/二次快；损坏 JSON 可恢复；无 GPU CI 绿

## T3: Vendor cubin（manifest 供应链）

- `cubins/manifest.json`（sha256/arch/来源）入库；加载+校验；离线→硬失败（无静默下载）；失败→回退 Jit(fmha)+warn；vendored release 分发路径
- Verification: 篡改 manifest → load 拒绝并回退；离线行为断言；checksum 来源=仓库内（非下载侧）

## T4: Graph pool（decode-only）

- 桶 8..128/8、128..256/16；单池；profile 记账；捕获全局锁；捕获期 `--no-overlap`；ExecUpdate 限 ptr 刷新（失败 re-instantiate）；运行期计数（replay/eager/padding per bucket）
- Verification: 各桶回放==eager 100% token + ≤1 ulp；execupdate 失败路径重实例化；内存增量实测记录

## T5: 双流重叠

- 模式①（事件入图）与 ②（--no-overlap）；捕获窗口外部多流允许
- Verification: 两模式结果一致；捕获期串行断言（无并发发射）；`--no-overlap` 开关生效

## T6: decode fused Q8_0 dequant-dot 核（sm90 门禁前提）

- 寄存器内解量化+dot（mmvq 级结构，算法不抄代码）；替代"dequant→GEMM"路径（dense fallback 保留）
- Verification: 差分 ≤D7；decode 相对 003 提升记录；**sm90 门禁在 T6 落地后才按 0.85× 判决**

## T7: 基准协议 + 回归门禁

- `bench/` 协议脚本（llama-bench 参数/commit/构建 flags/UUID 记录）；`baseline.json`（5 次中位数）；`--perf` 对比输出；CI：中位数 ≤0.9× 基线 → 红（10% 阈值，与 000 的 5% CPU 档并存注明）
- Verification: 重放基准两遍一致；CI 模拟阈值判定（fixture）通过

---

Completion gate：T1–T7 accepted；sm100 decode ≥0.85× / sm90（T6 后）≥0.85×，未达 T6 前回退档（≥max(0.7× CUDA, 6× CPU)）记录；prefill ≥0.7×；notes+baseline 入库。下一步：007-core-inference（CPU 全链路，为 005 `--backend cpu` 与无 GPU CI 提供载体）与 008-ci-infra。

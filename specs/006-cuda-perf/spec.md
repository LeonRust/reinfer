# Spec: CUDA performance upgrade — vendor tier + CUDA graphs

> Status: approved (review 2026-08-25) · Parent: specs/003 · 修订记录：门禁 arch 分档；仅 decode 捕获；单池实测记账；供应链 manifest；spec 层去厂商化；'007 RFC' 改为"后续增量 spec"。

## Problem Statement

003 提供了正确性路径（Jit 档 dense kernel）；本切片把性能拉到对标杆：① prefill 用 vendor/Jit FMHA 替换两段 GEMM；② decode 高频路径引入 fused 量化核（sm90 门禁前提）；③ CUDA Graph 桶化（decode-only）。回退链始终保留：性能提升不得以失去 003 正确性路径为代价。

## Success Metrics

- **正确性（硬门）**：同形状图重放与 eager **100% 逐 token 一致 + kernel 差分 ≤1 ulp**；回退率另记（回退不等于错误）；实现与 eager 不合时强制回退（D4/D6）。
- **性能门禁（arch 分档，协议锁定见基准协议文档）**：
  - sm100：decode ≥ **0.85×** llama.cpp CUDA（cuBLAS tcgen05 路径先验）；
  - sm90：decode ≥ 0.85× 当且仅当 **decode fused Q8_0 dequant-dot 核（T6）** 落地；未落地前回退档 = `max(0.7× llama.cpp CUDA, 6× llama.cpp CPU)`；
  - prefill ≥ **0.7×** llama.cpp CUDA（CUTLASS FMHA 路径）；
  - 对比条件固定：KV dtype=f16、graph on（双侧）、同模型 sha、同 batch、预热≥3 取中位数、commit f280b2698 + 构建标志锁定（基准协议）。
- **内存**：graph 捕获内存**实测记账**（profile 法）并计入预算公式；桶池化缓冲增幅记录（预期 ≤8%，以实测为准）。
- **运行期信号**：graph_replay / eager_fallback / padding_ratio 计数器；5 分钟内 eager 比例 >20% → 告警指标（不作为硬门禁，记录于监控）。
- **工程**：无 GPU CI 仍绿；006 门禁仅 gpu-runner 判定；无 GPU 档仅测 JitCache 键/锁、TuneDb 读写、选择器回退链（恒定回退 003）。

## User Stories

1. 作为引擎作者：`select()` 对同一 OpConfig 自动选 Vendor(cubin) > Jit(fmha) > Jit(dense)，并可从 `TuneDb` 读取实测量；回退传递对引擎透明。
2. 作为服务者：`--perf` 输出与 llama.cpp 同协议的对比；`bench/notes.md` + `bench/baseline.json` 机器可读。
3. 作为维护者：离线（无 GPU/无网）构建与单元测试不受影响；供应商资产经 manifest 变更受审。

## Acceptance Criteria

- [ ] FMHA prefill 落地（引擎自有源码 + 运行时编译；sm90a gencode；heuristics 按实测），与 003 dense 路径差分 ≤ 容差表；无相应 arch 时回退 003
- [ ] decode fused Q8_0 dequant-dot 核（T6）独立交付并过差分；sm90 门禁在此核存在后才生效
- [ ] vendor cubin 档：期望 sha256 从**入库 manifest**校验；离线 → 硬失败（不静默下载）；优先随 release vendored 分发；校验失败回退 Jit(fmha) 并 warn
- [ ] Graph：decode-only 桶（8..128 步进 8、128..256 步进 16）；单一共享内存池；捕获全局串行化；捕获期强制 `--no-overlap`；ExecUpdate 仅限同形状指针刷新（失败 re-instantiate）；prefill 不捕获（实验性小桶须显式开关）
- [ ] 双流重叠两模式：图内事件节点（llama.cpp 式）或 `--no-overlap`（vLLM 式）；捕获期唯一
- [ ] 基准协议 + `baseline.json`（5 次中位数）+ 回归门禁（CI 红 = 中位数 ≤ 0.9× 基线）；10% 为 GPU 档阈值（000 的 5% 为 CPU 档，两档并存）

## Non-Goals

- Flash MoE/MLA（P4）；warp specialization 手工调（后续增量 spec 006-2）；IR 编译/跨卡调度；非 decode 的图捕获（除实验性开关外）

## Constraints

- 引擎自有源码 + 供应商预编译资产两条线并存；**任何 vendor 覆盖必须有显式回退链**（Vendor → Jit(fmha) → Jit(dense)/003）
- 裁判 = llama.cpp CUDA（commit/参数锁定）；所有比值固定 KV dtype=f16、graph on
- 供应链：manifest 提交入库、变更走 PR 审查；禁止"脚本下载+同源自校验"

# Tasks: CUDA L3 — single-request full loop

> Derived from specs/014-cuda-l3-single-request/plan.md · 依赖：见文末依赖表；提交拆分：每个 T 一块小提交（`feat(gguf/arch/tokenizer/cuda/memory/cli): ...`）
> 模型获取走 specs/013 resolver；模型标识零硬编码（013 铁律）。

## T1: GGUF 读取器（header/meta/tensor + mmap）

- `crates/gguf`：魔数/版本/元数据(含嵌套数组)/tensor 表；mmap 视图；`GgufTensor`（对齐/偏移/形状）；Qwen2 元数据探针
- Verification: golden-file（fixture gguf：自生成小文件）+ proptest（随机元数据）；tensor 读取与 llama.cpp 转储对拍（元数据层）

## T2: 量化 codec（Q8_0 / F16 / FP32）+ proptest

- Q8_0 block 256 算法（llama.cpp 语义）；F16→fp32 转换；codec 纯函数 + proptest（≤1 ULP 金块：llama.cpp `llama-quant` 产出）
- Verification: golden blocks（固定 seed 的量化块）≤1 ulp；roundtrip 性质（dequant ∘ quant 有界误差）

## T3: arch typed config（Llama/Qwen2）

- `crates/arch`：head_size/n_layer/kv heads/rope θ/vocab 等映射 → `LlamaConfig`；缺字段 → 明确错误
- Verification: 从 T1 fixture 元数据解析出与 llama.cpp dump 一致的值；缺字段错误单测

## T4: tokenizer（004：SPM / BPE）

- GGUF `tokenizer.model` 解析（SPM/BPE 两种容器）；encode + 增量 decode-step；特殊 token 边界
- Verification: 004 golden（20 prompts token 100% 对齐 llama.cpp）；中文/emoji/换行边界用例

## T5: dequant 内核（003 T8）+ 真机差分

- `crates/cuda`：Q8_0 并行核（block 256 语义与 T2 CPU 参考一致）+ F16 直读视图；走 JitCache 管线（012）
- Verification: 真机 diff（随机块：≤1 ulp）+ 确定性

## T6: cuBLAS GEMM wrapper（003 T9；vendor tier 首件）

- workspace/cudarc 增 `cublas` feature；`Gemm::f16_16acc`（对齐 llama.cpp 惯用参数）；ProviderTier::Vendor 语义接入
- Verification: vs CPU matmul（f16 rel 1e-4 / f32 rel 1e-5；100 随机形状）+ perf sanity 记录（cpu 参考时间/GEMM 时间写入 notes）

## T7: Prefill attention（003 T10：两段 GEMM，NHD）

- `prefill_attention`：QK^T→masked softmax(012 参考)→PV；fp16 输入；CPU 参考（naive）
- Verification: seq 1k fp16 diff（判定 ≤1 ulp vs 参考声明记录；容差 D7）；注意力一致性（全行 sum≈1、掩码行 0）

## T8: Paged decode attention GQA + MemOps（003 T11）

- `crates/memory`：页块池（16/32 参数化，refcount+free list）+ 泄漏计数；`crates/cuda`：decode_step_gqa（smem 累加、paged 索引、GQA 组映射）+ MemOps alloc/free
- Verification: 随机页表 batch 1..64 diff（含未初始化页 != 0 的防护）；泄漏运行：1M 页循环后"在用页==0 且空闲表==预热长度"

## T9: 最小 Runner + cli 流式闭环（003 T12 装配）

- bin/cli：`reinfer cli --backend cuda --model <由 013 获取的 gguf> "<prompt>"`：加载（T1/T3）→ 预填充（T7）→ decode 循环（T5/T6/T8 + 012 sampler 管线，temp=0/固定 seed）→ 流式 stdout
- Verification: 0.5B F16/Q8_0 各跑 200 token 稳出（无 NaN、固定 seed 复现）；Q8_0 结果与 llama.cpp 同参对拍（parity harness 接入）

## T10: parity + 性能判据

- parity.md 三层执行（F16 三层 / Q8_0 ≥99.9%；20 prompts）；decode ≥3× llama.cpp CPU（008 D5 协议：同机同参、warmup、中位数）
- Verification: `gate_throughput.sh` 扩展 + notes 记录（模型 sha256/参数/硬件四元组/结果）

## T11: 文档与状态

- feature-list（L3 状态/锚 014）、phase-plan L3 勾选、notes-l3（复盘/命令/一次性大坑）、README 模型段补 `reinfer model get` 关联
- Verification: 文档一致性（勾选+锚点）；无"模型名写死"残留（grep 检查：代码路径无模型常量）

---

Completion gate: T1–T11 accepted；数据管道全绿 + 真机 diff（D5-D7 判据）+ cli 流式 + parity 三层 + 3× CPU 记录；评审通过。下一片：005（serving，L3 解码路径为真实载体）与 006（perf/图编排）。

依赖表：T2←T1；T3←T1；T4←T1；T5←T2；T7←T3,T6；T8←T6；T9←T3,T5,T6,T7,T8；T10←T9；T11←T10。T1-T3 与 T4（tokenizer 可独立先于 T1?——tokenizer 模型文件也由 gguf 容器内嵌——T4←T1）。T1 先行，T2/T3 依赖 T1，可并行推进（无文件冲突：gguf/codes vs arch）。

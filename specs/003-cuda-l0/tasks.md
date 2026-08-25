# Tasks: CUDA L0 — GPU inference loop

> Derived from specs/003-cuda-l0/plan.md · 每条独立可验证；容差一律用 plan.md §D7 数值容差表

## T1: CUDA wiring (workspace)

- `cudarc` optional workspace dep；feature `cuda` 门控（`crates/cuda` + `bin/reinfer` 转发）；`.gitignore` 加 jit 缓存目录
- Verification: 无 toolkit `cargo check --workspace` 绿；GPU 机 `cargo check --features cuda` 绿

## T2: Device/Stream/Event/Buffer wrappers

- `CudaContext`（设备数/set per thread）、`CudaStream`、`CudaEvent`、`CudaBuffer`(Send)、`HostBuffer`(pinned)、memcpy
- Verification: 无 GPU 单测（构造/Debug/错误映射）；GPU smoke（alloc/free/copy roundtrip）

## T3: Error mapping (whitelist)

- `map_error` 白名单表（Oom/Driver/Fatal；未知→Fatal）；对齐 002/plan 表
- Verification: `cargo test -p reinfer-cuda` 每类 code 单测 + unknown→Fatal

## T4: JitCache v1 (加固版)

- 键 = sha256(源码+`nvcc -M` 头闭包+gencode/flags+nvcc version+capability)；temp+rename 原子写；load 失败删+重建一次；启动阻塞式 prewarm；按 key 文件锁+双检；锁文件 /tmp；`REINFER_CUDA_ARCH` 离线预烘焙；toolkit 梯度检查（sm90a≥12.3/sm100a≥12.8/sm120a≥13.0）；nvcc 缺失→`LaunchError::Fatal` 专用消息（不静默 CPU）
- Verification: 编译-再命中（<50ms）；改头文件→失效重建；nvcc 缺失→专用错误；无 GPU 用 REINFER_CUDA_ARCH 预烘焙成功

## T5-D7: diff 内核：norm/rope/softmax + sampler

- RMSNorm、RoPE(f32 累积)、masked softmax(online-max)、sampler(masked softmax + SplitMix64 纯函数 + argmax)
- Verification: host 差分按 D7 容差（fp32 输出 rtol/atol；随机形状 head_dim 64/128）+ 确定性单测（同 seed 同输入→bit 一致）；GPU runner 执行

## T8: dequant kernels (Q8_0 / F16)

- Q8_0 块解量化（block 256，llama.cpp 算法）、F16→fp16
- Verification: 与 CPU 参考差分（出 fp32→D7 fp32 判据；金块来自 001 golden）≤1 ulp

## T9: cuBLAS GEMM wrapper

- `gemm_f16/f32`（f16-acc/f32-acc 可选，F16 默认 16F-acc 与 llama.cpp 同）
- Verification: 与 CPU matmul 差分（f16：rel 1e-4；f32：rel 1e-5，100 形状）; perf sanity vs cuBLAS 理论峰值（记录项，无 torch）

## T10: Prefill attention（两段 GEMM，本切片路径）

- QK^T+softmax+PV；NHD 布局；与 referee flash-attn 的差异声明入 notes
- Verification: seq 1k fp16 差分（输出舍入后 ≤1 ulp）

## T11: Paged decode attention (GQA) + MemOps

- 块 16/32、GQA 组映射、smem staging；`crates/memory` 块分配器 + CUDA `MemOps`；refcount + 泄漏计数
- Verification: 差分（随机页表，batch 1..64）；泄漏运行：1M 页 alloc/free 后"在用页==0 且空闲表==预热长度"（005 池基线定义）

## T12: Sampler 集成 + engine 单请求闭环

- ModelRunner 路径：001 GGUF → 004 tokenizer（依赖）→ 003 kernels → sampler → 流式 decode；`crates/engine` 最小宿主或经 `arch` 组装（归属见 design review A-M4：engine crate 于 005 切片正式建）
- Verification: `reinfer cli --backend cuda --model Llama-8B-Q8_0.gguf "prompt"` 流式；parity.md 矩阵跑通（F16 三层门禁/Q8_0 ≥99.9%）；decode≥3× llama.cpp CPU（基准协议，gpu-runner 判定）+ notes 记录

## T13: CI 门禁

- 无 GPU 档：check/test（含 TuneDb/JitCache 键与锁、错误映射单测）+ `#[ignore]` 标注 GPU 测试
- GPU 档：差分 + parity + 3× CPU，矩阵入 008-ci-infra 定义
- Verification: 两档在 CI 绿并文档化

---

Completion gate: T1–T13 accepted；parity（三层门禁）+ 3× CPU 记录于 notes；评审通过。下一片：specs/005（serving）与 specs/006（vendor+graph，含 decode quant-dot 任务）。

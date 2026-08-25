# 用 Rust 从零构建支持 CUDA + 昇腾 CANN 的推理引擎——设计报告

> 生成日期：2026-08-25
> 背景：基于 ai-tokens 下 7 个开源项目（vllm / sglang / llama.cpp / lightllm / flashinfer / flash-linear-attention / mini-sglang）的深度分析，结合 2026-08 的 Rust 生态现状检索，给出 Rust 语言新推理引擎的完整设计。

---

## 0. 先定调：可行性判断

**Rust 值得做引擎，但"全 Rust"要严谨区分两层：**

| 层 | Rust 的胜算 | 原因 |
|---|---|---|
| 上层（调度/缓存/内存/通信/服务） | ✅ **极强** | 无 GIL、无 GC，`Arc`/借用可静态证明 KV 页生命周期（C++ 引擎的 UAF/悬挂指针恰恰是 vLLM/SGLang csrc 里最常见的崩溃类）；单二进制分发（llama.cpp 哲学）；axum/llguidance 原生生态 |
| 下层（kernel） | ⚠️ **分档** | 峰值性能内核（FA3、CUTLASS MLA、FP8 GEMM）在 NVIDIA 侧仍是 C++/PTX 天下；昇腾侧 CANN 只有 C/C++ API，AscendC 是 C++ 编写、须经插件编译。全 Rust 写这些内核在 2026 会输掉性能战 |

**核心决策 D1——分层语言策略**：引擎/调度/内存 100% Rust；kernel 分三档，用 trait 统一：

```
档1 Vendor 预编译（性能天花板）: FlashInfer cubin / FlashAttn-3 / cuDNN / CANN aclnn+aclblas —— FFI 调用
档2 Rust 原生内核（性能 90% 档）: cudarc + 手写 PTX/rust-cuda 或 CubeCL 写 norm/sampler/量化/gather/KV 搬运 —— 简单、易测、可自控
档3 外部 DSL（研究实验）: Triton/TileLang/AscendC 通过 JIT 桥接 —— 留给线性注意力/稀疏注意力等新算法
```

---

## 1. 总体架构

```
┌────────────────────────────────────────────────────────────┐
│ 发布面: 单一二进制定reinfer (server/cli/bench) + pyo3 绑定 crate   │
├────────────────────────────────────────────────────────────┤
│ 协议层: axum (OpenAI/Anthropic/Cohere/MCP 兼容) · gRPC · 流式 │
├────────────────────────────────────────────────────────────┤
│ 服务层: llguidance 结构化生成(纯Rust原生嵌入!) · 路由/鉴权 · 监控 │
├────────────────────────────────────────────────────────────┤
│ 引擎核心 (tokio + 单线程调度 loop):                          │
│   Scheduler(连续批处理+chunked prefill+精确预算准入)          │
│   RadixCache(前缀树复用) · URL/请求状态机 · 采样器链 · 投机解码 │
├────────────────────────────────────────────────────────────┤
│ IR/执行层: Typed IR(强类型算子) → 桶化图(graph per shape 桶)   │
│   → CUDA Graph 捕获 / CANN Graph 捕获 / CPU 直执行            │
├────────────────────────────────────────────────────────────┤
│ 硬件抽象: Backend trait: Cuda · Can · Cpu ...               │
│   backend trait 管: 设备init/上下文、流与事件、内存、通信句柄    │
├────────────────────────────────────────────────────────────┤
│ 内存层: VMM 池 · 页表式KV分配器 · mmap 权重 · CPU offload 栈   │
├────────────────────────────────────────────────────────────┤
│ 通信层: NCCL/HCCL FFI · 自研 ring allreduce · NVLink/RoCE     │
└────────────────────────────────────────────────────────────┘
```

**核心决策 D2——进程模型**：**单进程、多线程（每 GPU 一个 worker 线程 + 每线程一个 CUDA/CANN 上下文）**。这与 vLLM/SGLang 的多进程模型不同——Rust 没有 GIL，零拷贝消息（crossbeam SPSC + 共享内存环形缓冲）天然可用，跨 rank 共享 KV 页表通过 Rust 所有权静态保证无竞态；保留可选"进程隔离模式"（GPU 故障隔离，参考 llama.cpp 的崩溃重启思路）。

---

## 2. 引擎核心设计（集结 7 项目精华）

**2.1 调度器**（对标 SGLang 5291 行调度器，拆成小模块）
- 单线程事件循环 + tokio 异步前端；决策纯 CPU，只传递 `ReqId`/整数索引/`&Batch` 引用——"瓶颈是 GPU 不是调度器"被反复验证；
- Continuous batching：token 预算准入（lightllm 的"预计峰值 token 数"）+ chunked prefill（1024/动态切块，让长 prefill 插入 decode 流）；
- **确定性铁律**：decode batch 一律按 `req_id` 排序（mini-sglang #113 一行修复的教训——跨 TP rank 槽位错位会导致结果归属错误）；
- Cache-aware 调度：LPM/DFS-weight（按前缀长度收益优先）与 FCFS/LOF 自适应切换；
- 抢占：swap 级（页式 KV 天然支持）与暂停-恢复两级；
- 请求生命周期用显式状态机（mini-sglang `Req` 双游标 cached_len/device_len 原样继承，用 Rust enum 建模过不了编译器的非法状态）。

**2.2 前缀缓存**：RadixAttention 移植（SGLang 核心创新）。Rust 版本用 `radix` crate + 链表 arena，key 按 token hash 分页（page_size=1 连续布局 / >1 分页布局统一——mini-sglang 的"一套机制"设计直接可抄）；ref_count 保护 + 堆式 LRU 驱逐。

**2.3 采样器链**：top-p/top-k/min-p/温度/贪心/beam 静态组合为 `Sampler` 链（不可变配置 + 每批随机种子状态）；投影、噪声、multinomial 全部 Rust 原生 kernel（档2）——这块在 C++ 引擎里以 C++ 实现，Rust 反而最顺手；multi-sample/beam 支持并行批次。

**2.4 投机解码**：草案模型 + EAGLE/DRAFT + MTP（对齐 lightllm 的 EAGLE3/DFlash、SGLang 的 DSpark 投机）；rejection sampler 在 Rust 侧实现；支持 `step=5` 级联。

**2.5 结构化生成（本方案最强的差异化点）**：SGLang/vLLM 都要 FFI 包装 C++ 的 xgrammar；而 **llguidance 本身就是 Rust 库**（Guidance.ai 出品，regex FSM + JSON schema + CFG），全程零 FFI，语法约束直接挂在调度器 grammar 状态上、逐 token 填 vocab mask + jump-forward——这正好落在 Rust 生态的甜点区。

**2.6 图执行**：per-(batch_shape, seqlen) 桶 → CUDA Graph 捕获池（vLLM `cuda_graph_buffer_registry` 思路）；CANN 侧对应 `aclrtGraph`；小 batch 场景维护"graph + eager 双路"，按桶命中率切换。

---

## 3. 算子层设计

**3.1 KernelProvider trait（三档）**，每个算子给出 provider 优先级链，运行时自动选择＋autotune 缓存 `kernels/{device}/{dtype}/{shape}/bench.json`：

```rust
trait KernelProvider<E> {      // E: CudaKernel / AscendKernel / CpuKernel
    fn priority(&self) -> i32;
    fn max_batch_factor(&self) -> f32;              // 某档在 batch 上的衰减
    fn matches(&self, cfg: &OpConfig) -> bool;
    fn launch(&mut self, ...) -> Result<()>;
}
```

**3.2 NVIDIA 侧**：
- 注意力/GEMM/FFN 头部：**FFI 复用 FlashInfer**（其 `flashinfer-cubin` 按 GPU 下载预编译 cubin 的协议可以直接沿用以避免本地编译开销）＋ FlashAttn-3（sm100a/sm120a 路径走 cutlass）+ cuBLAS；
- norm/RoPE/采样/量化/位宽转换/gather/KV 搬运：**cudarc 写原生 kernel**（或用 CubeCL 增量积累）——这些算子结构简单、数值易验证，是 Rust 原生 kernel 的自留地；
- FP8/NVFP4/MXFP4：走 CUTLASS 底座；W4A16 Marlin 已有 Rust 先例（ferrum-runtime）可参考。

**3.3 昇腾侧（无先例区，设计要点）**：
- 上层全部走 CANN C-API（aclrt/aclrtStream/aclrtSetDevice/aclGraph）；
- 标准算子（Matmul/Softmax/TopK/LayerNorm 等）用 **aclnn/aclblas 官方算子**（ATC 编译后的 aclnn 调用），大 GEMM 用 `aclblasLiteMatmul`/FP16 accumulate；
- 关键自研算子（与 flash decode 同款：KV 装载、分页注意力、MoE 路由）用 **AscendC（C++，bisheng 编译）写成外部 kernel 模块，经 C ABI 桥接**——CANN 生态现阶段没有 Rust 编写路径，这是唯一的现实选择，设计上把"AscendC 源"作为供应商可替换资产（同 CUTLASS cubin 的地位）；
- 多维并行：TP（TensorParallel）+ PP 在 Rust 侧编排，HCCL 通信。

**3.4 量化与数值**：GGUF 兼容量化表（Q4_0/Q8_0/K-quants/IQ 家族——MIT 许可，重排版与 SIMD 用 `std::arch`/`wide` + LLVM auto-vectorize 移植）；KV cache 量化（Q8_0/Q4_0、FP8）；数值纪律：kernel 档2 必须与档1/naive 参考在 `assert_close` 阈值内 fwd+bwd 一致（FLA 的验证文化直接纳入 CI 门禁）。

**3.5 模型定义**：不走 vLLM 的"285 个文件"，走 llama.cpp 路线——**架构枚举（arch 参数化描述）+ 权重映射驱动加载器**：GGUF/safetensors 元数据 + 权重表 → 通用构建器拼装计算图；特例用 trait 扩展（DeepSeek MLA/MoE、Qwen 稀疏 MoE、GQA、多模态 only 后期）。首期支持：Llama 系 / Qwen2-3 系 / Mistral / DeepSeek V3+MLA / Qwen3-MoE。

---

## 4. 内存管理（Rust 相对 C++ 引擎最大的增量价值）

| 组件 | 设计 | 对应借鉴 |
|---|---|---|
| 缓冲区抽象 | `Arc<DeviceBuffer>`（设备内存句柄）+ `&'batch BatchBuffer` 借用；workspace 每步 arena；epoch 回收池 | vLLM 引用计数 block |
| VMM | CUDA VMM（cuMemAddressReserve/vmm 池化 2MiB 粒度）；CANN `aclrtMalloc`+池复用 | lightllm/FlashInfer 的 VMM 实践 |
| KV 页表 | free-list + radix 树；页级 ref_count；`PageQueue` 双状态（待写/可读） | PagedAttention 思想 |
| 权重加载 | `memmap2` 零拷贝映射 GGUF/safetensors（4KB 对齐只映射元数据+懒入页）；大模型层级 offload（`n_gpu_layers` 式） | llama.cpp |
| CPU offload | pinned 内存池（cudaHostAlloc/aclrtMallocHost）+ 交换流水线预取 | vLLM simple_kv_offload / SGLang hybrid cache |
| OOM 防护 | mem_fraction 预留水位 + swap + 提前告警；逐层释放权重 | lightllm max_total_token_num 自动估算 |
| **安全增量** | 所有权静态保证"批次存活期内 KV 页必存活"——把 C++ 里最常见的悬挂指针 bug 变成编译错误；批次间缓存用 epoch 标记避免 ABA | —— |

零拷贝链路：前端→调度经 SPSC 环形缓冲（跨线程）；跨进程（PD 分离）用共享内存 + 页表引用传递 KV（对齐 NIXL 的分页传输，Rust 侧用 serde+bytes 只传元数据）。

---

## 5. GPU 性能优化清单（分四层）

**微内核层**：TMA/async copy 路径、warp specialization（sm90）、PDL（programmatic dependent launch）；Rust 原生 kernel 用 cudarc 暴露其下 driver 能力（`cuLaunchKernelEx`）；Blackwell tcgen05 走 vendor cubin。

**算子层**：按 head_dim/dtype 选择 kernel 变体（heuristic + autotune）；tensor core epilogue 融合（norm+rope+attn、MoE gate+gather）；权重重排在加载期完成（repack，对齐 llama.cpp repack.cpp）；分块 GEMM 布局（GQA 共享 KV 路径）。

**流水线层**：双 stream 计算/通信重叠（TBO two-batch overlap 思路）；**prefill/decode 资源节奏分离**——对齐 lightllm DSPark：同机两 worker 分别固定 prefill/decode 分配，KV 页迁移关着走；纯 Python 引擎的 GIL/编排开销在 Rust 里直接消失（vLLM 为此写 C++ batch_invariant——我们的调度器本身就是 Rust）。

**系统层**：每设备线程 pin 核 + 每流独立事件查询；CUDA Graph 按桶重放（+18%~85% 的收益已被 ferrum-runtime 验证）；通信 NCCL/HCCL FFI + 节点内自研 allreduce（vLLM quickreduce 的拓扑感知思想）；PDL 替代尾部同步；CPU 打包全路径优化（tokenizer 并行、detoken 增量 UTF-8 处理——mini-sglang 的双 offset 细节必抄）。

---

## 6. 正确性、工程化与发布

- **数值等价**：每算子 naive 参考 + 全后端差分；make_naive 与 vendor 交叉验证；NaN 毒化测试（FLA 文化）；
- **确定性**：固定种子 + decode 排序 + 跨 rank 一致；端到端与 vLLM/SGLang/llama.cpp 逐 token golden 对比（差分测试基线）；
- **错误恢复**：CUDA error（UVA/同步失败）→ 隔离上下文重建与请求重试；Rust panic 不跨 FFI（catch_unwind 临界区），内核热路径 panic=abort；
- **CI/基准**：双硬件矩阵（NVIDIA H20/B200/SM100 + 昇腾 Atlas 800/900 A2）X 每模型每量化；cargo bench + nightly 回归门禁；kernel 性能衰减即失败；
- **可观测**：tracing/OTel、per-token 指标、nsight/aclprof 采集接口、collect_env 诊断（flashinfer CLAUDE.md 的做法）；
- **分发**：`cargo build --release` 单二进制，GPU 后端按 feature 编译；同时发布 pyo3 绑定让 Python 侧（LangChain/数据管道）零成本接入。

---

## 7. 路线图（供排期参考）

| 阶段 | 里程碑 | 验证门禁 |
|---|---|---|
| P0 | IR + CPU 后端 + GGUF 加载 + Llama-7B 单机跑通 | 与 llama.cpp CPU 逐 token 对齐 |
| P1 | CUDA 档1/档2 kernel + 页式 KV + 连续批处理 + OpenAI 服务 | Go-NoGO：单 H100/B200 上 decode 吞吐 ≥ SGLang 同期 85% |
| P2 | CANN 后端（aclnn + AscendC 自研核）+ 双后端 CI 门禁 | Atlas 900 上吞吐/时延对齐官方 CANN 基线 |
| P3 | RadixCache + 投机解码 + llguidance 生成 + TP/PP/CP | 各功能 95% 单测覆盖 |
| P4 | MoE/MLA/FP8 + autotune 体系 + KV offload + PD 分离 + 插件系统 | 对标 vLLM 完整服务能力矩阵 |

---

## 8. 风险与对策

| 风险 | 等级 | 对策 |
|---|---|---|
| Rust 原生 kernel 性能不敌 vendor | 高 | D1 分档策略：天花板永远交给 cuBLAS/FlashInfer/CUTLASS；Rust kernel 只攻简单算子 |
| 昇腾侧零先例、CANN 文档/工具链面向 C++ | 高 | 所有 CANN 调用收敛到单一 can_backend crate；AscendC 外部模块 C ABI 隔离；没有 C++ 就不会被工具链绑架 |
| 生态人才与维护成本 | 中 | 发布 crates（gllm-kernels 路线）回馈社区；pyo3 绑定降低接入门槛 |
| 双硬件维护节奏差异（CUDA 季更 vs CANN 版本漂移） | 中 | Backend trait 版本探测 + feature 矩阵 CI |
| 错过 Python 生态（HF 权重、微调工具链） | 低 | GGUF/safetensors 直读 + pyo3 + 转换工具三路齐开 |
| 与 mistral.rs/ferrum 等已存在项的关系 | — | 差异化在"生产级 serving（radix/PD/grammar/EP）＋双端厂商优化"；mistral.rs 更适合个人实验（可移植其 FA3/MoE kernel 成果作档2 参考） |

**一句话概括本设计**：用 Rust 重写 7 个项目里"最好抄的部分"（SGLang 的调度与 radix、mini-sglang 的模块与状态机、llama.cpp 的格式与量化、FlashInfer 的 vendor 复用协议、lightllm 的准入与 PD），把"最不好写但仍值得写的部分"——内存安全、确定性、单二进制的调度核心——完全交给 Rust；kernel 层则坦诚地站在巨人肩上（vendor cubin + AscendC + 少量 Rust 原生核），用三档 provider 设计保持"什么时候都可以自己补全"的能力。

---

## 附录：参考生态事实来源（2026-08 检索）

- [mistral.rs v0.9.1（Candle 系 Rust 引擎，FA3/PagedAttention/NCCL/MoE 定制核）](https://ossaihub.com/tool/mistral-rs/)
- [mistral.rs v0.8.2 CUDA 性能（prefill 超 llama.cpp GGUF Q8_0）](https://www.creativeainews.com/blog/mistral-rs-v082-cuda-local-llm-creators-2026/#main)
- [gllm-kernels（burn 生态原生 CUDA attention kernel：FA3 async/MLA/Mamba/Paged）](https://lib.rs/crates/gllm-kernels)
- [CubeCL 实施指南（纯 Rust 编写 FlashAttention kernel）](https://github.com/tzervas/unsloth-rs/blob/main/CUBECL_IMPLEMENTATION_GUIDE.md)
- [T0-GPU：纯 Rust GPU 编译器，AMD 上 79 TFLOPS 追平 Triton（证明 Rust 免 LLVM 写 kernel 可行）](https://zhuanlan.zhihu.com/p/2021897491537208329)
- [atomr-accel（CUDA 加速器：FlashAttn 绑定/CUDA Graph/P2P/NCCL）](https://lib.rs/crates/atomr-accel-flashattn)
- [trustformers-core（多后端 FlashAttention-2）](https://docs.rs/crate/trustformers-core/latest)
- [ferrum-runtime（Candle 系：CUDA Graph +18%、INT4 Marlin +85%）](https://lib.rs/crates/ferrum-runtime)
- [C/C++ 系 + Rust 系 GGUF/LLM 推理引擎综述](https://task.bioinfo.online/articleList/20264629806.html)

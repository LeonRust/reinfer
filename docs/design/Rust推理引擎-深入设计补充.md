# Rust 推理引擎——深入设计补充

> 生成日期：2026-08-25
> 配套文档：`Rust推理引擎设计报告.md`
> 内容：① KernelProvider 三档接口 Rust 代码骨架 ② RadixCache 数据结构设计 ③ 7 项目可复用资产清单

---

## 补充一：KernelProvider 三档接口——Rust 代码骨架

### 1.1 设计目标

- 与 vLLM 的 attention backend registry、FLA 的 `@dispatch`/BackendRegistry 同构，但用 Rust trait 把它写出来；
- 三档语义：**Vendor**（预编译 cubin / CANN aclnn 官方算子）> **Native**（Rust 原生 kernel）> **Jit**（外部 DSL：Triton/TileLang/AscendC 桥接）；
- autotune 结果落盘 `kernels/{device}/{op}/{cfg-key}/tune.json`，冷启动无调参也能跑（预置 heuristic）；
- **窄 FFI 原则**：所有 `unsafe` 收敛在三个 Provider 各自的 `launch` 内部，调度/服务层零 unsafe。

### 1.2 核心类型

```rust
// ---- 类型化的算子配置：编译器能检查，运行时无字符串魔法 ----
pub enum OpKind { Attention, Gemm, Ffn, Norm, Rotary, Sampler, Quant, Gather }
pub struct OpConfig {
    pub op: OpKind,
    pub device: DeviceId,                 // Cuda(0..n) | Can(0..n)
    pub in_dt: DType, pub out_dt: DType,  // F16/BF16/FP8/NVFP4/I4...
    pub head_dim: usize, pub batch: usize,
    pub seq: SeqShape,                    // Bucket(usize) | VarLen   <- 图捕获桶化
    pub layout: Layout,                   // NHD | HND | Paged | Contiguous
    pub flags: Flags,                     // IS_VARLEN | RETURNS_LOGITS | ...
}

// ---- 三档 Provider trait ----
pub trait KernelProvider {
    fn tier(&self) -> ProviderTier;
    fn matches(&self, cfg: &OpConfig) -> bool;
    /// 启发式优先级（无调参时也给出确定选择）
    fn base_priority(&self, cfg: &OpConfig) -> i32;
    fn workspace_size(&self, cfg: &OpConfig) -> usize;
    /// # Safety: 调用方保证 cfg/workspace/ctx 三者一致
    unsafe fn launch(&self, cfg: &OpConfig, ctx: &ExecCtx) -> Result<(), LaunchError>;
}

/// ExecCtx：设备 + 流 + 图栈 + 本步 arena。所有 buffer 以 `&'batch` 借用传入。
pub struct ExecCtx<'ex> {
    pub dev: &'ex dyn DeviceBackend,       // Cuda | Can | Cpu
    pub stream: StreamHandle,
    pub graph: Option<GraphCapture>,       // 桶化捕获中
    pub ws: &'ex mut WorkspaceArena,
    pub batch: &'ex BatchSlice,            // &'batch 借用，所有权保证 KV 页存活
}

/// LaunchError 可分类：决定上层是重试、换档降级、还是重建上下文
pub enum LaunchError {
    Oom,                       // -> 驱逐/swap 后重试
    Driver,                    // -> 上下文重建（读界恢复）
    Fatal,                     // -> 不可恢复
}
```

### 1.3 三个实现（unsafe 只在各自的 launch 内）

```rust
// 档1 -------- Vendor：FlashInfer cubin / cuDNN / CANN aclnn --------
pub struct VendorCubinProvider {
    libs: HashMap<JitKey, Arc<JLib>>,     // cuLibraryLoad/aclrtLoad 的句柄
    cache: JitCache,
}
// launch 内部：cuLaunchKernelEx / aclrtLaunchKernel + FFI 边界 catch_unwind

// 档2 -------- Native：cudarc / CubeCL 原生 kernel（norm、采样、量化、gather）--------
pub struct NativeProvider { kernels: HashMap<OpKind, NativeKernel> }
// NativeKernel::ptr 是编译期固定的 PTX/CUDA-C 符号，launch 用 `cudarc` 驱动层

// 档3 -------- Jit：Triton/TileLang/AscendC 桥接 --------
pub struct JitProvider { engine: JitEngine }
impl JitProvider {
    /// 与 FlashInfer JitSpec 同构的三段式：try_load -> build -> load
    fn get_or_build(&self, key: &JitKey, src: &KernelSource) -> Result<JLib, LaunchError> {
        if let Some(lib) = self.cache.try_load(key) { return Ok(lib); }   // 哈希文件命中
        let _guard = self.cache.lock(key)?;                               // FileLock 跨进程
        if let Some(lib) = self.cache.try_load(key) { return Ok(lib); }   // 双检查
        let lib = self.cache.build(key, src)?;                            // nvcc/ninja(与JitSpecNvcc同路)
        Ok(lib)
    }
}
// AscendC 外部模块同样走这条链：kernels/{op}/ascendc/*.cpp -> ATC bisheng 编译 -> JLib
```

### 1.4 选择与自动调优

```rust
/// 确定性选择：调优测量值优先，同分按 tier 稳定排序（Vendor > Native > Jit）
pub fn select<'a>(providers: &'a [Box<dyn KernelProvider>], cfg: &OpConfig)
    -> &'a dyn KernelProvider {
    providers.iter()
        .filter(|p| p.matches(cfg))
        .max_by_key(|p| (tune_score(cfg, p.tier()), p.base_priority(cfg), p.tier() as i32))
        .expect("至少一个 fallback provider")
}

/// 调优器：CUDA event / aclrtEvent 计时，结果写入 tune.json
/// 环境变量 AUTOTUNE=offline 时 CI 批量调优并入库（FlashInfer tuning_configs 模式）
pub fn bench_and_record(p: &dyn KernelProvider, cfg: &OpConfig) -> TuneEntry { ... }
```

选择样板（Attention，读写次数最多的路径）：

| 条件 | 选择 |
|---|---|
| CUDA sm90/100 & batch 桶 ≤ 64 & FP16 | Vendor FA3（flashinfer cubin） |
| CUDA & FP8/NVFP4 | Vendor CUTLASS FA3 fp8 路径 |
| CUDA & 变长 varlen 小 batch | Native FA2（CubeCL/gllm-kernels） |
| CANN & batch 大 | aclnn 官方注意力/大 GEMM 拆分 |
| CANN & MLA/稀疏 | AscendC 自研核（档3 编好即并入） |
| CPU | llama.cpp 风格 tiled f16（Rust SIMD 重写） |

### 1.5 安全与工程细节

- 所有 `launch` 调用外层 `catch_unwind(AssertUnwindSafe(..))`，且 unsafe block 带 `// SAFETY:` 注释——FFI 边界是审计重点；
- `WorkspaceArena` 每 forward 复用，`workspace_size` 不足时返回 `Oom` 由上层扩池；
- 内核热路径 `panic = "abort"`，但 FFI 盒区（call cuda 库）之内可捕获并分类；
- 数量纪律：每个 OpKind 至少有 **1 个 Rust 纯 CPU 参考实现**，永远可以 without 显卡 debug（FLA 四方对照文化）。

---

## 补充二：RadixCache 数据结构设计

### 2.1 为什么用"索引 + arena"，而不是指针/Rc 树

- **自引用问题**：Rust 里构造自我引用的树（子节点指针回指父节点）很别扭；用 `Vec<RadixNode>` + `u32` 索引彻底绕开；
- **缓存局部性**：所有节点连续存放，哈希表 `page_key -> PageRef` 的扫描友好（SGLang 的"哈希代替指针树"用了同一个理由）；
- **防悬垂**：`PageRef { idx, gen }` 代际校验，arena 复用槽位时旧引用立即失效——把 C++ 版"索引越界回到旧实例"这类 bug 变成 100% 可复现的 debug assert；
- **并发模型**：单写者（调度器线程）+ 只读（worker 读页表），写者用 `Epoch` 版本号让旧快照读安全（同 mini-sglang 的 integrity check，ref_count 保护+驱动式不变量）。

### 2.2 结构定义

```rust
pub type PageIdx = u32;
pub struct PageRef { pub idx: PageIdx, pub gen: u32 }        // 代际校验，Copy

pub struct RadixCache {
    arena: Vec<RadixNode>,                // 节点池：连续内存、索引寻址
    table: HashMap<PageKey, PageRef>,     // 页内容 -> 页槽位（SGLang 哈希化）
    evict: BinaryHeap<HeapEntry>,         // LRU 堆（懒删除：弹出时再核对 lru_seq）
    lru_seq: u64,                         // 单调递增访问序号
    free_pages: Vec<PageIdx>,             // KV 池空闲页（返回给 KV allocator）
    policy: CachePolicy,                  // LRU | Hybrid(LFU 混合)
}

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct PageKey { tokens: Box<[u32]> } // 页内 token 窗口哈希（预计算，dense 比较）

pub struct RadixNode {                    // 页段节点："一段连续页 + 长度"
    ref_count: u32,                       // 活跃 batch 引用（>0 不可驱逐）
    span: PageSpan,                       // (起始页, 页数)
    lru_seq: u64,
    evicted: bool,                        // 懒删除标记
}

#[derive(Copy, Clone)]
pub struct PageSpan { pub start: PageIdx, pub len: u32 }
```

**统一布局**（mini-sglang 的"一套机制"）：`page_size = 1` 时 `PageKey` 退化为单个 token，`PageSpan` 直接指向连续 KV 槽——连续与分页两种 KV 布局共用同一套 radix 代码，只差页长参数。

### 2.3 核心方法签名与语义

```rust
impl RadixCache {
    /// 匹配最长公共前缀（"命中率"就是 LPM 策略的收益权重）；
    /// 返回 (已匹配的段, 匹配 token 数)——包含"停在页中间"的情况
    pub fn match_prefix(&mut self, tokens: &[u32]) -> MatchOutcome;

    /// 插入新 token：结束时无匹配页 -> 分配；页内混合匹配 -> 页分裂（copy-on-write）
    pub fn insert_token(&mut self, tokens: &[u32]) -> Result<PageSpan, CacheError>;

    /// 请求完成回填（SGLang 术语 cache_finished_req）：合并相邻段、更新 lru_seq
    pub fn cache_finished_req(&mut self, req: &Req) -> Result<(), CacheError>;

    /// 驱逐到预算内：栈弹出 + 懒删除核对；ref_count>0 的页跳过，交回 KV 池
    /// 被驱逐的页需要向设备下发"重置页表"（zero + kv_reset 批量命令）
    pub fn evict_lru(&mut self, budget: usize) -> Result<usize, CacheError>;
}
```

**匹配/分裂的关键流程**（伪代码级别的核心不变量）：

```
match_prefix(tokens):
  for page in token 窗口序列:
      key = PageKey{page 内容}
      if let Some(pr) = table.get(key): 命中 -> 顺延（ref++）
      else: 若此页一半命中一半未命中 -> split_page(拷贝半页到新页)，在此停止
  返回 MatchOutcome { matched_span, matched_tokens, need_insert_from: pos }

split_page: 新页从 KV 池取，拷贝后半段前先写入 token 内容；旧页 ref_count 递减，
            table 更新为两个新 key -> PageRef（代际 ++，旧 PageRef 失效）
```

### 2.4 与批处理的衔接与泄漏防护

```rust
pub struct BatchSpan {                 // RAII：析构时自动归还 ref_count（防泄漏的最后防线）
    cache: *const RadixCache,          // 内部专用指针（调度器唯一写者）
    span: PageSpan,
    _epoch: EpochGuard,                // 持有 epoch 号——驱逐时先穿透，epoch 即失效
}
```

- **OOM 级联路径**：驱逐 → 抢占（swap 页到 CPU pinned 内存）→ 暂停新请求准入 → 保底（大容器预留）。每级有日志与指标；
- **确定性**：`PageKey` 的 token 哈希用 rolling hash 预计算；steady state 零分配（所有容器 `SmallVec`/`Box<[u32]>` 池化）；`debug_assert!` 逐页校验"页内容==表记录"（mini-sglang integrity check 的思路）。

---

## 补充三：7 项目可复用资产清单

### 3.1 总表

| 项目 | 许可证 | 可复用资产 | 复用方式 | 价值 |
|---|---|---|---|---|
| llama.cpp | MIT（含少量附加条款，以 LICENSE 为准） | GGUF 格式与全部量化表；convert_hf_to_gguf.py；llama-arch 枚举法（149 架构）；repack 权重重排；SIMD vec 算子 | 格式/算法以 Rust 重写；转换脚本直接作为工具调用 | **一次接入全网模型生态** |
| FlashInfer | Apache-2.0 | cubin 下载协议（artifactory+版本校验）；JitSpec 三段式（try_load/build/load + FileLock）；autotuner JSON schema；plan/workspace/run FFI 形态 | 黑盒 FFI 调用 + 协议兼容（vendor 档主力） | 免写注意力/MoE/Mamba 顶峰内核 |
| SGLang | Apache-2.0 | RadixAttention 页面化树算法；schedule_policy 策略族（LPM/DFS-weight/FCFS/LOF）；TBO 两批重叠；mem_cache hybrid CPU offload；sglang.proto | 算法/协议参考，proto 直接采用 | 复杂工作流的设计蓝本 |
| mini-sglang | 以 LICENSE 为准 | Req 状态机双游标；模块↔生产组件映射表；page_size 统一布局；decode 按 req_id 排序确定性；tokenizer 三消息 worker | 几乎逐行参考（教学→实现直通车） | **最佳"抄作业指南"** |
| vLLM | Apache-2.0 | 页式 KV block 引用计数；SchedulingBudget 准入；batch_invariant 语义（Rust 化）；attention/量化 backend registry 模式；entry_points 插件；EngineArgs 配置中心 | 模式迁移（架构对标，不抄代码） | 最大的架构蓝图 |
| lightllm | Apache-2.0 | token 预算准入（预计峰值 token 数）；ChunkedPrefillQueue；DSPark 部署拓扑 + NIXL KV 分页传输协议；PDSelector 四策略；SubmoduleManager 进程树监督 | PD 分离阶段协议兼容 | 前置的生产级验证 |
| FLA | MIT | `@dispatch`/BackendRegistry 的 priority/verifier 设计；naive/chunk/fused 四方对照测试框架；autotune cache 三态（strict/fuzzy/always）；NaN 毒化测试；CONTRIBUTING 五原则 | 方法论整体迁移 | 档3 JIT 与测试文化来源 |

### 3.2 "直接抄"的协议/格式（零开发成本）

| 类别 | 资产 | 抄法 |
|---|---|---|
| 二进制格式 | GGUF（llama.cpp）、safetensors（HF） | Rust 原生读写器，4KB 对齐只映射元数据 |
| 量化表 | Q4_0/Q8_0/K-quants/IQ 家族 | 从 llama.cpp `ggml-quants.c` 移植为 Rust SIMD（std::arch/wide） |
| RPC / 公共 API | `sglang.proto`（gRPC 流式 Generate） | 直接采用；内部 IPC 用 rkyv/zerocopy 自研 |
| kernel 分发 | flashinfer-cubin 的 artifactory 下载 + 版本校验 | 照抄 URL/版本模式（昇腾侧采用同类"预编译供应商包"模式） |

### 3.3 "学算法不要抄代码"的部分

- GGML 的 `GGML_ASSERT` + 崩溃回溯 → Rust `panic` hook + backtrace + `debug_assert`；
- FLA 的 autotune 缓存演进（FLA_CACHE_MODE 四态）→ 我们的 `TuneDb` schema；
- vLLM 的 `docs/design/*` 设计文档文化 → 每阶段交付 RFC 文档；
- 全部 7 家的 AGENTS.md/CLAUDE.md 治理形态 → 自建 `AGENTS.md`，含"AI 贡献必须披露、禁止完全自主提交"（llama.cpp 版）与"精度变更须 RFC"（FLA 版）。

### 3.4 明确"不抄"的（写我们的理由）

| 不抄什么 | 来自 | 我们的做法 |
|---|---|---|
| 285 个模型文件范式 | vLLM | 架构枚举（llama-arch 式）+ 权重映射驱动加载器 |
| 5291 行巨型调度器类 | SGLang | 拆成 scheduler/schedule_policy/prefill_plan 小模块 |
| CMake+Make 双轨 | llama.cpp | 纯 cargo features（后端按 feature 编译） |
| rpyc 多进程模型 | lightllm | 线程 + SPSC + 共享内存（无 GIL 不需要多进程） |
| C++ 裸指针生命周期 | vLLM csrc | Rust 所有权 + epoch 代际 |

### 3.5 复用地图（engine crate 模块 → 资产来源）

```
scheduler/          ← mini-sglang Req 状态机 + SGLang 调度策略 + lightllm 预算准入
radix_cache/        ← SGLang RadixAttention + mini-sglang 统一 page 布局
kv_pool/            ← vLLM block 引用计数 + lightllm KV 池 + 自研 VMM
kernel_provider/    ← FlashInfer JitSpec + FLA dispatch 三层思想
quant/              ← llama.cpp 量化谱系 + vLLM registry 模式
model_loading/      ← llama.cpp GGUF + safetensors 直读
sampler/            ← 自研（Rust 原生重写，但语义对齐 vLLM/V1 samplers）
grammar/            ← llguidance（直接依赖，零 FFI —— SGLang 的 xgrammar 我们不需要）
comm/               ← vLLM quickreduce 拓扑感知 + NIXL 分页传输协议
server/             ← llama.cpp tools/server + vLLM OpenAI API 曲面
observability/      ← vLLM OTel 指标 + flashinfer collect_env 诊断
testing/            ← FLA verify 门禁 + mini-sglang 确定性（Differential test vs 三引擎）
```

### 3.6 建议开发顺序（与资产清单的对应关系）

1. 先写 `model_loading`（GGUF）+ CPU 后端 → 当天就能跑 Llama 小模型（对齐 llama.cpp 数值）；
2. 再写 `kernel_provider` 骨架 + JitCache（FlashInfer 协议）→ 立刻获得 CUDA 注意力；
3. 同时写 `kv_pool` + `radix_cache`（vLLM/mini-sglang 蓝本）→ PagedAttention；
4. 然后 `scheduler`：先用 mini-sglang 的 Req/DoubleCursor 把数据流打通，再逐步替换为 SGLang 策略族；
5. 最后 `grammar`（llguidance）+ PD 分离（lightllm 协议）→ 功能对标 vLLM 完成。

---

## 结语

三份文档构成完整闭环：

| 文档 | 回答的问题 |
|---|---|
| `ai-tokens-分析报告.md` | 7 个项目各自是什么、怎么设计的 |
| `Rust推理引擎设计报告.md` | 我们要做成什么样（架构/策略/路线图/风险） |
| `Rust推理引擎-深入设计补充.md` | 具体到代码骨架与资产抄写清单（本文件） |

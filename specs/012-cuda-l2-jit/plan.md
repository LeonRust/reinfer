# Plan: CUDA L2 — JitCache v1 & Jit-tier kernels

> Derived from specs/012-cuda-l2-jit/spec.md · 评审修订 r1（4 代理评审 2026-08-27）

## Architecture Decisions

- **D1 分层归属**（009 约束原样 + 评审 A-H1/A-M1/A-M2；r1 修正：编译子进程在 jit）：
  - `crates/jit`（零 unsafe、零 CUDA、POSIX）：JitCache 实体 + **nvcc 编译子进程**（工具链探测/版本提取/梯度检查/`-M` 漂移校验）——`std::process`/`std::fs` 不是 CUDA 依赖；AscendC（bisheng）按同一机制扩展为第二个编译后端；
  - `crates/kernels`：`KernelProvider`/`OpConfig`/`ProviderTier`/`select` + TuneEntry 最小结构 + CPU 参考（评审 A-H1）；
  - `crates/cuda`：nvcc 解析链（`REINFER_CUDA_NVCC`→`CUDA_HOME`→`CUDA_PATH`→`PATH`）、**cubin 加载**（`cuLibraryLoadData`/`cuLibraryGetKernel`/`cuLibraryUnload`，cudarc `driver::sys` 已含，无需新 feature）、launch（`cuLaunchKernel` 复用 009 `launch_kernel` 裸函数接口）、Jit provider（unsafe 收敛于此）。
- **D2 键组成**（r1 重写）：前缀编码 + sha256，元素为——
  1. `source`（字节）；
  2. headers 内容哈希列表（**按路径排序后再取哈希排序**；路径字符串不入键；读不到条目 → 直接报错，不占位——TOCTOU 防治）；
  3. flags **原始顺序**（`-I`/`-L`/`-include`/`-Xcompiler` 顺序敏感，禁止排序；入键者为**最终展开参数**，含 env 注入）；
  4. toolchain 版本行 + nvcc 可执行 realpath + `-ccbin` 宿主编译器 realpath/版本首行；
  5. capability 规范串（device query 归一 `sm_120`；`-a` 后缀仅在内核声明且支持时入键）；
  6. `std::env::consts::TARGET`（triple；防交叉编译误命中）。
  每个元素长度前缀；`key()` 为关联函数（无需 &self）。
- **D3 缓存布局**：`REINFER_JIT_CACHE` 显式覆盖，否则 `<系统缓存>/reinfer/jit/`；布局 `<key[..2]>/<key>.cubin + <key>.meta.json`；锁目录默认 `<cache>/locks/<key>.lock`（同挂载、同命名空间；`REINFER_JIT_LOCK_DIR` 可覆盖；flock `LOCK_NB` 轮询，`REINFER_JIT_LOCK_TIMEOUT` 默认 300s）；temp 与目标同目录（`<key[..2]>/.<key>.tmp.<pid>.<rand>`），rename 原子、不跨挂载点；提交顺序 = .cubin 先、meta 最后（meta 为提交点），meta 含 `.cubin` sha256 + key 全字段 + toolchain realpath + gencode 全量数组 + created_at。
- **D4 预烘焙**：`REINFER_CUDA_ARCH`（如 `sm_120a`）→ 同一 nvcc+原子写路径；验收"同工具链同 arch 二次命中 <50ms"（跨机命中受系统头漂移制约 → notes 记录，不承诺）；判定机默认 `sm_120a`（与真机产物同构）。
- **D5 第一次内核**：vec_add 链路最小闭环（无头文件、无布局魔法）；之后按 T5-D7 累积。
- **D6 差分 harness**：CPU 参考纯函数在 `crates/kernels`；差分 = GPU 输出 vs CPU 参考（fp32 比对，003/plan §D7：rtol 1e-5 / atol 1e-7 逐项 allclose）；**掩码位规则：掩码一致即视为匹配**（不比较 −inf 值，显式跳过 NaN 语义）；固定 seed（如 `0x5eed`），CPU 参考与 GPU 读同一份输入；**bit-exact 仅承诺"同机、同产物（同 key）、固定 grid/block 配置"的 GPU-vs-GPU 两次运行**；内核编译禁 `-use_fast_math`；rms_norm eps=1e-5 与 CPU 参考同语义（全零行两侧 NaN 相同）；dtype 矩阵 f16 入/f32 出为主档（f32 入/f32 出为可选第二档）。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/jit/src/{key,cache,lock,meta,toolchain,compile}.rs` | JitKey 编码；JitCache（提交点/双检/重建一次/清扫）；flock（rustix/fs2 safe wrapper）；meta 读写 + 校验；nvcc 解析链/版本/梯度检查；`-M` 漂移校验；`nvcc -cubin` 编译子进程 |
| `crates/kernels/src/{provider,refs}.rs` | `KernelProvider`/`OpConfig` 最小型/`ProviderTier`（显式 discriminant）/`select`（含"无 provider → 明确错误"）；TuneEntry 最小结构；vec_add/rms_norm/rope/masked_softmax/sampler-host 的 CPU 参考（纯函数） |
| `crates/cuda/src/{jit_provider,launch}.rs` | nvcc 解析链（env 顺序）；`JLib`（`.cubin` 加载句柄）；Jit provider（`matches`/`base_priority`/`launch`）；launch 原语复用 009 |
| `crates/cuda/kernels/` | `.cu` 源码 + 头文件资产（003 约束：资产路径 + PR 审查） |
| `crates/cuda/tests/jit_smoke.rs` | 真机 ignore 用例（差分/确定性/命中/跨进程锁并发首发/产物损坏重建） |
| `crates/jit/tests/prebake.rs` | 无 GPU 预烘焙路径（需本机 nvcc；`REINFER_CUDA_ARCH`） |
| `crates/cuda/examples/kernel_diff.rs` | 手动差分示例（真机核对入口） |

## Interface Contracts

```rust
// ---- crates/jit（零 unsafe、零 CUDA、POSIX）----
pub struct JitKey([u8; 32]);
pub struct KernelSource {
    pub name: &'static str,               // 导出符号名（extern "C" __global__）
    pub src: &'static str,                // .cu 源码（include_str! 自 crates/cuda/kernels/）
    pub headers: Vec<crate::HeaderFile>,  // (path, content)，内容哈希入键；路径仅诊断用
    pub flags: Vec<String>,               // 原始顺序（调用方保序；含 -gencode 全数组）
    pub arch: String,                     // 规范串："sm_120" / "sm_120a" / "ascend910b3"
    pub toolchain_ver: String,            // 编译器版本行（平台无关命名；nvcc/bisheng）
}
pub struct JLibMeta {
    pub key: JitKey, pub arch: String, pub toolchain_ver: String,
    pub sha256: String, pub size: u64, pub gencode: Vec<String>, pub created_at: u64,
}
pub struct JitCache { dir: PathBuf }
impl JitCache {
    pub fn open(dir: Option<PathBuf>) -> Result<Self, LaunchError>;        // 建目录 + 清扫残留 temp
    pub fn key(src: &KernelSource, toolchain: &ToolchainId) -> JitKey;     // 关联函数
    pub fn try_load(&self, key: &JitKey) -> Result<Option<(JLibMeta, PathBuf)>, LaunchError>; // .cubin 存在且 sha 一致
    pub fn lock(&self, key: &JitKey) -> Result<JitLockGuard, LaunchError>; // flock NB + 超时
    pub fn store(&self, key: &JitKey, _guard: &JitLockGuard, bytes: &[u8], meta: &JLibMeta) -> Result<(), LaunchError>; // .cubin 先、meta 提交点
    pub fn remove(&self, key: &JitKey, _guard: &JitLockGuard) -> Result<(), LaunchError>;      // 删 .cubin+meta（重建一次路径用）
    /// 锁 + 双检 + 至多一次删重建；`compile` 返回产物字节；二次失败 → 包装原错误上抛（不循环）
    pub fn build_once(&self, key: &JitKey, src: &KernelSource, compile: impl FnOnce() -> Result<Vec<u8>, LaunchError>) -> Result<(JLibMeta, PathBuf), LaunchError>;
}
pub struct ToolchainId { pub ver_line: String, pub realpath: PathBuf, pub ccbin: (PathBuf, String) }

// ---- crates/kernels（评审 A-H1；r1 补充）----
pub enum ProviderTier { Vendor = 0, Jit = 1, Native = 2, CpuRef = 3 }   // 显式 discriminant；select 不得返回 CpuRef
pub trait KernelProvider {
    fn tier(&self) -> ProviderTier;
    fn matches(&self, cfg: &OpConfig) -> bool;
    fn base_priority(&self, cfg: &OpConfig) -> i32;
    fn workspace_size(&self, cfg: &OpConfig) -> usize;
    /// # Safety: 调用方保证 cfg/ws 与设备上下文一致（catch_unwind 在实现内）
    unsafe fn launch(&self, cfg: &OpConfig, ws: &mut dyn LaunchCtx) -> Result<(), LaunchError>;
}
pub struct OpConfig { pub op: &'static str, pub device: DeviceId, pub in_dt: DType, pub out_dt: DType, pub head_dim: usize, pub batch: usize, pub seq: usize }
pub fn select<'a>(providers: &'a [&'a dyn KernelProvider], cfg: &OpConfig) -> Result<&'a dyn KernelProvider, LaunchError>; // tier+priority；全不匹配或仅 CpuRef → Err(明确消息)
```

三方拿铁：key/cache/编译在 jit；加载/launch 在 cuda；trait/select/参考在 kernels。错误分类：缓存磁盘/锁超时/IO → `Fatal`；磁盘满 → `Oom`；编译失败 → `Fatal` 附 nvcc stderr 尾；驱动加载失败 → 009 白名单分类。

## 差异注记（003 T4 原文 vs 本切片）

| 项 | 003 T4 | 本切片 | 原因 / 证据 |
|---|---|---|---|
| 产物形态 | 预烘焙 **cubin** 缓存 | 同（`.cubin`，`nvcc -cubin`；禁用 `-shared`） | 实测 `-shared -fPIC` 产物在 sm_120 判定机 `cuLibraryGetKernel` 恒 200（r1） |
| 符号导出 | 未声明 | 一律 `extern "C" __global__` | 实测 12.8/13.2 的 cubin 符号均为 C++ mangled，未 mangle 名 500（r1） |
| 工具链梯度 | sm120a ≥ 13.0 | **sm_120a ≥ 12.8（实测）** | 10 个 nvcc 版本实测：12.6 不支持 sm_120，12.8/12.9/13.0/13.2 支持且 12.8 产物在本机 launch 位精确 |
| nvcc 解析 | 未定义 | `REINFER_CUDA_NVCC` → `CUDA_HOME` → `CUDA_PATH` → `PATH` | 本机 PATH nvcc=12.6（cudarc 检测面 12.6），须可覆 |
| 键组成 | 源码 + 头闭包 + gencode/flags + nvcc 版本 + capability | 同改为：嵌入内容（headers 内容哈希，路径无关）+ flags **原始顺序** + toolchain realpath + `-ccbin` + triple；`-M` 闭包降为**构建期漂移校验** | `-M` 恒包含 ~185 系统头、单次 ~90ms，与 <50ms 命中预算冲突；嵌入内容零子进程 |
| 加载 API | cuModuleLoad/Launch（009 文本） | cuLibraryLoadData/cuLibraryGetKernel/cuLibraryUnload + CUkernel→CUfunction cast（cuda.h，实测启用） | CUDA 12+ 库加载 API；cudarc 0.19.9 `driver::sys` 已含，无新 feature；API 签名与 12.0 老 ABI 不同（跨老驱动需复核） |
| prewarm | 启动阻塞式 prewarm | 延至 L3 引擎启动切片（005） | 本切片无引擎宿主；离线预烘焙与懒构建承担拉通 |
| sampler | 四件套含 GPU sampler | sampler = host 管线（SplitMix64 + 温度 + argmax）；GPU 侧仅 masked_softmax logits | 无独立采样核时差分对象为组合管线；005 的 RNG 数学以 kernels 纯函数为锚 |
| 档位顺序 | （深入设计 §1.1/§1.4：Vendor>Native>Jit） | Vendor > Jit > Native（Jit=自有核为数值主路径；Native 保留档位） | r1 裁决；回写深入设计与 003 D2（T9） |
| 错误分类 | 未定义缓存侧 | 磁盘/锁/IO → Fatal；磁盘满 → Oom；编译失败 → Fatal 附 stderr 尾 | fail-closed；重试归上层 |

## Risk Assessment

| Risk | Mitigation |
|---|---|
| nvcc 版本/解析链错位（本机 PATH 12.6） | 解析链 env 优先 + 梯度检查早失败；三轴版本表（nvcc 判 arch / 驱动判加载 API / cudarc 检测面判 sys 绑定：要 cudarc 13.x 用 `CUDARC_CUDA_VERSION`） |
| 系统头漂移（gcc/glibc）导致跨机命中 miss | key 不含系统头内容 / `-M` 仅为构建期校验；跨机命中 notes 记录不承诺 |
| 并发首发竞态 | 锁（NB+超时）+ 双检；并发测试模拟两进程 |
| 产物损坏而 meta 完好 | meta 含 sha256 + try_load 校验 → miss → 重建一次（单测注入坏字节） |
| 陈旧产物（env 派生 flag 变化） | flags 入键为**最终展开参数**；无法枚举的环境依赖由 meta 校验 + 文档兜底 |
| 驱动加载失败 | 009 白名单分类（Oom/Driver/Fatal）+ 锁内删重建一次 |
| 加载 API 老 ABI 漂移（搬运老驱动） | notes 记录；本切片只承诺判定机/同代驱动 |
| `-M` 校验误报（读不到时的 TOCTOU） | key 计算遇不可读头直接报错（不占位），编译失败自然暴露 |
| 上下文混用（误新建非 primary context） | JLib 契约：持 `Arc<CudaContext>`（或显式"仅所属 context 存活期有效"），launch 前置 current；禁止在 Jit provider 内新建 cudarc safety-layer context |
| 锁超时分类为 Fatal 但语义暂态 | 记 changelog：重试语义归上层（L3/引擎层接管前保持 fail-closed） |

## 里程碑（建议排期）

- M0：本 spec 评审（已完成 r1）
- M1（T1/T3）：`crates/jit` JitCache 实体 + 预烘焙（CPU 单测 ≥8；`forbid(unsafe_code)` 翻转）
- M2（T2/T4/T7）：nvcc 链路 + vec_add 闭环 + KernelProvider/select 最小落地（T7 入 M2：vec_add 注册即验证选择链）
- M3（T5/T6）：diff 四件套 + host sampler + CPU 参考 + 差分（D7 容差）
- M4（T8/T9）：真机 + 008 `l2-jit` 接线 + 文档回写（R1-R5 涉及的 009/003/深入设计/边界文）

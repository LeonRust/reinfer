# Spec: CUDA runtime base (L1 — device/stream/event/buffer wrappers)

> Status: approved (review 2026-08-27, see docs/design/review-cuda-l1-2026-08-27.md) · Parent: specs/003-cuda-l0 (T2 展开) · Alignment: docs/design/cuda-phase-plan.md
> 评审要点回填：接口按 cudarc 0.19.9 实测改写（A/B/D 四组）；spec 层去 HOW（D-H1）；拷贝方向修正（C-F2）；query/NotReady 特判（A-M1/C-F3）；跨设备 D2D=运行时探测（B#8/C-F4）；Send 决策锚=002/plan（D-M1）。

## Problem Statement

L1 提供**可安全消费**的 CUDA 运行时基座：五个类型 + 拷贝族，全部 RAII + `Result<_, LaunchError>`（T3 白名单，fail-closed），为 L2/L3 提供唯一 CUDA FFI 入口。本切片不写任何 kernel。

## Success Metrics

- **接口完整**：`CudaContext`/`CudaStream`/`CudaEvent`/`DeviceBuffer`/`HostBuffer` + `copy`/`copy_async`；语义与 002 契约同构（跨端对称）。
- **真机 smoke（判定机 RTX 5090 sm_120）**：
  - 设备列表：`device_count() >= 1`；`device_info(0)` 断言 name 非空、major.minor 与判定机算力一致、total_mem > 0、uuid 可解析为 8-4-4-4-12 十六进制；
  - **三链往返 H2D → D2D → D2H**（两个不重叠 `DeviceBuffer` 参与 D2D 段），同步版与异步版（`CudaStream::synchronize()` 后）最终 dst 与 1 MiB 确定性填充源**逐字节相等**；
  - 事件 record → synchronize 成功；`query()`（未 record 事件）== `Ok(true)`——**完成态**（2026-08-27 真机实测回填：从未 record 的事件即完成态，评审 C-F3"未 record→false"预设作废；`NotReady(600)→false` 分支由 error.rs 纯函数单测覆盖）；
  - 泄漏：循环前快照 `(free_before, total)`，循环 1000 次 `alloc(1 MiB)→写读回验→free`，结束后 `free_after >= free_before - (1 MiB×1000×1%) - 8 MiB(slack)` 且 `total` 不变；**独占设备 + 单线程**前提（`CUDA_VISIBLE_DEVICES` 固定、`--test-threads=1`）。
- **无 GPU 单测（具名清单，防空转）**：error 5 例（已有）+ `DeviceInfo` Debug/Clone/parse + `DeviceId` 伪造的归属校验纯函数 + `MemRef` 方向/边界校验；以 `cargo test -p reinfer-cuda -- --list | grep -c ': test$' >= 7` 为闸，禁止"空跑绿"措辞。
- **工程**：默认 feature 零 CUDA 依赖；`--features cuda` 编译 + smoke 在（gpu.yml `smoke` job 或本地真机命令，见 008 接线表）通过。
- **线程语义**：per-thread 设备绑定（`cudaSetDevice` 一次/线程）文档化，`current_device()` 供调试校验。

## User Stories

1. 作为后端作者：初始化上下文后即可创建流/事件/缓冲区并拷贝，全部返回分类错误。
2. 作为引擎调度者：缓冲区可在 worker 线程间传递（Send），语义为"仅限归属设备使用"。
3. 作为维护者：无驱动环境也有可执行的单测集（具名、带断言），不会"空跑绿"。

## Acceptance Criteria

- [ ] `CudaContext`：init/device_count/device_info/current_device；per-thread 设备绑定文档化；子资源释放次序由用户纪律保证（Context 无子资源注册表）
- [ ] `CudaStream`：创建/销毁 RAII；synchronize；句柄仅内部可见（`pub(crate)`）
- [ ] `CudaEvent`：record/synchronize/query；**query 特判未完成（NotReady 码）→ Ok(false)**，其余错误走白名单；创建默认 **BlockingSync 标志**（否则 synchronize 不阻塞 CPU）；Drop 自动 synchronize（**语义：仅等待本事件**，非设备级同步）
- [ ] `DeviceBuffer`/`HostBuffer`：alloc/free/as_ptr/size；`DeviceBuffer` 实现 `Send`（SAFETY：仅限归属设备使用；锚=002/plan 契约行）
- [ ] 拷贝：`copy/copy_async` 以 **MemRef（Device/Host 视图）** 为入参——内部做方向/边界/归属校验，**不对外暴露裸指针**；`copy_async` 返回 `CudaEvent` 作为同步凭证；**跨设备 D2D = 运行时探测**（peer 能力），不支持/失败按白名单分类——不得硬编码"不支持"
- [ ] 注入式错误精确变体断言：(a) `alloc(size = total_mem + 1)` → `Err(Oom)`；(b) `init(device_count())` → `Err(Fatal)`（码 101 不在白名单，注释 fail-closed 语义）；(c) 涉及 `IllegalAddress` 的用例置于独立进程/测试二进制末尾（上下文毒化）
- [ ] 无 GPU 具名单测集达标（计数闸 ≥7）；（工程）`--features cuda` 编译 + smoke 在 gpu.yml `smoke` job（008 接线表）或本地真机命令下通过；`bench/runner-info.json` 记录设备四元组（UUID/驱动/sm/cuBLAS）

## Non-Goals

- 任何 kernel / JitCache（L2）；内存池/VMM 策略（crates/memory 的 MemOps）；CUDA Graph 捕获（006）；多设备拓扑/TP；平台适配说明外的内容；CUresult 双轨错误体系（本切片统一 cudaError_t）
- 跨设备 D2D 的"保证支持"——行为是运行时探测，非本切片承诺项

## Constraints

- 仅经 cudarc（runtime/driver 子集，`default-features=false`）；错误收敛 T3 白名单（fail-closed）；本 crate 是 CUDA FFI 唯一宿主——`crates/jit` 只做编译流水线（nvcc 子进程/缓存/锁），装载/执行留在此处（决策见 plan D1）。changelog(2026-08-27, 012 r1)：加载 API 更新为 `cuLibraryLoadData/cuLibraryGetKernel/cuKernelGetFunction/cuLaunchKernel`（模块型旧 API 在现代驱动面弱化；另三条真机纪律见 specs/012 差异注记）
- 不修改 cudarc 类型；不新增第三方 GPU 依赖；`DeviceId` 使用 `crates/core` 类型

# Tasks: CUDA runtime base (L1)

> Derived from specs/009-cuda-runtime-base/plan.md · 每任务可独立验证；T3 的错误前置为 T4/T5 依赖。

## T1: CudaContext + DeviceInfo

- 实现 `init/device_count/device_info`（D2/D5）；每线程绑定文档；`TODO(probe)`：设备属性函数（cudaGetDeviceProperties vs cudaGetDeviceAttribute 组合、UUID API）
- Verification: 无 GPU `cargo test -p reinfer-cuda` 空跑绿；真机 `cargo test -p reinfer-cuda --features cuda -- --ignored` 的 `device_info_smoke` 通过（输出 `name/major.minor/total_mem/uuid`）

## T2: Stream / Event

- RAII + 句柄暴露；`record/device_sync/query`；Drop 次序（D3）
- Verification: 无 GPU 单测（构造/Debug）；真机事件往返（record→device_sync 成功）；`query()` 结果状态正确

## T3: DeviceBuffer + HostBuffer

- `alloc/free/as_ptr/size` + `Send`（SAFETY） + `debug_assert` device 归属；pinned host 同款
- Verification: 无 GPU 构造/Debug；真机 alloc/free 1000 循环后驱动内存计数稳定（`cudaGetMemInfo` 快照差 ≈0，处理驱动缓存语义——允许 ±1% 阈值）

## T4: memcpy

- `copy()` 同步/异步（D4）；跨设备 D2D → 明确 "unsupported"（Fatal + 文档）
- Verification: 真机三链往返（H2D→D2D→D2H 与源逐字节一致）；异步到 `Stream.synchronize()` 后一致

## T5: 错误贯通与日志

- 每个 cudarc 返回路径 → `map_err`（T3 白名单）；`tracing::warn` 附 `error_string()`（D6）；`LaunchError` 透传正确
- Verification: 注入式单测（mock 不可行——用真实失败路径：非法 size 分配、错误 device index 启动）在真机断言 `Oom/Driver/Fatal` 分类正确

## T6: 集成与文档（验收闸）

- `tests/smoke.rs`（`#[ignore]` + allowlist 登记 `gpu-smoke` job → 008 T2）；`bench/notes.md` 记录设备信息；no-GPU 单测集（T1-T3 空跑）
- Verification: `cargo test --workspace`（无GPU）绿；`--features cuda -- --ignored` 真机 smoke 全绿；clippy/fmt 绿

---

Completion gate：T1–T6 完成；真机 smoke 通过；设备信息与行为记录入 notes；无 GPU CI 档不变红。下一步 L2（specs/010-cuda-ops-v1，JitCache+首批内核）。

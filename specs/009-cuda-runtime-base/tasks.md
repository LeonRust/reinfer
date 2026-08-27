# Tasks: CUDA runtime base (L1)

> Derived from specs/009-cuda-runtime-base/plan.md

## T0: core::DeviceId

- `crates/core/src/device.rs`：`DeviceId(u32)` + Debug/Clone/PartialEq + `From<u32>`
- Verification: `cargo test -p reinfer-core`；无 feature 依赖确认

## T1: CudaContext + DeviceInfo

- `init/device_count/device_info/current_device`（runtime sys 窄绑定；`TODO(probe)` 项实现期逐一 ±编译断言：`cudaDeviceGetProperties` 签名、uuid 字段类型）
- Verification: 无 GPU 具名单测 `context::tests::device_info_prop_parse`（**构造 DeviceInfo 纯数据**：uuid 格式/name 长度/Debug 含字段——不依赖驱动）+ `--list` 计数闸；真机 `device_info_smoke`（断言见 spec 指标 2a）

## T2: CudaStream / CudaEvent

- RAII + handle(pub(crate))；事件 `BlockingSync` + record/synchronize/query（**600 → Ok(false) 特判**，`cudaEventQuery` 自包）
- Verification: 无 GPU 单测（Event 语义的纯函数：`query` 的 NotReady 特判分支用注入码值测试）；真机：未 record 事件 `query()==Ok(false)`；record→synchronize→`query()==Ok(true)`；**禁止**在存在未完成异步工作的 stream 上断言 `==false`（flaky）；Drop 兜底后再 query 仍 ==true

## T3: DeviceBuffer / HostBuffer + 归属校验

- alloc/free/as_ptr/size；`DeviceBuffer` unsafe impl Send（SAFETY 注释=锚引用）；边界/方向校验为**纯函数**（`fn validate_memref(kind, len, dev_current) -> Result<..>`）
- Verification: 无 GPU 纯函数单测（**伪造不同 DeviceId 的两个 buffer** → 跨设备校验 Err；len 越界 Err）；真机 alloc/free 泄漏（F7 公式：1 MiB×1000 + slack + total 不变，独占前提）

## T4: copy / copy_async

- `MemRef` 封装 + copy/copy_async（同步用 `memcpy_*_sync`；异步用 async + 返回 Event；跨设备 peer 探测）
- Verification: 真机三链 H2D→D2D→D2H（1 MiB 确定性填充源，同步+异步（sync 后）逐字节相等）；非法方向/越界 → Err（纯函数已测）；peer 双卡可用时递补用例（不可用时 skip+记录）

## T5: 错误面贯通（精确变体断言）

- 全部 cudarc 返回路径 → `map_err`；`error_string()`→`to_string_lossy()`→`tracing::warn`；query 特判 600 注释
- Verification: 真机注入（每例断言**精确变体**）：(a) `alloc(total_mem+1)` → `Err(Oom)`；(b) `init(device_count())` → `Err(Fatal)`（101 不在白名单，注释 fail-closed）；(c) IllegalAddress 用例 → **独立测试二进制且置于末尾**；日志仅人工观察（`RUST_LOG=reinfer_cuda=debug`），不入断言面
- Verification(无 GPU): `error.rs` 现有 5 例保持绿

## T6: 集成、ward与 008 接线（验收闸）

- `tests/smoke.rs`（`#[cfg(feature="cuda")]`+`#[ignore]`）；allowlist 预登记（`smoke::device_info_smoke`/`memcpy_roundtrip`/`event_query_states`/`alloc_free_1000_no_leak`/`error_injection`，注释 `# gpu.yml: smoke`）；`bench/` 建目录 + `runner-info.json`（四元组）+ `bench/notes.md`（真机执行痕迹：commit hash/驱动/UUID/sm_120/命令）；**修 F1 之后的 checked-ignores.sh 应验**（一个故意未登记的 ignore 必须 exit 1）
- Verification: `cargo test --workspace`（无 GPU）绿且 `--list` 计数 ≥7；本地真机命令 `CUDA_VISIBLE_DEVICES=0 cargo test -p reinfer-cuda --features cuda --test smoke -- --ignored --test-threads=1` 全绿并记录；clippy/fmt 绿；008 接线表新增 `smoke` 行（本章 T-Gate=008 D5 行生效）
- 完成后：`cuda-phase-plan.md` L1 标记完成（提交说明引用本切片 commit）

---

Completion gate：T0-T6 完成；真机 smoke 全绿（notes 留痕）；无 GPU 计数闸 ≥7；008 `smoke` job 行合并。下一步：L2（003 T4/T5；若展开则建 specs/010 并同步 phase-plan 与 003 tasks——见 009 评审 D-M4 措辞）。

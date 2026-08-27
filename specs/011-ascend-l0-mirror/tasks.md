# Tasks: Ascend L0 mirror

> Derived from specs/011-ascend-l0-mirror/plan.md · 可独立推进（M0-M3 里程碑）

## T1: 共享校验层回填（已完成）

- `kernels::mem_check` 已上移（commit 847b026）——将 011 spec 中"复用共享校验"需求闭环
- Verification: 已过（kernels 8 tests + CUDA 真机 25）

## T2: cann-rs DeviceProps（跨仓库，用户节奏）

- 0001 L0 扩展：`DeviceProps { soc_name, total_mem, compute_major/minor, uuid_str }`
- Verification: cann-rs 自身单测 + (NPU) smoke；reinfer 侧仅消费

## T3: crates/ascend 消费层（可先无 NPU 编写）

- `context/stream/event/buffer` + `AscendMemRef` + `copy/copy_async`（cann-safe + `kernels::mem_check` 复用）
- → **仅编译**（NPU 缺失时 `cargo check --features ascend` 需能过：cann 为 optional/patch，ffi feature 关）
- Verification: 无 feature 全 workspace `cargo check`；`--features ascend`（有 SDK 无 NPU）编译通过；mem_check 复用路径单测（纯逻辑部分可 CPU 测）

## T4: 无 NPU 单测集

- DeviceProps 格式化/校验纯测（同 009 的 16-count gate 思路：≥7）
- Verification: `cargo test -p reinfer-ascend` 计数达标

## T5: NPU 真机 smoke（设备供给后）

- `tests/smoke.rs` 5 类同构用例（设备/往返/事件状态/泄漏/注入）+ allowlist `# npu.yml: smoke`
- **执行包已就绪**：`npu-test-checklist.md`（本目录；前置/构建/运行/用例对照/探针 P1-P6/失败排查/回填去向）
- 运行：`cargo test -p reinfer-ascend --features ffi --test smoke -- --ignored --test-threads=1`
  （`ffi = ["cann/ffi"]`：目标机构建开关；本地开发机禁止开启——无 CANN 无法链接）
- Verification: NPU 上全绿；**先测后钉**（P1 未 record 事件行为、P3/P4 实际错误码、P5 peer 语义走 R3 回填）

## T6: 008 接线 + 差异注记

- 008 D5 接线表 `can-gpu` 行启用；011 差异注记表入 002/plan（Event 无 flag、peer 语义、TODO(probe) 清单）
- Verification: 008 文档与 011 一致；无未划线的差异项

---

Completion gate: T1–T6（T5 需设备）；"镜像验收"达成 = 对称矩阵全 ✅ + 差值注记归档。

# Plan: Ascend L0 mirror

> Derived from specs/011-ascend-l0-mirror/spec.md

## Architecture Decisions

- **D1 复用优先**：`crates/ascend` 不复刻 ANY 校验/策略——全部走 `crates/kernels::mem_check`（011 的直接理由）；DeviceId 同 core。
- **D2 消费层 API 同构**（对齐 009 契约）：

```rust
// crates/ascend（feature ascend；经 cann crate）
pub struct AscendContext;                       // cann::Context RAII + device count/set（per-thread 语义与 CUDA 同）
pub struct AscendStream;                        // cann::Stream（Drop 仅销毁）
pub struct AscendEvent;                         // cann::Event；synchronize 天然线程阻塞（ACL 语义——无 BlockingSync 对等物，差异注记）
pub struct AscendDeviceBuffer;                  // cann::DeviceBuffer；Send（锚 0001）
pub struct AscendHostBuffer;                    // cann::HostBuffer（pinned）
pub enum AscendMemRef<'a> { Device(&'a AscendDeviceBuffer), Host(&'a AscendHostBuffer) }
pub fn copy(dst, src, bytes, stream: Option<&AscendStream>) -> Result<(), LaunchError>;
pub fn copy_async(dst, src, bytes, stream) -> Result<AscendEvent, LaunchError>;
// 内部：kind_of + kernels::mem_check::validate_memref（同 CUDA 路径算法）
```

- **D3 跨设备 D2D（差异注记）**：`aclrtMemcpyPeer(dst, dstDev, src, srcDev, count)`——无能力探测 API；执行失败 → `cann::Error` → 白名单分类（fail-closed）；`validate_memref` 的 `allow_peer=true` 分支语义 = "按后端能力承担"。
- **D4 DeviceProps 缺口给 cann-rs**：0001 的 L0 扩展（`DeviceProps`：SoC、显存、算力、uuid 表示），reinfer 侧 `device_info` 镜像消费（cuBLAS 等价物无——用 SocName）。属于 cann-rs 0001 增量；本 plan 只列"消费契约"。
- **D5 测试镜像**：`crates/ascend/tests/smoke.rs`（5 类同构用例，`#[ignore]`）+ allowlist `# cuda? can-gpu` 映射；无 NPU 档：mem_check（已共享）+ DeviceProps 纯格式化。
- **D6 验收顺序**：先共享校验（已完成 ✅）→ cann-rs 侧 DeviceProps → ascen 消费层 + 无 NPU 单测 → NPU smoke（设备供给）→ 008 接线。

## Module Breakdown

| 模块 | 内容 |
|---|---|
| `crates/ascend/src/{context,stream,event,buffer}.rs` | cann-safe 消费（D2） |
| `crates/ascend/src/mem_check_use.rs`（或并入 buffer.rs） | kind_of + validate 调用（复用 kernels） |
| `cann-rs`（跨仓库） | `DeviceProps` 增量（0001 L0 扩展，待用户节奏） |
| `crates/ascend/tests/smoke.rs` | 5 类同构用例（D5） |

## Interface Contracts — 差异注记表

| 项 | CUDA (009) | Ascend (011) | 差异 |
|---|---|---|---|
| Event 同步 | 需显式 `CU_EVENT_BLOCKING_SYNC` 才阻塞 CPU | ACL 天然阻塞 | 无 flag 对等物；文档注记 |
| 事件"未 record" | 完成态（实测回填） | 待 NPU 实测（推测同构——R3 先测后钉） | TODO(probe) |
| peer 跨设备 | CanAccess 探测 + PeerAsync | `aclrtMemcpyPeer` 直接调，错误分类 | Ascend 无探测 API |
| per-thread 绑定 | `cudaSetDevice` | `aclrtSetDevice` | 语义同构 |
| Send | DeviceBuffer (0001 决策) | 同（0001 契约已定） | 无差异 |
| 错误码体系 | `cudaError_t` 白名单 | `aclError` 码段白名单（002 表） | 同 fail-closed 规则 |

## Risk Assessment

| Risk | Mitigation |
|---|---|
| NPU 设备二次供给延迟（D6 挡点） | 无 NPU 单测 + `can-gpu` 接线先行（记录档不卡） |
| `aclrtMemcpyPeer` 语义/签名漂移 | 0001 verify-list 风格 + 真机后回填（R3） |
| Event 未 record 行为与 CUDA 不同 | TODO(probe) + changelog；不假定对称 |
| 消费层与 CUDA 面漂移 | D2 同构签名 + 双向 review（契约为锚） |

## 里程碑（建议排期）

1. M0：本 spec 评审（同 009 4 代理流程，若需）
2. M1：cann-rs `DeviceProps`（用户侧节奏）
3. M2：`crates/ascend` 消费层 + 无 NPU 单测（可先行，DeviceProps 用 SoCName 起步）
4. M3：NPU smoke + 008 `can-gpu` 接线（设备就绪）

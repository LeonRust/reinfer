# 009-cuda-runtime-base 评审裁决记录（2026-08-27）

> 评审方式：4 个独立代理（A-契约一致性 / B-CUDA API 事实核查 / C-测试与验证 / D-SDD 质量），只读分析，cudarc 0.19.9 本地源码逐项核对，实测验证（rustc 1.98.0、`--list` 行为、构建失败传播）。
> 本记录为采纳裁决的唯一依据；已落实见 specs/009 与下述修订。

## 采纳并落实

| # | 变更 | 来源 | 位置 |
|---|---|---|---|
| 1 | **checked-ignores.sh 双bug修复**（`--list` 不标注 ignore；构建失败被吞恒绿）——先行修复提交 | C-F1 | `scripts/ci/checked-ignores.sh`（重写：`--list --ignored` + 构建失败 exit 1）+ 008 spec AC 命名修正；提交 `7d2f7b0` |
| 2 | **接口契约按 cudarc 实测全量改写**：runtime 层窄绑定单码体系（cudaError_t）；类型/构造/句柄名对齐（get_device_prop、uuid 直取、`memcpy_*_sync` 现成、`cudaDeviceGetAttribute`/`cudaMemGetInfo` 正名、`cudaSetDeviceAsync` 不存在）；Event 创建显式 BlockingSync（否则 synchronize 不阻塞） | B 全部 + A-L6 | 009 plan/spec D2-D7 |
| 3 | **copy 去裸指针**：改成 `MemRef` 视图（Device/Host buffer-pair），内部边界/归属校验；异步返回 Event 作为同步凭证 | A-H2 + A-L8 | 009 plan D4/契约 |
| 4 | **跨设备 D2D 改为运行时探测**（`cudaDeviceCanAccessPeer` + peer 路径），撤销"硬编码不支持"；spec 不设"不支持"Non-Goal | B#8 + C-F4d + D-L1 | 009 spec AC/plan D4 |
| 5 | `Event::query()` 特判 `cudaErrorNotReady=600 → Ok(false)`（否则 fail-closed 变 Fatal 与契约矛盾） | A-M1 + C-F3（实测 600） | 009 T2/plan；（error.rs 常量待实现加） |
| 6 | `DeviceId`（core 纯类型）替代裸 u32 | A-L2 | 009 契约 + T0（新建 crates/core/src/device.rs） |
| 7 | 无 GPU 单测具名清单 + 计数闸（≥7），删除"空跑绿"措辞；泄漏口径改绑定分配体积公式 + 独占前提（原 ±1% 对 24GB 失效） | C-F5/F7 + D-M3 | 009 spec 指标/T1-T6 |
| 8 | 008 接线表新增 `smoke` 行（job 名并入 008 唯一接线表，不再自造）；allowlist 预登记；bench/ 建目录（runner-info.json + notes.md 真机留痕）；完成门槛=notes 记录+本地真机全绿（008 gpu.yml 落地前 CI 无证据不卡） | A-M2 + C-F6 + D-H2 | 008 plan D5/spec 引用方/tasks T5；009 T6 |
| 9 | spec 层去 HOW（`debug_assert`/`SAFETY 注释`/`pub(crate)`/`map_err` 等 7 处逐行删除） | D-H1（行清单） | 009 spec 重写 |
| 10 | 消除双轨/重复：003 spec AC+T2 改指针引 009；Send 决策锚=002/plan 契约行（补"决策锚=cann-rs 0001"）；phase-plan L1 锚 009；feature-list P1-01 补 009 锚 | D-M1/M2/M4/L2 + A-L3 | 003/002/phase-plan/feature-list |
| 11 | cudarc feature 收窄 `default-features=false + driver/runtime/dynamic-linking` | C-F10 | 根 Cargo.toml（编译验证通过） |
| 12 | FFI 归属决策明示：jit=crates/jit 仅编译流水线（不触 FFI），装载/执行留 crates/cuda；"唯一 FFI 入口"措辞修正 | A-H1 | 009 plan D1 |
| 13 | Driver 分类语义降级：L1 仅分类，重建/重试语义属上层（L3/010）；error.rs 注释同步 | A-M3 | 009 T5/plan（error.rs 注释修订列实现变更） |
| 14 | Event Drop 语义明示"仅等待本事件"（非设备全同步）；CudaContext 弱校验 Drop 删除（不可实现）；Event 与 0001 侧不对称在文档注明 | A-L1/L7 + B#4 | 009 plan D3 |

## 驳回/说明

- "±1% 严格 ==0"（D-M3 建议的 ==0）：驱动缓存/碎片下 flaky，采纳 C-F7 的带松弛公式（绑定体积 + 1% + 8MiB slack）。
- "Hash 退化 UUID"（原 plan D5）：B 确认 uuid 直取可读，方案移除。
- "100 次 iter 后池回基线"（无此项）——本切片无池。

## 正面确认

六要素完备；跨端对称（Send/白名单三分类/API 形状）；范围克制（无 kernel/无策略/无 Graph）；Reference assets 单向锚定；Non-Goals 无越界；cudarc 的 `CudaContext/CudaStream/CudaEvent` 本身 `Send+Sync`（对跨线程无害，F12）。

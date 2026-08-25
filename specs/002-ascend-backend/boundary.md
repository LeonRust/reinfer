# Ascend ownership boundary: reinfer ↔ cann-rs

> Status: agreed 2026-08-25 · Anchor: `specs/002-ascend-backend/plan.md` (L0 contract)
> Mirror: `cann-rs/docs/boundary-with-reinfer.md` (Chinese) — when in doubt, this anchor wins.

## 1. Purpose

reinfer (engine) and cann-rs (SDK bindings) grow in parallel without overlap or gaps. This document is the **single source of truth for ownership** — every Ascend capability below belongs to exactly one side.

## 2. The single ownership rule

Apply the SDD granularity test to the hardware stack:

> **If the work survives an SDK swap (replace CANN with CUDA) → reinfer.**
> **If the work is specific to the CANN SDK surface → cann-rs.**

Example: "KV page eviction policy" survives the swap (CUDA needs it too) → reinfer. "aclrtMalloc signature" does not → cann-rs.

## 3. Layering

```
CANN SDK (8.x/9.x) → cann-sys (bare FFI) → cann (safe API)   ← cann-rs owns
        → reinfer-ascend (backend consumption) → reinfer engine   ← reinfer owns
```

Consistent with cann-rs constitution §1 (`cann → cann-sys`, no reverse/cross-layer).

## 4. Responsibility matrix

| Capability | Owner | Boundary note |
|---|---|---|
| aclInit/aclFinalize, Context RAII | cann-rs (`cann`) | reinfer consumes `cann::Context` |
| Device count / set / reset | cann-rs (`cann-sys` + `cann`) | reinfer maps to `core::DeviceId` |
| Stream / Event (create, record, sync) | cann-rs (bindings + safe API) | reinfer orchestrates them in `ExecCtx` |
| Device/host memory alloc/free primitives | cann-rs (`DeviceBuffer`, `HostBuffer`) | **policy** (page pools, refcount, offload, VMM-like semantics) = reinfer `crates/memory` |
| ACL error codes & `is_oom/is_recoverable` | cann-rs (SDK semantics) | reinfer maps to engine `LaunchError` (table in 002/plan.md) |
| Device properties (SoC, memory, L2…) | cann-rs (`DeviceProps`) | reinfer: capability gating + TuneDb keys |
| aclnn operator wrappers (Matmul/Softmax/RMSNorm/TopK…) | cann-rs (`cann-ops`-style crate) | reinfer: KernelProvider Vendor-tier selection + autotune |
| Graph capture (GE graph engine `aclgrph*`: aclgrphParseONNX/BuildModel/SaveModel + Session) | cann-rs (bindings) | reinfer: what goes into the graph, bucket pooling, memory reuse |

> Note (2026-08-25): CANN 8.5.0 has no `aclrtGraph*` symbols; graph APIs are GE graph engine (`aclcppdevg`/`API/ascendgraphapi`: aclgrph* + Session). Earlier `aclrtGraph*` naming referred to legacy dynamic-graph APIs.
| HCCL primitives (comm init, collective, send/recv) | cann-rs (`cann-sys` + `cann`) | reinfer: algorithms, topology, fallback (shared with CUDA `crates/comm`) |
| AscendC kernel **compile pipeline** (bisheng/AOC, cache, locks) | **reinfer `crates/jit`** | mirrors FlashInfer JitSpec; kernel source assets are engine-owned |
| AscendC custom-op **load/execute API** (aclnnCustomOp…) | cann-rs (bindings) | reinfer drives compile → load → launch |
| Version / env probing | cann-rs (already in 0.1.x) | reinfer `diag` formats the report |
| Autotune, TuneDb, benches, differential tests | reinfer | engine-level; cann-rs never benchmarks |
| Binding smoke tests | cann-rs repos' tests | reinfer: integration tests on `ascend-gpu` runner |
| Contract doc + changelog | reinfer `specs/002` (anchor) | cann-rs mirrors + links; changes go through spec changelog |

## 5. Hard rules

- **R1 One-way dependency**: cann-rs never imports reinfer types; reinfer never consumes cann-sys types directly.
- **R2 No duplicate bindings**: every SDK symbol is bound exactly once (in cann-sys).
- **R3 Contract first**: signature changes update `specs/002` before implementation in either repo (SDD anti-spec-rot, §6.4b).
- **R4 SDK 8.x/9.x gating**: cann-rs detects & reports; reinfer decides behavior (fallback vs fail).
- **R5 Cadence**: cann-rs semver — 0.1.x = L0, 0.2.x = L1, 0.3.x = L2; reinfer pins crates.io versions in release mode, uses `[patch]` in dev mode.
- **R6 Same governance ideals**: both repos use SDD (spec/plan/tasks), Conventional Commits, and no AI co-author trailers; mismatches open an issue upstream first, never silently fork.

## 6. Owner lists (recap)

**cann-rs implements** (SDK surface): Context + device/stream/event lifecycle, memory primitives, aclnn op wrappers, graph bindings, HCCL bindings, CustomOp execute API, DeviceProps, error codes/classification, version probing.

**reinfer implements** (engine surface): `crates/ascend` backend (consumption, capability gating, `diag`), KernelProvider selection + TuneDb/autotune, memory policy, comm algorithms, AscendC pipeline, benchmarks/differential tests, contract governance.

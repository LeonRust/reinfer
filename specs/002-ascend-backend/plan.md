# Plan: Ascend backend L0 integration contract

> Derived from specs/002-ascend-backend/spec.md

## Architecture Decision

- **One-way dependency**: `crates/ascend` depends only on `cann` (never `cann-sys` directly); all `unsafe` lives inside cann-rs. This preserves constitution §2.1 and keeps the FFI audit surface entirely in the partner repo.
- **Development wiring**: `[patch.crates-io]` points `cann`/`cann-sys` to `../cann-rs/{cann,cann-sys}`. Release mode: `cann = { version = "0.1", optional = true }` + `cann-sys` ffi feature handled internally by `cann` (transparent to us).
- **`crates/ascend` module layout**: `context.rs` (Context RAII over aclInit/aclFinalize), `device.rs`, `stream.rs`, `event.rs`, `buffer.rs` (DeviceBuffer / HostBuffer), `error.rs` (classification mapping).
- **Feature surface**: `crates/ascend` has `[features] default = []`; `bin/reinfer` forwards `ascend = ["dep:reinfer-ascend", "dep:cann"]` when implemented in P2 — L0 keeps only the diagnose command, gated by the same feature.

## Module Breakdown

1. `crates/ascend` — L0 host-side wrappers + `diag()` + `Error` classification
2. `bin/reinfer` — `diag` subcommand (prints versions/device count; human-readable on error)
3. `docs/sdd` — contract changelog entry when cann-rs signatures change

## Interface Contracts — L0 (proposal for `cann` crate)

> Symbol names follow CANN SDK 8.x/9.x headers. All L0 symbols were verified against official CANN 8.5.0 docs on 2026-08-25 (see cann-rs `docs/cann-850-catalog.md` §2).

```rust
// ---- lifecycle ----
pub struct Context;                                 // RAII: aclInit / aclFinalize
impl Context {
    pub fn init() -> Result<Self, Error>;
}
pub fn device_count() -> Result<u32, Error>;       // aclrtGetDeviceCount(uint32_t*) — verified vs CANN 8.5 (aclcppdevg_03_0045)
pub fn set_device(dev: u32) -> Result<(), Error>;  // aclrtSetDevice

// ---- streams & events ----
pub struct Stream;                                  // aclrtCreateStream / aclrtDestroyStream
pub struct Event;                                   // aclrtCreateEvent / Record / Synchronize / Destroy

// ---- memory (device + pinned host) ----
pub struct DeviceBuffer { /* internal ptr + size */ }
impl DeviceBuffer {
    pub fn alloc(size: usize) -> Result<Self, Error>;           // aclrtMalloc
    pub fn as_ptr(&self) -> *const u8;
}
pub struct HostBuffer;                              // aclrtMallocHost (pinned, for swap/offload)

// ---- error ----
pub enum Error { code: aclError, message: String }  // existing shape, extended:
impl Error {
    pub fn is_oom(&self) -> bool;                   // ACL_ERROR_RT_MEMORY_ALLOCATION | ACL_ERROR_RT_MEMORY_FREE...
    pub fn is_recoverable(&self) -> bool;           // driver/context class (context rebuild path)
}
```

### Error mapping table (contracted semantics)

| CANN class (aclError) | reinfer `LaunchError` | Action |
|---|---|---|
| `ACL_ERROR_RT_MEMORY_ALLOCATION` / free-class | `Oom` | evict → swap → retry |
| context/driver errors (`ACL_ERROR_RT_INTERNAL_ERROR`, context lost class) | `Driver` | rebuild context, retry request |
| parameter/invalid-class, unknown | `Fatal` | fail request, keep process |

### Pure FFI list for `cann-sys` (L0)

`aclInit` · `aclFinalize` · `aclrtGetDeviceCount` · `aclrtSetDevice` · `aclrtCreateStream` · `aclrtDestroyStream` · `aclrtCreateEvent` · `aclrtRecordEvent` · `aclrtSynchronizeEvent` · `aclrtDestroyEvent` · `aclrtMalloc` · `aclrtFree` · `aclrtMallocHost` · `aclrtFreeHost` · `aclrtMemcpy` · `ACL_ERROR_RT_*` consts
(note: `aclrtMalloc` third arg is enum `aclrtMemMallocPolicy`; `aclrtGetSocName` is `const char *aclrtGetSocName(void)` — not in L0 surface, but for DeviceProps later)

## Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Symbol drift across CANN 8/9 | High | `TODO(verify-symbol)` gate before L1; contrived `build.rs` version check in cann-rs (they already do version print) |
| cann-sys `ffi` feature default-off hides link failures until ascend feature on | Medium | Our CI compiles `--features ascend` on the dedicated `ascend-gpu` runner only; local dev runs `cargo check -p reinfer-ascend` |
| Patch vs locked versions conflict (cargo warning seen) | Low | `cargo update -p cann` after cann-rs publishes; release mode relies on crates.io only |
| Contract drift between repos | Medium | Contract table is the anchor; mismatches resolved via spec changelog (spec rot rule, constitution §6.4b) |

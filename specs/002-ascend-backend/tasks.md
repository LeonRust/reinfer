# Tasks: Ascend backend L0 integration contract

> From specs/002-ascend-backend/plan.md · Each task independently verifiable

## Task 1: Error classification in crates/ascend

- Implement `error.rs`: `LaunchError` classification from `cann::Error` (via `is_oom()` / `is_recoverable()` / code ranges), `thiserror` enums
- Verification: unit tests with synthetic `cann::Error` codes; `cargo test -p reinfer-ascend`

## Task 2: Context + diag

- Implement `Context::init()` wrapper, `device_count()`, `set_device()`; `DiagInfo { version_str, version_num, device_count }`
- Verification: with CANN SDK and driver — `reinfer diag` prints correct values; without driver — readable error, exit code 1, no panic

## Task 3: crate wiring

- `crates/ascend/Cargo.toml`: `cann = { workspace = true }` (optional via feature `ascend`); `bin/reinfer` forwards `ascend` feature + `diag` subcommand behind it
- Verification: `cargo check --workspace --no-default-features` (no SDK) passes; `cargo check --features ascend` with SDK present passes

## Task 4: Contract conformance checklist (cross-repo)

- Mirror the L0 contract table into cann-rs repo as its own implementation checklist (one smoke test per signature)
- Verification: both repos' checks pass on their respective CI; any signature mismatch → spec changelog entry before merging either side

## Task 5: CI + docs

- `ci.yml.template`: add `ascend-gpu` self-hosted job (`cargo test --features ascend`)
- README/CLAUDE env docs updated for `ASCEND_TOOLKIT_HOME` chain
- Verification: CI template rendered; env docs consistent with cann-rs README

---

Completion gate: Tasks 1–5 accepted; contract table sync'd with cann-rs HEAD; `cargo check --workspace --no-default-features` + fmt + clippy clean.

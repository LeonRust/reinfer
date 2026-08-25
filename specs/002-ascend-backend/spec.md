# Spec: Ascend backend L0 integration contract (cann / cann-sys)

> Status: proposal · Owner: maintainers · Created: 2026-08-25
> Parent: specs/000-project-mvp · Partner repo: [cann-rs](https://github.com/cann-rs/cann-rs)
> This spec defines WHAT reinfer needs from `cann`; HOW is split in this repo's plan.md and cann-rs's own implementation.

## Problem Statement

The reinfer Ascend backend (`crates/ascend`) needs the CANN SDK basics — device/context, streams, events, device/host memory, and a classified error surface — but `cann`/`cann-sys` (v0.1.x) currently only expose version probing. Both projects are in development; the cheapest correct path is a **contract as single source of truth**: reinfer publishes the L0 API contract, cann-rs implements against it, and reinfer wires a local `[patch.crates-io]` so development proceeds in parallel without blocking either repo. Deliverable of this slice: **compile + version/device diagnostics closed loop** (also runnable on NPU-less machines for CI).

## Success Metrics

- `cargo check --workspace` (default features) passes with no CANN SDK installed — Ascend stays inert unless `--features ascend` is used
- With CANN SDK 8.x/9.x installed, `cargo build --features ascend` succeeds and `reinfer diag` prints: CANN version string + version number (from `cann::Version`), device count, and a readable error path (no panic)
- `crates/ascend` contains **zero unsafe**; the `Error → LaunchError` classification mapping is unit-tested in CI
- cann-rs side: every L0 signature in plan.md §Interface Contracts compiles in `cann` with a smoke test

## User Stories

1. As a maintainer on a machine without CANN SDK, I can build and test the default feature set.
2. As a backend author, I only consume `cann`'s safe API — I never touch CANN headers or FFI.
3. As a cann-rs author, I can develop independently against the contract without reinfer blocking me (and vice versa).

## Acceptance Criteria

- [ ] `crates/ascend` exposes `ascend::diag() -> DiagInfo` (version str/num, device count); error classification unit tests green
- [ ] Workspace contains `[patch.crates-io] cann/cann-sys` (dev mode) — plan.md documents the release-mode switch back to crates.io versions
- [ ] Contract table in plan.md matches the actual `cann` signatures; mismatches are resolved against the contract (updated via spec-changelog)
- [ ] CI: default-features job is green without GPU/SDK; `ascend` job tagged `ascend-gpu` (self-hosted)

## Non-Goals

- Inference path, aclnn operator bindings (L1), HCCL / graph capture (L2), AscendC compile chain (belongs to `crates/jit`)
- Merging cann-rs into this monorepo — cann-rs stays an independent publishable repo
- Changing cann-rs's own API philosophy beyond the contract (naming/typography owned by cann-rs)

## Constraints

- cann-rs license MIT OR Apache-2.0 (compatible with constitution appendix); CANN SDK 8.x/9.x; SDK path detection per cann-rs order (`ASCEND_TOOLKIT_HOME` → `ASCEND_HOME_PATH` → `ASCEND_HOME` → `~/Ascend/cann` → `/usr/local/Ascend`)
- Legal: no `unsafe` beyond cann/cann-sys (constitution §2.1); torch forbidden (§1.3)

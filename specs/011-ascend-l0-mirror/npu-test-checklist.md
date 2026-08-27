# NPU Test Checklist — Ascend L0 mirror (specs/011 T5)

> Execution package for `crates/ascend/tests/smoke.rs` on the target NPU machine.
> Project rule: the dev machine does not build `reinfer-ascend` (no NPU); compile +
> run happen on the target machine only. Any assertion change here must be committed,
> then re-run on the target.

## A. Preconditions

- NPU device + driver present: `npu-smi info` shows ≥1 device, no `[ERROR]` lines.
- CANN toolkit (SDK 8.x) installed; `libascendcl` discoverable both at build
  (headers/libs) and runtime (`LD_LIBRARY_PATH`).
- Repo checkout at latest `main` (this suite + ffi gate).
- cann/cann-sys source: locally-patched checkout preferred (copy
  `.cargo/config.toml.example` to `.cargo/config.toml`; keep reinfer and
  cann-rs as sibling directories so `../cann-rs/*` resolves on any machine) —
  the repo default is crates.io, and the published 0.1.2 predates the memcpy
  primitives (expect a 0.1.3 publish before relying on it).
- Rust toolchain ≥ 1.85 (edition 2024), same as dev machine.

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest   # adjust per install
export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/lib64:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH
```

## B. Build (link check; `#[ignore]` means nothing runs yet)

```bash
cd reinfer
cargo test -p reinfer-ascend --features ffi --test smoke --no-run        # must link
cargo test -p reinfer-ascend --features ffi --test smoke -- --list --ignored
```

Expected 5 listed: `device_info_smoke`, `memcpy_roundtrip`, `event_query_states`,
`alloc_free_1000_no_leak`, `error_injection`.

## C. Run (acceptance gate)

```bash
cargo test -p reinfer-ascend --features ffi --test smoke -- --ignored --test-threads=1
```

Expectation: 5 passed, 0 failed, 0 ignored. `--test-threads=1` is mandatory
(ACL per-thread device binding; parallel tests perturb each other).

Sanity pass (non-smoke tests under ffi — only the 3 pure error-classification tests
run; stub tests are `#[cfg(not(feature = "ffi"))]`-gated):

```bash
cargo test -p reinfer-ascend --features ffi --lib
```

Expectation: 3 passed (error.rs classification: 207001→Oom, 507000/507033→Driver,
unclassified→Fatal).

## C2. Manual examples (L1 demo, optional)

Same env as above; results are printed for human inspection, no assertions:

```bash
cargo run -p reinfer-ascend --features ffi --example device_info
cargo run -p reinfer-ascend --features ffi --example basic_ops
```

- `device_info`: device count + SoC name per device (DeviceProps gap until 011 T2).
- `basic_ops`: sections [1]-[5] — device, stream/event (record+sync), 1 MiB
  deterministic checksum roundtrip both sync and async, 100× alloc/free, error
  injection (1 TiB over-alloc → Oom; bad dev index → non-Oom).

Manual checkpoints: checksum equality lines (src == out both chains), all
"100 次 alloc/free 全部成功", error variants match expected classification
(record actual variants back as probes P3/P4 if different).

## D. Case matrix (mirror vs crates/cuda/tests/smoke.rs)

| Test | CUDA counterpart | Assertions | Notes / differing |
|---|---|---|---|
| `device_info_smoke` | `device_info(major≥10, uuid format)` | count≥1; soc_name non-empty | DeviceProps gap (011 T2) — expand after cann-rs lands it |
| `memcpy_roundtrip` | sync + async 3-chain, event vouchers | byte-equality after each chain | Same shape; sync path = `aclrtMemcpy`, async = `aclrtMemcpyAsync` |
| `event_query_states` | code path: `evt.query()` complete states | record→sync→`stream.query()==idle` | ACL has no event query in cann-rs surface (aclrtQueryEventStatus not exposed); stream idle used as completion proof |
| `alloc_free_1000_no_leak` | memGetInfo free-before/after | 1000 × 1 MiB alloc/free all OK | No mem info in reinfer layer yet — add `aclrtGetMemInfo`-based assertion after T2 |
| `error_injection` | over-alloc `total+1 → Oom`; bad dev Fatal | 1 TiB over-alloc → Oom; bad dev → non-Oom | Exact codes recorded via probes P3/P4 |

## E. Probes (R3 backfill — record results, then adjust code/doc and commit)

| # | Probe | How | Hypothesis | Record |
|---|---|---|---|---|
| P1 | Event never-recorded → `synchronize()` | `timeout 30` guard on a one-off test (or empty suite run with `AscendEvent::new().expect("e").synchronize()`) | returns immediately (CUDA-observed completed-state behavior) or blocks — **do not** guess; if blocking, must not enter the suite | |
| P2 | Repeated `aclInit` in one process | Two `AscendContext::new()` in one test | Cann-rs Context non-refcount (design already avoids: `ensure_ctx` leaks a single ctx) — confirm no double-aclFinalize crash at process end | |
| P3 | `set_device(999999)` actual `aclError` code + classification | `error_injection` failure output, or one-off print | Driver(507xxx) or Fatal; never Oom | |
| P4 | Over-alloc (1 TiB) actual code | `error_injection` failure output | 207001 → Oom (if Driver/Fatal → cann-rs `is_oom` whitelist gap) | |
| P5 | (deferred, needs ≥2 NPUs) cross-device D2D via `aclrtMemcpyPeer` | after removing the explicit Fatal hook in `buffer.rs` | record code + classification; decide whether to open peer path in 011 diff note | |
| P6 | (optional) `npu-smi info` SoC name | `npu-smi info` | matches `soc_name` string from device_info — record for DeviceProps work | |

## F. Failure triage

| Symptom | Likely cause | Action |
|---|---|---|
| link error (`undefined reference` / no `libascendcl`) | toolkit path / `LD_LIBRARY_PATH` | fix env (A), rebuild |
| `aclInit` fails on first test | driver/toolkit mismatch | `npu-smi info`, CANN version vs driver |
| first test hangs | `aclInit` or `set_device` wait | run with `timeout 60`; then follow P1/P2 |
| over-alloc maps Driver/Fatal | cann-rs whitelist gap | **not a reinfer bug** — add code to cann-rs `is_oom`, commit there |
| single test fails but others pass | divergence between backends | read mirror table (D); backfill diff note, do not "fix" by weakening the test |

## G. Non-goals (this suite)

- No performance/benchmark measurements (reinfer's job after 002 revival).
- No multi-card / HCCL; no `aclrtMemcpyPeer` path until P5.
- No AscendC/compute kernels — L1 is runtime base surface only.

## H. Backfill destination (after machine run)

1. Record probe results into this checklists (E columns) or the 011 plan diff-note table
   (`plan.md` rows: Event未record TODO(probe), peer 语义, DeviceProps).
2. Adjust any test assumption (e.g. soc_name format) and commit.
3. Re-run section C to confirm green; then update `tasks.md` T5 → done and note in
   `008` ci-infra wiring (`npu.yml` mapping) per spec.md acceptance criteria.

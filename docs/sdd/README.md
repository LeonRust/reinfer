# Spec-Driven Development at reinfer

reinfer follows **Spec-Driven Development (SDD)**: the specification is the single source of truth; code is its derived product. Humans define **WHAT**, AI implements **HOW** (constitution §6.4).

## Workflow

```
Specify (human, spec.md) → Plan (human+AI, plan.md) → Implement (AI, code+tests) → Validate (human+AI, CI+review)
```

## Repository layout

```
specs/<NNN>-<slug>/
  spec.md     requirements: Problem Statement / Success Metrics (testable) /
              User Stories / Acceptance Criteria / Non-Goals / Constraints
  plan.md     architecture: Decision / Module Breakdown / Interface Contracts / Risk Assessment
  tasks.md    atomic tasks, each with an independent verification
CONSTITUTION.md   immutable project principles (Spec Kit's constitution.md slot)
```

## Discipline

Single source of discipline = CONSTITUTION.md §6.4 (not duplicated here). Review verdicts: `docs/design/review-2026-08-25.md`.

### Exemption lane (micro-change channel, §6.4c)

- **Eligibility**: does NOT change AC / metrics / contract signatures / feature-list status, and validates unchanged (fmt/clippy/test green).
- **Record**: reviewer notes "exempt (conditions met)" in the PR description to merge.
- **Boundary**: any change touching AC / metrics / contracts always goes through an incremental spec (non-exemptible); when in doubt, treat as non-exempt.

## Cross-repo agreements

- [`specs/002-ascend-backend/boundary.md`](../specs/002-ascend-backend/boundary.md) — Ascend ownership boundary between reinfer and [cann-rs](https://github.com/cann-rs/cann-rs) (mirror: cann-rs/docs/boundary-with-reinfer.md)

## Current specs

- `specs/000-project-mvp/` — P0 CPU feasibility loop
- `specs/001-gguf-loader/` — GGUF loader + typed core layer
- `specs/002-ascend-backend/` — L0 integration contract with cann/cann-sys (partner repo: cann-rs)
- `specs/003-cuda-l0/` — NVIDIA GPU base + single-request inference loop (first P1 slice)
- `specs/004-tokenizer/` — GGUF SPM/BPE tokenizer + incremental UTF-8 decode (feeds 003/005)
- `specs/005-scheduler-serving/` — continuous batching + OpenAI-compatible HTTP serving
- `specs/006-cuda-perf/` — FA3/CUTLASS vendor tier + CUDA graph (arch-tiered llama.cpp-CUDA gate)
- `specs/007-core-inference/` — 🔒 to write (CPU full-path loop; carrier for 005 `--backend cpu` and GPU-less CI)
- `specs/008-ci-infra/` — CI artifacts contract (jobs / runners / `#[ignore]` matrix / bench gates) — spec ready
- Parity matrix — [`specs/000-project-mvp/parity.md`](../specs/000-project-mvp/parity.md)

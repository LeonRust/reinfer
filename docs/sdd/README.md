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

## Discipline (all specs obey)

1. **Granularity check** — re-implement the spec with a different tech stack; if it breaks, you smuggled HOW into WHAT.
2. **Incremental specs** — new requirements or bug fixes immediately get an incremental spec; never let spec rot.
3. **Micro-change exemption** — purely local tweaks (copy, style, local variables) skip the full cycle.
4. **Validate is never skipped** — a spec replaces the requirements document, never Code Review or CI.
5. **RFC vs Spec** — RFC decides *whether/why*; Spec decides *to what extent*; on conflict, RFC wins.

## Cross-repo agreements

- [`specs/002-ascend-backend/boundary.md`](../specs/002-ascend-backend/boundary.md) — Ascend ownership boundary between reinfer and [cann-rs](https://github.com/cann-rs/cann-rs) (mirror: cann-rs/docs/boundary-with-reinfer.md)

## Current specs

- `specs/000-project-mvp/` — P0 CPU feasibility loop
- `specs/001-gguf-loader/` — GGUF loader + typed core layer
- `specs/002-ascend-backend/` — L0 integration contract with cann/cann-sys (partner repo: cann-rs)
- `specs/003-cuda-l0/` — NVIDIA GPU base + single-request inference loop (first P1 slice)

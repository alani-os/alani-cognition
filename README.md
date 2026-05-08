# alani-cognition

Cognitive execution service for model loading, inference requests, planning, memory retrieval, and accelerator integration.

| Field | Value |
|---|---|
| Tier | MVK required |
| Owner | Cognition team |
| Aliases | None |
| Architectural dependencies | `alani-lib`, `alani-protocol`, `alani-memory`, `alani-models`, `alani-devices`, `alani-security`, `alani-observability` |

## Quick start

```bash
cargo fmt -- --check
cargo test --all-features
cargo test --no-default-features
cargo check --no-default-features
cargo clippy --all-features -- -D warnings
```

## Scope

`alani-cognition` is the cognitive execution boundary for Alani. The current skeleton is dependency-free and provides:

- model metadata, capability flags, lifecycle state, selection, and a fixed-capacity registry;
- explicit `InferenceBudget`, `InferenceRequest`, `CognitiveContext`, `EvidenceBundle`, and `InferenceResult` contracts;
- deterministic host-mode `MockCognitionEngine` for MVK tests and corpus fixtures;
- knowledge records with provenance, confidence, redaction class, fixed-capacity stores, and retrieval results;
- bounded plan goals, steps, deterministic planner output, and authorization-aware plan status.

The crate is `no_std` when built without the default `std` feature. Sibling architectural dependencies remain in Cargo metadata until their public APIs are intentionally wired.

Keep public API changes synchronized with `docs/repositories/alani-cognition.md`, Doc 07, Doc 14, Doc 29, Doc 30, Doc 42, and Doc 43 in `alani-spec`.

## Engine Restructure Execution Index

Purpose: single-page tracker for sequence, status, and non-negotiable decisions.

Detailed planning document:
- ENGINE_RESTRUCTURE_FULL_PLAN.md

Baseline and parity contract:
- ENGINE_RESTRUCTURE_BASELINE.md

Implementation timeline:
- ENGINE_RESTRUCTURE_PROGRESS.md

## Workstream Sequence

1. A: Baseline guardrails and parity checks.
2. B1: Service-layer extraction from app.py.
3. B2: Domain module split inside engine.
4. C: Structured scouting page rebuild with rating sorting.
5. D: Structured graph/state fan-out with no CFBD in structured mode.
6. E: Open chat page with session-only memory.
7. F: Navigation and shared diagnostics shell.
8. G: Verification, hardening, and handoff.

Execution dependencies:
1. A -> B1 -> B2 -> C
2. B2 -> D -> E
3. C + E -> F
4. A + B1 + B2 + C + D + E + F -> G

## Locked Decisions

1. Dropdown sorting basis: recruiting rating only.
2. School dropdown semantics: target team context.
3. Memory scope: session-only.
4. Persona options for this phase: Fan and Scout.
5. CFBD tools enabled only on open chat page.
6. athlete_id persistence rule: positive integer only, otherwise null.

## Current Snapshot

1. Portion A complete.
2. Portion B1 complete.
3. Portion B2 complete.
4. Portion C complete.
5. Portion D complete.
6. Portion E complete.
7. Portion F complete.
8. Portion G complete.

## Engine Restructure Execution Index (Phase 2)

Purpose: single-page tracker for the Phase 2 architecture shift.

Detailed planning document:
- ENGINE_RESTRUCTURE_FULL_PLAN.md

Baseline and parity contract:
- ENGINE_RESTRUCTURE_BASELINE.md

Implementation timeline and active risks:
- ENGINE_RESTRUCTURE_PROGRESS.md

## Non-Negotiable Architecture Decisions

1. Identity split is required:
	- gi_recruit_master
	- gi_college_master
	- gi_player_link_bridge
2. Graph topology is required:
	- lead_delegator -> parallel workers -> lead_synthesizer
3. Identity matching policy is SQL-first fuzzy matching (search_text + pg_trgm), not embeddings.
4. Shared graph state stores summaries only, never raw scraped HTML/JSON payloads.
5. Streamlit UI stays thin; orchestration and business logic live under engine/.

## Workstream Sequence

1. A: Diff hygiene and safety checks.
2. B: Schema/notebook alignment for split identity model.
3. C: Engine identity helpers and state/tool contract updates.
4. D: Graph reshape to delegator + fan-out/fan-in synthesis.
5. E: UI and orchestration service separation.
6. F: Runtime validation, parity checks, and handoff.

Execution dependencies:
1. A -> B
2. B -> C
3. C -> D
4. D -> E
5. B + C + D + E -> F

## Current Snapshot (2026-03-24)

1. A complete: diff review performed; major artifact risks identified.
2. B in progress: notebook/schema updates present but needs final SQL policy validation.
3. C complete: identity-aware helpers and summary-first state fields implemented.
4. D complete: graph rewired to lead_delegator + three workers + lead_synthesizer.
5. E complete: orchestration extracted to engine service layer and app handlers rewired.
6. F in progress: runtime validation and cleanup of unexpected artifacts still pending.

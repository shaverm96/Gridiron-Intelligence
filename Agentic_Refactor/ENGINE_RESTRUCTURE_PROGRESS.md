## Engine Restructure Implementation Log (Phase 2)

Purpose: maintain one concise execution log for the current architecture migration.

Related docs:
1. ENGINE_RESTRUCTURE_PLAN.md (index)
2. ENGINE_RESTRUCTURE_FULL_PLAN.md (master plan)
3. ENGINE_RESTRUCTURE_BASELINE.md (parity contract)

## Completed Milestones

### A. Diff Hygiene and Safety Review

Status: Complete

1. Reviewed current workspace diffs before continuing rollout.
2. Identified high-risk unexpected artifacts and isolated them for follow-up.

### C. Engine Identity and State Contract

Status: Complete

1. Added identity helpers in engine Supabase client:
	- resolve_player_identity
	- fetch_player_bundle_by_identity
2. Added scoring/tokenization helpers for candidate matching.
3. Expanded ScoutState with:
	- target_player_name
	- cfbd_athlete_id
	- delegator_plan
	- worker summary fields
	- trace_log
4. Added tool wrappers for delegated planning, identity resolution, and summary generation.

### D. Graph and Agent Reshape

Status: Complete

1. Added lead_delegator and worker node implementations.
2. Rewired graph to explicit fan-out/fan-in topology:
	- lead_delegator -> cfbd_analyst, recruiting_scout, team_scout
	- workers -> lead_synthesizer
3. Added trace helper for route observability.
4. Kept compatibility aliases for previous node naming.

### E. UI and Service Separation

Status: Complete

1. Added engine/orchestration_service.py with:
	- orchestrate_structured_report
	- orchestrate_chat_turn
2. Updated app page handlers to call orchestration service instead of inline orchestration logic.
3. Exported orchestration entrypoints from engine package boundary.

## In-Progress Milestones

### B. Schema and Notebook Alignment

Status: In Progress

1. Notebook changes reflect split recruit/college/bridge model and payload generation.
2. Migration SQL and lookup examples require final policy pass for explicit pg_trgm/search_text alignment.
3. End-to-end notebook execution against live Supabase writes is still pending.

### F. Validation and Cleanup

Status: In Progress

1. Runtime smoke validation after graph/app rewiring is partially complete.
2. Unexpected generated artifact cleanup is pending repository decision.

## Current Update Snapshot (2026-04-08)

Status: Significant implementation completed since prior baseline.

1. Structured Report now runs a lightweight multi-agent web scout path (recruiting + team) while preserving deterministic report data sections.
2. Open Chat remains full multi-agent orchestration with delegator and worker fan-out.
3. Structured Report remains CFBD-excluded by execution scope.
4. Local-only CFBD debugger page added for delegator/identity/endpoint diagnostics.
5. Security hardening expanded:
	- stricter delegator schema validation and failure handling
	- model output sanitization guards
	- prompt truncation/date-context wrappers
	- structured report submission rate limiting
6. Identity resolution upgraded with context-aware scoring and clarification handling.
7. Model score card rendering improved with end-user-friendly threshold probability bars and duplicate suppression logic.
8. Skill-grade fields are omitted from scouting-clean context for future recruits.

## Updated Risk and Follow-up Notes

1. Medium: Structured report and open chat flows are now intentionally divergent; regression tests should ensure this separation remains explicit.
2. Medium: Additional smoke checks are still recommended after integrating partner updates on main.
3. Medium: Controlled CSV fallback for index loading remains a future hardening task when Supabase is unavailable.

## Next Verification Checklist (Revised)

1. Structured report smoke check confirms:
	- comparables and model card render
	- recruiting/team web summaries render
	- final synthesis includes web context
	- no CFBD dependency in report route
2. Open chat smoke check confirms full orchestrated path and trace output.
3. Identity ambiguity scenario confirms candidate clarification and recovery behavior.
4. Model score card confirms no duplicate threshold rows and no internal key leakage (for example ge80).
5. Merge-time review confirms no accidental reintroduction of skill-grade mention in final narrative.

## Current Diff Review Findings

1. High: A large generated binary artifact (engine_zip.zip) is staged as a new file and appears unrelated to runtime logic.
2. Medium: Full runtime validation (Streamlit + graph path + live data integration) is not yet closed.

## Next Verification Checklist

1. Remove or explicitly exclude generated artifact files from merge scope.
2. Run structured-report and open-chat smoke checks with trace validation.
3. Re-run targeted problem checks and confirm no regressions in updated engine/app files.

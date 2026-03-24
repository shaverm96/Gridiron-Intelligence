## Engine Restructure Implementation Log

Purpose: maintain one concise execution log instead of multiple step-by-step note files.

Related docs:
1. ENGINE_RESTRUCTURE_PLAN.md (index)
2. ENGINE_RESTRUCTURE_FULL_PLAN.md (master plan)
3. ENGINE_RESTRUCTURE_BASELINE.md (parity contract)

## Completed Milestones

### A. Baseline Guardrails

Status: Complete

1. Fixed baseline cohort captured for 2026 to 2028.
2. Structured output contract and invariants captured.
3. Validation table template created for regression runs.

### B1. Service Extraction from app.py

Status: Complete

1. Step 1: Extracted player index and player bundle retrieval logic.
2. Step 2: Extracted DuckDuckGo web search and web summarization logic.
3. Step 3: Extracted vector insight retrieval logic.
4. Step 4: Extracted historical comparables retrieval logic.
5. Step 5: Extracted final prompt assembly and final synthesis helper logic.
6. Step 6: Extracted scouting/profile normalization and score card rendering helpers.
7. Step 7: Added this milestone log and executed a lightweight dry-path verification.

### B2. Domain Module Split

Status: Complete

1. Split the consolidated service layer into domain modules and rewired imports.
2. Kept structured_report_services.py as compatibility exports.
3. Validated syntax/import integrity after split.

### Modules in Current Boundary

- engine/data_access.py
- engine/data_transforms.py
- engine/web_research_service.py
- engine/vector_service.py
- engine/comparables_service.py
- engine/synthesis_service.py
- engine/structured_report_services.py (compatibility layer)

### Validation Notes

1. Dry-path check passed with deterministic fallback behavior.
2. Compile checks passed for app and split engine modules.

### Refactor Pattern

1. app.py retains public helper function names used by the UI flow.
2. app.py helper functions now delegate to service functions.
3. External dependencies are injected into service helpers where needed.
4. This keeps behavior stable while reducing app.py coupling.

### Next Step

1. Restructure delivery plan portions A through G are complete.
2. Next optional iteration: add automated tests for athlete_id normalization and persona-prompt branching.

## In-Progress Milestones

### C1. Structured Sorting Update

Status: Complete

1. Updated player index ordering to class-year partition with rating descending and name ascending tie-break.
2. Added null-safe rating handling to avoid UI breaks when rating is missing or malformed.
3. Preserved existing player label formatting and structured flow behavior.
4. Validated baseline cohort presence and per-year monotonic ordering for 2026 to 2028.
5. Confirmed structured section order and controls remain unchanged in app flow.

Validation evidence:
1. `COHORT_PRESENT True`
2. `YEAR_SORT_CHECK` returned `True` for each year bucket (2026, 2027, 2028).

### D1. Structured Graph/State Web Split

Status: Complete

1. Added explicit structured state fields for `player_web_summary` and `team_web_summary`.
2. Split structured web routing into deterministic sequence: `sql_analyst` -> `player_web_scout` -> `team_web_scout` -> `vector_analyst` -> `comparables` -> `synthesizer`.
3. Preserved chat-mode web path using existing `web_scout` node.
4. Kept backward-compatible `web_research_context` as a synthesized aggregate of player/team summaries.
5. Added explicit structured-route safety guard to block chat-only nodes (`web_scout`, `chat_followup`) in structured mode.

Validation evidence:
1. Structured supervisor smoke check returned route order:
	`STEP1 sql_analyst`, `STEP2 player_web_scout`, `STEP3 team_web_scout`, `STEP4 vector_analyst`, `STEP5 comparables`, `STEP6 synthesizer`.
2. Graph smoke invocation returned `FINAL_NEXT end`, `HAS_WEB True`, `REPORT_SET True`.
3. Route guard is enforced in both conditional routing and fallback runner for structured mode.

### E1. Open Chat Page + Session Memory Isolation

Status: Complete

1. Added app workspace routing in sidebar with separate pages: `Structured Report` and `Open Chat`.
2. Kept structured report flow encapsulated in its own renderer with existing output contract preserved.
3. Implemented dedicated open chat renderer backed by engine graph invocation (`mode=chat`).
4. Added session-scoped open chat memory keys (`open_chat_messages`, `open_chat_agent_state`) isolated from structured page state.
5. Added explicit chat reset control (`Clear Chat`) that wipes only open chat session memory.

Validation evidence:
1. `app.py` diagnostics returned no errors after page split.
2. `python -m py_compile app.py` completed successfully.

### F1. Persona Layer + Navigation Stability

Status: Complete

1. Added shared sidebar persona selector with constrained options: `Scout` and `Fan`.
2. Threaded persona into structured prompt assembly path (app -> synthesis service).
3. Threaded persona into graph chat synthesis path (state -> agents -> tools prompt builder).
4. Preserved workspace routing stability across `Structured Report` and `Open Chat` while keeping diagnostics centralized in sidebar.
5. Kept open chat memory session-scoped and isolated, with persona synchronized into chat state on reset and each invocation.

Validation evidence:
1. Structured output metadata now surfaces selected persona.
2. Graph synthesis prompt path now receives persona from state.

### G1. athlete_id + Validation + Handoff

Status: Complete

1. Added athlete_id normalization utility in engine data access layer with rule: retain positive integer only; otherwise null.
2. Applied athlete_id normalization to player index loading and player bundle retrieval path.
3. Exposed athlete_id in normalized player profile view payload.
4. Updated Supabase upsert notebook schema DDL to include `athlete_id bigint` on `gi_player_master` plus index.
5. Updated notebook id-standardization and payload cells to normalize athlete_id and include it in player master upserts.
6. Updated notebook SQL join example to surface athlete_id in retrieval patterns.
7. Corrected structured output metadata formatting artifact in app summary line.

Validation evidence:
1. `ATHLETE_ID_POSITIVE_OR_NULL True` from runtime data-path smoke check.
2. `ATHLETE_ID_SAMPLE [None, None, None, None, None, 12, 12]` confirms positive-only normalization behavior.
3. Baseline parity rerun passed: `COHORT_PRESENT True` and `YEAR_SORT_CHECK` true for 2026/2027/2028.
4. Compile checks passed for updated files: `app.py`, `engine/data_access.py`, `engine/data_transforms.py`.

Known limitations:
1. Notebook cells were updated but not fully re-executed end-to-end against a live Supabase write in this run.
2. Current runtime flow still keys retrieval primarily on recruit_id; athlete_id is added for persistence/interoperability and profile visibility.

## Next Verification and Integration Plan

Purpose: provide an execution-ready checklist to verify end-to-end behavior after portions A through G, with special focus on open chat reliability, athlete_id/CFBD identity flow, and agent observability.

### V1. Open Chat End-to-End Review

Status: Planned

1. Validate routing behavior in chat mode for representative prompts:
	- "latest news on <player>" should route to web path.
	- "compare <player> to historical comps" should route to comparables path.
	- "full scouting report on <player>" should attempt SQL path.
2. Confirm session memory isolation still holds after repeated chat resets and persona switches.
3. Verify error surfaces are user-readable when required IDs are missing.
4. Record per-prompt expected vs observed node path and response quality.

Acceptance evidence:
1. Prompt matrix with route decisions and response outcomes.
2. Confirmation that open chat state remains session-scoped and independent from structured flow.

### V2. athlete_id and CFBD Linkage Hardening

Status: Planned

1. Add deterministic mapping flow from 247 recruit identity to CFBD athlete_id.
2. Extend runtime retrieval policy to prefer athlete_id for CFBD-backed pulls, with recruit_id fallback only when athlete_id is unavailable.
3. Explicitly instruct synthesis/agent prompt path that CFBD retrieval should use athlete_id whenever present.
4. Backfill athlete_id coverage metrics for active cohort and flag unresolved identities.

Acceptance evidence:
1. Sample validation table showing recruit_id -> athlete_id mapping quality and coverage.
2. Runtime logs showing athlete_id-first retrieval behavior on CFBD paths.

### V3. Agent Interaction Debug Mode

Status: Planned

1. Add UI debug toggle in sidebar for graph tracing.
2. Capture per-turn node execution trace with:
	- entered node sequence
	- next_step decisions
	- key identity fields present (recruit_id, athlete_id)
	- error list and citations count
3. Render trace in open chat as collapsible diagnostics block.
4. Keep debug mode off by default to avoid noisy UI.

Acceptance evidence:
1. For each prompt, trace shows full node chain and final route rationale.
2. Debug output supports fast diagnosis when chat path skips SQL/comparables due to missing IDs.

### V4. Schema and Identity Model Improvements

Status: Planned

1. Propose an identity map table to centralize recruit_id, athlete_id, sports_ref_id, source, confidence, and verification timestamps.
2. Add/confirm indexes for hot retrieval paths (athlete_id and recruit_id joins).
3. Decide whether to include athlete_id in optional news ingestion entities for future multi-source joins.
4. Document migration order and rollback notes for schema changes.

Acceptance evidence:
1. Schema recommendation document with DDL candidates and migration sequencing.
2. Join-path review showing reduced ambiguity in runtime identity resolution.

### Suggested Execution Order

1. Execute V1 first (open chat reliability and route quality).
2. Execute V3 second (debug trace visibility to accelerate V2).
3. Execute V2 third (athlete_id-first CFBD retrieval contract).
4. Execute V4 last (schema refinements and migration planning).

### Immediate Notes from Current Code Review

1. Open chat runtime is functioning and persona-aware, but SQL/comparables routes currently depend on recruit_id being present in state.
2. athlete_id is now normalized and persisted in player paths, but CFBD retrieval is not yet explicitly athlete_id-first in runtime orchestration.
3. A debug trace mode is not yet implemented, so agent route verification is currently manual.

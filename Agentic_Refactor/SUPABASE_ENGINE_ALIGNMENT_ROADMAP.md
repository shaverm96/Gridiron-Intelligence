# Supabase DB Update + Agentic App Alignment Roadmap

Audience: Project partner handoff and implementation guide
Last updated: 2026-04-08

## 1) Why this document exists

This document explains:
1. What changed in the Supabase database construction pipeline.
2. Why recruit and college entities are now separated.
3. How the linkage table connects both entities.
4. What must change in the Streamlit app and engine wrappers to stay schema-aligned.
5. A step-by-step roadmap to support both:
   - Structured Scouting Report page
   - Multi-agent Open Chat page

This is intended to be implementation-ready, not just conceptual.

## 2) Current Supabase schema direction (source of truth)

Primary source file:
- notebooks/supabase_upserts_pipeline.ipynb

Primary operational tables:
1. gi_recruit_master
2. gi_college_master
3. gi_player_link_bridge
4. gi_scouting_report_features
5. gi_model_prediction_score
6. gi_model_prediction_thresholds
7. gi_factoid_vectors
8. gi_news_chunks (optional workflow extension)
9. gi_news_summaries (optional workflow extension)

## 3) What was updated in DB construction

### 3.1 Recruit and college split is now explicit

Instead of treating all player data as one logical object, the pipeline now models:
1. Recruit identity and recruiting profile in gi_recruit_master.
2. College career identity/history in gi_college_master.
3. Crosswalk logic in gi_player_link_bridge.

Why this matters:
- Reduces identity ambiguity between HS recruit records and later college records.
- Supports future cases where one side exists without the other.
- Makes multi-agent resolution easier (identity-first retrieval).

### 3.2 Filled recruiting summary is prioritized for ID enrichment

The upsert notebook now prefers:
- recruiting_summaries_2015_2028_athlete_id_filled.csv
with fallback to:
- recruiting_summaries_2015_2028.csv

Impact:
- Better athlete_id and cfbd_athlete_id coverage in recruit payloads.
- Stronger linkage across recruit/college/CFBD.

### 3.3 Height/weight handling is now explicit and source-controlled

Recruit payload enrichment uses master recruits as the raw source of truth:
1. height_raw, weight_raw from master recruits text fields.
2. height_inches parsed from height_raw.
3. weight_lbs parsed from weight_raw.

Impact:
- Preserves original values and normalized numeric values.
- Improves feature consistency for scoring/comparables.

## 4) Recruit, college, and linkage model (how it should be understood)

### 4.1 Entity boundaries

Recruit entity (gi_recruit_master) should answer:
- Who was this player as a recruit?
- What was the recruiting context and projection?

College entity (gi_college_master) should answer:
- Who is this player in college records?
- What team/season context exists on the college side?

Bridge entity (gi_player_link_bridge) should answer:
- What evidence links recruit_id to college_player_id and/or cfbd_athlete_id?
- How confident is linkage and by which method?

### 4.2 Expected key behavior

1. recruit_id remains the app-level anchor for structured scouting workflows.
2. cfbd_athlete_id is the cross-system identity anchor.
3. athlete_id remains as backward-compatible mirror when needed.
4. bridge table is the canonical join path from recruit to college player object.

## 5) Where app and engine currently stand

### 5.1 Good alignment already in place

1. Engine config table map already points to:
   - gi_recruit_master
   - gi_college_master
   - gi_player_link_bridge
2. Identity resolution helpers already exist in engine/supabase_client.py:
   - resolve_player_identity
   - fetch_player_bundle_by_identity
   - search_text-based fuzzy lookup should stay SQL-first with pg_trgm-aware matching, not embeddings.
3. Agent graph supports structured report and chat orchestration through shared state.

### 5.2 Current drift / risk areas

1. app.py player index is Supabase-backed; current risk is lack of a controlled CSV fallback path when Supabase is unavailable.
2. app.py TABLES mapping is partial and still named around player_master conventions.
3. Data mapping wrappers in app.py (first_non_null, build_player_profile_view wrapper) are effectively schema-map functions but are not centralized as a formal schema contract module.
4. Structured report page profile rendering currently favors recruit-side payload only; college and bridge context are not surfaced in a first-class way.
5. Open chat uses agent orchestration, but there is no explicit prompt-level contract guaranteeing consistent recruit-vs-college identity framing in every turn.

### 5.3 Current app state snapshot (2026-04-08)

1. Structured Report currently combines deterministic data retrieval (bundle/model card/comparables/vector context) with a lightweight multi-agent web scout pipeline (recruiting + team summaries).
2. Structured Report remains CFBD-excluded by scope and route design.
3. Open Chat remains full multi-agent orchestration (delegator + workers + synthesizer) with clarification-aware identity flow.
4. Local-only CFBD debugger exists as a separate route for endpoint and identity diagnostics.
5. Prompt/model hardening is now active in multiple layers (sanitization, date context, payload truncation, validation handling).
6. Model score UX is now end-user oriented (friendly threshold wording and probability bars).

## 6) Clarifying the schema-map concept for this codebase

Even though there is no file literally named schema_map.py yet, the schema-map behavior is currently spread across:
1. engine/data_access.py
2. engine/data_transforms.py
3. app.py wrappers around those transforms

Key mapping behavior today:
- Column fallback selection (first_non_null)
- Scouting JSON normalization and merge logic
- Profile projection from raw row fields

Recommendation:
- Create a dedicated schema contract module (for example: engine/schema_contract.py) with canonical field maps and fallback lists for recruit, college, bridge, and UI view models.

## 7) Implementation roadmap (app + engine + workflow)

## Phase 1: Lock DB contract and data quality gates

Objective:
- Freeze the operational schema contract and validate upsert quality.

Tasks:
1. Re-run upsert notebook in DRY_RUN mode with current pipeline and capture diagnostics.
2. Confirm non-null rates for:
   - recruit_id
   - cfbd_recruiting_id
   - athlete_id
   - cfbd_athlete_id
   - sports_ref_id
   - height_raw
   - weight_raw
   - height_inches
   - weight_lbs
3. Confirm bridge joinability rates:
   - percent recruit rows with any bridge row
   - percent bridge rows resolving to college_player_id
4. Store a dated validation snapshot in Agentic_Refactor.

Deliverable:
- DB contract snapshot + validation markdown with row counts and null-rate table.

## Phase 2: Centralize schema-map logic

Objective:
- Eliminate mapping drift between app and engine.

Tasks:
1. Add engine/schema_contract.py containing:
   - canonical table names
   - canonical output shape for recruit profile
   - canonical output shape for college profile
   - bridge resolution precedence
2. Move first_non_null and profile fallback lists into schema_contract module.
3. Update engine/data_transforms.py to consume schema_contract constants.
4. Update app.py wrappers to consume engine schema contract instead of local ad-hoc fallback definitions.

Deliverable:
- One source of truth for schema maps used by both report and chat pathways.

## Phase 3: Update Structured Report page to new DB model

Objective:
- Make report page explicitly recruit+college aware and bridge-aware.

Tasks:
1. Keep Supabase-backed player index as primary path and add explicit controlled CSV fallback behavior for degraded mode.
2. In report generation, show:
   - recruit profile section
   - linked college profile section (if bridge resolution exists)
   - identity confidence/trace section
3. Surface bridge metadata where available:
   - match_method
   - match_confidence
4. Ensure report text generator prompt receives both recruit and college context consistently.

Deliverable:
- Structured report page fully powered by current Supabase schema contract.

## Phase 4: Multi-agent open chat hardening

Objective:
- Ensure open chat is truly multi-agent and identity-consistent.

Tasks:
1. Add explicit routing checks in delegator for chat intents:
   - identity lookup
   - report-style synthesis
   - web update context
   - team-fit context
2. Add trace completeness checks so each turn logs:
   - route chosen
   - identity method used
   - workers executed
3. Ensure chat state stores resolved recruit_id and cfbd_athlete_id when discovered and reuses them in follow-up turns.
4. Add failure-safe behavior when identity cannot be resolved (ask targeted clarification).

Deliverable:
- Robust open chat flow that stays anchored to the same identity object across turns.

## Phase 5: End-to-end QA and partner handoff

Objective:
- Validate parity between report and chat paths.

Tasks:
1. Build a shared test matrix with at least:
   - known recruit_id case
   - name-only identity case
   - no-match case
   - bridge-only case
2. Validate both pages produce consistent identity and profile context.
3. Add quick regression checklist to Agentic_Refactor/ENGINE_RESTRUCTURE_PROGRESS.md.

Deliverable:
- Signed-off QA checklist and stable handoff state.

## 8) Priority implementation order (recommended next sprint)

1. Phase 1 validation snapshot (fast, high confidence gain).
2. Phase 2 schema-contract centralization (prevents rework).
3. Phase 3 structured report migration to DB-backed index.
4. Phase 4 open chat hardening with identity persistence.
5. Phase 5 QA closeout.

Execution note:
1. Items from Phase 4 have partially advanced (identity clarification and orchestrator hardening); remaining work should prioritize schema-contract centralization and recruit/college/bridge presentation parity in Structured Report.

## 9) Concrete file touch list for upcoming work

High-priority edits expected in:
1. notebooks/supabase_upserts_pipeline.ipynb
2. app.py
3. engine/config.py
4. engine/supabase_client.py
5. engine/data_access.py
6. engine/data_transforms.py
7. engine/agents.py
8. engine/orchestration_service.py
9. Agentic_Refactor/ENGINE_RESTRUCTURE_PROGRESS.md

New file recommended:
- engine/schema_contract.py

## 10) Risk register and mitigations

Risk 1: Hidden field drift between app and engine.
- Mitigation: central schema contract module + one integration test per page.

Risk 2: Identity mismatch between recruit and college contexts in chat.
- Mitigation: persist resolved IDs in chat state and require trace logs per turn.

Risk 3: Report page still tied to local CSV assumptions.
- Mitigation: DB-first index loader with explicit fallback and warning banner.

Risk 4: Upsert appears successful but key fields degrade silently.
- Mitigation: required diagnostics block and null-rate threshold checks before live upsert.

## 11) Definition of done

This migration is done when all statements below are true:
1. Structured Report page reads from current Supabase recruit/college/bridge model.
2. Open Chat uses same identity resolution contract and persists identity across turns.
3. Schema-map behavior is centralized in engine code (not duplicated in app ad hoc logic).
4. Upsert notebook diagnostics are clean and documented.

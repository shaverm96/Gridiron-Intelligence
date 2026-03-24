## Engine Restructure Master Plan

This is the canonical detailed plan for the restructure. It consolidates previously split planning notes and aligns with current scope decisions.

## Goals

1. Preserve structured dropdown scouting flow.
2. Sort players by recruiting rating within selected year.
3. Keep target team context explicit.
4. Include player and team web context in structured reports.
5. Keep CFBD out of structured report path.
6. Provide a separate open chat page with tools and session memory.
7. Introduce Fan and Scout personas across both pages.
8. Add athlete_id to player master paths with positive-only normalization.

## Scope

Included:
1. UI split to structured page plus open chat page.
2. Engine/state/graph updates for clear mode-specific routing.
3. Domain module split for structured report services.
4. Validation and parity checks against baseline cohort.

Excluded:
1. Cross-session persistent memory.
2. CFBD use in structured mode.
3. Large pipeline overhauls unrelated to athlete_id and current retrieval needs.

## Architecture Direction

1. Keep app.py as orchestration shell and UI wrappers.
2. Maintain domain service modules under engine:
   - data_access.py
   - data_transforms.py
   - web_research_service.py
   - vector_service.py
   - comparables_service.py
   - synthesis_service.py
3. Keep structured_report_services.py as a temporary compatibility export layer.
4. Move toward graph-routed structured and chat modes with isolated state.

## Delivery Plan

### Portion A: Baseline and Invariants

Status: Complete

Deliverables:
1. Cohort and parity contract in ENGINE_RESTRUCTURE_BASELINE.md.
2. Invariant checklist for structured behavior and error handling.

### Portion B1: Service Extraction from app.py

Status: Complete

Deliverables:
1. Service wrappers in app.py with preserved signatures.
2. Extracted logic no longer embedded in UI blocks.

### Portion B2: Domain Module Split

Status: Complete

Deliverables:
1. Domain-focused engine modules created.
2. app.py imports rewired to domain modules.
3. Compatibility exports retained for transition safety.

### Portion C: Structured Scouting Page Update

Status: Complete

Tasks:
1. Implement explicit sorting by rating descending within selected year.
2. Keep tie-break by player name ascending.
3. Preserve section order and rendering contract from baseline.

### Portion D: Structured Graph and State Updates

Status: Complete

Tasks:
1. Add explicit structured-mode state fields for player/team web summaries.
2. Remove CFBD dependency from structured synthesis path.
3. Ensure deterministic fan-out/fan-in routing.

### Portion E: Open Chat Page

Status: Complete

Tasks:
1. Add dedicated open chat UI route/page.
2. Ensure tool access parity with current chat capabilities.
3. Keep memory session-scoped and isolated from structured mode state.

### Portion F: Persona Layer and Navigation Stability

Status: Complete

Tasks:
1. Add Fan and Scout persona selector usable in both pages.
2. Thread persona through synthesis prompts and chat prompts.
3. Finalize shared diagnostics and cold-start behavior.

### Portion G: athlete_id + Validation + Handoff

Status: Complete

Tasks:
1. Add athlete_id in Supabase schema/path updates.
2. Normalize athlete_id: retain positive integer only, otherwise null.
3. Re-run baseline parity checks and smoke validation.
4. Publish concise handoff notes and known limitations.

## Dependencies

1. A -> B1 -> B2 -> C
2. B2 -> D -> E
3. C + E -> F
4. F + athlete_id updates -> G

## Acceptance Criteria

1. Structured flow remains stable and deterministic for baseline inputs.
2. Player dropdown ordering follows rating desc within year.
3. Structured output includes required sections and deterministic fallbacks.
4. CFBD is absent from structured path and available only in open chat path for this phase.
5. Persona selection (Fan/Scout) is available in both pages.
6. athlete_id normalization rule is consistently applied.

## Risks and Mitigations

1. Risk: import churn during modularization.
   Mitigation: compatibility export layer and compile checks after each split.
2. Risk: state leakage between structured and chat paths.
   Mitigation: separate session keys and mode-specific initialization.
3. Risk: external API variability affecting summaries.
   Mitigation: deterministic fallback messaging and retry limits.

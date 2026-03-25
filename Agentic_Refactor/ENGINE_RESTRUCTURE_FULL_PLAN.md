## Engine Restructure Master Plan (Phase 2)

This is the canonical detailed plan for the current architecture shift.

## Goals

1. Move to explicit recruit and college identity separation with bridge linking.
2. Replace supervisor-loop behavior with explicit fan-out/fan-in graph execution.
3. Enforce SQL-first identity resolution (search_text + pg_trgm), no embedding identity match.
4. Keep shared graph state summary-first and compact.
5. Keep Streamlit UI layer thin and delegate orchestration to engine services.
6. Preserve existing structured report and open chat usability while changing internals.

## Scope

Included:
1. Schema/notebook alignment for split identity tables.
2. Engine state, tools, graph, and agent rewiring to the new architecture.
3. App to service-layer orchestration extraction.
4. Validation and diff hygiene before rollout.

Excluded:
1. Cross-session memory redesign.
2. Embedding-based identity retrieval.
3. Broad product UI redesign beyond architecture-supporting updates.

## Target Architecture

1. Data model:
   - gi_recruit_master
   - gi_college_master
   - gi_player_link_bridge
2. Graph model:
   - lead_delegator
   - cfbd_analyst, recruiting_scout, team_scout (parallel workers)
   - lead_synthesizer (fan-in)
3. State model:
   - explicit delegator_plan
   - summary fields per worker
   - trace_log for observability
4. UI model:
   - app.py for rendering and controls
   - engine/orchestration_service.py for report/chat orchestration

## Delivery Plan

### Portion A: Diff Hygiene and Safety

Status: Complete

Deliverables:
1. Diff sweep completed before continuing implementation.
2. High-risk unexpected artifacts identified and recorded.

### Portion B: Schema and Notebook Alignment

Status: In Progress

Tasks:
1. Align notebook DDL and payload steps to recruit/college/bridge tables.
2. Confirm SQL fuzzy identity strategy includes pg_trgm-ready search surface.
3. Validate migration safety and backward-compatible table key usage where needed.

### Portion C: Engine Identity and State Contract

Status: Complete

Tasks:
1. Add identity helpers and bridge-aware retrieval path in engine Supabase client.
2. Expand ScoutState with delegator plan, summary buckets, and trace log.
3. Add tool wrappers for identity resolution, delegated planning, and summary condensation.

### Portion D: Graph and Agent Reshape

Status: Complete

Tasks:
1. Implement lead_delegator and three worker nodes.
2. Rewire graph to explicit fan-out/fan-in topology.
3. Keep compatibility aliases where practical to reduce transition break risk.

### Portion E: UI and Service Separation

Status: Complete

Tasks:
1. Add orchestration service entrypoints for structured report and open chat turns.
2. Update app page handlers to call orchestration layer instead of inline orchestration logic.
3. Keep UI output behavior stable while changing backend flow.

### Portion F: Validation and Handoff

Status: In Progress

Tasks:
1. Resolve or intentionally exclude unrelated artifact diffs before handoff.
2. Run runtime checks for Streamlit, graph invocation, and identity fallback behavior.
3. Confirm parity expectations and publish final migration notes.

## Dependencies

1. A -> B
2. B -> C
3. C -> D
4. D -> E
5. B + C + D + E -> F

## Acceptance Criteria

1. Graph invokes in delegator + parallel worker + synthesizer sequence.
2. Identity resolution path supports recruit_id, cfbd_athlete_id, and bridge-assisted lookup.
3. Shared state stores summaries, not raw web/sql payload blobs.
4. app.py pages run through orchestration service entrypoints.
5. Notebook/schema layer reflects split identity architecture and SQL fuzzy match policy.

## Current Risks

1. A large generated artifact file was added to version control accidentally (engine_zip.zip).
   Mitigation: remove from commit scope unless explicitly required.
2. Notebook lookup examples currently show LIKE-based search examples and may not fully enforce pg_trgm strategy.
   Mitigation: add explicit pg_trgm/search_text query examples and index notes.
3. Runtime validation is not fully closed out after graph/app rewiring.
   Mitigation: run focused smoke checks before merge.

## Plan: Engine Restructure with Structured Scouting + Open Chat

This document provides a highly detailed implementation plan split into manageable portions for execution and handoff. It preserves existing structured scouting behavior, adds requested sorting logic, removes CFBD from structured reports, and introduces a separate open-ended chat experience with session memory.

## Objectives

1. Preserve current structured scouting workflow with dropdown controls.
2. Sort player dropdown options by recruiting rating within each year.
3. Keep target team dropdown on structured report page.
4. Include player web context and school/team web context in structured report.
5. Exclude CFBD context from structured report path.
6. Add separate page for open-ended chat with all existing tools and session memory.
7. Maintain existing secrets/config behavior and reliability diagnostics.

## Scope Boundaries

### Included

1. UI restructuring into two clear pages/modes.
2. Engine graph/state updates for structured and chat paths.
3. Structured path fan-out/fan-in around web summaries.
4. Session-only memory continuity for open chat.
5. Regression-safe migration that reuses existing logic where possible.

### Excluded

1. Persistent cross-session memory storage.
2. CFBD tools and data on the structured scouting page.
3. Large database schema changes unrelated to this feature.
4. Rebuilding data pipelines beyond fields already used for sorting and context.

## Current-State Baseline

1. Existing app is currently a single-page flow.
2. Existing controls already include year, player, and target team dropdowns.
3. Existing player ordering is year then player name.
4. Existing engine uses supervisor-driven node routing and synthesis.
5. Existing chat memory is session-scoped in state conversation history.

## Manageable Portions (Execution Chunks)

## Portion A: Safety Baseline and Refactor Guardrails

### Goal

Create a safe migration path before changing behavior.

### Tasks

1. Snapshot current report output behavior using two representative recruits per year.
2. Capture expected output sections and minimal quality checks.
3. Add internal notes for key assumptions and invariants:
4. Structured page must still run from dropdown selection.
5. Target team control remains independent from player selection.
6. Synthesis must still include SQL baseline and vector/comparables context.

### Deliverable

1. Baseline checklist for parity validation after each subsequent portion.

### Dependencies

1. None.

### Risks

1. Hidden behavior regressions if baseline expectations are incomplete.

### Mitigation

1. Use a small fixed set of known recruits and store expected output headings and minimum content checks.

## Portion B: Service Layer Extraction from Monolith UI

### Goal

Decouple business logic from single-page UI so two-page navigation can be introduced safely.

### Tasks

1. Extract reusable functions from app-level flow into a service-oriented module boundary.
2. Preserve signatures for:
3. Player index load and normalization.
4. Player bundle retrieval from Supabase.
5. Web search and web summarization calls.
6. Vector retrieval and historical comparables retrieval.
7. Final synthesis prompt construction and model call.
8. Keep current error semantics and user-facing diagnostics unchanged.
9. Keep current secrets resolution precedence unchanged.

### Deliverable

1. Refactored callable services with no functional behavior changes.

### Dependencies

1. Portion A baseline complete.

### Risks

1. Tight coupling to UI session state causing subtle runtime issues.

### Mitigation

1. Use thin wrappers that preserve old function contracts during transition.

## Portion C: Structured Scouting Page Rebuild

### Goal

Implement dedicated structured scouting page while preserving current experience and adding requested sorting.

### Tasks

1. Implement page shell with controls:
2. Recruiting year dropdown.
3. Player dropdown filtered to selected year.
4. Target team dropdown.
5. Generate scouting report action.
6. Update dropdown sort logic:
7. Primary sort by year partition.
8. Within selected year sort by recruiting rating descending.
9. Tie-break by player name ascending.
10. Null-safe behavior for missing ratings using nulls last.
11. Keep player label formatting readable and deterministic.
12. Preserve report rendering sections and order.
13. Preserve loading spinners and failure messages.

### Deliverable

1. Structured scouting page with required controls and rating-based ordering.

### Dependencies

1. Portion B extraction complete.

### Risks

1. Sorting confusion if rating scale not visible.

### Mitigation

1. Optionally display compact rating in player labels during final polish.

## Portion D: Engine State and Graph Updates for Structured Mode

### Goal

Align orchestration to explicit structured-mode fan-out/fan-in around web contexts while removing CFBD from structured path.

### Tasks

1. Update state schema to include explicit fields:
2. Player web summary.
3. Team web summary.
4. Structured mode metadata for traceability and prompt assembly.
5. Remove CFBD field dependency from structured synthesis path.
6. Update graph routes:
7. Router identifies structured mode path.
8. Fan-out to recruiting web scout and team web scout.
9. Rejoin before synthesis.
10. Keep existing SQL, vector, and comparables participation as configured.
11. Ensure summary model assignment remains lightweight for web nodes.
12. Ensure final synthesis node uses heavyweight final model.

### Deliverable

1. Structured orchestration path with separate player/team web summaries and no CFBD requirement.

### Dependencies

1. Portions B and C complete.

### Risks

1. Supervisor route loops if next-step logic not updated consistently.

### Mitigation

1. Add explicit state markers for completed nodes and deterministic router branching.

## Portion E: Dedicated Open-Ended Chat Page with Session Memory

### Goal

Provide separate chat page with all tools and memory continuity in current session.

### Tasks

1. Implement standalone chat page UI with message history display and user input.
2. Route chat prompts through chat-capable graph path.
3. Ensure tool access remains available:
4. SQL retrieval tools.
5. Web search and summarization tools.
6. Vector factoid retrieval tools.
7. Comparables context tools.
8. Keep session-only memory behavior using conversation history in state.
9. Ensure follow-up turns include prior conversation context.
10. Keep chat state isolated from structured page control state.

### Deliverable

1. Open-ended chat page with working tools and session memory continuity.

### Dependencies

1. Portion D graph update complete.

### Risks

1. State leakage between pages leading to incorrect context mixing.

### Mitigation

1. Use separate session keys and clear mode-specific state initialization.

## Portion F: Navigation, Integration, and UX Stability

### Goal

Create reliable page navigation and shared runtime shell.

### Tasks

1. Add page navigation strategy:
2. Multipage architecture preferred for stronger isolation.
3. Shared sidebar for diagnostics and configuration indicators.
4. Keep existing environment checks and one-click diagnostic available.
5. Confirm both pages can initialize independently from cold start.

### Deliverable

1. Stable two-page app with shared diagnostics and isolated mode state.

### Dependencies

1. Portions C and E complete.

### Risks

1. Duplicate initialization causing slower startup.

### Mitigation

1. Use caching and lightweight lazy init for mode-specific resources.

## Portion G: Verification, Hardening, and Handoff

### Goal

Ensure production confidence with explicit acceptance checks.

### Tasks

1. Structured page acceptance tests:
2. Verify player ordering by rating desc within selected year.
3. Verify team dropdown still controls team context in report.
4. Verify reports include player and team web context.
5. Verify reports exclude CFBD content path.
6. Chat page acceptance tests:
7. Verify multi-turn continuity in same session.
8. Verify tool invocation pathways remain operational.
9. Reliability tests:
10. Verify missing credential behavior remains fail-fast and clear.
11. Verify Supabase fetch failure path is handled without app crash.
12. Run available lint or test commands and perform manual UI smoke checks.
13. Capture known limitations and next-step improvements.

### Deliverable

1. Verified release candidate and handoff notes.

### Dependencies

1. Portions A through F complete.

### Risks

1. Runtime issues discovered late due to external API variability.

### Mitigation

1. Add deterministic fallback messages and retry constraints for noncritical web steps.

## Dependency Map

1. A -> B -> C
2. B -> D
3. D -> E
4. C + E -> F
5. A + B + C + D + E + F -> G

## Parallelization Opportunities

1. During Portion B, extraction of data-loading utilities and rendering utilities can proceed in parallel.
2. During Portion D, state schema updates and synthesis prompt updates can proceed in parallel if interfaces are frozen first.
3. During Portion G, structured and chat manual verification can run in parallel by two reviewers.

## Acceptance Criteria

1. Structured page exists and independently generates reports from dropdown selection.
2. Player dropdown is sorted by rating descending within selected year.
3. Team dropdown remains present and controls team context.
4. Structured report uses player web and team web context and does not depend on CFBD.
5. Open-ended chat page exists with tool access and session-only memory continuity.
6. Existing config and diagnostics behavior remains functional.

## Suggested PR Slicing

## PR 1

1. Portion A and Portion B.
2. No visible behavior change expected.

## PR 2

1. Portion C and Portion F structured subset.
2. Introduces structured page with sorting changes.

## PR 3

1. Portion D.
2. Structured engine fan-out/fan-in and CFBD removal for structured path.

## PR 4

1. Portion E and remaining Portion F.
2. Adds open-ended chat page and isolated session memory behavior.

## PR 5

1. Portion G.
2. Hardening, verification artifacts, and final documentation.

## Post-Implementation Follow-Ups

1. Optional UX enhancement: show rating in dropdown labels for transparency.
2. Optional analytics: instrument latency per node to validate fan-out gains.
3. Optional persistence roadmap: add durable memory layer only if product requires cross-session continuity.

## Notes for Memory Recall

1. Core user intent preserved:
2. Keep dropdown scouting path.
3. Sort players by rating within year.
4. Keep team dropdown.
5. Use player and team web context in report.
6. Skip CFBD for structured report.
7. Add separate open-ended chat with session memory.
8. Design principle:
9. Separate structured deterministic flow from exploratory chat flow.
10. Keep orchestration modular so both pages share tools without sharing fragile UI state.

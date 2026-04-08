# SECURITY_UPDATES

Date: 2026-04-08
Scope: Consolidated application, orchestration, security, and structured-report updates since the last sync pull.

## Commit Context Summary

This update hardens the multi-agent scouting application for safer production behavior, restores strict separation between Structured Report and Open Chat responsibilities, introduces a simplified structured multi-agent web-scout workflow, and improves user-facing report clarity.

Core intent:
1. Keep Structured Report deterministic and CFBD-excluded.
2. Keep Open Chat fully agentic and stateful.
3. Harden model/tool prompt handling against malformed/unsafe outputs.
4. Improve identity resolution quality and clarification behavior.
5. Improve model score UX for end users (probability bars, plain threshold language).

## Major Code Updates

### 1. Structured Report and Open Chat Workflow Separation

Files:
1. app.py
2. engine/graph.py
3. engine/orchestration_service.py
4. engine/state.py
5. engine/__init__.py

What changed:
1. Structured Report now uses a lightweight structured web-scout orchestration path that runs Recruiting Scout + Team Scout and feeds summaries into synthesizer context.
2. Structured Report still preserves deterministic sections:
   - Historical Comparables
   - Projected Model Score card
   - Development information expander (temporary)
3. Open Chat remains full multi-agent flow with delegator + workers + synthesis.
4. Added graph progress callbacks and compacted chat state handling for safer, lighter turn-to-turn execution.

Why this matters:
1. Reduces scope bleed between chat and report modes.
2. Preserves report determinism while still surfacing recent web intelligence.
3. Prevents accidental CFBD re-coupling in the structured page.

### 2. Security and Prompt Hygiene Hardening

Files:
1. engine/state.py
2. engine/tools.py
3. engine/web_research_service.py
4. engine/synthesis_service.py
5. app.py

What changed:
1. DelegatorPlan schema tightened with field sanitization and whitelist validation.
2. Delegator output validation errors now explicitly halt unsafe execution paths.
3. Added summary sanitizers to strip HTML/script/code-block artifacts from model output before state write.
4. Added date-context wrappers to model prompts for recency-aware reasoning.
5. Added prompt-size controls and truncation guards to avoid oversized payload injection.
6. Added structured report submit rate limiting.
7. Added optional strict secrets mode for sensitive config sourcing.

Why this matters:
1. Limits prompt/data injection surface.
2. Improves deterministic safety behavior when LLM output is malformed.
3. Protects app stability under heavier prompts and repeated requests.

### 3. Identity Resolution and Clarification Improvements

Files:
1. engine/supabase_client.py
2. engine/agents.py
3. engine/orchestration_service.py
4. app.py
5. engine/config.py

What changed:
1. Identity scoring upgraded from simple token overlap to weighted context scoring (name + team + year + position + CFBD ID preference).
2. Added ambiguity detection and clarification prompt generation.
3. Added clarification response handling in chat turns.
4. Added normalized query sanitization for fuzzy matching input safety.

Why this matters:
1. Reduces wrong-player resolution in common duplicate-name scenarios.
2. Creates safer and clearer follow-up behavior when confidence is low.

### 4. CFBD Tooling and Local Debugger Enhancements

Files:
1. app.py
2. engine/cfbd_service.py
3. engine/tools.py
4. engine/agents.py

What changed:
1. Added local-only CFBD debugger page with stepwise delegator -> identity -> CFBD execution visibility.
2. Added endpoint expansion and parameter coverage (recruiting, player usage, season stats, roster, player search).
3. Added retry/backoff behavior and improved handling for HTML/non-JSON responses.
4. Improved endpoint selection heuristics by intent.

Why this matters:
1. Speeds local diagnostics and endpoint parity checks.
2. Reduces opaque failures from rate-limit/transient API responses.

### 5. User-Facing Report UX and Model Card Improvements

Files:
1. engine/data_transforms.py
2. engine/comparables_service.py
3. engine/synthesis_service.py
4. engine/vector_service.py
5. app.py

What changed:
1. Model card switched to dark theme and clearer visual hierarchy.
2. Predicted score now shown as /100 with a score progress bar.
3. Threshold probabilities rendered as bars with plain language labels.
4. Duplicate probability rows eliminated (for example prob vs odds columns).
5. Internal threshold key leakage (for example ge80) suppressed in prompt context.
6. Omitted unused/unhelpful lines:
   - removed Target Tier from comparables block
   - hide Threshold Band when unavailable
7. Skill-grade fields are excluded from scouting profile context and final narrative cues.
8. Vector insights now permit no-position fallback while still using position when available.

Why this matters:
1. End-user outputs are clearer and less technical.
2. Reduces confusion from internal model nomenclature.
3. Improves resilience when sparse fields are missing.

## Documentation and Dependency Updates

Files:
1. requirements.txt
2. Agentic_Refactor/ENGINE_RESTRUCTURE_PROGRESS.md
3. Agentic_Refactor/SUPABASE_ENGINE_ALIGNMENT_ROADMAP.md

What changed:
1. Added tenacity dependency for retry policies.
2. Updated restructuring docs to reflect current orchestration/app state and active risk focus.

## Reviewer Guide (for PR / agentic online review)

Focus areas:
1. Validate Structured Report remains CFBD-free while Open Chat remains full-agentic.
2. Validate delegator validation failures do not cause unsafe fallthrough behavior.
3. Validate identity clarification flow (top candidates, tie handling, response resolution).
4. Validate model-card probability extraction does not mix odds and probability fields.
5. Validate prompt sanitization/truncation does not remove critical factual context.

High-risk files:
1. app.py
2. engine/agents.py
3. engine/orchestration_service.py
4. engine/state.py
5. engine/tools.py
6. engine/data_transforms.py
7. engine/supabase_client.py

## Validation Checklist

1. Structured Report run:
   - renders comparables + score card + recruiting/team summaries + synthesis
   - no CFBD summary calls in report page route
2. Open Chat run:
   - still executes full graph path with progress and trace
3. Identity ambiguity scenario:
   - returns clarification candidates and safely resumes on selection
4. Model card scenario:
   - no duplicate threshold rows
   - no internal keys like ge80 in visible copy
5. Skill-grade omission:
   - no skill_* dependency in cleaned scouting context or final synthesis prompt narrative

## Known Follow-ups

1. Optional controlled CSV fallback for player index when Supabase is unavailable.
2. Optional explicit structured-web citations section in Structured Report.
3. Additional smoke testing pass after merge with partner changes on main.

# Engine Optimization Summary

This document captures the current state of the scouting engine optimization work, what has already been applied, and what remains worth doing. The operating assumption is still query-scoped telemetry only: report-level and chat-turn-level visibility, with no persistent session cost history.

## Current State

The system already has the main low-risk optimizations in place:
- query-level telemetry for model calls
- caching for repeated summaries and vector lookups
- parallel recruiting web execution
- transfer CFBD context reuse
- prompt-budget tightening for the largest final synthesis payloads

The current model structure is fixed and intentional:
- Flash-Lite handles summary-style calls
- Flash handles final synthesis
- the existing workflow delegates those paths correctly

## Changes Already Applied

### Telemetry and visibility
- `engine/tools.py` emits per-call telemetry for latency, prompt size, token usage, and estimated cost.
- `engine/orchestration_service.py` rolls up query-level telemetry and branch timing.
- `app.py` renders per-query telemetry for Recruiting and Transfer.
- Recruiting telemetry now includes both the Flash-Lite summary calls and the Flash final synthesis call.
- Recruiting web-scout telemetry is preserved through the graph/state path and rolled into the same per-query cost total.

### Caching and reuse
- `engine/tools.py` memoizes deterministic summary calls.
- `engine/vector_service.py` caches repeated vector-query results.
- `engine/orchestration_service.py` caches repeated Transfer CFBD context pulls.
- `engine/config.py` centralizes the cache toggles and pricing assumptions.

### Execution-path tuning
- `engine/agents.py` routes Recruiting web work through a combined parallel node.
- `engine/graph.py` wires the graph to use that node end to end.
- `engine/synthesis_service.py` trims the largest final-synthesis sections a bit earlier.

## Rough Cost Picture

The current cost estimates are directional, not benchmark-grade. They are good enough for ranking work, but not for precise budgeting.

Reference pricing currently used in telemetry:
- gemini-3.1-flash-lite-preview: about $0.25 / 1M input tokens and $1.50 / 1M output tokens
- gemini-3-flash-preview: about $0.50 / 1M input tokens and $3.00 / 1M output tokens

The existing planning numbers still look reasonable as rough order-of-magnitude estimates:
- Recruiting report: roughly 37.8k input tokens and 3.15k output tokens
- Transfer report: roughly 18k input tokens and 1.5k output tokens
- Follow-up chat turn: roughly 12k input tokens and 800 output tokens

Fixed non-model costs are still only rough planning inputs:
- Supabase Pro baseline: about $25/month
- hosting/ops: about $50/month at the current scale assumption
- CFBD and web-search costs may matter more as usage grows

## Remaining Work, Ranked by ROI

### 1. Manual benchmark pass
- Run a small manual sample set for Recruiting and Transfer.
- Record p50/p95 latency, model call count, token totals, and estimated cost per query.
- Keep q4 manual and document the results directly in this summary or a short appendix.

### 2. Improve reporting clarity
- If needed, surface the per-model breakdown more clearly in the UI.
- Keep the current Flash-Lite / Flash split unchanged.
- Do not add complexity-based model switching unless a measured gain justifies it.

### 3. Keep the high-ROI frugal work
- Preserve deterministic memoization for summaries and vector queries.
- Tighten prompt budgets further only if the manual sample shows synthesis is still the bottleneck.
- Reuse cached report artifacts where possible.
- Prefer explicit workflow reuse over hidden routing rules.

### 4. Defer low-ROI complexity
- Quantization is a luxury at this stage unless the stack moves to local model inference.
- FrugalGPT-style cascades are only worth doing if the benchmark clearly shows a repeatable gain.
- Large benchmark infrastructure is not worth building until the manual sample shows a specific bottleneck.

## Main Trade-offs

- Query-scoped telemetry keeps the product simple and actionable, but it does not create historical spend trends.
- Memoization and vector caching reduce latency and cost, but only when prompt construction and cache bounds stay stable.
- Shared pricing keeps telemetry consistent, but pricing updates still require config changes.

## Key Scaling Bottlenecks

- Final synthesis still carries the heaviest prompt payloads.
- Open chat still replays a full context stack for every turn.
- The next useful step is measurement, not more optimization theory.

## Notes

- The summary intentionally excludes persistent cost history.
- The goal is to use per-query telemetry to guide decisions, not to build a full accounting system.

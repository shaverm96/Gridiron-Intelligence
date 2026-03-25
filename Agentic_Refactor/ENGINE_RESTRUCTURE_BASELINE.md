## Engine Restructure Baseline (Phase 2)

Purpose: canonical parity and behavior contract while migrating to split identity model and fan-out/fan-in orchestration.

## Baseline Cohort

Source file:
- data/modeling_datasets/recruits/master_recruits_2015_2028.csv

Selection rule:
1. Filter years: 2026, 2027, 2028.
2. Sort by rating descending, then player name ascending.
3. Select top 2 recruits per year.
4. Canonical id fallback: recruit_id, else player_id.

Representative cohort:

| year | recruit_id | name | position | rating | high_school | committed_to |
|---|---|---|---|---:|---|---|
| 2026 | 202600001 | Jared Curtis | QB | 0.9992 | Nashville Christian | Vanderbilt |
| 2026 | 202600002 | Lamar Brown | ATH | 0.9992 | University Lab | LSU |
| 2027 | 202700001 | John Meredith III | CB | 0.9988 | North Crowley | Uncommitted |
| 2027 | 202700002 | Mark Matthews | OT | 0.9986 | St. Thomas Aquinas | Uncommitted |
| 2028 | 202800001 | Brysen Wright | WR | 0.9994 | Mandarin | Uncommitted |
| 2028 | 202800002 | Jalanie George | Edge | 0.9990 | Desert Edge | Uncommitted |

## Phase 2 Runtime Contract

Execution contract:
1. lead_delegator runs once per turn.
2. Worker fan-out runs across:
	- cfbd_analyst
	- recruiting_scout
	- team_scout
3. lead_synthesizer runs as fan-in aggregator.

Identity contract:
1. Recruit and college identities are split across dedicated master tables.
2. Bridge table carries cross-source links (recruit_id, cfbd_athlete_id, sports_ref_id, and related metadata).
3. Identity lookup is SQL fuzzy matching oriented (search_text + pg_trgm), not embeddings.

State contract:
1. Shared state keeps summary fields, not raw payload blobs.
2. trace_log is populated for route/debug visibility.
3. Structured and open chat modes remain session-isolated in UI state.

## Required Invariants

1. Structured report rendering order remains stable for existing UI sections.
2. Player sorting is rating desc within year (null-safe), tie-break player name asc.
3. app.py uses orchestration service entrypoints, not inline graph orchestration.
4. Graph can produce deterministic output even when one worker source is sparse.
5. Missing config/retrieval failures return deterministic user-facing fallback text.

## Pass Criteria

A validation pass requires:
1. No unhandled exceptions for the baseline cohort.
2. Graph route trace confirms delegator -> worker fan-out -> synthesizer.
3. Final report is non-empty or fallback text is deterministic.
4. Identity resolution path returns a stable bundle when recruit_id or linked id exists.

## Validation Table Template

| year | recruit_id | player | target_team | route_ok | identity_ok | sections_ok | synthesis_ok | fallback_ok | errors |
|---|---|---|---|---|---|---|---|---|---|
| 2026 | 202600001 | Jared Curtis | TBD |  |  |  |  |  |  |
| 2026 | 202600002 | Lamar Brown | TBD |  |  |  |  |  |  |
| 2027 | 202700001 | John Meredith III | TBD |  |  |  |  |  |  |
| 2027 | 202700002 | Mark Matthews | TBD |  |  |  |  |  |  |
| 2028 | 202800001 | Brysen Wright | TBD |  |  |  |  |  |  |
| 2028 | 202800002 | Jalanie George | TBD |  |  |  |  |  |  |

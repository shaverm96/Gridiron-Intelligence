## Engine Restructure Baseline

Purpose: Provide one canonical baseline document for parity checks during the engine restructure.

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

## Structured Report Contract

Expected sections in order:
1. Scouting Workbench header with player name.
2. Metadata lines: recruit_id, year, target team.
3. Player Profile.
4. Score card HTML block.
5. Historical Comparables.
6. Filtered Scouting Profile.
7. Web Intelligence Summary.
8. Vector Insights.
9. Final Synthesis.

## Required Invariants

1. Year filter constrains player dropdown options.
2. Player selection resolves to a single canonical recruit_id.
3. Target team stays independent from player selection.
4. Structured synthesis includes SQL/player, scouting, web, vector, and comparables context.
5. Structured path excludes CFBD context.
6. Sorting in structured view is rating desc within year, tie-break name asc, null-safe.
7. Missing config or retrieval failures return deterministic user-facing errors.
8. Secrets precedence and diagnostics visibility remain unchanged.
9. Structured/chat page state isolation is preserved in multipage architecture.

## Pass Criteria

A run passes when all baseline recruits satisfy:
1. All required sections render in order.
2. Data blocks render valid JSON where expected.
3. Synthesis is non-empty or deterministic fallback is shown.
4. No unhandled exceptions occur.

## Validation Table Template

| year | recruit_id | player | target_team | sections_ok | profile_ok | scorecard_ok | comparables_ok | scouting_ok | web_ok | vector_ok | synthesis_ok | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026 | 202600001 | Jared Curtis | TBD |  |  |  |  |  |  |  |  |  |
| 2026 | 202600002 | Lamar Brown | TBD |  |  |  |  |  |  |  |  |  |
| 2027 | 202700001 | John Meredith III | TBD |  |  |  |  |  |  |  |  |  |
| 2027 | 202700002 | Mark Matthews | TBD |  |  |  |  |  |  |  |  |  |
| 2028 | 202800001 | Brysen Wright | TBD |  |  |  |  |  |  |  |  |  |
| 2028 | 202800002 | Jalanie George | TBD |  |  |  |  |  |  |  |  |  |

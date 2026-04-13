# Gridiron Intelligence

Gridiron Intelligence is a Streamlit scouting app with two production workflows:
- Recruiting Portal: build structured recruiting evaluations for high school prospects with a required position filter and a default top-100 candidate window by rating.
- Transfer Portal: evaluate transfer candidates with CFBD usage and season stats plus web context using a required position filter and live search results.

The app opens on a Landing Page where users choose a portal.

## What You Can Do

### Recruiting Portal
1. Select recruiting class year, position, and target team.
2. Browse the top 100 candidates by rating for the selected class year and position.
3. Use text search to expand beyond the default window when the first 100 results do not include what you need.
4. Generate a full recruiting report with:
- profile and projection context
- web scouting summaries
- historical comparables
- final synthesis
5. Use Open Chat for follow-up questions tied to the generated recruiting context.

### Transfer Portal
1. Select a position filter, then type at least 3 letters to load matching transfer candidates.
2. Pick a candidate and target team.
3. Generate transfer impact output with:
- final synthesis
- side-by-side charts (usage line plus season stat bar)
- transfer tables
- debug details with branch health, raw payloads, and compact payloads
4. Use Open Chat for transfer-specific follow-up analysis that reuses the generated report context.

### Local CFBD Debugger (if enabled)
1. Run canonical CFBD pulls for a selected transfer candidate.
2. Inspect raw yearly pulls, compact payloads, diagnostics, and charts.
3. Use it to validate parity against Transfer Portal data pulls.

## Quick Start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Configure environment
Set required keys using your preferred method (for example environment variables, local env files, or Streamlit secrets):
- SUPABASE_URL
- SUPABASE_SERVICE_ROLE_KEY
- GEMINI_API_KEY
- CFBD_API_KEY (or CFBD_API)

### 3) Run the app
```bash
python -m streamlit run app.py
```

Open the local URL shown in terminal output.

## Current App Structure

Primary app entry:
- app.py

Core engine modules:
- engine/orchestration_service.py
- engine/graph.py
- engine/agents.py
- engine/data_access.py
- engine/data_transforms.py
- engine/synthesis_service.py
- engine/tools.py

## Documentation and Diagrams

Production architecture summary:
- Agentic_Refactor/PRODUCTION_APP_STRUCTURE.md

Mermaid workflow sources:
- Agentic_Refactor/diagrams/app_navigation_workflow.mmd
- Agentic_Refactor/diagrams/recruiting_portal_workflow.mmd
- Agentic_Refactor/diagrams/transfer_portal_workflow.mmd
- Agentic_Refactor/diagrams/multi_agent_setup.mmd
- Agentic_Refactor/diagrams/data_context_layers.mmd

## Troubleshooting

### Transfer shows empty sections
Check Transfer Debug Details first:
- Pipeline Branch Health
- Pull Config
- Per-Year Diagnostics

Then compare with Local CFBD Debugger for the same player/settings.

### Summaries unavailable
If summaries are empty or skipped:
- verify GEMINI_API_KEY
- verify web search dependency availability
- inspect branch reason fields in Transfer Debug Details

### CFBD pulls unavailable
If CFBD payloads are empty or skipped:
- verify CFBD_API_KEY
- inspect pull diagnostics and queried teams
- validate candidate has a CFBD athlete ID in transfer index

## Notes

This README reflects the current production app behavior and intentionally focuses on user workflows and operation.

---
title: "Gridiron Intelligence: Multi-Agent Football Scouting Application"
author:
	- name: "Matthew Shaver"
		affiliation: "UNC Charlotte"
		email: "mshaver5@charlotte.edu"
	- name: "Elliott Kervin"
		affiliation: "UNC Charlotte"
		email: "ekervin@charlotte.edu"
format:
	html:
		toc: true
		number-sections: true
	pdf:
		toc: true
		number-sections: true
---

## Abstract

[Write one paragraph that covers: problem, target users, method, and outcomes.]

Suggested focus:
- Problem: fragmented recruiting and transfer scouting workflows are difficult to standardize and compare.
- Solution: a production-focused Streamlit system with deterministic report generation plus multi-agent chat reasoning.
- Core technical contribution: explicit delegator plus worker fan-out/fan-in graph with secure tool integration.
- Outcome summary: improved consistency, traceability, and decision support for football personnel analysis.

## 1. Introduction

### 1.1 Background and Motivation

[Explain why college football scouting needs both structured analytics and up-to-date contextual intelligence.]

Key points to include:
- Recruiting and transfer decisions combine historical performance, projection models, and current news context.
- Single-source or ad hoc workflows produce inconsistent recommendations.
- Need for a system that balances deterministic outputs (score cards, comparables) with agentic exploration.

### 1.2 Problem Statement

[Define the central problem your system solves and who benefits.]

Suggested statement:
- Build a unified scouting application that supports two decision workflows:
	- Recruiting evaluations (high school prospects).
	- Transfer impact analysis (college players).
- Preserve reliability and explainability while adding multi-agent reasoning and tool use.

### 1.3 Project Objectives

Map this section to course learning objectives:
- Prompt engineering for consistent report quality.
- Tool calling for external data systems.
- Agent architecture for decomposition and orchestration.
- Security and robustness controls for production behavior.
- Evaluation and performance measurement across workflows.

## 2. Methodology

### 2.1 System Overview

Reference implementation artifacts:
- UI entry and routing: `app.py`.
- Engine boundary and orchestration exports: `engine/__init__.py`.
- User-facing workflow summary: `README.md`.

Narrative to write:
- Landing page routes users into Recruiting Portal or Transfer Portal.
- Structured workflows produce report artifacts and seed context for follow-up chat.

### 2.2 Application Architecture

Use the architecture layers documented in:
- `Agentic_Refactor/PRODUCTION_APP_STRUCTURE.md`

Describe layers:
1. UI and routing.
2. Orchestration and graph execution.
3. Data access and transforms.
4. Prompt composition and synthesis.
5. External systems (Supabase, CFBD, web search, LLM).

### 2.3 Multi-Agent Workflow Design

Primary graph source:
- `engine/graph.py`

Node implementation source:
- `engine/agents.py`

Document both orchestration paths:
1. Full chat graph:
	 - `lead_delegator` -> (`cfbd_analyst`, `recruiting_scout`, `team_scout`) -> `lead_synthesizer`.
2. Structured web graph:
	 - `recruiting_scout` -> `team_scout`.

Design rationale to explain:
- Delegator plans work before execution.
- Specialized workers operate on different evidence channels.
- Synthesizer merges evidence with grounding rules and fallback behavior.

### 2.4 Workflow-Specific Orchestration

Reference:
- `engine/orchestration_service.py`
- `app.py`

Cover three execution patterns:
1. Structured recruiting report generation (deterministic report sections plus web-scout summaries).
2. Transfer report generation (parallel CFBD plus web branches with final synthesis).
3. Open chat turn orchestration (stateful, context-grounded, clarification-aware).

### 2.5 State and Memory Contract

Reference:
- `engine/state.py`

Include:
- `ScoutState` fields for identity, summaries, contexts, citations, errors, and trace log.
- `DelegatorPlan` schema and validation constraints.
- Conversation compaction and bounded-memory behavior for stable session performance.

### 2.6 Prompting Strategy

Reference:
- `engine/prompt_architecture.py`
- `engine/synthesis_service.py`
- `engine/agents.py`

Document:
- Separation of summarization prompts vs final synthesis prompts.
- Grounding policies and source priority rules.
- Deterministic handling for report-referential follow-up queries.

### 2.7 Security and Robustness Controls

Reference:
- `Agentic_Refactor/SECURITY_UPDATES.md`
- `engine/tools.py`
- `engine/agents.py`
- `app.py`

Explain implemented controls:
- Input sanitization and schema validation.
- Safe handling when delegator output is malformed.
- Prompt truncation and payload-size controls.
- Summary sanitization against unsafe HTML/script artifacts.
- Submission rate limiting for structured report generation.

### 2.8 Architecture Figures

Insert or adapt diagrams from:
- `Agentic_Refactor/diagrams/app_navigation_workflow.mmd`
- `Agentic_Refactor/diagrams/multi_agent_setup.mmd`
- `Agentic_Refactor/diagrams/recruiting_portal_workflow.mmd`
- `Agentic_Refactor/diagrams/transfer_portal_workflow.mmd`

[Add each figure with caption and short interpretation paragraph.]

## 3. Data Used

### 3.1 Data Sources

[Describe all sources and how they map to workflows.]

Use this structure:
- Recruit profile and scouting features from Supabase tables.
- College/transfer context from Supabase plus CFBD API pulls.
- Unstructured context from web search snippets.
- Vector factoids from embedding-backed retrieval.

### 3.2 Core Tables and Schema Direction

Reference:
- `Agentic_Refactor/SUPABASE_ENGINE_ALIGNMENT_ROADMAP.md`

Include key entities:
- `gi_recruit_master`
- `gi_college_master`
- `gi_player_link_bridge`
- `gi_scouting_report_features`
- `gi_model_prediction_score`
- `gi_model_prediction_thresholds`
- `gi_factoid_vectors`

### 3.3 Data Characteristics and Quality

[Add concrete counts and null-rate statistics from your notebooks or DB diagnostics.]

Suggested table template:

| Dataset/Table | Rows | Key fields | Missingness notes | Primary use |
|---|---:|---|---|---|
| gi_recruit_master | [fill] | recruit_id, player_name, class_year | [fill] | Recruiting report |
| gi_college_master | [fill] | college_player_id, cfbd_athlete_id | [fill] | Transfer report |
| gi_player_link_bridge | [fill] | recruit_id, cfbd_athlete_id | [fill] | Identity resolution |

## 4. Concepts Implemented

### 4.1 Prompting

[Summarize instruction design, output formatting constraints, and grounding rules.]

### 4.2 RAG and Retrieval

[Describe vector insight retrieval and how retrieved context is merged into synthesis.]

### 4.3 Tool Calling

[Document tools for Supabase access, CFBD pulls, web search, summarization, and final synthesis.]

### 4.4 Agent Design

[Explain delegator-worker-synthesizer decomposition and why it improves modularity and observability.]

### 4.5 Safety and Security

[Summarize prompt hygiene, validation, fallback pathways, and rate-limiting controls.]

### 4.6 APIs and Integrations

[List external services and integration responsibilities.]

### 4.7 Scaling and Reliability

[Describe retries, batching, timeout handling, checkpoint behavior, and compact state constraints.]

## 5. Experiments

[Use this section for architecture-first evidence with light quantitative placeholders.]

### 5.1 Model and Prompt Experiments

Suggested comparisons:
- Summary model variants and prompt formats.
- Final synthesis model settings (temperature, token budget).
- Deterministic vs flexible response style under follow-up queries.

Template table:

| Experiment | Setup A | Setup B | Metric(s) | Result |
|---|---|---|---|---|
| Summarizer prompt style | [fill] | [fill] | Relevance, clarity | [fill] |
| Final synthesis consistency | [fill] | [fill] | Section compliance | [fill] |

### 5.2 Agent Workflow Experiments

Suggested scenarios:
- Ambiguous player identity.
- Sparse CFBD response.
- Missing/weak web context.
- Report-referential follow-up in open chat.

### 5.3 Retrieval and Context Experiments

[Add retrieval setup and quality checks for vector factoids and web summaries.]

## 6. Results

### 6.1 Functional Outcomes

[Quantify and summarize what works end to end across Recruiting, Transfer, and Open Chat flows.]

### 6.2 Qualitative Outputs

Include:
- Screenshots of recruiting report cards and final synthesis.
- Transfer charts and diagnostic views.
- Example trace output for agent execution.

### 6.3 Quantitative Outcomes

Template table:

| Metric family | Metric | Value | Notes |
|---|---|---:|---|
| Latency | p50 response time | [fill] | [fill] |
| Reliability | Failure rate | [fill] | [fill] |
| Quality | Human relevance score | [fill] | [fill] |
| Cost | Avg cost per report | [fill] | [fill] |

## 7. Evaluation: How We Measured System Performance

### 7.1 Evaluation Dimensions

Cover at minimum:
- Accuracy and grounding.
- Relevance and coherence.
- Faithfulness to available evidence.
- Task completion and error recovery.
- Stability under missing data.

### 7.2 Evaluation Procedure

[Describe test set construction, manual scoring rubric, and any automated checks used.]

### 7.3 Failure Modes and Mitigations

[List top observed failures and the mitigation implemented in code.]

## 8. Conclusion

[Summarize what was built, what technical decisions mattered most, and what value the system provides.]

## 9. Limitations and Future Work

Ground this section in:
- `Agentic_Refactor/ENGINE_RESTRUCTURE_FULL_PLAN.md`
- `Agentic_Refactor/ENGINE_RESTRUCTURE_PROGRESS.md`
- `Agentic_Refactor/SUPABASE_ENGINE_ALIGNMENT_ROADMAP.md`

Suggested items:
- Remaining schema-contract centralization work.
- Additional parity and smoke testing closure.
- Controlled degraded-mode fallback when Supabase is unavailable.
- Expanded evaluation automation and cost instrumentation.
- Optional multimodal extension path.

## 10. References

[Add formal citations for all external resources, APIs, frameworks, and related tools used.]

Suggested reference categories:
- Streamlit documentation.
- LangGraph or orchestration framework documentation.
- Google Gemini API documentation.
- CFBD API documentation.
- Supabase documentation.
- Any evaluation framework docs used.

## Appendix A. Optional Deliverable Checklist Mapping

[Use this appendix to map report content to milestone and rubric requirements.]

| Requirement | Section(s) in this report | Evidence artifact |
|---|---|---|
| Architecture and methodology | Sections 2, 4 | `app.py`, `engine/*`, `Agentic_Refactor/*` |
| Data used | Section 3 | Supabase tables and notebooks |
| Experiments and results | Sections 5-7 | Metrics tables, screenshots, logs |
| Security and robustness | Sections 2.7, 4.5, 7.3 | `SECURITY_UPDATES.md`, engine controls |
| Limitations and future work | Section 9 | Restructure and roadmap docs |

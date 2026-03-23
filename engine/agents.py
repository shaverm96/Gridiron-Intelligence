from __future__ import annotations

import json
from typing import Any

from .state import ScoutState
from .tools import (
    build_synthesis_prompt,
    fetch_player_bundle_tool,
    final_synthesis_tool,
    historical_comparables_tool,
    normalize_position_group,
    search_web_tool,
    summarize_web_tool,
    vector_insights_tool,
)


def _append_citations(state: ScoutState, new_items: list[dict[str, str]]) -> None:
    citations = list(state.get("citations", []))
    citations.extend(new_items)
    state["citations"] = citations


def _append_error(state: ScoutState, message: str) -> None:
    errors = list(state.get("errors", []))
    errors.append(message)
    state["errors"] = errors


def _infer_chat_route(query: str) -> str:
    q = query.lower()
    if any(word in q for word in ["compare", "similar", "historical"]):
        return "comparables"
    if any(word in q for word in ["news", "transfer", "portal", "update", "recent"]):
        return "web_scout"
    if any(word in q for word in ["report", "fit", "recommend", "scout"]):
        return "sql_analyst"
    return "web_scout"


def supervisor_node(state: ScoutState) -> ScoutState:
    mode = state.get("mode", "chat")

    if mode == "structured_report":
        missing = []
        if not state.get("recruit_id"):
            missing.append("recruit_id")
        if not state.get("target_team"):
            missing.append("target_team")
        if missing:
            state["missing_fields"] = missing
            state["next_step"] = "synthesizer"
            _append_error(state, f"Missing required fields: {', '.join(missing)}")
            return state

        if not state.get("sql_data_context"):
            state["next_step"] = "sql_analyst"
            return state
        if not state.get("web_research_context"):
            state["next_step"] = "web_scout"
            return state
        if not state.get("vector_factoids"):
            state["next_step"] = "vector_analyst"
            return state
        if not state.get("comparables_context"):
            state["next_step"] = "comparables"
            return state

        state["next_step"] = "synthesizer"
        return state

    # Chat mode
    if not state.get("sql_data_context") and state.get("recruit_id"):
        state["next_step"] = "sql_analyst"
        return state

    state["next_step"] = _infer_chat_route(state.get("user_query", ""))
    return state


def sql_analyst_node(state: ScoutState) -> ScoutState:
    recruit_id = str(state.get("recruit_id", "")).strip()
    if not recruit_id:
        _append_error(state, "SQL analyst skipped: no recruit_id in state")
        state["next_step"] = "synthesizer"
        return state

    bundle_result = fetch_player_bundle_tool(recruit_id)
    if bundle_result.get("status") != "ok":
        _append_error(state, f"SQL bundle fetch failed: {bundle_result.get('reason', 'unknown error')}")

    state["sql_data_context"] = bundle_result.get("data", {})
    _append_citations(state, bundle_result.get("citations", []))

    player = state["sql_data_context"].get("player", {}) if state.get("sql_data_context") else {}
    if not state.get("player_name"):
        state["player_name"] = str(player.get("player_name") or "")

    state["next_step"] = "supervisor"
    return state


def web_scout_node(state: ScoutState) -> ScoutState:
    bundle = state.get("sql_data_context", {})
    player = bundle.get("player", {}) if isinstance(bundle, dict) else {}

    player_name = state.get("player_name") or player.get("player_name") or "Unknown Player"
    position = str(player.get("position") or "")
    high_school = str(player.get("high_school") or "")
    year = int(state.get("year") or 0)

    search_result = search_web_tool(player_name, position, high_school, year, max_results=12)
    summary_result = summarize_web_tool(player_name, position, search_result.get("data", []))

    state["web_research_context"] = str(summary_result.get("data", ""))
    _append_citations(state, search_result.get("citations", []))
    _append_citations(state, summary_result.get("citations", []))

    state["next_step"] = "supervisor"
    return state


def vector_analyst_node(state: ScoutState) -> ScoutState:
    bundle = state.get("sql_data_context", {})
    player = bundle.get("player", {}) if isinstance(bundle, dict) else {}
    scouting = bundle.get("scouting", {}) if isinstance(bundle, dict) else {}

    position = str(player.get("position") or "")
    position_group = normalize_position_group(position)
    player_state = str(player.get("state") or "").strip().upper()

    query_text = (
        f"Player: {state.get('player_name', '')}\n"
        f"Position: {position}\n"
        f"Position Group: {position_group}\n"
        f"State: {player_state}\n"
        f"Scouting: {json.dumps(scouting, default=str)}\n"
        f"Web Summary: {state.get('web_research_context', '')}"
    )

    vector_result = vector_insights_tool(
        query_text=query_text,
        position=position_group,
        state=player_state,
    )

    state["vector_factoids"] = list(vector_result.get("data", []))
    _append_citations(state, vector_result.get("citations", []))
    state["next_step"] = "supervisor"
    return state


def comparables_node(state: ScoutState) -> ScoutState:
    recruit_id = str(state.get("recruit_id", "")).strip()
    if not recruit_id:
        state["comparables_context"] = "Historical comparables unavailable: missing recruit_id."
        state["next_step"] = "supervisor"
        return state

    comp_result = historical_comparables_tool(recruit_id)
    state["comparables_context"] = str(comp_result.get("data", ""))
    _append_citations(state, comp_result.get("citations", []))
    state["next_step"] = "supervisor"
    return state


def synthesizer_node(state: ScoutState) -> ScoutState:
    prompt = build_synthesis_prompt(
        state_bundle=state.get("sql_data_context", {}),
        web_summary=state.get("web_research_context", ""),
        vector_factoids=state.get("vector_factoids", []),
        comparables_md=state.get("comparables_context", ""),
        target_team=state.get("target_team", ""),
        year=int(state.get("year") or 0),
        user_query=state.get("user_query", "Generate a scouting report."),
    )

    final_result = final_synthesis_tool(prompt)
    report_text = str(final_result.get("data", ""))

    if state.get("mode") == "chat":
        history = list(state.get("conversation_history", []))
        if state.get("user_query"):
            history.append({"role": "user", "content": str(state.get("user_query"))})
        history.append({"role": "assistant", "content": report_text})
        state["conversation_history"] = history

    state["final_report"] = report_text
    _append_citations(state, final_result.get("citations", []))
    state["next_step"] = "end"
    return state


def chat_followup_node(state: ScoutState) -> ScoutState:
    # For now, follow-up uses the same synthesis path with accumulated context.
    return synthesizer_node(state)

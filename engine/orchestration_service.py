from __future__ import annotations

import re
from typing import Any, Callable

from .graph import get_scout_graph, get_structured_web_graph
from .state import ScoutState, initial_chat_state, initial_structured_state, initial_structured_web_state


ProgressCallback = Callable[[dict[str, str]], None]

CHAT_STATE_MAX_TURNS = 6
CHAT_STATE_MAX_TRACE = 10
CHAT_STATE_MAX_ERRORS = 6
CHAT_STATE_MAX_CITATIONS = 16
CHAT_STATE_MAX_CANDIDATES = 3


def _ensure_base_state(state: dict[str, Any] | None) -> ScoutState:
    base = dict(state or {})
    if "mode" not in base:
        base["mode"] = "chat"
    if "conversation_history" not in base:
        base["conversation_history"] = []
    if "errors" not in base:
        base["errors"] = []
    if "citations" not in base:
        base["citations"] = []
    if "trace_log" not in base:
        base["trace_log"] = []
    return _compact_chat_state(base)


def _compact_chat_state(state: dict[str, Any] | None) -> ScoutState:
    src = dict(state or {})
    compact: ScoutState = {
        "mode": "chat",
        "user_query": str(src.get("user_query") or ""),
        "target_player_name": str(src.get("target_player_name") or ""),
        "player_name": str(src.get("player_name") or ""),
        "recruit_id": str(src.get("recruit_id") or ""),
        "cfbd_athlete_id": str(src.get("cfbd_athlete_id") or ""),
        "target_team": str(src.get("target_team") or ""),
        "year": int(src.get("year") or 0),
        "delegator_plan": dict(src.get("delegator_plan") or {}),
        "cfbd_data_summary": str(src.get("cfbd_data_summary") or ""),
        "web_recruiting_summary": str(src.get("web_recruiting_summary") or ""),
        "web_team_summary": str(src.get("web_team_summary") or ""),
        "final_report": str(src.get("final_report") or ""),
        "next_step": str(src.get("next_step") or "supervisor"),
        "missing_fields": list(src.get("missing_fields") or []),
        "requires_identity_clarification": bool(src.get("requires_identity_clarification")),
        "clarification_prompt": str(src.get("clarification_prompt") or ""),
        "pending_identity_query": str(src.get("pending_identity_query") or ""),
        "security_halt": bool(src.get("security_halt")),
        "security_message": str(src.get("security_message") or ""),
    }

    compact["identity_candidates"] = list(src.get("identity_candidates") or [])[-CHAT_STATE_MAX_CANDIDATES:]
    compact["conversation_history"] = list(src.get("conversation_history") or [])[-CHAT_STATE_MAX_TURNS * 2:]
    compact["trace_log"] = list(src.get("trace_log") or [])[-CHAT_STATE_MAX_TRACE:]
    compact["errors"] = list(src.get("errors") or [])[-CHAT_STATE_MAX_ERRORS:]
    compact["citations"] = list(src.get("citations") or [])[-CHAT_STATE_MAX_CITATIONS:]
    compact["sql_data_context"] = {}
    compact["web_research_context"] = ""
    compact["vector_factoids"] = []
    compact["comparables_context"] = ""

    return compact


def _emit_progress(progress_callback: ProgressCallback | None, node: str, status: str) -> None:
    if progress_callback is None:
        return
    progress_callback({"node": str(node), "status": str(status)})


def _merge_update(state: ScoutState, update: dict[str, Any]) -> ScoutState:
    merged = dict(state)
    for key, value in dict(update or {}).items():
        if key in {"citations", "errors", "trace_log"}:
            merged[key] = list(merged.get(key, [])) + list(value or [])
        else:
            merged[key] = value
    return merged


def _candidate_name(row: dict[str, Any]) -> str:
    for key in ("player_name", "full_name", "recruit_name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _try_resolve_clarification_response(state: ScoutState, user_prompt: str) -> ScoutState:
    if not bool(state.get("requires_identity_clarification")):
        return state

    prompt = str(user_prompt or "").strip()
    candidates = list(state.get("identity_candidates") or [])
    if not prompt or not candidates:
        return state

    selected: dict[str, Any] | None = None

    if re.fullmatch(r"\d+", prompt):
        idx = int(prompt) - 1
        if 0 <= idx < len(candidates):
            selected = dict(candidates[idx])
    else:
        normalized = prompt.lower()
        for row in candidates:
            name = _candidate_name(row)
            if name and name.lower() == normalized:
                selected = dict(row)
                break

    if selected is None:
        return state

    pending_query = str(state.get("pending_identity_query") or "").strip()
    selected_name = _candidate_name(selected)
    selected_recruit_id = str(selected.get("recruit_id") or "").strip()
    selected_athlete_id = str(selected.get("cfbd_athlete_id") or "").strip()

    updates: ScoutState = {
        "requires_identity_clarification": False,
        "clarification_prompt": "",
        "identity_candidates": [],
        "recruit_id": selected_recruit_id,
        "cfbd_athlete_id": selected_athlete_id,
        "target_player_name": selected_name or str(state.get("target_player_name") or ""),
        "missing_fields": [],
        "pending_identity_query": "",
    }
    if pending_query:
        updates["user_query"] = pending_query

    return _merge_update(state, updates)


def _invoke_graph(
    graph_runner: Any,
    state: ScoutState,
    progress_callback: ProgressCallback | None = None,
) -> ScoutState:
    if hasattr(graph_runner, "stream"):
        _emit_progress(progress_callback, "workflow", "started")
        latest_state: ScoutState = dict(state)
        for event in graph_runner.stream(state, stream_mode="updates"):
            if not isinstance(event, dict):
                continue
            for node_name, update in event.items():
                if node_name == "__start__":
                    _emit_progress(progress_callback, "workflow", "running")
                    continue
                if node_name == "__end__":
                    _emit_progress(progress_callback, "workflow", "completed")
                    continue
                _emit_progress(progress_callback, str(node_name), "completed")
                if isinstance(update, dict):
                    latest_state = _merge_update(latest_state, update)
        return latest_state

    if hasattr(graph_runner, "invoke_with_progress"):
        return graph_runner.invoke_with_progress(state, progress_callback=progress_callback)

    _emit_progress(progress_callback, "workflow", "started")
    result = graph_runner.invoke(state)
    _emit_progress(progress_callback, "workflow", "completed")
    return result


def orchestrate_structured_report(
    player_name: str,
    recruit_id: str,
    target_team: str,
    year: int,
    user_query: str | None = None,
    graph: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ScoutState:
    graph_runner = graph or get_scout_graph()
    state = initial_structured_state(
        player_name=player_name,
        recruit_id=str(recruit_id),
        target_team=target_team,
        year=int(year),
    )
    state["target_player_name"] = player_name
    state["user_query"] = user_query or (
        f"Create a scouting report for {player_name} and evaluate fit for {target_team}."
    )
    state["mode"] = "structured_report"
    return _invoke_graph(graph_runner, state, progress_callback=progress_callback)


def orchestrate_chat_turn(
    user_prompt: str,
    current_state: dict[str, Any] | None = None,
    target_team: str = "",
    target_player_name: str = "",
    graph: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ScoutState:
    graph_runner = graph or get_scout_graph()
    state = _ensure_base_state(current_state)
    if not state:
        state = initial_chat_state(user_prompt)

    state["mode"] = "chat"
    state["user_query"] = str(user_prompt or "").strip()
    if target_team:
        state["target_team"] = target_team
    if target_player_name:
        state["target_player_name"] = target_player_name

    state = _try_resolve_clarification_response(state, user_prompt)
    state = _compact_chat_state(state)

    return _invoke_graph(graph_runner, state, progress_callback=progress_callback)


def orchestrate_structured_web_scouting(
    player_name: str,
    recruit_id: str,
    target_team: str,
    year: int,
    user_query: str | None = None,
    graph: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ScoutState:
    graph_runner = graph or get_structured_web_graph()
    state = initial_structured_web_state(
        player_name=player_name,
        recruit_id=str(recruit_id),
        target_team=target_team,
        year=int(year),
    )
    state["target_player_name"] = player_name
    state["user_query"] = user_query or (
        f"Create a structured scouting brief for {player_name} and evaluate fit for {target_team}."
    )
    state["mode"] = "structured_report"
    return _invoke_graph(graph_runner, state, progress_callback=progress_callback)

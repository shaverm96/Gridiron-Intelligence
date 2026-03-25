from __future__ import annotations

from typing import Any

from .graph import get_scout_graph
from .state import ScoutState, initial_chat_state, initial_structured_state


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
    return base


def orchestrate_structured_report(
    player_name: str,
    recruit_id: str,
    target_team: str,
    year: int,
    user_query: str | None = None,
    graph: Any | None = None,
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
    return graph_runner.invoke(state)


def orchestrate_chat_turn(
    user_prompt: str,
    current_state: dict[str, Any] | None = None,
    target_team: str = "",
    target_player_name: str = "",
    graph: Any | None = None,
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

    return graph_runner.invoke(state)

from __future__ import annotations

from typing import Any, Literal, TypedDict


class ScoutState(TypedDict, total=False):
    # User request context
    mode: Literal["structured_report", "chat"]
    user_query: str
    player_name: str
    recruit_id: str
    target_team: str
    year: int

    # Gathered contexts
    sql_data_context: dict[str, Any]
    web_research_context: str
    vector_factoids: list[str]
    comparables_context: str

    # Output and traceability
    final_report: str
    citations: list[dict[str, str]]

    # Follow-up memory
    conversation_history: list[dict[str, str]]

    # Routing
    next_step: str
    missing_fields: list[str]
    errors: list[str]


def initial_structured_state(
    player_name: str,
    recruit_id: str,
    target_team: str,
    year: int,
) -> ScoutState:
    return {
        "mode": "structured_report",
        "user_query": "",
        "player_name": player_name,
        "recruit_id": recruit_id,
        "target_team": target_team,
        "year": int(year),
        "sql_data_context": {},
        "web_research_context": "",
        "vector_factoids": [],
        "comparables_context": "",
        "final_report": "",
        "citations": [],
        "conversation_history": [],
        "next_step": "supervisor",
        "missing_fields": [],
        "errors": [],
    }


def initial_chat_state(user_query: str) -> ScoutState:
    return {
        "mode": "chat",
        "user_query": user_query,
        "player_name": "",
        "recruit_id": "",
        "target_team": "",
        "year": 0,
        "sql_data_context": {},
        "web_research_context": "",
        "vector_factoids": [],
        "comparables_context": "",
        "final_report": "",
        "citations": [],
        "conversation_history": [{"role": "user", "content": user_query}],
        "next_step": "supervisor",
        "missing_fields": [],
        "errors": [],
    }

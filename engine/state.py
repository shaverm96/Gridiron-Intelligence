from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class DelegatorPlan(BaseModel):
    cfbd_search_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keys: name, position, college_team. Leave blank if unknown.",
    )
    recruiting_web_query: str = Field(default="", description="DuckDuckGo query for recruiting context.")
    team_context_query: str = Field(default="", description="DuckDuckGo query for team context.")
    user_intent: str = Field(default="", description="One-sentence user intent summary.")


class ScoutState(TypedDict, total=False):
    # User request context
    mode: Literal["structured_report", "chat"]
    user_query: str
    target_player_name: str
    player_name: str
    recruit_id: str
    cfbd_athlete_id: str
    target_team: str
    year: int

    # Delegator and worker summaries
    delegator_plan: dict[str, Any]
    cfbd_data_summary: str
    web_recruiting_summary: str
    web_team_summary: str

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
    trace_log: list[dict[str, Any]]


def initial_structured_state(
    player_name: str,
    recruit_id: str,
    target_team: str,
    year: int,
) -> ScoutState:
    return {
        "mode": "structured_report",
        "user_query": "",
        "target_player_name": player_name,
        "player_name": player_name,
        "recruit_id": recruit_id,
        "cfbd_athlete_id": "",
        "target_team": target_team,
        "year": int(year),
        "delegator_plan": DelegatorPlan(
            cfbd_search_params={
                "name": player_name,
                "college_team": target_team,
            },
            recruiting_web_query=f"{player_name} recruiting profile",
            team_context_query=f"{target_team} depth chart",
            user_intent="Generate a structured scouting report.",
        ).model_dump(),
        "cfbd_data_summary": "",
        "web_recruiting_summary": "",
        "web_team_summary": "",
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
        "trace_log": [],
    }


def initial_chat_state(user_query: str) -> ScoutState:
    return {
        "mode": "chat",
        "user_query": user_query,
        "target_player_name": "",
        "player_name": "",
        "recruit_id": "",
        "cfbd_athlete_id": "",
        "target_team": "",
        "year": 0,
        "delegator_plan": {},
        "cfbd_data_summary": "",
        "web_recruiting_summary": "",
        "web_team_summary": "",
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
        "trace_log": [],
    }

from __future__ import annotations

import operator
import re
from typing import Annotated, Any, ClassVar, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


class DelegatorPlan(BaseModel):
    ALLOWED_CFBD_KEYS: ClassVar[set[str]] = {"name", "position", "college_team"}

    cfbd_search_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keys: name, position, college_team. Leave blank if unknown.",
    )
    recruiting_web_query: str = Field(
        default="",
        max_length=200,
        description="DuckDuckGo query for recruiting context.",
    )
    team_context_query: str = Field(
        default="",
        max_length=200,
        description="DuckDuckGo query for team context.",
    )
    user_intent: str = Field(default="", max_length=300, description="One-sentence user intent summary.")

    @staticmethod
    def _sanitize_text(value: Any, max_len: int) -> str:
        text = str(value or "")
        text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_len]

    @field_validator("recruiting_web_query", "team_context_query", "user_intent", mode="before")
    @classmethod
    def _sanitize_text_fields(cls, value: Any) -> str:
        return cls._sanitize_text(value, 200)

    @field_validator("cfbd_search_params", mode="before")
    @classmethod
    def _validate_cfbd_search_params(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}

        cleaned: dict[str, Any] = {}
        for key in cls.ALLOWED_CFBD_KEYS:
            if key not in value:
                continue
            sanitized = cls._sanitize_text(value.get(key), 120)
            if sanitized:
                cleaned[key] = sanitized
        return cleaned


class ScoutState(TypedDict, total=False):
    # User request context
    mode: Literal["structured_report", "chat"]
    user_query: str
    target_player_name: str
    player_name: str
    recruit_id: str
    cfbd_athlete_id: str
    identity_candidates: list[dict[str, Any]]
    requires_identity_clarification: bool
    clarification_prompt: str
    pending_identity_query: str
    security_halt: bool
    security_message: str
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
    citations: Annotated[list[dict[str, str]], operator.add]

    # Follow-up memory
    conversation_history: list[dict[str, str]]

    # Routing
    next_step: str
    missing_fields: list[str]
    errors: Annotated[list[str], operator.add]
    trace_log: Annotated[list[dict[str, Any]], operator.add]


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
        "identity_candidates": [],
        "requires_identity_clarification": False,
        "clarification_prompt": "",
        "pending_identity_query": "",
        "security_halt": False,
        "security_message": "",
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
        "identity_candidates": [],
        "requires_identity_clarification": False,
        "clarification_prompt": "",
        "pending_identity_query": "",
        "security_halt": False,
        "security_message": "",
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


def initial_structured_web_state(
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
        "identity_candidates": [],
        "requires_identity_clarification": False,
        "clarification_prompt": "",
        "pending_identity_query": "",
        "security_halt": False,
        "security_message": "",
        "target_team": target_team,
        "year": int(year),
        "delegator_plan": DelegatorPlan(
            cfbd_search_params={},
            recruiting_web_query=f"{player_name} recruiting injury transfer update",
            team_context_query=f"{target_team} depth chart roster outlook",
            user_intent="Generate structured recruiting and team web summaries.",
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
        "next_step": "web_scout",
        "missing_fields": [],
        "errors": [],
        "trace_log": [],
    }

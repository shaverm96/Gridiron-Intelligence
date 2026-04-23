from __future__ import annotations

import operator
import re
from typing import Annotated, Any, ClassVar, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

from .web_query_templates import recruiting_player_query, recruiting_team_query


class DelegatorPlan(BaseModel):
    ALLOWED_CFBD_KEYS: ClassVar[set[str]] = {"name", "position", "college_team"}

    cfbd_search_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Keys: name, position, college_team. Leave blank if unknown.",
    )
    recruiting_web_query: str = Field(
        default="",
        max_length=200,
        description="Tavily query for recruiting context.",
    )
    team_context_query: str = Field(
        default="",
        max_length=200,
        description="Tavily query for team context.",
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


class TransferDelegatorPlan(BaseModel):
    player_news_query: str = Field(
        default="",
        max_length=200,
        description="Tavily query for transfer portal player news.",
    )
    team_news_query: str = Field(
        default="",
        max_length=200,
        description="Tavily query for team roster/depth chart transfer context.",
    )
    user_intent: str = Field(
        default="",
        max_length=300,
        description="One-sentence user intent summary for transfer chat follow-up.",
    )
    should_refresh_web: bool = Field(
        default=True,
        description="Whether to refresh web search or use cached context only.",
    )

    @staticmethod
    def _sanitize_text(value: Any, max_len: int) -> str:
        text = str(value or "")
        text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_len]

    @field_validator("player_news_query", "team_news_query", "user_intent", mode="before")
    @classmethod
    def _sanitize_text_fields(cls, value: Any) -> str:
        return cls._sanitize_text(value, 200)


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
    active_report_context: dict[str, Any]

    # Delegator and worker summaries
    delegator_plan: dict[str, Any]
    transfer_delegator_plan: dict[str, Any]
    cfbd_data_summary: str
    web_recruiting_summary: str
    web_team_summary: str
    web_recruiting_retrieval: dict[str, Any]
    web_team_retrieval: dict[str, Any]
    transfer_web_player_summary: str
    transfer_web_team_summary: str
    transfer_web_player_retrieval: dict[str, Any]
    transfer_web_team_retrieval: dict[str, Any]

    # Gathered contexts
    sql_data_context: dict[str, Any]
    transfer_report_context: dict[str, Any]
    web_research_context: str
    web_recruiting_used: bool
    web_team_used: bool
    transfer_web_player_used: bool
    transfer_web_team_used: bool
    allow_web_refresh: bool
    vector_factoids: list[str]
    comparables_context: str
    telemetry: dict[str, Any]

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
        "active_report_context": {},
        "delegator_plan": DelegatorPlan(
            cfbd_search_params={
                "name": player_name,
                "college_team": target_team,
            },
            recruiting_web_query=recruiting_player_query(player_name),
            team_context_query=recruiting_team_query(target_team),
            user_intent="Generate a structured scouting report.",
        ).model_dump(),
        "cfbd_data_summary": "",
        "web_recruiting_summary": "",
        "web_team_summary": "",
        "sql_data_context": {},
        "web_research_context": "",
        "web_recruiting_used": False,
        "web_team_used": False,
        "vector_factoids": [],
        "comparables_context": "",
        "telemetry": {},
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
        "active_report_context": {},
        "delegator_plan": {},
        "cfbd_data_summary": "",
        "web_recruiting_summary": "",
        "web_team_summary": "",
        "sql_data_context": {},
        "web_research_context": "",
        "web_recruiting_used": False,
        "web_team_used": False,
        "vector_factoids": [],
        "comparables_context": "",
        "telemetry": {},
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
        "active_report_context": {},
        "delegator_plan": DelegatorPlan(
            cfbd_search_params={},
            recruiting_web_query=recruiting_player_query(player_name),
            team_context_query=recruiting_team_query(target_team),
            user_intent="Generate structured recruiting and team web summaries.",
        ).model_dump(),
        "cfbd_data_summary": "",
        "web_recruiting_summary": "",
        "web_team_summary": "",
        "sql_data_context": {},
        "web_research_context": "",
        "vector_factoids": [],
        "comparables_context": "",
        "telemetry": {},
        "final_report": "",
        "citations": [],
        "conversation_history": [],
        "next_step": "web_scout",
        "missing_fields": [],
        "errors": [],
        "trace_log": [],
    }


def compact_open_chat_state(
    state: dict[str, Any] | None,
    max_turns: int = 6,
    max_trace: int = 10,
    max_errors: int = 6,
    max_citations: int = 16,
    max_candidates: int = 3,
) -> dict[str, Any]:
    src = dict(state or {})
    compact: dict[str, Any] = {
        "mode": "chat",
        "user_query": str(src.get("user_query") or ""),
        "target_player_name": str(src.get("target_player_name") or ""),
        "player_name": str(src.get("player_name") or ""),
        "recruit_id": str(src.get("recruit_id") or ""),
        "cfbd_athlete_id": str(src.get("cfbd_athlete_id") or ""),
        "target_team": str(src.get("target_team") or ""),
        "year": int(src.get("year") or 0),
        "active_report_context": dict(src.get("active_report_context") or {}),
        "requires_identity_clarification": bool(src.get("requires_identity_clarification")),
        "clarification_prompt": str(src.get("clarification_prompt") or ""),
        "pending_identity_query": str(src.get("pending_identity_query") or ""),
        "security_halt": bool(src.get("security_halt")),
        "security_message": str(src.get("security_message") or ""),
        "next_step": str(src.get("next_step") or "supervisor"),
    }

    compact["identity_candidates"] = list(src.get("identity_candidates") or [])[-max_candidates:]
    compact["conversation_history"] = list(src.get("conversation_history") or [])[-max_turns * 2 :]
    compact["trace_log"] = list(src.get("trace_log") or [])[-max_trace:]
    compact["errors"] = list(src.get("errors") or [])[-max_errors:]
    compact["citations"] = list(src.get("citations") or [])[-max_citations:]

    compact["sql_data_context"] = {}
    compact["web_research_context"] = ""
    compact["vector_factoids"] = []
    compact["comparables_context"] = ""
    return compact


def compact_transfer_chat_state(
    state: dict[str, Any] | None,
    max_turns: int = 6,
    max_trace: int = 10,
) -> dict[str, Any]:
    src = dict(state or {})
    return {
        "transfer_report_context": dict(src.get("transfer_report_context") or {}),
        "conversation_history": list(src.get("conversation_history") or [])[-max_turns * 2 :],
        "trace_log": list(src.get("trace_log") or [])[-max_trace:],
    }

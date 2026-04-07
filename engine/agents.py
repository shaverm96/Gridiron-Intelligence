from __future__ import annotations

from typing import Any

from .prompt_architecture import build_master_prompt
from .state import DelegatorPlan, ScoutState
from .tools import (
    cfbd_fetch_tool,
    cfbd_search_players_tool,
    delegator_plan_tool,
    fetch_player_bundle_by_identity_tool,
    final_synthesis_tool,
    historical_comparables_tool,
    resolve_player_identity_tool,
    search_web_query_tool,
    summarize_payload_tool,
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


def _append_trace(state: ScoutState, node_name: str, note: str = "") -> None:
    trace_log = list(state.get("trace_log", []))
    trace_log.append(
        {
            "node": node_name,
            "note": note,
            "recruit_id": state.get("recruit_id", ""),
            "cfbd_athlete_id": state.get("cfbd_athlete_id", ""),
        }
    )
    state["trace_log"] = trace_log


def _truncate_text_block(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()} ...[truncated]"


def _is_meaningful_summary(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    weak_markers = [
        "summary unavailable",
        "no data",
        "insufficient",
        "skipped",
        "failed",
    ]
    if any(marker in text for marker in weak_markers):
        return False
    return len(text) >= 80


def _has_internal_grounding(state: ScoutState) -> bool:
    bundle = dict(state.get("sql_data_context") or {})
    player_profile = dict(bundle.get("player") or {})
    has_player_profile = bool(player_profile)
    has_cfbd_summary = _is_meaningful_summary(state.get("cfbd_data_summary", ""))
    return has_player_profile or has_cfbd_summary


def _needs_web_recency_enrichment(state: ScoutState) -> bool:
    query = str(state.get("user_query") or "").lower()
    recency_terms = ["news", "recent", "update", "transfer", "portal", "injury", "latest"]
    return any(term in query for term in recency_terms)


def _should_use_duckduckgo(state: ScoutState, scope: str) -> tuple[bool, str]:
    # Primary-first policy: web search is only fallback/enrichment.
    if not _has_internal_grounding(state):
        return True, f"{scope}_fallback_missing_internal"
    if _needs_web_recency_enrichment(state):
        return True, f"{scope}_enrichment_recent_updates"
    return False, f"{scope}_skipped_internal_sufficient"


def _infer_chat_route(query: str) -> str:
    q = str(query or "").lower()
    if any(word in q for word in ["compare", "similar", "historical"]):
        return "comparables"
    if any(word in q for word in ["news", "transfer", "portal", "update", "recent"]):
        return "web_scout"
    if any(word in q for word in ["report", "fit", "recommend", "scout"]):
        return "report"
    return "report"


def lead_delegator_node(state: ScoutState) -> ScoutState:
    route_hint = _infer_chat_route(str(state.get("user_query") or ""))
    plan_dict = delegator_plan_tool(
        user_query=str(state.get("user_query", "") or "Generate a scouting report."),
        target_team=str(state.get("target_team", "")),
        target_player_name=str(state.get("target_player_name") or state.get("player_name") or ""),
    )

    try:
        state["delegator_plan"] = DelegatorPlan(**plan_dict).model_dump()
    except Exception:
        state["delegator_plan"] = DelegatorPlan().model_dump()
        _append_error(state, "Delegator plan parse failed; using defaults.")

    _append_trace(state, "lead_delegator", f"delegator_plan_ready route={route_hint}")
    state["next_step"] = "synthesizer"
    return state


def cfbd_analyst_node(state: ScoutState) -> ScoutState:
    plan = state.get("delegator_plan") or {}
    cfbd_params = plan.get("cfbd_search_params") if isinstance(plan, dict) else {}
    cfbd_params = cfbd_params if isinstance(cfbd_params, dict) else {}

    name = str(
        cfbd_params.get("name")
        or state.get("target_player_name")
        or state.get("player_name")
        or ""
    ).strip()
    team = str(cfbd_params.get("college_team") or state.get("target_team") or "").strip()

    if name and not state.get("cfbd_athlete_id"):
        identity_result = resolve_player_identity_tool(name)
        identity_data = identity_result.get("data") or {}
        if identity_data:
            state["recruit_id"] = str(identity_data.get("recruit_id") or state.get("recruit_id") or "")
            state["cfbd_athlete_id"] = str(identity_data.get("cfbd_athlete_id") or "")

    bundle_result = fetch_player_bundle_by_identity_tool(
        recruit_id=str(state.get("recruit_id") or "") or None,
        cfbd_athlete_id=str(state.get("cfbd_athlete_id") or "") or None,
        name_query=name or None,
    )
    if bundle_result.get("status") == "ok":
        state["sql_data_context"] = dict(bundle_result.get("data") or {})
        _append_citations(state, list(bundle_result.get("citations") or []))

    resolved_athlete_id = str(
        (state.get("sql_data_context") or {}).get("resolved_cfbd_athlete_id")
        or state.get("cfbd_athlete_id")
        or ""
    ).strip()

    position = str(cfbd_params.get("position") or "").strip()
    year_value = int(state.get("year") or 0) or None

    # If we still do not have an athlete id, run CFBD player search as deterministic fallback.
    if not resolved_athlete_id and name:
        search_result = cfbd_search_players_tool(
            search_term=name,
            year=year_value,
            team=team or None,
            position=position or None,
        )
        _append_citations(state, list(search_result.get("citations") or []))

        search_rows = list(search_result.get("data") or [])
        if search_rows:
            preferred_team = (team or "").strip().lower()

            def _candidate_score(row: dict[str, Any]) -> tuple[int, int]:
                row_team = str(row.get("team") or row.get("school") or "").strip().lower()
                has_id = 1 if str(row.get("athleteId") or row.get("athlete_id") or "").strip() else 0
                team_match = 1 if preferred_team and row_team == preferred_team else 0
                return (team_match, has_id)

            best_row = sorted(search_rows, key=_candidate_score, reverse=True)[0]
            resolved_athlete_id = str(best_row.get("athleteId") or best_row.get("athlete_id") or "").strip()
            if resolved_athlete_id:
                state["cfbd_athlete_id"] = resolved_athlete_id

    if resolved_athlete_id:
        state["cfbd_athlete_id"] = resolved_athlete_id

    if not resolved_athlete_id and not team:
        cfbd_result = {
            "status": "skipped",
            "reason": "missing athlete_id and team",
            "data": [],
            "citations": [],
        }
    else:
        cfbd_result = cfbd_fetch_tool(
            athlete_id=resolved_athlete_id or None,
            team=team or None,
            year=year_value,
            endpoint="player/stats" if resolved_athlete_id else "roster",
        )
    summary_result = summarize_payload_tool(
        summary_prompt=(
            "Summarize CFBD data in concise markdown bullets for scouting synthesis. "
            "Include only grounded facts and note if data is sparse."
        ),
        payload=cfbd_result.get("data", []),
    )

    state["cfbd_data_summary"] = str(summary_result.get("data", "")).strip()
    _append_citations(state, list(cfbd_result.get("citations") or []))
    _append_citations(state, list(summary_result.get("citations") or []))
    _append_trace(state, "cfbd_analyst", "cfbd_summary_ready")
    return state


def recruiting_scout_node(state: ScoutState) -> ScoutState:
    use_web, reason = _should_use_duckduckgo(state, "recruiting")
    state["web_recruiting_used"] = use_web
    if not use_web:
        state["web_recruiting_summary"] = "Web enrichment not used: internal backend evidence is sufficient for current request."
        _append_trace(state, "recruiting_scout", reason)
        return state

    plan = state.get("delegator_plan") or {}
    query = ""
    if isinstance(plan, dict):
        query = str(plan.get("recruiting_web_query") or "").strip()
    if not query:
        fallback_name = state.get("target_player_name") or state.get("player_name") or "player"
        query = f"{fallback_name} recruiting injury transfer update"

    search_result = search_web_query_tool(query=query, max_results=6)
    summary_result = summarize_payload_tool(
        summary_prompt=(
            "Summarize recruiting and player web context in markdown bullets for a scouting report as "
            "supplemental enrichment only. Do not override internal backend facts. "
            "Use only supplied snippets and include caveats when uncertain."
        ),
        payload=search_result.get("data", []),
    )

    state["web_recruiting_summary"] = str(summary_result.get("data", "")).strip()
    _append_citations(state, list(search_result.get("citations") or []))
    _append_citations(state, list(summary_result.get("citations") or []))
    _append_trace(state, "recruiting_scout", f"web_recruiting_summary_ready reason={reason}")
    return state


def team_scout_node(state: ScoutState) -> ScoutState:
    use_web, reason = _should_use_duckduckgo(state, "team")
    state["web_team_used"] = use_web
    if not use_web:
        state["web_team_summary"] = "Web enrichment not used: internal backend evidence is sufficient for current request."
        _append_trace(state, "team_scout", reason)
        return state

    plan = state.get("delegator_plan") or {}
    query = ""
    if isinstance(plan, dict):
        query = str(plan.get("team_context_query") or "").strip()
    if not query:
        fallback_team = state.get("target_team") or "team"
        query = f"{fallback_team} depth chart roster defensive backfield outlook"

    search_result = search_web_query_tool(query=query, max_results=6)
    summary_result = summarize_payload_tool(
        summary_prompt=(
            "Summarize team context for roster fit in concise markdown bullets as supplemental enrichment only. "
            "Use only supplied snippets and avoid unsupported claims. Do not override internal backend facts."
        ),
        payload=search_result.get("data", []),
    )

    state["web_team_summary"] = str(summary_result.get("data", "")).strip()
    _append_citations(state, list(search_result.get("citations") or []))
    _append_citations(state, list(summary_result.get("citations") or []))
    _append_trace(state, "team_scout", f"web_team_summary_ready reason={reason}")
    return state


def lead_synthesizer_node(state: ScoutState) -> ScoutState:
    plan = state.get("delegator_plan") or {}
    user_intent = ""
    if isinstance(plan, dict):
        user_intent = str(plan.get("user_intent") or "").strip()
    if not user_intent:
        user_intent = str(state.get("user_query") or "Generate a scouting report.")

    bundle = dict(state.get("sql_data_context") or {})
    player_profile = dict(bundle.get("player") or {})
    profile_position = str(player_profile.get("position") or "").strip()
    profile_state = str(player_profile.get("state") or "").strip() or None

    if profile_position:
        vector_result = vector_insights_tool(
            query_text=user_intent,
            position=profile_position,
            state=profile_state,
            top_k=6,
        )
        raw_factoids = [
            _truncate_text_block(item, 800)
            for item in list(vector_result.get("data") or [])[:6]
        ]
        state["vector_factoids"] = raw_factoids
        _append_citations(state, list(vector_result.get("citations") or []))

    recruit_id = str(state.get("recruit_id") or "").strip()
    if recruit_id:
        comparables_result = historical_comparables_tool(recruit_id)
        state["comparables_context"] = _truncate_text_block(comparables_result.get("data") or "", 5000)
        _append_citations(state, list(comparables_result.get("citations") or []))

    synthesis_payload = {
        "player_name": state.get("target_player_name") or state.get("player_name") or "",
        "user_intent": _truncate_text_block(user_intent, 500),
        "user_query": _truncate_text_block(state.get("user_query") or "", 2200),
        "player_profile": player_profile,
        "cfbd_summary": _truncate_text_block(state.get("cfbd_data_summary", ""), 6000),
        "recruiting_summary": _truncate_text_block(state.get("web_recruiting_summary", ""), 5000),
        "team_summary": _truncate_text_block(state.get("web_team_summary", ""), 5000),
        "vector_factoids": list(state.get("vector_factoids") or []),
        "historical_comparables": _truncate_text_block(state.get("comparables_context", ""), 5000),
        "source_priority": {
            "primary": "internal_backend_data_vectors_and_repository_context",
            "secondary": "duckduckgo_supplemental_enrichment_only",
            "final": "model_reasoning_supported_by_available_evidence_only",
        },
        "source_usage": {
            "internal_grounding_available": _has_internal_grounding(state),
            "duckduckgo_recruiting_used": bool(state.get("web_recruiting_used", False)),
            "duckduckgo_team_used": bool(state.get("web_team_used", False)),
        },
    }

    synthesis_prompt = build_master_prompt(
        player_name=str(synthesis_payload.get("player_name") or "Unknown Player"),
        target_team=str(state.get("target_team") or ""),
        year=int(state.get("year") or 0),
        user_prompt=str(state.get("user_query") or user_intent),
        retrieved_context=synthesis_payload,
    )

    final_result = final_synthesis_tool(synthesis_prompt)
    report_text = str(final_result.get("data", "")).strip()
    state["final_report"] = report_text
    _append_citations(state, list(final_result.get("citations") or []))
    _append_trace(state, "lead_synthesizer", "final_report_ready")

    if state.get("mode") == "chat":
        history = list(state.get("conversation_history", []))
        if state.get("user_query"):
            history.append({"role": "user", "content": str(state.get("user_query"))})
        history.append({"role": "assistant", "content": report_text})
        state["conversation_history"] = history

    state["next_step"] = "end"
    return state


# Backward-compatible aliases kept for older call sites.
def supervisor_node(state: ScoutState) -> ScoutState:
    return lead_delegator_node(state)


def sql_analyst_node(state: ScoutState) -> ScoutState:
    return cfbd_analyst_node(state)


def web_scout_node(state: ScoutState) -> ScoutState:
    return recruiting_scout_node(state)


def vector_analyst_node(state: ScoutState) -> ScoutState:
    return team_scout_node(state)


def comparables_node(state: ScoutState) -> ScoutState:
    return team_scout_node(state)


def synthesizer_node(state: ScoutState) -> ScoutState:
    return lead_synthesizer_node(state)


def chat_followup_node(state: ScoutState) -> ScoutState:
    return lead_synthesizer_node(state)

from __future__ import annotations

import json
import re
from typing import Any

from .config import CONFIG
from .prompt_architecture import build_master_prompt
from .state import DelegatorPlan, ScoutState
from .tools import (
    DelegatorOutputValidationError,
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


def _trace_entry(state: ScoutState, node_name: str, note: str = "") -> dict[str, Any]:
    return {
        "node": node_name,
        "note": note,
        "recruit_id": state.get("recruit_id", ""),
        "cfbd_athlete_id": state.get("cfbd_athlete_id", ""),
    }


def _truncate_text_block(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()} ...[truncated]"


def _append_citations(state: ScoutState, citations: list[dict[str, Any]] | None) -> None:
    if not citations:
        return
    try:
        merged = list(state.get("citations") or [])
        merged.extend(item for item in citations if isinstance(item, dict))
        state["citations"] = merged
    except Exception:
        # Citation enrichment is best-effort and must not break chat responses.
        return


def _fallback_master_prompt(*, user_prompt: str, payload: dict[str, Any]) -> str:
    # Keep synthesis running if prompt architecture helpers are unavailable.
    safe_query = str(user_prompt or "Generate a scouting report.").strip() or "Generate a scouting report."
    payload_json = json.dumps(payload, indent=2, default=str)
    return (
        "You are Gridiron Intelligence Scout, a professional football scouting analyst. "
        "Use only the provided context and avoid unsupported claims.\n\n"
        f"USER REQUEST:\n{safe_query}\n\n"
        f"CONTEXT:\n{payload_json}\n"
    )


def _is_report_referential_query(query: str) -> bool:
    text = str(query or "").strip().lower()
    if not text:
        return False
    markers = [
        "listed above",
        "above",
        "scorecard",
        "report",
        "that recommendation",
        "confidence score",
        "second comparable",
        "second player",
        "that section",
        "this section",
        "what you listed",
        "the comparables",
        "development risk",
    ]
    return any(marker in text for marker in markers)


def _render_report_comparables(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines: list[str] = []
    for idx, row in enumerate(rows, start=1):
        name = str(row.get("name") or row.get("player_name") or "").strip()
        if not name:
            continue
        match = str(row.get("match") or "").strip()
        year = str(row.get("year") or "").strip()
        state = str(row.get("state") or "").strip()
        bits = [f"{idx}. {name}"]
        if match:
            bits.append(f"Match: {match}")
        if year:
            bits.append(f"Class: {year}")
        if state:
            bits.append(f"State: {state}")
        lines.append(" | ".join(bits))
    return "\n".join(lines)


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


def _sanitize_query_input(value: str) -> str:
    text = str(value or "").strip()
    text = text.replace("%", " ").replace("_", " ")
    return " ".join(text.split())[:120]


def _infer_year_from_text(*values: str) -> int | None:
    for value in values:
        text = str(value or "")
        matches = re.findall(r"\b(20\d{2})\b", text)
        for match in matches:
            year = int(match)
            if 2010 <= year <= 2035:
                return year
    return None


def _select_cfbd_endpoint(user_intent: str, resolved_athlete_id: str) -> str:
    q = str(user_intent or "").strip().lower()
    if any(token in q for token in ("recruit", "recruiting", "prospect")):
        return "recruiting"
    if any(token in q for token in ("usage", "snap share", "target share", "garbage time")):
        return "player/usage"
    if any(token in q for token in ("season stats", "player season", "aggregated stats", "stat line")):
        return "stats/player/season"
    return "player/usage" if resolved_athlete_id else "roster"


def lead_delegator_node(state: ScoutState) -> ScoutState:
    route_hint = _infer_chat_route(str(state.get("user_query") or ""))
    try:
        plan_dict = delegator_plan_tool(
            user_query=str(state.get("user_query", "") or "Generate a scouting report."),
            target_team=str(state.get("target_team", "")),
            target_player_name=str(state.get("target_player_name") or state.get("player_name") or ""),
        )
    except DelegatorOutputValidationError:
        state["security_halt"] = True
        state["security_message"] = "Unable to safely parse your request. Please rephrase with concise football-only instructions."
        state["errors"] = list(state.get("errors", [])) + [
            "Delegator validation failed; execution halted for safety."
        ]
        state["trace_log"] = list(state.get("trace_log", [])) + [
            _trace_entry(state, "lead_delegator", "security_halt_delegator_validation_failed")
        ]
        state["next_step"] = "security_halt"
        return state

    try:
        state["delegator_plan"] = DelegatorPlan(**plan_dict).model_dump()
    except Exception:
        state["delegator_plan"] = DelegatorPlan().model_dump()
        state["errors"] = list(state.get("errors", [])) + ["Delegator plan parse failed; using defaults."]

    state["trace_log"] = list(state.get("trace_log", [])) + [
        _trace_entry(state, "lead_delegator", f"delegator_plan_ready route={route_hint}")
    ]
    state["next_step"] = "synthesizer"
    return state


def cfbd_analyst_node(state: ScoutState) -> ScoutState:
    if bool(state.get("security_halt")):
        return {
            "trace_log": [_trace_entry(state, "cfbd_analyst", "skipped_security_halt")],
            "cfbd_data_summary": "CFBD analysis skipped due to security safeguards.",
        }

    updates: dict[str, Any] = {}
    citations: list[dict[str, str]] = []
    errors: list[str] = []
    traces: list[dict[str, Any]] = []

    plan = state.get("delegator_plan") or {}
    cfbd_params = plan.get("cfbd_search_params") if isinstance(plan, dict) else {}
    cfbd_params = cfbd_params if isinstance(cfbd_params, dict) else {}

    name = _sanitize_query_input(str(
        cfbd_params.get("name")
        or state.get("target_player_name")
        or state.get("player_name")
        or ""
    ))
    team = _sanitize_query_input(str(cfbd_params.get("college_team") or state.get("target_team") or ""))
    position = _sanitize_query_input(str(cfbd_params.get("position") or ""))
    year_value = int(state.get("year") or 0) or None
    if year_value is None:
        inferred_year = _infer_year_from_text(
            str(plan.get("user_intent") or ""),
            str(state.get("user_query") or ""),
            str(plan.get("recruiting_web_query") or ""),
            str(plan.get("team_context_query") or ""),
        )
        if inferred_year is not None:
            year_value = inferred_year
            updates["year"] = inferred_year

    recruit_id = _sanitize_query_input(str(state.get("recruit_id") or ""))
    cfbd_athlete_id = str(state.get("cfbd_athlete_id") or "").strip()

    if name and not cfbd_athlete_id:
        identity_result = resolve_player_identity_tool(name, year=year_value, position=position or None, team=team or None)
        identity_data = identity_result.get("data") or {}
        if bool(identity_data.get("requires_clarification")):
            pending_query = str(state.get("pending_identity_query") or state.get("user_query") or "")
            updates["requires_identity_clarification"] = True
            updates["identity_candidates"] = list(identity_data.get("top_candidates") or [])
            updates["clarification_prompt"] = str(identity_data.get("clarification_prompt") or "")
            updates["pending_identity_query"] = pending_query
            updates["missing_fields"] = ["player_identity"]
            updates["cfbd_data_summary"] = "Identity clarification required before CFBD lookup can continue."
            traces.append(_trace_entry(state, "cfbd_analyst", "identity_clarification_required"))
            updates["trace_log"] = traces
            return updates
        if identity_data:
            recruit_id = str(identity_data.get("recruit_id") or recruit_id or "")
            cfbd_athlete_id = str(identity_data.get("cfbd_athlete_id") or "")
            updates["recruit_id"] = recruit_id
            updates["cfbd_athlete_id"] = cfbd_athlete_id
            updates["requires_identity_clarification"] = False
            updates["identity_candidates"] = list(identity_data.get("top_candidates") or [])
            updates["clarification_prompt"] = ""
            updates["pending_identity_query"] = ""

    bundle_result = fetch_player_bundle_by_identity_tool(
        recruit_id=recruit_id or None,
        cfbd_athlete_id=cfbd_athlete_id or None,
        name_query=name or None,
        year=year_value,
        position=position or None,
        team=team or None,
    )
    if bundle_result.get("status") == "ok":
        bundle_data = dict(bundle_result.get("data") or {})
        identity = dict(bundle_data.get("identity") or {})
        if bool(identity.get("requires_clarification")):
            pending_query = str(state.get("pending_identity_query") or state.get("user_query") or "")
            updates["requires_identity_clarification"] = True
            updates["identity_candidates"] = list(identity.get("top_candidates") or [])
            updates["clarification_prompt"] = str(identity.get("clarification_prompt") or "")
            updates["pending_identity_query"] = pending_query
            updates["missing_fields"] = ["player_identity"]
            updates["cfbd_data_summary"] = "Identity clarification required before CFBD lookup can continue."
            traces.append(_trace_entry(state, "cfbd_analyst", "bundle_identity_clarification_required"))
            updates["trace_log"] = traces
            return updates

        updates["sql_data_context"] = bundle_data
        citations.extend(list(bundle_result.get("citations") or []))

    resolved_athlete_id = str(
        (updates.get("sql_data_context") or state.get("sql_data_context") or {}).get("resolved_cfbd_athlete_id")
        or updates.get("cfbd_athlete_id")
        or cfbd_athlete_id
        or ""
    ).strip()

    # If we still do not have an athlete id, run CFBD player search as deterministic fallback.
    if not resolved_athlete_id and name:
        search_result = cfbd_search_players_tool(
            search_term=name,
            year=year_value,
            team=team or None,
            position=position or None,
        )
        citations.extend(list(search_result.get("citations") or []))

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
                updates["cfbd_athlete_id"] = resolved_athlete_id

    if resolved_athlete_id:
        updates["cfbd_athlete_id"] = resolved_athlete_id

    user_intent = str(plan.get("user_intent") or state.get("user_query") or "").strip()
    selected_endpoint = _select_cfbd_endpoint(user_intent=user_intent, resolved_athlete_id=resolved_athlete_id)
    cfbd_result = cfbd_fetch_tool(
        athlete_id=resolved_athlete_id or None,
        team=team or None,
        year=year_value,
        position=position or None,
        endpoint=selected_endpoint,
        search_term=name or None,
    )

    cfbd_meta = cfbd_result.get("meta") if isinstance(cfbd_result.get("meta"), dict) else {}
    cfbd_rows = list(cfbd_result.get("data") or [])
    summary_payload = {
        "endpoint": selected_endpoint,
        "status": str(cfbd_result.get("status") or ""),
        "reason": str(cfbd_result.get("reason") or ""),
        "request_params": dict(cfbd_meta.get("params") or {}),
        "row_count": len(cfbd_rows),
        "data_preview": cfbd_rows[:5],
    }

    if str(cfbd_result.get("status") or "") != "ok" and not cfbd_rows:
        updates["cfbd_data_summary"] = (
            f"- CFBD endpoint: {selected_endpoint}\n"
            f"- Status: {cfbd_result.get('status')}\n"
            f"- Reason: {cfbd_result.get('reason')}\n"
            f"- Request params: {json.dumps(summary_payload['request_params'], default=str)}\n"
            "- Data rows returned: 0"
        )
        citations.extend(list(cfbd_result.get("citations") or []))
        traces.append(_trace_entry(state, "cfbd_analyst", "cfbd_summary_unavailable_non_ok"))
        if citations:
            updates["citations"] = citations
        if traces:
            updates["trace_log"] = traces
        return updates

    summary_result = summarize_payload_tool(
        summary_prompt=(
            "You are a secure summarization node. Output ONLY plain markdown bullet points (no HTML, no JSON, no links). "
            f"Summarize CFBD data from endpoint '{selected_endpoint}' in concise bullets for scouting synthesis using only provided payload. "
            "Always include endpoint used, request filters, and number of rows returned before any player insights. "
            "Include grounded facts and explicitly note sparse/missing data."
        ),
        payload=summary_payload,
    )

    updates["cfbd_data_summary"] = str(summary_result.get("data", "")).strip()
    citations.extend(list(cfbd_result.get("citations") or []))
    citations.extend(list(summary_result.get("citations") or []))
    traces.append(_trace_entry(state, "cfbd_analyst", "cfbd_summary_ready"))

    if citations:
        updates["citations"] = citations
    if errors:
        updates["errors"] = errors
    if traces:
        updates["trace_log"] = traces
    return updates


def recruiting_scout_node(state: ScoutState) -> ScoutState:
    if bool(state.get("security_halt")):
        return {
            "web_recruiting_summary": "Recruiting summary skipped due to security safeguards.",
            "trace_log": [_trace_entry(state, "recruiting_scout", "skipped_security_halt")],
        }

    if bool(state.get("requires_identity_clarification")):
        return {
            "web_recruiting_summary": "Recruiting web context skipped until player identity is clarified.",
            "trace_log": [_trace_entry(state, "recruiting_scout", "skipped_needs_identity_clarification")],
        }

    plan = state.get("delegator_plan") or {}
    query = ""
    if isinstance(plan, dict):
        query = str(plan.get("recruiting_web_query") or "").strip()
    if not query:
        fallback_name = state.get("target_player_name") or state.get("player_name") or "player"
        query = f"{fallback_name} recruiting injury transfer update"

    search_result = search_web_query_tool(query=query, max_results=int(CONFIG.get("WEB_QUERY_MAX_RESULTS", 6)))
    summary_result = summarize_payload_tool(
        summary_prompt=(
            "You are a secure summarization node. Output ONLY plain markdown bullet points (no HTML, no JSON, no links). "
            "Summarize recruiting and player web context in bullets for a scouting report. "
            "Use only supplied snippets and include caveats when uncertain."
        ),
        payload=search_result.get("data", []),
    )

    return {
        "web_recruiting_summary": str(summary_result.get("data", "")).strip(),
        "citations": list(search_result.get("citations") or []) + list(summary_result.get("citations") or []),
        "trace_log": [_trace_entry(state, "recruiting_scout", "web_recruiting_summary_ready")],
    }


def team_scout_node(state: ScoutState) -> ScoutState:
    if bool(state.get("security_halt")):
        return {
            "web_team_summary": "Team context skipped due to security safeguards.",
            "trace_log": [_trace_entry(state, "team_scout", "skipped_security_halt")],
        }

    if bool(state.get("requires_identity_clarification")):
        return {
            "web_team_summary": "Team context skipped until player identity is clarified.",
            "trace_log": [_trace_entry(state, "team_scout", "skipped_needs_identity_clarification")],
        }

    plan = state.get("delegator_plan") or {}
    query = ""
    if isinstance(plan, dict):
        query = str(plan.get("team_context_query") or "").strip()
    if not query:
        fallback_team = state.get("target_team") or "team"
        query = f"{fallback_team} depth chart roster defensive backfield outlook"

    search_result = search_web_query_tool(query=query, max_results=int(CONFIG.get("WEB_QUERY_MAX_RESULTS", 6)))
    summary_result = summarize_payload_tool(
        summary_prompt=(
            "You are a secure summarization node. Output ONLY plain markdown bullet points (no HTML, no JSON, no links). "
            "Summarize team context for roster fit in concise bullets. "
            "Use only supplied snippets and avoid unsupported claims."
        ),
        payload=search_result.get("data", []),
    )

    return {
        "web_team_summary": str(summary_result.get("data", "")).strip(),
        "citations": list(search_result.get("citations") or []) + list(summary_result.get("citations") or []),
        "trace_log": [_trace_entry(state, "team_scout", "web_team_summary_ready")],
    }


def lead_synthesizer_node(state: ScoutState) -> ScoutState:
    if bool(state.get("security_halt")):
        message = str(state.get("security_message") or "Unable to process this request safely.")
        state["final_report"] = message
        state["next_step"] = "end"
        state["trace_log"] = list(state.get("trace_log", [])) + [
            _trace_entry(state, "lead_synthesizer", "security_halt_response_returned")
        ]
        return state

    if bool(state.get("requires_identity_clarification")):
        prompt = str(state.get("clarification_prompt") or "Please clarify which player you want to analyze.")
        state["final_report"] = prompt
        state["next_step"] = "clarify_identity"
        state["trace_log"] = list(state.get("trace_log", [])) + [
            _trace_entry(state, "lead_synthesizer", "identity_clarification_prompt_ready")
        ]
        if state.get("mode") == "chat":
            history = list(state.get("conversation_history", []))
            if state.get("user_query"):
                history.append({"role": "user", "content": str(state.get("user_query"))})
            history.append({"role": "assistant", "content": prompt})
            state["conversation_history"] = history
        return state

    plan = state.get("delegator_plan") or {}
    user_intent = ""
    if isinstance(plan, dict):
        user_intent = str(plan.get("user_intent") or "").strip()
    if not user_intent:
        user_intent = str(state.get("user_query") or "Generate a scouting report.")

    active_report_context = dict(state.get("active_report_context") or {})
    report_follow_up = _is_report_referential_query(str(state.get("user_query") or ""))
    report_comparables = list(active_report_context.get("comparables_list") or [])

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
    if report_follow_up and report_comparables:
        state["comparables_context"] = _truncate_text_block(_render_report_comparables(report_comparables), 5000)
    elif recruit_id:
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
        "active_report_context": active_report_context,
        "grounding_policy": {
            "report_follow_up_detected": report_follow_up,
            "primary_source_for_report_followups": "active_report_context",
            "when_report_is_referenced": "Use report artifacts first and do not substitute alternate comparable names.",
            "fallback_when_report_context_missing": "Use available state summaries and clearly mark additional interpretation.",
        },
        "source_priority": {
            "primary": "active_rendered_report_context_when_referenced_then_internal_backend_data_vectors_and_repository_context",
            "secondary": "duckduckgo_supplemental_enrichment_only",
            "final": "model_reasoning_supported_by_available_evidence_only",
        },
        "source_usage": {
            "internal_grounding_available": _has_internal_grounding(state),
            "duckduckgo_recruiting_used": bool(state.get("web_recruiting_used", False)),
            "duckduckgo_team_used": bool(state.get("web_team_used", False)),
        },
    }

    try:
        synthesis_prompt = build_master_prompt(
            player_name=str(synthesis_payload.get("player_name") or "Unknown Player"),
            target_team=str(state.get("target_team") or ""),
            year=int(state.get("year") or 0),
            user_prompt=str(state.get("user_query") or user_intent),
            retrieved_context=synthesis_payload,
        )
    except Exception:
        synthesis_prompt = _fallback_master_prompt(
            user_prompt=str(state.get("user_query") or user_intent),
            payload=synthesis_payload,
        )

    final_result = final_synthesis_tool(synthesis_prompt)
    report_text = str(final_result.get("data", "")).strip()
    state["final_report"] = report_text
    state["citations"] = list(state.get("citations", [])) + list(final_result.get("citations") or [])
    state["trace_log"] = list(state.get("trace_log", [])) + [_trace_entry(state, "lead_synthesizer", "final_report_ready")]

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

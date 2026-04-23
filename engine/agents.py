from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import CONFIG
from .prompt_architecture import build_master_prompt
from .state import DelegatorPlan, ScoutState, TransferDelegatorPlan
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
from .web_query_templates import recruiting_player_query, recruiting_team_query


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
        "score",
        "confidence",
        "recommendation",
        "scheme fit",
        "recruiting summary",
        "final synthesis",
        "development risk",
    ]
    return any(marker in text for marker in markers)


def _is_comparables_referential_query(query: str) -> bool:
    text = str(query or "").strip().lower()
    if not text:
        return False
    markers = [
        "comparables",
        "comp list",
        "players listed above",
        "top match",
        "second comparable",
        "third comparable",
        "match percentages",
        "why are these players similar",
        "break down those comparables",
    ]
    return any(marker in text for marker in markers)


def _normalize_active_comparables(active_report_context: dict[str, Any]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    raw_rows = list(active_report_context.get("comparables") or [])
    if not raw_rows:
        raw_rows = list(active_report_context.get("comparables_list") or [])

    for idx, row in enumerate(raw_rows, start=1):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("player_name") or "").strip()
        if not name:
            continue
        match = str(
            row.get("match_pct_display")
            or row.get("match")
            or ""
        ).strip()
        year = str(row.get("year") or row.get("class") or "").strip()
        state = str(row.get("state") or "").strip()
        rating = str(row.get("rating") or "").strip()
        normalized.append(
            {
                "index": str(row.get("index") or idx),
                "name": name,
                "match": match,
                "year": year,
                "state": state,
                "rating": rating,
            }
        )
    return normalized


def _deterministic_comparables_response(query: str, comparables: list[dict[str, str]]) -> str:
    q = str(query or "").lower()
    if not comparables:
        return (
            "I do not have the active comparables card in context for this session, "
            "so I cannot safely restate exact comparable names or match percentages. "
            "I can still explain the matching methodology in general terms."
        )

    lines = [
        "Using the active comparables card shown above, here are the exact matches:",
    ]
    for row in comparables:
        lines.append(f"- {row['name']} - Match {row['match']}")

    if "second comparable" in q or "2nd comparable" in q:
        selected = comparables[1] if len(comparables) > 1 else comparables[0]
        lines.append("")
        lines.append(
            f"Second comparable: {selected['name']} ({selected['match']})."
        )
    elif "top match" in q:
        top = comparables[0]
        lines.append("")
        lines.append(f"Top match: {top['name']} ({top['match']}).")

    lines.append("")
    lines.append(
        "Interpretation: these matches indicate profile similarity to the listed players; "
        "higher match percentages reflect closer similarity under the current model inputs."
    )
    lines.append(
        "I am intentionally anchoring to the currently rendered card and not adding alternate comparables."
    )
    return "\n".join(lines)


def _query_mentions_displayed_comparable(query: str, comparables: list[dict[str, str]]) -> bool:
    q = str(query or "").strip().lower()
    if not q:
        return False
    if "match" not in q and "comparable" not in q and "similar" not in q:
        return False
    for row in comparables:
        name = str(row.get("name") or "").strip().lower()
        if name and name in q:
            return True
    return False


def _build_rendered_report_grounding_context(
    active_report_context: dict[str, Any],
    query: str,
    comparables: list[dict[str, str]],
) -> dict[str, Any]:
    context = dict(active_report_context or {})
    context["report_referential_query"] = _is_report_referential_query(query)
    context["comparables_referential_query"] = _is_comparables_referential_query(query)
    context["comparables_exact_ordered"] = comparables
    context["grounding_contract"] = {
        "instruction": "When question references the report above, use this context first.",
        "comparables_rule": "If comparables are present here, preserve names, order, and match percentages exactly.",
        "no_silent_substitution": True,
    }
    return context


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


def _normalize_prior_colleges(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    if "," in text:
        return [part.strip() for part in text.split(",") if part.strip()]
    return [text]


def _player_summary_context(state: ScoutState, *, transfer_mode: bool = False) -> dict[str, Any]:
    if transfer_mode:
        transfer_ctx = dict(state.get("transfer_report_context") or {})
        player_row = dict(transfer_ctx.get("college_player") or {})
        prior_colleges = _normalize_prior_colleges(player_row.get("teams") or player_row.get("prior_colleges"))
        return {
            "player_name": str(transfer_ctx.get("player_name") or state.get("target_player_name") or state.get("player_name") or "").strip(),
            "player_position": str(player_row.get("position") or transfer_ctx.get("position") or "").strip(),
            "high_school": str(player_row.get("high_school") or transfer_ctx.get("high_school") or "").strip(),
            "prior_colleges": prior_colleges,
        }

    bundle = dict(state.get("sql_data_context") or {})
    player_row = dict(bundle.get("player") or {})
    prior_colleges = _normalize_prior_colleges(
        player_row.get("prior_colleges")
        or player_row.get("college_history")
        or player_row.get("teams")
    )
    return {
        "player_name": str(state.get("target_player_name") or state.get("player_name") or player_row.get("name") or "").strip(),
        "player_position": str(player_row.get("position") or "").strip(),
        "high_school": str(player_row.get("high_school") or player_row.get("high_school_name") or "").strip(),
        "prior_colleges": prior_colleges,
    }


def _team_summary_context(state: ScoutState, *, transfer_mode: bool = False) -> dict[str, Any]:
    if transfer_mode:
        transfer_ctx = dict(state.get("transfer_report_context") or {})
        return {
            "team_name": str(transfer_ctx.get("target_team") or state.get("target_team") or "").strip(),
            "conference": str(transfer_ctx.get("conference") or "").strip(),
        }

    bundle = dict(state.get("sql_data_context") or {})
    team_row = dict(bundle.get("team") or {})
    player_row = dict(bundle.get("player") or {})
    return {
        "team_name": str(state.get("target_team") or team_row.get("name") or player_row.get("college_team") or "").strip(),
        "conference": str(team_row.get("conference") or player_row.get("conference") or "").strip(),
    }


def _should_use_web_search_enrichment(state: ScoutState, scope: str) -> tuple[bool, str]:
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

    # Canonical follow-up behavior: the delegator controls optional web enrichment,
    # while respecting explicit orchestration-level disable flags.
    explicit_web_refresh_disabled = not bool(state.get("allow_web_refresh", True))
    if explicit_web_refresh_disabled:
        should_refresh_web = False
    else:
        should_refresh_web = route_hint == "web_scout"
    if not explicit_web_refresh_disabled and not should_refresh_web and not _has_internal_grounding(state):
        should_refresh_web = True
    state["allow_web_refresh"] = bool(should_refresh_web)

    if not bool(state.get("allow_web_refresh")) and isinstance(state.get("delegator_plan"), dict):
        state["delegator_plan"]["recruiting_web_query"] = ""
        state["delegator_plan"]["team_context_query"] = ""

    state["trace_log"] = list(state.get("trace_log", [])) + [
        _trace_entry(
            state,
            "lead_delegator",
            f"delegator_plan_ready route={route_hint} web_refresh={bool(state.get('allow_web_refresh'))}",
        )
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
        role="player",
        entity_kind="player",
        target_name=str(state.get("target_player_name") or state.get("player_name") or ""),
        target_team=str(state.get("target_team") or ""),
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
            "web_recruiting_retrieval": {
                "status": "skipped",
                "reason": "security_halt",
                "data": [],
            },
            "web_recruiting_used": False,
            "trace_log": [_trace_entry(state, "recruiting_scout", "skipped_security_halt")],
        }

    if bool(state.get("requires_identity_clarification")):
        return {
            "web_recruiting_summary": "Recruiting web context skipped until player identity is clarified.",
            "web_recruiting_retrieval": {
                "status": "skipped",
                "reason": "needs_identity_clarification",
                "data": [],
            },
            "web_recruiting_used": False,
            "trace_log": [_trace_entry(state, "recruiting_scout", "skipped_needs_identity_clarification")],
        }

    plan = state.get("delegator_plan") or {}
    query = ""
    if isinstance(plan, dict):
        query = str(plan.get("recruiting_web_query") or "").strip()
    if not query:
        fallback_name = state.get("target_player_name") or state.get("player_name") or "player"
        query = recruiting_player_query(str(fallback_name))

    search_result = search_web_query_tool(
        query=query,
        max_results=int(CONFIG.get("WEB_QUERY_MAX_RESULTS", 10)),
        timelimit="m",
    )
    summary_result = summarize_payload_tool(
        summary_prompt=(
            "You are a secure summarization filter. Output ONLY plain markdown bullet points (no HTML, no JSON, no links). "
            "Extract high-signal, player-relevant recruiting context with richer factual detail from the provided snippets. "
            "Keep useful scouting information (traits, athletic background, timeline notes, measurable stats, offers/visits, injuries, eligibility) when present. "
            "Remove legal boilerplate, copyright text, nav junk, and repetitive formatting noise. "
            "Use known player context (name, position, high school, prior colleges) to exclude unrelated players, teams, sports, and duplicate facts. "
            "Return 8-14 detailed bullets when evidence supports it, and include uncertainty caveats where needed. "
            "Use strictly the supplied snippets. "
        ),
        payload=search_result.get("data", []),
        role="recruiting_player",
        entity_kind="player",
        target_name=str(state.get("target_player_name") or state.get("player_name") or ""),
        target_team=str(state.get("target_team") or ""),
        entity_context=_player_summary_context(state),
    )

    return {
        "web_recruiting_summary": str(summary_result.get("data", "")).strip(),
        "web_recruiting_retrieval": dict(search_result or {}),
        "web_recruiting_used": True,
        "citations": list(search_result.get("citations") or []) + list(summary_result.get("citations") or []),
        "telemetry": dict(summary_result.get("telemetry") or {}),
        "trace_log": [_trace_entry(state, "recruiting_scout", "web_recruiting_summary_ready")],
    }


def team_scout_node(state: ScoutState) -> ScoutState:
    if bool(state.get("security_halt")):
        return {
            "web_team_summary": "Team context skipped due to security safeguards.",
            "web_team_retrieval": {
                "status": "skipped",
                "reason": "security_halt",
                "data": [],
            },
            "web_team_used": False,
            "trace_log": [_trace_entry(state, "team_scout", "skipped_security_halt")],
        }

    if bool(state.get("requires_identity_clarification")):
        return {
            "web_team_summary": "Team context skipped until player identity is clarified.",
            "web_team_retrieval": {
                "status": "skipped",
                "reason": "needs_identity_clarification",
                "data": [],
            },
            "web_team_used": False,
            "trace_log": [_trace_entry(state, "team_scout", "skipped_needs_identity_clarification")],
        }

    plan = state.get("delegator_plan") or {}
    query = ""
    if isinstance(plan, dict):
        query = str(plan.get("team_context_query") or "").strip()
    if not query:
        fallback_team = state.get("target_team") or "team"
        query = recruiting_team_query(str(fallback_team))

    search_result = search_web_query_tool(
        query=query,
        max_results=int(CONFIG.get("WEB_QUERY_MAX_RESULTS", 10)),
        timelimit="m",
    )
    summary_result = summarize_payload_tool(
        summary_prompt=(
            "You are a secure summarization filter. Output ONLY plain markdown bullet points (no HTML, no JSON, no links). "
            "Extract high-signal team context for roster fit with useful detail: roster needs, depth chart competition, staff turnover, coaching changes, positional battles, and spring standouts. "
            "Remove legal boilerplate, site disclaimers, and formatting clutter while preserving actionable facts. "
            "Use known team context (college name and conference) to ignore unrelated programs and other sports. "
            "Prioritize recent evidence, but keep still-relevant context if it improves fit evaluation. "
            "Return 6-12 detailed bullets when evidence supports it, and mark uncertainty when needed. "
            "Use strictly the supplied snippets. "
        ),
        payload=search_result.get("data", []),
        role="recruiting_team",
        entity_kind="team",
        target_name=str(state.get("target_player_name") or state.get("player_name") or ""),
        target_team=str(state.get("target_team") or ""),
        entity_context=_team_summary_context(state),
    )

    return {
        "web_team_summary": str(summary_result.get("data", "")).strip(),
        "web_team_retrieval": dict(search_result or {}),
        "web_team_used": True,
        "citations": list(search_result.get("citations") or []) + list(summary_result.get("citations") or []),
        "telemetry": dict(summary_result.get("telemetry") or {}),
        "trace_log": [_trace_entry(state, "team_scout", "web_team_summary_ready")],
    }


def parallel_web_scout_node(state: ScoutState) -> ScoutState:
    if bool(state.get("security_halt")):
        return {
            "web_recruiting_summary": "Recruiting summary skipped due to security safeguards.",
            "web_team_summary": "Team context skipped due to security safeguards.",
            "web_recruiting_retrieval": {
                "status": "skipped",
                "reason": "security_halt",
                "data": [],
            },
            "web_team_retrieval": {
                "status": "skipped",
                "reason": "security_halt",
                "data": [],
            },
            "trace_log": [_trace_entry(state, "parallel_web_scout", "skipped_security_halt")],
        }

    if bool(state.get("requires_identity_clarification")):
        return {
            "web_recruiting_summary": "Recruiting web context skipped until player identity is clarified.",
            "web_team_summary": "Team context skipped until player identity is clarified.",
            "web_recruiting_retrieval": {
                "status": "skipped",
                "reason": "needs_identity_clarification",
                "data": [],
            },
            "web_team_retrieval": {
                "status": "skipped",
                "reason": "needs_identity_clarification",
                "data": [],
            },
            "web_recruiting_used": False,
            "web_team_used": False,
            "trace_log": [_trace_entry(state, "parallel_web_scout", "skipped_needs_identity_clarification")],
        }

    if state.get("mode") == "chat" and not bool(state.get("allow_web_refresh", True)):
        return {
            "web_recruiting_summary": "Web enrichment skipped for this follow-up turn.",
            "web_team_summary": "Web enrichment skipped for this follow-up turn.",
            "web_recruiting_retrieval": {
                "status": "skipped",
                "reason": "delegator_no_web_refresh",
                "data": [],
            },
            "web_team_retrieval": {
                "status": "skipped",
                "reason": "delegator_no_web_refresh",
                "data": [],
            },
            "web_recruiting_used": False,
            "web_team_used": False,
            "trace_log": [_trace_entry(state, "parallel_web_scout", "skipped_delegator_no_web_refresh")],
        }

    def _run_recruiting() -> dict[str, Any]:
        return recruiting_scout_node(state)

    def _run_team() -> dict[str, Any]:
        return team_scout_node(state)

    parallel_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        recruiting_future = executor.submit(_run_recruiting)
        team_future = executor.submit(_run_team)
        recruiting_result = recruiting_future.result()
        team_result = team_future.result()

    merged_trace = list(recruiting_result.get("trace_log") or []) + list(team_result.get("trace_log") or [])
    merged_citations = list(recruiting_result.get("citations") or []) + list(team_result.get("citations") or [])
    model_telemetry_rows = [
        dict(recruiting_result.get("telemetry") or {}),
        dict(team_result.get("telemetry") or {}),
    ]
    model_telemetry_rows = [row for row in model_telemetry_rows if row]
    parallel_latency_ms = int((time.perf_counter() - parallel_started) * 1000)
    telemetry_rollup = {
        "model_call_count": len(model_telemetry_rows),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in model_telemetry_rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in model_telemetry_rows),
        "total_tokens": sum(int(row.get("total_tokens") or 0) for row in model_telemetry_rows),
        "estimated_cost_usd": round(
            sum(float(row.get("estimated_cost_usd") or 0.0) for row in model_telemetry_rows),
            8,
        ),
        "latency_ms": parallel_latency_ms,
    }
    if telemetry_rollup["total_tokens"] == 0:
        telemetry_rollup["total_tokens"] = telemetry_rollup["input_tokens"] + telemetry_rollup["output_tokens"]
    return {
        "web_recruiting_summary": str(recruiting_result.get("web_recruiting_summary") or "").strip(),
        "web_team_summary": str(team_result.get("web_team_summary") or "").strip(),
        "web_recruiting_retrieval": dict(recruiting_result.get("web_recruiting_retrieval") or {}),
        "web_team_retrieval": dict(team_result.get("web_team_retrieval") or {}),
        "web_recruiting_used": bool(recruiting_result.get("web_recruiting_used")),
        "web_team_used": bool(team_result.get("web_team_used")),
        "citations": merged_citations,
        "telemetry": {
            "model_telemetry": model_telemetry_rows,
            "model_rollup": telemetry_rollup,
        },
        "trace_log": merged_trace or [_trace_entry(state, "parallel_web_scout", "web_summaries_ready")],
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
    user_query_text = str(state.get("user_query") or "")
    comparables_follow_up = _is_comparables_referential_query(user_query_text)
    report_comparables = _normalize_active_comparables(active_report_context)
    if not comparables_follow_up and report_comparables:
        comparables_follow_up = _query_mentions_displayed_comparable(user_query_text, report_comparables)

    if comparables_follow_up:
        reply = _deterministic_comparables_response(
            query=user_query_text,
            comparables=report_comparables,
        )
        state["final_report"] = reply
        state["trace_log"] = list(state.get("trace_log", [])) + [
            _trace_entry(state, "lead_synthesizer", "comparables_report_grounded_response")
        ]
        if state.get("mode") == "chat":
            history = list(state.get("conversation_history", []))
            if state.get("user_query"):
                history.append({"role": "user", "content": str(state.get("user_query"))})
            history.append({"role": "assistant", "content": reply})
            state["conversation_history"] = history
        state["next_step"] = "end"
        return state

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

    rendered_report_context = _build_rendered_report_grounding_context(
        active_report_context=active_report_context,
        query=user_query_text,
        comparables=report_comparables,
    )
    supplemental_reasoning_context = {
        "player_name": state.get("target_player_name") or state.get("player_name") or "",
        "user_intent": _truncate_text_block(user_intent, 500),
        "user_query": _truncate_text_block(state.get("user_query") or "", 2200),
        "player_profile": player_profile,
        "cfbd_summary": _truncate_text_block(state.get("cfbd_data_summary", ""), 6000),
        "recruiting_summary": _truncate_text_block(state.get("web_recruiting_summary", ""), 5000),
        "team_summary": _truncate_text_block(state.get("web_team_summary", ""), 5000),
        "vector_factoids": list(state.get("vector_factoids") or []),
        "historical_comparables": _truncate_text_block(state.get("comparables_context", ""), 5000),
    }

    synthesis_payload = {
        "player_name": state.get("target_player_name") or state.get("player_name") or "",
        "user_intent": _truncate_text_block(user_intent, 500),
        "user_query": _truncate_text_block(state.get("user_query") or "", 2200),
        "current_rendered_report_context": rendered_report_context,
        "supplemental_scout_reasoning_context": supplemental_reasoning_context,
        "grounding_policy": {
            "report_follow_up_detected": report_follow_up,
            "primary_source_for_report_followups": "current_rendered_report_context",
            "supplemental_source": "supplemental_scout_reasoning_context",
            "when_report_is_referenced": "Answer from rendered report context first; supplement without contradiction.",
            "fallback_when_report_context_missing": "Be explicit that rendered report context is unavailable and avoid inventing exact on-card values.",
        },
        "source_priority": {
            "primary": "active_rendered_report_context_when_referenced_then_internal_backend_data_vectors_and_repository_context",
            "secondary": "tavily_supplemental_enrichment_only",
            "final": "model_reasoning_supported_by_available_evidence_only",
        },
        "source_usage": {
            "internal_grounding_available": _has_internal_grounding(state),
            "tavily_recruiting_used": bool(state.get("web_recruiting_used", False)),
            "tavily_team_used": bool(state.get("web_team_used", False)),
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
    return parallel_web_scout_node(state)


def vector_analyst_node(state: ScoutState) -> ScoutState:
    return parallel_web_scout_node(state)


def comparables_node(state: ScoutState) -> ScoutState:
    return parallel_web_scout_node(state)


def synthesizer_node(state: ScoutState) -> ScoutState:
    return lead_synthesizer_node(state)


def chat_followup_node(state: ScoutState) -> ScoutState:
    return lead_synthesizer_node(state)

def transfer_delegator_node(state: ScoutState) -> ScoutState:
    from .tools import transfer_delegator_plan_tool, TransferDelegatorOutputValidationError
    try:
        plan_dict = transfer_delegator_plan_tool(
            user_query=str(state.get("user_query", "") or "Analyze transfer portal opportunity."),
            target_team=str(state.get("target_team", "")),
            target_player_name=str(state.get("target_player_name") or state.get("player_name") or ""),
        )
    except TransferDelegatorOutputValidationError:
        state["security_halt"] = True
        state["security_message"] = "Unable to safely parse your request. Please rephrase with concise football-only instructions."
        state["errors"] = list(state.get("errors", [])) + [
            "Transfer Delegator validation failed; execution halted for safety."
        ]
        state["trace_log"] = list(state.get("trace_log", [])) + [
            _trace_entry(state, "transfer_delegator", "security_halt_delegator_validation_failed")
        ]
        state["next_step"] = "security_halt"
        return state

    try:
        state["transfer_delegator_plan"] = TransferDelegatorPlan(**plan_dict).model_dump()
    except Exception:
        state["transfer_delegator_plan"] = TransferDelegatorPlan().model_dump()
        state["errors"] = list(state.get("errors", [])) + ["Transfer Delegator plan parse failed; using defaults."]

    if not bool(state.get("allow_web_refresh", True)):
        state["transfer_delegator_plan"]["should_refresh_web"] = False
        state["transfer_delegator_plan"]["player_news_query"] = ""
        state["transfer_delegator_plan"]["team_news_query"] = ""

    state["trace_log"] = list(state.get("trace_log", [])) + [
        _trace_entry(state, "transfer_delegator", "transfer_delegator_plan_ready")
    ]
    
    plan = state.get("transfer_delegator_plan") or {}
    if plan.get("should_refresh_web"):
        state["next_step"] = "transfer_web_scout"
    else:
        state["next_step"] = "transfer_synthesizer"
    return state


def transfer_web_scout_node(state: ScoutState) -> ScoutState:
    if bool(state.get("security_halt")):
        return {
            "trace_log": [_trace_entry(state, "transfer_web_scout", "skipped")],
            "transfer_web_player_used": False,
            "transfer_web_team_used": False,
        }

    plan = state.get("transfer_delegator_plan") or {}
    if not bool(state.get("allow_web_refresh", True)) or not plan.get("should_refresh_web"):
        plan["player_news_query"] = ""
        plan["team_news_query"] = ""
    player_query = str(plan.get("player_news_query") or "").strip()
    team_query = str(plan.get("team_news_query") or "").strip()
    
    updates: dict[str, Any] = {}
    traces = []
    
    if not player_query and not team_query:
        traces.append(_trace_entry(state, "transfer_web_scout", "skipped_no_queries"))
        updates["transfer_web_player_summary"] = ""
        updates["transfer_web_team_summary"] = ""
        updates["transfer_web_player_retrieval"] = {"status": "skipped", "reason": "no_queries", "data": []}
        updates["transfer_web_team_retrieval"] = {"status": "skipped", "reason": "no_queries", "data": []}
        updates["transfer_web_player_used"] = False
        updates["transfer_web_team_used"] = False
        updates["trace_log"] = traces
        updates["next_step"] = "transfer_synthesizer"
        return updates

    def process_query(q: str, prompt_hint: str) -> dict[str, Any]:
        if not q:
            return {
                "summary": "",
                "payload": {},
                "retrieval": {"status": "skipped", "reason": "empty_query", "data": []},
            }
        rows = search_web_query_tool(query=q, max_results=10, timelimit="m")
        is_player_prompt = "player" in prompt_hint.lower()
        summary_result = summarize_payload_tool(
            summary_prompt=prompt_hint,
            payload=rows.get("data") or [],
            role="transfer_player" if is_player_prompt else "transfer_team",
            entity_kind="player" if is_player_prompt else "team",
            target_name=str(state.get("target_player_name") or ""),
            target_team=str(state.get("target_team") or ""),
            entity_context=_player_summary_context(state, transfer_mode=True)
            if is_player_prompt
            else _team_summary_context(state, transfer_mode=True),
        )
        return {
            "summary": str(summary_result.get("data") or "").strip(),
            "payload": summary_result,
            "retrieval": dict(rows or {}),
        }

    parallel_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_player = executor.submit(
            process_query, 
            player_query, 
            "Summarize transfer-portal player updates as a high-signal filter. Keep detailed facts on portal intent, eligibility remaining, NIL mentions, visits, injuries, scouting notes, and draft-stock movement when present. Remove legal/disclaimer bloat and noisy formatting. Use known player context (name, position, high school, prior colleges) to exclude unrelated players, teams, sports, and duplicates. Return 8-14 detailed bullets when evidence supports it."
        )
        f_team = executor.submit(
            process_query, 
            team_query, 
            "Summarize team transfer context as a high-signal filter. Keep detailed facts on roster needs, depth-chart competition, coaching/staff changes, positional battles, scholarship pressure, and spring standouts when present. Remove legal/disclaimer bloat and noisy formatting. Use known team context (college name and conference) to exclude unrelated programs and other sports. Use Wikipedia as grounding when present without overstating speculation. Return 6-12 detailed bullets when evidence supports it."
        )
        
        try:
            player_res = f_player.result(timeout=25)
        except Exception as e:
            player_res = {
                "summary": f"Player web search failed: {e}",
                "payload": {},
                "retrieval": {"status": "error", "reason": str(e), "data": []},
            }
            
        try:
            team_res = f_team.result(timeout=25)
        except Exception as e:
            team_res = {
                "summary": f"Team web search failed: {e}",
                "payload": {},
                "retrieval": {"status": "error", "reason": str(e), "data": []},
            }

    parallel_latency_ms = int((time.perf_counter() - parallel_started) * 1000)

    updates["transfer_web_player_summary"] = player_res["summary"]
    updates["transfer_web_team_summary"] = team_res["summary"]
    updates["transfer_web_player_retrieval"] = dict(player_res.get("retrieval") or {})
    updates["transfer_web_team_retrieval"] = dict(team_res.get("retrieval") or {})
    updates["transfer_web_player_used"] = bool(player_query)
    updates["transfer_web_team_used"] = bool(team_query)

    # Gather telemetry from summarizations if any
    summaries_tel = []
    from .orchestration_service import _extract_tool_telemetry
    for r in [player_res, team_res]:
        tel = _extract_tool_telemetry(r["payload"])
        if tel: summaries_tel.append(tel)
        
    if summaries_tel:
        current_telemetry = state.get("telemetry") or {}
        existing_models = current_telemetry.get("model_telemetry") or []
        updates["telemetry"] = {"model_telemetry": list(existing_models) + summaries_tel}

    traces.append({
        "node": "transfer_web_scout",
        "status": "parallel_fetch_complete",
        "latency_ms": parallel_latency_ms
    })
    updates["trace_log"] = traces
    updates["next_step"] = "transfer_synthesizer"
    return updates


def transfer_synthesizer_node(state: ScoutState) -> ScoutState:
    if bool(state.get("security_halt")):
        return {
            "trace_log": [_trace_entry(state, "transfer_synthesizer", "skipped")],
            "final_report": state.get("security_message", "Halted."),
        }

    from .orchestration_service import _build_transfer_synthesis_prompt, _merge_text_blocks, _extract_tool_telemetry
    
    context = dict(state.get("transfer_report_context") or {})
    player_name = str(context.get("player_name") or state.get("player_name") or "Unknown")
    target_team = str(context.get("target_team") or state.get("target_team") or "")
    user_query = str(state.get("user_query") or "").strip()

    web_player = str(state.get("transfer_web_player_summary") or "")
    web_team = str(state.get("transfer_web_team_summary") or "")
    merged_web = _merge_text_blocks(web_player, web_team)
    
    player_news = _merge_text_blocks(str(context.get("player_news_summary") or ""), merged_web)

    prompt = _build_transfer_synthesis_prompt(
        player_name=player_name,
        target_team=target_team,
        player_row=dict(context.get("college_player") or {}),
        cfbd_usage_2025=dict(context.get("cfbd_usage_2025") or {}),
        cfbd_stats_2025=dict(context.get("cfbd_stats_2025") or {}),
        cfbd_usage_career=list(context.get("cfbd_usage_career") or []),
        cfbd_stats_career=list(context.get("cfbd_stats_career") or []),
        usage_table_compact=list(context.get("usage_table_compact") or []),
        usage_yoy_compact=list(context.get("usage_yoy_compact") or []),
        season_stats_table_compact=list(context.get("season_stats_table_compact") or []),
        career_context=dict(context.get("career_context") or {}),
        player_news_summary=player_news,
        team_news_summary=str(context.get("team_news_summary") or ""),
        exclude_garbage_time=bool(context.get("exclude_garbage_time", True)),
        branch_status=dict(context.get("branch_status") or {}),
        follow_up_question=user_query,
    )
    
    synthesis_started = time.perf_counter()
    result = final_synthesis_tool(prompt)
    latency = int((time.perf_counter() - synthesis_started) * 1000)
    
    model_telemetry = _extract_tool_telemetry(result)
    
    answer_text = str(result.get("data") or "").strip() or "No response generated."

    updates: dict[str, Any] = {
        "final_report": answer_text,
        "next_step": "end",
    }
    
    if model_telemetry:
        current_telemetry = state.get("telemetry") or {}
        existing_models = current_telemetry.get("model_telemetry") or []
        updates["telemetry"] = {"model_telemetry": list(existing_models) + [model_telemetry]}

    updates["trace_log"] = [_trace_entry(state, "transfer_synthesizer", f"success_latency_{latency}ms")]
    
    return updates

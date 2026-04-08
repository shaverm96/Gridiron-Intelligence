from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from pydantic import ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import CONFIG, TABLES
from .cfbd_service import (
    fetch_player_season_stats,
    fetch_player_stats,
    fetch_player_usage,
    fetch_recruits,
    fetch_team_roster,
    search_player_candidates,
)
from .state import DelegatorPlan
from .supabase_client import (
    fetch_player_bundle,
    fetch_player_bundle_by_identity,
    get_supabase_client,
    query_vector_factoids,
    resolve_player_identity,
)

try:
    from ddgs import DDGS
except Exception:  # pragma: no cover
    DDGS = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None

try:
    from sklearn.preprocessing import MinMaxScaler
except Exception:  # pragma: no cover
    MinMaxScaler = None


EMBED_MODEL = None
logger = logging.getLogger(__name__)
MODEL_ALIAS_MAP = {
    "gemini-3.0-flash": "gemini-3-flash-preview",
    "gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
}


class DelegatorOutputValidationError(Exception):
    """Raised when LLM delegator output fails strict schema validation."""
POS_MAP = {
    "CB": "DB",
    "S": "DB",
    "FS": "DB",
    "SS": "DB",
    "DB": "DB",
    "DE": "EDGE",
    "EDGE": "EDGE",
    "DT": "IDL",
    "NT": "IDL",
    "DL": "IDL",
    "LB": "LB",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",
    "OL": "OL",
    "OT": "OL",
    "OG": "OL",
    "C": "OL",
    "QB": "QB",
    "RB": "RB",
    "HB": "RB",
    "FB": "RB",
    "K": "SPEC",
    "P": "SPEC",
    "PK": "SPEC",
    "LS": "SPEC",
    "RET": "SPEC",
    "SPEC": "SPEC",
    "TE": "TE",
    "WR": "WR",
}


def _get_llm(model_name: str, temperature: float, max_output_tokens: int):
    if ChatGoogleGenerativeAI is None or not CONFIG["GEMINI_API_KEY"]:
        return None
    resolved_model = MODEL_ALIAS_MAP.get(str(model_name or "").strip(), str(model_name or "").strip())
    return ChatGoogleGenerativeAI(
        model=resolved_model,
        google_api_key=CONFIG["GEMINI_API_KEY"],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )


def _llm_response_to_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                chunks.append(str(item["text"]))
            else:
                chunks.append(str(item))
        return "\n".join(chunks).strip()
    return str(content).strip()


def sanitize_model_summary_text(text: str) -> str:
    sanitized = str(text or "")
    sanitized = re.sub(r"```(?:json|javascript|html)?[\s\S]*?```", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"<iframe\b[^>]*>[\s\S]*?</iframe>", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"<a\b[^>]*>[\s\S]*?</a>", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"<[^>]+>", "", sanitized)

    lines: list[str] = []
    for line in sanitized.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("{", "}", "[", "]")):
            continue
        if re.match(r'^"[^"]+"\s*:\s*', stripped):
            continue
        lines.append(stripped)

    return "\n".join(lines).strip()


def _truncate_text(value: str, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[truncated {len(text) - max_chars} chars]"


def _payload_to_text(payload: Any, max_chars: int | None = None) -> str:
    payload_text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str)
    cap = int(max_chars if max_chars is not None else CONFIG.get("PROMPT_PAYLOAD_MAX_CHARS", 12000))
    return _truncate_text(payload_text, max_chars=cap)


def _log_prompt_size(label: str, prompt_text: str) -> None:
    try:
        logger.info("prompt_size label=%s chars=%s", label, len(str(prompt_text or "")))
    except Exception:
        pass


def _with_current_date_context(prompt_text: str) -> str:
    today_iso = date.today().isoformat()
    date_context = (
        "Date Context:\n"
        f"- Current date: {today_iso}\n"
        "- Treat this as today's date when reasoning about recency and up-to-date information.\n"
        "- If recency is uncertain, state the uncertainty explicitly.\n\n"
    )
    return f"{date_context}{str(prompt_text or '').strip()}"


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
    reraise=True,
)
def _ddgs_text_search(query: str, max_results: int) -> list[dict[str, Any]]:
    if DDGS is None:
        return []
    rows: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        for result in ddgs.text(str(query or ""), max_results=max_results):
            rows.append(result)
    return rows


def normalize_position_group(position_value: str | None) -> str:
    raw = str(position_value or "").strip().upper()
    return POS_MAP.get(raw, raw)


def fetch_player_bundle_tool(recruit_id: str) -> dict[str, Any]:
    return fetch_player_bundle(recruit_id)


def fetch_player_bundle_by_identity_tool(
    recruit_id: str | None = None,
    cfbd_athlete_id: str | None = None,
    name_query: str | None = None,
    year: int | None = None,
    position: str | None = None,
    team: str | None = None,
) -> dict[str, Any]:
    return fetch_player_bundle_by_identity(
        recruit_id=recruit_id,
        cfbd_athlete_id=cfbd_athlete_id,
        name_query=name_query,
        year=year,
        position=position,
        team=team,
    )


def resolve_player_identity_tool(
    name_query: str,
    year: int | None = None,
    position: str | None = None,
    team: str | None = None,
) -> dict[str, Any]:
    return resolve_player_identity(name_query, year=year, position=position, team=team)


def delegator_plan_tool(user_query: str, target_team: str = "", target_player_name: str = "") -> dict[str, Any]:
    llm = _get_llm(CONFIG["SUMMARY_MODEL"], temperature=0.0, max_output_tokens=500)
    if llm is None:
        fallback_player = target_player_name or ""
        fallback_team = target_team or ""
        return DelegatorPlan(
            cfbd_search_params={
                "name": fallback_player,
                "college_team": fallback_team,
                "position": "",
            },
            recruiting_web_query=f"{fallback_player} recruiting scouting report".strip(),
            team_context_query=f"{fallback_team} depth chart roster".strip(),
            user_intent=(user_query or "Generate a scouting report.")[:220],
        ).model_dump()

    try:
        structured = llm.with_structured_output(DelegatorPlan)
    except Exception as exc:
        raise DelegatorOutputValidationError(f"Delegator structured output setup failed: {exc}") from exc

    prompt = (
        "Create a delegator plan for a college football scouting workflow. "
        "Infer likely player/team context from the user request. "
        "Return concise search params and queries.\n\n"
        f"User query: {user_query}\n"
        f"Known target team: {target_team}\n"
        f"Known target player: {target_player_name}\n"
    )
    try:
        plan = structured.invoke(_with_current_date_context(prompt))
        if isinstance(plan, DelegatorPlan):
            return plan.model_dump()
        if hasattr(plan, "model_dump"):
            return plan.model_dump()
        if isinstance(plan, dict):
            return DelegatorPlan(**plan).model_dump()
        raise DelegatorOutputValidationError("Delegator returned an unexpected output type.")
    except ValidationError as exc:
        raise DelegatorOutputValidationError(f"Delegator validation failed: {exc}") from exc
    except DelegatorOutputValidationError:
        raise
    except Exception as exc:
        raise DelegatorOutputValidationError(f"Delegator invoke failed: {exc}") from exc

    return DelegatorPlan(
        cfbd_search_params={
            "name": target_player_name or "",
            "college_team": target_team or "",
            "position": "",
        },
        recruiting_web_query=f"{target_player_name} recruiting scouting report".strip(),
        team_context_query=f"{target_team} depth chart roster".strip(),
        user_intent=(user_query or "Generate a scouting report.")[:220],
    ).model_dump()


def search_web_tool(
    player_name: str,
    position: str,
    high_school: str,
    year: int,
    max_results: int = 12,
) -> dict[str, Any]:
    if DDGS is None:
        return {"status": "skipped", "reason": "DDGS not installed", "data": [], "citations": []}

    query = (
        f"{player_name} {position} {high_school} {year} football recruiting "
        f"(site:maxpreps.com OR site:247sports.com OR site:rivals.com OR site:espn.com OR site:on3.com)"
    )

    rows: list[dict[str, str]] = []
    citations: list[dict[str, str]] = []

    try:
        results = _ddgs_text_search(query, max_results=max_results)
        for result in results:
            url = result.get("href", "") or ""
            if not any(site in url for site in CONFIG["TARGET_SEARCH_SITES"]):
                continue
            row = {
                "title": result.get("title", ""),
                "url": url,
                "snippet": result.get("body", ""),
            }
            rows.append(row)
            citations.append(
                {
                    "source_type": "web",
                    "source_name": row["title"] or "DDGS result",
                    "source_url": row["url"],
                }
            )
    except Exception as exc:
        return {"status": "skipped", "reason": f"DDGS search failed: {exc}", "data": [], "citations": []}

    return {"status": "ok", "reason": "search complete", "data": rows, "citations": citations}


def search_web_query_tool(query: str, max_results: int | None = None) -> dict[str, Any]:
    if DDGS is None:
        return {"status": "skipped", "reason": "DDGS not installed", "data": [], "citations": []}

    effective_max_results = int(max_results if max_results is not None else CONFIG.get("WEB_QUERY_MAX_RESULTS", 6))

    rows: list[dict[str, str]] = []
    citations: list[dict[str, str]] = []
    try:
        results = _ddgs_text_search(str(query or ""), max_results=effective_max_results)
        for result in results:
            row = {
                "title": str(result.get("title") or ""),
                "url": str(result.get("href") or ""),
                "snippet": str(result.get("body") or ""),
            }
            rows.append(row)
            citations.append(
                {
                    "source_type": "web",
                    "source_name": row["title"] or "DDGS result",
                    "source_url": row["url"],
                }
            )
        return {"status": "ok", "reason": "search complete", "data": rows, "citations": citations}
    except Exception as exc:
        return {"status": "skipped", "reason": f"DDGS search failed: {exc}", "data": [], "citations": []}


def summarize_web_tool(player_name: str, position: str, search_rows: list[dict[str, str]]) -> dict[str, Any]:
    if not search_rows:
        return {
            "status": "ok",
            "reason": "no rows",
            "data": "No relevant web articles were found from target recruiting sites.",
            "citations": [],
        }

    llm = _get_llm(CONFIG["SUMMARY_MODEL"], temperature=0.0, max_output_tokens=1200)
    if llm is None:
        return {
            "status": "skipped",
            "reason": "Gemini summary model unavailable",
            "data": "Gemini summary skipped: API key/model client not configured.",
            "citations": [],
        }

    max_sources = max(1, min(int(CONFIG.get("WEB_QUERY_MAX_RESULTS", 6)), 10))
    context_chunks = []
    for idx, row in enumerate(search_rows[:max_sources], start=1):
        context_chunks.append(
            f"[{idx}] Title: {row.get('title', '')}\nURL: {row.get('url', '')}\nSnippet: {row.get('snippet', '')}"
        )
    sources_block = _truncate_text(
        "\n".join(context_chunks),
        max_chars=int(CONFIG.get("PROMPT_PAYLOAD_MAX_CHARS", 12000)),
    )

    prompt = (
        f"You are a recruiting research assistant. Summarize recent web intelligence for {player_name} ({position}).\n"
        "Only use provided sources. Do not invent facts.\n\n"
        "Output:\n"
        "1) Key facts\n2) Recruiting updates\n3) Source list\n\n"
        f"Sources:\n{sources_block}"
    )

    prompt_with_date = _with_current_date_context(prompt)
    _log_prompt_size("summarize_web_tool", prompt_with_date)
    response = llm.invoke(prompt_with_date)
    cleaned = sanitize_model_summary_text(_llm_response_to_text(response))
    return {
        "status": "ok",
        "reason": "summary complete",
        "data": cleaned,
        "citations": [
            {"source_type": "model", "source_name": CONFIG["SUMMARY_MODEL"], "source_url": ""}
        ],
    }


def summarize_payload_tool(summary_prompt: str, payload: Any) -> dict[str, Any]:
    llm = _get_llm(CONFIG["SUMMARY_MODEL"], temperature=0.0, max_output_tokens=1200)
    if llm is None:
        return {
            "status": "skipped",
            "reason": "Gemini summary model unavailable",
            "data": "Summary unavailable: Gemini summary model is not configured.",
            "citations": [],
        }

    payload_text = _payload_to_text(payload)
    full_prompt = f"{summary_prompt}\n\nPayload:\n{payload_text}"
    prompt_with_date = _with_current_date_context(full_prompt)
    _log_prompt_size("summarize_payload_tool", prompt_with_date)
    response = llm.invoke(prompt_with_date)
    cleaned = sanitize_model_summary_text(_llm_response_to_text(response))
    return {
        "status": "ok",
        "reason": "summary complete",
        "data": cleaned,
        "citations": [{"source_type": "model", "source_name": CONFIG["SUMMARY_MODEL"], "source_url": ""}],
    }


def cfbd_fetch_tool(
    athlete_id: str | None = None,
    team: str | None = None,
    year: int | None = None,
    position: str | None = None,
    conference: str | None = None,
    start_week: int | None = None,
    end_week: int | None = None,
    season_type: str | None = None,
    category: str | None = None,
    state: str | None = None,
    classification: str | None = None,
    player_id: int | None = None,
    exclude_garbage_time: bool | None = None,
    search_term: str | None = None,
    endpoint: str = "player/usage",
) -> dict[str, Any]:
    endpoint_name = endpoint.strip().lower()
    if endpoint_name == "player/stats":
        # Legacy alias kept for compatibility; mapped via service shim.
        return fetch_player_stats(athlete_id=athlete_id, team=team, year=year)
    if endpoint_name == "roster":
        return fetch_team_roster(team=team or "", year=year, classification=classification)
    if endpoint_name == "player/search":
        return search_player_candidates(search_term=search_term or "", year=year, team=team, position=position)
    if endpoint_name == "recruiting":
        return fetch_recruits(
            year=year,
            team=team,
            position=position,
            state=state,
            classification=classification,
        )
    if endpoint_name == "stats/player/season":
        return fetch_player_season_stats(
            year=year,
            conference=conference,
            team=team,
            start_week=start_week,
            end_week=end_week,
            season_type=season_type,
            category=category,
        )
    if endpoint_name == "player/usage":
        effective_player_id = player_id
        if effective_player_id is None and athlete_id and str(athlete_id).strip().isdigit():
            effective_player_id = int(str(athlete_id).strip())
        return fetch_player_usage(
            year=year,
            conference=conference,
            position=position,
            team=team,
            player_id=effective_player_id,
            exclude_garbage_time=exclude_garbage_time,
        )

    # Keep a generic fallback for compatibility with any future endpoint callers.
    params: dict[str, Any] = {}
    if athlete_id:
        params["athleteId"] = str(athlete_id)
    if team:
        params["team"] = str(team)
    if year:
        params["year"] = int(year)
    if position:
        params["position"] = str(position)
    if conference:
        params["conference"] = str(conference)
    if start_week is not None:
        params["startWeek"] = int(start_week)
    if end_week is not None:
        params["endWeek"] = int(end_week)
    if season_type:
        params["seasonType"] = str(season_type)
    if category:
        params["category"] = str(category)
    if state:
        params["state"] = str(state)
    if classification:
        params["classification"] = str(classification)
    if player_id is not None:
        params["playerId"] = int(player_id)
    if exclude_garbage_time is not None:
        params["excludeGarbageTime"] = bool(exclude_garbage_time)
    if search_term:
        params["searchTerm"] = str(search_term)
    from .cfbd_service import cfbd_fetch

    return cfbd_fetch(endpoint=endpoint, params=params)


def cfbd_search_players_tool(
    search_term: str,
    year: int | None = None,
    team: str | None = None,
    position: str | None = None,
) -> dict[str, Any]:
    return search_player_candidates(
        search_term=search_term,
        year=year,
        team=team,
        position=position,
    )


def cfbd_roster_tool(team: str, year: int | None = None) -> dict[str, Any]:
    return fetch_team_roster(team=team, year=year)


def cfbd_recruits_tool(
    year: int | None = None,
    team: str | None = None,
    position: str | None = None,
    state: str | None = None,
    classification: str | None = None,
) -> dict[str, Any]:
    return fetch_recruits(
        year=year,
        team=team,
        position=position,
        state=state,
        classification=classification,
    )


def cfbd_player_season_stats_tool(
    year: int | None,
    conference: str | None = None,
    team: str | None = None,
    start_week: int | None = None,
    end_week: int | None = None,
    season_type: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    return fetch_player_season_stats(
        year=year,
        conference=conference,
        team=team,
        start_week=start_week,
        end_week=end_week,
        season_type=season_type,
        category=category,
    )


def cfbd_player_usage_tool(
    year: int | None,
    conference: str | None = None,
    position: str | None = None,
    team: str | None = None,
    player_id: int | None = None,
    exclude_garbage_time: bool | None = None,
) -> dict[str, Any]:
    return fetch_player_usage(
        year=year,
        conference=conference,
        position=position,
        team=team,
        player_id=player_id,
        exclude_garbage_time=exclude_garbage_time,
    )


def _get_embedding_model():
    global EMBED_MODEL
    if EMBED_MODEL is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return EMBED_MODEL


def vector_insights_tool(
    query_text: str,
    position: str,
    state: str | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    if not position:
        return {"status": "skipped", "reason": "Missing player position", "data": [], "citations": []}

    try:
        model = _get_embedding_model()
    except Exception as exc:
        return {"status": "skipped", "reason": str(exc), "data": [], "citations": []}

    embedding = model.encode([query_text], normalize_embeddings=True)[0].tolist()
    rpc_result = query_vector_factoids(
        query_embedding=embedding,
        filter_position=normalize_position_group(position),
        filter_state=state,
        threshold=float(threshold if threshold is not None else CONFIG["VECTOR_MATCH_THRESHOLD"]),
        top_k=int(top_k if top_k is not None else CONFIG["VECTOR_MATCH_COUNT"]),
    )

    insights: list[str] = []
    citations: list[dict[str, str]] = []
    for row in rpc_result.get("data", []):
        text = row.get("factoid_text") or ""
        similarity = row.get("similarity")
        if not text:
            continue
        if similarity is None:
            insights.append(text)
        else:
            insights.append(f"[sim={float(similarity):.3f}] {text}")
        citations.append(
            {
                "source_type": "sql-rpc",
                "source_name": CONFIG["VECTOR_RPC_NAME"],
                "source_url": "",
            }
        )

    return {
        "status": rpc_result.get("status", "ok"),
        "reason": rpc_result.get("reason", "vector complete"),
        "data": insights,
        "citations": citations,
    }


def historical_comparables_tool(recruit_id: str) -> dict[str, Any]:
    sb = get_supabase_client()
    if sb is None:
        return {
            "status": "skipped",
            "reason": "Supabase client unavailable",
            "data": "Historical comparables unavailable: Supabase client is not configured.",
            "citations": [],
        }
    if MinMaxScaler is None:
        return {
            "status": "skipped",
            "reason": "scikit-learn unavailable",
            "data": "Historical comparables unavailable: scikit-learn is not installed.",
            "citations": [],
        }

    rid = str(recruit_id).strip()
    target_rows = (
        sb.table(TABLES["player_master"])
        .select("recruit_id, player_name, year, position, rating, height_inches, weight_lbs, state")
        .eq("recruit_id", rid)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not target_rows:
        return {
            "status": "ok",
            "reason": "no target row",
            "data": f"Historical comparables unavailable: recruit_id {rid} not found.",
            "citations": [],
        }

    target = target_rows[0]
    target_pos = str(target.get("position") or "").strip()
    if not target_pos:
        return {
            "status": "ok",
            "reason": "missing position",
            "data": "Historical comparables unavailable: target player has no position.",
            "citations": [],
        }

    pool_rows = (
        sb.table(TABLES["player_master"])
        .select("recruit_id, player_name, year, position, rating, height_inches, weight_lbs, state")
        .eq("position", target_pos)
        .lte("year", 2022)
        .execute()
        .data
        or []
    )
    if not pool_rows:
        return {
            "status": "ok",
            "reason": "no pool rows",
            "data": f"No historical comparables found for position {target_pos}.",
            "citations": [],
        }

    df_pool = pd.DataFrame(pool_rows)
    df_pool = df_pool[df_pool["recruit_id"].astype(str) != rid].copy()
    if df_pool.empty:
        return {
            "status": "ok",
            "reason": "empty pool after exclusion",
            "data": "No historical comparables found after filtering.",
            "citations": [],
        }

    df = pd.concat([pd.DataFrame([target]), df_pool], ignore_index=True)

    def _calc_sim(cols: list[str]) -> np.ndarray:
        numeric_df = df[cols].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.fillna(numeric_df.mean()).fillna(0.0)
        scaled = MinMaxScaler().fit_transform(numeric_df)
        dists = np.linalg.norm(scaled - scaled[0], axis=1)
        max_dist = np.sqrt(float(len(cols)))
        if max_dist == 0:
            return np.ones(len(df), dtype=float)
        return np.clip(1.0 - (dists / max_dist), 0.0, 1.0)

    rating_sim = _calc_sim(["rating"])
    size_sim = _calc_sim(["height_inches", "weight_lbs"])
    target_state = str(target.get("state") or "").strip().upper()
    if target_state:
        state_sim = (df["state"].astype(str).str.strip().str.upper() == target_state).astype(float).to_numpy()
    else:
        state_sim = np.zeros(len(df), dtype=float)

    final_sim = (rating_sim * 3.0 + size_sim * 2.5 + state_sim * 0.5) / 6.0
    df["similarity"] = np.round(final_sim * 100.0, 2)

    comps = df.drop(index=0).sort_values("similarity", ascending=False).head(5).copy()
    if comps.empty:
        return {
            "status": "ok",
            "reason": "no comparable rows",
            "data": "No historical comparables found after similarity scoring.",
            "citations": [],
        }

    lines = [f"### Historical Comparables for {target.get('player_name', rid)}", "---"]
    for _, row in comps.iterrows():
        lines.append(
            f"- **{row.get('player_name', 'Unknown')}** ({row.get('year', 'N/A')}, {row.get('state', 'N/A')}) | "
            f"Match: {row.get('similarity', 'N/A')}% | Rating: {row.get('rating', 'N/A')}"
        )

    return {
        "status": "ok",
        "reason": "comparables generated",
        "data": "\n".join(lines),
        "citations": [
            {"source_type": "sql", "source_name": TABLES["player_master"], "source_url": ""},
            {"source_type": "sql", "source_name": TABLES["pred_score"], "source_url": ""},
        ],
    }


def build_synthesis_prompt(
    state_bundle: dict[str, Any],
    web_summary: str,
    vector_factoids: list[str],
    comparables_md: str,
    target_team: str,
    year: int,
    user_query: str,
) -> str:
    vector_text = "\n".join([f"- {x}" for x in vector_factoids]) if vector_factoids else "No vector insights returned."
    state_bundle_text = _payload_to_text(state_bundle)
    web_summary_text = _truncate_text(str(web_summary or ""), max_chars=4000)
    vector_text = _truncate_text(vector_text, max_chars=3000)
    comparables_text = _truncate_text(str(comparables_md or ""), max_chars=3000)
    return (
        "You are a senior college football recruiting scout.\n\n"
        "Generate a structured, grounded scouting answer using only the context below.\n\n"
        f"User Query: {user_query}\n"
        f"Target Team: {target_team}\n"
        f"Year: {year}\n\n"
        f"Player Bundle JSON:\n{state_bundle_text}\n\n"
        f"Web Summary:\n{web_summary_text}\n\n"
        f"Vector Insights:\n{vector_text}\n\n"
        f"Historical Comparables:\n{comparables_text}\n\n"
        "Output:\n"
        "1) Player Snapshot\n2) Trait Evaluation\n3) Fit Recommendation\n4) Final Recommendation with confidence\n"
        "Cite key evidence inline when possible."
    )


def final_synthesis_tool(prompt: str) -> dict[str, Any]:
    llm = _get_llm(CONFIG["FINAL_MODEL"], temperature=0.25, max_output_tokens=2200)
    if llm is None:
        return {
            "status": "skipped",
            "reason": "Gemini final model unavailable",
            "data": "Final synthesis skipped: Gemini client not configured.",
            "citations": [],
        }

    safe_prompt = _truncate_text(prompt, max_chars=int(CONFIG.get("FINAL_PROMPT_MAX_CHARS", 20000)))
    prompt_with_date = _with_current_date_context(safe_prompt)
    _log_prompt_size("final_synthesis_tool", prompt_with_date)
    response = llm.invoke(prompt_with_date)
    return {
        "status": "ok",
        "reason": "final synthesis complete",
        "data": _llm_response_to_text(response),
        "citations": [{"source_type": "model", "source_name": CONFIG["FINAL_MODEL"], "source_url": ""}],
    }

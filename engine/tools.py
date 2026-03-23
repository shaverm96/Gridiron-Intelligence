from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .config import CONFIG, TABLES
from .supabase_client import fetch_player_bundle, get_supabase_client, query_vector_factoids

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
    return ChatGoogleGenerativeAI(
        model=model_name,
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


def normalize_position_group(position_value: str | None) -> str:
    raw = str(position_value or "").strip().upper()
    return POS_MAP.get(raw, raw)


def fetch_player_bundle_tool(recruit_id: str) -> dict[str, Any]:
    return fetch_player_bundle(recruit_id)


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

    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
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

    return {"status": "ok", "reason": "search complete", "data": rows, "citations": citations}


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

    context_chunks = []
    for idx, row in enumerate(search_rows[:10], start=1):
        context_chunks.append(
            f"[{idx}] Title: {row.get('title', '')}\nURL: {row.get('url', '')}\nSnippet: {row.get('snippet', '')}"
        )

    prompt = (
        f"You are a recruiting research assistant. Summarize recent web intelligence for {player_name} ({position}).\n"
        "Only use provided sources. Do not invent facts.\n\n"
        "Output:\n"
        "1) Key facts\n2) Recruiting updates\n3) Source list\n\n"
        f"Sources:\n{chr(10).join(context_chunks)}"
    )

    response = llm.invoke(prompt)
    return {
        "status": "ok",
        "reason": "summary complete",
        "data": _llm_response_to_text(response),
        "citations": [
            {"source_type": "model", "source_name": CONFIG["SUMMARY_MODEL"], "source_url": ""}
        ],
    }


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
    return (
        "You are a senior college football recruiting scout.\n\n"
        "Generate a structured, grounded scouting answer using only the context below.\n\n"
        f"User Query: {user_query}\n"
        f"Target Team: {target_team}\n"
        f"Year: {year}\n\n"
        f"Player Bundle JSON:\n{json.dumps(state_bundle, indent=2, default=str)}\n\n"
        f"Web Summary:\n{web_summary}\n\n"
        f"Vector Insights:\n{vector_text}\n\n"
        f"Historical Comparables:\n{comparables_md}\n\n"
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

    response = llm.invoke(prompt)
    return {
        "status": "ok",
        "reason": "final synthesis complete",
        "data": _llm_response_to_text(response),
        "citations": [{"source_type": "model", "source_name": CONFIG["FINAL_MODEL"], "source_url": ""}],
    }

from __future__ import annotations

import ast
import html
import json
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st

from engine import (
    get_scout_graph,
    get_structured_web_graph,
    orchestrate_chat_turn,
    orchestrate_structured_web_scouting,
)
from engine.state import initial_chat_state

from engine.comparables_service import (
    get_historical_player_comparables_data,
)
from engine.data_access import (
    fetch_player_bundle_data,
)
from engine.data_transforms import (
    build_player_profile_view_data,
    build_score_card_html_data,
    clean_scouting_profile_data,
    merge_scouting_sources_data,
)
from engine.synthesis_service import (
    build_final_prompt_data,
    run_final_synthesis_data,
)
from engine.vector_service import (
    vector_insights_query_data,
)
from engine.web_research_service import (
    duckduckgo_search_data,
    summarize_web_with_flash_lite_data,
)
from engine.tools import (
    cfbd_fetch_tool,
    cfbd_search_players_tool,
    delegator_plan_tool,
    resolve_player_identity_tool,
)

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from supabase import create_client
except ImportError:
    create_client = None


def _cfg_with_source(key: str, default: str = "") -> tuple[str, str]:
    sensitive_keys = {
        "SUPABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "GEMINI_API_KEY",
        "CFBD_API_KEY",
        "CFBD_API",
    }
    require_secrets = str(os.getenv("GI_REQUIRE_STREAMLIT_SECRETS", "")).strip().lower() in {
        "1",
        "true",
        "yes",
    }

    try:
        if key in st.secrets:
            value = st.secrets.get(key, default)
            return (str(value).strip() if value is not None else "", "streamlit_secrets")
    except Exception:
        pass

    if require_secrets and key in sensitive_keys:
        return default, "required_streamlit_secrets_missing"

    env_value = os.getenv(key)
    if env_value is not None:
        return env_value.strip(), "environment"
    return default, "default"


def _cfg(key: str, default: str = "") -> str:
    value, _ = _cfg_with_source(key, default)
    return value


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _cfg_flag_with_source(key: str, default: bool = False) -> tuple[bool, str]:
    try:
        if key in st.secrets:
            return _parse_bool(st.secrets.get(key), default), "streamlit_secrets"
    except Exception:
        pass

    env_value = os.getenv(key)
    if env_value is not None:
        return _parse_bool(env_value, default), "environment"
    return default, "default"


def resolve_project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for candidate in candidates:
        if (candidate / "data" / "modeling_datasets").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = resolve_project_root()

if load_dotenv is not None:
    for env_name in ("SECRETS.env", "SUPABASE.env", "GEMINI_API_KEY.env"):
        env_file = PROJECT_ROOT / env_name
        if env_file.exists():
            load_dotenv(env_file, override=False)

SUPABASE_URL, SUPABASE_URL_SOURCE = _cfg_with_source("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY, SUPABASE_SERVICE_ROLE_KEY_SOURCE = _cfg_with_source("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY, GEMINI_API_KEY_SOURCE = _cfg_with_source("GEMINI_API_KEY")
CFBD_API_KEY, CFBD_API_KEY_SOURCE = _cfg_with_source("CFBD_API_KEY")
if not CFBD_API_KEY:
    CFBD_API_KEY, CFBD_API_KEY_SOURCE = _cfg_with_source("CFBD_API")
LOCAL_CFBD_DEBUGGER_ENABLED, LOCAL_CFBD_DEBUGGER_SOURCE = _cfg_flag_with_source("GI_ENABLE_LOCAL_CFBD_DEBUGGER", default=False)

CONFIG = {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "CFBD_API_KEY": CFBD_API_KEY,
    "CFBD_BASE_URL": _cfg("CFBD_BASE_URL", "https://api.collegefootballdata.com"),
    "YEARS": [2026, 2027, 2028],
    "FINAL_MODEL": "gemini-3-flash-preview",
    "SUMMARY_MODEL": "gemini-3.1-flash-lite-preview",
    "VECTOR_MATCH_COUNT": 6,
    "VECTOR_MATCH_THRESHOLD": 0.15,
    "VECTOR_RPC_NAME": "match_gi_factoids",
    "LOCAL_CFBD_DEBUGGER_ENABLED": LOCAL_CFBD_DEBUGGER_ENABLED,
}

CHAT_STATE_MAX_TURNS = 6
CHAT_STATE_MAX_TRACE = 10
CHAT_STATE_MAX_ERRORS = 6
CHAT_STATE_MAX_CITATIONS = 16
CHAT_STATE_MAX_CANDIDATES = 3
STRUCTURED_REPORT_RATE_LIMIT_COUNT = 3
STRUCTURED_REPORT_RATE_LIMIT_WINDOW_SECONDS = 60

CONFIG_SOURCES = {
    "SUPABASE_URL": SUPABASE_URL_SOURCE,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY_SOURCE,
    "GEMINI_API_KEY": GEMINI_API_KEY_SOURCE,
    "CFBD_API_KEY": CFBD_API_KEY_SOURCE,
    "GI_ENABLE_LOCAL_CFBD_DEBUGGER": LOCAL_CFBD_DEBUGGER_SOURCE,
}

TABLES = {
    "player_master": "gi_recruit_master",
    "scouting_features": "gi_scouting_report_features",
    "pred_score": "gi_model_prediction_score",
    "pred_threshold": "gi_model_prediction_thresholds",
}

TARGET_TEAMS = [
    "Alabama", "Auburn", "Clemson", "Colorado", "Duke", "Florida", "Florida State",
    "Georgia", "Georgia Tech", "LSU", "Miami", "Michigan", "NC State", "Notre Dame",
    "Ohio State", "Ole Miss", "Oregon", "South Carolina", "Tennessee", "Texas",
    "Texas A&M", "Charlotte", "USC", "Virginia Tech", "Wake Forest",
]

TARGET_SEARCH_SITES = ["maxpreps.com", "247sports.com", "rivals.com", "espn.com", "on3.com"]
POS_MAP = {"CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DE": "EDGE", "DT": "IDL", "NT": "IDL", "LB": "LB", "OLB": "LB", "ILB": "LB", "OL": "OL", "OT": "OL", "OG": "OL", "C": "OL", "QB": "QB", "PRO": "QB", "DUAL": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "K": "SPEC", "P": "SPEC", "PK": "SPEC", "LS": "SPEC", "RET": "SPEC", "TE": "TE", "WR": "WR"}

EMBED_MODEL = None
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_MODEL_LOAD_ERROR = None


def get_supabase_client():
    if create_client is None:
        return None
    if not CONFIG["SUPABASE_URL"] or not CONFIG["SUPABASE_SERVICE_ROLE_KEY"]:
        return None
    return create_client(CONFIG["SUPABASE_URL"], CONFIG["SUPABASE_SERVICE_ROLE_KEY"])


def get_supabase_config_issues() -> list[str]:
    issues = []
    if create_client is None:
        issues.append("Python package 'supabase' is not installed (or failed to import).")
    if not CONFIG["SUPABASE_URL"]:
        issues.append(f"SUPABASE_URL is missing (source: {CONFIG_SOURCES['SUPABASE_URL']}).")
    if not CONFIG["SUPABASE_SERVICE_ROLE_KEY"]:
        issues.append(
            "SUPABASE_SERVICE_ROLE_KEY is missing "
            f"(source: {CONFIG_SOURCES['SUPABASE_SERVICE_ROLE_KEY']})."
        )
    return issues


def get_gemini_config_issues() -> list[str]:
    issues = []
    if ChatGoogleGenerativeAI is None:
        issues.append("Python package 'langchain-google-genai' is not installed (or failed to import).")
    if not CONFIG["GEMINI_API_KEY"]:
        issues.append(f"GEMINI_API_KEY is missing (source: {CONFIG_SOURCES['GEMINI_API_KEY']}).")
    return issues


def run_one_click_diagnostics() -> dict:
    checks: list[dict[str, str]] = []

    def add_check(name: str, status: str, detail: str):
        checks.append({"name": name, "status": status, "detail": detail})

    add_check(
        "Config sources",
        "pass",
        (
            "SUPABASE_URL="
            f"{CONFIG_SOURCES['SUPABASE_URL']}, SUPABASE_SERVICE_ROLE_KEY={CONFIG_SOURCES['SUPABASE_SERVICE_ROLE_KEY']}, "
            f"GEMINI_API_KEY={CONFIG_SOURCES['GEMINI_API_KEY']}"
        ),
    )

    supabase_issues = get_supabase_config_issues()
    if supabase_issues:
        add_check("Supabase preflight", "fail", "; ".join(supabase_issues))
    else:
        try:
            sb = get_supabase_client()
            response = sb.table(TABLES["player_master"]).select("recruit_id").limit(1).execute()
            row_count = len(response.data or [])
            add_check("Supabase connectivity", "pass", f"Connected and queried {TABLES['player_master']} (rows returned: {row_count}).")
        except Exception as exc:
            add_check("Supabase connectivity", "fail", f"Query test failed: {exc}")

    gemini_issues = get_gemini_config_issues()
    if gemini_issues:
        add_check("Gemini preflight", "fail", "; ".join(gemini_issues))
    else:
        try:
            llm = get_llm(CONFIG["SUMMARY_MODEL"], temperature=0.0, max_output_tokens=20)
            if llm is None:
                add_check("Gemini connectivity", "fail", "Gemini client could not be created.")
            else:
                today_iso = date.today().isoformat()
                response = llm.invoke(
                    f"Date Context: Current date is {today_iso}. Reply with exactly: OK"
                )
                text = llm_response_to_text(response).strip()
                add_check("Gemini connectivity", "pass", f"Model responded: {text[:80] if text else 'empty response'}")
        except Exception as exc:
            add_check("Gemini connectivity", "fail", f"Invocation test failed: {exc}")

    overall = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {"overall": overall, "checks": checks}


def _normalize_model_name(model_name: str) -> str:
    alias_map = {
        "gemini-3.0-flash": "gemini-3-flash-preview",
        "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
    }
    value = str(model_name or "").strip()
    return alias_map.get(value, value)


def get_llm(model_name: str, temperature: float = 0.2, max_output_tokens: int = 1800):
    if ChatGoogleGenerativeAI is None or not CONFIG["GEMINI_API_KEY"]:
        return None
    resolved_model = _normalize_model_name(model_name)
    return ChatGoogleGenerativeAI(model=resolved_model, google_api_key=CONFIG["GEMINI_API_KEY"], temperature=temperature, max_output_tokens=max_output_tokens)


def get_embedding_model():
    global EMBED_MODEL, EMBED_MODEL_LOAD_ERROR
    if EMBED_MODEL is not None:
        return EMBED_MODEL
    if EMBED_MODEL_LOAD_ERROR is not None:
        raise RuntimeError(EMBED_MODEL_LOAD_ERROR)
    try:
        from sentence_transformers import SentenceTransformer

        EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
        return EMBED_MODEL
    except Exception as exc:
        EMBED_MODEL_LOAD_ERROR = f"Embedding model load failed: {exc}"
        raise RuntimeError(EMBED_MODEL_LOAD_ERROR)


def llm_response_to_text(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text") or item.get("output_text") or ""
                if txt:
                    parts.append(str(txt))
            else:
                txt = getattr(item, "text", None)
                if txt:
                    parts.append(str(txt))
        return "\n".join(parts) if parts else str(content)
    if isinstance(content, dict):
        txt = content.get("text") or content.get("output_text")
        if txt:
            return str(txt)
    return str(content)


def to_float_or_none(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def parse_jsonish(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan"}:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        try:
            parsed = ast.literal_eval(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}


def first_non_null(row: dict, candidates: list[str]):
    for key in candidates:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        return value
    return None


def normalize_position_group(position_value: str | None) -> str:
    raw = str(position_value or "").strip().upper()
    return POS_MAP.get(raw, raw)


def _supabase_fetch_all_rows(
    sb: Any,
    table_name: str,
    columns: str,
    batch_size: int = 1000,
    max_rows: int = 50000,
) -> list[dict[str, Any]]:
    if sb is None:
        return []

    rows: list[dict[str, Any]] = []
    start = 0
    while start < max_rows:
        end = start + batch_size - 1
        response = sb.table(table_name).select(columns).range(start, end).execute()
        chunk = list(response.data or [])
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < batch_size:
            break
        start += batch_size
    return rows


@st.cache_data
def load_model_tiers() -> pd.DataFrame:
    base_cols = ["Score Range", "Career Designation", "College Outlook", "Professional Outlook", "low", "high", "count"]
    sb = get_supabase_client()
    if sb is None:
        return pd.DataFrame(columns=base_cols)

    try:
        rows = _supabase_fetch_all_rows(
            sb=sb,
            table_name=TABLES["pred_score"],
            columns="predictive_score_0_100,contrib_tier_raw",
            batch_size=1000,
            max_rows=50000,
        )
    except Exception:
        return pd.DataFrame(columns=base_cols)

    if not rows:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(rows)
    if "predictive_score_0_100" not in df.columns:
        return pd.DataFrame(columns=base_cols)

    df["predictive_score_0_100"] = pd.to_numeric(df["predictive_score_0_100"], errors="coerce")
    df["contrib_tier_raw"] = df.get("contrib_tier_raw", "").fillna("Unknown").astype(str).str.strip()
    df.loc[df["contrib_tier_raw"] == "", "contrib_tier_raw"] = "Unknown"
    df = df[df["predictive_score_0_100"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=base_cols)

    grouped = (
        df.groupby("contrib_tier_raw", dropna=False)["predictive_score_0_100"]
        .agg(["min", "max", "count"])
        .reset_index()
        .rename(columns={"contrib_tier_raw": "Career Designation", "min": "low", "max": "high"})
    )
    grouped["Score Range"] = grouped.apply(lambda r: f"{r['low']:.2f}-{r['high']:.2f}", axis=1)
    grouped["College Outlook"] = "Derived from prediction score table"
    grouped["Professional Outlook"] = "Derived from prediction score table"
    return grouped[["Score Range", "Career Designation", "College Outlook", "Professional Outlook", "low", "high", "count"]].sort_values(["low", "high"]).reset_index(drop=True)


def score_tier(score: float | None) -> str:
    tiers = load_model_tiers()
    if score is None or tiers.empty:
        return "Unknown"
    for _, row in tiers.iterrows():
        low, high = to_float_or_none(row.get("low")), to_float_or_none(row.get("high"))
        if low is not None and high is not None and low <= score <= high:
            return str(row.get("Career Designation", "Unknown"))
    return "Unknown"


def tier_definitions_markdown() -> str:
    tiers = load_model_tiers()
    if tiers.empty:
        return "Tier definitions unavailable."
    return "\n".join([
        f"- **{r.get('Career Designation', '')}** ({r.get('Score Range', '')}): "
        f"Samples: {int(r.get('count', 0) or 0)}"
        for _, r in tiers.iterrows()
    ])


def merge_scouting_sources(scouting_row: dict) -> dict:
    return merge_scouting_sources_data(scouting_row=scouting_row, parse_jsonish=parse_jsonish)


def clean_scouting_profile(scouting_json: dict) -> dict:
    return clean_scouting_profile_data(scouting_json=scouting_json, to_float_or_none=to_float_or_none)


def build_player_profile_view(player_row: dict) -> dict:
    return build_player_profile_view_data(player_row=player_row, first_non_null=first_non_null)


@st.cache_data
def load_player_index() -> pd.DataFrame:
    base_cols = [
        "recruit_id",
        "player_name",
        "position",
        "high_school",
        "year",
        "rating",
        "player_label",
        "athlete_id",
    ]

    sb = get_supabase_client()
    if sb is None:
        raise RuntimeError("Supabase is not configured. Recruit dropdown requires gi_recruit_master.")

    select_cols = (
        "recruit_id,recruit_name,full_name,player_name,position_group,position,"
        "recruit_class,year,composite_rating,rating,high_school,hs_city,hs_state,"
        "athlete_id,cfbd_athlete_id"
    )
    rows = _supabase_fetch_all_rows(
        sb=sb,
        table_name=TABLES["player_master"],
        columns=select_cols,
        batch_size=1000,
        max_rows=50000,
    )
    if not rows:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(rows)
    if "recruit_id" not in df.columns:
        return pd.DataFrame(columns=base_cols)

    recruit_class_series = df["recruit_class"] if "recruit_class" in df.columns else pd.Series([None] * len(df), index=df.index)
    year_series = df["year"] if "year" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["year"] = pd.to_numeric(recruit_class_series.where(recruit_class_series.notna(), year_series), errors="coerce")
    year_filter = [int(y) for y in CONFIG["YEARS"]]
    df = df[df["year"].isin(year_filter)].copy()

    def _coalesce_text(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
        out = pd.Series(["" for _ in range(len(frame))], index=frame.index, dtype="object")
        for col in cols:
            if col in frame.columns:
                vals = frame[col].fillna("").astype(str).str.strip()
                out = out.where(out.astype(str).str.strip() != "", vals)
        return out

    df["recruit_id"] = df["recruit_id"].astype(str).str.strip()
    df["player_name"] = _coalesce_text(df, ["recruit_name", "full_name", "player_name"])
    df["position"] = _coalesce_text(df, ["position_group", "position"]).apply(normalize_position_group)
    df["high_school"] = _coalesce_text(df, ["high_school", "hs_city", "hs_state"])
    composite_rating_series = df["composite_rating"] if "composite_rating" in df.columns else pd.Series([None] * len(df), index=df.index)
    rating_series = df["rating"] if "rating" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["rating"] = pd.to_numeric(composite_rating_series.where(composite_rating_series.notna(), rating_series), errors="coerce")

    athlete_series = df.get("athlete_id") if "athlete_id" in df.columns else pd.Series([None] * len(df), index=df.index)
    cfbd_athlete_series = df.get("cfbd_athlete_id") if "cfbd_athlete_id" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["athlete_id"] = athlete_series.where(athlete_series.notna(), cfbd_athlete_series)
    df["athlete_id"] = pd.to_numeric(df["athlete_id"], errors="coerce").astype("Int64")

    df["player_label"] = (
        df["player_name"].astype(str)
        + " | "
        + df["position"].astype(str)
        + " | "
        + df["high_school"].astype(str)
        + " | "
        + df["year"].astype("Int64").astype(str)
    )

    df = df[df["recruit_id"].str.len() > 0].drop_duplicates(subset=["recruit_id"]).copy()
    df = df.sort_values(["year", "rating", "player_name"], ascending=[True, False, True]).reset_index(drop=True)
    return df[base_cols]


def fetch_player_bundle(sb, recruit_id: str) -> dict:
    return fetch_player_bundle_data(
        sb=sb,
        recruit_id=recruit_id,
        tables=TABLES,
        build_player_profile_view=build_player_profile_view,
        clean_scouting_profile=clean_scouting_profile,
        merge_scouting_sources=merge_scouting_sources,
    )


def duckduckgo_search(player_name: str, position: str, high_school: str, year: int, max_results: int = 12) -> list[dict]:
    return duckduckgo_search_data(
        ddgs_class=DDGS,
        player_name=player_name,
        position=position,
        high_school=high_school,
        year=year,
        target_search_sites=TARGET_SEARCH_SITES,
        max_results=max_results,
    )


def summarize_web_with_flash_lite(player_name: str, position: str, search_rows: list[dict]) -> str:
    return summarize_web_with_flash_lite_data(
        player_name=player_name,
        position=position,
        search_rows=search_rows,
        summary_model=CONFIG["SUMMARY_MODEL"],
        get_llm=get_llm,
        llm_response_to_text=llm_response_to_text,
    )


def vector_insights_query(sb, query_text: str, position: str | None = None, top_k: int = 6, threshold: float | None = None) -> dict:
    return vector_insights_query_data(
        sb=sb,
        query_text=query_text,
        position=position,
        top_k=top_k,
        threshold=threshold,
        vector_match_threshold=CONFIG["VECTOR_MATCH_THRESHOLD"],
        vector_rpc_name=CONFIG["VECTOR_RPC_NAME"],
        get_embedding_model=get_embedding_model,
        to_float_or_none=to_float_or_none,
    )


def get_historical_player_comparables(recruit_id: str) -> str:
    sb = get_supabase_client()
    if sb is None:
        return "Historical comparables unavailable: Supabase client is not configured."
    return get_historical_player_comparables_data(
        sb=sb,
        recruit_id=recruit_id,
        tables=TABLES,
        to_float_or_none=to_float_or_none,
        score_tier=score_tier,
    )


def build_score_card_html(pred_score: dict, pred_threshold: dict) -> str:
    return build_score_card_html_data(
        pred_score=pred_score,
        pred_threshold=pred_threshold,
        to_float_or_none=to_float_or_none,
        score_tier=score_tier,
    )


def build_final_prompt(year: int, target_team: str, player_row: dict, scouting_clean: dict, hs_athletic_background: str, pred_score_row: dict, pred_thr_row: dict, web_summary: str, vector_result: dict, historical_comparables_md: str) -> str:
    return build_final_prompt_data(
        year=year,
        target_team=target_team,
        persona=st.session_state.get("selected_persona", "Scout"),
        player_row=player_row,
        scouting_clean=scouting_clean,
        hs_athletic_background=hs_athletic_background,
        pred_score_row=pred_score_row,
        pred_thr_row=pred_thr_row,
        web_summary=web_summary,
        vector_result=vector_result,
        historical_comparables_md=historical_comparables_md,
        tier_definitions_markdown=tier_definitions_markdown,
    )


def run_final_synthesis(prompt: str) -> str:
    return run_final_synthesis_data(
        prompt=prompt,
        final_model=CONFIG["FINAL_MODEL"],
        get_llm=get_llm,
        llm_response_to_text=llm_response_to_text,
    )


def _is_local_debug_page_enabled() -> bool:
    force_enable = str(os.getenv("GI_ENABLE_LOCAL_DEBUG_PAGE", "")).strip().lower() in {"1", "true", "yes"}
    force_disable = str(os.getenv("GI_DISABLE_LOCAL_DEBUG_PAGE", "")).strip().lower() in {"1", "true", "yes"}
    if force_disable:
        return False
    if force_enable:
        return True

    cloud_markers = [
        os.getenv("STREAMLIT_SHARING_MODE", ""),
        os.getenv("STREAMLIT_CLOUD", ""),
        os.getenv("IS_STREAMLIT_CLOUD", ""),
    ]
    running_in_cloud = any(str(marker).strip() for marker in cloud_markers)
    has_local_env_files = any(
        (PROJECT_ROOT / name).exists()
        for name in ("SECRETS.env", "SUPABASE.env", "GEMINI_API_KEY.env")
    )
    return has_local_env_files and not running_in_cloud


def _build_cfbd_debug_url(meta: dict[str, Any]) -> str:
    endpoint = str(meta.get("endpoint") or "").strip().lstrip("/")
    params = meta.get("params") if isinstance(meta.get("params"), dict) else {}
    query = urlencode({k: v for k, v in params.items() if v is not None and str(v).strip() != ""})
    base = str(CONFIG.get("CFBD_BASE_URL") or "https://api.collegefootballdata.com").rstrip("/")
    return f"{base}/{endpoint}?{query}" if query else f"{base}/{endpoint}"


def render_local_cfbd_debugger_page() -> None:
    st.subheader("Local CFBD Agent Debugger")
    st.caption("Local-only debugger for delegator task detection, player identity matching, and CFBD query validation.")

    user_query = st.text_area(
        "User Query",
        value="Build a scouting report on Jeremiah Smith for Ohio State this season.",
        height=100,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        target_player_name = st.text_input("Target Player (hint)", value="Jeremiah Smith")
    with col2:
        target_team = st.text_input("Target Team (hint)", value="Ohio State")
    with col3:
        year = st.number_input("Year", min_value=2010, max_value=2035, value=2026, step=1)

    col4, col5 = st.columns(2)
    with col4:
        position_hint = st.text_input("Position (optional)", value="WR")
    with col5:
        athlete_id_override = st.text_input("CFBD Athlete ID override (optional)", value="")

    endpoint_mode = st.selectbox(
        "Endpoint to Test",
        [
            "auto",
            "player/usage",
            "roster",
            "player/search",
            "recruiting",
            "stats/player/season",
        ],
        index=0,
    )

    col6, col7, col8 = st.columns(3)
    with col6:
        conference = st.text_input("Conference (optional)", value="")
    with col7:
        start_week = st.number_input("Start Week (optional)", min_value=0, max_value=20, value=0, step=1)
    with col8:
        end_week = st.number_input("End Week (optional)", min_value=0, max_value=20, value=0, step=1)

    col9, col10, col11 = st.columns(3)
    with col9:
        season_type = st.selectbox("Season Type (optional)", ["", "regular", "postseason", "both"], index=0)
    with col10:
        category = st.text_input("Category (optional)", value="")
    with col11:
        state_filter = st.text_input("State (optional)", value="")

    col12, col13, col14 = st.columns(3)
    with col12:
        classification = st.selectbox("Classification (optional)", ["", "fbs", "fcs", "HighSchool", "Juco", "PrepSchool"], index=0)
    with col13:
        player_id_override = st.text_input("Player ID override (optional)", value="")
    with col14:
        exclude_garbage_time = st.checkbox("Exclude garbage time", value=False)

    if not st.button("Run Local CFBD Debugger", type="primary"):
        return

    if not str(user_query or "").strip():
        st.warning("Provide a user query to run the debugger.")
        return

    with st.spinner("Running delegator -> player finder -> CFBD checks..."):
        try:
            delegator_result = delegator_plan_tool(
                user_query=str(user_query),
                target_team=str(target_team),
                target_player_name=str(target_player_name),
            )
        except Exception as exc:
            st.error(f"Delegator failed: {exc}")
            return

        st.markdown("### Step 1: Delegator Output")
        st.code(json.dumps(delegator_result, indent=2, default=str), language="json")

        cfbd_params = delegator_result.get("cfbd_search_params") if isinstance(delegator_result, dict) else {}
        cfbd_params = cfbd_params if isinstance(cfbd_params, dict) else {}
        planned_name = str(cfbd_params.get("name") or target_player_name or "").strip()
        planned_team = str(cfbd_params.get("college_team") or target_team or "").strip()
        planned_position = str(cfbd_params.get("position") or position_hint or "").strip()

        st.markdown("### Step 2: Player Finder (Identity Resolution)")
        identity_result: dict[str, Any] = {"status": "skipped", "reason": "No name available", "data": {}}
        if planned_name:
            try:
                identity_result = resolve_player_identity_tool(
                    name_query=planned_name,
                    year=int(year),
                    position=planned_position or None,
                    team=planned_team or None,
                )
            except Exception as exc:
                identity_result = {"status": "error", "reason": f"Identity lookup failed: {exc}", "data": {}}

        st.code(json.dumps(identity_result, indent=2, default=str), language="json")

        identity_data = identity_result.get("data") if isinstance(identity_result, dict) else {}
        identity_data = identity_data if isinstance(identity_data, dict) else {}
        if bool(identity_data.get("requires_clarification")):
            st.warning("Identity resolver returned multiple low-confidence candidates.")

        resolved_athlete_id = str(athlete_id_override or identity_data.get("cfbd_athlete_id") or "").strip()

        search_fallback_result: dict[str, Any] = {}
        if not resolved_athlete_id and planned_name:
            try:
                search_fallback_result = cfbd_search_players_tool(
                    search_term=planned_name,
                    year=int(year),
                    team=planned_team or None,
                    position=planned_position or None,
                )
            except Exception as exc:
                search_fallback_result = {"status": "error", "reason": f"CFBD player search failed: {exc}", "data": []}

            rows = list(search_fallback_result.get("data") or [])
            for row in rows:
                candidate_id = str(row.get("athleteId") or row.get("athlete_id") or "").strip()
                if candidate_id:
                    resolved_athlete_id = candidate_id
                    break

            with st.expander("CFBD Player Search Fallback"):
                st.code(json.dumps(search_fallback_result, indent=2, default=str), language="json")

        st.markdown("### Step 3: CFBD Query + Response")
        endpoint = endpoint_mode
        if endpoint == "auto":
            endpoint = "player/usage" if resolved_athlete_id else "roster"

        parsed_player_id: int | None = None
        if str(player_id_override or "").strip().isdigit():
            parsed_player_id = int(str(player_id_override).strip())
        try:
            cfbd_result = cfbd_fetch_tool(
                athlete_id=resolved_athlete_id or None,
                team=planned_team or None,
                year=int(year),
                endpoint=endpoint,
                search_term=planned_name or None,
                position=planned_position or None,
                conference=str(conference or "").strip() or None,
                start_week=int(start_week) if int(start_week) > 0 else None,
                end_week=int(end_week) if int(end_week) > 0 else None,
                season_type=str(season_type or "").strip() or None,
                category=str(category or "").strip() or None,
                state=str(state_filter or "").strip() or None,
                classification=str(classification or "").strip() or None,
                player_id=parsed_player_id,
                exclude_garbage_time=bool(exclude_garbage_time),
            )
        except Exception as exc:
            st.error(f"CFBD fetch failed: {exc}")
            return

        meta = cfbd_result.get("meta") if isinstance(cfbd_result, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        if not meta:
            meta = {
                "endpoint": endpoint,
                "params": {
                    "athleteId": resolved_athlete_id or None,
                    "searchTerm": planned_name or None,
                    "team": planned_team or None,
                    "year": int(year),
                    "position": planned_position or None,
                },
            }

        debug_url = _build_cfbd_debug_url(meta)
        data_rows = list(cfbd_result.get("data") or []) if isinstance(cfbd_result, dict) else []

        st.write(f"Status: {cfbd_result.get('status', 'unknown')}")
        st.write(f"Reason: {cfbd_result.get('reason', '')}")
        st.write(f"Resolved athlete ID: {resolved_athlete_id or 'n/a'}")
        st.write(f"Endpoint: {meta.get('endpoint', endpoint)}")
        st.write(f"Record count: {len(data_rows)}")
        st.write(f"Query URL: {debug_url}")

        with st.expander("CFBD Request Meta"):
            st.code(json.dumps(meta, indent=2, default=str), language="json")

        with st.expander("CFBD Raw Result"):
            st.code(json.dumps(cfbd_result, indent=2, default=str), language="json")


st.set_page_config(page_title="Gridiron Intelligence - Scouting Workbench", page_icon="🏈", layout="wide")
st.markdown("<h1 class='football-title'>Gridiron Intelligence 🏈</h1>", unsafe_allow_html=True)
st.markdown("<p class='football-subtitle'>Interactive Scouting Workbench (Streamlit)</p>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://raw.githubusercontent.com/shaverm96/Gridiron-Intelligence/main/Logos/Main.svg", width=150)
    st.title("Gridiron Intelligence")
    workspace_options = ["Structured Report", "Structured Report + Open Chat", "Open Chat"]
    if _is_local_debug_page_enabled():
        workspace_options.append("Local CFBD Debugger")
    app_page = st.radio("Workspace", workspace_options, index=0)
    selected_persona = st.selectbox("Persona", ["Scout", "Fan"], index=0, key="selected_persona")
    st.write("---")
    st.caption(f"Gemini configured: {'Yes' if bool(CONFIG['GEMINI_API_KEY']) else 'No'}")
    st.caption(f"Supabase configured: {'Yes' if bool(CONFIG['SUPABASE_URL'] and CONFIG['SUPABASE_SERVICE_ROLE_KEY']) else 'No'}")
    with st.expander("Configuration diagnostics"):
        st.write(f"SUPABASE_URL source: {CONFIG_SOURCES['SUPABASE_URL']}")
        st.write(f"SUPABASE_SERVICE_ROLE_KEY source: {CONFIG_SOURCES['SUPABASE_SERVICE_ROLE_KEY']}")
        st.write(f"GEMINI_API_KEY source: {CONFIG_SOURCES['GEMINI_API_KEY']}")
        st.write(
            "GI_ENABLE_LOCAL_CFBD_DEBUGGER source: "
            f"{CONFIG_SOURCES['GI_ENABLE_LOCAL_CFBD_DEBUGGER']} "
            f"(enabled: {'Yes' if CONFIG['LOCAL_CFBD_DEBUGGER_ENABLED'] else 'No'})"
        )
        st.write(f"Supabase package import: {'Yes' if create_client is not None else 'No'}")
        if st.button("Run One-Click Diagnostic", key="run_one_click_diagnostic"):
            with st.spinner("Running connectivity and configuration checks..."):
                st.session_state["one_click_diag"] = run_one_click_diagnostics()

        diag = st.session_state.get("one_click_diag")
        if isinstance(diag, dict):
            if diag.get("overall") == "pass":
                st.success("One-click diagnostic passed.")
            else:
                st.error("One-click diagnostic found issues.")

            for item in diag.get("checks", []):
                icon = "✅" if item.get("status") == "pass" else "❌"
                st.write(f"{icon} {item.get('name')}: {item.get('detail')}")

    if CONFIG["LOCAL_CFBD_DEBUGGER_ENABLED"]:
        with st.expander("Local CFBD Debugger (opt-in)"):
            from engine.cfbd_service import cfbd_fetch

            endpoint = st.selectbox(
                "CFBD endpoint",
                ["player/stats", "player/search", "roster"],
                key="cfbd_debug_endpoint",
            )
            team = st.text_input("Team", value="", key="cfbd_debug_team")
            athlete_id = st.text_input("Athlete ID", value="", key="cfbd_debug_athlete_id")
            search_term = st.text_input("Search term", value="", key="cfbd_debug_search_term")
            year_raw = st.text_input("Year", value="", key="cfbd_debug_year")

            params: dict[str, Any] = {}
            if team.strip():
                params["team"] = team.strip()
            if athlete_id.strip():
                params["athleteId"] = athlete_id.strip()
            if search_term.strip():
                params["searchTerm"] = search_term.strip()
            if year_raw.strip().isdigit():
                params["year"] = int(year_raw.strip())

            if st.button("Run local CFBD debug request", key="run_local_cfbd_debug_request"):
                st.session_state["local_cfbd_debug_result"] = cfbd_fetch(endpoint=endpoint, params=params)

            debug_result = st.session_state.get("local_cfbd_debug_result")
            if isinstance(debug_result, dict):
                st.code(json.dumps(debug_result, indent=2, default=str), language="json")

    supabase_issues = get_supabase_config_issues()
    if supabase_issues:
        st.warning("Supabase preflight issues detected. Open diagnostics for details.")


@st.cache_resource
def get_cached_agent_graph():
    return get_scout_graph()


@st.cache_resource
def get_cached_structured_web_graph():
    return get_structured_web_graph()


def _compact_open_chat_state(state: dict[str, Any] | None) -> dict[str, Any]:
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
        "requires_identity_clarification": bool(src.get("requires_identity_clarification")),
        "clarification_prompt": str(src.get("clarification_prompt") or ""),
        "pending_identity_query": str(src.get("pending_identity_query") or ""),
        "security_halt": bool(src.get("security_halt")),
        "security_message": str(src.get("security_message") or ""),
        "next_step": str(src.get("next_step") or "supervisor"),
    }

    compact["identity_candidates"] = list(src.get("identity_candidates") or [])[-CHAT_STATE_MAX_CANDIDATES:]
    compact["conversation_history"] = list(src.get("conversation_history") or [])[-CHAT_STATE_MAX_TURNS * 2:]
    compact["trace_log"] = list(src.get("trace_log") or [])[-CHAT_STATE_MAX_TRACE:]
    compact["errors"] = list(src.get("errors") or [])[-CHAT_STATE_MAX_ERRORS:]
    compact["citations"] = list(src.get("citations") or [])[-CHAT_STATE_MAX_CITATIONS:]

    # Intentionally drop bulky payloads between chat turns.
    compact["sql_data_context"] = {}
    compact["web_research_context"] = ""
    compact["vector_factoids"] = []
    compact["comparables_context"] = ""

    return compact


def _allow_structured_report_submission() -> tuple[bool, int]:
    now = time.time()
    history_key = "structured_report_submit_timestamps"
    timestamps = list(st.session_state.get(history_key, []))
    cutoff = now - STRUCTURED_REPORT_RATE_LIMIT_WINDOW_SECONDS
    timestamps = [ts for ts in timestamps if float(ts) >= cutoff]

    if len(timestamps) >= STRUCTURED_REPORT_RATE_LIMIT_COUNT:
        st.session_state[history_key] = timestamps
        oldest_kept = min(timestamps) if timestamps else now
        retry_after = int(max(1, STRUCTURED_REPORT_RATE_LIMIT_WINDOW_SECONDS - (now - oldest_kept)))
        return False, retry_after

    timestamps.append(now)
    st.session_state[history_key] = timestamps
    return True, 0


def render_structured_report_page() -> None:
    try:
        player_index = load_player_index()
    except Exception as exc:
        st.error(f"Unable to load player index: {exc}")
        st.stop()

    selected_year = st.selectbox("Recruiting Class Year", CONFIG["YEARS"], index=0)
    year_players = player_index[player_index["year"] == selected_year]
    selected_label = st.selectbox("Player", year_players["player_label"].tolist()) if not year_players.empty else ""
    target_team = st.selectbox("Target Team", TARGET_TEAMS, index=0)

    if st.button("Generate Scouting Report", type="primary"):
        allowed, retry_after = _allow_structured_report_submission()
        if not allowed:
            st.warning(
                f"Rate limit reached: max {STRUCTURED_REPORT_RATE_LIMIT_COUNT} reports per "
                f"{STRUCTURED_REPORT_RATE_LIMIT_WINDOW_SECONDS} seconds. Try again in ~{retry_after}s."
            )
            return

        if not selected_label:
            st.warning("No players available for selected year.")
            return

        supabase_issues = get_supabase_config_issues()
        if supabase_issues:
            msg = "Cannot run report until Supabase is configured:\n" + "\n".join([f"- {issue}" for issue in supabase_issues])
            st.error(msg)
            st.stop()

        lookup = dict(zip(player_index["player_label"], player_index["recruit_id"]))
        recruit_id = lookup.get(selected_label)
        if not recruit_id:
            st.warning("Pick a valid player from the dropdown list.")
            st.stop()

        selected_label_parts = [part.strip() for part in str(selected_label).split("|")]
        selected_player_name = selected_label_parts[0] if selected_label_parts else ""
        selected_position_hint = selected_label_parts[1] if len(selected_label_parts) > 1 else ""
        milestone_slot = st.empty()

        def _render_structured_milestone(event: dict[str, str]) -> None:
            node = str(event.get("node") or "workflow")
            status = str(event.get("status") or "running")
            labels = {
                "workflow": "Web Scout Pipeline",
                "recruiting_scout": "Recruiting Scout",
                "team_scout": "Team Scout",
            }
            node_label = labels.get(node, node.replace("_", " ").title())
            if status == "completed":
                milestone_slot.success(f"{node_label}: complete")
            else:
                milestone_slot.info(f"{node_label}: running")

        with st.spinner("Building structured scouting report..."):
            try:
                sb = get_supabase_client()
                bundle = fetch_player_bundle(sb=sb, recruit_id=str(recruit_id))
            except Exception as exc:
                st.error(f"Structured report failed while fetching player bundle: {exc}")
                st.stop()

            player_row = dict(bundle.get("player") or {})
            player_profile = dict(bundle.get("player_profile") or {})
            scouting_clean = dict(bundle.get("scouting_clean") or {})
            pred_score_row = dict(bundle.get("pred_score") or {})
            pred_thr_row = dict(bundle.get("pred_threshold") or {})

            player_name = (
                str(player_profile.get("player_name") or "").strip()
                or str(player_row.get("player_name") or "").strip()
                or selected_player_name
            )
            position = str(
                player_profile.get("position")
                or player_row.get("position")
                or player_row.get("pos")
                or player_row.get("primary_position")
                or scouting_clean.get("position")
                or selected_position_hint
                or ""
            ).strip()
            high_school = str(player_row.get("high_school") or player_row.get("school") or "").strip()

            vector_query = (
                f"Player: {player_name}. Position: {position}. High school: {high_school}. "
                f"Target team: {target_team}. Class year: {selected_year}. "
                "Provide grounded trait/development insights relevant for recruiting projection."
            )
            vector_result = vector_insights_query(
                sb=sb,
                query_text=vector_query,
                position=position or None,
                top_k=CONFIG["VECTOR_MATCH_COUNT"],
                threshold=None,
            )

            web_recruiting_summary = ""
            web_team_summary = ""
            try:
                structured_web_graph = get_cached_structured_web_graph()
                web_state = orchestrate_structured_web_scouting(
                    player_name=player_name,
                    recruit_id=str(recruit_id),
                    target_team=str(target_team),
                    year=int(selected_year),
                    graph=structured_web_graph,
                    progress_callback=_render_structured_milestone,
                )
                web_recruiting_summary = str(web_state.get("web_recruiting_summary") or "").strip()
                web_team_summary = str(web_state.get("web_team_summary") or "").strip()
            except Exception as exc:
                web_recruiting_summary = f"Recruiting web summary unavailable: {exc}"
                web_team_summary = f"Team web summary unavailable: {exc}"
            finally:
                milestone_slot.success("Web Scout Pipeline complete")

            historical_comparables_md = get_historical_player_comparables(str(recruit_id))
            score_card_html = build_score_card_html(pred_score=pred_score_row, pred_threshold=pred_thr_row)
            web_summary = (
                "Recruiting Scout Summary:\n"
                f"{web_recruiting_summary or 'No recruiting summary available.'}\n\n"
                "Team Scout Summary:\n"
                f"{web_team_summary or 'No team summary available.'}"
            )

            final_prompt = build_final_prompt(
                year=int(selected_year),
                target_team=str(target_team),
                player_row=player_row,
                scouting_clean=scouting_clean,
                hs_athletic_background=str(scouting_clean.get("athletic_background") or "N/A"),
                pred_score_row=pred_score_row,
                pred_thr_row=pred_thr_row,
                web_summary=web_summary,
                vector_result=vector_result,
                historical_comparables_md=historical_comparables_md,
            )
            final_report = run_final_synthesis(final_prompt)

        st.markdown(f"## Scouting Workbench Output - {player_name}")
        st.markdown(
            f"- Recruit ID: `{recruit_id}`  \\\n+- Year: `{selected_year}`  \\\n+- Target Team: `{target_team}`  \\\n+- Persona: `{st.session_state.get('selected_persona', 'Scout')}`"
        )
        st.markdown("### Historical Comparables")
        st.markdown(historical_comparables_md or "No historical comparables available.")

        st.markdown("### Projected Model Score")
        st.markdown(score_card_html, unsafe_allow_html=True)

        st.markdown("### Recruiting Scout Summary")
        st.markdown(web_recruiting_summary or "No recruiting summary available.")

        st.markdown("### Team Scout Summary")
        st.markdown(web_team_summary or "No team summary available.")

        st.markdown("### Final Synthesis")
        st.markdown(final_report or "No final synthesis generated.")

        with st.expander("Development Information (temporary)"):
            st.markdown("#### Player Profile")
            st.code(json.dumps(player_profile or player_row, indent=2, default=str), language="json")

            st.markdown("#### Vector Insights")
            vector_insights = list(vector_result.get("insights") or []) if isinstance(vector_result, dict) else []
            if vector_insights:
                for insight in vector_insights[:8]:
                    st.write(f"- {insight}")
            else:
                reason = ""
                if isinstance(vector_result, dict):
                    reason = str(vector_result.get("reason") or "").strip()
                st.write(f"No vector insights returned for the selected player. {reason}".strip())

            st.markdown("#### Scouting Profile (General)")
            if scouting_clean:
                st.code(json.dumps(scouting_clean, indent=2, default=str), language="json")
            else:
                st.write("No scouting profile fields were available for this player.")


def render_structured_report_with_chat_page() -> None:
    report_state_key = "structured_chat_report_output"

    def _parse_selected_player_label(label: str | None) -> tuple[str, str, str, str]:
        parts = [part.strip() for part in str(label or "").split("|")]
        name = parts[0] if len(parts) > 0 else ""
        position = parts[1] if len(parts) > 1 else ""
        high_school = parts[2] if len(parts) > 2 else ""
        year = parts[3] if len(parts) > 3 else ""
        return name, position, high_school, year

    def _extract_predicted_score_display(score_card_html: str | None, pred_score_row: dict[str, Any] | None) -> str:
        html_text = str(score_card_html or "")
        if html_text:
            plain_text = re.sub(r"<[^>]+>", " ", html_text)
            plain_text = " ".join(plain_text.split())

            for pattern in [
                r"Predicted\s*Score\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?(?:\s*/\s*100)?)",
                r"([0-9]+(?:\.[0-9]+)?\s*/\s*100)",
            ]:
                match = re.search(pattern, plain_text, flags=re.IGNORECASE)
                if match:
                    return str(match.group(1)).replace(" / ", "/").strip()

        row = pred_score_row if isinstance(pred_score_row, dict) else {}
        for value in row.values():
            score = to_float_or_none(value)
            if score is None:
                continue
            if 0.0 <= score <= 1.0:
                return f"{score * 100.0:.3f}"
            return f"{score:.3f}"

        return "N/A"

    def _parse_historical_comparables_md(raw_md: str | None) -> dict[str, Any]:
        text = str(raw_md or "")
        lines = [line.strip() for line in text.splitlines() if line and line.strip()]

        target_position = ""
        rows: list[dict[str, Any]] = []

        def _is_placeholder(value: Any) -> bool:
            normalized = str(value or "").strip().lower()
            return normalized in {"", "-", "--", "n/a", "na", "none", "null", "unknown", "?"}

        def _clean_name(value: Any) -> str:
            name = str(value or "")
            # Strip markdown emphasis and extra symbols from raw text payloads.
            name = re.sub(r"[*_`~]+", "", name)
            return re.sub(r"\s+", " ", name).strip(" -|")

        def _match_numeric(match_text: str) -> float | None:
            number_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(match_text or ""))
            if not number_match:
                return None
            return to_float_or_none(number_match.group(1))

        for line in lines:
            clean = re.sub(r"^#{1,6}\s*", "", line).strip()
            if not clean:
                continue

            if clean.lower().startswith("target position:"):
                target_position = clean.split(":", 1)[1].strip() if ":" in clean else ""
                continue

            if clean.startswith("-") or clean.startswith("*"):
                body = clean[1:].strip()
                parsed = re.match(
                    r"^(?P<name>.+?)\s*\((?P<year>\d{4})\s*,\s*(?P<state>[A-Za-z]{2})\)\s*\|\s*Match:\s*(?P<match>[^|]+?)\s*\|\s*Rating:\s*(?P<rating>.+)$",
                    body,
                )
                if parsed:
                    raw_match = str(parsed.group("match") or "").strip()
                    match_value = _match_numeric(raw_match)
                    match_display = ""
                    if match_value is not None:
                        match_display = f"{match_value:.2f}%"
                    elif not _is_placeholder(raw_match):
                        match_display = raw_match

                    rows.append(
                        {
                            "name": _clean_name(parsed.group("name")),
                            "year": str(parsed.group("year") or "").strip(),
                            "state": str(parsed.group("state") or "").strip(),
                            "match": match_display,
                            "match_value": match_value,
                            "rating": str(parsed.group("rating") or "").strip(),
                            "raw": body,
                        }
                    )
                else:
                    cleaned_name = _clean_name(body)
                    rows.append(
                        {
                            "name": cleaned_name,
                            "year": "",
                            "state": "",
                            "match": "",
                            "match_value": None,
                            "rating": "",
                            "raw": body,
                        }
                    )

        filtered_rows: list[dict[str, Any]] = []
        for row in rows:
            name = _clean_name(row.get("name"))
            year = str(row.get("year") or "").strip()
            state = str(row.get("state") or "").strip()
            rating = str(row.get("rating") or "").strip()
            match = str(row.get("match") or "").strip()
            match_value = row.get("match_value")

            # Skip rows that are empty placeholders with no meaningful data.
            if _is_placeholder(name):
                continue

            has_real_metadata = any(not _is_placeholder(value) for value in [year, state, rating, match])
            if not has_real_metadata and match_value is None:
                continue

            # Comparables panel should only include entries with a usable match metric.
            if _is_placeholder(match) and match_value is None:
                continue

            filtered_rows.append(
                {
                    "name": name,
                    "year": "" if _is_placeholder(year) else year,
                    "state": "" if _is_placeholder(state) else state,
                    "rating": "" if _is_placeholder(rating) else rating,
                    "match": "" if _is_placeholder(match) else match,
                    "match_value": match_value,
                    "raw": str(row.get("raw") or "").strip(),
                }
            )

        filtered_rows.sort(key=lambda row: (row.get("match_value") is not None, row.get("match_value") or -1.0), reverse=True)

        return {
            "target_position": target_position,
            "rows": filtered_rows,
            "raw": text,
        }

    def _parse_summary_notes(raw_text: str | None) -> list[dict[str, str]]:
        text = str(raw_text or "")
        lines = [line.rstrip() for line in text.splitlines() if line and line.strip()]
        notes: list[dict[str, str]] = []

        for line in lines:
            clean = re.sub(r"^\s*[-*•]+\s*", "", line).strip()
            if not clean:
                continue

            label = ""
            body = clean
            if ":" in clean:
                left, right = clean.split(":", 1)
                left_clean = left.strip()
                right_clean = right.strip()
                if left_clean and right_clean and len(left_clean) <= 36:
                    label = left_clean
                    body = right_clean

            notes.append({"label": label, "body": body})

        return notes

    def _render_summary_card(title: str, raw_text: str | None, section_key: str, compact: bool = False) -> None:
        notes = _parse_summary_notes(raw_text)
        if not notes:
            st.markdown(f"### {title}")
            st.markdown(str(raw_text or "No summary available."))
            return

        card_variant = " structured-summary-card--dense" if compact else ""
        list_variant = " structured-summary-list--dense" if compact else ""
        note_variant = " structured-summary-note--dense" if compact else ""

        notes_html = "".join(
            [
                (
                    f"<div class='structured-summary-note{note_variant}'>"
                    f"<div class='structured-summary-note-label'>{html.escape(note.get('label') or 'Note')}</div>"
                    f"<div class='structured-summary-note-body'>{html.escape(note.get('body') or '')}</div>"
                    "</div>"
                    if str(note.get("label") or "").strip()
                    else (
                        f"<div class='structured-summary-note structured-summary-note--plain{note_variant}'>"
                        f"<div class='structured-summary-note-body'>{html.escape(note.get('body') or '')}</div>"
                        "</div>"
                    )
                )
                for note in notes
            ]
        )

        st.markdown(
            (
                f"<section class='structured-summary-card structured-summary-card--{html.escape(section_key)}{card_variant}'>"
                f"<h3 class='structured-summary-title'>{html.escape(title)}</h3>"
                f"<div class='structured-summary-list{list_variant}'>{notes_html}</div>"
                "</section>"
            ),
            unsafe_allow_html=True,
        )

    try:
        player_index = load_player_index()
    except Exception as exc:
        st.error(f"Unable to load player index: {exc}")
        st.stop()

    selected_year = st.selectbox("Recruiting Class Year", CONFIG["YEARS"], index=0, key="structured_chat_year")
    year_players = player_index[player_index["year"] == selected_year]
    selected_label = (
        st.selectbox("Player", year_players["player_label"].tolist(), key="structured_chat_player")
        if not year_players.empty
        else ""
    )
    target_team = st.selectbox("Target Team", TARGET_TEAMS, index=0, key="structured_chat_target_team")

    if st.button("Generate Scouting Report", type="primary", key="structured_chat_generate_report"):
        allowed, retry_after = _allow_structured_report_submission()
        if not allowed:
            st.warning(
                f"Rate limit reached: max {STRUCTURED_REPORT_RATE_LIMIT_COUNT} reports per "
                f"{STRUCTURED_REPORT_RATE_LIMIT_WINDOW_SECONDS} seconds. Try again in ~{retry_after}s."
            )
            return

        if not selected_label:
            st.warning("No players available for selected year.")
            return

        supabase_issues = get_supabase_config_issues()
        if supabase_issues:
            msg = "Cannot run report until Supabase is configured:\n" + "\n".join([f"- {issue}" for issue in supabase_issues])
            st.error(msg)
            st.stop()

        lookup = dict(zip(player_index["player_label"], player_index["recruit_id"]))
        recruit_id = lookup.get(selected_label)
        if not recruit_id:
            st.warning("Pick a valid player from the dropdown list.")
            st.stop()

        selected_player_name, selected_position_hint, selected_high_school_hint, _ = _parse_selected_player_label(selected_label)
        milestone_slot = st.empty()

        def _render_structured_milestone(event: dict[str, str]) -> None:
            node = str(event.get("node") or "workflow")
            status = str(event.get("status") or "running")
            labels = {
                "workflow": "Web Scout Pipeline",
                "recruiting_scout": "Recruiting Scout",
                "team_scout": "Team Scout",
            }
            node_label = labels.get(node, node.replace("_", " ").title())
            if status == "completed":
                milestone_slot.success(f"{node_label}: complete")
            else:
                milestone_slot.info(f"{node_label}: running")

        with st.spinner("Building structured scouting report..."):
            try:
                sb = get_supabase_client()
                bundle = fetch_player_bundle(sb=sb, recruit_id=str(recruit_id))
            except Exception as exc:
                st.error(f"Structured report failed while fetching player bundle: {exc}")
                st.stop()

            player_row = dict(bundle.get("player") or {})
            player_profile = dict(bundle.get("player_profile") or {})
            scouting_clean = dict(bundle.get("scouting_clean") or {})
            pred_score_row = dict(bundle.get("pred_score") or {})
            pred_thr_row = dict(bundle.get("pred_threshold") or {})

            player_name = (
                str(player_profile.get("player_name") or "").strip()
                or str(player_row.get("player_name") or "").strip()
                or selected_player_name
            )
            position = str(
                player_profile.get("position")
                or player_row.get("position")
                or player_row.get("pos")
                or player_row.get("primary_position")
                or scouting_clean.get("position")
                or selected_position_hint
                or ""
            ).strip()
            high_school = str(
                player_profile.get("high_school")
                or player_row.get("high_school")
                or player_row.get("school")
                or selected_high_school_hint
                or ""
            ).strip()

            vector_query = (
                f"Player: {player_name}. Position: {position}. High school: {high_school}. "
                f"Target team: {target_team}. Class year: {selected_year}. "
                "Provide grounded trait/development insights relevant for recruiting projection."
            )
            vector_result = vector_insights_query(
                sb=sb,
                query_text=vector_query,
                position=position or None,
                top_k=CONFIG["VECTOR_MATCH_COUNT"],
                threshold=None,
            )

            web_recruiting_summary = ""
            web_team_summary = ""
            try:
                structured_web_graph = get_cached_structured_web_graph()
                web_state = orchestrate_structured_web_scouting(
                    player_name=player_name,
                    recruit_id=str(recruit_id),
                    target_team=str(target_team),
                    year=int(selected_year),
                    graph=structured_web_graph,
                    progress_callback=_render_structured_milestone,
                )
                web_recruiting_summary = str(web_state.get("web_recruiting_summary") or "").strip()
                web_team_summary = str(web_state.get("web_team_summary") or "").strip()
            except Exception as exc:
                web_recruiting_summary = f"Recruiting web summary unavailable: {exc}"
                web_team_summary = f"Team web summary unavailable: {exc}"
            finally:
                milestone_slot.success("Web Scout Pipeline complete")

            historical_comparables_md = get_historical_player_comparables(str(recruit_id))
            score_card_html = build_score_card_html(pred_score=pred_score_row, pred_threshold=pred_thr_row)
            web_summary = (
                "Recruiting Scout Summary:\n"
                f"{web_recruiting_summary or 'No recruiting summary available.'}\n\n"
                "Team Scout Summary:\n"
                f"{web_team_summary or 'No team summary available.'}"
            )

            final_prompt = build_final_prompt(
                year=int(selected_year),
                target_team=str(target_team),
                player_row=player_row,
                scouting_clean=scouting_clean,
                hs_athletic_background=str(scouting_clean.get("athletic_background") or "N/A"),
                pred_score_row=pred_score_row,
                pred_thr_row=pred_thr_row,
                web_summary=web_summary,
                vector_result=vector_result,
                historical_comparables_md=historical_comparables_md,
            )
            final_report = run_final_synthesis(final_prompt)

        st.session_state[report_state_key] = {
            "player_name": player_name,
            "position": position,
            "high_school": high_school,
            "selected_player_label": selected_label,
            "recruit_id": str(recruit_id),
            "selected_year": selected_year,
            "target_team": target_team,
            "pred_score_row": pred_score_row,
            "historical_comparables_md": historical_comparables_md,
            "score_card_html": score_card_html,
            "web_recruiting_summary": web_recruiting_summary,
            "web_team_summary": web_team_summary,
            "final_report": final_report,
            "player_profile": player_profile,
            "player_row": player_row,
            "vector_result": vector_result,
            "scouting_clean": scouting_clean,
        }

    report_output = st.session_state.get(report_state_key)
    if isinstance(report_output, dict):
        st.markdown(
            """
            <style>
            .structured-report-player-header {
                text-align: center;
                margin: 0.25rem auto 1.2rem auto;
                max-width: 900px;
            }
            .structured-report-player-name {
                font-size: clamp(1.7rem, 1.9vw, 2.25rem);
                font-weight: 700;
                letter-spacing: 0.01em;
                line-height: 1.2;
                color: var(--text-color);
            }
            .structured-report-player-meta {
                margin-top: 0.35rem;
                font-size: 0.98rem;
                color: color-mix(in srgb, var(--text-color) 72%, transparent);
            }
            .structured-report-kpi-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 1.25rem;
                margin: 0;
            }
            .structured-report-kpi-wrap {
                width: min(940px, 100%);
                margin: 0.1rem auto 1.5rem auto;
            }
            .structured-report-kpi-card {
                background:
                    linear-gradient(
                        165deg,
                        color-mix(in srgb, var(--secondary-background-color) 92%, #0a1224 8%),
                        color-mix(in srgb, var(--background-color) 82%, #081020 18%)
                    );
                border: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
                border-radius: 16px;
                padding: 0.9rem 1.0rem;
                min-height: 112px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                text-align: center;
                box-shadow:
                    0 16px 36px color-mix(in srgb, #000 42%, transparent),
                    inset 0 1px 0 color-mix(in srgb, #fff 7%, transparent);
            }
            .structured-report-kpi-label {
                font-size: 0.72rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: color-mix(in srgb, var(--text-color) 52%, transparent);
                margin-bottom: 0.5rem;
                text-align: center;
            }
            .structured-report-kpi-value {
                font-size: clamp(1.35rem, 1.9vw, 1.85rem);
                font-weight: 800;
                line-height: 1.12;
                color: var(--text-color);
                word-break: break-word;
                letter-spacing: 0.01em;
                text-align: center;
            }
            .structured-report-kpi-card--score {
                border-color: color-mix(in srgb, #3b82f6 46%, transparent);
                background:
                    radial-gradient(circle at 100% 0%, color-mix(in srgb, #3b82f6 18%, transparent), transparent 52%),
                    linear-gradient(
                        165deg,
                        color-mix(in srgb, var(--secondary-background-color) 84%, #0f1a33 16%),
                        color-mix(in srgb, var(--background-color) 72%, #101f3d 28%)
                    );
            }
            .structured-report-kpi-card--score .structured-report-kpi-value {
                color: color-mix(in srgb, #9ec5ff 70%, var(--text-color) 30%);
            }
            .structured-comps-wrap {
                background: color-mix(in srgb, var(--secondary-background-color) 90%, var(--background-color));
                border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
                border-radius: 16px;
                padding: 0.95rem 1rem 0.9rem 1rem;
                margin: 0 0 1.1rem 0;
            }
            .structured-comps-position-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                font-size: 0.76rem;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: color-mix(in srgb, #9ec5ff 66%, var(--text-color) 34%);
                background: color-mix(in srgb, #3b82f6 14%, transparent);
                border: 1px solid color-mix(in srgb, #3b82f6 35%, transparent);
                border-radius: 999px;
                padding: 0.33rem 0.68rem;
                margin: 0 0 0.85rem 0;
            }
            .structured-comps-list {
                display: flex;
                flex-direction: column;
                gap: 0.7rem;
            }
            .structured-comps-item {
                border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
                border-radius: 12px;
                padding: 0.86rem 0.92rem;
                background: color-mix(in srgb, var(--background-color) 90%, var(--secondary-background-color));
            }
            .structured-comps-item-top {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 0.9rem;
                margin-bottom: 0.56rem;
            }
            .structured-comps-player {
                font-size: clamp(1.12rem, 1.25vw, 1.24rem);
                font-weight: 820;
                color: var(--text-color);
                line-height: 1.14;
                letter-spacing: 0.012em;
                padding-left: 0.52rem;
                border-left: 2px solid color-mix(in srgb, #3b82f6 42%, transparent);
                text-shadow: 0 1px 0 color-mix(in srgb, #000 30%, transparent);
            }
            .structured-comps-match {
                white-space: nowrap;
                font-size: 0.84rem;
                font-weight: 760;
                color: color-mix(in srgb, #9ec5ff 72%, var(--text-color) 28%);
                background: color-mix(in srgb, #3b82f6 18%, transparent);
                border: 1px solid color-mix(in srgb, #3b82f6 34%, transparent);
                border-radius: 999px;
                padding: 0.24rem 0.62rem;
            }
            .structured-comps-meta {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem 0.9rem;
                font-size: 0.83rem;
                color: color-mix(in srgb, var(--text-color) 68%, transparent);
                margin-left: 0.62rem;
            }
            .structured-comps-meta-item {
                display: inline-flex;
                align-items: baseline;
                gap: 0.26rem;
                white-space: nowrap;
            }
            .structured-comps-meta-label {
                color: color-mix(in srgb, var(--text-color) 52%, transparent);
                text-transform: uppercase;
                letter-spacing: 0.06em;
                font-size: 0.69rem;
                font-weight: 650;
            }
            .structured-comps-meta-value {
                color: color-mix(in srgb, var(--text-color) 86%, transparent);
                font-weight: 600;
            }
            .structured-comps-fallback {
                margin: 0;
                font-size: 0.92rem;
                color: color-mix(in srgb, var(--text-color) 74%, transparent);
            }
            .structured-summary-card {
                background: color-mix(in srgb, var(--secondary-background-color) 90%, var(--background-color));
                border: 1px solid color-mix(in srgb, var(--text-color) 12%, transparent);
                border-radius: 16px;
                padding: 1rem 1rem 0.92rem 1rem;
                margin: 0 0 1.1rem 0;
            }
            .structured-summary-title {
                margin: 0 0 0.8rem 0;
                font-size: 1.28rem;
                font-weight: 760;
                letter-spacing: 0.01em;
                line-height: 1.2;
                color: var(--text-color);
            }
            .structured-summary-list {
                display: flex;
                flex-direction: column;
                gap: 0.62rem;
            }
            .structured-summary-card--dense {
                padding: 0.95rem 1rem 0.88rem 1rem;
            }
            .structured-summary-card--dense .structured-summary-title {
                margin-bottom: 0.68rem;
                font-size: 1.2rem;
            }
            .structured-summary-list--dense {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.5rem 0.62rem;
            }
            .structured-summary-note {
                border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
                border-radius: 12px;
                background: color-mix(in srgb, var(--background-color) 88%, var(--secondary-background-color));
                padding: 0.72rem 0.82rem;
            }
            .structured-summary-note--dense {
                padding: 0.58rem 0.66rem;
                border-radius: 10px;
                min-height: 92px;
            }
            .structured-summary-note-label {
                margin: 0 0 0.28rem 0;
                font-size: 0.73rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: color-mix(in srgb, var(--text-color) 54%, transparent);
            }
            .structured-summary-note-body {
                margin: 0;
                font-size: 0.95rem;
                line-height: 1.5;
                color: color-mix(in srgb, var(--text-color) 94%, transparent);
            }
            .structured-summary-note--dense .structured-summary-note-label {
                margin-bottom: 0.2rem;
                font-size: 0.66rem;
                letter-spacing: 0.07em;
            }
            .structured-summary-note--dense .structured-summary-note-body {
                font-size: 0.9rem;
                line-height: 1.38;
            }
            .structured-summary-note--plain {
                border-left: 2px solid color-mix(in srgb, #3b82f6 35%, transparent);
                padding-left: 0.74rem;
            }
            @media (max-width: 1000px) {
                .structured-report-kpi-wrap {
                    width: min(720px, 100%);
                }
                .structured-report-kpi-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
            }
            @media (max-width: 640px) {
                .structured-report-kpi-wrap {
                    width: 100%;
                }
                .structured-report-kpi-grid {
                    grid-template-columns: 1fr;
                    gap: 0.9rem;
                }
                .structured-report-kpi-card {
                    min-height: 96px;
                }
                .structured-comps-item-top {
                    align-items: flex-start;
                    flex-direction: column;
                    gap: 0.45rem;
                }
                .structured-comps-match {
                    white-space: normal;
                }
                .structured-summary-card {
                    padding: 0.9rem 0.9rem 0.82rem 0.9rem;
                }
                .structured-summary-title {
                    font-size: 1.16rem;
                }
                .structured-summary-list--dense {
                    grid-template-columns: 1fr;
                    gap: 0.5rem;
                }
                .structured-summary-note--dense {
                    min-height: unset;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        selected_name_hint, selected_position_hint, selected_high_school_hint, _ = _parse_selected_player_label(selected_label)

        player_name = str(report_output.get("player_name") or selected_name_hint or "Unknown Player").strip()
        position = str(report_output.get("position") or selected_position_hint or "").strip()
        high_school = str(report_output.get("high_school") or selected_high_school_hint or "").strip()
        player_meta = " | ".join([part for part in [position, high_school] if part])

        st.markdown(
            ""
            f"<div class='structured-report-player-header'>"
            f"<div class='structured-report-player-name'>{html.escape(player_name)}</div>"
            f"<div class='structured-report-player-meta'>{html.escape(player_meta)}</div>"
            f"</div>"
            "",
            unsafe_allow_html=True,
        )

        predicted_score_display = _extract_predicted_score_display(
            score_card_html=str(report_output.get("score_card_html") or ""),
            pred_score_row=report_output.get("pred_score_row") or {},
        )
        kpi_cards = [
            ("Recruit ID", str(report_output.get("recruit_id") or "N/A")),
            ("Year", str(report_output.get("selected_year") or "N/A")),
            ("Target Team", str(report_output.get("target_team") or "N/A")),
            ("Predicted Score", predicted_score_display),
        ]
        kpi_cards_html = "".join(
            [
                (
                    f"<div class='structured-report-kpi-card{' structured-report-kpi-card--score' if label == 'Predicted Score' else ''}'>"
                    f"<div class='structured-report-kpi-label'>{html.escape(label)}</div>"
                    f"<div class='structured-report-kpi-value'>{html.escape(value)}</div>"
                    "</div>"
                )
                for label, value in kpi_cards
            ]
        )
        st.markdown(
            (
                "<div class='structured-report-kpi-wrap'>"
                f"<div class='structured-report-kpi-grid'>{kpi_cards_html}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

        st.markdown("### Historical Comparables")
        comps_data = _parse_historical_comparables_md(report_output.get("historical_comparables_md"))
        comps_rows = list(comps_data.get("rows") or [])
        comps_target_position = str(comps_data.get("target_position") or "").strip()

        if comps_rows:
            comps_position_html = (
                (
                    "<div class='structured-comps-position-badge'>"
                    "<span>Target Position</span>"
                    f"<span>{html.escape(comps_target_position)}</span>"
                    "</div>"
                )
                if comps_target_position
                else ""
            )
            comps_items_html = "".join(
                [
                    (
                        "<div class='structured-comps-item'>"
                        "<div class='structured-comps-item-top'>"
                        f"<div class='structured-comps-player'>{html.escape(str(row.get('name') or ''))}</div>"
                        f"<div class='structured-comps-match'>Match {html.escape(str(row.get('match') or ''))}</div>"
                        "</div>"
                        "<div class='structured-comps-meta'>"
                        f"<span class='structured-comps-meta-item'><span class='structured-comps-meta-label'>Class</span><span class='structured-comps-meta-value'>{html.escape(str(row.get('year') or ''))}</span></span>"
                        f"<span class='structured-comps-meta-item'><span class='structured-comps-meta-label'>State</span><span class='structured-comps-meta-value'>{html.escape(str(row.get('state') or ''))}</span></span>"
                        f"<span class='structured-comps-meta-item'><span class='structured-comps-meta-label'>Rating</span><span class='structured-comps-meta-value'>{html.escape(str(row.get('rating') or ''))}</span></span>"
                        "</div>"
                        "</div>"
                    )
                    for row in comps_rows
                ]
            )
            st.markdown(
                (
                    "<div class='structured-comps-wrap'>"
                    f"{comps_position_html}"
                    f"<div class='structured-comps-list'>{comps_items_html}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        else:
            fallback_text = str(comps_data.get("raw") or "").strip() or "No historical comparables available."
            st.markdown(f"<p class='structured-comps-fallback'>{html.escape(fallback_text)}</p>", unsafe_allow_html=True)

        st.markdown("### Projected Model Score")
        st.markdown(str(report_output.get("score_card_html") or ""), unsafe_allow_html=True)

        _render_summary_card(
            title="Recruiting Scout Summary",
            raw_text=str(report_output.get("web_recruiting_summary") or "No recruiting summary available."),
            section_key="recruiting",
            compact=True,
        )

        _render_summary_card(
            title="Team Scout Summary",
            raw_text=str(report_output.get("web_team_summary") or "No team summary available."),
            section_key="team",
        )

        st.markdown("### Final Synthesis")
        st.markdown(report_output.get("final_report") or "No final synthesis generated.")

        with st.expander("Development Information (temporary)"):
            st.markdown("#### Player Profile")
            st.code(
                json.dumps(report_output.get("player_profile") or report_output.get("player_row") or {}, indent=2, default=str),
                language="json",
            )

            st.markdown("#### Vector Insights")
            vector_result = report_output.get("vector_result") or {}
            vector_insights = list(vector_result.get("insights") or []) if isinstance(vector_result, dict) else []
            if vector_insights:
                for insight in vector_insights[:8]:
                    st.write(f"- {insight}")
            else:
                reason = ""
                if isinstance(vector_result, dict):
                    reason = str(vector_result.get("reason") or "").strip()
                st.write(f"No vector insights returned for the selected player. {reason}".strip())

            st.markdown("#### Scouting Profile (General)")
            scouting_clean = report_output.get("scouting_clean") or {}
            if scouting_clean:
                st.code(json.dumps(scouting_clean, indent=2, default=str), language="json")
            else:
                st.write("No scouting profile fields were available for this player.")

        st.write("---")
        st.subheader("Open Chat")
        st.caption(
            "Session-scoped memory is isolated to this combined page. "
            f"Current persona: {st.session_state.get('selected_persona', 'Scout')}"
        )

        if "structured_chat_messages" not in st.session_state:
            st.session_state["structured_chat_messages"] = []
        if "structured_chat_agent_state" not in st.session_state:
            st.session_state["structured_chat_agent_state"] = initial_chat_state("")

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Clear Chat", key="structured_chat_clear"):
                st.session_state["structured_chat_messages"] = []
                st.session_state["structured_chat_agent_state"] = initial_chat_state("")
                st.rerun()

        for message in st.session_state["structured_chat_messages"]:
            with st.chat_message(message.get("role", "assistant")):
                st.markdown(str(message.get("content", "")))

        user_prompt = st.chat_input(
            "Ask anything about players, comparables, web intel, or team fit...",
            key="structured_chat_input",
        )
        if not user_prompt:
            return

        st.session_state["structured_chat_messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            milestone_slot = st.empty()

            def _render_milestone(event: dict[str, str]) -> None:
                node = str(event.get("node") or "workflow")
                status = str(event.get("status") or "running")
                labels = {
                    "workflow": "Pipeline",
                    "lead_delegator": "Delegator",
                    "cfbd_analyst": "CFBD Analyst",
                    "recruiting_scout": "Recruiting Scout",
                    "team_scout": "Team Scout",
                    "lead_synthesizer": "Lead Synthesizer",
                }
                node_label = labels.get(node, node.replace("_", " ").title())
                if status == "completed":
                    milestone_slot.success(f"{node_label}: complete")
                else:
                    milestone_slot.info(f"{node_label}: running")

            with st.spinner("Thinking..."):
                try:
                    graph = get_cached_agent_graph()
                    current_state = _compact_open_chat_state(st.session_state.get("structured_chat_agent_state", {}))
                    result_state = orchestrate_chat_turn(
                        user_prompt=user_prompt,
                        current_state=current_state,
                        graph=graph,
                        progress_callback=_render_milestone,
                    )
                    assistant_text = str(result_state.get("final_report") or "No response generated.")

                    st.session_state["structured_chat_agent_state"] = _compact_open_chat_state(result_state)
                    st.session_state["structured_chat_messages"].append({"role": "assistant", "content": assistant_text})
                    milestone_slot.success("Pipeline complete")
                    st.markdown(assistant_text)

                    if bool(result_state.get("requires_identity_clarification")):
                        candidate_rows = list(result_state.get("identity_candidates") or [])
                        if candidate_rows:
                            with st.expander("Identity clarification candidates"):
                                for idx, row in enumerate(candidate_rows, start=1):
                                    name = str(
                                        row.get("player_name")
                                        or row.get("full_name")
                                        or row.get("recruit_name")
                                        or "Unknown"
                                    ).strip()
                                    position = str(row.get("position") or row.get("position_group") or "?").strip() or "?"
                                    year = str(row.get("recruit_class") or row.get("year") or "?").strip() or "?"
                                    team = str(row.get("committed_to") or row.get("teams") or "").strip()
                                    rid = str(row.get("recruit_id") or "").strip()
                                    score = row.get("score")
                                    score_text = f"{float(score):.2f}" if score is not None else "n/a"
                                    team_text = f" | Team: {team}" if team else ""
                                    rid_text = f" | Recruit ID: {rid}" if rid else " | Recruit ID: n/a"
                                    st.write(
                                        f"{idx}. {name} | Pos: {position} | Year: {year}{team_text}{rid_text} | Score: {score_text}"
                                    )

                    trace_log = list(result_state.get("trace_log") or [])
                    if trace_log:
                        with st.expander("Execution Trace"):
                            st.code(json.dumps(trace_log, indent=2, default=str), language="json")

                    errors = result_state.get("errors", [])
                    if errors:
                        with st.expander("Agent notes"):
                            for err in errors[-3:]:
                                st.write(f"- {err}")
                except Exception as exc:
                    err_text = f"Open chat failed: {exc}"
                    st.session_state["structured_chat_messages"].append({"role": "assistant", "content": err_text})
                    st.error(err_text)


def render_open_chat_page() -> None:
    st.subheader("Open Chat")
    st.caption(
        "Session-scoped memory is isolated to this page. "
        f"Current persona: {st.session_state.get('selected_persona', 'Scout')}"
    )

    if "open_chat_messages" not in st.session_state:
        st.session_state["open_chat_messages"] = []
    if "open_chat_agent_state" not in st.session_state:
        st.session_state["open_chat_agent_state"] = initial_chat_state("")

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Clear Chat"):
            st.session_state["open_chat_messages"] = []
            st.session_state["open_chat_agent_state"] = initial_chat_state("")
            st.rerun()

    for message in st.session_state["open_chat_messages"]:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(str(message.get("content", "")))

    user_prompt = st.chat_input("Ask anything about players, comparables, web intel, or team fit...")
    if not user_prompt:
        return

    st.session_state["open_chat_messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        milestone_slot = st.empty()

        def _render_milestone(event: dict[str, str]) -> None:
            node = str(event.get("node") or "workflow")
            status = str(event.get("status") or "running")
            labels = {
                "workflow": "Pipeline",
                "lead_delegator": "Delegator",
                "cfbd_analyst": "CFBD Analyst",
                "recruiting_scout": "Recruiting Scout",
                "team_scout": "Team Scout",
                "lead_synthesizer": "Lead Synthesizer",
            }
            node_label = labels.get(node, node.replace("_", " ").title())
            if status == "completed":
                milestone_slot.success(f"{node_label}: complete")
            else:
                milestone_slot.info(f"{node_label}: running")

        with st.spinner("Thinking..."):
            try:
                graph = get_cached_agent_graph()
                current_state = _compact_open_chat_state(st.session_state.get("open_chat_agent_state", {}))
                result_state = orchestrate_chat_turn(
                    user_prompt=user_prompt,
                    current_state=current_state,
                    graph=graph,
                    progress_callback=_render_milestone,
                )
                assistant_text = str(result_state.get("final_report") or "No response generated.")

                st.session_state["open_chat_agent_state"] = _compact_open_chat_state(result_state)
                st.session_state["open_chat_messages"].append({"role": "assistant", "content": assistant_text})
                milestone_slot.success("Pipeline complete")
                st.markdown(assistant_text)

                if bool(result_state.get("requires_identity_clarification")):
                    candidate_rows = list(result_state.get("identity_candidates") or [])
                    if candidate_rows:
                        with st.expander("Identity clarification candidates"):
                            for idx, row in enumerate(candidate_rows, start=1):
                                name = str(row.get("player_name") or row.get("full_name") or row.get("recruit_name") or "Unknown").strip()
                                position = str(row.get("position") or row.get("position_group") or "?").strip() or "?"
                                year = str(row.get("recruit_class") or row.get("year") or "?").strip() or "?"
                                team = str(row.get("committed_to") or row.get("teams") or "").strip()
                                rid = str(row.get("recruit_id") or "").strip()
                                score = row.get("score")
                                score_text = f"{float(score):.2f}" if score is not None else "n/a"
                                team_text = f" | Team: {team}" if team else ""
                                rid_text = f" | Recruit ID: {rid}" if rid else " | Recruit ID: n/a"
                                st.write(f"{idx}. {name} | Pos: {position} | Year: {year}{team_text}{rid_text} | Score: {score_text}")

                trace_log = list(result_state.get("trace_log") or [])
                if trace_log:
                    with st.expander("Execution Trace"):
                        st.code(json.dumps(trace_log, indent=2, default=str), language="json")

                errors = result_state.get("errors", [])
                if errors:
                    with st.expander("Agent notes"):
                        for err in errors[-3:]:
                            st.write(f"- {err}")
            except Exception as exc:
                err_text = f"Open chat failed: {exc}"
                st.session_state["open_chat_messages"].append({"role": "assistant", "content": err_text})
                st.error(err_text)


if app_page == "Structured Report":
    render_structured_report_page()
elif app_page == "Structured Report + Open Chat":
    render_structured_report_with_chat_page()
elif app_page == "Open Chat":
    render_open_chat_page()
else:
    render_local_cfbd_debugger_page()

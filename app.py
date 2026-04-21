from __future__ import annotations

import html
import importlib
import io
import json
import os
import re
import time
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import streamlit as st

try:
    import altair as alt
except ImportError:
    alt = None

from engine import (
    get_scout_graph,
    get_structured_web_graph,
    orchestrate_chat_turn,
    orchestrate_structured_web_scouting,
)
from engine.state import (
    compact_open_chat_state,
    compact_transfer_chat_state,
    initial_chat_state,
)
from engine.streamlit_config import build_streamlit_runtime_config_data

from engine.comparables_service import (
    get_historical_player_comparables_data,
)
from engine.data_access import (
    fetch_player_bundle_data,
    load_model_tiers_from_supabase_data,
    load_recruit_candidate_window_from_supabase_data,
    search_transfer_player_index_from_supabase_data,
    search_recruit_candidate_matches_from_supabase_data,
    score_tier_from_tiers_data,
    tier_definitions_markdown_data,
)
from engine.diagnostics import run_one_click_diagnostics_data
from engine.diagnostics import (
    get_gemini_config_issues_data,
    get_model_pricing_issues_data,
    get_supabase_config_issues_data,
)
from engine.data_transforms import (
    build_recruiting_summary_layout_data,
    build_player_profile_view_data,
    build_score_card_html_data,
    build_transfer_usage_with_yoy_table,
    clean_scouting_profile_data,
    extract_predicted_score_display_data,
    merge_scouting_sources_data,
    parse_historical_comparables_md_data,
    parse_selected_player_label_data,
    parse_summary_notes_data,
    rows_to_dynamic_table,
    split_team_tokens_text,
    transfer_position_stat_order,
    transfer_position_usage_order,
    transfer_to_percent_points,
)
from engine.orchestration_service import (
    orchestrate_transfer_cfbd_context,
    orchestrate_transfer_chat_turn,
    orchestrate_transfer_report,
)
from engine.synthesis_service import (
    build_final_prompt_data,
    run_final_synthesis_with_telemetry_data,
)
from engine.vector_service import (
    vector_insights_query_data,
)
from engine.tools import (
    cfbd_fetch_tool,
    cfbd_search_players_tool,
    delegator_plan_tool,
    resolve_player_identity_tool,
)
from engine.utils import (
    first_non_null,
    image_data_uri_data,
    llm_response_to_text,
    parse_jsonish,
    to_float_or_none,
)

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from supabase import create_client
except ImportError:
    create_client = None

try:
    _docx_module = importlib.import_module("docx")
    Document = getattr(_docx_module, "Document", None)
except Exception:
    Document = None


RUNTIME_CFG = build_streamlit_runtime_config_data(secrets=st.secrets)
PROJECT_ROOT = RUNTIME_CFG["project_root"]
CONFIG = dict(RUNTIME_CFG["config"])

CHAT_STATE_MAX_TURNS = 6
CHAT_STATE_MAX_TRACE = 10
CHAT_STATE_MAX_ERRORS = 6
CHAT_STATE_MAX_CITATIONS = 16
CHAT_STATE_MAX_CANDIDATES = 3
POSITION_FILTER_PLACEHOLDER = "Select Position"
POSITION_FILTER_OPTIONS = [
    "QB",
    "RB",
    "WR",
    "TE",
    "OT",
    "OG",
    "C",
    "OL",
    "EDGE",
    "DE",
    "DT",
    "DL",
    "LB",
    "CB",
    "S",
    "DB",
    "ATH",
    "K",
    "P",
    "LS",
]
RECRUIT_POSITION_OPTIONS = list(POSITION_FILTER_OPTIONS)
STRUCTURED_REPORT_RATE_LIMIT_COUNT = 3
STRUCTURED_REPORT_RATE_LIMIT_WINDOW_SECONDS = 60

CONFIG_SOURCES = dict(RUNTIME_CFG["config_sources"])
TABLES = dict(RUNTIME_CFG["tables"])
TARGET_TEAMS = list(RUNTIME_CFG["target_teams"])
POS_MAP = dict(RUNTIME_CFG["pos_map"])

EMBED_MODEL = None
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_MODEL_LOAD_ERROR = None


@st.cache_resource
def _get_cached_supabase_client(url: str, service_role_key: str):
    if create_client is None:
        return None
    if not url or not service_role_key:
        return None
    return create_client(url, service_role_key)


def get_supabase_client():
    return _get_cached_supabase_client(CONFIG["SUPABASE_URL"], CONFIG["SUPABASE_SERVICE_ROLE_KEY"])


def get_supabase_config_issues() -> list[str]:
    return get_supabase_config_issues_data(
        config=CONFIG,
        config_sources=CONFIG_SOURCES,
        has_create_client=create_client is not None,
    )


def get_gemini_config_issues() -> list[str]:
    return get_gemini_config_issues_data(
        config=CONFIG,
        config_sources=CONFIG_SOURCES,
        has_chat_model=ChatGoogleGenerativeAI is not None,
    )


def get_model_pricing_issues() -> list[str]:
    return get_model_pricing_issues_data(config=CONFIG)


def run_one_click_diagnostics() -> dict:
    return run_one_click_diagnostics_data(
        config=CONFIG,
        config_sources=CONFIG_SOURCES,
        tables=TABLES,
        summary_model=CONFIG["SUMMARY_MODEL"],
        get_supabase_config_issues=get_supabase_config_issues,
        get_supabase_client=get_supabase_client,
        get_gemini_config_issues=get_gemini_config_issues,
        get_model_pricing_issues=get_model_pricing_issues,
        get_llm=get_llm,
        llm_response_to_text=llm_response_to_text,
    )


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


@st.cache_data
def load_model_tiers() -> pd.DataFrame:
    sb = get_supabase_client()
    return load_model_tiers_from_supabase_data(sb=sb, pred_score_table=TABLES["pred_score"])


def score_tier(score: float | None) -> str:
    tiers = load_model_tiers()
    return score_tier_from_tiers_data(score=score, tiers=tiers, to_float_or_none=to_float_or_none)


def tier_definitions_markdown() -> str:
    tiers = load_model_tiers()
    return tier_definitions_markdown_data(tiers)


@st.cache_data(ttl=300, show_spinner=False)
def load_recruit_candidate_window(year: int, position: str, limit: int = 100) -> pd.DataFrame:
    sb = get_supabase_client()
    if sb is None:
        raise RuntimeError("Supabase is not configured. Recruit dropdown requires gi_recruit_master.")

    return load_recruit_candidate_window_from_supabase_data(
        sb=sb,
        table_name=TABLES["player_master"],
        year=year,
        position=position,
        position_map=POS_MAP,
        limit=limit,
    )


@st.cache_data(ttl=300, show_spinner=False)
def search_recruit_candidate_matches(year: int, position: str, search_text: str, limit: int = 100) -> pd.DataFrame:
    sb = get_supabase_client()
    if sb is None:
        raise RuntimeError("Supabase is not configured. Recruit search requires gi_recruit_master.")

    return search_recruit_candidate_matches_from_supabase_data(
        sb=sb,
        table_name=TABLES["player_master"],
        year=year,
        position=position,
        search_text=search_text,
        position_map=POS_MAP,
        limit=limit,
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_transfer_candidate_matches(position: str, search_text: str, limit: int = 25) -> pd.DataFrame:
    sb = get_supabase_client()
    if sb is None:
        raise RuntimeError("Supabase is not configured. Transfer search requires gi_college_master.")

    return search_transfer_player_index_from_supabase_data(
        sb=sb,
        table_name="gi_college_master",
        position=position,
        search_text=search_text,
        limit=limit,
    )


def _render_transfer_candidate_live_picker(
    widget_prefix: str,
    selected_position: str,
    limit: int = 25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_label_key = f"{widget_prefix}_selected_label"
    query_text = st.text_input(
        "Player Lookup",
        value=str(st.session_state.get(f"{widget_prefix}_candidate_query") or ""),
        placeholder="Type at least 3 letters to live-search transfer players",
        key=f"{widget_prefix}_candidate_query_input",
    )
    st.session_state[f"{widget_prefix}_candidate_query"] = str(query_text or "")

    if selected_position == POSITION_FILTER_PLACEHOLDER:
        st.info("Choose a position to load transfer candidates.")
        st.session_state.pop(selected_label_key, None)
        return pd.DataFrame(), {}

    if len(str(query_text or "").strip()) < 3:
        st.info("Type at least 3 letters to live-load player names.")
        st.session_state.pop(selected_label_key, None)
        return pd.DataFrame(), {}

    try:
        candidate_df = load_transfer_candidate_matches(selected_position, str(query_text).strip(), limit=limit)
    except Exception as exc:
        st.error(f"Unable to search transfer candidates: {exc}")
        return pd.DataFrame(), {}

    if candidate_df.empty:
        st.info("No transfer candidates matched the current position and text.")
        st.session_state.pop(selected_label_key, None)
        return candidate_df, {}

    st.caption("Live matches")
    selected_label = str(st.session_state.get(selected_label_key) or "")
    records = candidate_df.to_dict(orient="records")
    for idx, row in enumerate(records):
        label = str(row.get("player_label") or "")
        team_text = str(row.get("teams") or "").strip() or "Team N/A"
        years_text = f"{row.get('first_season') or '?'}-{row.get('last_season') or '?'}"
        button_text = f"{str(row.get('player_name') or 'Unknown')} | {str(row.get('position') or '?')} | {team_text} | {years_text}"
        if st.button(button_text, key=f"{widget_prefix}_pick_{idx}", width="stretch"):
            selected_label = label
            st.session_state[selected_label_key] = label

    if not selected_label:
        return candidate_df, {}

    selected_matches = candidate_df[candidate_df["player_label"] == selected_label]
    if selected_matches.empty:
        st.session_state.pop(selected_label_key, None)
        return candidate_df, {}

    selected_row = dict(selected_matches.iloc[0].to_dict())
    st.success(f"Selected player: {selected_label}")
    return candidate_df, selected_row


def _render_recruit_candidate_live_picker(
    widget_prefix: str,
    selected_year: int,
    selected_position: str,
    limit: int = 100,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_label_key = f"{widget_prefix}_selected_label"

    if selected_position == POSITION_FILTER_PLACEHOLDER:
        st.info("Choose a position to load candidates.")
        st.session_state.pop(selected_label_key, None)
        return pd.DataFrame(), {}

    base_df = pd.DataFrame()
    try:
        base_df = load_recruit_candidate_window(int(selected_year), str(selected_position), limit=limit)
    except Exception as exc:
        st.error(f"Unable to load recruit candidates: {exc}")
        return pd.DataFrame(), {}

    if base_df.empty:
        st.info("No candidates matched the selected year and position.")
        st.session_state.pop(selected_label_key, None)
        return base_df, {}

    if "rating" in base_df.columns:
        base_df = base_df.copy()
        base_df["_rating_sort"] = pd.to_numeric(base_df["rating"], errors="coerce")
        base_df = base_df.sort_values("_rating_sort", ascending=False, na_position="last").drop(columns=["_rating_sort"])

    st.caption("Top 100 by rating")
    selected_label = str(st.session_state.get(selected_label_key) or "")
    top_100_df = base_df.head(100).copy()
    display_to_label: dict[str, str] = {}
    top_100_display_options: list[str] = []
    for _, row in top_100_df.iterrows():
        raw_label = str(row.get("player_label") or "").strip()
        if not raw_label:
            continue
        player_name, position, school, year = parse_selected_player_label_data(raw_label)
        stars = _star_icons(_recruit_star_value(row.get("rating")))
        display = (
            f"{stars} {player_name} | {school or 'School N/A'} | {position or '?'} | {year or '?'}"
            if stars
            else f"{player_name} | {school or 'School N/A'} | {position or '?'} | {year or '?'}"
        )
        if display in display_to_label:
            recruit_id = str(row.get("recruit_id") or "").strip()
            display = f"{display} | ID:{recruit_id or 'dup'}"
        display_to_label[display] = raw_label
        top_100_display_options.append(display)

    placeholder = "Select player from Top 100"
    selected_display = ""
    for display_text, raw_label in display_to_label.items():
        if raw_label == selected_label:
            selected_display = display_text
            break

    options = [placeholder] + top_100_display_options
    default_index = options.index(selected_display) if selected_display in options else 0
    top_100_selected = st.selectbox(
        "Top 100",
        options,
        index=default_index,
        key=f"{widget_prefix}_top_100_select",
    )
    if str(top_100_selected) != placeholder:
        selected_label = str(display_to_label.get(str(top_100_selected)) or "")
        st.session_state[selected_label_key] = selected_label

    search_text = st.text_input(
        "Search for other players",
        value=str(st.session_state.get(f"{widget_prefix}_candidate_query") or ""),
        placeholder="Type at least 3 letters to search outside the top 100",
        key=f"{widget_prefix}_candidate_query_input",
    )
    st.session_state[f"{widget_prefix}_candidate_query"] = str(search_text or "")

    search_df = pd.DataFrame()
    query = str(search_text or "").strip()
    if len(query) >= 3:
        try:
            search_df = search_recruit_candidate_matches(
                int(selected_year),
                str(selected_position),
                query,
                limit=limit,
            )
        except Exception as exc:
            st.error(f"Unable to search recruit candidates: {exc}")
            return base_df, {}

        if not search_df.empty:
            st.caption("Search matches")
            for idx, row in enumerate(search_df.to_dict(orient="records")[:40]):
                label = str(row.get("player_label") or "")
                hs_text = str(row.get("high_school") or "HS N/A")
                rating_text = str(row.get("rating") or "N/A")
                button_text = (
                    f"{str(row.get('player_name') or 'Unknown')} | "
                    f"{str(row.get('position') or '?')} | "
                    f"{hs_text} | Rating: {rating_text}"
                )
                if st.button(button_text, key=f"{widget_prefix}_search_pick_{idx}", width="stretch"):
                    selected_label = label
                    st.session_state[selected_label_key] = label
        else:
            st.info("No additional players matched your search.")

    if not selected_label:
        return base_df, {}

    selected_matches = base_df[base_df["player_label"] == selected_label]
    if selected_matches.empty and not search_df.empty:
        selected_matches = search_df[search_df["player_label"] == selected_label]
    if selected_matches.empty:
        st.session_state.pop(selected_label_key, None)
        return base_df, {}

    selected_row = dict(selected_matches.iloc[0].to_dict())
    st.success(f"Selected player: {selected_label}")
    return base_df, selected_row


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


def _render_json_lazy(payload: Any, key: str, label: str = "Render JSON") -> None:
    if st.checkbox(label, key=key):
        st.code(json.dumps(payload, indent=2, default=str), language="json")


def _target_team_name(team_option: str) -> str:
    text = str(team_option or "").strip()
    if "|" in text:
        return text.split("|", 1)[0].strip()
    return text


def _recruit_star_value(rating_value: Any) -> float | None:
    rating = to_float_or_none(rating_value)
    if rating is None:
        return None
    if rating >= 0.98:
        return 5.0
    if rating >= 0.90:
        return 4.0
    if rating >= 0.80:
        return 3.0
    if rating >= 0.70:
        return 2.0
    return None


def _star_icons(star_value: float | None) -> str:
    if star_value is None:
        return ""
    full = int(star_value)
    return "⭐" * full


def render_local_cfbd_debugger_page() -> None:
    st.subheader("Local Transfer Portal CFBD Debugger")
    st.caption(
        "Simplified debugger for Transfer Portal CFBD pulls: live player lookup -> athlete ID -> "
        "2025 usage plus all-years usage and all-years player season stats checks."
    )

    selected_position = st.selectbox(
        "Position Filter",
        [POSITION_FILTER_PLACEHOLDER] + POSITION_FILTER_OPTIONS,
        index=0,
        key="transfer_debug_position",
    )
    filtered_df, selected_row = _render_transfer_candidate_live_picker(
        widget_prefix="transfer_debug",
        selected_position=selected_position,
        limit=25,
    )
    selected_label = str(selected_row.get("player_label") or "")

    col1, col2 = st.columns(2)
    with col1:
        usage_year = st.number_input("Usage Year", min_value=2010, max_value=2035, value=2025, step=1, key="transfer_debug_usage_year")
    with col2:
        exclude_garbage_time = st.checkbox("Exclude garbage time", value=True, key="transfer_debug_exclude_gt")

    debug_state_key = "transfer_debugger_report_output"
    run_requested = st.button("Run Transfer Portal CFBD Pulls", type="primary", key="transfer_debug_run")

    if run_requested:
        if not selected_label:
            st.warning("Select a transfer candidate.")
            return

        selected_row = dict(selected_row or {})

        player_name = str(selected_row.get("player_name") or "").strip()
        athlete_id_text = str(selected_row.get("cfbd_athlete_id") or "").strip()
        position = str(selected_row.get("position") or "").strip()
        teams_text = str(selected_row.get("teams") or "").strip()
        team_filters = split_team_tokens_text(teams_text)

        first_season = pd.to_numeric(selected_row.get("first_season"), errors="coerce")
        last_season = pd.to_numeric(selected_row.get("last_season"), errors="coerce")

        if not athlete_id_text.isdigit():
            st.error(
                f"Selected player '{player_name}' does not have a numeric cfbd_athlete_id. "
                "Cannot run exact Transfer Portal CFBD pull test."
            )
            return

        with st.spinner("Running CFBD pulls for transfer verification..."):
            cfbd_context = orchestrate_transfer_cfbd_context(
                player_name=player_name,
                cfbd_athlete_id=athlete_id_text,
                position=position,
                teams=teams_text,
                year=int(usage_year),
                first_season=int(first_season) if pd.notna(first_season) else None,
                last_season=int(last_season) if pd.notna(last_season) else None,
                exclude_garbage_time=bool(exclude_garbage_time),
                progress_callback=None,
            )

        st.session_state[debug_state_key] = {
            "selected_label": selected_label,
            "player_name": player_name,
            "athlete_id_text": athlete_id_text,
            "athlete_id": int(athlete_id_text),
            "position": position,
            "teams_text": teams_text,
            "team_filters": team_filters,
            "first_season": int(first_season) if pd.notna(first_season) else None,
            "last_season": int(last_season) if pd.notna(last_season) else None,
            "usage_year": int(usage_year),
            "exclude_garbage_time": bool(exclude_garbage_time),
            "cfbd_context": cfbd_context,
        }

    debug_output = st.session_state.get(debug_state_key)
    if not isinstance(debug_output, dict):
        st.info("Run the debugger once, then you can interact with tables/charts without resetting output.")
        return

    selected_label = str(debug_output.get("selected_label") or "")
    player_name = str(debug_output.get("player_name") or "")
    athlete_id = int(debug_output.get("athlete_id") or 0)
    position = str(debug_output.get("position") or "")
    teams_text = str(debug_output.get("teams_text") or "")
    team_filters = list(debug_output.get("team_filters") or [])
    first_season = debug_output.get("first_season")
    last_season = debug_output.get("last_season")
    usage_year = int(debug_output.get("usage_year") or usage_year)

    cfbd_context = dict(debug_output.get("cfbd_context") or {})
    usage_2025_result = dict(cfbd_context.get("cfbd_usage_for_year") or {})
    career_usage_by_year = list(cfbd_context.get("cfbd_usage_career") or [])
    career_stats_by_year = list(cfbd_context.get("cfbd_stats_career") or [])
    usage_table_compact = list(cfbd_context.get("usage_table_compact") or [])
    usage_yoy_compact = list(cfbd_context.get("usage_yoy_compact") or [])
    season_stats_table_compact = list(cfbd_context.get("season_stats_table_compact") or [])
    pull_diagnostics = list(cfbd_context.get("pull_diagnostics") or [])
    pull_config = dict(cfbd_context.get("pull_config") or {})

    st.markdown("### Exact Mapping Confirmation")
    st.write(f"- Player label: {selected_label}")
    st.write(f"- Player name: {player_name or 'N/A'}")
    st.write(f"- Position: {position or 'N/A'}")
    st.write(f"- Team context (all teams): {', '.join(team_filters) if team_filters else 'N/A'}")
    st.write(f"- Mapped CFBD athlete ID: {athlete_id}")
    st.write(f"- Career season span from college table: {first_season if first_season is not None else 'N/A'} to {last_season if last_season is not None else 'N/A'}")

    st.markdown(f"### Pull 1: {usage_year} Usage")
    usage_rows = list(usage_2025_result.get("data") or []) if isinstance(usage_2025_result, dict) else []
    usage_meta = usage_2025_result.get("meta") if isinstance(usage_2025_result, dict) else {}
    usage_meta = usage_meta if isinstance(usage_meta, dict) else {}
    st.write(f"- Status: {usage_2025_result.get('status', 'unknown')}")
    st.write(f"- Reason: {usage_2025_result.get('reason', '')}")
    st.write(f"- Record count: {len(usage_rows)}")
    st.write(f"- Query URL: {_build_cfbd_debug_url(usage_meta)}")
    with st.expander("2025 Usage Raw Result"):
        _render_json_lazy(usage_2025_result, key="transfer_debug_usage_2025_json")

    st.markdown("### Pull Configuration")
    _render_json_lazy(pull_config, key="transfer_debug_pull_config_json")

    st.markdown("### Pull 2: Player Career Usage Stats")
    career_summary_df = pd.DataFrame(
        [
            {
                "year": row["year"],
                "status": row["status"],
                "record_count": row["record_count"],
                "reason": row["reason"],
            }
            for row in career_usage_by_year
        ]
    )
    if not career_summary_df.empty:
        st.dataframe(career_summary_df, width="stretch")
    else:
        st.write("No career seasons were available to test.")

    with st.expander("Career Usage Raw Results"):
        _render_json_lazy(career_usage_by_year, key="transfer_debug_career_usage_json")

    st.markdown("### Pull 3: Player Career Season Stats")
    career_stats_summary_df = pd.DataFrame(
        [
            {
                "year": row["year"],
                "status": row["status"],
                "record_count": row["record_count"],
                "raw_record_count": row["raw_record_count"],
                "reason": row["reason"],
            }
            for row in career_stats_by_year
        ]
    )
    if not career_stats_summary_df.empty:
        st.dataframe(career_stats_summary_df, width="stretch")
    else:
        st.write("No career season stats were available to test.")

    with st.expander("Career Season Stats Raw Results"):
        _render_json_lazy(career_stats_by_year, key="transfer_debug_career_stats_json")

    st.markdown("### Pull Diagnostics")
    diagnostics_df = rows_to_dynamic_table(
        pull_diagnostics,
        leading_columns=[
            "year",
            "endpoint",
            "status",
            "reason",
            "rows_pre_filter",
            "rows_post_filter",
            "queried_teams",
            "queried_team_count",
            "fallback_policy",
            "fallback_teamless_attempted",
            "params_text",
        ],
    )
    if not diagnostics_df.empty:
        st.dataframe(diagnostics_df, width="stretch")
    else:
        st.write("No diagnostics available.")

    st.markdown("### Compacted Payload Preview (Passed To Gemini)")
    st.caption("These are token-optimized structured payloads used for transfer synthesis. Garbage time exclusion is enabled by default.")
    debug_artifacts = _get_transfer_render_artifacts(
        {
            "cfbd_athlete_id": athlete_id,
            "target_team": "debugger",
            "year": usage_year,
            "pull_config": pull_config,
            "usage_table_compact": usage_table_compact,
            "usage_yoy_compact": usage_yoy_compact,
            "season_stats_table_compact": season_stats_table_compact,
        },
        position_hint=position,
    )
    st.markdown("#### Charts")
    _render_transfer_charts_side_by_side(section_key="transfer_debugger", artifacts=debug_artifacts)
    _render_transfer_tables(artifacts=debug_artifacts)

    with st.expander("Compact JSON Payloads"):
        st.markdown("#### Compact Usage Table JSON")
        _render_json_lazy(usage_table_compact, key="transfer_debug_usage_table_compact_json")
        st.markdown("#### Usage YoY Delta Table JSON")
        _render_json_lazy(usage_yoy_compact, key="transfer_debug_usage_yoy_compact_json")
        st.markdown("#### Compact Season Stats Table JSON")
        _render_json_lazy(season_stats_table_compact, key="transfer_debug_season_stats_compact_json")


def _safe_int_telemetry(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(float(value))
    except Exception:
        return 0


def _safe_float_telemetry(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _normalize_telemetry_payload(payload: dict[str, Any] | None, trace_log: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    src = dict(payload or {})
    branch_latency_raw = dict(src.get("branch_latency_ms") or {})
    branch_latency: dict[str, int] = {
        str(k): _safe_int_telemetry(v)
        for k, v in branch_latency_raw.items()
        if str(k).strip()
    }

    model_rollup_raw = dict(src.get("model_rollup") or {})
    model_rollup = {
        "model_call_count": _safe_int_telemetry(model_rollup_raw.get("model_call_count")),
        "input_tokens": _safe_int_telemetry(model_rollup_raw.get("input_tokens")),
        "output_tokens": _safe_int_telemetry(model_rollup_raw.get("output_tokens")),
        "total_tokens": _safe_int_telemetry(model_rollup_raw.get("total_tokens")),
        "estimated_cost_usd": round(_safe_float_telemetry(model_rollup_raw.get("estimated_cost_usd")), 8),
        "latency_ms": _safe_int_telemetry(model_rollup_raw.get("latency_ms")),
    }
    if model_rollup["total_tokens"] <= 0:
        model_rollup["total_tokens"] = model_rollup["input_tokens"] + model_rollup["output_tokens"]

    pipeline_latency_ms = _safe_int_telemetry(src.get("pipeline_latency_ms"))
    if pipeline_latency_ms <= 0:
        orchestration_latencies = [
            _safe_int_telemetry(row.get("latency_ms"))
            for row in list(trace_log or [])
            if str(row.get("node") or "").strip() == "orchestration"
        ]
        pipeline_latency_ms = max(orchestration_latencies) if orchestration_latencies else 0
        if pipeline_latency_ms > 0 and "orchestration" not in branch_latency:
            branch_latency["orchestration"] = pipeline_latency_ms

    return {
        "pipeline_latency_ms": pipeline_latency_ms,
        "branch_latency_ms": branch_latency,
        "model_telemetry": list(src.get("model_telemetry") or []),
        "model_rollup": model_rollup,
    }


def _render_telemetry_summary(telemetry: dict[str, Any] | None, key_prefix: str, title: str = "Telemetry") -> None:
    normalized = _normalize_telemetry_payload(telemetry)
    rollup = dict(normalized.get("model_rollup") or {})
    branch_latency = dict(normalized.get("branch_latency_ms") or {})
    pipeline_latency_ms = _safe_int_telemetry(normalized.get("pipeline_latency_ms"))

    with st.expander(title, expanded=False):
        st.markdown("#### Model Rollup")
        st.write(
            f"Calls: {_safe_int_telemetry(rollup.get('model_call_count'))} | "
            f"Tokens: {_safe_int_telemetry(rollup.get('total_tokens')):,} | "
            f"Cost: ${_safe_float_telemetry(rollup.get('estimated_cost_usd')):.4f} | "
            f"Pipeline latency: {pipeline_latency_ms:,} ms"
        )

        if branch_latency:
            branch_rows = [
                {"branch": key, "latency_ms": _safe_int_telemetry(value)}
                for key, value in branch_latency.items()
            ]
            branch_df = pd.DataFrame(branch_rows).sort_values("latency_ms", ascending=False)
            with st.expander("Branch Latency Breakdown", expanded=False):
                st.dataframe(branch_df, width="stretch", hide_index=True)

        with st.expander("Telemetry JSON", expanded=False):
            _render_json_lazy(normalized, key=f"{key_prefix}_telemetry_json")


st.set_page_config(page_title="Gridiron Intelligence - Scouting Workbench", page_icon="🏈", layout="wide")
st.markdown("<h1 class='football-title'>Gridiron Intelligence 🏈</h1>", unsafe_allow_html=True)
st.markdown("<p class='football-subtitle'>Interactive Scouting Workbench (Streamlit)</p>", unsafe_allow_html=True)

if "app_page" not in st.session_state:
    st.session_state["app_page"] = "Landing Page"

app_page = str(st.session_state.get("app_page") or "Landing Page")

if app_page == "Landing Page":
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

if app_page != "Landing Page":
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/shaverm96/Gridiron-Intelligence/main/Logos/Main.svg", width=150)
        st.title("Gridiron Intelligence")

        workspace_options = [
            "Recruiting Portal",
            "Transfer Portal",
        ]
        if _is_local_debug_page_enabled():
            workspace_options.append("Local CFBD Debugger")

        default_index = workspace_options.index(app_page) if app_page in workspace_options else 0
        app_page = st.radio("Workspace", workspace_options, index=default_index)
        st.session_state["app_page"] = app_page

        if st.button("Back To Landing Page", key="back_to_landing"):
            st.session_state["app_page"] = "Landing Page"
            st.rerun()

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
                    _render_json_lazy(debug_result, key="sidebar_local_cfbd_debug_result_json")

        supabase_issues = get_supabase_config_issues()
        pricing_issues = get_model_pricing_issues()
        if supabase_issues or pricing_issues:
            st.warning("Configuration preflight issues detected. Open diagnostics for details.")


@st.cache_resource
def get_cached_agent_graph():
    return get_scout_graph()


@st.cache_resource
def get_cached_structured_web_graph():
    return get_structured_web_graph()


def _usage_delta_cell_style(value: Any) -> str:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        numeric = float(value)
    except Exception:
        return ""
    if numeric > 0:
        return "color: #0f9d58; font-weight: 600;"
    if numeric < 0:
        return "color: #c62828; font-weight: 600;"
    return ""


def _transfer_report_cache_key(report_output: dict[str, Any]) -> str:
    pull_config = dict(report_output.get("pull_config") or {})
    return "|".join(
        [
            str(report_output.get("cfbd_athlete_id") or ""),
            str(report_output.get("target_team") or ""),
            str(report_output.get("year") or ""),
            str(bool(pull_config.get("exclude_garbage_time", True))),
        ]
    )


def _get_transfer_render_artifacts(
    report_output: dict[str, Any],
    position_hint: str,
) -> dict[str, Any]:
    cache_store = dict(st.session_state.get("transfer_render_cache") or {})
    cache_key = _transfer_report_cache_key(report_output)
    cached = cache_store.get(cache_key)
    if isinstance(cached, dict):
        return cached

    usage_table_compact = list(report_output.get("usage_table_compact") or [])
    usage_yoy_compact = list(report_output.get("usage_yoy_compact") or [])
    season_stats_table_compact = list(report_output.get("season_stats_table_compact") or [])

    usage_display_df, usage_cols, delta_cols = build_transfer_usage_with_yoy_table(
        usage_table_compact=usage_table_compact,
        usage_yoy_compact=usage_yoy_compact,
        position_hint=position_hint,
    )
    season_stats_df = rows_to_dynamic_table(
        season_stats_table_compact,
        leading_columns=["year", "team", "games"],
    )
    stat_cols = [col for col in season_stats_df.columns if col not in {"year", "team", "games"}]
    ordered_stats = transfer_position_stat_order(position_hint, stat_cols)
    ordered_stat_cols = [col for col in ["year", "team", "games", *ordered_stats] if col in season_stats_df.columns]

    artifact = {
        "usage_display_df": usage_display_df,
        "usage_cols": usage_cols,
        "delta_cols": delta_cols,
        "season_stats_df": season_stats_df,
        "ordered_stat_cols": ordered_stat_cols,
    }
    cache_store[cache_key] = artifact
    st.session_state["transfer_render_cache"] = cache_store
    return artifact


def _render_transfer_usage_chart(section_key: str, artifacts: dict[str, Any]) -> None:
    usage_display_df = artifacts.get("usage_display_df")
    usage_cols = list(artifacts.get("usage_cols") or [])
    if alt is None or not isinstance(usage_display_df, pd.DataFrame) or usage_display_df.empty:
        st.write("No usage rows available for charting.")
        return

    usage_chart_cols: list[str] = []
    for col in usage_cols:
        if col not in usage_display_df.columns:
            continue
        metric_series = pd.to_numeric(usage_display_df[col], errors="coerce")
        if metric_series.replace([np.inf, -np.inf], np.nan).notna().any():
            usage_chart_cols.append(col)
    if not usage_chart_cols:
        st.write("No usage metrics available for charting.")
        return

    metric_key = f"{section_key}_usage_metric"
    usage_metric = st.selectbox("Usage metric", usage_chart_cols, key=metric_key)
    usage_plot_df = usage_display_df[["year", usage_metric]].copy()
    usage_plot_df["year"] = pd.to_numeric(usage_plot_df["year"], errors="coerce")
    usage_plot_df[usage_metric] = pd.to_numeric(usage_plot_df[usage_metric], errors="coerce")
    finite_mask = np.isfinite(usage_plot_df["year"].to_numpy()) & np.isfinite(usage_plot_df[usage_metric].to_numpy())
    usage_plot_df = usage_plot_df.loc[finite_mask].copy()
    if usage_plot_df.empty:
        st.write("No usage points available for selected metric.")
        return
    usage_plot_df["year"] = usage_plot_df["year"].astype(int)
    usage_plot_df = usage_plot_df.sort_values("year").drop_duplicates(subset=["year"], keep="last")
    if usage_plot_df.empty:
        st.write("No usage points available for selected metric.")
        return
    usage_plot_df["usage_label"] = usage_plot_df[usage_metric].map(lambda v: f"{float(v):.1f}")

    base = alt.Chart(usage_plot_df).encode(
        x=alt.X("year:Q", title="Year", axis=alt.Axis(format="d")),
        y=alt.Y(f"{usage_metric}:Q", title="Usage %", axis=alt.Axis(format=".1f")),
        tooltip=[alt.Tooltip("year:Q", title="Year", format="d"), alt.Tooltip(f"{usage_metric}:Q", title="Usage %", format=".1f")],
    )
    line = base.mark_line()
    points = base.mark_point(size=80, filled=True)
    point_labels = base.mark_text(dy=14, baseline="top", fontSize=14, fontWeight="bold", color="#FFFFFF").encode(
        text=alt.Text("usage_label:N")
    )
    chart = (line + points + point_labels).properties(height=260)
    st.altair_chart(chart, width="stretch")


def _render_transfer_stat_bar_chart(section_key: str, artifacts: dict[str, Any]) -> None:
    season_stats_df = artifacts.get("season_stats_df")
    ordered_stat_cols = list(artifacts.get("ordered_stat_cols") or [])
    if alt is None or not isinstance(season_stats_df, pd.DataFrame) or season_stats_df.empty:
        st.write("No season-stat rows available for charting.")
        return

    stat_cols: list[str] = []
    for col in ordered_stat_cols:
        if col in {"year", "team", "games"} or col not in season_stats_df.columns:
            continue
        metric_series = pd.to_numeric(season_stats_df[col], errors="coerce")
        if metric_series.replace([np.inf, -np.inf], np.nan).notna().any():
            stat_cols.append(col)
    if not stat_cols:
        st.write("No season-stat metrics available for charting.")
        return

    metric_key = f"{section_key}_stat_metric"
    stat_metric = st.selectbox("Season stat metric", stat_cols, key=metric_key)
    stat_plot_df = season_stats_df[["year", stat_metric]].copy()
    stat_plot_df["year"] = pd.to_numeric(stat_plot_df["year"], errors="coerce")
    stat_plot_df[stat_metric] = pd.to_numeric(stat_plot_df[stat_metric], errors="coerce")
    finite_mask = np.isfinite(stat_plot_df["year"].to_numpy()) & np.isfinite(stat_plot_df[stat_metric].to_numpy())
    stat_plot_df = stat_plot_df.loc[finite_mask].copy()
    if stat_plot_df.empty:
        st.write("No numeric values available for selected stat metric.")
        return
    stat_plot_df["year"] = stat_plot_df["year"].astype(int)
    # Aggregate to a single bar per season to avoid repeated year labels when multiple team rows exist.
    stat_plot_df = (
        stat_plot_df.groupby("year", as_index=False)[stat_metric]
        .max()
        .sort_values("year")
    )
    if stat_plot_df.empty:
        st.write("No numeric values available for selected stat metric.")
        return

    stat_plot_df["baseline"] = 0.0
    stat_plot_df["stat_label"] = stat_plot_df[stat_metric].map(lambda v: f"{float(v):.1f}")
    base = alt.Chart(stat_plot_df).encode(
        x=alt.X("year:O", title="Year"),
        tooltip=[alt.Tooltip("year:Q", title="Year", format="d"), alt.Tooltip(f"{stat_metric}:Q", title="Value", format=".1f")],
    )
    bars = base.mark_bar(size=48).encode(
        y=alt.Y(f"{stat_metric}:Q", title=stat_metric.replace("_", " ").title())
    )
    bar_labels = base.mark_text(dy=-4, baseline="bottom", fontSize=14, fontWeight="bold", color="#111827").encode(
        y=alt.Y("baseline:Q"),
        text=alt.Text("stat_label:N"),
    )
    chart = (bars + bar_labels).properties(height=260)
    st.altair_chart(chart, width="stretch")


def _render_transfer_charts_side_by_side(section_key: str, artifacts: dict[str, Any]) -> None:
    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("#### Usage Line Chart")
        _render_transfer_usage_chart(section_key=section_key, artifacts=artifacts)
    with right_col:
        st.markdown("#### Season Stat Bar Chart")
        _render_transfer_stat_bar_chart(section_key=section_key, artifacts=artifacts)


def _render_transfer_tables(artifacts: dict[str, Any]) -> None:
    usage_display_df = artifacts.get("usage_display_df")
    usage_cols = list(artifacts.get("usage_cols") or [])
    delta_cols = list(artifacts.get("delta_cols") or [])
    season_stats_df = artifacts.get("season_stats_df")
    ordered_stat_cols = list(artifacts.get("ordered_stat_cols") or [])

    st.markdown("### Usage Table")
    if not isinstance(usage_display_df, pd.DataFrame) or usage_display_df.empty:
        st.write("No usage rows available.")
    else:
        usage_view_df = usage_display_df.drop(columns=[col for col in ["record_count", "status"] if col in usage_display_df.columns])
        subset_delta = [col for col in delta_cols if col in usage_display_df.columns]
        percent_cols = [col for col in usage_cols + subset_delta if col in usage_view_df.columns]
        styled_usage = usage_view_df.style.hide(axis="index")
        if percent_cols:
            styled_usage = styled_usage.format({col: "{:.1f}%" for col in percent_cols})
        if subset_delta:
            styled_usage = styled_usage.map(_usage_delta_cell_style, subset=subset_delta)
        styled_usage = styled_usage.set_properties(**{
            "font-size": "0.97rem",
            "font-weight": "600",
            "color": "#F9FAFB",
        }).set_table_styles(
            [
                {"selector": "th", "props": "background: #0f172a; color: #e2e8f0; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;"},
                {"selector": "td", "props": "background: #111827; border-bottom: 1px solid #1f2937;"},
            ]
        )
        st.dataframe(styled_usage, width="stretch")

    st.markdown("### Season Stats Table")
    if not isinstance(season_stats_df, pd.DataFrame) or season_stats_df.empty:
        st.write("No season stat rows available.")
    else:
        display_cols = ordered_stat_cols if ordered_stat_cols else list(season_stats_df.columns)
        stats_view_df = season_stats_df[display_cols].drop(columns=[col for col in ["record_count", "status"] if col in season_stats_df.columns], errors="ignore")
        stats_formatters = {}
        for col in stats_view_df.columns:
            if pd.api.types.is_numeric_dtype(stats_view_df[col]):
                stats_formatters[col] = "{:.3f}" if "pct" in str(col).lower() else "{:.0f}"

        styled_stats = stats_view_df.style.hide(axis="index")
        if stats_formatters:
            styled_stats = styled_stats.format(stats_formatters)
        styled_stats = styled_stats.set_properties(**{
            "font-size": "0.97rem",
            "font-weight": "600",
            "color": "#F9FAFB",
        }).set_table_styles(
            [
                {"selector": "th", "props": "background: #0f172a; color: #e2e8f0; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em;"},
                {"selector": "td", "props": "background: #111827; border-bottom: 1px solid #1f2937;"},
            ]
        )
        st.dataframe(styled_stats, width="stretch")


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


@st.cache_data
def _asset_data_uri(relative_path: tuple[str, ...], mime_type: str = "image/png") -> str:
    return image_data_uri_data(project_root=PROJECT_ROOT, relative_path=relative_path, mime_type=mime_type)


def _build_recruiting_layout_safe(raw_text: str | None, context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build recruiting layout with backwards-compatible signature handling and schema guards."""
    builder = build_recruiting_summary_layout_data
    result: Any

    try:
        result = builder(raw_text, context=context)
    except TypeError as exc:
        # Backward compatibility: some environments may still have the pre-context signature.
        if "context" in str(exc).lower() and "unexpected" in str(exc).lower():
            result = builder(raw_text)
        else:
            raise

    if not isinstance(result, dict):
        result = {}

    notes = result.get("notes")
    if not isinstance(notes, list):
        notes = parse_summary_notes_data(raw_text)

    grid_items_raw = result.get("grid_items")
    grid_items: list[dict[str, str]] = []
    if isinstance(grid_items_raw, list):
        for row in grid_items_raw:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            value = str(row.get("value") or "").strip()
            if not title or not value:
                continue
            grid_items.append(
                {
                    "key": str(row.get("key") or "").strip(),
                    "title": title,
                    "value": value,
                }
            )

    return {
        "hero_name": str(result.get("hero_name") or "").strip(),
        "hero_subtitle": str(result.get("hero_subtitle") or "").strip(),
        "physical_profile": str(result.get("physical_profile") or "").strip(),
        "grid_items": grid_items,
        "note_on_recency": str(result.get("note_on_recency") or "").strip(),
        "notes": notes,
    }


def _render_recruiting_summary_card(raw_text: str | None, context: dict[str, Any] | None = None) -> None:
    data = _build_recruiting_layout_safe(raw_text, context=context)
    notes = list(data.get("notes") or [])
    if not notes:
        st.markdown("### Recruiting Scout Summary")
        st.markdown(str(raw_text or "No recruiting summary available."))
        return

    ctx = context if isinstance(context, dict) else {}
    context_name = str(ctx.get("player_name") or "").strip()
    context_position = str(ctx.get("position") or "").strip()
    context_school = str(ctx.get("high_school") or "").strip()
    context_year = str(ctx.get("selected_year") or "").strip()

    hero_name = str(data.get("hero_name") or "").strip()
    hero_subtitle = str(data.get("hero_subtitle") or "").strip()

    if not hero_name or hero_name.lower() == "prospect" or len(hero_name.split()) > 6:
        hero_name = context_name or "Prospect"

    context_subtitle_parts = [part for part in [context_position, context_school, (f"Class {context_year}" if context_year else "")] if part]
    context_subtitle = " | ".join(context_subtitle_parts)

    # Avoid prose-style hero subtitles when we have structured player metadata.
    if context_subtitle and (
        not hero_subtitle
        or " is " in hero_subtitle.lower()
        or "commit for" in hero_subtitle.lower()
        or "committed to" in hero_subtitle.lower()
        or len(hero_subtitle) > 120
    ):
        hero_subtitle = context_subtitle

    if not hero_subtitle:
        hero_subtitle = context_subtitle

    physical_profile = str(data.get("physical_profile") or "").strip() or "Physical profile unavailable"
    grid_items = list(data.get("grid_items") or [])
    recency_note = str(data.get("note_on_recency") or "").strip()
    helmet_data_uri = _asset_data_uri(("Logos", "Helmate.png"))
    football_data_uri = _asset_data_uri(("Logos", "Football.png"))
    helmet_html = (
        f"<img class='recruiting-dossier-helmet-img' src='{helmet_data_uri}' alt='Helmet' />"
        if helmet_data_uri
        else "<div class='recruiting-dossier-helmet-fallback'>🏈</div>"
    )

    grid_parts: list[str] = []
    for item in grid_items:
        item_key = str(item.get("key") or "")
        icon_html = ""
        if item_key == "performance_notes" and football_data_uri:
            icon_html = (
                "<span class='recruiting-dossier-note-icon'>"
                f"<img class='recruiting-dossier-note-icon-img' src='{football_data_uri}' alt='Football' />"
                "</span>"
            )

        grid_parts.append(
            (
                "<article class='recruiting-dossier-note'>"
                "<div class='recruiting-dossier-note-head'>"
                f"<h4 class='recruiting-dossier-note-title'>{html.escape(str(item.get('title') or 'Note'))}</h4>"
                "</div>"
                "<div class='recruiting-dossier-note-body'>"
                f"{icon_html}"
                f"<span>{html.escape(str(item.get('value') or ''))}</span>"
                "</div>"
                "</article>"
            )
        )

    grid_html = "".join(grid_parts)

    recency_html = (
        (
            "<article class='recruiting-dossier-recency'>"
            "<h4 class='recruiting-dossier-note-title'>Note On Recency</h4>"
            f"<p class='recruiting-dossier-recency-body'>{html.escape(recency_note)}</p>"
            "</article>"
        )
        if recency_note
        else ""
    )

    st.markdown(
        (
            "<section class='recruiting-dossier-card'>"
            "<header class='recruiting-dossier-header recruiting-dossier-kpi-surface'>"
            "<h3 class='recruiting-dossier-title'>Recruiting Scout Summary</h3>"
            "</header>"
            "<div class='recruiting-dossier-hero recruiting-dossier-kpi-surface'>"
            "<div class='recruiting-dossier-hero-left'>"
            f"<div class='recruiting-dossier-helmet'>{helmet_html}</div>"
            "<div class='recruiting-dossier-hero-copy'>"
            f"<div class='recruiting-dossier-player'>{html.escape(hero_name)}</div>"
            f"<div class='recruiting-dossier-player-meta'>{html.escape(hero_subtitle)}</div>"
            "</div>"
            "</div>"
            f"<div class='recruiting-dossier-physical'>{html.escape(physical_profile)}</div>"
            "</div>"
            f"<div class='recruiting-dossier-grid'>{grid_html}</div>"
            f"{recency_html}"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_structured_summary_card(
    title: str,
    raw_text: str | None,
    section_key: str,
    compact: bool = False,
    context: dict[str, Any] | None = None,
) -> None:
    if section_key == "recruiting":
        _render_recruiting_summary_card(raw_text, context=context)
        return

    notes = parse_summary_notes_data(raw_text)
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


def _make_milestone_renderer(milestone_slot: Any, labels: dict[str, str], workflow_label: str) -> Any:
    def _render_milestone(event: dict[str, str]) -> None:
        node = str(event.get("node") or "workflow")
        status = str(event.get("status") or "running")
        node_label = labels.get(node, node.replace("_", " ").title())
        if node == "workflow":
            node_label = workflow_label
        if status == "completed":
            milestone_slot.success(f"{node_label}: complete")
        else:
            milestone_slot.info(f"{node_label}: running")

    return _render_milestone


def _render_structured_report_kpi_cards(report_output: dict[str, Any]) -> None:
    predicted_score_display = extract_predicted_score_display_data(
        score_card_html=str(report_output.get("score_card_html") or ""),
        pred_score_row=report_output.get("pred_score_row") or {},
        to_float_or_none=to_float_or_none,
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


def _render_structured_historical_comparables(report_output: dict[str, Any]) -> None:
    st.markdown("### Historical Comparables")
    comps_data = parse_historical_comparables_md_data(
        report_output.get("historical_comparables_md"),
        to_float_or_none=to_float_or_none,
    )
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


FINAL_SYNTH_SECTION_TITLES = [
    "Player Snapshot",
    "Trait Evaluation",
    "Scheme and Team Fit",
    "Development Risks",
    "Final Recommendation and Confidence",
]


def _clean_final_synth_line(line: str) -> str:
    text = str(line or "").strip()
    text = re.sub(r"[*_`]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_final_synth_heading(line: str) -> str | None:
    clean = _clean_final_synth_line(line)
    clean = re.sub(r"^#+\s*", "", clean)
    clean = re.sub(r"^\d+\s*[\)\.-]\s*", "", clean)
    clean = clean.strip(" :")
    for title in FINAL_SYNTH_SECTION_TITLES:
        if clean.lower() == title.lower():
            return title
    return None


def _parse_final_synthesis_sections(final_report: str | None) -> dict[str, Any]:
    raw = str(final_report or "")
    lines = [line.rstrip() for line in raw.splitlines()]
    sections: dict[str, list[str]] = {title: [] for title in FINAL_SYNTH_SECTION_TITLES}
    current_section: str | None = None

    for line in lines:
        heading = _normalize_final_synth_heading(line)
        if heading:
            current_section = heading
            continue
        if current_section is None:
            continue
        if not str(line).strip():
            continue
        sections[current_section].append(str(line).strip())

    ordered_sections = [
        {"title": title, "lines": lines_for_title}
        for title, lines_for_title in sections.items()
        if any(str(item).strip() for item in lines_for_title)
    ]
    return {"sections": ordered_sections, "raw": raw}


def _split_kv_and_body(lines: list[str]) -> tuple[list[tuple[str, str]], list[str]]:
    kv: list[tuple[str, str]] = []
    body: list[str] = []
    for line in lines:
        clean = _clean_final_synth_line(re.sub(r"^\s*[-*•]+\s*", "", str(line or "")))
        if not clean:
            continue
        if ":" in clean:
            left, right = clean.split(":", 1)
            key = left.strip()
            value = right.strip()
            if key and value and len(key) <= 40:
                kv.append((key, value))
                continue
        body.append(clean)
    return kv, body


def _is_bullet_line(line: str) -> bool:
    return bool(re.match(r"^\s*[-*•]+\s+", str(line or "")))


def _render_final_synthesis(report_output: dict[str, Any], player_name: str) -> None:
    final_text = str(report_output.get("final_report") or "").strip()
    if not final_text:
        st.markdown("No final synthesis generated.")
        return

    parsed = _parse_final_synthesis_sections(final_text)
    sections = list(parsed.get("sections") or [])
    if not sections:
        st.markdown(final_text)
        return

    col_left, col_right = st.columns([4, 1])
    with col_left:
        st.caption("Executive scouting report")

    with col_right:
        docx_bytes = _build_final_synthesis_docx_bytes(parsed=parsed, player_name=player_name)
        safe_name = re.sub(r"[^a-z0-9]+", "_", str(player_name or "player").strip().lower()).strip("_") or "player"
        if docx_bytes:
            st.download_button(
                "Download .docx",
                data=docx_bytes,
                file_name=f"{safe_name}_final_synthesis.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_final_synthesis_docx",
            )
        else:
            st.caption("Install python-docx to enable export.")

    report_html_parts: list[str] = [
        "<div class='final-synth-report'>",
        "<div class='final-synth-headline'>",
        "<div class='final-synth-headline-label'>Executive scouting report</div>",
        f"<div class='final-synth-headline-player'>{html.escape(str(player_name or ''))}</div>",
        "</div>",
    ]

    for idx, section in enumerate(sections):
        title = str(section.get("title") or "Section").strip()
        lines = [str(line) for line in list(section.get("lines") or []) if str(line).strip()]
        kv, body = _split_kv_and_body(lines)
        section_classes = "final-synth-section"
        if title == "Final Recommendation and Confidence":
            section_classes += " final-synth-conclusion"

        report_html_parts.append(f"<section class='{section_classes}'><h4>{html.escape(title)}</h4>")

        if title == "Player Snapshot" and kv:
            kv_html = "".join(
                [
                    (
                        "<div class='final-synth-kv-item'>"
                        f"<span class='final-synth-kv-label'>{html.escape(key)}</span>"
                        f"<span class='final-synth-kv-value'>{html.escape(value)}</span>"
                        "</div>"
                    )
                    for key, value in kv
                ]
            )
            report_html_parts.append(f"<div class='final-synth-kv-grid'>{kv_html}</div>")
            for paragraph in body:
                p_text = _clean_final_synth_line(paragraph)
                if p_text:
                    report_html_parts.append(f"<p class='final-synth-paragraph'>{html.escape(p_text)}</p>")
        else:
            bullets: list[str] = []
            subblocks: list[tuple[str, str]] = []
            paragraphs: list[str] = []

            for line in lines:
                clean = _clean_final_synth_line(re.sub(r"^\s*[-*•]+\s*", "", line))
                if not clean:
                    continue

                if _is_bullet_line(line):
                    bullets.append(clean)
                    continue

                if ":" in clean:
                    left, right = clean.split(":", 1)
                    left_key = left.strip()
                    right_value = right.strip()
                    if left_key and right_value and len(left_key) <= 36:
                        subblocks.append((left_key, right_value))
                        continue

                paragraphs.append(clean)

            for label, value in subblocks:
                report_html_parts.append(
                    (
                        "<div class='final-synth-subblock'>"
                        f"<span class='final-synth-subblock-title'>{html.escape(label)}</span>"
                        f"<span class='final-synth-subblock-body'>{html.escape(value)}</span>"
                        "</div>"
                    )
                )

            for paragraph in paragraphs:
                report_html_parts.append(f"<p class='final-synth-paragraph'>{html.escape(paragraph)}</p>")

            if bullets:
                bullets_html = "".join([f"<li>{html.escape(item)}</li>" for item in bullets if item])
                report_html_parts.append(f"<ul class='final-synth-bullets'>{bullets_html}</ul>")

        report_html_parts.append("</section>")
        if idx < len(sections) - 1:
            report_html_parts.append("<div class='final-synth-divider'></div>")

    report_html_parts.append("</div>")
    st.markdown("".join(report_html_parts), unsafe_allow_html=True)


def _extract_recommendation_confidence_from_final_report(
    final_report: str | None,
    parsed_sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    sections = list(parsed_sections or _parse_final_synthesis_sections(final_report).get("sections") or [])
    section = next(
        (item for item in sections if str(item.get("title") or "").strip().lower() == "final recommendation and confidence"),
        None,
    )
    lines = [str(line) for line in list((section or {}).get("lines") or []) if str(line).strip()]
    kv_pairs, body_lines = _split_kv_and_body(lines)
    kv_map = {str(k).strip().lower(): str(v).strip() for k, v in kv_pairs}

    recommendation = (
        kv_map.get("final recommendation")
        or kv_map.get("recommendation")
        or kv_map.get("verdict")
        or ""
    )
    confidence = (
        kv_map.get("confidence")
        or kv_map.get("rating confidence")
        or ""
    )
    return {
        "recommendation": recommendation,
        "confidence": confidence,
        "section_lines": lines,
        "section_body": body_lines,
    }


def _build_active_recruiting_report_context(report_output: dict[str, Any], persona: str) -> dict[str, Any]:
    if not isinstance(report_output, dict):
        return {}

    player_name = str(report_output.get("player_name") or "").strip()
    position = str(report_output.get("position") or "").strip()
    high_school = str(report_output.get("high_school") or "").strip()
    recruit_id = str(report_output.get("recruit_id") or "").strip()
    selected_year = int(report_output.get("selected_year") or 0)
    target_team = str(report_output.get("target_team") or "").strip()

    comps_data = parse_historical_comparables_md_data(
        report_output.get("historical_comparables_md"),
        to_float_or_none=to_float_or_none,
    )
    final_report = str(report_output.get("final_report") or "").strip()
    final_sections = list(_parse_final_synthesis_sections(final_report).get("sections") or [])
    final_reco = _extract_recommendation_confidence_from_final_report(
        final_report,
        parsed_sections=final_sections,
    )
    predicted_score_display = extract_predicted_score_display_data(
        score_card_html=str(report_output.get("score_card_html") or ""),
        pred_score_row=report_output.get("pred_score_row") or {},
        to_float_or_none=to_float_or_none,
    )

    tier_value = "N/A"
    score_for_tier = to_float_or_none(predicted_score_display)
    if score_for_tier is not None:
        tier_value = score_tier(score_for_tier)

    recruiting_summary = str(report_output.get("web_recruiting_summary") or "")
    recruiting_layout = _build_recruiting_layout_safe(
        recruiting_summary,
        context={
            "player_name": player_name,
            "position": position,
            "high_school": high_school,
            "selected_year": selected_year,
        },
    )

    comparable_rows = list(comps_data.get("rows") or [])
    normalized_comparables: list[dict[str, Any]] = []
    for idx, row in enumerate(comparable_rows, start=1):
        if not isinstance(row, dict):
            continue
        normalized_comparables.append(
            {
                "index": idx,
                "name": str(row.get("name") or "").strip(),
                "match_pct_display": str(row.get("match") or "").strip(),
                "class": str(row.get("year") or "").strip(),
                "state": str(row.get("state") or "").strip(),
                "rating": str(row.get("rating") or "").strip(),
                "match_value": row.get("match_value"),
            }
        )

    return {
        "context_type": "recruiting_structured_report",
        "player_name": player_name,
        "position": position,
        "high_school": high_school,
        "recruit_id": recruit_id,
        "selected_year": selected_year,
        "target_team": target_team,
        "persona": str(persona or "Scout"),
        "comparables_target_position": str(comps_data.get("target_position") or "").strip(),
        "comparables": normalized_comparables,
        "comparables_list": comparable_rows,
        "scorecard": {
            "predicted_score_display": predicted_score_display,
            "tier": tier_value,
            "pred_score_row": dict(report_output.get("pred_score_row") or {}),
            "score_card_html": str(report_output.get("score_card_html") or ""),
        },
        "recruiting_summary": recruiting_summary,
        "team_summary": str(report_output.get("web_team_summary") or ""),
        "recruiting_summary_layout": recruiting_layout,
        "final_synthesis": {
            "raw": final_report,
            "sections": final_sections,
            "recommendation": str(final_reco.get("recommendation") or ""),
            "confidence": str(final_reco.get("confidence") or ""),
            "recommendation_section_lines": list(final_reco.get("section_lines") or []),
        },
        "player_profile": dict(report_output.get("player_profile") or report_output.get("player_row") or {}),
        "scouting_clean": dict(report_output.get("scouting_clean") or {}),
    }


def _sync_recruiting_chat_state_with_report(
    state: dict[str, Any] | None,
    report_output: dict[str, Any],
    persona: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    active_context = _build_active_recruiting_report_context(report_output, persona=persona)
    base_state = dict(state or initial_chat_state(""))
    base_state["active_report_context"] = active_context
    base_state["target_player_name"] = str(report_output.get("player_name") or "")
    base_state["player_name"] = str(report_output.get("player_name") or "")
    base_state["recruit_id"] = str(report_output.get("recruit_id") or "")
    base_state["target_team"] = str(report_output.get("target_team") or "")
    base_state["year"] = int(report_output.get("selected_year") or 0)
    return base_state, active_context


def _chat_context_snapshot(active_report_context: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(active_report_context or {})
    return {
        "player_name": str(src.get("player_name") or "").strip(),
        "target_team": str(src.get("target_team") or "").strip(),
        "recruit_id": str(src.get("recruit_id") or "").strip(),
        "selected_year": int(src.get("selected_year") or 0),
        "persona": str(src.get("persona") or "Scout"),
    }


def _append_chat_transcript_turn(
    transcript_key: str,
    user_message: str,
    assistant_response: str,
    persona: str,
    context_snapshot: dict[str, Any] | None = None,
) -> None:
    transcript = list(st.session_state.get(transcript_key, []))
    turn_id = len(transcript) + 1
    transcript.append(
        {
            "turn_id": turn_id,
            "user_message": str(user_message or ""),
            "assistant_response": str(assistant_response or ""),
            "persona": str(persona or "Scout"),
            "sequence": turn_id,
            "timestamp_unix": int(time.time()),
            "context_snapshot": dict(context_snapshot or {}),
        }
    )
    st.session_state[transcript_key] = transcript


def _history_meta_response(user_prompt: str, transcript: list[dict[str, Any]]) -> str | None:
    q = str(user_prompt or "").strip().lower()
    if not q:
        return None

    asks_first_question = any(
        token in q
        for token in [
            "what was my first question",
            "what did i ask first",
            "show me my first question",
            "first thing i asked",
        ]
    )
    asks_last_question = any(
        token in q
        for token in [
            "what was my last question",
            "what did i ask last",
        ]
    )
    asks_previous_turn = any(
        token in q
        for token in [
            "what did i ask before that",
            "what did you say before",
            "show me the answer you gave me earlier",
            "show me the first answer",
        ]
    )

    if not (asks_first_question or asks_last_question or asks_previous_turn):
        return None

    turns = [row for row in list(transcript or []) if isinstance(row, dict)]
    if not turns:
        return "There is no earlier question in this current chat session yet."

    def _format_turn(prefix: str, turn: dict[str, Any]) -> str:
        user_text = str(turn.get("user_message") or "").strip()
        answer_text = str(turn.get("assistant_response") or "").strip()
        if not answer_text:
            answer_text = "Original output is unavailable for that turn."
        return (
            f"{prefix}: \"{user_text}\"\n\n"
            f"Associated output from that turn:\n{answer_text}"
        )

    if asks_first_question:
        return _format_turn("Your first question in this chat was", turns[0])

    if asks_last_question:
        return _format_turn("Your last question in this chat was", turns[-1])

    if asks_previous_turn:
        if len(turns) < 2:
            return "There is no earlier turn before the current one in this chat session."
        return _format_turn("The previous question before your most recent turn was", turns[-2])

    return None


def _build_final_synthesis_docx_bytes(parsed: dict[str, Any], player_name: str) -> bytes | None:
    if Document is None:
        return None

    doc = Document()
    doc.add_heading("Final Synthesis", level=0)
    if str(player_name or "").strip():
        doc.add_paragraph(f"Player: {str(player_name).strip()}")

    sections = list(parsed.get("sections") or [])
    for section in sections:
        title = str(section.get("title") or "Section").strip()
        lines = [str(line) for line in list(section.get("lines") or []) if str(line).strip()]
        if not lines:
            continue

        doc.add_heading(title, level=1)
        kv, body = _split_kv_and_body(lines)

        if title == "Player Snapshot" and kv:
            for key, value in kv:
                p = doc.add_paragraph()
                p.add_run(f"{key}: ").bold = True
                p.add_run(value)
            for paragraph in body:
                p_text = _clean_final_synth_line(paragraph)
                if p_text:
                    doc.add_paragraph(p_text)
            continue

        for line in lines:
            clean = _clean_final_synth_line(re.sub(r"^\s*[-*•]+\s*", "", line))
            if not clean:
                continue
            if _is_bullet_line(line):
                doc.add_paragraph(clean, style="List Bullet")
            else:
                doc.add_paragraph(clean)

    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def render_landing_page() -> None:
    st.subheader("Welcome To Gridiron Intelligence")
    st.markdown(
        """
Gridiron Intelligence combines model-based analysis and web intelligence to support college football scouting decisions.

Choose one of the two portal workflows:
1. Recruiting Portal: evaluate high school recruits for team fit and long-term projection.
2. Transfer Portal: evaluate college players from the 2025 season for likely transfer impact.
        """
    )

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("### Recruiting Portal")
        st.write("Build a structured scouting report for recruits and ask follow-up questions.")
        if st.button("Open Recruiting Portal", type="primary", key="landing_open_recruiting"):
            st.session_state["app_page"] = "Recruiting Portal"
            st.rerun()

    with right_col:
        st.markdown("### Transfer Portal")
        st.write("Assess transfer fit, impact, and transfer-likelihood context for college players.")
        if st.button("Open Transfer Portal", type="primary", key="landing_open_transfer"):
            st.session_state["app_page"] = "Transfer Portal"
            st.rerun()


def render_structured_report_with_chat_page() -> None:
    st.subheader("Recruiting Portal")
    report_state_key = "structured_chat_report_output"

    selected_year = st.selectbox("Recruiting Class Year", CONFIG["YEARS"], index=0, key="structured_chat_year")
    selected_position = st.selectbox(
        "Position Filter",
        [POSITION_FILTER_PLACEHOLDER] + RECRUIT_POSITION_OPTIONS,
        index=0,
        key="structured_chat_position",
    )
    candidate_df, selected_row = _render_recruit_candidate_live_picker(
        widget_prefix="structured_chat",
        selected_year=int(selected_year),
        selected_position=str(selected_position),
        limit=100,
    )
    selected_label = str(selected_row.get("player_label") or "")
    target_team_option = st.selectbox("Target Team", TARGET_TEAMS, index=0, key="structured_chat_target_team")
    target_team = _target_team_name(target_team_option)

    if st.button("Generate Scouting Report", type="primary", key="structured_chat_generate_report"):
        allowed, retry_after = _allow_structured_report_submission()
        if not allowed:
            st.warning(
                f"Rate limit reached: max {STRUCTURED_REPORT_RATE_LIMIT_COUNT} reports per "
                f"{STRUCTURED_REPORT_RATE_LIMIT_WINDOW_SECONDS} seconds. Try again in ~{retry_after}s."
            )
            return

        if not selected_label:
            st.warning("No players available for the selected year, position, and search settings.")
            return

        supabase_issues = get_supabase_config_issues()
        if supabase_issues:
            msg = "Cannot run report until Supabase is configured:\n" + "\n".join([f"- {issue}" for issue in supabase_issues])
            st.error(msg)
            st.stop()

        recruit_id = str(selected_row.get("recruit_id") or "").strip()
        if not recruit_id:
            st.warning("Pick a valid player from the live search results.")
            st.stop()

        selected_player_name, selected_position_hint, selected_high_school_hint, _ = parse_selected_player_label_data(selected_label)
        milestone_slot = st.empty()
        _render_structured_milestone = _make_milestone_renderer(
            milestone_slot=milestone_slot,
            labels={
                "recruiting_scout": "Recruiting Scout",
                "team_scout": "Team Scout",
            },
            workflow_label="Web Scout Pipeline",
        )
        report_pipeline_started = time.perf_counter()
        web_scout_latency_ms = 0

        with st.spinner("Building structured scouting report..."):
            try:
                sb = get_supabase_client()
                bundle = fetch_player_bundle_data(
                    sb=sb,
                    recruit_id=str(recruit_id),
                    tables=TABLES,
                    build_player_profile_view=lambda row: build_player_profile_view_data(row, first_non_null=first_non_null),
                    clean_scouting_profile=lambda scouting: clean_scouting_profile_data(scouting, to_float_or_none=to_float_or_none),
                    merge_scouting_sources=lambda scouting_row: merge_scouting_sources_data(scouting_row=scouting_row, parse_jsonish=parse_jsonish),
                )
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
            vector_result = vector_insights_query_data(
                sb=sb,
                query_text=vector_query,
                position=position or None,
                top_k=CONFIG["VECTOR_MATCH_COUNT"],
                threshold=None,
                vector_match_threshold=CONFIG["VECTOR_MATCH_THRESHOLD"],
                vector_rpc_name=CONFIG["VECTOR_RPC_NAME"],
                get_embedding_model=get_embedding_model,
                to_float_or_none=to_float_or_none,
            )

            web_recruiting_summary = ""
            web_team_summary = ""
            web_state: dict[str, Any] = {}
            web_scout_started = time.perf_counter()
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
                web_scout_latency_ms = int((time.perf_counter() - web_scout_started) * 1000)
                milestone_slot.success("Web Scout Pipeline complete")

            web_telemetry = _normalize_telemetry_payload(
                web_state.get("telemetry") if isinstance(web_state, dict) else None,
                trace_log=list(web_state.get("trace_log") or []) if isinstance(web_state, dict) else [],
            )

            historical_comparables_md = get_historical_player_comparables_data(
                sb=sb,
                recruit_id=str(recruit_id),
                tables=TABLES,
                to_float_or_none=to_float_or_none,
                score_tier=score_tier,
            )
            score_card_html = build_score_card_html_data(
                pred_score=pred_score_row,
                pred_threshold=pred_thr_row,
                to_float_or_none=to_float_or_none,
                score_tier=score_tier,
            )
            web_summary = (
                "Recruiting Scout Summary:\n"
                f"{web_recruiting_summary or 'No recruiting summary available.'}\n\n"
                "Team Scout Summary:\n"
                f"{web_team_summary or 'No team summary available.'}"
            )

            final_prompt = build_final_prompt_data(
                year=int(selected_year),
                target_team=str(target_team),
                persona=st.session_state.get("selected_persona", "Scout"),
                player_row=player_row,
                scouting_clean=scouting_clean,
                hs_athletic_background=str(scouting_clean.get("athletic_background") or "N/A"),
                pred_score_row=pred_score_row,
                pred_thr_row=pred_thr_row,
                web_summary=web_summary,
                vector_result=vector_result,
                historical_comparables_md=historical_comparables_md,
                tier_definitions_markdown=tier_definitions_markdown,
            )
            final_synthesis_result = run_final_synthesis_with_telemetry_data(
                prompt=final_prompt,
                final_model=CONFIG["FINAL_MODEL"],
                get_llm=get_llm,
                llm_response_to_text=llm_response_to_text,
            )
            final_report = str(final_synthesis_result.get("data") or "").strip()
            synthesis_telemetry = dict(final_synthesis_result.get("telemetry") or {})
            synthesis_latency_ms = _safe_int_telemetry(synthesis_telemetry.get("latency_ms"))

            web_model_telemetry = list(web_telemetry.get("model_telemetry") or [])
            model_telemetry_rows = web_model_telemetry + ([synthesis_telemetry] if synthesis_telemetry else [])
            model_call_count = len(model_telemetry_rows)
            input_tokens = sum(_safe_int_telemetry(row.get("input_tokens")) for row in model_telemetry_rows)
            output_tokens = sum(_safe_int_telemetry(row.get("output_tokens")) for row in model_telemetry_rows)
            total_tokens = sum(_safe_int_telemetry(row.get("total_tokens")) for row in model_telemetry_rows)
            estimated_cost_usd = round(
                sum(_safe_float_telemetry(row.get("estimated_cost_usd")) for row in model_telemetry_rows),
                8,
            )
            latency_ms = _safe_int_telemetry(web_telemetry.get("model_rollup", {}).get("latency_ms")) + synthesis_latency_ms
            if total_tokens <= 0:
                total_tokens = input_tokens + output_tokens

            report_telemetry = {
                "pipeline_latency_ms": int((time.perf_counter() - report_pipeline_started) * 1000),
                "branch_latency_ms": {
                    "web_scout_pipeline": web_scout_latency_ms,
                    "final_synthesis": synthesis_latency_ms,
                },
                "model_telemetry": model_telemetry_rows,
                "model_rollup": {
                    "model_call_count": model_call_count,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                    "latency_ms": latency_ms,
                },
            }

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
            "telemetry": report_telemetry,
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
            .recruiting-dossier-card {
                --recruiting-asset-size: clamp(2.7rem, 5vw, 3.25rem);
                background: linear-gradient(
                    160deg,
                    color-mix(in srgb, var(--secondary-background-color) 92%, #0c1225 8%),
                    color-mix(in srgb, var(--background-color) 84%, #11182f 16%)
                );
                border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
                border-radius: 18px;
                padding: 0.95rem;
                margin: 0 0 1.15rem 0;
                box-shadow: 0 16px 36px color-mix(in srgb, #000 34%, transparent);
            }
            .recruiting-dossier-kpi-surface,
            .recruiting-dossier-note,
            .recruiting-dossier-recency {
                background:
                    linear-gradient(
                        165deg,
                        color-mix(in srgb, var(--secondary-background-color) 92%, #0a1224 8%),
                        color-mix(in srgb, var(--background-color) 82%, #081020 18%)
                    );
                border: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
                box-shadow:
                    0 16px 36px color-mix(in srgb, #000 42%, transparent),
                    inset 0 1px 0 color-mix(in srgb, #fff 7%, transparent);
            }
            .recruiting-dossier-header {
                display: flex;
                align-items: center;
                justify-content: flex-start;
                border-radius: 10px;
                padding: 0.74rem 0.96rem;
                margin-bottom: 0.66rem;
            }
            .recruiting-dossier-title {
                margin: 0;
                font-size: 1.32rem;
                line-height: 1.2;
                letter-spacing: 0.01em;
                font-weight: 800;
                color: var(--text-color);
            }
            .recruiting-dossier-hero {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 0.9rem;
                border-radius: 10px;
                padding: 0.9rem 0.98rem;
                margin-bottom: 0.66rem;
            }
            .recruiting-dossier-hero-left {
                display: flex;
                align-items: center;
                gap: 0.76rem;
                min-width: 0;
            }
            .recruiting-dossier-helmet {
                width: var(--recruiting-asset-size);
                height: var(--recruiting-asset-size);
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: transparent;
                border: none;
                border-radius: 0;
                overflow: visible;
                flex: 0 0 auto;
            }
            .recruiting-dossier-helmet-img {
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
            }
            .recruiting-dossier-helmet-fallback {
                font-size: 2rem;
                line-height: 1;
            }
            .recruiting-dossier-hero-copy {
                min-width: 0;
            }
            .recruiting-dossier-player {
                font-size: clamp(1.55rem, 2.05vw, 2rem);
                font-weight: 820;
                color: color-mix(in srgb, var(--text-color) 96%, transparent);
                line-height: 1.12;
                margin-bottom: 0.2rem;
                text-wrap: balance;
            }
            .recruiting-dossier-player-meta {
                font-size: 1.01rem;
                line-height: 1.32;
                color: color-mix(in srgb, var(--text-color) 84%, transparent);
                text-wrap: pretty;
            }
            .recruiting-dossier-physical {
                font-size: clamp(1.7rem, 2.2vw, 2.2rem);
                font-weight: 810;
                color: color-mix(in srgb, var(--text-color) 96%, transparent);
                white-space: nowrap;
                text-align: right;
            }
            .recruiting-dossier-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.68rem;
                align-items: start;
            }
            .recruiting-dossier-note {
                border-radius: 10px;
                padding: 0.72rem 0.8rem;
                align-self: start;
            }
            .recruiting-dossier-note-head {
                position: relative;
                margin-bottom: 0.36rem;
                padding-left: 0.46rem;
            }
            .recruiting-dossier-note-head::before {
                content: "";
                position: absolute;
                left: 0;
                top: 0.18rem;
                bottom: 0.18rem;
                width: 2px;
                border-radius: 2px;
                background: color-mix(in srgb, var(--text-color) 58%, transparent);
            }
            .recruiting-dossier-note-title {
                margin: 0;
                font-size: 0.8rem;
                font-weight: 780;
                letter-spacing: 0.085em;
                text-transform: uppercase;
                color: color-mix(in srgb, var(--text-color) 82%, transparent);
            }
            .recruiting-dossier-note-body {
                margin: 0;
                display: flex;
                align-items: center;
                gap: 0.44rem;
                font-size: 0.97rem;
                line-height: 1.42;
                color: color-mix(in srgb, var(--text-color) 94%, transparent);
            }
            .recruiting-dossier-note-icon {
                flex: 0 0 auto;
                width: var(--recruiting-asset-size);
                height: var(--recruiting-asset-size);
                line-height: 1;
                opacity: 0.9;
            }
            .recruiting-dossier-note-icon-img {
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
            }
            .recruiting-dossier-recency {
                margin-top: 0.68rem;
                border-radius: 10px;
                padding: 0.74rem 0.82rem;
            }
            .recruiting-dossier-recency-body {
                margin: 0.34rem 0 0 0;
                font-size: 0.96rem;
                line-height: 1.43;
                color: color-mix(in srgb, var(--text-color) 93%, transparent);
            }
            .final-synth-section {
                background: color-mix(in srgb, var(--secondary-background-color) 90%, var(--background-color));
                border: 1px solid color-mix(in srgb, var(--text-color) 13%, transparent);
                border-radius: 12px;
                padding: 0.86rem 0.92rem;
                margin: 0.45rem 0;
            }
            .final-synth-report {
                max-width: 1040px;
                margin: 0.2rem auto 0.95rem auto;
            }
            .final-synth-headline {
                margin: 0 0 0.68rem 0;
                padding: 0.72rem 0.85rem;
                border-radius: 12px;
                border: 1px solid color-mix(in srgb, var(--text-color) 11%, transparent);
                background: color-mix(in srgb, var(--secondary-background-color) 88%, var(--background-color));
            }
            .final-synth-headline-label {
                font-size: 0.76rem;
                letter-spacing: 0.09em;
                text-transform: uppercase;
                font-weight: 700;
                color: color-mix(in srgb, var(--text-color) 60%, transparent);
            }
            .final-synth-headline-player {
                margin-top: 0.18rem;
                font-size: 1rem;
                font-weight: 620;
                color: color-mix(in srgb, var(--text-color) 92%, transparent);
            }
            .final-synth-section h4 {
                margin: 0 0 0.55rem 0;
                font-size: 1.18rem;
                letter-spacing: 0.005em;
                color: var(--text-color);
            }
            .final-synth-kv-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.5rem 1.0rem;
            }
            .final-synth-kv-item {
                display: flex;
                flex-direction: column;
                gap: 0.16rem;
                border-left: 2px solid color-mix(in srgb, var(--text-color) 45%, transparent);
                padding-left: 0.45rem;
            }
            .final-synth-kv-label {
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: color-mix(in srgb, var(--text-color) 58%, transparent);
            }
            .final-synth-kv-value {
                font-size: 0.98rem;
                line-height: 1.4;
                color: color-mix(in srgb, var(--text-color) 95%, transparent);
            }
            .final-synth-paragraph {
                margin: 0 0 0.45rem 0;
                font-size: 1rem;
                line-height: 1.58;
                color: color-mix(in srgb, var(--text-color) 95%, transparent);
            }
            .final-synth-bullets {
                margin: 0.12rem 0 0.2rem 1rem;
                padding: 0;
            }
            .final-synth-bullets li {
                margin: 0.24rem 0;
                font-size: 0.99rem;
                line-height: 1.54;
                color: color-mix(in srgb, var(--text-color) 95%, transparent);
            }
            .final-synth-subblock {
                margin: 0.2rem 0 0.6rem 0;
                padding-left: 0.55rem;
                border-left: 2px solid color-mix(in srgb, var(--text-color) 30%, transparent);
            }
            .final-synth-subblock-title {
                display: block;
                font-size: 0.78rem;
                font-weight: 720;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: color-mix(in srgb, var(--text-color) 66%, transparent);
                margin-bottom: 0.12rem;
            }
            .final-synth-subblock-body {
                display: block;
                font-size: 0.98rem;
                line-height: 1.56;
                color: color-mix(in srgb, var(--text-color) 95%, transparent);
            }
            .final-synth-conclusion {
                background: linear-gradient(
                    165deg,
                    color-mix(in srgb, var(--secondary-background-color) 92%, #0d1326 8%),
                    color-mix(in srgb, var(--background-color) 84%, #0f1730 16%)
                );
                border-color: color-mix(in srgb, var(--text-color) 20%, transparent);
            }
            .final-synth-divider {
                border-top: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
                margin: 0.32rem 0 0.1rem 0;
            }
            @media (max-width: 1000px) {
                .structured-report-kpi-wrap {
                    width: min(720px, 100%);
                }
                .structured-report-kpi-grid {
                    grid-template-columns: repeat(2, minmax(0, 1fr));
                }
                .recruiting-dossier-grid {
                    grid-template-columns: 1fr;
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
                .recruiting-dossier-card {
                    --recruiting-asset-size: 2.9rem;
                    padding: 0.82rem;
                }
                .recruiting-dossier-header {
                    padding: 0.64rem 0.72rem;
                }
                .recruiting-dossier-title {
                    font-size: 1.12rem;
                }
                .recruiting-dossier-hero {
                    flex-direction: column;
                    align-items: flex-start;
                    padding: 0.72rem 0.74rem;
                }
                .recruiting-dossier-physical {
                    text-align: left;
                    white-space: normal;
                }
                .final-synth-kv-grid {
                    grid-template-columns: 1fr;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        selected_name_hint, selected_position_hint, selected_high_school_hint, _ = parse_selected_player_label_data(selected_label)

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

        _render_structured_report_kpi_cards(report_output)
        _render_structured_historical_comparables(report_output)

        st.markdown("### Projected Model Score")
        st.markdown(str(report_output.get("score_card_html") or ""), unsafe_allow_html=True)

        report_player_profile = dict(report_output.get("player_profile") or {})
        report_player_row = dict(report_output.get("player_row") or {})
        report_scouting_clean = dict(report_output.get("scouting_clean") or {})
        resolved_height = first_non_null(
            report_player_profile,
            ["height", "height_display", "height_ft_in", "ht", "height_inches"],
        )
        if resolved_height is None:
            resolved_height = first_non_null(
                report_player_row,
                ["height", "height_display", "height_ft_in", "ht", "height_inches"],
            )
        if resolved_height is None:
            resolved_height = first_non_null(
                report_scouting_clean,
                ["height", "height_display", "height_ft_in", "ht", "height_inches"],
            )

        resolved_weight = first_non_null(
            report_player_profile,
            ["weight", "weight_display", "weight_lbs", "wt"],
        )
        if resolved_weight is None:
            resolved_weight = first_non_null(
                report_player_row,
                ["weight", "weight_display", "weight_lbs", "wt"],
            )
        if resolved_weight is None:
            resolved_weight = first_non_null(
                report_scouting_clean,
                ["weight", "weight_display", "weight_lbs", "wt"],
            )

        recruiting_summary_context = {
            "player_name": player_name,
            "position": position,
            "high_school": high_school,
            "selected_year": report_output.get("selected_year"),
            "height": resolved_height,
            "weight": resolved_weight,
        }

        render_structured_summary_card(
            title="Recruiting Scout Summary",
            raw_text=str(report_output.get("web_recruiting_summary") or "No recruiting summary available."),
            section_key="recruiting",
            context=recruiting_summary_context,
        )

        render_structured_summary_card(
            title="Team Scout Summary",
            raw_text=str(report_output.get("web_team_summary") or "No team summary available."),
            section_key="team",
        )

        st.markdown("### Final Synthesis")
        _render_final_synthesis(report_output=report_output, player_name=player_name)

        _render_telemetry_summary(
            _normalize_telemetry_payload(
                report_output.get("telemetry"),
                trace_log=list(report_output.get("trace_log") or []),
            ),
            key_prefix="recruiting_report",
            title="Recruiting Report Telemetry",
        )

        with st.expander("Development Information (temporary)"):
            st.markdown("#### Player Profile")
            _render_json_lazy(
                report_output.get("player_profile") or report_output.get("player_row") or {},
                key="recruiting_dev_player_profile_json",
            )

            st.markdown("#### Recruiting Summary Diagnostics")
            recruiting_raw_summary = str(report_output.get("web_recruiting_summary") or "")
            recruiting_layout = _build_recruiting_layout_safe(
                recruiting_raw_summary,
                context=recruiting_summary_context,
            )
            st.write(f"Selected label: {report_output.get('selected_player_label', selected_label)}")
            st.write(f"Selected year: {report_output.get('selected_year', selected_year)}")
            st.write(f"Target team: {report_output.get('target_team', target_team)}")
            st.write(f"Recruit ID: {report_output.get('recruit_id', '')}")
            st.write(f"Raw recruiting summary chars: {len(recruiting_raw_summary)}")
            st.write(f"Parsed notes count: {len(recruiting_layout.get('notes') or [])}")
            st.write(f"Grid items count: {len(recruiting_layout.get('grid_items') or [])}")
            _render_json_lazy(
                {
                    "hero_name": recruiting_layout.get("hero_name"),
                    "hero_subtitle": recruiting_layout.get("hero_subtitle"),
                    "physical_profile": recruiting_layout.get("physical_profile"),
                    "grid_item_titles": [item.get("title") for item in list(recruiting_layout.get("grid_items") or [])],
                    "has_note_on_recency": bool(str(recruiting_layout.get("note_on_recency") or "").strip()),
                },
                key="recruiting_dev_summary_layout_json",
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
                _render_json_lazy(scouting_clean, key="recruiting_dev_scouting_profile_json")
            else:
                st.write("No scouting profile fields were available for this player.")

        st.write("---")
        st.subheader("Open Chat")
        st.caption(
            "Session-scoped memory is isolated to this combined page. "
            f"Current persona: {st.session_state.get('selected_persona', 'Scout')}"
        )

        structured_persona = str(st.session_state.get("selected_persona", "Scout"))
        synced_structured_state, active_report_context = _sync_recruiting_chat_state_with_report(
            state=st.session_state.get("structured_chat_agent_state"),
            report_output=report_output,
            persona=structured_persona,
        )
        st.session_state["active_rendered_report_context"] = dict(active_report_context)
        st.session_state["structured_chat_agent_state"] = synced_structured_state

        if "structured_chat_messages" not in st.session_state:
            st.session_state["structured_chat_messages"] = []
        if "structured_chat_agent_state" not in st.session_state:
            st.session_state["structured_chat_agent_state"] = dict(synced_structured_state)
        if "structured_chat_transcript" not in st.session_state:
            st.session_state["structured_chat_transcript"] = []

        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Clear Chat", key="structured_chat_clear"):
                st.session_state["structured_chat_messages"] = []
                st.session_state["structured_chat_agent_state"] = dict(synced_structured_state)
                st.session_state["structured_chat_transcript"] = []
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

        transcript_key = "structured_chat_transcript"
        persona_value = str(st.session_state.get("selected_persona", "Scout"))
        history_reply = _history_meta_response(
            user_prompt=user_prompt,
            transcript=list(st.session_state.get(transcript_key, [])),
        )
        if history_reply is not None:
            st.session_state["structured_chat_messages"].append({"role": "assistant", "content": history_reply})
            _append_chat_transcript_turn(
                transcript_key=transcript_key,
                user_message=user_prompt,
                assistant_response=history_reply,
                persona=persona_value,
                context_snapshot=_chat_context_snapshot(active_report_context),
            )
            with st.chat_message("assistant"):
                st.markdown(history_reply)
            return

        with st.chat_message("assistant"):
            milestone_slot = st.empty()
            _render_milestone = _make_milestone_renderer(
                milestone_slot=milestone_slot,
                labels={
                    "lead_delegator": "Delegator",
                    "cfbd_analyst": "CFBD Analyst",
                    "recruiting_scout": "Recruiting Scout",
                    "team_scout": "Team Scout",
                    "lead_synthesizer": "Lead Synthesizer",
                },
                workflow_label="Pipeline",
            )

            with st.spinner("Thinking..."):
                try:
                    graph = get_cached_agent_graph()
                    seeded_state = dict(st.session_state.get("structured_chat_agent_state", {}))
                    seeded_state["active_report_context"] = dict(active_report_context)
                    st.session_state["structured_chat_agent_state"] = seeded_state
                    current_state = compact_open_chat_state(
                        seeded_state,
                        max_turns=CHAT_STATE_MAX_TURNS,
                        max_trace=CHAT_STATE_MAX_TRACE,
                        max_errors=CHAT_STATE_MAX_ERRORS,
                        max_citations=CHAT_STATE_MAX_CITATIONS,
                        max_candidates=CHAT_STATE_MAX_CANDIDATES,
                    )
                    result_state = orchestrate_chat_turn(
                        user_prompt=user_prompt,
                        current_state=current_state,
                        graph=graph,
                        progress_callback=_render_milestone,
                    )
                    assistant_text = str(result_state.get("final_report") or "No response generated.")

                    result_state["active_report_context"] = dict(active_report_context)

                    st.session_state["structured_chat_agent_state"] = compact_open_chat_state(
                        result_state,
                        max_turns=CHAT_STATE_MAX_TURNS,
                        max_trace=CHAT_STATE_MAX_TRACE,
                        max_errors=CHAT_STATE_MAX_ERRORS,
                        max_citations=CHAT_STATE_MAX_CITATIONS,
                        max_candidates=CHAT_STATE_MAX_CANDIDATES,
                    )
                    st.session_state["structured_chat_messages"].append({"role": "assistant", "content": assistant_text})
                    _append_chat_transcript_turn(
                        transcript_key=transcript_key,
                        user_message=user_prompt,
                        assistant_response=assistant_text,
                        persona=persona_value,
                        context_snapshot=_chat_context_snapshot(active_report_context),
                    )
                    milestone_slot.success("Pipeline complete")
                    st.markdown(assistant_text)

                    chat_telemetry = _normalize_telemetry_payload(
                        result_state.get("telemetry"),
                        trace_log=list(result_state.get("trace_log") or []),
                    )
                    result_state["telemetry"] = chat_telemetry
                    _render_telemetry_summary(
                        chat_telemetry,
                        key_prefix="recruiting_chat_turn",
                        title="Chat Turn Telemetry",
                    )

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
                            _render_json_lazy(trace_log, key="recruiting_chat_execution_trace_json")

                    errors = result_state.get("errors", [])
                    if errors:
                        with st.expander("Agent notes"):
                            for err in errors[-3:]:
                                st.write(f"- {err}")
                except Exception as exc:
                    err_text = f"Open chat failed: {exc}"
                    st.session_state["structured_chat_messages"].append({"role": "assistant", "content": err_text})
                    _append_chat_transcript_turn(
                        transcript_key=transcript_key,
                        user_message=user_prompt,
                        assistant_response=err_text,
                        persona=persona_value,
                        context_snapshot=_chat_context_snapshot(active_report_context),
                    )
                    st.error(err_text)


def render_potential_transfers_with_chat_page() -> None:
    st.subheader("Transfer Portal")
    _render_transfer_portal_style_block()
    st.markdown(
        (
            "<section class='transfer-dossier-card'>"
            "<header class='transfer-dossier-header'>"
            "<h3 class='transfer-dossier-title'>Transfer Report Setup</h3>"
            "</header>"
            "<div class='transfer-dossier-hero'>"
            "<div class='transfer-dossier-hero-left'>"
            "<div class='transfer-dossier-hero-copy'>"
            "<div class='transfer-dossier-player'>Match transfer impact to team fit</div>"
            "<div class='transfer-dossier-player-meta'>Choose a position, then type at least 3 letters to live-render player names from Supabase.</div>"
            "</div>"
            "</div>"
            "<div class='transfer-dossier-physical'>Live lookup</div>"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )
    report_state_key = "transfer_chat_report_output"
    selected_position = st.selectbox(
        "Position Filter",
        [POSITION_FILTER_PLACEHOLDER] + POSITION_FILTER_OPTIONS,
        index=0,
        key="transfer_chat_position_filter",
    )
    candidate_df, selected_row = _render_transfer_candidate_live_picker(
        widget_prefix="transfer_chat",
        selected_position=selected_position,
        limit=25,
    )
    selected_label = str(selected_row.get("player_label") or "")
    target_team_option = st.selectbox("Target Team", TARGET_TEAMS, index=0, key="transfer_chat_target_team")
    target_team = _target_team_name(target_team_option)

    if st.button("Generate Transfer Impact Report", type="primary", key="transfer_chat_generate_report"):
        allowed, retry_after = _allow_structured_report_submission()
        if not allowed:
            st.warning(
                f"Rate limit reached: max {STRUCTURED_REPORT_RATE_LIMIT_COUNT} reports per "
                f"{STRUCTURED_REPORT_RATE_LIMIT_WINDOW_SECONDS} seconds. Try again in ~{retry_after}s."
            )
            return

        if not selected_label:
            st.warning("No transfer candidates available for the selected position and search text.")
            return

        selected_row = dict(selected_row or {})
        college_player_id = str(selected_row.get("college_player_id") or "").strip()
        cfbd_athlete_id = str(selected_row.get("cfbd_athlete_id") or "").strip()
        position = str(selected_row.get("position") or "").strip()

        milestone_slot = st.empty()

        def _render_transfer_milestone(event: dict[str, str]) -> None:
            node = str(event.get("node") or "transfer_pipeline")
            status = str(event.get("status") or "running")
            labels = {
                "transfer_pipeline": "Transfer Pipeline",
                "profile_lookup": "College Profile Lookup",
                "parallel_fetch": "Parallel Data Fetch",
                "cfbd_usage": "CFBD Usage Pull",
                "cfbd_stats": "CFBD Season Stats Pull",
                "player_news_search": "Player News Search",
                "team_news_search": "Team News Search",
                "summarization": "News Summarization",
                "final_synthesis": "Final Synthesis",
            }
            label = labels.get(node, node.replace("_", " ").title())
            if status == "completed":
                milestone_slot.success(f"{label}: complete")
            else:
                milestone_slot.info(f"{label}: running")

        with st.spinner("Building transfer impact report..."):
            result = orchestrate_transfer_report(
                college_player_id=college_player_id,
                cfbd_athlete_id=cfbd_athlete_id,
                target_team=str(target_team),
                position=position,
                year=2025,
                exclude_garbage_time=True,
                progress_callback=_render_transfer_milestone,
            )

        trace_rows = list(result.get("trace_log") or [])
        result_telemetry = _normalize_telemetry_payload(
            result.get("telemetry"),
            trace_log=trace_rows,
        )
        result["telemetry"] = result_telemetry

        milestone_slot.success("Transfer pipeline complete")
        st.session_state[report_state_key] = result
        _ = _get_transfer_render_artifacts(result, position_hint=position)
        st.session_state["transfer_chat_messages"] = []
        st.session_state["transfer_chat_state"] = {
            "transfer_report_context": dict(result.get("transfer_report_context") or {}),
            "conversation_history": [],
            "trace_log": list(result.get("trace_log") or []),
        }

    report_output = st.session_state.get(report_state_key)
    if not isinstance(report_output, dict):
        return

    st.markdown("### Transfer Evaluation")

    player_name = str(report_output.get("player_name") or "Unknown Player").strip()
    position = str(report_output.get("position") or "").strip()
    st.markdown(f"## Potential Transfer Evaluation - {player_name}")
    _render_transfer_report_hero(report_output)
    _render_transfer_report_kpi_cards(report_output)

    st.markdown("### Final Transfer Impact Synthesis")
    st.markdown(str(report_output.get("final_report") or "No synthesis generated."))

    pull_config = dict(report_output.get("pull_config") or {})
    artifacts = _get_transfer_render_artifacts(report_output, position_hint=position)

    st.markdown("### Charts")
    st.caption(
        "Usage values are displayed as percentages with one decimal place. "
        f"Garbage time excluded: {'Yes' if bool(pull_config.get('exclude_garbage_time', True)) else 'No'}."
    )
    _render_transfer_charts_side_by_side(section_key="transfer_portal", artifacts=artifacts)

    st.markdown("### Transfer Tables")
    _render_transfer_tables(artifacts=artifacts)

    usage_table_compact = list(report_output.get("usage_table_compact") or [])
    usage_yoy_compact = list(report_output.get("usage_yoy_compact") or [])
    season_stats_table_compact = list(report_output.get("season_stats_table_compact") or [])
    pull_diagnostics = list(report_output.get("pull_diagnostics") or [])

    with st.expander("Transfer Debug Details"):
        branch_status = dict(report_output.get("branch_status") or {})
        if branch_status:
            st.markdown("#### Pipeline Branch Health")
            _render_json_lazy(branch_status, key="transfer_portal_branch_status_json", label="Render branch health JSON")

            cfbd_branch = dict(branch_status.get("cfbd_context") or {})
            player_branch = dict(branch_status.get("player_news_search") or {})
            team_branch = dict(branch_status.get("team_news_search") or {})
            summary_branch = dict(branch_status.get("summarization") or {})

            st.write(
                f"- CFBD context: {cfbd_branch.get('status', 'unknown')} | "
                f"Usage rows: {cfbd_branch.get('usage_year_rows', 0)} | "
                f"Stats rows: {cfbd_branch.get('stats_year_rows', 0)} | "
                f"Reason: {cfbd_branch.get('reason', '')}"
            )
            st.write(
                f"- Player news search: {player_branch.get('status', 'unknown')} | "
                f"Rows: {player_branch.get('row_count', 0)} | "
                f"Reason: {player_branch.get('reason', '')}"
            )
            st.write(
                f"- Team news search: {team_branch.get('status', 'unknown')} | "
                f"Rows: {team_branch.get('row_count', 0)} | "
                f"Reason: {team_branch.get('reason', '')}"
            )
            st.write(
                f"- Summarization: player={summary_branch.get('player_status', 'unknown')} "
                f"({summary_branch.get('player_reason', '')}) | "
                f"team={summary_branch.get('team_status', 'unknown')} "
                f"({summary_branch.get('team_reason', '')})"
            )

        st.markdown("#### College Player Profile")
        _render_json_lazy(report_output.get("college_player") or {}, key="transfer_portal_college_player_json")

        st.markdown("#### 2025 Season Usage (CFBD)")
        _render_json_lazy(report_output.get("cfbd_usage_2025") or {}, key="transfer_portal_cfbd_usage_2025_json")

        st.markdown("#### 2025 Season Stats (CFBD)")
        _render_json_lazy(report_output.get("cfbd_stats_2025") or {}, key="transfer_portal_cfbd_stats_2025_json")

        st.markdown("#### Career Context")
        _render_json_lazy(report_output.get("career_context") or {}, key="transfer_portal_career_context_json")

        st.markdown("#### Career Usage By Year (CFBD)")
        _render_json_lazy(report_output.get("cfbd_usage_career") or [], key="transfer_portal_cfbd_usage_career_json")

        st.markdown("#### Career Season Stats By Year (CFBD)")
        _render_json_lazy(report_output.get("cfbd_stats_career") or [], key="transfer_portal_cfbd_stats_career_json")

        st.markdown("#### Compact Usage Table JSON")
        _render_json_lazy(usage_table_compact, key="transfer_portal_usage_table_compact_json")
        st.markdown("#### Usage YoY Delta Table JSON")
        _render_json_lazy(usage_yoy_compact, key="transfer_portal_usage_yoy_compact_json")
        st.markdown("#### Compact Season Stats Table JSON")
        _render_json_lazy(season_stats_table_compact, key="transfer_portal_season_stats_compact_json")

        st.markdown("#### Pull Config")
        _render_json_lazy(pull_config, key="transfer_portal_pull_config_json")
        st.markdown("#### Per-Year Diagnostics")
        diagnostics_df = rows_to_dynamic_table(
            pull_diagnostics,
            leading_columns=[
                "year",
                "endpoint",
                "status",
                "reason",
                "rows_pre_filter",
                "rows_post_filter",
                "queried_teams",
                "queried_team_count",
                "fallback_policy",
                "fallback_teamless_attempted",
                "params_text",
            ],
        )
        if not diagnostics_df.empty:
            st.dataframe(diagnostics_df, width="stretch")
        else:
            st.write("No diagnostics available.")

    _render_telemetry_summary(
        _normalize_telemetry_payload(
            report_output.get("telemetry"),
            trace_log=list(report_output.get("trace_log") or []),
        ),
        key_prefix="transfer_report",
        title="Transfer Report Telemetry",
    )

    st.write("---")
    st.subheader("Open Chat")
    st.caption("Follow-up chat is context-first. CFBD refresh is disabled; optional DDG recency refresh may be used.")

    if "transfer_chat_messages" not in st.session_state:
        st.session_state["transfer_chat_messages"] = []
    if "transfer_chat_state" not in st.session_state:
        st.session_state["transfer_chat_state"] = {
            "transfer_report_context": dict(report_output.get("transfer_report_context") or {}),
            "conversation_history": [],
            "trace_log": list(report_output.get("trace_log") or []),
        }

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Clear Chat", key="transfer_chat_clear"):
            st.session_state["transfer_chat_messages"] = []
            st.session_state["transfer_chat_state"] = {
                "transfer_report_context": dict(report_output.get("transfer_report_context") or {}),
                "conversation_history": [],
                "trace_log": [],
            }
            st.rerun()

    for message in st.session_state["transfer_chat_messages"]:
        with st.chat_message(message.get("role", "assistant")):
            st.markdown(str(message.get("content", "")))

    user_prompt = st.chat_input(
        "Ask follow-up questions about this transfer scenario...",
        key="transfer_chat_input",
    )
    if not user_prompt:
        return

    st.session_state["transfer_chat_messages"].append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result_state = orchestrate_transfer_chat_turn(
                user_prompt=user_prompt,
                current_state=compact_transfer_chat_state(
                    st.session_state.get("transfer_chat_state"),
                    max_turns=CHAT_STATE_MAX_TURNS,
                    max_trace=CHAT_STATE_MAX_TRACE,
                ),
                allow_web_refresh=True,
            )
            telemetry = _normalize_telemetry_payload(
                result_state.get("telemetry"),
                trace_log=list(result_state.get("trace_log") or []),
            )
            result_state["telemetry"] = telemetry
            assistant_text = str(result_state.get("final_report") or "No response generated.")
            st.session_state["transfer_chat_state"] = compact_transfer_chat_state(
                result_state,
                max_turns=CHAT_STATE_MAX_TURNS,
                max_trace=CHAT_STATE_MAX_TRACE,
            )
            st.session_state["transfer_chat_messages"].append({"role": "assistant", "content": assistant_text})
            st.markdown(assistant_text)
            _render_telemetry_summary(
                telemetry,
                key_prefix="transfer_chat_turn",
                title="Chat Turn Telemetry",
            )

            trace_log = list(result_state.get("trace_log") or [])
            if trace_log:
                with st.expander("Execution Trace"):
                    _render_json_lazy(trace_log, key="transfer_chat_execution_trace_json")


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
    if "open_chat_transcript" not in st.session_state:
        st.session_state["open_chat_transcript"] = []

    structured_report_output = st.session_state.get("structured_chat_report_output")
    active_report_context: dict[str, Any] = {}
    if isinstance(structured_report_output, dict):
        open_chat_persona = str(st.session_state.get("selected_persona", "Scout"))
        synced_state, active_report_context = _sync_recruiting_chat_state_with_report(
            state=st.session_state.get("open_chat_agent_state"),
            report_output=structured_report_output,
            persona=open_chat_persona,
        )
        st.session_state["active_rendered_report_context"] = dict(active_report_context)
        st.session_state["open_chat_agent_state"] = synced_state
    elif isinstance(st.session_state.get("active_rendered_report_context"), dict):
        active_report_context = dict(st.session_state.get("active_rendered_report_context") or {})

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Clear Chat"):
            st.session_state["open_chat_messages"] = []
            st.session_state["open_chat_transcript"] = []
            if isinstance(structured_report_output, dict):
                st.session_state["open_chat_agent_state"] = dict(synced_state)
            else:
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

    transcript_key = "open_chat_transcript"
    persona_value = str(st.session_state.get("selected_persona", "Scout"))
    history_reply = _history_meta_response(
        user_prompt=user_prompt,
        transcript=list(st.session_state.get(transcript_key, [])),
    )
    if history_reply is not None:
        st.session_state["open_chat_messages"].append({"role": "assistant", "content": history_reply})
        _append_chat_transcript_turn(
            transcript_key=transcript_key,
            user_message=user_prompt,
            assistant_response=history_reply,
            persona=persona_value,
            context_snapshot=_chat_context_snapshot(active_report_context),
        )
        with st.chat_message("assistant"):
            st.markdown(history_reply)
        return

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
                seeded_state = dict(st.session_state.get("open_chat_agent_state", {}))
                if active_report_context:
                    seeded_state["active_report_context"] = dict(active_report_context)
                st.session_state["open_chat_agent_state"] = seeded_state
                current_state = compact_open_chat_state(
                    seeded_state,
                    max_turns=CHAT_STATE_MAX_TURNS,
                    max_trace=CHAT_STATE_MAX_TRACE,
                    max_errors=CHAT_STATE_MAX_ERRORS,
                    max_citations=CHAT_STATE_MAX_CITATIONS,
                    max_candidates=CHAT_STATE_MAX_CANDIDATES,
                )
                result_state = orchestrate_chat_turn(
                    user_prompt=user_prompt,
                    current_state=current_state,
                    graph=graph,
                    progress_callback=_render_milestone,
                )
                assistant_text = str(result_state.get("final_report") or "No response generated.")
                if active_report_context:
                    result_state["active_report_context"] = dict(active_report_context)

                st.session_state["open_chat_agent_state"] = compact_open_chat_state(
                    result_state,
                    max_turns=CHAT_STATE_MAX_TURNS,
                    max_trace=CHAT_STATE_MAX_TRACE,
                    max_errors=CHAT_STATE_MAX_ERRORS,
                    max_citations=CHAT_STATE_MAX_CITATIONS,
                    max_candidates=CHAT_STATE_MAX_CANDIDATES,
                )
                st.session_state["open_chat_messages"].append({"role": "assistant", "content": assistant_text})
                _append_chat_transcript_turn(
                    transcript_key=transcript_key,
                    user_message=user_prompt,
                    assistant_response=assistant_text,
                    persona=persona_value,
                    context_snapshot=_chat_context_snapshot(active_report_context),
                )
                milestone_slot.success("Pipeline complete")
                st.markdown(assistant_text)

                chat_telemetry = _normalize_telemetry_payload(
                    result_state.get("telemetry"),
                    trace_log=list(result_state.get("trace_log") or []),
                )
                result_state["telemetry"] = chat_telemetry
                _render_telemetry_summary(
                    chat_telemetry,
                    key_prefix="open_chat_turn",
                    title="Chat Turn Telemetry",
                )

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
                        _render_json_lazy(trace_log, key="open_chat_execution_trace_json")

                errors = result_state.get("errors", [])
                if errors:
                    with st.expander("Agent notes"):
                        for err in errors[-3:]:
                            st.write(f"- {err}")
            except Exception as exc:
                err_text = f"Open chat failed: {exc}"
                st.session_state["open_chat_messages"].append({"role": "assistant", "content": err_text})
                _append_chat_transcript_turn(
                    transcript_key=transcript_key,
                    user_message=user_prompt,
                    assistant_response=err_text,
                    persona=persona_value,
                    context_snapshot=_chat_context_snapshot(active_report_context),
                )
                st.error(err_text)


def _render_transfer_portal_style_block() -> None:
    st.markdown(
        """
        <style>
        .transfer-dossier-card {
            background: linear-gradient(
                160deg,
                color-mix(in srgb, var(--secondary-background-color) 92%, #0c1225 8%),
                color-mix(in srgb, var(--background-color) 84%, #11182f 16%)
            );
            border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
            border-radius: 18px;
            padding: 0.95rem;
            margin: 0 0 1.15rem 0;
            box-shadow: 0 16px 36px color-mix(in srgb, #000 34%, transparent);
        }
        .transfer-dossier-header,
        .transfer-dossier-hero {
            border-radius: 10px;
            padding: 0.74rem 0.96rem;
            margin-bottom: 0.66rem;
        }
        .transfer-dossier-header {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            background: linear-gradient(
                165deg,
                color-mix(in srgb, var(--secondary-background-color) 90%, #0a1224 10%),
                color-mix(in srgb, var(--background-color) 80%, #081020 20%)
            );
            border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
        }
        .transfer-dossier-title {
            margin: 0;
            font-size: 1.32rem;
            line-height: 1.2;
            letter-spacing: 0.01em;
            font-weight: 800;
            color: var(--text-color);
        }
        .transfer-dossier-hero {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.9rem;
            background: linear-gradient(
                165deg,
                color-mix(in srgb, var(--secondary-background-color) 92%, #0a1224 8%),
                color-mix(in srgb, var(--background-color) 82%, #081020 18%)
            );
            border: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
            box-shadow:
                0 16px 36px color-mix(in srgb, #000 42%, transparent),
                inset 0 1px 0 color-mix(in srgb, #fff 7%, transparent);
        }
        .transfer-dossier-hero-left {
            display: flex;
            align-items: center;
            gap: 0.76rem;
            min-width: 0;
        }
        .transfer-dossier-hero-copy {
            min-width: 0;
        }
        .transfer-dossier-player {
            font-size: clamp(1.55rem, 2.05vw, 2rem);
            font-weight: 820;
            color: color-mix(in srgb, var(--text-color) 96%, transparent);
            line-height: 1.12;
            margin-bottom: 0.2rem;
            text-wrap: balance;
        }
        .transfer-dossier-player-meta {
            font-size: 1.01rem;
            line-height: 1.32;
            color: color-mix(in srgb, var(--text-color) 84%, transparent);
            text-wrap: pretty;
        }
        .transfer-dossier-physical {
            font-size: clamp(1.7rem, 2.2vw, 2.2rem);
            font-weight: 810;
            color: color-mix(in srgb, var(--text-color) 96%, transparent);
            white-space: nowrap;
            text-align: right;
        }
        .transfer-report-kpi-wrap {
            width: min(940px, 100%);
            margin: 0.1rem auto 1.2rem auto;
        }
        .transfer-report-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1.0rem;
        }
        .transfer-report-kpi-card {
            background:
                linear-gradient(
                    165deg,
                    color-mix(in srgb, var(--secondary-background-color) 92%, #0a1224 8%),
                    color-mix(in srgb, var(--background-color) 82%, #081020 18%)
                );
            border: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
            border-radius: 16px;
            padding: 0.9rem 1rem;
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
        .transfer-report-kpi-label {
            font-size: 0.72rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: color-mix(in srgb, var(--text-color) 52%, transparent);
            margin-bottom: 0.5rem;
            text-align: center;
        }
        .transfer-report-kpi-value {
            font-size: clamp(1.3rem, 1.8vw, 1.75rem);
            font-weight: 800;
            line-height: 1.12;
            color: var(--text-color);
            word-break: break-word;
            letter-spacing: 0.01em;
            text-align: center;
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
        .structured-summary-note {
            border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
            border-radius: 12px;
            background: color-mix(in srgb, var(--background-color) 88%, var(--secondary-background-color));
            padding: 0.72rem 0.82rem;
        }
        .structured-summary-note--plain {
            border-left: 2px solid color-mix(in srgb, #3b82f6 35%, transparent);
            padding-left: 0.74rem;
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
        @media (max-width: 1000px) {
            .transfer-report-kpi-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 640px) {
            .transfer-dossier-card {
                padding: 0.82rem;
            }
            .transfer-dossier-header,
            .transfer-dossier-hero {
                padding: 0.64rem 0.72rem;
            }
            .transfer-dossier-title {
                font-size: 1.12rem;
            }
            .transfer-dossier-hero {
                flex-direction: column;
                align-items: flex-start;
                padding: 0.72rem 0.74rem;
            }
            .transfer-dossier-physical {
                text-align: left;
                white-space: normal;
            }
            .transfer-report-kpi-wrap {
                width: 100%;
            }
            .transfer-report-kpi-grid {
                grid-template-columns: 1fr;
                gap: 0.9rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_transfer_summary_card(title: str, raw_text: str | None) -> None:
    notes = parse_summary_notes_data(raw_text)
    if not notes:
        notes = [{"label": "", "body": str(raw_text or "No summary available.").strip() or "No summary available."}]

    notes_html = "".join(
        [
            (
                f"<div class='structured-summary-note'>"
                f"<div class='structured-summary-note-label'>{html.escape(note.get('label') or 'Note')}</div>"
                f"<div class='structured-summary-note-body'>{html.escape(note.get('body') or '')}</div>"
                "</div>"
                if str(note.get("label") or "").strip()
                else (
                    f"<div class='structured-summary-note structured-summary-note--plain'>"
                    f"<div class='structured-summary-note-body'>{html.escape(note.get('body') or '')}</div>"
                    "</div>"
                )
            )
            for note in notes
        ]
    )

    st.markdown(
        (
            f"<section class='structured-summary-card'>"
            f"<h3 class='structured-summary-title'>{html.escape(title)}</h3>"
            f"<div class='structured-summary-list'>{notes_html}</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def _render_transfer_report_kpi_cards(report_output: dict[str, Any]) -> None:
    kpi_cards = [
        ("Player", str(report_output.get("player_name") or "N/A")),
        ("Position", str(report_output.get("position") or "N/A")),
        ("Target Team", str(report_output.get("target_team") or "N/A")),
        ("CFBD Athlete ID", str(report_output.get("cfbd_athlete_id") or "N/A")),
    ]
    kpi_cards_html = "".join(
        [
            (
                "<div class='transfer-report-kpi-card'>"
                f"<div class='transfer-report-kpi-label'>{html.escape(label)}</div>"
                f"<div class='transfer-report-kpi-value'>{html.escape(value)}</div>"
                "</div>"
            )
            for label, value in kpi_cards
        ]
    )
    st.markdown(
        (
            "<div class='transfer-report-kpi-wrap'>"
            f"<div class='transfer-report-kpi-grid'>{kpi_cards_html}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _build_transfer_physical_profile(player_row: dict[str, Any]) -> str:
    height_inches = to_float_or_none(first_non_null(player_row, ["height_inches", "height", "height_in"]))
    weight_lbs = to_float_or_none(first_non_null(player_row, ["weight_lbs", "weight"]))
    class_year = str(first_non_null(player_row, ["class", "class_year", "year"]) or "").strip()
    season_span_start = first_non_null(player_row, ["first_season"])
    season_span_end = first_non_null(player_row, ["last_season"])

    parts: list[str] = []
    if height_inches is not None and height_inches > 0:
        feet = int(height_inches) // 12
        inches = int(round(height_inches - feet * 12))
        parts.append(f"{feet}-{inches} H")
    if weight_lbs is not None and weight_lbs > 0:
        parts.append(f"{int(round(weight_lbs))} lbs")
    if class_year:
        parts.append(f"Class {class_year}")
    if season_span_start or season_span_end:
        parts.append(f"Span {season_span_start or '?'}-{season_span_end or '?'}")
    return " | ".join(parts) if parts else "Physical profile unavailable"


def _render_transfer_report_hero(report_output: dict[str, Any]) -> None:
    player_row = dict(report_output.get("college_player") or {})
    player_name = str(report_output.get("player_name") or player_row.get("player_name") or "Unknown Player").strip()
    position = str(report_output.get("position") or player_row.get("position") or "N/A").strip() or "N/A"
    target_team = str(report_output.get("target_team") or "N/A").strip() or "N/A"
    athlete_id = str(report_output.get("cfbd_athlete_id") or "N/A").strip() or "N/A"
    subtitle = f"{position} | Target: {target_team} | CFBD ID: {athlete_id}"
    physical_profile = _build_transfer_physical_profile(player_row)

    st.markdown(
        (
            "<section class='transfer-dossier-card'>"
            "<header class='transfer-dossier-header'>"
            "<h3 class='transfer-dossier-title'>Transfer Report</h3>"
            "</header>"
            "<div class='transfer-dossier-hero'>"
            "<div class='transfer-dossier-hero-left'>"
            "<div class='transfer-dossier-hero-copy'>"
            f"<div class='transfer-dossier-player'>{html.escape(player_name)}</div>"
            f"<div class='transfer-dossier-player-meta'>{html.escape(subtitle)}</div>"
            "</div>"
            "</div>"
            f"<div class='transfer-dossier-physical'>{html.escape(physical_profile)}</div>"
            "</div>"
            "</section>"
        ),
        unsafe_allow_html=True,
    )


if app_page == "Landing Page":
    render_landing_page()
elif app_page == "Recruiting Portal":
    render_structured_report_with_chat_page()
elif app_page == "Transfer Portal":
    render_potential_transfers_with_chat_page()
elif app_page == "Open Chat":
    render_open_chat_page()
else:
    render_local_cfbd_debugger_page()

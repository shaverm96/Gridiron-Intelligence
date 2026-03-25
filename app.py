from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from engine import get_scout_graph, orchestrate_chat_turn, orchestrate_structured_report
from engine.state import initial_chat_state

from engine.comparables_service import (
    get_historical_player_comparables_data,
)
from engine.data_access import (
    fetch_player_bundle_data,
    load_player_index_data,
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
    try:
        if key in st.secrets:
            value = st.secrets.get(key, default)
            return (str(value).strip() if value is not None else "", "streamlit_secrets")
    except Exception:
        pass

    env_value = os.getenv(key)
    if env_value is not None:
        return env_value.strip(), "environment"
    return default, "default"


def _cfg(key: str, default: str = "") -> str:
    value, _ = _cfg_with_source(key, default)
    return value


def resolve_project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for candidate in candidates:
        if (candidate / "data" / "modeling_datasets").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = resolve_project_root()
RECRUITS_PATH = PROJECT_ROOT / "data" / "modeling_datasets" / "recruits" / "master_recruits_2015_2028.csv"
MODEL_TIERS_PATH = PROJECT_ROOT / "data" / "modeling_datasets" / "final" / "models" / "input_data" / "Model_Tiers.csv"

if load_dotenv is not None:
    for env_name in ("SECRETS.env", "SUPABASE.env", "GEMINI_API_KEY.env"):
        env_file = PROJECT_ROOT / env_name
        if env_file.exists():
            load_dotenv(env_file, override=False)

SUPABASE_URL, SUPABASE_URL_SOURCE = _cfg_with_source("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY, SUPABASE_SERVICE_ROLE_KEY_SOURCE = _cfg_with_source("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY, GEMINI_API_KEY_SOURCE = _cfg_with_source("GEMINI_API_KEY")
CFBD_API_KEY, CFBD_API_KEY_SOURCE = _cfg_with_source("CFBD_API_KEY")

CONFIG = {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "CFBD_API_KEY": CFBD_API_KEY,
    "CFBD_BASE_URL": _cfg("CFBD_BASE_URL", "https://api.collegefootballdata.com"),
    "YEARS": [2026, 2027, 2028],
    "FINAL_MODEL": "gemini-3-flash-preview",
    "SUMMARY_MODEL": "gemini-2.5-flash-lite",
    "VECTOR_MATCH_COUNT": 6,
    "VECTOR_MATCH_THRESHOLD": 0.15,
    "VECTOR_RPC_NAME": "match_gi_factoids",
}

CONFIG_SOURCES = {
    "SUPABASE_URL": SUPABASE_URL_SOURCE,
    "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_SERVICE_ROLE_KEY_SOURCE,
    "GEMINI_API_KEY": GEMINI_API_KEY_SOURCE,
    "CFBD_API_KEY": CFBD_API_KEY_SOURCE,
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
                response = llm.invoke("Reply with exactly: OK")
                text = llm_response_to_text(response).strip()
                add_check("Gemini connectivity", "pass", f"Model responded: {text[:80] if text else 'empty response'}")
        except Exception as exc:
            add_check("Gemini connectivity", "fail", f"Invocation test failed: {exc}")

    overall = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {"overall": overall, "checks": checks}


def get_llm(model_name: str, temperature: float = 0.2, max_output_tokens: int = 1800):
    if ChatGoogleGenerativeAI is None or not CONFIG["GEMINI_API_KEY"]:
        return None
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=CONFIG["GEMINI_API_KEY"], temperature=temperature, max_output_tokens=max_output_tokens)


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


@st.cache_data
def load_model_tiers() -> pd.DataFrame:
    if not MODEL_TIERS_PATH.exists():
        return pd.DataFrame(columns=["Score Range", "Career Designation", "College Outlook", "Professional Outlook", "low", "high"])
    df = pd.read_csv(MODEL_TIERS_PATH)
    for c in ["Score Range", "Career Designation", "College Outlook", "Professional Outlook"]:
        if c not in df.columns:
            df[c] = ""
    bounds = df["Score Range"].astype(str).str.extract(r"(?P<low>\d+(?:\.\d+)?).*?(?P<high>\d+(?:\.\d+)?)")
    df["low"] = pd.to_numeric(bounds["low"], errors="coerce")
    df["high"] = pd.to_numeric(bounds["high"], errors="coerce")
    return df.sort_values(["low", "high"]).reset_index(drop=True)


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
    return "\n".join([f"- **{r.get('Career Designation', '')}** ({r.get('Score Range', '')}): College Outlook: {r.get('College Outlook', '')}; Professional Outlook: {r.get('Professional Outlook', '')}" for _, r in tiers.iterrows()])


def merge_scouting_sources(scouting_row: dict) -> dict:
    return merge_scouting_sources_data(scouting_row=scouting_row, parse_jsonish=parse_jsonish)


def clean_scouting_profile(scouting_json: dict) -> dict:
    return clean_scouting_profile_data(scouting_json=scouting_json, to_float_or_none=to_float_or_none)


def build_player_profile_view(player_row: dict) -> dict:
    return build_player_profile_view_data(player_row=player_row, first_non_null=first_non_null)


@st.cache_data
def load_player_index() -> pd.DataFrame:
    return load_player_index_data(recruits_path=RECRUITS_PATH, years=CONFIG["YEARS"])


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


st.set_page_config(page_title="Gridiron Intelligence - Scouting Workbench", page_icon="🏈", layout="wide")
st.markdown("<h1 class='football-title'>Gridiron Intelligence 🏈</h1>", unsafe_allow_html=True)
st.markdown("<p class='football-subtitle'>Interactive Scouting Workbench (Streamlit)</p>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://raw.githubusercontent.com/shaverm96/Gridiron-Intelligence/main/Logos/Main.svg", width=150)
    st.title("Gridiron Intelligence")
    app_page = st.radio("Workspace", ["Structured Report", "Open Chat"], index=0)
    selected_persona = st.selectbox("Persona", ["Scout", "Fan"], index=0, key="selected_persona")
    st.write("---")
    st.caption(f"Gemini configured: {'Yes' if bool(CONFIG['GEMINI_API_KEY']) else 'No'}")
    st.caption(f"Supabase configured: {'Yes' if bool(CONFIG['SUPABASE_URL'] and CONFIG['SUPABASE_SERVICE_ROLE_KEY']) else 'No'}")
    with st.expander("Configuration diagnostics"):
        st.write(f"SUPABASE_URL source: {CONFIG_SOURCES['SUPABASE_URL']}")
        st.write(f"SUPABASE_SERVICE_ROLE_KEY source: {CONFIG_SOURCES['SUPABASE_SERVICE_ROLE_KEY']}")
        st.write(f"GEMINI_API_KEY source: {CONFIG_SOURCES['GEMINI_API_KEY']}")
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

    supabase_issues = get_supabase_config_issues()
    if supabase_issues:
        st.warning("Supabase preflight issues detected. Open diagnostics for details.")


@st.cache_resource
def get_cached_agent_graph():
    return get_scout_graph()


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

        selected_player_name = selected_label.split("|")[0].strip()
        graph = get_cached_agent_graph()

        with st.spinner("Running multi-agent pipeline... delegating CFBD and web workers in parallel."):
            try:
                result_state = orchestrate_structured_report(
                    player_name=selected_player_name,
                    recruit_id=str(recruit_id),
                    target_team=str(target_team),
                    year=int(selected_year),
                    graph=graph,
                )
            except Exception as exc:
                st.error(f"Pipeline failed: {exc}")
                st.stop()

        bundle = dict(result_state.get("sql_data_context") or {})
        player_profile = dict(bundle.get("player") or {})
        player_name = (
            result_state.get("target_player_name")
            or result_state.get("player_name")
            or player_profile.get("player_name")
            or selected_player_name
        )

        st.markdown(f"## Scouting Workbench Output - {player_name}")
        st.markdown(
            f"- Recruit ID: `{recruit_id}`  \\\n+- Year: `{selected_year}`  \\\n+- Target Team: `{target_team}`  \\\n+- Persona: `{st.session_state.get('selected_persona', 'Scout')}`"
        )
        st.markdown("### Player Profile")
        st.code(json.dumps(player_profile, indent=2, default=str), language="json")

        st.markdown("### CFBD Analyst Summary")
        st.markdown(result_state.get("cfbd_data_summary") or "No CFBD summary available.")

        st.markdown("### Recruiting Scout Summary")
        st.markdown(result_state.get("web_recruiting_summary") or "No recruiting web summary available.")

        st.markdown("### Team Scout Summary")
        st.markdown(result_state.get("web_team_summary") or "No team context summary available.")

        st.markdown("### Final Synthesis")
        st.markdown(result_state.get("final_report") or "No final synthesis generated.")

        trace_log = list(result_state.get("trace_log") or [])
        if trace_log:
            with st.expander("Execution Trace"):
                st.code(json.dumps(trace_log, indent=2, default=str), language="json")

        errors = list(result_state.get("errors") or [])
        if errors:
            with st.expander("Agent Notes"):
                for err in errors[-5:]:
                    st.write(f"- {err}")


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
        with st.spinner("Thinking..."):
            try:
                graph = get_cached_agent_graph()
                current_state = dict(st.session_state.get("open_chat_agent_state", {}))
                result_state = orchestrate_chat_turn(
                    user_prompt=user_prompt,
                    current_state=current_state,
                    graph=graph,
                )
                assistant_text = str(result_state.get("final_report") or "No response generated.")

                st.session_state["open_chat_agent_state"] = result_state
                st.session_state["open_chat_messages"].append({"role": "assistant", "content": assistant_text})
                st.markdown(assistant_text)

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
else:
    render_open_chat_page()

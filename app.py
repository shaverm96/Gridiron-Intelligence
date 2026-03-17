from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

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


def _cfg(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


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
    for env_name in ("SUPABASE.env", "GEMINI_API_KEY.env"):
        env_file = PROJECT_ROOT / env_name
        if env_file.exists():
            load_dotenv(env_file, override=False)

CONFIG = {
    "SUPABASE_URL": _cfg("SUPABASE_URL"),
    "SUPABASE_SERVICE_ROLE_KEY": _cfg("SUPABASE_SERVICE_ROLE_KEY"),
    "GEMINI_API_KEY": _cfg("GEMINI_API_KEY"),
    "YEARS": [2026, 2027, 2028],
    "FINAL_MODEL": "gemini-3-flash-preview",
    "SUMMARY_MODEL": "gemini-2.5-flash-lite",
    "VECTOR_MATCH_COUNT": 6,
    "VECTOR_MATCH_THRESHOLD": 0.15,
    "VECTOR_RPC_NAME": "match_gi_factoids",
}

TABLES = {
    "player_master": "gi_player_master",
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
    merged = parse_jsonish(scouting_row.get("scouting_json"))
    for key, value in scouting_row.items():
        if (str(key).startswith("skill_") or str(key).startswith("flag_")) and (key not in merged or merged.get(key) in (None, "", "none", "nan")):
            merged[key] = value
    return merged


def clean_scouting_profile(scouting_json: dict) -> dict:
    cleaned = {}
    for key, value in scouting_json.items():
        if key.startswith("skill_"):
            if value is None or str(value).strip().lower() in {"", "none", "nan"}:
                continue
            cleaned[key] = value
        elif key.startswith("flag_"):
            num = to_float_or_none(value)
            if num is not None and num != 0:
                cleaned[key] = int(num) if float(num).is_integer() else num
    return cleaned


def build_player_profile_view(player_row: dict) -> dict:
    return {"recruit_id": first_non_null(player_row, ["recruit_id", "player_id"]), "player_name": first_non_null(player_row, ["player_name", "name"]), "position": first_non_null(player_row, ["position"]), "rating": first_non_null(player_row, ["rating"]), "height_raw": first_non_null(player_row, ["height_raw", "height"]), "weight_raw": first_non_null(player_row, ["weight_raw", "weight"]), "height_inches": first_non_null(player_row, ["height_inches"]), "weight_lbs": first_non_null(player_row, ["weight_lbs"]), "high_school": first_non_null(player_row, ["high_school"]), "city": first_non_null(player_row, ["city"]), "state": first_non_null(player_row, ["state"]), "committed_to": first_non_null(player_row, ["committed_to"]), "year": first_non_null(player_row, ["year"])}


@st.cache_data
def load_player_index() -> pd.DataFrame:
    df = pd.read_csv(RECRUITS_PATH)
    if "recruit_id" not in df.columns and "player_id" in df.columns:
        df["recruit_id"] = df["player_id"]
    required = ["recruit_id", "name", "position", "high_school", "year"]
    df = df[required + [c for c in ["committed_to"] if c in df.columns]].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df[df["year"].isin(CONFIG["YEARS"])].copy()
    for col in ["recruit_id", "name", "position", "high_school"]:
        df[col] = df[col].astype(str).str.strip()
    df = df[df["recruit_id"] != ""].drop_duplicates(subset=["recruit_id"])
    df["player_label"] = df.apply(lambda r: f"{r['name']} | {r['position']} | {r['high_school']}", axis=1)
    return df.sort_values(["year", "name"]).reset_index(drop=True)


def fetch_player_bundle(sb, recruit_id: str) -> dict:
    if sb is None:
        raise RuntimeError("Supabase client is not configured.")
    recruit_id = str(recruit_id).strip()
    player_master = sb.table(TABLES["player_master"]).select("*").eq("recruit_id", recruit_id).limit(1).execute().data
    scouting = sb.table(TABLES["scouting_features"]).select("*").eq("recruit_id", recruit_id).limit(1).execute().data
    pred_score = sb.table(TABLES["pred_score"]).select("*").eq("recruit_id", recruit_id).order("as_of_date", desc=True).order("updated_at", desc=True).limit(1).execute().data
    pred_threshold = sb.table(TABLES["pred_threshold"]).select("*").eq("recruit_id", recruit_id).order("as_of_date", desc=True).order("updated_at", desc=True).limit(1).execute().data
    player_row = player_master[0] if player_master else {}
    scouting_row = scouting[0] if scouting else {}
    return {"recruit_id": recruit_id, "player": player_row, "player_profile": build_player_profile_view(player_row), "scouting_raw": scouting_row, "scouting_clean": clean_scouting_profile(merge_scouting_sources(scouting_row)), "pred_score": pred_score[0] if pred_score else {}, "pred_threshold": pred_threshold[0] if pred_threshold else {}}


def duckduckgo_search(player_name: str, position: str, high_school: str, year: int, max_results: int = 12) -> list[dict]:
    if DDGS is None:
        return []
    site_filter = " OR ".join([f"site:{site}" for site in TARGET_SEARCH_SITES])
    query = f"{player_name} {position} {high_school} {year} football recruiting ({site_filter})"
    rows = []
    with DDGS() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            url = result.get("href", "") or ""
            if any(site in url for site in TARGET_SEARCH_SITES):
                rows.append({"title": result.get("title", ""), "url": url, "snippet": result.get("body", "")})
    return rows


def summarize_web_with_flash_lite(player_name: str, position: str, search_rows: list[dict]) -> str:
    if not search_rows:
        return "No relevant web articles were found from target recruiting sites."
    llm = get_llm(CONFIG["SUMMARY_MODEL"], temperature=0.0, max_output_tokens=1200)
    if llm is None:
        return "Gemini summary skipped: API key/model client not configured."
    context = "\n".join([f"[{i}] Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet']}" for i, r in enumerate(search_rows[:10], start=1)])
    prompt = f"You are a recruiting research assistant. Summarize recent web intelligence for {player_name} ({position}). Only use provided sources.\nSources:\n{context}"
    return llm_response_to_text(llm.invoke(prompt))


def vector_insights_query(sb, query_text: str, position: str | None = None, top_k: int = 6, threshold: float | None = None) -> dict:
    if sb is None:
        return {"status": "skipped", "reason": "Supabase client unavailable", "insights": []}
    pos = (position or "").strip().upper()
    if not pos:
        return {"status": "skipped", "reason": "Missing player position for exact positional filter", "insights": []}
    try:
        model = get_embedding_model()
        embedding = model.encode([query_text], normalize_embeddings=True)[0].tolist()
        use_threshold = float(threshold if threshold is not None else CONFIG["VECTOR_MATCH_THRESHOLD"])
        payload = {"query_embedding": embedding, "match_threshold": use_threshold, "match_count": int(top_k), "filter_position": pos}
        rows = sb.rpc(CONFIG["VECTOR_RPC_NAME"], payload).execute().data or []
        insights = []
        for row in rows:
            text = row.get("factoid_text") or ""
            if not text:
                continue
            sim = to_float_or_none(row.get("similarity"))
            ftype = row.get("factoid_type") or "insight"
            insights.append(f"[{ftype} | sim={sim:.3f}] {text}" if sim is not None else f"[{ftype}] {text}")
        return {"status": "ok", "reason": f"exact position={pos}, threshold={use_threshold:.3f}", "insights": insights[:top_k]}
    except Exception as exc:
        return {"status": "skipped", "reason": f"Vector RPC unavailable: {exc}", "insights": []}


def get_historical_player_comparables(recruit_id: str) -> str:
    sb = get_supabase_client()
    if sb is None:
        return "Historical comparables unavailable: Supabase client is not configured."
    try:
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        return "Historical comparables unavailable: sklearn.preprocessing.MinMaxScaler is not installed in this environment."

    target_rows = sb.table(TABLES["player_master"]).select("recruit_id, player_name, year, position, rating, height_inches, weight_lbs, state").eq("recruit_id", str(recruit_id).strip()).limit(1).execute().data or []
    if not target_rows:
        return f"Historical comparables unavailable: recruit_id {recruit_id} was not found in gi_player_master."
    target_row = target_rows[0]
    target_pos = str(target_row.get("position") or "").strip()
    if not target_pos:
        return "Historical comparables unavailable: player position is missing."

    pool = sb.table(TABLES["player_master"]).select("recruit_id, player_name, year, position, rating, height_inches, weight_lbs, state").eq("position", target_pos).lte("year", 2022).execute().data or []
    df_pool = pd.DataFrame(pool)
    if df_pool.empty:
        return f"No historical comparables found for position {target_pos} with class year <= 2022."
    df_pool = df_pool[df_pool["recruit_id"].astype(str) != str(recruit_id)].copy()
    if df_pool.empty:
        return f"No historical comparables found for position {target_pos} with class year <= 2022."

    df = pd.concat([pd.DataFrame([target_row]), df_pool], ignore_index=True)

    def calc_sim(feature_df: pd.DataFrame) -> np.ndarray:
        numeric_df = feature_df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        if numeric_df.empty:
            return np.ones(len(df), dtype=float)
        filled = numeric_df.fillna(numeric_df.mean()).fillna(0.0)
        scaled = MinMaxScaler().fit_transform(filled)
        dists = np.linalg.norm(scaled - scaled[0], axis=1)
        max_dist = np.sqrt(float(numeric_df.shape[1]))
        sim = np.ones(len(df), dtype=float) if max_dist == 0 else 1.0 - (dists / max_dist)
        return np.clip(np.nan_to_num(sim, nan=0.0), 0.0, 1.0)

    target_state = str(target_row.get("state") or "").strip().upper()
    state_sim = (df["state"].astype(str).str.strip().str.upper() == target_state).astype(float).to_numpy() if target_state else np.zeros(len(df), dtype=float)
    final_sim = (calc_sim(df[["rating"]]) * 3.0 + calc_sim(df[["height_inches", "weight_lbs"]]) * 2.5 + state_sim * 0.5) / 6.0
    df["similarity_score"] = np.round(final_sim * 100.0, 2)
    comps_top10 = df.drop(index=0).sort_values("similarity_score", ascending=False).head(10).copy()
    if comps_top10.empty:
        return "No comparables found after similarity scoring."

    comp_ids = comps_top10["recruit_id"].astype(str).tolist()
    score_rows = sb.table(TABLES["pred_score"]).select("recruit_id, predictive_score_0_100, as_of_date, updated_at").in_("recruit_id", comp_ids).execute().data or []
    latest_by_id: dict[str, dict] = {}
    for row in score_rows:
        rid = str(row.get("recruit_id") or "").strip()
        if not rid:
            continue
        previous = latest_by_id.get(rid)
        if previous is None or (str(row.get("as_of_date") or ""), str(row.get("updated_at") or "")) > (str(previous.get("as_of_date") or ""), str(previous.get("updated_at") or "")):
            latest_by_id[rid] = row

    score_by_id = {rid: to_float_or_none(rec.get("predictive_score_0_100")) for rid, rec in latest_by_id.items()}
    comps = comps_top10[comps_top10["recruit_id"].astype(str).map(lambda rid: score_by_id.get(rid) is not None)].head(5).copy()
    if comps.empty:
        return "No historical comparables with recorded college outcomes were found among the 10 closest matches."

    lines = [f"### Historical Comparables for {target_row.get('player_name', recruit_id)}", "---"]
    for _, row in comps.iterrows():
        rid = str(row.get("recruit_id"))
        hist_score = score_by_id.get(rid)
        lines.append(
            f"- **{row.get('player_name', 'Unknown')}** (Class of {row.get('year', 'N/A')}, {row.get('state', 'N/A')})  \n"
            f"  *Match:* {row.get('similarity_score', 'N/A')}% Sim | Rating: {row.get('rating', 'N/A')} | Size: {row.get('height_inches', 'N/A')} in, {row.get('weight_lbs', 'N/A')} lbs  \n"
            f"  *Actual College Outcome:* **{score_tier(hist_score)}** (Score: {'N/A' if hist_score is None else f'{hist_score:.1f}/100'})"
        )
    return "\n".join(lines)


def build_score_card_html(pred_score: dict, pred_threshold: dict) -> str:
    score = to_float_or_none(pred_score.get("predictive_score_0_100"))
    p20 = to_float_or_none(pred_threshold.get("prob_ge20"))
    p50 = to_float_or_none(pred_threshold.get("prob_ge50"))
    p80 = to_float_or_none(pred_threshold.get("prob_ge80"))
    format_percentage = lambda x: "N/A" if x is None else f"{x*100:.1f}%"
    create_progress_bar = lambda x: f"<div style='background:#e9ecef;border-radius:8px;height:14px;'><div style='width:{0 if x is None else max(0,min(100,x*100)):.1f}%;background:#0d6efd;height:14px;border-radius:8px;'></div></div>"
    score_width = 0 if score is None else max(0, min(100, score))
    return f"<div style='border:1px solid #d9d9d9;border-radius:12px;padding:14px 16px;margin:8px 0 16px 0;background:#ffffff;'><h3 style='margin:0 0 8px 0;'>Model Output</h3><div style='font-size:28px;font-weight:700;'>{'N/A' if score is None else f'{score:.1f}'}/100</div><div>Career Designation: <b>{score_tier(score)}</b></div><div style='background:#e9ecef;border-radius:8px;height:16px;'><div style='width:{score_width:.1f}%;background:#198754;height:16px;border-radius:8px;'></div></div><div><b>Contributor (&gt;20):</b> {format_percentage(p20)} {create_progress_bar(p20)}</div><div><b>Multi-Year Starter (&gt;50):</b> {format_percentage(p50)} {create_progress_bar(p50)}</div><div><b>Elite (&gt;80):</b> {format_percentage(p80)} {create_progress_bar(p80)}</div></div>"


def build_final_prompt(year: int, target_team: str, player_row: dict, scouting_clean: dict, hs_athletic_background: str, pred_score_row: dict, pred_thr_row: dict, web_summary: str, vector_result: dict, historical_comparables_md: str) -> str:
    vector_text = "\n".join([f"- {x}" for x in vector_result.get("insights", [])]) or f"No vector insights available ({vector_result.get('reason', 'not returned')})."
    return f"""You are a senior college football recruiting scout.

## Request Context
- Recruiting Year: {year}
- Target Team: {target_team}

## Player Profile
{json.dumps(player_row, indent=2, default=str)}

## High School Athletic Background
{hs_athletic_background}

## Filtered Scouting Attributes
{json.dumps(scouting_clean, indent=2, default=str)}

## Model Outputs
{json.dumps(pred_score_row, indent=2, default=str)}
{json.dumps(pred_thr_row, indent=2, default=str)}

## Official Tier Definitions
{tier_definitions_markdown()}

## Web Intelligence Summary
{web_summary}

## Historical/Vector Insights
{vector_text}

## Historical Player Comparables
{historical_comparables_md}
"""


def run_final_synthesis(prompt: str) -> str:
    llm = get_llm(CONFIG["FINAL_MODEL"], temperature=0.25, max_output_tokens=2200)
    if llm is None:
        return "Final synthesis skipped: Gemini client not configured."
    return llm_response_to_text(llm.invoke(prompt))


st.set_page_config(page_title="Gridiron Intelligence - Scouting Workbench", page_icon="🏈", layout="wide")
st.markdown("<h1 class='football-title'>Gridiron Intelligence 🏈</h1>", unsafe_allow_html=True)
st.markdown("<p class='football-subtitle'>Interactive Scouting Workbench (Streamlit)</p>", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://raw.githubusercontent.com/shaverm96/Gridiron-Intelligence/main/Logos/Main.svg", width=150)
    st.title("Gridiron Intelligence")
    st.write("---")
    st.caption(f"Gemini configured: {'Yes' if bool(CONFIG['GEMINI_API_KEY']) else 'No'}")
    st.caption(f"Supabase configured: {'Yes' if bool(CONFIG['SUPABASE_URL'] and CONFIG['SUPABASE_SERVICE_ROLE_KEY']) else 'No'}")

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
    else:
        lookup = dict(zip(player_index["player_label"], player_index["recruit_id"]))
        recruit_id = lookup.get(selected_label)
        if not recruit_id:
            st.warning("Pick a valid player from the dropdown list.")
            st.stop()

        with st.spinner("Running pipeline... retrieving SQL profile, web summary, vector insights, and final synthesis."):
            try:
                sb = get_supabase_client()
                bundle = fetch_player_bundle(sb, recruit_id)
            except Exception as exc:
                st.error(f"Data retrieval failed: {exc}")
                st.stop()

            player_profile = bundle.get("player_profile", {})
            scouting_raw = bundle.get("scouting_raw", {})
            scouting_clean = bundle.get("scouting_clean", {})
            pred_score_row = bundle.get("pred_score", {})
            pred_thr_row = bundle.get("pred_threshold", {})

            player_name = player_profile.get("player_name") or selected_label.split("|")[0].strip()
            position = player_profile.get("position") or ""
            vector_position = normalize_position_group(position)
            player_state = str(player_profile.get("state") or "").strip().upper()
            high_school = player_profile.get("high_school") or ""
            hs_athletic_background = scouting_raw.get("hs_athletic_background") or ""

            search_rows = duckduckgo_search(player_name, position, high_school, int(selected_year), max_results=12)
            web_summary = summarize_web_with_flash_lite(player_name, position, search_rows)

            vector_query_text = (
                f"Player: {player_name}\nPosition: {position}\nPosition Group: {vector_position}\nState: {player_state}\n"
                f"High School: {high_school}\nHS Athletic Background:\n{hs_athletic_background}\n\n"
                f"Filtered scouting report:\n{json.dumps(scouting_clean, default=str)}\n\nWeb Intelligence Summary:\n{web_summary}"
            )
            vector_result = vector_insights_query(sb, query_text=vector_query_text, position=vector_position, top_k=CONFIG["VECTOR_MATCH_COUNT"], threshold=CONFIG["VECTOR_MATCH_THRESHOLD"])
            historical_comparables_md = get_historical_player_comparables(str(recruit_id))
            final_prompt = build_final_prompt(
                year=int(selected_year),
                target_team=target_team,
                player_row=player_profile if player_profile else bundle.get("player", {}),
                scouting_clean=scouting_clean,
                hs_athletic_background=hs_athletic_background,
                pred_score_row=pred_score_row,
                pred_thr_row=pred_thr_row,
                web_summary=web_summary,
                vector_result=vector_result,
                historical_comparables_md=historical_comparables_md,
            )
            final_report = run_final_synthesis(final_prompt)

        st.markdown(f"## Scouting Workbench Output - {player_name}")
        st.markdown(f"- Recruit ID: `{recruit_id}`  \\\n- Year: `{selected_year}`  \\\n- Target Team: `{target_team}`")
        st.markdown("### Player Profile (from gi_player_master)")
        st.code(json.dumps(player_profile, indent=2, default=str), language="json")
        st.markdown(build_score_card_html(pred_score_row, pred_thr_row), unsafe_allow_html=True)
        st.markdown("### Historical Comparables")
        st.markdown(historical_comparables_md)
        st.markdown("### Filtered Scouting Profile")
        st.code(json.dumps(scouting_clean, indent=2, default=str), language="json")
        st.markdown("### Web Intelligence Summary")
        st.markdown(web_summary)
        st.markdown("### Vector Insights")
        if vector_result.get("insights"):
            for i, insight in enumerate(vector_result["insights"], start=1):
                st.markdown(f"{i}. {insight}")
        else:
            st.info(vector_result.get("reason", "No vector insights returned."))
        st.markdown("### Final Synthesis")
        st.markdown(final_report)

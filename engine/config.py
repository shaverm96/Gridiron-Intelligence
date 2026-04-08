from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _normalize_model_name(model_name: str, default_model: str) -> str:
    value = str(model_name or "").strip() or default_model
    alias_map = {
        "gemini-3.0-flash": "gemini-3-flash-preview",
        "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
    }
    return alias_map.get(value, value)


def resolve_project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = resolve_project_root()

if load_dotenv is not None:
    secrets_env = PROJECT_ROOT / "SECRETS.env"
    supabase_env = PROJECT_ROOT / "SUPABASE.env"
    gemini_env = PROJECT_ROOT / "GEMINI_API_KEY.env"
    if secrets_env.exists():
        load_dotenv(secrets_env, override=False)
    if supabase_env.exists():
        load_dotenv(supabase_env, override=False)
    if gemini_env.exists():
        load_dotenv(gemini_env, override=False)


CONFIG = {
    "SUPABASE_URL": os.getenv("SUPABASE_URL", ""),
    "SUPABASE_SERVICE_ROLE_KEY": os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
    "CFBD_API_KEY": os.getenv("CFBD_API_KEY", "") or os.getenv("CFBD_API", ""),
    "CFBD_BASE_URL": os.getenv("CFBD_BASE_URL", "https://api.collegefootballdata.com"),
    "FINAL_MODEL": _normalize_model_name(
        os.getenv("GI_FINAL_MODEL", "gemini-3-flash-preview"),
        "gemini-3-flash-preview",
    ),
    "SUMMARY_MODEL": _normalize_model_name(
        os.getenv("GI_SUMMARY_MODEL", "gemini-3.1-flash-lite-preview"),
        "gemini-3.1-flash-lite-preview",
    ),
    "WEB_QUERY_MAX_RESULTS": int(os.getenv("GI_WEB_QUERY_MAX_RESULTS", "6")),
    "PROMPT_PAYLOAD_MAX_CHARS": int(os.getenv("GI_PROMPT_PAYLOAD_MAX_CHARS", "12000")),
    "FINAL_PROMPT_MAX_CHARS": int(os.getenv("GI_FINAL_PROMPT_MAX_CHARS", "20000")),
    "VECTOR_MATCH_COUNT": int(os.getenv("GI_VECTOR_MATCH_COUNT", "6")),
    "VECTOR_MATCH_THRESHOLD": float(os.getenv("GI_VECTOR_MATCH_THRESHOLD", "0.15")),
    "VECTOR_RPC_NAME": os.getenv("GI_VECTOR_RPC_NAME", "match_gi_factoids"),
    "IDENTITY_CANDIDATE_LIMIT": int(os.getenv("GI_IDENTITY_CANDIDATE_LIMIT", "8")),
    "IDENTITY_TOP_CANDIDATES": int(os.getenv("GI_IDENTITY_TOP_CANDIDATES", "3")),
    "IDENTITY_CONFIDENCE_THRESHOLD": float(os.getenv("GI_IDENTITY_CONFIDENCE_THRESHOLD", "0.65")),
    "TARGET_SEARCH_SITES": [
        "maxpreps.com",
        "247sports.com",
        "rivals.com",
        "espn.com",
        "on3.com",
    ],
}

TABLES = {
    "recruit_master": "gi_recruit_master",
    "college_master": "gi_college_master",
    "player_link_bridge": "gi_player_link_bridge",
    "player_master": "gi_recruit_master",
    "scouting_features": "gi_scouting_report_features",
    "pred_score": "gi_model_prediction_score",
    "pred_threshold": "gi_model_prediction_thresholds",
}

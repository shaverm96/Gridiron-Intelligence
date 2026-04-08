from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


MODEL_ALIAS_MAP = {
    "gemini-3.0-flash": "gemini-3-flash-preview",
}


def normalize_model_name(model_name: str, default_model: str = "") -> str:
    value = str(model_name or "").strip() or str(default_model or "").strip()
    return MODEL_ALIAS_MAP.get(value, value)


def env_flag(name: str, default: bool = False) -> bool:
    value = str(os.getenv(name, "")).strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


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
    "FINAL_MODEL": normalize_model_name(
        os.getenv("GI_FINAL_MODEL", "gemini-3-flash-preview"),
        "gemini-3-flash-preview",
    ),
    "SUMMARY_MODEL": normalize_model_name(
        os.getenv("GI_SUMMARY_MODEL", "gemini-2.5-flash-lite"),
        "gemini-2.5-flash-lite",
    ),
    "LOCAL_CFBD_DEBUGGER_ENABLED": env_flag("GI_ENABLE_LOCAL_CFBD_DEBUGGER", default=False),
    "VECTOR_MATCH_COUNT": int(os.getenv("GI_VECTOR_MATCH_COUNT", "6")),
    "VECTOR_MATCH_THRESHOLD": float(os.getenv("GI_VECTOR_MATCH_THRESHOLD", "0.15")),
    "VECTOR_RPC_NAME": os.getenv("GI_VECTOR_RPC_NAME", "match_gi_factoids"),
    "IDENTITY_CANDIDATE_LIMIT": int(os.getenv("GI_IDENTITY_CANDIDATE_LIMIT", "8")),
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

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def resolve_project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for candidate in candidates:
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = resolve_project_root()

if load_dotenv is not None:
    supabase_env = PROJECT_ROOT / "SUPABASE.env"
    gemini_env = PROJECT_ROOT / "GEMINI_API_KEY.env"
    if supabase_env.exists():
        load_dotenv(supabase_env, override=False)
    if gemini_env.exists():
        load_dotenv(gemini_env, override=False)


CONFIG = {
    "SUPABASE_URL": os.getenv("SUPABASE_URL", ""),
    "SUPABASE_SERVICE_ROLE_KEY": os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
    "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", ""),
    "FINAL_MODEL": os.getenv("GI_FINAL_MODEL", "gemini-2.5-flash"),
    "SUMMARY_MODEL": os.getenv("GI_SUMMARY_MODEL", "gemini-2.5-flash-lite"),
    "VECTOR_MATCH_COUNT": int(os.getenv("GI_VECTOR_MATCH_COUNT", "6")),
    "VECTOR_MATCH_THRESHOLD": float(os.getenv("GI_VECTOR_MATCH_THRESHOLD", "0.15")),
    "VECTOR_RPC_NAME": os.getenv("GI_VECTOR_RPC_NAME", "match_gi_factoids"),
    "TARGET_SEARCH_SITES": [
        "maxpreps.com",
        "247sports.com",
        "rivals.com",
        "espn.com",
        "on3.com",
    ],
}

TABLES = {
    "player_master": "gi_player_master",
    "scouting_features": "gi_scouting_report_features",
    "pred_score": "gi_model_prediction_score",
    "pred_threshold": "gi_model_prediction_thresholds",
}

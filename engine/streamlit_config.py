from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


def _cfg_with_source(
    key: str,
    secrets: Mapping[str, Any] | None = None,
    default: str = "",
) -> tuple[str, str]:
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
        if secrets and key in secrets:
            value = secrets.get(key, default)
            return (str(value).strip() if value is not None else "", "streamlit_secrets")
    except Exception:
        pass

    if require_secrets and key in sensitive_keys:
        return default, "required_streamlit_secrets_missing"

    env_value = os.getenv(key)
    if env_value is not None:
        return env_value.strip(), "environment"
    return default, "default"


def _cfg(key: str, secrets: Mapping[str, Any] | None = None, default: str = "") -> str:
    value, _ = _cfg_with_source(key, secrets=secrets, default=default)
    return value


def _parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "on"}


def _cfg_flag_with_source(
    key: str,
    secrets: Mapping[str, Any] | None = None,
    default: bool = False,
) -> tuple[bool, str]:
    try:
        if secrets and key in secrets:
            return _parse_bool(secrets.get(key), default), "streamlit_secrets"
    except Exception:
        pass

    env_value = os.getenv(key)
    if env_value is not None:
        return _parse_bool(env_value, default), "environment"
    return default, "default"


def resolve_streamlit_project_root() -> Path:
    candidates = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
    for candidate in candidates:
        if (candidate / "data" / "modeling_datasets").exists():
            return candidate
    return Path.cwd()


def build_streamlit_runtime_config_data(
    secrets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    project_root = resolve_streamlit_project_root()

    if load_dotenv is not None:
        for env_name in ("SECRETS.env", "SUPABASE.env", "GEMINI_API_KEY.env"):
            env_file = project_root / env_name
            if env_file.exists():
                load_dotenv(env_file, override=False)

    supabase_url, supabase_url_source = _cfg_with_source("SUPABASE_URL", secrets=secrets)
    service_role_key, service_role_key_source = _cfg_with_source("SUPABASE_SERVICE_ROLE_KEY", secrets=secrets)
    gemini_api_key, gemini_api_key_source = _cfg_with_source("GEMINI_API_KEY", secrets=secrets)
    cfbd_api_key, cfbd_api_key_source = _cfg_with_source("CFBD_API_KEY", secrets=secrets)
    if not cfbd_api_key:
        cfbd_api_key, cfbd_api_key_source = _cfg_with_source("CFBD_API", secrets=secrets)

    local_debugger_enabled, local_debugger_source = _cfg_flag_with_source(
        "GI_ENABLE_LOCAL_CFBD_DEBUGGER",
        secrets=secrets,
        default=False,
    )

    config = {
        "SUPABASE_URL": supabase_url,
        "SUPABASE_SERVICE_ROLE_KEY": service_role_key,
        "GEMINI_API_KEY": gemini_api_key,
        "CFBD_API_KEY": cfbd_api_key,
        "CFBD_BASE_URL": _cfg("CFBD_BASE_URL", secrets=secrets, default="https://api.collegefootballdata.com"),
        "YEARS": [2026, 2027, 2028],
        "FINAL_MODEL": "gemini-3-flash-preview",
        "SUMMARY_MODEL": "gemini-3.1-flash-lite-preview",
        "VECTOR_MATCH_COUNT": 6,
        "VECTOR_MATCH_THRESHOLD": 0.15,
        "VECTOR_RPC_NAME": "match_gi_factoids",
        "LOCAL_CFBD_DEBUGGER_ENABLED": local_debugger_enabled,
    }

    config_sources = {
        "SUPABASE_URL": supabase_url_source,
        "SUPABASE_SERVICE_ROLE_KEY": service_role_key_source,
        "GEMINI_API_KEY": gemini_api_key_source,
        "CFBD_API_KEY": cfbd_api_key_source,
        "GI_ENABLE_LOCAL_CFBD_DEBUGGER": local_debugger_source,
    }

    tables = {
        "player_master": "gi_recruit_master",
        "scouting_features": "gi_scouting_report_features",
        "pred_score": "gi_model_prediction_score",
        "pred_threshold": "gi_model_prediction_thresholds",
    }

    target_teams = [
        "Alabama", "Auburn", "Clemson", "Colorado", "Duke", "Florida", "Florida State",
        "Georgia", "Georgia Tech", "LSU", "Miami", "Michigan", "NC State", "Notre Dame",
        "Ohio State", "Ole Miss", "Oregon", "South Carolina", "Tennessee", "Texas",
        "Texas A&M", "Charlotte", "USC", "Virginia Tech", "Wake Forest",
    ]

    pos_map = {
        "CB": "DB", "S": "DB", "FS": "DB", "SS": "DB", "DE": "EDGE", "DT": "IDL", "NT": "IDL", "LB": "LB", "OLB": "LB", "ILB": "LB", "OL": "OL", "OT": "OL", "OG": "OL", "C": "OL", "QB": "QB", "PRO": "QB", "DUAL": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "K": "SPEC", "P": "SPEC", "PK": "SPEC", "LS": "SPEC", "RET": "SPEC", "TE": "TE", "WR": "WR",
    }

    return {
        "project_root": project_root,
        "config": config,
        "config_sources": config_sources,
        "tables": tables,
        "target_teams": target_teams,
        "pos_map": pos_map,
    }

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
        "MODEL_TOKEN_COSTS_PER_1M": {
            "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
            "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
        },
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
        "Army Black Knights | American Athletic Conference (AAC)",
        "Charlotte 49ers | American Athletic Conference (AAC)",
        "East Carolina Pirates | American Athletic Conference (AAC)",
        "Florida Atlantic Owls | American Athletic Conference (AAC)",
        "Memphis Tigers | American Athletic Conference (AAC)",
        "Navy Midshipmen | American Athletic Conference (AAC)",
        "North Texas Mean Green | American Athletic Conference (AAC)",
        "Rice Owls | American Athletic Conference (AAC)",
        "South Florida Bulls | American Athletic Conference (AAC)",
        "Temple Owls | American Athletic Conference (AAC)",
        "Tulane Green Wave | American Athletic Conference (AAC)",
        "Tulsa Golden Hurricane | American Athletic Conference (AAC)",
        "UAB Blazers | American Athletic Conference (AAC)",
        "UTSA Roadrunners | American Athletic Conference (AAC)",
        "Boston College Eagles | Atlantic Coast Conference (ACC)",
        "California Golden Bears | Atlantic Coast Conference (ACC)",
        "Clemson Tigers | Atlantic Coast Conference (ACC)",
        "Duke Blue Devils | Atlantic Coast Conference (ACC)",
        "Florida State Seminoles | Atlantic Coast Conference (ACC)",
        "Georgia Tech Yellow Jackets | Atlantic Coast Conference (ACC)",
        "Louisville Cardinals | Atlantic Coast Conference (ACC)",
        "Miami Hurricanes | Atlantic Coast Conference (ACC)",
        "NC State Wolfpack | Atlantic Coast Conference (ACC)",
        "North Carolina Tar Heels | Atlantic Coast Conference (ACC)",
        "Pittsburgh Panthers | Atlantic Coast Conference (ACC)",
        "SMU Mustangs | Atlantic Coast Conference (ACC)",
        "Stanford Cardinal | Atlantic Coast Conference (ACC)",
        "Syracuse Orange | Atlantic Coast Conference (ACC)",
        "Virginia Cavaliers | Atlantic Coast Conference (ACC)",
        "Virginia Tech Hokies | Atlantic Coast Conference (ACC)",
        "Wake Forest Demon Deacons | Atlantic Coast Conference (ACC)",
        "Illinois Fighting Illini | Big Ten Conference (B1G)",
        "Indiana Hoosiers | Big Ten Conference (B1G)",
        "Iowa Hawkeyes | Big Ten Conference (B1G)",
        "Maryland Terrapins | Big Ten Conference (B1G)",
        "Michigan Wolverines | Big Ten Conference (B1G)",
        "Michigan State Spartans | Big Ten Conference (B1G)",
        "Minnesota Golden Gophers | Big Ten Conference (B1G)",
        "Nebraska Cornhuskers | Big Ten Conference (B1G)",
        "Northwestern Wildcats | Big Ten Conference (B1G)",
        "Ohio State Buckeyes | Big Ten Conference (B1G)",
        "Oregon Ducks | Big Ten Conference (B1G)",
        "Penn State Nittany Lions | Big Ten Conference (B1G)",
        "Purdue Boilermakers | Big Ten Conference (B1G)",
        "Rutgers Scarlet Knights | Big Ten Conference (B1G)",
        "UCLA Bruins | Big Ten Conference (B1G)",
        "USC Trojans | Big Ten Conference (B1G)",
        "Washington Huskies | Big Ten Conference (B1G)",
        "Wisconsin Badgers | Big Ten Conference (B1G)",
        "Arizona Wildcats | Big 12 Conference",
        "Arizona State Sun Devils | Big 12 Conference",
        "Baylor Bears | Big 12 Conference",
        "BYU Cougars | Big 12 Conference",
        "Cincinnati Bearcats | Big 12 Conference",
        "Colorado Buffaloes | Big 12 Conference",
        "Houston Cougars | Big 12 Conference",
        "Iowa State Cyclones | Big 12 Conference",
        "Kansas Jayhawks | Big 12 Conference",
        "Kansas State Wildcats | Big 12 Conference",
        "Oklahoma State Cowboys | Big 12 Conference",
        "TCU Horned Frogs | Big 12 Conference",
        "Texas Tech Red Raiders | Big 12 Conference",
        "UCF Knights | Big 12 Conference",
        "Utah Utes | Big 12 Conference",
        "West Virginia Mountaineers | Big 12 Conference",
        "FIU Panthers | Conference USA (C-USA)",
        "Jacksonville State Gamecocks | Conference USA (C-USA)",
        "Kennesaw State Owls | Conference USA (C-USA)",
        "Liberty Flames | Conference USA (C-USA)",
        "Louisiana Tech Bulldogs | Conference USA (C-USA)",
        "Middle Tennessee Blue Raiders | Conference USA (C-USA)",
        "New Mexico State Aggies | Conference USA (C-USA)",
        "Sam Houston Bearkats | Conference USA (C-USA)",
        "UTEP Miners | Conference USA (C-USA)",
        "Western Kentucky Hilltoppers | Conference USA (C-USA)",
        "Akron Zips | Mid-American Conference (MAC)",
        "Ball State Cardinals | Mid-American Conference (MAC)",
        "Bowling Green Falcons | Mid-American Conference (MAC)",
        "Buffalo Bulls | Mid-American Conference (MAC)",
        "Central Michigan Chippewas | Mid-American Conference (MAC)",
        "Eastern Michigan Eagles | Mid-American Conference (MAC)",
        "Kent State Golden Flashes | Mid-American Conference (MAC)",
        "Miami (OH) RedHawks | Mid-American Conference (MAC)",
        "Northern Illinois Huskies | Mid-American Conference (MAC)",
        "Ohio Bobcats | Mid-American Conference (MAC)",
        "Toledo Rockets | Mid-American Conference (MAC)",
        "Western Michigan Broncos | Mid-American Conference (MAC)",
        "Air Force Falcons | Mountain West Conference (MWC)",
        "Boise State Broncos | Mountain West Conference (MWC)",
        "Colorado State Rams | Mountain West Conference (MWC)",
        "Fresno State Bulldogs | Mountain West Conference (MWC)",
        "Hawaii Rainbow Warriors | Mountain West Conference (MWC)",
        "Nevada Wolf Pack | Mountain West Conference (MWC)",
        "New Mexico Lobos | Mountain West Conference (MWC)",
        "San Diego State Aztecs | Mountain West Conference (MWC)",
        "San Jose State Spartans | Mountain West Conference (MWC)",
        "UNLV Rebels | Mountain West Conference (MWC)",
        "Utah State Aggies | Mountain West Conference (MWC)",
        "Wyoming Cowboys | Mountain West Conference (MWC)",
        "Oregon State Beavers | Pac-12 Conference",
        "Washington State Cougars | Pac-12 Conference",
        "Alabama Crimson Tide | Southeastern Conference (SEC)",
        "Arkansas Razorbacks | Southeastern Conference (SEC)",
        "Auburn Tigers | Southeastern Conference (SEC)",
        "Florida Gators | Southeastern Conference (SEC)",
        "Georgia Bulldogs | Southeastern Conference (SEC)",
        "Kentucky Wildcats | Southeastern Conference (SEC)",
        "LSU Tigers | Southeastern Conference (SEC)",
        "Mississippi State Bulldogs | Southeastern Conference (SEC)",
        "Missouri Tigers | Southeastern Conference (SEC)",
        "Oklahoma Sooners | Southeastern Conference (SEC)",
        "Ole Miss Rebels | Southeastern Conference (SEC)",
        "South Carolina Gamecocks | Southeastern Conference (SEC)",
        "Tennessee Volunteers | Southeastern Conference (SEC)",
        "Texas Longhorns | Southeastern Conference (SEC)",
        "Texas A&M Aggies | Southeastern Conference (SEC)",
        "Vanderbilt Commodores | Southeastern Conference (SEC)",
        "Appalachian State Mountaineers | Sun Belt Conference",
        "Arkansas State Red Wolves | Sun Belt Conference",
        "Coastal Carolina Chanticleers | Sun Belt Conference",
        "Georgia Southern Eagles | Sun Belt Conference",
        "Georgia State Panthers | Sun Belt Conference",
        "James Madison Dukes | Sun Belt Conference",
        "Louisiana Ragin' Cajuns | Sun Belt Conference",
        "Louisiana-Monroe (ULM) Warhawks | Sun Belt Conference",
        "Marshall Thundering Herd | Sun Belt Conference",
        "Old Dominion Monarchs | Sun Belt Conference",
        "South Alabama Jaguars | Sun Belt Conference",
        "Southern Miss Golden Eagles | Sun Belt Conference",
        "Texas State Bobcats | Sun Belt Conference",
        "Troy Trojans | Sun Belt Conference",
        "Notre Dame Fighting Irish | Independents",
        "UConn Huskies | Independents",
        "UMass Minutemen | Independents",
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

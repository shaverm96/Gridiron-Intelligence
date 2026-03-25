from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import CONFIG


def _cfbd_api_key() -> str:
    return str(CONFIG.get("CFBD_API_KEY") or "").strip()


def _cfbd_base_url() -> str:
    return str(CONFIG.get("CFBD_BASE_URL") or "https://api.collegefootballdata.com").rstrip("/")


def cfbd_fetch(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    api_key = _cfbd_api_key()
    if not api_key:
        return {
            "status": "skipped",
            "reason": "CFBD_API_KEY missing",
            "data": [],
            "citations": [],
            "meta": {"endpoint": endpoint, "params": params or {}},
        }

    base_url = _cfbd_base_url()
    safe_params = {k: v for k, v in (params or {}).items() if v is not None and str(v).strip() != ""}
    url = f"{base_url}/{endpoint.lstrip('/')}"
    if safe_params:
        url = f"{url}?{urlencode(safe_params)}"

    request = Request(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
    )

    try:
        with urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body) if body else []
    except URLError as exc:
        return {
            "status": "skipped",
            "reason": f"CFBD request failed: {exc}",
            "data": [],
            "citations": [],
            "meta": {"endpoint": endpoint, "params": safe_params},
        }
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": f"CFBD parse failed: {exc}",
            "data": [],
            "citations": [],
            "meta": {"endpoint": endpoint, "params": safe_params},
        }

    return {
        "status": "ok",
        "reason": "CFBD fetch complete",
        "data": data,
        "citations": [{"source_type": "api", "source_name": "cfbd", "source_url": url}],
        "meta": {"endpoint": endpoint, "params": safe_params},
    }


def fetch_player_stats(
    athlete_id: str | None = None,
    team: str | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if athlete_id:
        params["athleteId"] = str(athlete_id)
    if team:
        params["team"] = str(team)
    if year:
        params["year"] = int(year)
    return cfbd_fetch(endpoint="player/stats", params=params)


def search_player_candidates(
    search_term: str,
    year: int | None = None,
    team: str | None = None,
    position: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"searchTerm": str(search_term or "").strip()}
    if year:
        params["year"] = int(year)
    if team:
        params["team"] = str(team)
    if position:
        params["position"] = str(position)
    return cfbd_fetch(endpoint="player/search", params=params)


def fetch_team_roster(team: str, year: int | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"team": str(team or "").strip()}
    if year:
        params["year"] = int(year)
    return cfbd_fetch(endpoint="roster", params=params)

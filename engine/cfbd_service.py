from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import CONFIG


def _cfbd_api_key() -> str:
    return str(CONFIG.get("CFBD_API_KEY") or "").strip()


def _cfbd_base_url() -> str:
    return str(CONFIG.get("CFBD_BASE_URL") or "https://api.collegefootballdata.com").rstrip("/")


class _CFBDRateLimitError(Exception):
    pass


@retry(
    retry=retry_if_exception_type(URLError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
    reraise=True,
)
def _fetch_cfbd_json(request: Request) -> Any:
    try:
        with urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
            if not body or not str(body).strip():
                return []
            stripped = str(body).lstrip()
            if stripped.startswith(("<!DOCTYPE", "<html", "<!--")):
                raise ValueError("CFBD returned HTML instead of JSON (likely wrong endpoint or unsupported params)")
            return json.loads(body)
    except HTTPError as exc:
        if int(getattr(exc, "code", 0) or 0) == 429:
            raise _CFBDRateLimitError("CFBD rate limit exceeded") from exc
        # Retry transient upstream failures through URLError channel.
        if int(getattr(exc, "code", 0) or 0) >= 500:
            raise URLError(str(exc)) from exc
        raise


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
        data = _fetch_cfbd_json(request)
    except _CFBDRateLimitError:
        return {
            "status": "skipped",
            "reason": "CFBD rate limit (429) reached",
            "data": [],
            "citations": [],
            "meta": {"endpoint": endpoint, "params": safe_params},
        }
    except HTTPError as exc:
        return {
            "status": "skipped",
            "reason": f"CFBD HTTP error: {exc}",
            "data": [],
            "citations": [],
            "meta": {"endpoint": endpoint, "params": safe_params},
        }
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
    # Backward-compatible shim: legacy callers expect player-level stats.
    # Route to supported player/usage endpoint with playerId when possible.
    params: dict[str, Any] = {}
    if year is not None:
        params["year"] = int(year)
    if team:
        params["team"] = str(team)
    athlete_text = str(athlete_id or "").strip()
    if athlete_text.isdigit():
        params["playerId"] = int(athlete_text)
    return cfbd_fetch(endpoint="player/usage", params=params)


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


def fetch_team_roster(
    team: str,
    year: int | None = None,
    classification: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"team": str(team or "").strip()}
    if year:
        params["year"] = int(year)
    if classification:
        params["classification"] = str(classification)
    return cfbd_fetch(endpoint="roster", params=params)


def fetch_recruits(
    year: int | None = None,
    team: str | None = None,
    position: str | None = None,
    state: str | None = None,
    classification: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if year is not None:
        params["year"] = int(year)
    if team:
        params["team"] = str(team)
    if position:
        params["position"] = str(position)
    if state:
        params["state"] = str(state)
    if classification:
        params["classification"] = str(classification)
    return cfbd_fetch(endpoint="recruiting", params=params)


def fetch_player_season_stats(
    year: int | None,
    conference: str | None = None,
    team: str | None = None,
    start_week: int | None = None,
    end_week: int | None = None,
    season_type: str | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    if year is None:
        return {
            "status": "skipped",
            "reason": "year is required for stats/player/season",
            "data": [],
            "citations": [],
            "meta": {"endpoint": "stats/player/season", "params": {}},
        }

    params: dict[str, Any] = {"year": int(year)}
    if conference:
        params["conference"] = str(conference)
    if team:
        params["team"] = str(team)
    if start_week is not None:
        params["startWeek"] = int(start_week)
    if end_week is not None:
        params["endWeek"] = int(end_week)
    if season_type:
        params["seasonType"] = str(season_type)
    if category:
        params["category"] = str(category)
    return cfbd_fetch(endpoint="stats/player/season", params=params)


def fetch_player_usage(
    year: int | None,
    conference: str | None = None,
    position: str | None = None,
    team: str | None = None,
    player_id: int | None = None,
    exclude_garbage_time: bool | None = None,
) -> dict[str, Any]:
    if year is None:
        return {
            "status": "skipped",
            "reason": "year is required for player/usage",
            "data": [],
            "citations": [],
            "meta": {"endpoint": "player/usage", "params": {}},
        }

    params: dict[str, Any] = {"year": int(year)}
    if conference:
        params["conference"] = str(conference)
    if position:
        params["position"] = str(position)
    if team:
        params["team"] = str(team)
    if player_id is not None:
        params["playerId"] = int(player_id)
    if exclude_garbage_time is not None:
        params["excludeGarbageTime"] = bool(exclude_garbage_time)
    return cfbd_fetch(endpoint="player/usage", params=params)

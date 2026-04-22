from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Callable

from .config import CONFIG
from .graph import get_scout_graph, get_structured_web_graph, get_transfer_chat_graph
from .state import ScoutState, initial_chat_state, initial_structured_state, initial_structured_web_state, compact_transfer_chat_state
from .supabase_client import fetch_college_player_bundle
from .tools import final_synthesis_tool, search_web_query_tool, summarize_payload_tool
from .cfbd_service import fetch_player_season_stats, fetch_player_usage


ProgressCallback = Callable[[dict[str, str]], None]

CHAT_STATE_MAX_TURNS = 6
CHAT_STATE_MAX_TRACE = 10
CHAT_STATE_MAX_ERRORS = 6
CHAT_STATE_MAX_CITATIONS = 16
CHAT_STATE_MAX_CANDIDATES = 3


LOGGER = logging.getLogger(__name__)

TRANSFER_CFBD_CONTEXT_CACHE: dict[str, dict[str, Any]] = {}


def _transfer_cfbd_context_cache_key(
    college_player_id: str,
    cfbd_athlete_id: str,
    year: int,
    exclude_garbage_time: bool,
) -> str:
    return "|".join([
        str(college_player_id or ""),
        str(cfbd_athlete_id or ""),
        str(int(year)),
        str(bool(exclude_garbage_time)),
    ])


def _transfer_cfbd_context_cache_get(cache_key: str) -> dict[str, Any] | None:
    if not bool(CONFIG.get("TRANSFER_CFBD_CACHE_ENABLED", True)):
        return None
    entry = TRANSFER_CFBD_CONTEXT_CACHE.get(cache_key)
    if not isinstance(entry, dict):
        return None
    ttl_seconds = int(CONFIG.get("TRANSFER_CFBD_CACHE_TTL_SECONDS", 1800))
    created_at = float(entry.get("created_at") or 0.0)
    if ttl_seconds > 0 and (time.time() - created_at) > ttl_seconds:
        TRANSFER_CFBD_CONTEXT_CACHE.pop(cache_key, None)
        return None
    value = entry.get("value")
    return dict(value) if isinstance(value, dict) else None


def _transfer_cfbd_context_cache_set(cache_key: str, value: dict[str, Any]) -> None:
    if not bool(CONFIG.get("TRANSFER_CFBD_CACHE_ENABLED", True)):
        return
    TRANSFER_CFBD_CONTEXT_CACHE[cache_key] = {
        "created_at": time.time(),
        "value": dict(value or {}),
    }
    max_entries = max(1, int(CONFIG.get("TRANSFER_CFBD_CACHE_MAX_ENTRIES", 256)))
    if len(TRANSFER_CFBD_CONTEXT_CACHE) <= max_entries:
        return
    ordered = sorted(
        TRANSFER_CFBD_CONTEXT_CACHE.items(),
        key=lambda item: float((item[1] or {}).get("created_at") or 0.0),
    )
    for key, _ in ordered[: max(0, len(ordered) - max_entries)]:
        TRANSFER_CFBD_CONTEXT_CACHE.pop(key, None)


def _extract_tool_telemetry(payload: dict[str, Any] | None) -> dict[str, Any]:
    src = dict(payload or {})
    telemetry = src.get("telemetry")
    if isinstance(telemetry, dict):
        return dict(telemetry)
    legacy = src.get("_telemetry")
    if isinstance(legacy, dict):
        return dict(legacy)
    return {}


def _rollup_telemetry_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rollup = {
        "model_call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
        "latency_ms": 0,
    }
    for row in rows:
        if not isinstance(row, dict) or not row:
            continue
        if bool(row.get("cache_hit")):
            continue
        rollup["model_call_count"] += 1
        input_tokens = row.get("input_tokens")
        output_tokens = row.get("output_tokens")
        total_tokens = row.get("total_tokens")
        latency_ms = row.get("latency_ms")
        cost = row.get("estimated_cost_usd")

        if isinstance(input_tokens, (int, float)):
            rollup["input_tokens"] += int(input_tokens)
        if isinstance(output_tokens, (int, float)):
            rollup["output_tokens"] += int(output_tokens)
        if isinstance(total_tokens, (int, float)):
            rollup["total_tokens"] += int(total_tokens)
        if isinstance(latency_ms, (int, float)):
            rollup["latency_ms"] += int(latency_ms)
        if isinstance(cost, (int, float)):
            rollup["estimated_cost_usd"] += float(cost)

    if rollup["total_tokens"] == 0:
        rollup["total_tokens"] = rollup["input_tokens"] + rollup["output_tokens"]
    rollup["estimated_cost_usd"] = round(float(rollup["estimated_cost_usd"]), 8)
    return rollup


def _ensure_base_state(state: dict[str, Any] | None) -> ScoutState:
    base = dict(state or {})
    if "mode" not in base:
        base["mode"] = "chat"
    if "conversation_history" not in base:
        base["conversation_history"] = []
    if "errors" not in base:
        base["errors"] = []
    if "citations" not in base:
        base["citations"] = []
    if "trace_log" not in base:
        base["trace_log"] = []
    return _compact_chat_state(base)


def _compact_chat_state(state: dict[str, Any] | None) -> ScoutState:
    src = dict(state or {})
    compact: ScoutState = {
        "mode": "chat",
        "user_query": str(src.get("user_query") or ""),
        "target_player_name": str(src.get("target_player_name") or ""),
        "player_name": str(src.get("player_name") or ""),
        "recruit_id": str(src.get("recruit_id") or ""),
        "cfbd_athlete_id": str(src.get("cfbd_athlete_id") or ""),
        "target_team": str(src.get("target_team") or ""),
        "year": int(src.get("year") or 0),
        "active_report_context": dict(src.get("active_report_context") or {}),
        "delegator_plan": dict(src.get("delegator_plan") or {}),
        "cfbd_data_summary": str(src.get("cfbd_data_summary") or ""),
        "web_recruiting_summary": str(src.get("web_recruiting_summary") or ""),
        "web_team_summary": str(src.get("web_team_summary") or ""),
        "final_report": str(src.get("final_report") or ""),
        "next_step": str(src.get("next_step") or "supervisor"),
        "missing_fields": list(src.get("missing_fields") or []),
        "requires_identity_clarification": bool(src.get("requires_identity_clarification")),
        "clarification_prompt": str(src.get("clarification_prompt") or ""),
        "pending_identity_query": str(src.get("pending_identity_query") or ""),
        "security_halt": bool(src.get("security_halt")),
        "security_message": str(src.get("security_message") or ""),
    }

    compact["identity_candidates"] = list(src.get("identity_candidates") or [])[-CHAT_STATE_MAX_CANDIDATES:]
    compact["conversation_history"] = list(src.get("conversation_history") or [])[-CHAT_STATE_MAX_TURNS * 2:]
    compact["trace_log"] = list(src.get("trace_log") or [])[-CHAT_STATE_MAX_TRACE:]
    compact["errors"] = list(src.get("errors") or [])[-CHAT_STATE_MAX_ERRORS:]
    compact["citations"] = list(src.get("citations") or [])[-CHAT_STATE_MAX_CITATIONS:]
    compact["sql_data_context"] = {}
    compact["web_research_context"] = ""
    compact["vector_factoids"] = []
    compact["comparables_context"] = ""

    return compact


def _emit_progress(progress_callback: ProgressCallback | None, node: str, status: str) -> None:
    if progress_callback is None:
        return
    progress_callback({"node": str(node), "status": str(status)})


def _batch_settings() -> dict[str, Any]:
    return {
        "enabled": bool(CONFIG.get("BATCH_ENABLED", True)),
        "batch_size": max(1, int(CONFIG.get("BATCH_SIZE", 4) or 4)),
        "concurrency": max(1, int(CONFIG.get("BATCH_CONCURRENCY", 3) or 3)),
        "retries": max(0, int(CONFIG.get("BATCH_RETRIES", 2) or 2)),
        "timeout_seconds": max(1, int(CONFIG.get("BATCH_TIMEOUT_SECONDS", 45) or 45)),
        "rate_limit_per_second": max(0.0, float(CONFIG.get("BATCH_RATE_LIMIT_PER_SECOND", 0) or 0)),
        "resume_enabled": bool(CONFIG.get("BATCH_RESUME_ENABLED", True)),
        "checkpoint_dir": str(CONFIG.get("BATCH_CHECKPOINT_DIR") or "").strip(),
    }


def _batch_checkpoint_path(checkpoint_scope: str | None, settings: dict[str, Any]) -> Path | None:
    checkpoint_dir = str(settings.get("checkpoint_dir") or "").strip()
    if not checkpoint_dir or not checkpoint_scope:
        return None
    safe_scope = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(checkpoint_scope)).strip("_") or "batch"
    root = Path(checkpoint_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{safe_scope}.json"


def _load_batch_checkpoint(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"completed": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"completed": {}}
    completed = dict(raw.get("completed") or {}) if isinstance(raw, dict) else {}
    return {"completed": completed}


def _save_batch_checkpoint(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    try:
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")
    except Exception as exc:
        LOGGER.warning("Failed to write batch checkpoint %s: %s", str(path), str(exc))


def _is_retryable_batch_reason(reason: str) -> bool:
    text = str(reason or "").strip().lower()
    if not text:
        return False
    retryable_markers = [
        "timed out",
        "rate limit",
        "429",
        "request failed",
        "http error",
        "failed",
        "tempor",
        "unavailable",
    ]
    return any(marker in text for marker in retryable_markers)


def _run_batched_tasks(
    items: list[Any],
    task_fn: Callable[[Any], dict[str, Any]],
    item_key_fn: Callable[[Any], str],
    fallback_fn: Callable[[Any, str], dict[str, Any]],
    checkpoint_scope: str | None = None,
) -> list[dict[str, Any]]:
    settings = _batch_settings()
    LOGGER.info(
        "Batch task start: scope=%s items=%s enabled=%s batch_size=%s concurrency=%s retries=%s timeout=%ss",
        str(checkpoint_scope or ""),
        len(items),
        bool(settings.get("enabled")),
        int(settings.get("batch_size") or 1),
        int(settings.get("concurrency") or 1),
        int(settings.get("retries") or 0),
        int(settings.get("timeout_seconds") or 0),
    )
    checkpoint_path = _batch_checkpoint_path(checkpoint_scope, settings)
    checkpoint = _load_batch_checkpoint(checkpoint_path)
    completed_map = dict(checkpoint.get("completed") or {})

    ordered_results: list[dict[str, Any] | None] = [None for _ in items]
    pending: list[tuple[int, Any, str]] = []

    for idx, item in enumerate(items):
        item_key = str(item_key_fn(item) or idx)
        cached = completed_map.get(item_key)
        if settings.get("resume_enabled") and isinstance(cached, dict) and isinstance(cached.get("value"), dict):
            ordered_results[idx] = {
                "value": dict(cached.get("value") or {}),
                "attempts": int(cached.get("attempts") or 0),
                "from_checkpoint": True,
                "item_key": item_key,
            }
            continue
        pending.append((idx, item, item_key))

    retries = int(settings.get("retries") or 0)

    def _execute_item(item: Any, item_key: str) -> dict[str, Any]:
        attempts = 0
        last_reason = ""
        while attempts <= retries:
            attempts += 1
            try:
                value = dict(task_fn(item) or {})
            except Exception as exc:
                last_reason = str(exc).strip() or repr(exc)
                LOGGER.warning("Batch task exception item_key=%s attempt=%s error=%s", item_key, attempts, last_reason)
                if attempts <= retries:
                    time.sleep(min(3.0, 0.5 * (2 ** (attempts - 1))))
                    continue
                value = fallback_fn(item, f"task failed after {attempts} attempts: {last_reason}")
                return {
                    "value": dict(value or {}),
                    "attempts": attempts,
                    "from_checkpoint": False,
                    "item_key": item_key,
                }

            status = str(value.get("status") or "").lower()
            reason = str(value.get("reason") or "")
            if status in {"ok", "completed", "success"}:
                return {
                    "value": value,
                    "attempts": attempts,
                    "from_checkpoint": False,
                    "item_key": item_key,
                }
            last_reason = reason
            if attempts <= retries and _is_retryable_batch_reason(reason):
                LOGGER.warning("Batch task retry item_key=%s attempt=%s reason=%s", item_key, attempts, reason)
                time.sleep(min(3.0, 0.5 * (2 ** (attempts - 1))))
                continue
            return {
                "value": value,
                "attempts": attempts,
                "from_checkpoint": False,
                "item_key": item_key,
            }

        value = fallback_fn(item, f"task failed after retries: {last_reason}")
        return {
            "value": dict(value or {}),
            "attempts": retries + 1,
            "from_checkpoint": False,
            "item_key": item_key,
        }

    def _persist_completed(result: dict[str, Any]) -> None:
        item_key = str(result.get("item_key") or "")
        if not item_key:
            return
        completed_map[item_key] = {
            "value": dict(result.get("value") or {}),
            "attempts": int(result.get("attempts") or 0),
        }
        _save_batch_checkpoint(checkpoint_path, {"completed": completed_map})

    if not pending:
        return [result for result in ordered_results if isinstance(result, dict)]

    use_parallel = bool(settings.get("enabled")) and int(settings.get("concurrency") or 1) > 1
    batch_size = max(1, int(settings.get("batch_size") or 1))
    concurrency = max(1, int(settings.get("concurrency") or 1))
    timeout_seconds = max(1, int(settings.get("timeout_seconds") or 45))
    rate_limit = max(0.0, float(settings.get("rate_limit_per_second") or 0.0))
    submit_interval = (1.0 / rate_limit) if rate_limit > 0 else 0.0

    if not use_parallel:
        for idx, item, item_key in pending:
            result = _execute_item(item, item_key)
            ordered_results[idx] = result
            _persist_completed(result)
    else:
        for start in range(0, len(pending), batch_size):
            chunk = pending[start : start + batch_size]
            worker_count = min(concurrency, len(chunk))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures: dict[Any, tuple[int, Any, str]] = {}
                last_submit_ts = 0.0
                for idx, item, item_key in chunk:
                    if submit_interval > 0 and last_submit_ts > 0:
                        elapsed = time.monotonic() - last_submit_ts
                        if elapsed < submit_interval:
                            time.sleep(submit_interval - elapsed)
                    future = executor.submit(_execute_item, item, item_key)
                    futures[future] = (idx, item, item_key)
                    last_submit_ts = time.monotonic()

                for future, (idx, item, item_key) in futures.items():
                    try:
                        result = dict(future.result(timeout=timeout_seconds) or {})
                    except TimeoutError:
                        LOGGER.warning("Batch task timeout item_key=%s timeout=%ss", item_key, timeout_seconds)
                        result = {
                            "value": fallback_fn(item, f"task timed out after {timeout_seconds}s"),
                            "attempts": retries + 1,
                            "from_checkpoint": False,
                            "item_key": item_key,
                        }
                    except Exception as exc:
                        LOGGER.warning("Batch task future failed item_key=%s error=%s", item_key, str(exc).strip() or repr(exc))
                        result = {
                            "value": fallback_fn(item, f"task failed: {str(exc).strip() or repr(exc)}"),
                            "attempts": retries + 1,
                            "from_checkpoint": False,
                            "item_key": item_key,
                        }

                    ordered_results[idx] = result
                    _persist_completed(result)

    final_results = [result for result in ordered_results if isinstance(result, dict)]
    checkpoint_completed = sum(1 for row in final_results if bool(row.get("from_checkpoint")))
    if checkpoint_completed:
        LOGGER.info("Batch resume reused %s checkpointed item(s) for scope=%s", checkpoint_completed, str(checkpoint_scope or ""))
    LOGGER.info("Batch task complete: scope=%s processed=%s", str(checkpoint_scope or ""), len(final_results))
    return final_results


def _diagnostic_scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list, tuple, set)) for item in value):
            return " | ".join([str(item) for item in value])
    try:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except Exception:
        return str(value)


def _merge_update(state: ScoutState, update: dict[str, Any]) -> ScoutState:
    merged = dict(state)
    for key, value in dict(update or {}).items():
        if key in {"citations", "errors", "trace_log"}:
            merged[key] = list(merged.get(key, [])) + list(value or [])
        else:
            merged[key] = value
    return merged


def _candidate_name(row: dict[str, Any]) -> str:
    for key in ("player_name", "full_name", "recruit_name"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _try_resolve_clarification_response(state: ScoutState, user_prompt: str) -> ScoutState:
    if not bool(state.get("requires_identity_clarification")):
        return state

    prompt = str(user_prompt or "").strip()
    candidates = list(state.get("identity_candidates") or [])
    if not prompt or not candidates:
        return state

    selected: dict[str, Any] | None = None

    if re.fullmatch(r"\d+", prompt):
        idx = int(prompt) - 1
        if 0 <= idx < len(candidates):
            selected = dict(candidates[idx])
    else:
        normalized = prompt.lower()
        for row in candidates:
            name = _candidate_name(row)
            if name and name.lower() == normalized:
                selected = dict(row)
                break

    if selected is None:
        return state

    pending_query = str(state.get("pending_identity_query") or "").strip()
    selected_name = _candidate_name(selected)
    selected_recruit_id = str(selected.get("recruit_id") or "").strip()
    selected_athlete_id = str(selected.get("cfbd_athlete_id") or "").strip()

    updates: ScoutState = {
        "requires_identity_clarification": False,
        "clarification_prompt": "",
        "identity_candidates": [],
        "recruit_id": selected_recruit_id,
        "cfbd_athlete_id": selected_athlete_id,
        "target_player_name": selected_name or str(state.get("target_player_name") or ""),
        "missing_fields": [],
        "pending_identity_query": "",
    }
    if pending_query:
        updates["user_query"] = pending_query

    return _merge_update(state, updates)


def _invoke_graph(
    graph_runner: Any,
    state: ScoutState,
    progress_callback: ProgressCallback | None = None,
) -> ScoutState:
    if hasattr(graph_runner, "stream"):
        _emit_progress(progress_callback, "workflow", "started")
        latest_state: ScoutState = dict(state)
        for event in graph_runner.stream(state, stream_mode="updates"):
            if not isinstance(event, dict):
                continue
            for node_name, update in event.items():
                if node_name == "__start__":
                    _emit_progress(progress_callback, "workflow", "running")
                    continue
                if node_name == "__end__":
                    _emit_progress(progress_callback, "workflow", "completed")
                    continue
                _emit_progress(progress_callback, str(node_name), "completed")
                if isinstance(update, dict):
                    latest_state = _merge_update(latest_state, update)
        return latest_state

    if hasattr(graph_runner, "invoke_with_progress"):
        return graph_runner.invoke_with_progress(state, progress_callback=progress_callback)

    _emit_progress(progress_callback, "workflow", "started")
    result = graph_runner.invoke(state)
    _emit_progress(progress_callback, "workflow", "completed")
    return result


def orchestrate_structured_report(
    player_name: str,
    recruit_id: str,
    target_team: str,
    year: int,
    user_query: str | None = None,
    graph: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ScoutState:
    started = time.perf_counter()
    graph_runner = graph or get_scout_graph()
    state = initial_structured_state(
        player_name=player_name,
        recruit_id=str(recruit_id),
        target_team=target_team,
        year=int(year),
    )
    state["target_player_name"] = player_name
    state["user_query"] = user_query or (
        f"Create a scouting report for {player_name} and evaluate fit for {target_team}."
    )
    state["mode"] = "structured_report"
    result_state = _invoke_graph(graph_runner, state, progress_callback=progress_callback)
    if isinstance(result_state, dict):
        trace = list(result_state.get("trace_log") or [])
        trace.append(
            {
                "node": "orchestration",
                "status": "completed",
                "mode": "structured_report",
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )
        result_state["trace_log"] = trace[-CHAT_STATE_MAX_TRACE:]
    return result_state


def orchestrate_chat_turn(
    user_prompt: str,
    current_state: dict[str, Any] | None = None,
    target_team: str = "",
    target_player_name: str = "",
    graph: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ScoutState:
    started = time.perf_counter()
    graph_runner = graph or get_scout_graph()
    state = _ensure_base_state(current_state)
    if not state:
        state = initial_chat_state(user_prompt)

    state["mode"] = "chat"
    state["user_query"] = str(user_prompt or "").strip()
    if target_team:
        state["target_team"] = target_team
    if target_player_name:
        state["target_player_name"] = target_player_name

    state = _try_resolve_clarification_response(state, user_prompt)
    state = _compact_chat_state(state)

    result_state = _invoke_graph(graph_runner, state, progress_callback=progress_callback)
    if isinstance(result_state, dict):
        trace = list(result_state.get("trace_log") or [])
        trace.append(
            {
                "node": "orchestration",
                "status": "completed",
                "mode": "chat",
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )
        result_state["trace_log"] = trace[-CHAT_STATE_MAX_TRACE:]
    return result_state


def orchestrate_structured_web_scouting(
    player_name: str,
    recruit_id: str,
    target_team: str,
    year: int,
    user_query: str | None = None,
    graph: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ScoutState:
    started = time.perf_counter()
    graph_runner = graph or get_structured_web_graph()
    state = initial_structured_web_state(
        player_name=player_name,
        recruit_id=str(recruit_id),
        target_team=target_team,
        year=int(year),
    )
    state["target_player_name"] = player_name
    state["user_query"] = user_query or (
        f"Create a structured scouting brief for {player_name} and evaluate fit for {target_team}."
    )
    state["mode"] = "structured_report"
    result_state = _invoke_graph(graph_runner, state, progress_callback=progress_callback)
    if isinstance(result_state, dict):
        trace = list(result_state.get("trace_log") or [])
        trace.append(
            {
                "node": "orchestration",
                "status": "completed",
                "mode": "structured_web",
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
        )
        result_state["trace_log"] = trace[-CHAT_STATE_MAX_TRACE:]
    return result_state


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return int(float(text))
    except Exception:
        return None


def _transfer_player_name(player_row: dict[str, Any]) -> str:
    full_name = str(player_row.get("full_name") or "").strip()
    if full_name:
        return full_name
    first = str(player_row.get("first_name") or "").strip()
    last = str(player_row.get("last_name") or "").strip()
    return " ".join([part for part in [first, last] if part]).strip()


def _career_year_bounds(player_row: dict[str, Any], reference_year: int) -> tuple[int, int]:
    first = _safe_int(player_row.get("first_season"))
    last = _safe_int(player_row.get("last_season"))
    ref = int(reference_year)
    start_year = first if first is not None else ref
    end_year = last if last is not None else ref
    start_year = max(2010, min(start_year, ref))
    end_year = max(start_year, min(end_year, ref))
    return start_year, end_year


def _split_team_tokens(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"[|,;/]+", text) if str(part).strip()]
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return deduped


def _normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _name_tokens(value: Any) -> list[str]:
    norm = _normalize_name(value)
    return [tok for tok in norm.split(" ") if tok]


def _extract_candidate_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    direct_keys = [
        "athleteId",
        "athlete_id",
        "cfbd_athlete_id",
        "athlete",
        "playerId",
        "player_id",
        "cfbd_player_id",
        "id",
    ]
    for key in direct_keys:
        value = row.get(key)
        text = str(value or "").strip()
        if text and text not in ids:
            ids.append(text)

    for key, value in row.items():
        lowered = str(key or "").lower()
        if not any(token in lowered for token in ["athlete", "player", "id"]):
            continue
        text = str(value or "").strip()
        if text and text.isdigit() and text not in ids:
            ids.append(text)

    return ids


def _name_match(candidate: str, target: str) -> bool:
    cand = _normalize_name(candidate)
    tgt = _normalize_name(target)
    if not cand or not tgt:
        return False
    if cand == tgt:
        return True
    cand_tokens = _name_tokens(cand)
    tgt_tokens = _name_tokens(tgt)
    if not cand_tokens or not tgt_tokens:
        return False

    # Match for abbreviated first names like "F Mendoza" vs "Fernando Mendoza".
    cand_last = cand_tokens[-1]
    tgt_last = tgt_tokens[-1]
    if cand_last == tgt_last:
        cand_first = cand_tokens[0]
        tgt_first = tgt_tokens[0]
        if cand_first == tgt_first:
            return True
        if len(cand_first) == 1 and tgt_first.startswith(cand_first):
            return True
        if len(tgt_first) == 1 and cand_first.startswith(tgt_first):
            return True

    # Token-subset fallback for multi-token variations.
    cand_set = set(cand_tokens)
    tgt_set = set(tgt_tokens)
    return bool(cand_set and tgt_set and (cand_set.issubset(tgt_set) or tgt_set.issubset(cand_set)))


def _filter_player_season_stats_rows(
    rows: list[dict[str, Any]],
    athlete_id_text: str,
    player_name: str,
) -> list[dict[str, Any]]:
    athlete = str(athlete_id_text or "").strip()
    athlete_int = _safe_int(athlete)
    target_name = _normalize_name(player_name)
    if not rows:
        return []

    filtered: list[dict[str, Any]] = []
    for row in rows:
        candidate_ids = _extract_candidate_ids(row)
        if athlete and athlete in candidate_ids:
            filtered.append(row)
            continue

        candidate_id_ints = [_safe_int(candidate) for candidate in candidate_ids]
        if athlete_int is not None and any(candidate_int == athlete_int for candidate_int in candidate_id_ints if candidate_int is not None):
            filtered.append(row)
            continue

        candidate_names = [
            str(row.get("player") or ""),
            str(row.get("playerName") or ""),
            str(row.get("name") or ""),
        ]
        if target_name and any(_name_match(name, target_name) for name in candidate_names if str(name or "").strip()):
            filtered.append(row)

    return filtered


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _compact_usage_table(career_usage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []
    for year_entry in career_usage:
        year = _safe_int(year_entry.get("year"))
        season_rows = list(year_entry.get("rows") or [])
        row0 = season_rows[0] if season_rows else {}
        usage = dict(row0.get("usage") or {}) if isinstance(row0, dict) else {}
        compact_rows.append(
            {
                "year": int(year) if year is not None else None,
                "team": str(row0.get("team") or ""),
                "position": str(row0.get("position") or ""),
                "overall": _to_float(usage.get("overall")),
                "pass": _to_float(usage.get("pass")),
                "rush": _to_float(usage.get("rush")),
                "third_down": _to_float(usage.get("thirdDown")),
                "passing_downs": _to_float(usage.get("passingDowns")),
                "record_count": int(year_entry.get("record_count") or 0),
                "status": str(year_entry.get("status") or "unknown"),
            }
        )
    compact_rows.sort(key=lambda r: int(r.get("year") or 0))
    return compact_rows


def _usage_yoy_deltas(compact_usage: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    prev: dict[str, Any] | None = None
    for current in compact_usage:
        if prev is None:
            prev = current
            continue

        def _delta(key: str) -> float | None:
            a = _to_float(current.get(key))
            b = _to_float(prev.get(key))
            if a is None or b is None:
                return None
            return round(a - b, 4)

        deltas.append(
            {
                "from_year": prev.get("year"),
                "to_year": current.get("year"),
                "overall_delta": _delta("overall"),
                "pass_delta": _delta("pass"),
                "rush_delta": _delta("rush"),
                "third_down_delta": _delta("third_down"),
                "passing_downs_delta": _delta("passing_downs"),
                "team_change": str(prev.get("team") or "") != str(current.get("team") or ""),
            }
        )
        prev = current
    return deltas


def _compact_season_stats_table(career_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_rows: list[dict[str, Any]] = []

    def _stat_key(category: str, stat_type: str) -> str:
        cat = re.sub(r"[^a-z0-9]+", "_", str(category or "").strip().lower()).strip("_")
        stat = re.sub(r"[^a-z0-9]+", "_", str(stat_type or "").strip().lower()).strip("_")
        if not cat:
            cat = "misc"
        if not stat:
            stat = "value"
        return f"{cat}_{stat}"

    for year_entry in career_stats:
        year = _safe_int(year_entry.get("year"))
        season_rows = list(year_entry.get("rows") or [])
        compact: dict[str, Any] = {
            "year": int(year) if year is not None else None,
            "record_count": int(year_entry.get("record_count") or 0),
            "status": str(year_entry.get("status") or "unknown"),
        }
        for row in season_rows:
            category = str(row.get("category") or "").strip().lower()
            stat_type = str(row.get("statType") or "").strip().upper()
            key = _stat_key(category, stat_type)
            value = _to_float(row.get("stat"))
            compact[key] = value if value is not None else row.get("stat")
        compact_rows.append(compact)

    compact_rows.sort(key=lambda r: int(r.get("year") or 0))
    return compact_rows


def orchestrate_transfer_cfbd_context(
    player_name: str,
    cfbd_athlete_id: str,
    position: str,
    teams: Any,
    year: int = 2025,
    first_season: int | None = None,
    last_season: int | None = None,
    exclude_garbage_time: bool | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    player_name_text = str(player_name or "").strip() or "Unknown Player"
    athlete_id_text = str(cfbd_athlete_id or "").strip()
    position_text = str(position or "").strip()
    ref_year = int(year)

    pseudo_player_row = {
        "first_season": first_season,
        "last_season": last_season,
    }
    career_start_year, career_end_year = _career_year_bounds(pseudo_player_row, ref_year)
    player_teams = _split_team_tokens(teams)

    _emit_progress(progress_callback, "cfbd_usage", "running")
    player_id = _safe_int(athlete_id_text)
    usage_position = position_text or None
    if player_id is not None:
        usage_position = None
    usage_by_year: list[dict[str, Any]] = []
    usage_citations: list[dict[str, str]] = []
    usage_diagnostics: list[dict[str, Any]] = []

    def _usage_task(season: int) -> dict[str, Any]:
        usage_result = fetch_player_usage(
            year=int(season),
            position=usage_position,
            player_id=player_id,
            exclude_garbage_time=exclude_garbage_time,
        )
        usage_result = dict(usage_result or {})
        usage_rows = list(usage_result.get("data") or [])
        usage_params = dict((usage_result.get("meta") or {}).get("params") or {})
        return {
            "status": str(usage_result.get("status") or "unknown"),
            "reason": str(usage_result.get("reason") or ""),
            "year_entry": {
                "year": int(season),
                "status": str(usage_result.get("status") or "unknown"),
                "reason": str(usage_result.get("reason") or ""),
                "record_count": len(usage_rows),
                "meta": usage_result.get("meta") if isinstance(usage_result, dict) else {},
                "rows": usage_rows,
                "result": usage_result,
            },
            "citations": list(usage_result.get("citations") or []),
            "diagnostic": {
                "year": int(season),
                "endpoint": "player/usage",
                "status": str(usage_result.get("status") or "unknown"),
                "reason": str(usage_result.get("reason") or ""),
                "rows_pre_filter": len(usage_rows),
                "rows_post_filter": len(usage_rows),
                "queried_teams": "",
                "queried_team_count": 0,
                "params_text": _diagnostic_scalar_text(usage_params),
                "fallback_policy": "player_usage_endpoint",
                "fallback_teamless_attempted": False,
            },
        }

    def _usage_fallback(season: int, reason: str) -> dict[str, Any]:
        safe_reason = str(reason or "usage pull failed")
        usage_result = {
            "status": "skipped",
            "reason": safe_reason,
            "data": [],
            "citations": [],
            "meta": {"params": {"year": int(season)}},
        }
        usage_rows: list[dict[str, Any]] = []
        return {
            "status": "skipped",
            "reason": safe_reason,
            "year_entry": {
                "year": int(season),
                "status": "skipped",
                "reason": safe_reason,
                "record_count": 0,
                "meta": usage_result.get("meta") or {},
                "rows": usage_rows,
                "result": usage_result,
            },
            "citations": [],
            "diagnostic": {
                "year": int(season),
                "endpoint": "player/usage",
                "status": "skipped",
                "reason": safe_reason,
                "rows_pre_filter": 0,
                "rows_post_filter": 0,
                "queried_teams": "",
                "queried_team_count": 0,
                "params_text": _diagnostic_scalar_text({"year": int(season)}),
                "fallback_policy": "player_usage_endpoint",
                "fallback_teamless_attempted": False,
            },
        }

    seasons = list(range(career_start_year, career_end_year + 1))
    usage_batch_results = _run_batched_tasks(
        items=seasons,
        task_fn=_usage_task,
        item_key_fn=lambda season: f"usage_{int(season)}",
        fallback_fn=_usage_fallback,
        checkpoint_scope=f"transfer_cfbd_usage_{athlete_id_text or player_name_text}_{ref_year}",
    )

    for row in usage_batch_results:
        payload = dict(row.get("value") or {})
        usage_by_year.append(dict(payload.get("year_entry") or {}))
        usage_citations.extend(list(payload.get("citations") or []))
        usage_diagnostics.append(dict(payload.get("diagnostic") or {}))
    _emit_progress(progress_callback, "cfbd_usage", "completed")

    _emit_progress(progress_callback, "cfbd_stats", "running")
    stats_by_year: list[dict[str, Any]] = []
    stats_citations: list[dict[str, str]] = []
    stats_diagnostics: list[dict[str, Any]] = []

    def _stats_task(season: int) -> dict[str, Any]:
        team_filters = list(player_teams)
        combined_rows: list[dict[str, Any]] = []
        combined_citations: list[dict[str, str]] = []
        reasons: list[str] = []
        statuses: list[str] = []
        raw_record_count = 0
        team_meta: list[dict[str, Any]] = []

        def _run_stats_pull(team_filter: str | None) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
            season_result = fetch_player_season_stats(
                year=int(season),
                team=team_filter,
                season_type="regular",
                category=None,
            )
            season_rows = list(season_result.get("data") or []) if isinstance(season_result, dict) else []
            filtered_rows = _filter_player_season_stats_rows(
                rows=season_rows,
                athlete_id_text=athlete_id_text,
                player_name=player_name_text,
            )
            return filtered_rows, season_result, season_rows

        if team_filters:
            for team_filter in team_filters:
                filtered_rows, season_result, season_rows = _run_stats_pull(team_filter)
                raw_record_count += len(season_rows)
                combined_rows.extend(filtered_rows)
                statuses.append(str(season_result.get("status") or "unknown"))
                reason_text = str(season_result.get("reason") or "").strip()
                if reason_text:
                    reasons.append(reason_text)
                combined_citations.extend(list(season_result.get("citations") or []))
                team_meta.append(
                    {
                        "team": team_filter,
                        "status": str(season_result.get("status") or "unknown"),
                        "record_count": len(filtered_rows),
                        "raw_record_count": len(season_rows),
                    }
                )
        else:
            statuses.append("skipped")
            reasons.append("No team filters supplied for season-stats pull; broad teamless fallback is disabled.")

        deduped_rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str, str, str, str]] = set()
        for row in combined_rows:
            key = (
                str(row.get("category") or ""),
                str(row.get("statType") or ""),
                str(row.get("team") or ""),
                str(row.get("player") or row.get("playerName") or row.get("name") or ""),
                str(row.get("athleteId") or row.get("athlete_id") or row.get("playerId") or ""),
                str(row.get("stat") or ""),
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_rows.append(row)

        season_status = "ok" if any(status == "ok" for status in statuses) else (statuses[0] if statuses else "unknown")
        season_reason = "; ".join([reason for reason in reasons if reason])
        year_meta = {"queried_teams": team_filters, "team_results": team_meta}
        return {
            "status": season_status,
            "reason": season_reason,
            "year_entry": {
                "year": int(season),
                "status": season_status,
                "reason": season_reason,
                "record_count": len(deduped_rows),
                "raw_record_count": raw_record_count,
                "meta": year_meta,
                "rows": deduped_rows,
                "result": {
                    "status": season_status,
                    "reason": season_reason,
                    "data": deduped_rows,
                    "meta": year_meta,
                    "citations": combined_citations,
                },
            },
            "citations": combined_citations,
            "diagnostic": {
                "year": int(season),
                "endpoint": "stats/player/season",
                "status": season_status,
                "reason": season_reason,
                "rows_pre_filter": raw_record_count,
                "rows_post_filter": len(deduped_rows),
                "queried_teams": _diagnostic_scalar_text(team_filters),
                "queried_team_count": len(team_filters),
                "params_text": _diagnostic_scalar_text(
                    {
                        "year": int(season),
                        "seasonType": "regular",
                        "teams": team_filters,
                    }
                ),
                "fallback_policy": "team_filtered_only",
                "fallback_teamless_attempted": False,
            },
        }

    def _stats_fallback(season: int, reason: str) -> dict[str, Any]:
        safe_reason = str(reason or "season stats pull failed")
        season_result = {
            "status": "skipped",
            "reason": safe_reason,
            "data": [],
            "meta": {"queried_teams": list(player_teams), "team_results": []},
            "citations": [],
        }
        return {
            "status": "skipped",
            "reason": safe_reason,
            "year_entry": {
                "year": int(season),
                "status": "skipped",
                "reason": safe_reason,
                "record_count": 0,
                "raw_record_count": 0,
                "meta": season_result.get("meta") or {},
                "rows": [],
                "result": season_result,
            },
            "citations": [],
            "diagnostic": {
                "year": int(season),
                "endpoint": "stats/player/season",
                "status": "skipped",
                "reason": safe_reason,
                "rows_pre_filter": 0,
                "rows_post_filter": 0,
                "queried_teams": _diagnostic_scalar_text(list(player_teams)),
                "queried_team_count": len(list(player_teams)),
                "params_text": _diagnostic_scalar_text(
                    {
                        "year": int(season),
                        "seasonType": "regular",
                        "teams": list(player_teams),
                    }
                ),
                "fallback_policy": "team_filtered_only",
                "fallback_teamless_attempted": False,
            },
        }

    stats_batch_results = _run_batched_tasks(
        items=seasons,
        task_fn=_stats_task,
        item_key_fn=lambda season: f"stats_{int(season)}",
        fallback_fn=_stats_fallback,
        checkpoint_scope=f"transfer_cfbd_stats_{athlete_id_text or player_name_text}_{ref_year}",
    )

    for row in stats_batch_results:
        payload = dict(row.get("value") or {})
        stats_by_year.append(dict(payload.get("year_entry") or {}))
        stats_citations.extend(list(payload.get("citations") or []))
        stats_diagnostics.append(dict(payload.get("diagnostic") or {}))
    _emit_progress(progress_callback, "cfbd_stats", "completed")

    usage_for_year = next((entry for entry in usage_by_year if int(entry.get("year") or 0) == ref_year), None)
    usage_for_year_result = dict(usage_for_year.get("result") or {}) if usage_for_year else {
        "status": "skipped",
        "reason": f"No usage result for {ref_year}",
        "data": [],
        "citations": [],
    }
    stats_for_year = next((entry for entry in stats_by_year if int(entry.get("year") or 0) == ref_year), None)
    stats_for_year_result = dict(stats_for_year.get("result") or {}) if stats_for_year else {
        "status": "skipped",
        "reason": f"No season stats result for {ref_year}",
        "data": [],
        "citations": [],
    }

    usage_table_compact = _compact_usage_table(usage_by_year)
    usage_yoy_compact = _usage_yoy_deltas(usage_table_compact)
    season_stats_table_compact = _compact_season_stats_table(stats_by_year)

    return {
        "status": "ok",
        "reason": "cfbd context complete",
        "year": ref_year,
        "career_start_year": career_start_year,
        "career_end_year": career_end_year,
        "cfbd_usage_for_year": usage_for_year_result,
        "cfbd_stats_for_year": stats_for_year_result,
        "cfbd_usage_career": usage_by_year,
        "cfbd_stats_career": stats_by_year,
        "usage_table_compact": usage_table_compact,
        "usage_yoy_compact": usage_yoy_compact,
        "season_stats_table_compact": season_stats_table_compact,
        "pull_config": {
            "player_name": player_name_text,
            "cfbd_athlete_id": athlete_id_text,
            "position": position_text,
            "teams": player_teams,
            "year": ref_year,
            "career_start_year": career_start_year,
            "career_end_year": career_end_year,
            "exclude_garbage_time": bool(exclude_garbage_time) if exclude_garbage_time is not None else None,
        },
        "pull_diagnostics": usage_diagnostics + stats_diagnostics,
        "citations": usage_citations + stats_citations,
    }


def _build_transfer_synthesis_prompt(
    player_name: str,
    target_team: str,
    player_row: dict[str, Any],
    cfbd_usage_2025: dict[str, Any],
    cfbd_stats_2025: dict[str, Any],
    cfbd_usage_career: list[dict[str, Any]],
    cfbd_stats_career: list[dict[str, Any]],
    usage_table_compact: list[dict[str, Any]],
    usage_yoy_compact: list[dict[str, Any]],
    season_stats_table_compact: list[dict[str, Any]],
    career_context: dict[str, Any],
    player_news_summary: str,
    team_news_summary: str,
    exclude_garbage_time: bool,
    branch_status: dict[str, Any] | None = None,
    follow_up_question: str | None = None,
) -> str:
    follow_up = str(follow_up_question or "").strip()
    task_line = (
        f"Follow-up user question: {follow_up}\n"
        if follow_up
        else "Create a transfer-impact scouting report for this player and team fit scenario.\n"
    )

    branch_status_block = ""
    if isinstance(branch_status, dict) and branch_status:
        branch_lines = ["Branch status summary:"]
        cfbd_branch = dict(branch_status.get("cfbd_context") or {})
        player_branch = dict(branch_status.get("player_news_search") or {})
        team_branch = dict(branch_status.get("team_news_search") or {})
        summary_branch = dict(branch_status.get("summarization") or {})
        branch_lines.append(
            f"- CFBD context: {cfbd_branch.get('status', 'unknown')}"
            f" | reason: {str(cfbd_branch.get('reason') or '')}"
        )
        branch_lines.append(
            f"- Player news search: {player_branch.get('status', 'unknown')}"
            f" | reason: {str(player_branch.get('reason') or '')}"
        )
        branch_lines.append(
            f"- Team news search: {team_branch.get('status', 'unknown')}"
            f" | reason: {str(team_branch.get('reason') or '')}"
        )
        branch_lines.append(
            f"- Summarization: player={summary_branch.get('player_status', 'unknown')}"
            f" | team={summary_branch.get('team_status', 'unknown')}"
        )
        branch_status_block = "\n".join(branch_lines) + "\n\n"

    return (
        "You are a senior college football transfer-portal scouting analyst.\n"
        "Use only provided context. Do not invent facts.\n"
        "If evidence is missing or stale, say so directly.\n\n"
        f"Player: {player_name}\n"
        f"Target Team: {target_team}\n"
        f"{task_line}\n"
        f"{branch_status_block}"
        "Context blocks:\n"
        f"- College Profile JSON: {player_row}\n"
        f"- CFBD 2025 Usage JSON: {cfbd_usage_2025}\n"
        f"- CFBD 2025 Season Stats JSON: {cfbd_stats_2025}\n"
        f"- CFBD Career Usage By Year JSON: {cfbd_usage_career}\n"
        f"- CFBD Career Season Stats By Year JSON: {cfbd_stats_career}\n"
        f"- Compact Usage Table JSON: {usage_table_compact}\n"
        f"- Usage YoY Delta Table JSON: {usage_yoy_compact}\n"
        f"- Compact Season Stats Table JSON: {season_stats_table_compact}\n"
        f"- Career Context JSON: {career_context}\n"
        f"- Exclude Garbage Time (CFBD pulls): {bool(exclude_garbage_time)}\n"
        f"- Player News Summary: {player_news_summary}\n"
        f"- Team News Summary: {team_news_summary}\n\n"
        "Critical analysis requirements:\n"
        "- Prioritize Compact Usage Table, Usage YoY Delta Table, and Compact Season Stats Table over narrative news claims.\n"
        "- Garbage-time plays were excluded from CFBD usage pulls by default; account for this when interpreting usage rates.\n"
        "- If any branch was skipped, failed, or timed out, state that explicitly and reduce certainty accordingly.\n"
        "- Evaluate year-to-year usage-rate changes and role volatility as transfer signals.\n"
        "- Explain key drivers and blockers using only provided evidence.\n\n"
        "Output sections in order:\n"
        "1) Player Snapshot\n"
        "2) 2025 Usage and Production\n"
        "3) Career Arc and Transfer Context\n"
        "4) Target Team Fit and Immediate Impact\n"
        "5) Transfer Conceivability Analysis\n"
        "   - Include line exactly: Likelihood Rating (out of 100): <integer 0-100>\n"
        "   - Include line exactly: Rating Confidence: <Low|Medium|High>\n"
        "   - Do NOT include rating tiers or slash-style formats (example forbidden: 15/100).\n"
        "   - Include top 3 evidence drivers and top blockers.\n\n"
        "Style constraints:\n"
        "- Keep output concise, evidence-grounded, and decision-oriented.\n"
        "- Avoid boilerplate and avoid repeating the same fact across sections."
    )


def orchestrate_transfer_report(
    college_player_id: str,
    cfbd_athlete_id: str,
    target_team: str,
    position: str,
    year: int = 2025,
    exclude_garbage_time: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    pipeline_started = time.perf_counter()
    branch_started = time.perf_counter()
    branch_latency_ms: dict[str, int] = {}
    LOGGER.info(
        "transfer_report_start college_player_id=%s cfbd_athlete_id=%s target_team=%s position=%s year=%s exclude_garbage_time=%s",
        str(college_player_id or "").strip(),
        str(cfbd_athlete_id or "").strip(),
        str(target_team or "").strip(),
        str(position or "").strip(),
        int(year),
        bool(exclude_garbage_time),
    )
    _emit_progress(progress_callback, "transfer_pipeline", "started")
    _emit_progress(progress_callback, "profile_lookup", "running")
    profile_result = fetch_college_player_bundle(
        college_player_id=str(college_player_id or "").strip() or None,
        cfbd_athlete_id=str(cfbd_athlete_id or "").strip() or None,
    )
    branch_latency_ms["profile_lookup"] = int((time.perf_counter() - branch_started) * 1000)
    _emit_progress(progress_callback, "profile_lookup", "completed")

    profile_data = dict(profile_result.get("data") or {})
    player_row = dict(profile_data.get("college_player") or {})
    resolved_athlete_id = str(profile_data.get("cfbd_athlete_id") or cfbd_athlete_id or "").strip()
    player_name = _transfer_player_name(player_row) or "Unknown Player"
    team_text = str(target_team or "").strip()
    position_text = str(position or player_row.get("position") or "").strip()
    _emit_progress(progress_callback, "transfer_pipeline", "running")
    _emit_progress(progress_callback, "parallel_fetch", "running")
    LOGGER.info("transfer_report_stage=parallel_fetch starting")
    branch_started = time.perf_counter()
    cfbd_cache_key = _transfer_cfbd_context_cache_key(
        college_player_id=str(college_player_id or "").strip(),
        cfbd_athlete_id=str(cfbd_athlete_id or "").strip(),
        year=int(year),
        exclude_garbage_time=bool(exclude_garbage_time),
    )

    def _empty_cfbd_context_payload(reason: str) -> dict[str, Any]:
        teams = _split_team_tokens(player_row.get("teams"))
        start_year = _safe_int(player_row.get("first_season")) or int(year)
        end_year = _safe_int(player_row.get("last_season")) or int(year)
        return {
            "status": "skipped",
            "reason": str(reason or "cfbd context unavailable"),
            "year": int(year),
            "career_start_year": start_year,
            "career_end_year": end_year,
            "cfbd_usage_for_year": {
                "status": "skipped",
                "reason": str(reason or "cfbd context unavailable"),
                "data": [],
                "meta": {},
                "citations": [],
            },
            "cfbd_stats_for_year": {
                "status": "skipped",
                "reason": str(reason or "cfbd context unavailable"),
                "data": [],
                "meta": {},
                "citations": [],
            },
            "cfbd_usage_career": [],
            "cfbd_stats_career": [],
            "usage_table_compact": [],
            "usage_yoy_compact": [],
            "season_stats_table_compact": [],
            "pull_config": {
                "player_name": player_name,
                "cfbd_athlete_id": resolved_athlete_id,
                "position": position_text,
                "teams": teams,
                "year": int(year),
                "career_start_year": start_year,
                "career_end_year": end_year,
                "exclude_garbage_time": bool(exclude_garbage_time),
            },
            "pull_diagnostics": [
                {
                    "year": int(year),
                    "endpoint": "cfbd_context",
                    "status": "skipped",
                    "reason": str(reason or "cfbd context unavailable"),
                    "rows_pre_filter": 0,
                    "rows_post_filter": 0,
                    "queried_teams": _diagnostic_scalar_text(teams),
                    "queried_team_count": len(teams) if isinstance(teams, list) else (1 if str(teams or "").strip() else 0),
                    "params_text": "",
                    "fallback_policy": "cfbd_context_unavailable",
                    "fallback_teamless_attempted": False,
                }
            ],
            "citations": [],
        }

    def _cfbd_context_task() -> dict[str, Any]:
        return orchestrate_transfer_cfbd_context(
            player_name=player_name,
            cfbd_athlete_id=resolved_athlete_id,
            position=position_text,
            teams=player_row.get("teams"),
            year=int(year),
            first_season=_safe_int(player_row.get("first_season")),
            last_season=_safe_int(player_row.get("last_season")),
            exclude_garbage_time=bool(exclude_garbage_time),
            # Worker thread: avoid Streamlit UI callbacks from background thread.
            progress_callback=None,
        )

    def _player_web_task() -> dict[str, Any]:
        query = (
            f"{player_name} transfer portal news college football recent {year}"
        )
        return search_web_query_tool(query=query, max_results=8, timelimit="y")

    def _team_web_task() -> dict[str, Any]:
        query = (
            f"{team_text} college football transfer portal roster needs depth chart coaching staff changes recent {year}"
        )
        return search_web_query_tool(query=query, max_results=8, timelimit="y")

    def _result_or_timeout(
        future: Any,
        label: str,
        timeout_seconds: int = 45,
        fallback_payload_factory: Callable[[str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            result = future.result(timeout=timeout_seconds)
            _emit_progress(progress_callback, label, "completed")
            return dict(result or {})
        except TimeoutError:
            try:
                future.cancel()
            except Exception:
                pass
            _emit_progress(progress_callback, label, "completed")
            reason = f"{label} timed out after {timeout_seconds}s"
            if fallback_payload_factory is not None:
                return fallback_payload_factory(reason)
            return {
                "status": "skipped",
                "reason": reason,
                "data": [],
                "citations": [],
            }
        except Exception as exc:
            _emit_progress(progress_callback, label, "completed")
            reason_text = str(exc).strip() or repr(exc)
            reason = f"{label} failed: {reason_text}"
            if fallback_payload_factory is not None:
                return fallback_payload_factory(reason)
            return {
                "status": "skipped",
                "reason": reason,
                "data": [],
                "citations": [],
            }

    _emit_progress(progress_callback, "cfbd_context", "running")
    _emit_progress(progress_callback, "player_news_search", "running")
    _emit_progress(progress_callback, "team_news_search", "running")

    executor = ThreadPoolExecutor(max_workers=3)
    try:
        cached_cfbd_context = _transfer_cfbd_context_cache_get(cfbd_cache_key)
        cfbd_context_future = None if cached_cfbd_context is not None else executor.submit(_cfbd_context_task)
        player_web_future = executor.submit(_player_web_task)
        team_web_future = executor.submit(_team_web_task)

        if cached_cfbd_context is not None:
            cfbd_context = dict(cached_cfbd_context)
            _emit_progress(progress_callback, "cfbd_context", "completed")
        else:
            LOGGER.info("transfer_report_stage=cfbd_context waiting timeout_seconds=%s", 90)
            cfbd_context = _result_or_timeout(
                cfbd_context_future,
                "cfbd_context",
                timeout_seconds=90,
                fallback_payload_factory=_empty_cfbd_context_payload,
            )
        LOGGER.info("transfer_report_stage=player_news_search waiting timeout_seconds=%s", 30)
        player_web = _result_or_timeout(player_web_future, "player_news_search", timeout_seconds=30)
        LOGGER.info("transfer_report_stage=team_news_search waiting timeout_seconds=%s", 30)
        team_web = _result_or_timeout(team_web_future, "team_news_search", timeout_seconds=30)
    finally:
        # Do not block request completion on stuck network worker threads.
        executor.shutdown(wait=False, cancel_futures=True)

    LOGGER.info(
        "transfer_report_stage=parallel_fetch done status cfbd=%s player_news=%s team_news=%s",
        str(cfbd_context.get("status") or "unknown"),
        str(player_web.get("status") or "unknown"),
        str(team_web.get("status") or "unknown"),
    )

    if cfbd_context.get("status") == "ok":
        _transfer_cfbd_context_cache_set(cfbd_cache_key, cfbd_context)

    branch_latency_ms["parallel_fetch"] = int((time.perf_counter() - branch_started) * 1000)
    _emit_progress(progress_callback, "parallel_fetch", "completed")

    _emit_progress(progress_callback, "summarization", "running")
    LOGGER.info("transfer_report_stage=summarization starting")
    branch_started = time.perf_counter()

    cfbd_usage_career = list(cfbd_context.get("cfbd_usage_career") or [])
    cfbd_stats_career = list(cfbd_context.get("cfbd_stats_career") or [])
    cfbd_usage_2025 = dict(cfbd_context.get("cfbd_usage_for_year") or {})
    cfbd_stats_2025 = dict(cfbd_context.get("cfbd_stats_for_year") or {})
    usage_table_compact = list(cfbd_context.get("usage_table_compact") or [])
    usage_yoy_compact = list(cfbd_context.get("usage_yoy_compact") or [])
    season_stats_table_compact = list(cfbd_context.get("season_stats_table_compact") or [])
    pull_diagnostics = list(cfbd_context.get("pull_diagnostics") or [])
    pull_config = dict(cfbd_context.get("pull_config") or {})

    summary_timeout_seconds = max(10, int(CONFIG.get("SUMMARY_TIMEOUT_SECONDS", 45) or 45))

    def _summary_result_or_timeout(future: Any, label: str) -> dict[str, Any]:
        try:
            return dict(future.result(timeout=summary_timeout_seconds) or {})
        except TimeoutError:
            try:
                future.cancel()
            except Exception:
                pass
            LOGGER.warning("transfer_summarization_timeout label=%s timeout_seconds=%s", label, summary_timeout_seconds)
            return {
                "status": "skipped",
                "reason": f"{label} timed out after {summary_timeout_seconds}s",
                "data": "Summary unavailable: summarization timed out.",
                "citations": [],
            }
        except Exception as exc:
            reason_text = str(exc).strip() or repr(exc)
            LOGGER.warning("transfer_summarization_failed label=%s reason=%s", label, reason_text)
            return {
                "status": "skipped",
                "reason": f"{label} failed: {reason_text}",
                "data": "Summary unavailable: summarization failed.",
                "citations": [],
            }

    summary_executor = ThreadPoolExecutor(max_workers=2)
    try:
        LOGGER.info("transfer_report_stage=player_news_summary submit")
        player_summary_future = summary_executor.submit(
            summarize_payload_tool,
            summary_prompt=(
                "You are a secure summarization node. Output ONLY plain markdown bullet points (no HTML, no JSON, no links). "
                "Extract and summarize only high-signal transfer-portal player news from the provided snippets. "
                "Focus strictly on transfer intent, eligibility remaining, role expectations, and timeline. "
                "Use strictly the supplied snippets. "
            ),
            payload=player_web.get("data") or [],
            role="transfer_player",
            entity_kind="player",
            target_name=player_name,
            target_team=team_text,
        )
        LOGGER.info("transfer_report_stage=team_news_summary submit")
        team_summary_future = summary_executor.submit(
            summarize_payload_tool,
            summary_prompt=(
                "You are a secure summarization node. Output ONLY plain markdown bullet points (no HTML, no JSON, no links). "
                "Extract and summarize only high-signal team transfer-portal context from the provided snippets. "
                "Focus strictly on roster needs, depth chart competition, current coaching staff, and recent staff changes. "
                "Prioritize the most recent evidence available, explicitly note if the source material appears outdated, and use strictly the supplied snippets. "
            ),
            payload=team_web.get("data") or [],
            role="transfer_team",
            entity_kind="team",
            target_name=player_name,
            target_team=team_text,
        )
        LOGGER.info("transfer_report_stage=player_news_summary waiting timeout_seconds=%s", summary_timeout_seconds)
        player_news_summary_result = _summary_result_or_timeout(player_summary_future, "player_news_summary")
        LOGGER.info("transfer_report_stage=team_news_summary waiting timeout_seconds=%s", summary_timeout_seconds)
        team_news_summary_result = _summary_result_or_timeout(team_summary_future, "team_news_summary")
    finally:
        # Avoid blocking completion when upstream model/network calls stall.
        summary_executor.shutdown(wait=False, cancel_futures=True)

    branch_latency_ms["summarization"] = int((time.perf_counter() - branch_started) * 1000)
    LOGGER.info(
        "transfer_report_stage=summarization done player_status=%s team_status=%s",
        str(player_news_summary_result.get("status") or "unknown"),
        str(team_news_summary_result.get("status") or "unknown"),
    )
    _emit_progress(progress_callback, "summarization", "completed")

    branch_status = {
        "cfbd_context": {
            "status": str(cfbd_context.get("status") or "unknown"),
            "reason": str(cfbd_context.get("reason") or ""),
            "usage_year_rows": len(list(cfbd_usage_2025.get("data") or [])),
            "stats_year_rows": len(list(cfbd_stats_2025.get("data") or [])),
            "diagnostic_rows": len(pull_diagnostics),
        },
        "player_news_search": {
            "status": str(player_web.get("status") or "unknown"),
            "reason": str(player_web.get("reason") or ""),
            "row_count": len(list(player_web.get("data") or [])),
        },
        "team_news_search": {
            "status": str(team_web.get("status") or "unknown"),
            "reason": str(team_web.get("reason") or ""),
            "row_count": len(list(team_web.get("data") or [])),
        },
        "summarization": {
            "player_status": str(player_news_summary_result.get("status") or "unknown"),
            "player_reason": str(player_news_summary_result.get("reason") or ""),
            "team_status": str(team_news_summary_result.get("status") or "unknown"),
            "team_reason": str(team_news_summary_result.get("reason") or ""),
        },
    }

    player_news_summary = str(player_news_summary_result.get("data") or "").strip()
    team_news_summary = str(team_news_summary_result.get("data") or "").strip()

    career_context = {
        "first_season": player_row.get("first_season"),
        "last_season": player_row.get("last_season"),
        "seasons_active": player_row.get("seasons_active"),
        "season_span": player_row.get("season_span"),
        "teams": player_row.get("teams"),
        "conference": player_row.get("conference"),
    }

    synthesis_prompt = _build_transfer_synthesis_prompt(
        player_name=player_name,
        target_team=team_text,
        player_row=player_row,
        cfbd_usage_2025=cfbd_usage_2025,
        cfbd_stats_2025=cfbd_stats_2025,
        cfbd_usage_career=cfbd_usage_career,
        cfbd_stats_career=cfbd_stats_career,
        usage_table_compact=usage_table_compact,
        usage_yoy_compact=usage_yoy_compact,
        season_stats_table_compact=season_stats_table_compact,
        career_context=career_context,
        player_news_summary=player_news_summary,
        team_news_summary=team_news_summary,
        exclude_garbage_time=bool(pull_config.get("exclude_garbage_time", exclude_garbage_time)),
        branch_status=branch_status,
    )
    _emit_progress(progress_callback, "final_synthesis", "running")
    LOGGER.info("transfer_report_stage=final_synthesis starting")
    branch_started = time.perf_counter()
    synthesis_result = final_synthesis_tool(synthesis_prompt)
    branch_latency_ms["final_synthesis"] = int((time.perf_counter() - branch_started) * 1000)
    LOGGER.info(
        "transfer_report_stage=final_synthesis done status=%s",
        str((synthesis_result or {}).get("status") or "unknown"),
    )
    _emit_progress(progress_callback, "final_synthesis", "completed")

    citations: list[dict[str, str]] = []
    for source in [
        profile_result.get("citations") or [],
        cfbd_context.get("citations") or [],
        player_web.get("citations") or [],
        team_web.get("citations") or [],
        player_news_summary_result.get("citations") or [],
        team_news_summary_result.get("citations") or [],
        synthesis_result.get("citations") or [],
    ]:
        citations.extend(list(source))

    model_telemetry_rows = [
        _extract_tool_telemetry(player_news_summary_result),
        _extract_tool_telemetry(team_news_summary_result),
        _extract_tool_telemetry(synthesis_result),
    ]
    telemetry_rollup = _rollup_telemetry_rows(model_telemetry_rows)
    orchestration_telemetry = {
        "pipeline_latency_ms": int((time.perf_counter() - pipeline_started) * 1000),
        "branch_latency_ms": branch_latency_ms,
        "model_telemetry": [row for row in model_telemetry_rows if row],
        "model_rollup": telemetry_rollup,
    }

    _emit_progress(progress_callback, "transfer_pipeline", "completed")

    return {
        "status": "ok",
        "player_name": player_name,
        "target_team": team_text,
        "position": position_text,
        "year": int(year),
        "college_player_id": str(profile_data.get("college_player_id") or college_player_id or ""),
        "cfbd_athlete_id": resolved_athlete_id,
        "college_player": player_row,
        "cfbd_usage_2025": cfbd_usage_2025,
        "cfbd_stats_2025": cfbd_stats_2025,
        "cfbd_usage_career": cfbd_usage_career,
        "cfbd_stats_career": cfbd_stats_career,
        "usage_table_compact": usage_table_compact,
        "usage_yoy_compact": usage_yoy_compact,
        "season_stats_table_compact": season_stats_table_compact,
        "pull_diagnostics": pull_diagnostics,
        "pull_config": pull_config,
        "career_context": career_context,
        "branch_status": branch_status,
        "exclude_garbage_time": bool(pull_config.get("exclude_garbage_time", exclude_garbage_time)),
        "player_news_summary": player_news_summary,
        "team_news_summary": team_news_summary,
        "final_report": str(synthesis_result.get("data") or "").strip(),
        "telemetry": orchestration_telemetry,
        "trace_log": [
            {"node": "profile_lookup", "status": "completed"},
            {"node": "parallel_fetch", "status": "completed"},
            {"node": "summarization", "status": "completed"},
            {"node": "final_synthesis", "status": "completed"},
            {"node": "transfer_pipeline", "status": "completed"},
            {
                "node": "telemetry",
                "status": "completed",
                "pipeline_latency_ms": orchestration_telemetry.get("pipeline_latency_ms"),
                "estimated_cost_usd": telemetry_rollup.get("estimated_cost_usd"),
            },
        ],
        "citations": citations,
        "transfer_report_context": {
            "player_name": player_name,
            "target_team": team_text,
            "position": position_text,
            "year": int(year),
            "college_player": player_row,
            "cfbd_usage_2025": cfbd_usage_2025,
            "cfbd_stats_2025": cfbd_stats_2025,
            "cfbd_usage_career": cfbd_usage_career,
            "cfbd_stats_career": cfbd_stats_career,
            "usage_table_compact": usage_table_compact,
            "usage_yoy_compact": usage_yoy_compact,
            "season_stats_table_compact": season_stats_table_compact,
            "pull_diagnostics": pull_diagnostics,
            "pull_config": pull_config,
            "career_context": career_context,
            "branch_status": branch_status,
            "exclude_garbage_time": bool(pull_config.get("exclude_garbage_time", exclude_garbage_time)),
            "player_news_summary": player_news_summary,
            "team_news_summary": team_news_summary,
            "telemetry": orchestration_telemetry,
        },
    }


def orchestrate_transfer_chat_turn(
    user_prompt: str,
    current_state: dict[str, Any] | None,
    allow_web_refresh: bool = True,
    graph: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    graph_runner = graph or get_transfer_chat_graph()
    state = dict(current_state or {})
    context = dict(state.get("transfer_report_context") or {})
    if not context:
        return {
            "status": "error",
            "final_report": "Transfer chat is unavailable until a transfer report is generated.",
            "conversation_history": list(state.get("conversation_history") or []),
            "trace_log": list(state.get("trace_log") or []) + [
                {"node": "transfer_chat", "status": "missing_context"},
            ],
            "transfer_report_context": {},
        }

    chat_state = compact_transfer_chat_state(state)
    chat_state["mode"] = "chat"
    chat_state["user_query"] = str(user_prompt or "").strip()
    chat_state["transfer_report_context"] = context
    chat_state["allow_web_refresh"] = bool(allow_web_refresh)

    result_state = _invoke_graph(graph_runner, chat_state, progress_callback=progress_callback)
    
    if isinstance(result_state, dict):
        trace = list(result_state.get("trace_log") or [])
        trace.append({
            "node": "transfer_orchestration",
            "status": "completed",
            "mode": "chat",
            "latency_ms": int((time.perf_counter() - started) * 1000)
        })
        result_state["trace_log"] = trace[-CHAT_STATE_MAX_TRACE:]
        
        model_telemetry_rows = []
        if isinstance(result_state.get("telemetry"), dict):
            model_telemetry_rows = result_state["telemetry"].get("model_telemetry", [])
            
        telemetry_rollup = _rollup_telemetry_rows(model_telemetry_rows)
        telemetry = {
            "pipeline_latency_ms": int((time.perf_counter() - started) * 1000),
            "model_telemetry": [row for row in model_telemetry_rows if row],
            "model_rollup": telemetry_rollup,
        }
        
        history = list(result_state.get("conversation_history") or [])
        history.append({"role": "user", "content": chat_state["user_query"]})
        answer_text = str(result_state.get("final_report") or "").strip()
        history.append({"role": "assistant", "content": answer_text})
        
        return {
            "status": "ok",
            "final_report": answer_text,
            "conversation_history": history[-(CHAT_STATE_MAX_TURNS * 2):],
            "trace_log": result_state["trace_log"],
            "telemetry": telemetry,
            "transfer_report_context": context,
        }
    return {"status": "error"}

def _merge_text_blocks(base_text: str, extra_text: str) -> str:
    base = str(base_text or "").strip()
    extra = str(extra_text or "").strip()
    if base and extra:
        return f"{base}\n\nRecent Follow-up Updates:\n{extra}"
    return base or extra

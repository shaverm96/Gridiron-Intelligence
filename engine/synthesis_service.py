from __future__ import annotations

import json
import re
import time
from datetime import date
from typing import Any

from .config import CONFIG


MODEL_ALIAS_MAP = {
    "gemini-3.0-flash": "gemini-3-flash-preview",
    "gemini-3.1-flash-lite-preview": "gemini-3.1-flash-lite-preview",
}
MODEL_TOKEN_COSTS_PER_1M = {
    "gemini-3.1-flash-lite-preview": {"input": 0.25, "output": 1.50},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
}


def _normalize_model_name(model_name: str) -> str:
    value = str(model_name or "").strip()
    return MODEL_ALIAS_MAP.get(value, value)


def _extract_token_usage(response: Any) -> dict[str, int | None]:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    usage_sources: list[dict[str, Any]] = []
    usage_direct = getattr(response, "usage_metadata", None)
    if isinstance(usage_direct, dict):
        usage_sources.append(usage_direct)

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        usage_meta = response_metadata.get("usage_metadata")
        if isinstance(usage_meta, dict):
            usage_sources.append(usage_meta)
        usage_block = response_metadata.get("usage")
        if isinstance(usage_block, dict):
            usage_sources.append(usage_block)

    for usage in usage_sources:
        input_tokens = input_tokens or usage.get("input_tokens") or usage.get("prompt_token_count")
        output_tokens = output_tokens or usage.get("output_tokens") or usage.get("candidates_token_count")
        total_tokens = total_tokens or usage.get("total_tokens") or usage.get("total_token_count")

    input_tokens = int(input_tokens) if isinstance(input_tokens, (int, float)) else None
    output_tokens = int(output_tokens) if isinstance(output_tokens, (int, float)) else None
    if isinstance(total_tokens, (int, float)):
        total_tokens = int(total_tokens)
    elif input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens + output_tokens)
    else:
        total_tokens = None

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _estimate_model_cost_usd(model_name: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    rates = MODEL_TOKEN_COSTS_PER_1M.get(_normalize_model_name(model_name), {})
    if not isinstance(rates, dict):
        return None

    input_rate = rates.get("input")
    output_rate = rates.get("output")
    if not isinstance(input_rate, (int, float)) or not isinstance(output_rate, (int, float)):
        return None

    if input_tokens is None and output_tokens is None:
        return None

    in_cost = (float(input_tokens or 0) / 1_000_000.0) * float(input_rate)
    out_cost = (float(output_tokens or 0) / 1_000_000.0) * float(output_rate)
    return round(in_cost + out_cost, 8)


def _truncate_text(value: str, max_chars: int) -> str:
    text = str(value or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n\n[truncated {len(text) - max_chars} chars]"


def _json_block(value: Any, max_chars: int) -> str:
    return _truncate_text(json.dumps(value, indent=2, default=str), max_chars=max_chars)


def _with_current_date_context(prompt_text: str) -> str:
    today_iso = date.today().isoformat()
    date_context = (
        "Date Context:\n"
        f"- Current date: {today_iso}\n"
        "- Treat this as today's date when reasoning about recency and up-to-date information.\n"
        "- If recency is uncertain, state the uncertainty explicitly.\n\n"
    )
    return f"{date_context}{str(prompt_text or '').strip()}"


def _friendly_threshold_block(pred_thr_row: dict[str, Any]) -> str:
    if not isinstance(pred_thr_row, dict) or not pred_thr_row:
        return "No threshold probabilities available."

    lines_map: dict[tuple[str, str], tuple[float, str, int]] = {}
    for key, value in pred_thr_row.items():
        key_text = str(key or "").strip()
        key_lower = key_text.lower()
        if key_lower in {"recruit_id", "id", "year", "class_year", "created_at", "updated_at"}:
            continue

        if "odds" in key_lower:
            continue

        threshold_match = re.search(r"(^|[_\-])(ge|gt|le|lt|gte|lte)(\d{1,3}(?:\.\d+)?)", key_lower)
        if not threshold_match:
            continue

        looks_probability = (
            "prob" in key_lower
            or "probability" in key_lower
        )
        if not looks_probability:
            continue

        try:
            numeric = float(value)
        except Exception:
            continue

        if numeric < 0:
            continue
        pct = numeric * 100.0 if numeric <= 1 else numeric
        if pct > 100:
            continue

        op = str(threshold_match.group(2) or "ge")
        threshold_num = str(threshold_match.group(3) or "")
        if "ge" in op or "gte" in op or ">=" in key_lower:
            label = f"Chance to reach >= {threshold_num}" if threshold_num else "Chance to reach upper threshold"
        elif "gt" in op or ">" in key_lower:
            label = f"Chance to exceed > {threshold_num}" if threshold_num else "Chance to exceed upper threshold"
        elif "le" in op or "lte" in op or "<=" in key_lower:
            label = f"Chance to stay <= {threshold_num}" if threshold_num else "Chance to stay below threshold"
        elif "lt" in op or "<" in key_lower:
            label = f"Chance to stay < {threshold_num}" if threshold_num else "Chance to stay below threshold"
        elif threshold_num:
            label = f"Chance to reach >= {threshold_num}"
        else:
            label = "Threshold probability"

        rank_num = float(threshold_num) if threshold_num else -1.0
        canonical_key = (op, threshold_num)
        priority = 2 if "prob" in key_lower or "probability" in key_lower else 1
        existing = lines_map.get(canonical_key)
        if existing is None or priority > existing[2]:
            lines_map[canonical_key] = (rank_num, f"- {label}: {pct:.1f}%", priority)

    if not lines_map:
        return "No threshold probabilities available."

    lines = [(rank, line) for rank, line, _ in lines_map.values()]
    lines.sort(key=lambda item: item[0], reverse=True)
    return "\n".join([line for _, line in lines[:6]])


def build_final_prompt_data(
    year: int,
    target_team: str,
    persona: str,
    player_row: dict[str, Any],
    scouting_clean: dict[str, Any],
    hs_athletic_background: str,
    pred_score_row: dict[str, Any],
    pred_thr_row: dict[str, Any],
    web_summary: str,
    vector_result: dict[str, Any],
    historical_comparables_md: str,
    tier_definitions_markdown: Any,
) -> str:
    vector_insights = vector_result.get("insights", []) if isinstance(vector_result, dict) else []
    vector_block = "\n".join([f"- {item}" for item in vector_insights]) if vector_insights else "No vector insights returned."
    threshold_block = _friendly_threshold_block(pred_thr_row)

    tier_defs = tier_definitions_markdown() if callable(tier_definitions_markdown) else str(tier_definitions_markdown)

    payload_cap = int(CONFIG.get("PROMPT_PAYLOAD_MAX_CHARS", 12000))
    prompt_cap = int(CONFIG.get("FINAL_PROMPT_MAX_CHARS", 20000))
    json_cap = max(1500, payload_cap // 3)

    prompt = (
        "You are a senior college football recruiting scout.\n"
        f"Persona: {persona}\n"
        "Use only provided context. If data is missing, say so clearly.\n\n"
        "Do not mention missing technical skill grades or any missing skill_* fields. "
        "Omit that uncertainty entirely from the narrative.\n\n"
        f"Year: {year}\n"
        f"Target Team: {target_team}\n\n"
        "Player Profile JSON:\n"
        f"{_json_block(player_row, max_chars=json_cap)}\n\n"
        "Filtered Scouting JSON:\n"
        f"{_json_block(scouting_clean, max_chars=json_cap)}\n\n"
        f"HS Athletic Background:\n{_truncate_text(hs_athletic_background or 'N/A', max_chars=2000)}\n\n"
        "Prediction Score Row JSON:\n"
        f"{_json_block(pred_score_row, max_chars=json_cap)}\n\n"
        f"Prediction Threshold Probabilities (user-friendly):\n{_truncate_text(threshold_block, max_chars=1800)}\n\n"
        f"Web Intelligence Summary:\n{_truncate_text(web_summary, max_chars=3000)}\n\n"
        f"Vector Insights:\n{_truncate_text(vector_block, max_chars=2500)}\n\n"
        f"Historical Comparables:\n{_truncate_text(historical_comparables_md, max_chars=2500)}\n\n"
        f"Tier Definitions:\n{_truncate_text(tier_defs, max_chars=2500)}\n\n"
        "When discussing threshold probabilities, never use raw internal key names like ge80, gt75, or p_ge80. "
        "Translate them to user-friendly language such as 'chance to reach >=80' or plain English equivalents.\n\n"
        "Output sections in order:\n"
        "1) Player Snapshot\n"
        "2) Trait Evaluation\n"
        "3) Scheme and Team Fit\n"
        "4) Development Risks\n"
        "5) Final Recommendation and Confidence\n"
    )
    return _truncate_text(prompt, max_chars=prompt_cap)


def run_final_synthesis_data(
    prompt: str,
    final_model: str,
    get_llm: Any,
    llm_response_to_text: Any,
) -> str:
    result = run_final_synthesis_with_telemetry_data(
        prompt=prompt,
        final_model=final_model,
        get_llm=get_llm,
        llm_response_to_text=llm_response_to_text,
    )
    return str(result.get("data") or "").strip()


def run_final_synthesis_with_telemetry_data(
    prompt: str,
    final_model: str,
    get_llm: Any,
    llm_response_to_text: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved_model = _normalize_model_name(final_model)
    llm = get_llm(final_model, temperature=0.25, max_output_tokens=2200)
    if llm is None:
        return {
            "status": "skipped",
            "reason": "Gemini model is not configured.",
            "data": "Final synthesis skipped: Gemini model is not configured.",
            "telemetry": {
                "tool": "run_final_synthesis_with_telemetry_data",
                "model": resolved_model,
                "status": "skipped",
                "reason": "Gemini model is not configured.",
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
        }

    try:
        response = llm.invoke(_with_current_date_context(prompt))
        text = llm_response_to_text(response)
        usage = _extract_token_usage(response)
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = int(input_tokens + output_tokens)
        estimated_cost = _estimate_model_cost_usd(resolved_model, input_tokens, output_tokens)
        clean_text = str(text).strip() if text else "Final synthesis returned empty output."
        return {
            "status": "ok",
            "reason": "",
            "data": clean_text,
            "telemetry": {
                "tool": "run_final_synthesis_with_telemetry_data",
                "model": resolved_model,
                "status": "ok",
                "reason": "",
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "total_tokens": int(total_tokens or 0),
                "estimated_cost_usd": float(estimated_cost or 0.0),
            },
        }
    except Exception as exc:
        reason = str(exc).strip() or repr(exc)
        return {
            "status": "error",
            "reason": reason,
            "data": f"Final synthesis failed: {reason}",
            "telemetry": {
                "tool": "run_final_synthesis_with_telemetry_data",
                "model": resolved_model,
                "status": "error",
                "reason": reason,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
            },
        }

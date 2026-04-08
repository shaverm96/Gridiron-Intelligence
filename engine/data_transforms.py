from __future__ import annotations

import json
import re
from typing import Any, Callable

import pandas as pd


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() in {"", "nan", "None", "none"}:
        return True
    return False


def _default_parse_jsonish(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if _is_blank(value):
        return {}
    text = str(value).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def merge_scouting_sources_data(
    scouting_row: dict[str, Any],
    parse_jsonish: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    parser = parse_jsonish or _default_parse_jsonish
    merged: dict[str, Any] = {}

    if not isinstance(scouting_row, dict):
        return merged

    # Merge known dict-like payload fields first.
    preferred_fields = [
        "composite_scouting_report",
        "athletic_profile",
        "manual_scouting_notes",
        "scouting_json",
        "profile_json",
    ]

    for field in preferred_fields:
        if field in scouting_row:
            parsed = parser(scouting_row.get(field))
            if isinstance(parsed, dict):
                merged.update(parsed)

    # Backfill any scalar fields not already represented.
    for key, value in scouting_row.items():
        if key in merged or _is_blank(value):
            continue
        merged[key] = value

    return merged


def clean_scouting_profile_data(
    scouting_json: dict[str, Any],
    to_float_or_none: Callable[[Any], float | None],
) -> dict[str, Any]:
    if not isinstance(scouting_json, dict):
        return {}

    cleaned: dict[str, Any] = {}
    numeric_hints = {
        "rating",
        "speed",
        "burst",
        "agility",
        "size",
        "arm_length",
        "wingspan",
        "weight",
        "height",
        "score",
        "pred_score",
    }

    for key, value in scouting_json.items():
        key_text = str(key).strip()
        key_lower = key_text.lower()

        # Ignore all skill-grade style fields (skill_*, skill *, skills_*); these are not populated for future recruits.
        if key_lower.startswith("skill"):
            continue

        if _is_blank(value):
            continue

        if any(h in key_lower for h in numeric_hints):
            numeric = to_float_or_none(value)
            cleaned[key_text] = value if numeric is None else numeric
        else:
            cleaned[key_text] = value

    return cleaned


def build_player_profile_view_data(
    player_row: dict[str, Any],
    first_non_null: Callable[[dict[str, Any], list[str]], Any],
) -> dict[str, Any]:
    row = player_row if isinstance(player_row, dict) else {}

    profile = {
        "recruit_id": first_non_null(row, ["recruit_id", "id"]),
        "athlete_id": first_non_null(row, ["athlete_id"]),
        "player_name": first_non_null(row, ["player_name", "name", "athlete_name"]),
        "position": first_non_null(row, ["position", "pos", "primary_position"]),
        "year": first_non_null(row, ["year", "class_year"]),
        "high_school": first_non_null(row, ["high_school", "hs_name", "school"]),
        "city": first_non_null(row, ["city", "home_city"]),
        "state": first_non_null(row, ["state", "home_state"]),
        "height_inches": first_non_null(row, ["height_inches", "height_in", "height"]),
        "weight_lbs": first_non_null(row, ["weight_lbs", "weight"]),
        "rating": first_non_null(row, ["rating", "composite_rating"]),
        "stars": first_non_null(row, ["stars"]),
        "committed_to": first_non_null(row, ["committed_to", "college", "commit_school"]),
    }

    return {k: v for k, v in profile.items() if not _is_blank(v)}


def build_score_card_html_data(
    pred_score: dict[str, Any],
    pred_threshold: dict[str, Any],
    to_float_or_none: Callable[[Any], float | None],
    score_tier: Callable[[float | None], str],
) -> str:
    ps = pred_score if isinstance(pred_score, dict) else {}
    pt = pred_threshold if isinstance(pred_threshold, dict) else {}

    def _first_numeric(keys: list[str], source: dict[str, Any]) -> float | None:
        for key in keys:
            val = to_float_or_none(source.get(key))
            if val is not None:
                return val
        return None

    def _first_numeric_by_name_hint(source: dict[str, Any], include_tokens: list[str], exclude_tokens: list[str]) -> float | None:
        for key, value in source.items():
            key_lower = str(key).lower()
            if any(token in key_lower for token in exclude_tokens):
                continue
            if any(token in key_lower for token in include_tokens):
                parsed = to_float_or_none(value)
                if parsed is not None:
                    return parsed
        return None

    def _percent_from_value(value: Any) -> float | None:
        parsed = to_float_or_none(value)
        if parsed is None:
            return None
        if parsed < 0:
            return None
        if parsed <= 1:
            return max(0.0, min(100.0, parsed * 100.0))
        if parsed <= 100:
            return max(0.0, min(100.0, parsed))
        return None

    def _friendly_probability_label(raw_key: str) -> str:
        key = str(raw_key or "").strip().lower()
        number_match = re.search(r"(\d{2,3}(?:\.\d+)?)", key)
        threshold_txt = number_match.group(1) if number_match else ""
        op_match = re.search(r"(^|[_\-])(ge|gt|le|lt|gte|lte)(\d{1,3}(?:\.\d+)?)", key)
        op = str(op_match.group(2)) if op_match else ""

        if op in {"ge", "gte"} or ">=" in key:
            return f"Chance to reach >= {threshold_txt}" if threshold_txt else "Chance to reach upper threshold"
        if op == "gt" or ">" in key:
            return f"Chance to exceed > {threshold_txt}" if threshold_txt else "Chance to exceed upper threshold"
        if op in {"le", "lte"} or "<=" in key:
            return f"Chance to stay <= {threshold_txt}" if threshold_txt else "Chance to stay below threshold"
        if op == "lt" or "<" in key:
            return f"Chance to stay < {threshold_txt}" if threshold_txt else "Chance to stay below threshold"
        if threshold_txt:
            return f"Chance to reach >= {threshold_txt}"
        pretty = re.sub(r"[_\-]+", " ", key).strip()
        return pretty.title() if pretty else "Threshold Probability"

    def _extract_probability_rows(source: dict[str, Any]) -> list[tuple[str, float, float]]:
        rows_map: dict[tuple[str, str], tuple[str, float, float, int]] = {}
        for key, value in source.items():
            key_text = str(key or "").strip()
            key_lower = key_text.lower()
            if key_lower in {"recruit_id", "id", "year", "class_year", "created_at", "updated_at"}:
                continue

            if "odds" in key_lower:
                continue

            threshold_match = re.search(r"(^|[_\-])(ge|gt|le|lt|gte|lte)(\d{1,3}(?:\.\d+)?)", key_lower)
            has_threshold_token = bool(threshold_match)
            if not has_threshold_token:
                continue

            looks_probability = (
                "prob" in key_lower
                or "probability" in key_lower
            )
            if not looks_probability:
                continue

            pct = _percent_from_value(value)
            if pct is None:
                continue

            op = str(threshold_match.group(2) or "ge")
            threshold_num_text = str(threshold_match.group(3) or "")
            threshold_num = to_float_or_none(threshold_num_text)
            if threshold_num is None or threshold_num < 0 or threshold_num > 100:
                continue
            rank_key = threshold_num if threshold_num is not None else -1.0
            label = _friendly_probability_label(key_text)
            canonical_key = (op, threshold_num_text)

            # Prefer explicit probability columns when multiple fields map to same threshold.
            priority = 2 if "prob" in key_lower or "probability" in key_lower else 1
            existing = rows_map.get(canonical_key)
            if existing is None or priority > existing[3]:
                rows_map[canonical_key] = (label, pct, rank_key, priority)

        rows = [(label, pct, rank) for (label, pct, rank, _) in rows_map.values()]
        rows = sorted(rows, key=lambda item: (item[2], item[0]), reverse=True)
        return rows[:5]

    score = _first_numeric(
        [
            "pred_score",
            "prediction_score",
            "overall_score",
            "model_score",
            "score",
            "probability",
            "pred_probability",
        ],
        ps,
    )
    if score is None:
        score = _first_numeric_by_name_hint(
            source=ps,
            include_tokens=["score", "prob", "prediction"],
            exclude_tokens=["id", "year", "rank", "tier", "threshold"],
        )

    low = _first_numeric(["low", "threshold_low", "floor", "min", "p25", "q1"], pt)
    high = _first_numeric(["high", "threshold_high", "ceiling", "max", "p75", "q3"], pt)

    tier = score_tier(score)

    score_pct = None
    if score is not None:
        raw_score = float(score)
        score_pct = max(0.0, min(100.0, raw_score * 100.0 if raw_score <= 1.0 else raw_score))
    score_text = "N/A" if score_pct is None else f"{score_pct:.1f}"
    threshold_text = (
        "N/A"
        if low is None and high is None
        else f"{'' if low is None else f'{low:.3f}'} - {'' if high is None else f'{high:.3f}'}"
    )
    threshold_band_html = (
        ""
        if threshold_text == "N/A"
        else f"<p style='margin:4px 0;color:#f9fafb;'><strong>Threshold Band:</strong> {threshold_text}</p>"
    )

    probability_rows = _extract_probability_rows(pt)

    score_bar_html = ""
    if score_pct is not None:
        score_bar_html = (
            "<div style='margin:8px 0 10px 0;'>"
            "<div style='display:flex;justify-content:space-between;font-size:12px;opacity:0.9;'>"
            "<span>Projected Score Level</span><span>"
            f"{score_pct:.1f}%"
            "</span></div>"
            "<div style='width:100%;height:10px;background:#374151;border-radius:999px;overflow:hidden;'>"
            f"<div style='height:10px;background:linear-gradient(90deg,#22d3ee 0%,#34d399 100%);width:{score_pct:.1f}%;'></div>"
            "</div></div>"
        )

    probability_bars_html = ""
    if probability_rows:
        bars = []
        for label, pct, _ in probability_rows:
            bars.append(
                "<div style='margin:8px 0;'>"
                "<div style='display:flex;justify-content:space-between;font-size:12px;opacity:0.95;'>"
                f"<span>{label}</span><span>{pct:.1f}%</span>"
                "</div>"
                "<div style='width:100%;height:10px;background:#374151;border-radius:999px;overflow:hidden;'>"
                f"<div style='height:10px;background:linear-gradient(90deg,#60a5fa 0%,#38bdf8 100%);width:{pct:.1f}%;'></div>"
                "</div>"
                "</div>"
            )
        probability_bars_html = (
            "<div style='margin-top:10px;'>"
            "<p style='margin:0 0 6px 0;color:#e5e7eb;'><strong>Threshold Probabilities</strong></p>"
            + "".join(bars)
            + "</div>"
        )

    return (
        "<div style='border:1px solid #1f2937;border-radius:10px;padding:14px;"
        "background:linear-gradient(135deg,#0b1220 0%,#1f2937 60%,#111827 100%);"
        "color:#f9fafb;box-shadow:0 4px 14px rgba(0,0,0,0.28);'>"
        "<h4 style='margin:0 0 8px 0;color:#ffffff;'>Projected Model Score</h4>"
        f"<p style='margin:4px 0;color:#f9fafb;'><strong>Predicted Score:</strong> {score_text}/100</p>"
        f"<p style='margin:4px 0;color:#f9fafb;'><strong>Tier:</strong> {tier}</p>"
        f"{threshold_band_html}"
        f"{score_bar_html}"
        f"{probability_bars_html}"
        "</div>"
    )

from __future__ import annotations

import json
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
        if _is_blank(value):
            continue

        key_lower = str(key).lower()
        if any(h in key_lower for h in numeric_hints):
            numeric = to_float_or_none(value)
            cleaned[key] = value if numeric is None else numeric
        else:
            cleaned[key] = value

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

    score = to_float_or_none(
        ps.get("pred_score")
        or ps.get("score")
        or ps.get("prediction_score")
        or ps.get("overall_score")
    )

    low = to_float_or_none(pt.get("low") or pt.get("threshold_low") or pt.get("floor"))
    high = to_float_or_none(pt.get("high") or pt.get("threshold_high") or pt.get("ceiling"))

    tier = score_tier(score)

    score_text = "N/A" if score is None else f"{score:.3f}"
    threshold_text = (
        "N/A"
        if low is None and high is None
        else f"{'' if low is None else f'{low:.3f}'} - {'' if high is None else f'{high:.3f}'}"
    )

    return (
        "<div style='border:1px solid #d1d5db;border-radius:10px;padding:12px;background:#f8fafc;'>"
        "<h4 style='margin:0 0 8px 0;'>Model Score Card</h4>"
        f"<p style='margin:4px 0;'><strong>Predicted Score:</strong> {score_text}</p>"
        f"<p style='margin:4px 0;'><strong>Tier:</strong> {tier}</p>"
        f"<p style='margin:4px 0;'><strong>Threshold Band:</strong> {threshold_text}</p>"
        "</div>"
    )

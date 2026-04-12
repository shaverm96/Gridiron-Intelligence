from __future__ import annotations

import json
import re
from typing import Any, Callable

import pandas as pd

THRESHOLD_OPERATOR_PATTERN = re.compile(
    r"(^|[_\-])(ge|gt|le|lt|gte|lte)(100(?:\.0+)?|[0-9]{1,2}(?:\.\d+)?)(?=$|[_\-])"
)


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


def transfer_to_percent_points(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
    except Exception:
        return None
    converted = numeric * 100.0 if abs(numeric) <= 1.0 else numeric
    return round(converted, 1)


def transfer_position_usage_order(position_hint: str, metric_cols: list[str]) -> list[str]:
    normalized = str(position_hint or "").strip().upper()
    custom_order = {
        "QB": ["pass", "rush", "overall", "third_down", "passing_downs"],
        "RB": ["rush", "pass", "overall", "third_down", "passing_downs"],
        "WR": ["pass", "overall", "third_down", "passing_downs", "rush"],
        "TE": ["pass", "overall", "third_down", "passing_downs", "rush"],
    }
    preferred = custom_order.get(normalized, ["overall", "pass", "rush", "third_down", "passing_downs"])
    ordered = [metric for metric in preferred if metric in metric_cols]
    for metric in metric_cols:
        if metric not in ordered:
            ordered.append(metric)
    return ordered


def transfer_position_stat_order(position_hint: str, stat_cols: list[str]) -> list[str]:
    normalized = str(position_hint or "").strip().upper()
    preferred_tokens = {
        "QB": ["passing", "pass", "rushing", "rush", "sack", "fumble"],
        "RB": ["rushing", "rush", "receiving", "pass", "fumble"],
        "WR": ["receiving", "rushing", "rush", "fumble"],
        "TE": ["receiving", "rushing", "rush", "fumble"],
    }
    tokens = preferred_tokens.get(normalized, ["passing", "rushing", "receiving", "defensive", "kicking", "punt"])

    def _score(column_name: str) -> tuple[int, str]:
        lowered = str(column_name or "").lower()
        for idx, token in enumerate(tokens):
            if token in lowered:
                return idx, lowered
        return len(tokens) + 1, lowered

    return sorted(stat_cols, key=_score)


def build_transfer_usage_with_yoy_table(
    usage_table_compact: list[dict[str, Any]],
    usage_yoy_compact: list[dict[str, Any]],
    position_hint: str = "",
) -> tuple[pd.DataFrame, list[str], list[str]]:
    usage_metrics = ["overall", "pass", "rush", "third_down", "passing_downs"]
    metric_cols = [metric for metric in usage_metrics if any(row.get(metric) is not None for row in usage_table_compact)]
    metric_cols = transfer_position_usage_order(position_hint, metric_cols)

    yoy_lookup: dict[int, dict[str, Any]] = {}
    for row in usage_yoy_compact:
        try:
            yoy_lookup[int(row.get("to_year") or 0)] = dict(row)
        except Exception:
            continue

    display_rows: list[dict[str, Any]] = []
    delta_cols: list[str] = []
    for row in usage_table_compact:
        out: dict[str, Any] = {
            "year": row.get("year"),
            "team": row.get("team"),
            "position": row.get("position"),
            "record_count": row.get("record_count"),
            "status": row.get("status"),
        }
        yoy_row = yoy_lookup.get(int(row.get("year") or 0), {})
        for metric in metric_cols:
            usage_col = f"{metric}_pct"
            delta_col = f"{metric}_yoy_delta_pct"
            out[usage_col] = transfer_to_percent_points(row.get(metric))
            out[delta_col] = transfer_to_percent_points(yoy_row.get(f"{metric}_delta"))
            if delta_col not in delta_cols:
                delta_cols.append(delta_col)
        display_rows.append(out)

    df = pd.DataFrame(display_rows)
    if df.empty:
        return df, [], []

    leading = ["year", "team", "position", "record_count", "status"]
    ordered_cols = list(leading)
    for metric in metric_cols:
        ordered_cols.append(f"{metric}_pct")
        ordered_cols.append(f"{metric}_yoy_delta_pct")

    existing_order = [col for col in ordered_cols if col in df.columns]
    df = df.reindex(columns=existing_order)
    usage_cols = [f"{metric}_pct" for metric in metric_cols if f"{metric}_pct" in df.columns]
    return df, usage_cols, delta_cols


def split_team_tokens_text(value: Any) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts = [part.strip() for part in re.split(r"[|,;/]+", text) if str(part).strip()]
    deduped: list[str] = []
    for part in parts:
        if part not in deduped:
            deduped.append(part)
    return deduped


def rows_to_dynamic_table(rows: list[dict[str, Any]], leading_columns: list[str] | None = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    leading = list(leading_columns or [])
    all_keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in all_keys:
                all_keys.append(key)

    trailing = [key for key in all_keys if key not in leading]
    ordered_cols = [key for key in leading if key in all_keys] + trailing

    df = pd.DataFrame(rows)
    return df.reindex(columns=ordered_cols)


def parse_selected_player_label_data(label: str | None) -> tuple[str, str, str, str]:
    parts = [part.strip() for part in str(label or "").split("|")]
    name = parts[0] if len(parts) > 0 else ""
    position = parts[1] if len(parts) > 1 else ""
    high_school = parts[2] if len(parts) > 2 else ""
    year = parts[3] if len(parts) > 3 else ""
    return name, position, high_school, year


def extract_predicted_score_display_data(
    score_card_html: str | None,
    pred_score_row: dict[str, Any] | None,
    to_float_or_none: Callable[[Any], float | None],
) -> str:
    html_text = str(score_card_html or "")
    if html_text:
        plain_text = re.sub(r"<[^>]+>", " ", html_text)
        plain_text = " ".join(plain_text.split())

        for pattern in [
            r"Predicted\s*Score\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?(?:\s*/\s*100)?)",
            r"([0-9]+(?:\.[0-9]+)?\s*/\s*100)",
        ]:
            match = re.search(pattern, plain_text, flags=re.IGNORECASE)
            if match:
                return str(match.group(1)).replace(" / ", "/").strip()

    row = pred_score_row if isinstance(pred_score_row, dict) else {}
    for value in row.values():
        score = to_float_or_none(value)
        if score is None:
            continue
        if 0.0 <= score <= 1.0:
            return f"{score * 100.0:.3f}"
        return f"{score:.3f}"

    return "N/A"


def parse_historical_comparables_md_data(
    raw_md: str | None,
    to_float_or_none: Callable[[Any], float | None],
) -> dict[str, Any]:
    text = str(raw_md or "")
    lines = [line.strip() for line in text.splitlines() if line and line.strip()]

    target_position = ""
    rows: list[dict[str, Any]] = []

    def _is_placeholder(value: Any) -> bool:
        normalized = str(value or "").strip().lower()
        return normalized in {"", "-", "--", "n/a", "na", "none", "null", "unknown", "?"}

    def _clean_name(value: Any) -> str:
        name = str(value or "")
        name = re.sub(r"[*_`~]+", "", name)
        return re.sub(r"\s+", " ", name).strip(" -|")

    def _match_numeric(match_text: str) -> float | None:
        number_match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(match_text or ""))
        if not number_match:
            return None
        return to_float_or_none(number_match.group(1))

    for line in lines:
        clean = re.sub(r"^#{1,6}\s*", "", line).strip()
        if not clean:
            continue

        if clean.lower().startswith("target position:"):
            target_position = clean.split(":", 1)[1].strip() if ":" in clean else ""
            continue

        if clean.startswith("-") or clean.startswith("*"):
            body = clean[1:].strip()
            parsed = re.match(
                r"^(?P<name>.+?)\s*\((?P<year>\d{4})\s*,\s*(?P<state>[A-Za-z]{2})\)\s*\|\s*Match:\s*(?P<match>[^|]+?)\s*\|\s*Rating:\s*(?P<rating>.+)$",
                body,
            )
            if parsed:
                raw_match = str(parsed.group("match") or "").strip()
                match_value = _match_numeric(raw_match)
                match_display = ""
                if match_value is not None:
                    match_display = f"{match_value:.2f}%"
                elif not _is_placeholder(raw_match):
                    match_display = raw_match

                rows.append(
                    {
                        "name": _clean_name(parsed.group("name")),
                        "year": str(parsed.group("year") or "").strip(),
                        "state": str(parsed.group("state") or "").strip(),
                        "match": match_display,
                        "match_value": match_value,
                        "rating": str(parsed.group("rating") or "").strip(),
                        "raw": body,
                    }
                )
            else:
                cleaned_name = _clean_name(body)
                rows.append(
                    {
                        "name": cleaned_name,
                        "year": "",
                        "state": "",
                        "match": "",
                        "match_value": None,
                        "rating": "",
                        "raw": body,
                    }
                )

    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        name = _clean_name(row.get("name"))
        year = str(row.get("year") or "").strip()
        state = str(row.get("state") or "").strip()
        rating = str(row.get("rating") or "").strip()
        match = str(row.get("match") or "").strip()
        match_value = row.get("match_value")

        if _is_placeholder(name):
            continue

        has_real_metadata = any(not _is_placeholder(value) for value in [year, state, rating, match])
        if not has_real_metadata and match_value is None:
            continue

        if _is_placeholder(match) and match_value is None:
            continue

        filtered_rows.append(
            {
                "name": name,
                "year": "" if _is_placeholder(year) else year,
                "state": "" if _is_placeholder(state) else state,
                "rating": "" if _is_placeholder(rating) else rating,
                "match": "" if _is_placeholder(match) else match,
                "match_value": match_value,
                "raw": str(row.get("raw") or "").strip(),
            }
        )

    filtered_rows.sort(key=lambda row: (row.get("match_value") is not None, row.get("match_value") or -1.0), reverse=True)

    return {
        "target_position": target_position,
        "rows": filtered_rows,
        "raw": text,
    }


def parse_summary_notes_data(raw_text: str | None) -> list[dict[str, str]]:
    text = str(raw_text or "")
    lines = [line.rstrip() for line in text.splitlines() if line and line.strip()]
    notes: list[dict[str, str]] = []

    for line in lines:
        clean = re.sub(r"^\s*(?:[-*•]+|\d+\s*[\.)-])\s*", "", line).strip()
        if not clean:
            continue

        clean = re.sub(r"[*_`~]+", "", clean).strip()

        label = ""
        body = clean
        if ":" in clean:
            left, right = clean.split(":", 1)
            left_clean = re.sub(r"^\s*\d+\s*[\.)-]?\s*", "", left).strip()
            right_clean = right.strip()
            if left_clean and right_clean and len(left_clean) <= 36:
                label = left_clean
                body = right_clean

        notes.append({"label": label, "body": body})

    return notes


def build_recruiting_summary_layout_data(raw_text: str | None) -> dict[str, Any]:
    notes = parse_summary_notes_data(raw_text)
    raw = str(raw_text or "")

    def _norm_label(label: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(label or "").strip().lower()).strip()

    def _norm_text(text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "").strip().lower())

    def _extract_physical_profile(text: str) -> str:
        value = str(text or "")
        feet = None
        inches = None
        weight = None

        feet_match = re.search(r"\b(\d)\s*(?:'|ft|foot)[\s\-]*(\d{1,2})\b", value, flags=re.IGNORECASE)
        if not feet_match:
            feet_match = re.search(r"\b(\d)\s*[-\s]?foot[-\s]?(\d{1,2})\b", value, flags=re.IGNORECASE)
        if feet_match:
            feet = str(feet_match.group(1)).strip()
            inches = str(feet_match.group(2)).strip()

        weight_match = re.search(r"\b(\d{2,3})\s*(?:lbs?|pounds?)\b", value, flags=re.IGNORECASE)
        if weight_match:
            weight = str(weight_match.group(1)).strip()

        if feet and inches and weight:
            return f"{feet}'{inches}\", {weight} lbs"
        if feet and inches:
            return f"{feet}'{inches}\""
        if weight:
            return f"{weight} lbs"

        compact = re.search(r"(\d\s*'\s*\d{1,2}\"?\s*,?\s*\d{2,3}\s*(?:lbs?|pounds?))", value, flags=re.IGNORECASE)
        if compact:
            return str(compact.group(1)).strip()

        simple_weight = re.search(r"(\d{2,3}\s*(?:lbs?|pounds?))", value, flags=re.IGNORECASE)
        if simple_weight:
            return str(simple_weight.group(1)).strip()
        return ""

    def _extract_labeled_blocks(text: str, labels: list[str]) -> str:
        if not text.strip():
            return ""

        label_pattern = "|".join([re.escape(label) for label in labels])
        all_aliases = [alias for aliases in field_map.values() for alias in aliases]
        next_label_pattern = "|".join([re.escape(alias) for alias in all_aliases])

        pattern = (
            r"(?:^|\n)\s*(?:[-*•]+|\d+\s*[\.)-])?\s*"
            r"(?:\*\*)?(?:" + label_pattern + r")(?:\*\*)?\s*[:\-–]\s*"
            r"(?P<body>.*?)"
            r"(?=(?:\n\s*(?:[-*•]+|\d+\s*[\.)-])?\s*(?:\*\*)?(?:" + next_label_pattern + r")(?:\*\*)?\s*[:\-–])|$)"
        )

        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""

        body = re.sub(r"[*_`~]+", "", str(match.group("body") or "")).strip()
        body = re.sub(r"\s+", " ", body)
        return body

    def _first_unmapped_note(unmapped: list[dict[str, str]]) -> str:
        while unmapped:
            note = unmapped.pop(0)
            body = str(note.get("body") or "").strip()
            if body:
                return body
        return ""

    def _pop_note_by_keywords(unmapped: list[dict[str, str]], keywords: list[str]) -> str:
        for idx, note in enumerate(unmapped):
            body = str(note.get("body") or "")
            if not body.strip():
                continue
            body_norm = _norm_text(body)
            if any(keyword in body_norm for keyword in keywords):
                unmapped.pop(idx)
                return body.strip()
        return ""

    def _classify_note(body: str) -> str:
        text = _norm_text(body)
        if not text:
            return ""

        has_date_token = bool(
            re.search(r"\b(?:jan|feb|mar|apr|may|jun|july?|aug|sep|sept|oct|nov|dec)\b", text)
            or re.search(r"\b20\d{2}\b", text)
        )

        if _extract_physical_profile(body):
            return "physical_profile"
        if any(token in text for token in ["as of", "current date", "recency", "between", "latest", "no information provided"]):
            return "note_on_recency"
        if any(token in text for token in ["committed on", "official visit", "decommitted", "flip", "timeline", "announced", "visit"]) or ("committed" in text and has_date_token):
            return "commitment_timeline"
        if any(token in text for token in ["commit", "committed", "uncommitted", "signed", "offer", "status"]):
            return "recruiting_status"
        if any(token in text for token in ["baseball", "basketball", "track", "wrestling", "multi-sport", "high school", "all-metro", "background"]):
            return "athletic_background"
        if any(token in text for token in ["touchdown", "yards", "production", "performance", "campaign", "season", "injury", "acl", "stats"]):
            return "performance_notes"
        return ""

    def _extract_hero_name_and_subtitle(prospect_text: str) -> tuple[str, str]:
        text = str(prospect_text or "").strip()
        if not text:
            return "Prospect", ""

        sentence = text.rstrip(".")
        is_match = re.match(r"^(?P<name>[A-Z][\w'’.-]+(?:\s+[A-Z][\w'’.-]+){0,3})\s+is\s+(?P<rest>.+)$", sentence)
        if is_match:
            name = str(is_match.group("name") or "").strip()
            rest = str(is_match.group("rest") or "").strip()
            rest = re.sub(r"^(?:a|an)\s+", "", rest, flags=re.IGNORECASE)
            return name or "Prospect", rest

        comma_parts = [part.strip() for part in sentence.split(",") if part.strip()]
        if len(comma_parts) >= 2:
            return comma_parts[0], ", ".join(comma_parts[1:])

        return sentence, ""

    def _normalize_school_name(name: str) -> str:
        school = str(name or "").strip(" .,:;-")
        if not school:
            return ""
        school = re.sub(r"\s+", " ", school)
        school = re.sub(r"\bUniversity\b\.?$", "", school, flags=re.IGNORECASE).strip(" .,:;-")
        school = re.sub(r"\bCollege\b\.?$", "", school, flags=re.IGNORECASE).strip(" .,:;-")
        return school

    def _extract_commit_school(text: str) -> str:
        value = str(text or "")
        patterns = [
            r"\bcommitted\s+to\s+([A-Z][A-Za-z&'\.\-\s]{2,80}?)(?:\s+for\b|\.|,|;|$)",
            r"\bcommit(?:ted)?\s+for\s+([A-Z][A-Za-z&'\.\-\s]{2,80}?)(?:\.|,|;|$)",
            r"\bsigned\s+with\s+([A-Z][A-Za-z&'\.\-\s]{2,80}?)(?:\.|,|;|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, value, flags=re.IGNORECASE)
            if match:
                school = _normalize_school_name(str(match.group(1) or ""))
                if school:
                    return school
        return ""

    def _derive_recruiting_status(extracted_fields: dict[str, str], full_text: str) -> str:
        candidate_blobs = [
            str(extracted_fields.get("recruiting_status") or ""),
            str(extracted_fields.get("commitment_timeline") or ""),
            str(extracted_fields.get("prospect") or ""),
            str(full_text or ""),
        ]
        merged = "\n".join([blob for blob in candidate_blobs if str(blob).strip()])
        merged_norm = _norm_text(merged)

        school = _extract_commit_school(merged)
        if school:
            return f"Currently committed to {school}"

        open_markers = ["open", "uncommitted", "unsigned", "still considering", "not committed"]
        if any(marker in merged_norm for marker in open_markers):
            return "Open"

        return "Open"

    field_map = {
        "prospect": ["prospect"],
        "physical_profile": ["physical profile"],
        "recruiting_status": ["recruiting status"],
        "commitment_timeline": ["commitment timeline"],
        "athletic_background": ["athletic background"],
        "performance_notes": ["performance notes"],
        "note_on_recency": ["note on recency", "recency"],
    }

    extracted: dict[str, str] = {key: "" for key in field_map.keys()}

    # Primary parse path: use note labels when present.
    for note in notes:
        label_norm = _norm_label(note.get("label") or "")
        for key, aliases in field_map.items():
            if label_norm in aliases and not extracted[key]:
                extracted[key] = str(note.get("body") or "").strip()
                break

    # Secondary parse path: recover from markdown/numbered formats where labels may not split cleanly.
    for key, aliases in field_map.items():
        if extracted.get(key):
            continue
        extracted[key] = _extract_labeled_blocks(raw, aliases)

    # Detect physical profile from any available content if explicit field is missing.
    if not extracted.get("physical_profile"):
        for note in notes:
            found = _extract_physical_profile(note.get("body") or "")
            if found:
                extracted["physical_profile"] = found
                break
        if not extracted.get("physical_profile"):
            extracted["physical_profile"] = _extract_physical_profile(raw)

    # Defensive fallback: map unlabeled note bodies into dossier fields so card remains populated.
    if notes:
        labeled_bodies = {str(value).strip() for value in extracted.values() if str(value).strip()}
        unmapped = [
            note
            for note in notes
            if str(note.get("body") or "").strip() and str(note.get("body") or "").strip() not in labeled_bodies
        ]

        # Remove rows that are clearly physical from generic fallback pool.
        residual = []
        for note in unmapped:
            body = str(note.get("body") or "").strip()
            if not body:
                continue
            if _classify_note(body) == "physical_profile":
                if not extracted.get("physical_profile"):
                    extracted["physical_profile"] = _extract_physical_profile(body)
                continue
            residual.append(note)
        unmapped = residual

        if not extracted.get("prospect"):
            extracted["prospect"] = _first_unmapped_note(unmapped)

        # Try semantic classification before loose keyword assignment.
        classified_values: dict[str, str] = {}
        for note in list(unmapped):
            body = str(note.get("body") or "").strip()
            note_type = _classify_note(body)
            if note_type and note_type in extracted and not extracted.get(note_type) and note_type not in classified_values:
                classified_values[note_type] = body

        for key, value in classified_values.items():
            extracted[key] = value
            for idx, note in enumerate(unmapped):
                if str(note.get("body") or "").strip() == value:
                    unmapped.pop(idx)
                    break

        if not extracted.get("recruiting_status"):
            extracted["recruiting_status"] = _pop_note_by_keywords(
                unmapped,
                ["committed", "offer", "status", "commitment", "signed", "decommit"],
            )
        if not extracted.get("commitment_timeline"):
            extracted["commitment_timeline"] = _pop_note_by_keywords(
                unmapped,
                ["timeline", "visit", "june", "july", "date", "official visit", "announced"],
            )
        if not extracted.get("athletic_background"):
            extracted["athletic_background"] = _pop_note_by_keywords(
                unmapped,
                ["baseball", "basketball", "track", "wrestling", "multi-sport", "athletic background"],
            )
        if not extracted.get("performance_notes"):
            extracted["performance_notes"] = _pop_note_by_keywords(
                unmapped,
                ["touchdown", "yards", "performance", "stats", "production", "campaign", "season"],
            )
        if not extracted.get("note_on_recency"):
            extracted["note_on_recency"] = _pop_note_by_keywords(
                unmapped,
                ["recency", "current date", "as of", "between", "today", "recent", "updated"],
            )

        # Fill remaining empty fields in display order using leftover note bodies.
        for key in [
            "recruiting_status",
            "commitment_timeline",
            "athletic_background",
            "performance_notes",
            "note_on_recency",
        ]:
            if extracted.get(key):
                continue
            extracted[key] = _first_unmapped_note(unmapped)

    # Deterministic status field: keep concise and schema-consistent.
    extracted["recruiting_status"] = _derive_recruiting_status(extracted, raw)

    prospect_text = str(extracted.get("prospect") or "").strip()
    hero_name, hero_subtitle = _extract_hero_name_and_subtitle(prospect_text)

    if not hero_subtitle and extracted.get("recruiting_status"):
        hero_subtitle = str(extracted.get("recruiting_status") or "").strip()

    if len(hero_subtitle) > 160:
        hero_subtitle = hero_subtitle[:157].rstrip() + "..."

    grid_fields = [
        ("recruiting_status", "Recruiting Status"),
        ("commitment_timeline", "Commitment Timeline"),
        ("athletic_background", "Athletic Background"),
        ("performance_notes", "Performance Notes"),
    ]

    grid_items = []
    for key, title in grid_fields:
        value = str(extracted.get(key) or "").strip()
        if value:
            grid_items.append({"key": key, "title": title, "value": value})

    return {
        "hero_name": hero_name,
        "hero_subtitle": hero_subtitle,
        "physical_profile": str(extracted.get("physical_profile") or "").strip(),
        "grid_items": grid_items,
        "note_on_recency": str(extracted.get("note_on_recency") or "").strip(),
        "notes": notes,
    }


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

        if "ge" in key or "gte" in key or ">=" in key:
            return f"Chance to reach >= {threshold_txt}" if threshold_txt else "Chance to reach upper threshold"
        if "gt" in key or ">" in key:
            return f"Chance to exceed > {threshold_txt}" if threshold_txt else "Chance to exceed upper threshold"
        if "le" in key or "lte" in key or "<=" in key:
            return f"Chance to stay <= {threshold_txt}" if threshold_txt else "Chance to stay below threshold"
        if "lt" in key or "<" in key:
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

    score_text = "N/A" if score is None else f"{score:.3f}"
    score_pct = None if score is None else max(0.0, min(100.0, float(score)))
    threshold_text = (
        "N/A"
        if low is None and high is None
        else f"{'' if low is None else f'{low:.1f}'} - {'' if high is None else f'{high:.1f}'}"
    )
    threshold_band_html = (
        ""
        if threshold_text == "N/A"
        else f"<p style='margin:4px 0;color:#f9fafb;'><strong>Threshold Band:</strong> {threshold_text}</p>"
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

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _query_one(
    sb: Any,
    table_name: str,
    recruit_id: str,
    order_columns: list[tuple[str, bool]] | None = None,
) -> dict[str, Any]:
    if sb is None:
        return {}

    query = sb.table(table_name).select("*").eq("recruit_id", recruit_id)
    for col, desc in order_columns or []:
        query = query.order(col, desc=desc)
    rows = query.limit(1).execute().data or []
    return rows[0] if rows else {}


def load_player_index_data(recruits_path: Path, years: list[int]) -> pd.DataFrame:
    base_cols = [
        "recruit_id",
        "player_name",
        "position",
        "high_school",
        "year",
        "rating",
        "player_label",
        "athlete_id",
    ]

    if not recruits_path.exists():
        return pd.DataFrame(columns=base_cols)

    df = pd.read_csv(recruits_path)
    for col in ["recruit_id", "player_name", "position", "high_school", "year", "rating", "athlete_id"]:
        if col not in df.columns:
            df[col] = None

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].isin([int(y) for y in years])].copy()

    df["recruit_id"] = df["recruit_id"].apply(_safe_str)
    df["player_name"] = df["player_name"].apply(_safe_str)
    df["position"] = df["position"].apply(_safe_str)
    df["high_school"] = df["high_school"].apply(_safe_str)

    # Keep athlete_id only when positive integer.
    df["athlete_id"] = df["athlete_id"].apply(_safe_int)
    df.loc[df["athlete_id"].fillna(0) <= 0, "athlete_id"] = None

    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    df["player_label"] = (
        df["player_name"].astype(str)
        + " | "
        + df["position"].astype(str)
        + " | "
        + df["high_school"].astype(str)
        + " | "
        + df["year"].astype("Int64").astype(str)
    )

    df = df.sort_values(["year", "rating_num", "player_name"], ascending=[True, False, True]).reset_index(drop=True)
    return df[base_cols]


def fetch_player_bundle_data(
    sb: Any,
    recruit_id: str,
    tables: dict[str, str],
    build_player_profile_view: Callable[[dict[str, Any]], dict[str, Any]],
    clean_scouting_profile: Callable[[dict[str, Any]], dict[str, Any]],
    merge_scouting_sources: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    rid = _safe_str(recruit_id)
    if not rid:
        return {
            "recruit_id": "",
            "player": {},
            "player_profile": {},
            "scouting_raw": {},
            "scouting_merged": {},
            "scouting_clean": {},
            "pred_score": {},
            "pred_threshold": {},
        }

    player_row = _query_one(sb, tables["player_master"], rid)
    scouting_row = _query_one(sb, tables["scouting_features"], rid)
    pred_score_row = _query_one(sb, tables["pred_score"], rid, order_columns=[("as_of_date", True), ("updated_at", True)])
    pred_thr_row = _query_one(sb, tables["pred_threshold"], rid, order_columns=[("as_of_date", True), ("updated_at", True)])

    scouting_merged = merge_scouting_sources(scouting_row) if scouting_row else {}
    scouting_clean = clean_scouting_profile(scouting_merged)
    player_profile = build_player_profile_view(player_row)

    return {
        "recruit_id": rid,
        "player": player_row,
        "player_profile": player_profile,
        "scouting_raw": scouting_row,
        "scouting_merged": scouting_merged,
        "scouting_clean": scouting_clean,
        "pred_score": pred_score_row,
        "pred_threshold": pred_thr_row,
    }

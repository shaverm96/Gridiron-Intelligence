from __future__ import annotations

import re
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


def supabase_fetch_all_rows_data(
    sb: Any,
    table_name: str,
    columns: str,
    batch_size: int = 1000,
    max_rows: int = 50000,
) -> list[dict[str, Any]]:
    if sb is None:
        return []

    rows: list[dict[str, Any]] = []
    start = 0
    while start < max_rows:
        end = start + batch_size - 1
        response = sb.table(table_name).select(columns).range(start, end).execute()
        chunk = list(response.data or [])
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < batch_size:
            break
        start += batch_size
    return rows


def load_player_index_from_supabase_data(
    sb: Any,
    table_name: str,
    years: list[int],
    position_map: dict[str, str],
) -> pd.DataFrame:
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

    if sb is None:
        return pd.DataFrame(columns=base_cols)

    select_cols = (
        "recruit_id,recruit_name,full_name,player_name,position_group,position,"
        "recruit_class,year,composite_rating,rating,high_school,hs_city,hs_state,"
        "athlete_id,cfbd_athlete_id"
    )
    rows = supabase_fetch_all_rows_data(
        sb=sb,
        table_name=table_name,
        columns=select_cols,
        batch_size=1000,
        max_rows=50000,
    )
    if not rows:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(rows)
    if "recruit_id" not in df.columns:
        return pd.DataFrame(columns=base_cols)

    recruit_class_series = df["recruit_class"] if "recruit_class" in df.columns else pd.Series([None] * len(df), index=df.index)
    year_series = df["year"] if "year" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["year"] = pd.to_numeric(recruit_class_series.where(recruit_class_series.notna(), year_series), errors="coerce")
    year_filter = [int(y) for y in years]
    df = df[df["year"].isin(year_filter)].copy()

    def _coalesce_text(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
        out = pd.Series(["" for _ in range(len(frame))], index=frame.index, dtype="object")
        for col in cols:
            if col in frame.columns:
                vals = frame[col].fillna("").astype(str).str.strip()
                out = out.where(out.astype(str).str.strip() != "", vals)
        return out

    def _normalize_position_group(position_value: str | None) -> str:
        raw = str(position_value or "").strip().upper()
        return position_map.get(raw, raw)

    df["recruit_id"] = df["recruit_id"].astype(str).str.strip()
    df["player_name"] = _coalesce_text(df, ["recruit_name", "full_name", "player_name"])
    df["position"] = _coalesce_text(df, ["position_group", "position"]).apply(_normalize_position_group)
    df["high_school"] = _coalesce_text(df, ["high_school", "hs_city", "hs_state"])
    composite_rating_series = df["composite_rating"] if "composite_rating" in df.columns else pd.Series([None] * len(df), index=df.index)
    rating_series = df["rating"] if "rating" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["rating"] = pd.to_numeric(composite_rating_series.where(composite_rating_series.notna(), rating_series), errors="coerce")

    athlete_series = df.get("athlete_id") if "athlete_id" in df.columns else pd.Series([None] * len(df), index=df.index)
    cfbd_athlete_series = df.get("cfbd_athlete_id") if "cfbd_athlete_id" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["athlete_id"] = athlete_series.where(athlete_series.notna(), cfbd_athlete_series)
    df["athlete_id"] = pd.to_numeric(df["athlete_id"], errors="coerce").astype("Int64")

    df["player_label"] = (
        df["player_name"].astype(str)
        + " | "
        + df["position"].astype(str)
        + " | "
        + df["high_school"].astype(str)
        + " | "
        + df["year"].astype("Int64").astype(str)
    )

    df = df[df["recruit_id"].str.len() > 0].drop_duplicates(subset=["recruit_id"]).copy()
    df = df.sort_values(["year", "rating", "player_name"], ascending=[True, False, True]).reset_index(drop=True)
    return df[base_cols]


def _normalize_recruit_search_text(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = text.replace("%", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:80]


def _build_recruit_candidate_frame(rows: list[dict[str, Any]], position_map: dict[str, str]) -> pd.DataFrame:
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

    if not rows:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(rows)
    if "recruit_id" not in df.columns:
        return pd.DataFrame(columns=base_cols)

    for col in ["recruit_name", "full_name", "player_name", "position_group", "position", "recruit_class", "year", "composite_rating", "rating", "high_school", "hs_city", "hs_state", "athlete_id", "cfbd_athlete_id"]:
        if col not in df.columns:
            df[col] = None

    recruit_class_series = df["recruit_class"] if "recruit_class" in df.columns else pd.Series([None] * len(df), index=df.index)
    year_series = df["year"] if "year" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["year"] = pd.to_numeric(recruit_class_series.where(recruit_class_series.notna(), year_series), errors="coerce")

    def _coalesce_text(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
        out = pd.Series(["" for _ in range(len(frame))], index=frame.index, dtype="object")
        for col in cols:
            if col in frame.columns:
                vals = frame[col].fillna("").astype(str).str.strip()
                out = out.where(out.astype(str).str.strip() != "", vals)
        return out

    def _normalize_position_group(position_value: str | None) -> str:
        raw = str(position_value or "").strip().upper()
        return position_map.get(raw, raw)

    df["recruit_id"] = df["recruit_id"].astype(str).str.strip()
    df["player_name"] = _coalesce_text(df, ["recruit_name", "full_name", "player_name"])
    df["position"] = _coalesce_text(df, ["position_group", "position"]).apply(_normalize_position_group)
    df["high_school"] = _coalesce_text(df, ["high_school", "hs_city", "hs_state"])
    composite_rating_series = df["composite_rating"] if "composite_rating" in df.columns else pd.Series([None] * len(df), index=df.index)
    rating_series = df["rating"] if "rating" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["rating"] = pd.to_numeric(composite_rating_series.where(composite_rating_series.notna(), rating_series), errors="coerce")

    athlete_series = df.get("athlete_id") if "athlete_id" in df.columns else pd.Series([None] * len(df), index=df.index)
    cfbd_athlete_series = df.get("cfbd_athlete_id") if "cfbd_athlete_id" in df.columns else pd.Series([None] * len(df), index=df.index)
    df["athlete_id"] = athlete_series.where(athlete_series.notna(), cfbd_athlete_series)
    df["athlete_id"] = pd.to_numeric(df["athlete_id"], errors="coerce").astype("Int64")

    df["player_label"] = (
        df["player_name"].astype(str)
        + " | "
        + df["position"].astype(str)
        + " | "
        + df["high_school"].astype(str)
        + " | "
        + df["year"].astype("Int64").astype(str)
    )

    df = df[df["recruit_id"].str.len() > 0].drop_duplicates(subset=["recruit_id"]).copy()
    df = df.sort_values(["rating", "player_name"], ascending=[False, True], na_position="last").reset_index(drop=True)
    return df[base_cols]


def _fetch_recruit_candidate_rows(
    sb: Any,
    table_name: str,
    year: int | None,
    position: str | None,
    search_text: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    if sb is None:
        return []

    normalized_search = _normalize_recruit_search_text(search_text or "")
    normalized_position = str(position or "").strip().upper()
    limit_value = int(limit) if int(limit or 0) > 0 else 100
    year_fields: list[tuple[str, int | None]] = [("year", None), ("recruit_class", None)] if year is not None else [("", None)]
    if year is not None:
        year_fields = [("year", int(year)), ("recruit_class", int(year))]
    position_fields: list[tuple[str, str | None]] = [("", None)]
    if normalized_position and normalized_position != "ALL":
        position_fields = [("position_group", normalized_position), ("position", normalized_position)]

    select_cols = (
        "recruit_id,recruit_name,full_name,player_name,position_group,position,"
        "recruit_class,year,composite_rating,rating,high_school,hs_city,hs_state,"
        "athlete_id,cfbd_athlete_id,search_text"
    )
    rows: list[dict[str, Any]] = []
    for year_column, year_value in year_fields:
        for position_column, position_value in position_fields:
            query = sb.table(table_name).select(select_cols)
            if year_column and year_value is not None:
                query = query.eq(year_column, year_value)
            if position_column and position_value:
                query = query.eq(position_column, position_value)
            if len(normalized_search) >= 3:
                query = query.ilike("search_text", f"%{normalized_search}%")
            query = query.order("rating", desc=True).order("player_name").limit(limit_value)
            rows.extend(query.execute().data or [])

    return rows


def load_recruit_candidate_window_from_supabase_data(
    sb: Any,
    table_name: str,
    year: int,
    position: str,
    position_map: dict[str, str],
    limit: int = 100,
) -> pd.DataFrame:
    rows = _fetch_recruit_candidate_rows(
        sb=sb,
        table_name=table_name,
        year=year,
        position=position,
        search_text=None,
        limit=limit,
    )
    df = _build_recruit_candidate_frame(rows, position_map=position_map)
    return df.head(int(limit)) if limit and int(limit) > 0 else df


def search_recruit_candidate_matches_from_supabase_data(
    sb: Any,
    table_name: str,
    year: int,
    position: str,
    search_text: str,
    position_map: dict[str, str],
    limit: int = 100,
) -> pd.DataFrame:
    rows = _fetch_recruit_candidate_rows(
        sb=sb,
        table_name=table_name,
        year=year,
        position=position,
        search_text=search_text,
        limit=limit,
    )
    df = _build_recruit_candidate_frame(rows, position_map=position_map)
    return df.head(int(limit)) if limit and int(limit) > 0 else df


def load_transfer_player_index_from_supabase_data(
    sb: Any,
    table_name: str = "gi_college_master",
) -> pd.DataFrame:
    base_cols = [
        "college_player_id",
        "cfbd_athlete_id",
        "player_name",
        "position",
        "teams",
        "conference",
        "first_season",
        "last_season",
        "season_span",
        "player_label",
    ]

    if sb is None:
        return pd.DataFrame(columns=base_cols)

    rows = supabase_fetch_all_rows_data(
        sb=sb,
        table_name=table_name,
        columns=(
            "college_player_id,cfbd_athlete_id,full_name,first_name,last_name,position,teams,"
            "conference,first_season,last_season,season_span"
        ),
        batch_size=1000,
        max_rows=50000,
    )

    if not rows:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(rows)
    if "college_player_id" not in df.columns:
        return pd.DataFrame(columns=base_cols)

    for col in ["full_name", "first_name", "last_name", "position", "teams", "conference", "season_span"]:
        if col not in df.columns:
            df[col] = ""

    df["last_season"] = pd.to_numeric(df.get("last_season"), errors="coerce")
    df = df[df["last_season"] == 2025].copy()

    df["cfbd_athlete_id"] = df.get("cfbd_athlete_id", "").astype(str).str.strip()
    df = df[df["cfbd_athlete_id"].str.len() > 0].copy()

    df["college_player_id"] = df["college_player_id"].astype(str).str.strip()
    df["position"] = df["position"].fillna("").astype(str).str.strip().str.upper()

    df["player_name"] = df["full_name"].fillna("").astype(str).str.strip()
    missing_name = df["player_name"].str.len() == 0
    df.loc[missing_name, "player_name"] = (
        df.loc[missing_name, "first_name"].fillna("").astype(str).str.strip()
        + " "
        + df.loc[missing_name, "last_name"].fillna("").astype(str).str.strip()
    ).str.strip()

    df["first_season"] = pd.to_numeric(df.get("first_season"), errors="coerce").astype("Int64")
    df["last_season"] = pd.to_numeric(df.get("last_season"), errors="coerce").astype("Int64")

    span = df["season_span"].fillna("").astype(str).str.strip()
    generated_span = (
        df["first_season"].astype(str)
        + "-"
        + df["last_season"].astype(str)
    )
    df["season_span"] = span.where(span.str.len() > 0, generated_span)

    df["player_label"] = (
        df["player_name"].astype(str)
        + " | "
        + df["position"].replace("", "UNK").astype(str)
        + " | "
        + df["teams"].fillna("").astype(str)
        + " | "
        + df["season_span"].fillna("").astype(str)
    )

    df = df[df["college_player_id"].str.len() > 0].drop_duplicates(subset=["college_player_id"]).copy()
    df = df.sort_values(["position", "player_name"], ascending=[True, True]).reset_index(drop=True)
    return df[base_cols]


def _normalize_transfer_search_text(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = text.replace("%", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:80]


def search_transfer_player_index_from_supabase_data(
    sb: Any,
    table_name: str = "gi_college_master",
    position: str | None = None,
    search_text: str | None = None,
    limit: int = 25,
) -> pd.DataFrame:
    base_cols = [
        "college_player_id",
        "cfbd_athlete_id",
        "player_name",
        "position",
        "teams",
        "conference",
        "first_season",
        "last_season",
        "season_span",
        "player_label",
    ]

    if sb is None:
        return pd.DataFrame(columns=base_cols)

    normalized_search = _normalize_transfer_search_text(search_text or "")
    normalized_position = str(position or "").strip().upper()

    if len(normalized_search) < 3:
        return pd.DataFrame(columns=base_cols)

    query = (
        sb.table(table_name)
        .select(
            "college_player_id,cfbd_athlete_id,full_name,first_name,last_name,position,teams,"
            "conference,first_season,last_season,season_span,search_text"
        )
        .eq("last_season", 2025)
        .ilike("search_text", f"%{normalized_search}%")
    )
    if normalized_position and normalized_position != "ALL":
        query = query.eq("position", normalized_position)
    if limit and limit > 0:
        query = query.limit(int(limit))

    rows = query.execute().data or []
    if not rows:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(rows)
    if "college_player_id" not in df.columns:
        return pd.DataFrame(columns=base_cols)

    for col in ["full_name", "first_name", "last_name", "position", "teams", "conference", "season_span"]:
        if col not in df.columns:
            df[col] = ""

    df["last_season"] = pd.to_numeric(df.get("last_season"), errors="coerce")
    df = df[df["last_season"] == 2025].copy()

    if normalized_position and normalized_position != "ALL":
        df["position"] = df["position"].fillna("").astype(str).str.strip().str.upper()
        df = df[df["position"] == normalized_position].copy()
    else:
        df["position"] = df["position"].fillna("").astype(str).str.strip().str.upper()

    df["cfbd_athlete_id"] = df.get("cfbd_athlete_id", "").astype(str).str.strip()
    df = df[df["cfbd_athlete_id"].str.len() > 0].copy()

    df["college_player_id"] = df["college_player_id"].astype(str).str.strip()
    df["player_name"] = df["full_name"].fillna("").astype(str).str.strip()
    missing_name = df["player_name"].str.len() == 0
    df.loc[missing_name, "player_name"] = (
        df.loc[missing_name, "first_name"].fillna("").astype(str).str.strip()
        + " "
        + df.loc[missing_name, "last_name"].fillna("").astype(str).str.strip()
    ).str.strip()

    df["first_season"] = pd.to_numeric(df.get("first_season"), errors="coerce").astype("Int64")
    df["last_season"] = pd.to_numeric(df.get("last_season"), errors="coerce").astype("Int64")

    span = df["season_span"].fillna("").astype(str).str.strip()
    generated_span = df["first_season"].astype(str) + "-" + df["last_season"].astype(str)
    df["season_span"] = span.where(span.str.len() > 0, generated_span)

    df["player_label"] = (
        df["player_name"].astype(str)
        + " | "
        + df["position"].replace("", "UNK").astype(str)
        + " | "
        + df["teams"].fillna("").astype(str)
        + " | "
        + df["season_span"].fillna("").astype(str)
    )

    df = df[df["college_player_id"].str.len() > 0].drop_duplicates(subset=["college_player_id"]).copy()
    df = df.sort_values(["position", "player_name"], ascending=[True, True]).reset_index(drop=True)
    return df[base_cols]


def load_model_tiers_from_supabase_data(
    sb: Any,
    pred_score_table: str,
) -> pd.DataFrame:
    base_cols = ["Score Range", "Career Designation", "College Outlook", "Professional Outlook", "low", "high", "count"]
    if sb is None:
        return pd.DataFrame(columns=base_cols)

    try:
        rows = supabase_fetch_all_rows_data(
            sb=sb,
            table_name=pred_score_table,
            columns="predictive_score_0_100,contrib_tier_raw",
            batch_size=1000,
            max_rows=50000,
        )
    except Exception:
        return pd.DataFrame(columns=base_cols)

    if not rows:
        return pd.DataFrame(columns=base_cols)

    df = pd.DataFrame(rows)
    if "predictive_score_0_100" not in df.columns:
        return pd.DataFrame(columns=base_cols)

    df["predictive_score_0_100"] = pd.to_numeric(df["predictive_score_0_100"], errors="coerce")
    df["contrib_tier_raw"] = df.get("contrib_tier_raw", "").fillna("Unknown").astype(str).str.strip()
    df.loc[df["contrib_tier_raw"] == "", "contrib_tier_raw"] = "Unknown"
    df = df[df["predictive_score_0_100"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=base_cols)

    grouped = (
        df.groupby("contrib_tier_raw", dropna=False)["predictive_score_0_100"]
        .agg(["min", "max", "count"])
        .reset_index()
        .rename(columns={"contrib_tier_raw": "Career Designation", "min": "low", "max": "high"})
    )
    grouped["Score Range"] = grouped.apply(lambda r: f"{r['low']:.2f}-{r['high']:.2f}", axis=1)
    grouped["College Outlook"] = "Derived from prediction score table"
    grouped["Professional Outlook"] = "Derived from prediction score table"
    return grouped[["Score Range", "Career Designation", "College Outlook", "Professional Outlook", "low", "high", "count"]].sort_values(["low", "high"]).reset_index(drop=True)


def score_tier_from_tiers_data(
    score: float | None,
    tiers: pd.DataFrame,
    to_float_or_none: Callable[[Any], float | None],
) -> str:
    if score is None or tiers.empty:
        return "Unknown"
    for _, row in tiers.iterrows():
        low, high = to_float_or_none(row.get("low")), to_float_or_none(row.get("high"))
        if low is not None and high is not None and low <= score <= high:
            return str(row.get("Career Designation", "Unknown"))
    return "Unknown"


def tier_definitions_markdown_data(tiers: pd.DataFrame) -> str:
    if tiers.empty:
        return "Tier definitions unavailable."
    return "\n".join([
        f"- **{r.get('Career Designation', '')}** ({r.get('Score Range', '')}): "
        f"Samples: {int(r.get('count', 0) or 0)}"
        for _, r in tiers.iterrows()
    ])

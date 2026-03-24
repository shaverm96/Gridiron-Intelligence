from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _safe_score(
    target_row: pd.Series,
    pool_df: pd.DataFrame,
    numeric_cols: list[str],
) -> pd.Series:
    df = pd.concat([target_row.to_frame().T, pool_df], ignore_index=True)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
        df[col] = df[col].fillna(df[col].mean()).fillna(0.0)

    arr = df[numeric_cols].to_numpy(dtype=float)
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    spans = np.where((max_vals - min_vals) == 0, 1.0, (max_vals - min_vals))
    scaled = (arr - min_vals) / spans

    dists = np.linalg.norm(scaled - scaled[0], axis=1)
    max_dist = np.sqrt(float(len(numeric_cols))) if numeric_cols else 1.0
    sims = np.clip(1.0 - (dists / max_dist), 0.0, 1.0)
    return pd.Series(sims[1:], index=pool_df.index)


def get_historical_player_comparables_data(
    sb: Any,
    recruit_id: str,
    tables: dict[str, str],
    to_float_or_none: Any,
    score_tier: Any,
) -> str:
    rid = str(recruit_id or "").strip()
    if not rid:
        return "Historical comparables unavailable: missing recruit_id."
    if sb is None:
        return "Historical comparables unavailable: Supabase client is not configured."

    target_rows = (
        sb.table(tables["player_master"])
        .select("recruit_id, player_name, year, position, rating, height_inches, weight_lbs, state")
        .eq("recruit_id", rid)
        .limit(1)
        .execute()
        .data
        or []
    )
    if not target_rows:
        return f"Historical comparables unavailable: recruit_id {rid} not found."

    target = target_rows[0]
    target_pos = str(target.get("position") or "").strip()
    if not target_pos:
        return "Historical comparables unavailable: target player has no position."

    pool_rows = (
        sb.table(tables["player_master"])
        .select("recruit_id, player_name, year, position, rating, height_inches, weight_lbs, state")
        .eq("position", target_pos)
        .lte("year", 2022)
        .execute()
        .data
        or []
    )
    if not pool_rows:
        return f"No historical comparables found for position {target_pos}."

    pool_df = pd.DataFrame(pool_rows)
    pool_df = pool_df[pool_df["recruit_id"].astype(str) != rid].copy()
    if pool_df.empty:
        return "No historical comparables found after filtering."

    numeric_cols = ["rating", "height_inches", "weight_lbs"]
    target_series = pd.Series(target)
    pool_df["similarity"] = _safe_score(target_series, pool_df, numeric_cols)

    target_state = str(target.get("state") or "").strip().upper()
    if target_state:
        state_bonus = (pool_df["state"].astype(str).str.strip().str.upper() == target_state).astype(float) * 0.05
        pool_df["similarity"] = np.clip(pool_df["similarity"] + state_bonus, 0.0, 1.0)

    pool_df["similarity_pct"] = (pool_df["similarity"] * 100.0).round(2)
    comps = pool_df.sort_values("similarity_pct", ascending=False).head(5)

    target_rating = to_float_or_none(target.get("rating"))
    target_tier = score_tier(target_rating)

    lines = [
        f"### Historical Comparables for {target.get('player_name', rid)}",
        f"Target Position: {target_pos}",
        f"Target Tier: {target_tier}",
        "---",
    ]

    for _, row in comps.iterrows():
        lines.append(
            "- **{name}** ({year}, {state}) | Match: {sim}% | Rating: {rating}".format(
                name=row.get("player_name", "Unknown"),
                year=row.get("year", "N/A"),
                state=row.get("state", "N/A"),
                sim=row.get("similarity_pct", "N/A"),
                rating=row.get("rating", "N/A"),
            )
        )

    return "\n".join(lines)

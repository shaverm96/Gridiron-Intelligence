from __future__ import annotations

import json
from typing import Any


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

    tier_defs = tier_definitions_markdown() if callable(tier_definitions_markdown) else str(tier_definitions_markdown)

    return (
        "You are a senior college football recruiting scout.\n"
        f"Persona: {persona}\n"
        "Use only provided context. If data is missing, say so clearly.\n\n"
        f"Year: {year}\n"
        f"Target Team: {target_team}\n\n"
        "Player Profile JSON:\n"
        f"{json.dumps(player_row, indent=2, default=str)}\n\n"
        "Filtered Scouting JSON:\n"
        f"{json.dumps(scouting_clean, indent=2, default=str)}\n\n"
        f"HS Athletic Background:\n{hs_athletic_background or 'N/A'}\n\n"
        "Prediction Score Row JSON:\n"
        f"{json.dumps(pred_score_row, indent=2, default=str)}\n\n"
        "Prediction Threshold Row JSON:\n"
        f"{json.dumps(pred_thr_row, indent=2, default=str)}\n\n"
        f"Web Intelligence Summary:\n{web_summary}\n\n"
        f"Vector Insights:\n{vector_block}\n\n"
        f"Historical Comparables:\n{historical_comparables_md}\n\n"
        f"Tier Definitions:\n{tier_defs}\n\n"
        "Output sections in order:\n"
        "1) Player Snapshot\n"
        "2) Trait Evaluation\n"
        "3) Scheme and Team Fit\n"
        "4) Development Risks\n"
        "5) Final Recommendation and Confidence\n"
    )


def run_final_synthesis_data(
    prompt: str,
    final_model: str,
    get_llm: Any,
    llm_response_to_text: Any,
) -> str:
    llm = get_llm(final_model, temperature=0.25, max_output_tokens=2200)
    if llm is None:
        return "Final synthesis skipped: Gemini model is not configured."

    try:
        response = llm.invoke(prompt)
        text = llm_response_to_text(response)
        return str(text).strip() if text else "Final synthesis returned empty output."
    except Exception as exc:
        return f"Final synthesis failed: {exc}"

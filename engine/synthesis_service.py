from __future__ import annotations

from typing import Any

from .prompt_architecture import build_master_prompt


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

    retrieved_context = {
        "player_profile": player_row,
        "filtered_scouting": scouting_clean,
        "hs_athletic_background": hs_athletic_background or "N/A",
        "prediction_score_row": pred_score_row,
        "prediction_threshold_row": pred_thr_row,
        "web_intelligence_summary": web_summary,
        "vector_insights": vector_block,
        "historical_comparables": historical_comparables_md,
        "tier_definitions": tier_defs,
    }

    user_prompt = (
        f"Generate a football scouting report for {player_row.get('player_name', 'the player')} "
        f"with fit evaluation for {target_team}."
    )

    return build_master_prompt(
        player_name=str(player_row.get("player_name") or "Unknown Player"),
        target_team=target_team,
        year=year,
        user_prompt=user_prompt,
        retrieved_context=retrieved_context,
        persona=persona,
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

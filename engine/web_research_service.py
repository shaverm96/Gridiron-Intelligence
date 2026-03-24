from __future__ import annotations

from typing import Any


def duckduckgo_search_data(
    ddgs_class: Any,
    player_name: str,
    position: str,
    high_school: str,
    year: int,
    target_search_sites: list[str],
    max_results: int = 12,
) -> list[dict[str, str]]:
    if ddgs_class is None:
        return []

    query = (
        f"{player_name} {position} {high_school} {year} football recruiting "
        "(site:maxpreps.com OR site:247sports.com OR site:rivals.com OR site:espn.com OR site:on3.com)"
    )

    rows: list[dict[str, str]] = []
    try:
        with ddgs_class() as ddgs:
            for result in ddgs.text(query, max_results=max_results):
                url = str(result.get("href") or "")
                if not url:
                    continue
                if target_search_sites and not any(site in url for site in target_search_sites):
                    continue
                rows.append(
                    {
                        "title": str(result.get("title") or ""),
                        "url": url,
                        "snippet": str(result.get("body") or ""),
                    }
                )
    except Exception:
        return []

    return rows


def summarize_web_with_flash_lite_data(
    player_name: str,
    position: str,
    search_rows: list[dict[str, str]],
    summary_model: str,
    get_llm: Any,
    llm_response_to_text: Any,
) -> str:
    if not search_rows:
        return "No relevant web articles were found from target recruiting sites."

    llm = get_llm(summary_model, temperature=0.0, max_output_tokens=1200)
    if llm is None:
        return "Web summary skipped: Gemini summary model is not configured."

    snippets: list[str] = []
    for idx, row in enumerate(search_rows[:10], start=1):
        snippets.append(
            f"[{idx}] Title: {row.get('title', '')}\n"
            f"URL: {row.get('url', '')}\n"
            f"Snippet: {row.get('snippet', '')}"
        )

    prompt = (
        f"You are a recruiting research assistant. Summarize recent web intelligence for {player_name} ({position}).\n"
        "Use only the provided sources. Do not invent facts.\n\n"
        "Output sections:\n"
        "1) Key facts\n"
        "2) Recruiting updates\n"
        "3) Source quality caveats\n\n"
        f"Sources:\n{chr(10).join(snippets)}"
    )

    try:
        response = llm.invoke(prompt)
        text = llm_response_to_text(response)
        return str(text).strip() if text else "Web summary returned empty output."
    except Exception as exc:
        return f"Web summary failed: {exc}"

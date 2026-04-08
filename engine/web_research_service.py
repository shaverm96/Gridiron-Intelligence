from __future__ import annotations

from datetime import date
import re
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


def _sanitize_model_summary_text(text: str) -> str:
    sanitized = str(text or "")
    sanitized = re.sub(r"```(?:json|javascript|html)?[\s\S]*?```", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"<script\b[^>]*>[\s\S]*?</script>", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"<iframe\b[^>]*>[\s\S]*?</iframe>", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"<a\b[^>]*>[\s\S]*?</a>", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"<[^>]+>", "", sanitized)

    lines: list[str] = []
    for line in sanitized.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("{", "}", "[", "]")):
            continue
        if re.match(r'^"[^"]+"\s*:\s*', stripped):
            continue
        lines.append(stripped)

    return "\n".join(lines).strip()


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
    reraise=True,
)
def _ddgs_text_search(ddgs_class: Any, query: str, max_results: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ddgs_class() as ddgs:
        for result in ddgs.text(query, max_results=max_results):
            rows.append(result)
    return rows


def _with_current_date_context(prompt_text: str) -> str:
    today_iso = date.today().isoformat()
    date_context = (
        "Date Context:\n"
        f"- Current date: {today_iso}\n"
        "- Treat this as today's date when reasoning about recency and up-to-date information.\n"
        "- If recency is uncertain, state the uncertainty explicitly.\n\n"
    )
    return f"{date_context}{str(prompt_text or '').strip()}"


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
        results = _ddgs_text_search(ddgs_class, query, max_results=max_results)
        for result in results:
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
        "Use only the provided sources. Do not invent facts.\n"
        "Output ONLY plain markdown bullet points (no HTML, no JSON, no links).\n\n"
        "Output sections:\n"
        "1) Key facts\n"
        "2) Recruiting updates\n"
        "3) Source quality caveats\n\n"
        f"Sources:\n{chr(10).join(snippets)}"
    )

    try:
        response = llm.invoke(_with_current_date_context(prompt))
        text = llm_response_to_text(response)
        cleaned = _sanitize_model_summary_text(str(text).strip()) if text else ""
        return cleaned or "Web summary returned empty output."
    except Exception as exc:
        return f"Web summary failed: {exc}"

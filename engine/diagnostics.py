from __future__ import annotations

from datetime import date
from typing import Any, Callable


def get_supabase_config_issues_data(
    config: dict[str, Any],
    config_sources: dict[str, Any],
    has_create_client: bool,
) -> list[str]:
    issues: list[str] = []
    if not has_create_client:
        issues.append("Python package 'supabase' is not installed (or failed to import).")
    if not config.get("SUPABASE_URL"):
        issues.append(f"SUPABASE_URL is missing (source: {config_sources.get('SUPABASE_URL', 'unknown')}).")
    if not config.get("SUPABASE_SERVICE_ROLE_KEY"):
        issues.append(
            "SUPABASE_SERVICE_ROLE_KEY is missing "
            f"(source: {config_sources.get('SUPABASE_SERVICE_ROLE_KEY', 'unknown')})."
        )
    return issues


def get_gemini_config_issues_data(
    config: dict[str, Any],
    config_sources: dict[str, Any],
    has_chat_model: bool,
) -> list[str]:
    issues: list[str] = []
    if not has_chat_model:
        issues.append("Python package 'langchain-google-genai' is not installed (or failed to import).")
    if not config.get("GEMINI_API_KEY"):
        issues.append(f"GEMINI_API_KEY is missing (source: {config_sources.get('GEMINI_API_KEY', 'unknown')}).")
    return issues


def run_one_click_diagnostics_data(
    config_sources: dict[str, Any],
    tables: dict[str, str],
    summary_model: str,
    get_supabase_config_issues: Callable[[], list[str]],
    get_supabase_client: Callable[[], Any],
    get_gemini_config_issues: Callable[[], list[str]],
    get_llm: Callable[..., Any],
    llm_response_to_text: Callable[[Any], str],
) -> dict[str, Any]:
    checks: list[dict[str, str]] = []

    def add_check(name: str, status: str, detail: str):
        checks.append({"name": name, "status": status, "detail": detail})

    add_check(
        "Config sources",
        "pass",
        (
            "SUPABASE_URL="
            f"{config_sources.get('SUPABASE_URL', 'unknown')}, "
            f"SUPABASE_SERVICE_ROLE_KEY={config_sources.get('SUPABASE_SERVICE_ROLE_KEY', 'unknown')}, "
            f"GEMINI_API_KEY={config_sources.get('GEMINI_API_KEY', 'unknown')}"
        ),
    )

    supabase_issues = get_supabase_config_issues()
    if supabase_issues:
        add_check("Supabase preflight", "fail", "; ".join(supabase_issues))
    else:
        try:
            sb = get_supabase_client()
            response = sb.table(tables["player_master"]).select("recruit_id").limit(1).execute()
            row_count = len(response.data or [])
            add_check(
                "Supabase connectivity",
                "pass",
                f"Connected and queried {tables['player_master']} (rows returned: {row_count}).",
            )
        except Exception as exc:
            add_check("Supabase connectivity", "fail", f"Query test failed: {exc}")

    gemini_issues = get_gemini_config_issues()
    if gemini_issues:
        add_check("Gemini preflight", "fail", "; ".join(gemini_issues))
    else:
        try:
            llm = get_llm(summary_model, temperature=0.0, max_output_tokens=20)
            if llm is None:
                add_check("Gemini connectivity", "fail", "Gemini client could not be created.")
            else:
                today_iso = date.today().isoformat()
                response = llm.invoke(f"Date Context: Current date is {today_iso}. Reply with exactly: OK")
                text = llm_response_to_text(response).strip()
                add_check("Gemini connectivity", "pass", f"Model responded: {text[:80] if text else 'empty response'}")
        except Exception as exc:
            add_check("Gemini connectivity", "fail", f"Invocation test failed: {exc}")

    overall = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {"overall": overall, "checks": checks}

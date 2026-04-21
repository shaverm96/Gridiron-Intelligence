from __future__ import annotations

from datetime import date
from typing import Any, Callable


def _normalize_model_name(value: Any) -> str:
    model_name = str(value or "").strip()
    alias_map = {
        "gemini-3.0-flash": "gemini-3-flash-preview",
        "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
    }
    return alias_map.get(model_name, model_name)


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


def get_model_pricing_issues_data(config: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    pricing = dict(config.get("MODEL_TOKEN_COSTS_PER_1M") or {})
    if not pricing:
        return ["MODEL_TOKEN_COSTS_PER_1M is missing or empty; cost telemetry cannot be estimated."]

    required_models = [
        _normalize_model_name(config.get("SUMMARY_MODEL")),
        _normalize_model_name(config.get("FINAL_MODEL")),
    ]

    missing_models: list[str] = []
    malformed_models: list[str] = []
    for model_name in required_models:
        if not model_name:
            continue
        model_rates = pricing.get(model_name)
        if not isinstance(model_rates, dict):
            missing_models.append(model_name)
            continue
        input_rate = model_rates.get("input")
        output_rate = model_rates.get("output")
        if not isinstance(input_rate, (int, float)) or not isinstance(output_rate, (int, float)):
            malformed_models.append(model_name)

    if missing_models:
        issues.append(
            "Missing pricing entries for active model(s): " + ", ".join(sorted(set(missing_models))) + "."
        )
    if malformed_models:
        issues.append(
            "Pricing entries must include numeric 'input' and 'output' rates for model(s): "
            + ", ".join(sorted(set(malformed_models)))
            + "."
        )
    return issues


def run_one_click_diagnostics_data(
    config: dict[str, Any],
    config_sources: dict[str, Any],
    tables: dict[str, str],
    summary_model: str,
    get_supabase_config_issues: Callable[[], list[str]],
    get_supabase_client: Callable[[], Any],
    get_gemini_config_issues: Callable[[], list[str]],
    get_model_pricing_issues: Callable[[], list[str]],
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

    pricing_issues = get_model_pricing_issues()
    if pricing_issues:
        add_check("Model pricing config", "fail", "; ".join(pricing_issues))
    else:
        summary_model_name = _normalize_model_name(config.get("SUMMARY_MODEL"))
        final_model_name = _normalize_model_name(config.get("FINAL_MODEL"))
        add_check(
            "Model pricing config",
            "pass",
            f"Pricing configured for active models: summary={summary_model_name}, final={final_model_name}.",
        )

    overall = "pass" if all(item["status"] == "pass" for item in checks) else "fail"
    return {"overall": overall, "checks": checks}

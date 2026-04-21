from __future__ import annotations

from typing import Callable
from typing import Any

from .agents import (
    cfbd_analyst_node,
    lead_delegator_node,
    lead_synthesizer_node,
    parallel_web_scout_node,
    recruiting_scout_node,
    team_scout_node,
    transfer_delegator_node,
    transfer_web_scout_node,
    transfer_synthesizer_node,
)
from .state import ScoutState

try:
    from langgraph.graph import END, StateGraph
except Exception:  # pragma: no cover
    END = "END"
    StateGraph = None


class SimpleScoutGraph:
    """Fallback sequential runner when langgraph is not available."""

    _SEQUENCE = [
        ("lead_delegator", lead_delegator_node),
        ("cfbd_analyst", cfbd_analyst_node),
        ("parallel_web_scout", parallel_web_scout_node),
        ("lead_synthesizer", lead_synthesizer_node),
    ]

    @staticmethod
    def _merge_update(state: ScoutState, update: ScoutState) -> ScoutState:
        merged = dict(state)
        for key, value in dict(update or {}).items():
            if key in {"citations", "errors", "trace_log"}:
                merged[key] = list(merged.get(key, [])) + list(value or [])
            elif key == "telemetry":
                merged[key] = dict(value or {})
            else:
                merged[key] = value
        return merged

    def invoke(self, state: ScoutState) -> ScoutState:
        for _, node_fn in self._SEQUENCE:
            update = node_fn(state)
            state = self._merge_update(state, update)
        return state

    def invoke_with_progress(
        self,
        state: ScoutState,
        progress_callback: Callable[[dict[str, str]], None] | None = None,
    ) -> ScoutState:
        if progress_callback is not None:
            progress_callback({"node": "workflow", "status": "started"})
        for node_name, node_fn in self._SEQUENCE:
            if progress_callback is not None:
                progress_callback({"node": node_name, "status": "running"})
            update = node_fn(state)
            state = self._merge_update(state, update)
            if progress_callback is not None:
                progress_callback({"node": node_name, "status": "completed"})
        if progress_callback is not None:
            progress_callback({"node": "workflow", "status": "completed"})
        return state


class SimpleStructuredWebGraph:
    """Fallback runner for structured web-scout-only workflow."""

    _SEQUENCE = [
        ("recruiting_scout", recruiting_scout_node),
        ("team_scout", team_scout_node),
    ]

    @staticmethod
    def _merge_update(state: ScoutState, update: ScoutState) -> ScoutState:
        merged = dict(state)
        for key, value in dict(update or {}).items():
            if key in {"citations", "errors", "trace_log"}:
                merged[key] = list(merged.get(key, [])) + list(value or [])
            elif key == "telemetry":
                previous = dict(merged.get("telemetry") or {})
                incoming = dict(value or {})
                previous_rows = list(previous.get("model_telemetry") or [])
                incoming_rows = list(incoming.get("model_telemetry") or [])
                merged_rows = previous_rows + incoming_rows
                previous_rollup = dict(previous.get("model_rollup") or {})
                incoming_rollup = dict(incoming.get("model_rollup") or {})
                merged[key] = {
                    "model_telemetry": merged_rows,
                    "model_rollup": {
                        "model_call_count": int(previous_rollup.get("model_call_count") or 0) + int(incoming_rollup.get("model_call_count") or 0),
                        "input_tokens": int(previous_rollup.get("input_tokens") or 0) + int(incoming_rollup.get("input_tokens") or 0),
                        "output_tokens": int(previous_rollup.get("output_tokens") or 0) + int(incoming_rollup.get("output_tokens") or 0),
                        "total_tokens": int(previous_rollup.get("total_tokens") or 0) + int(incoming_rollup.get("total_tokens") or 0),
                        "estimated_cost_usd": round(float(previous_rollup.get("estimated_cost_usd") or 0.0) + float(incoming_rollup.get("estimated_cost_usd") or 0.0), 8),
                        "latency_ms": int(previous_rollup.get("latency_ms") or 0) + int(incoming_rollup.get("latency_ms") or 0),
                    },
                }
            else:
                merged[key] = value
        return merged

    def invoke(self, state: ScoutState) -> ScoutState:
        for _, node_fn in self._SEQUENCE:
            update = node_fn(state)
            state = self._merge_update(state, update)
        return state

    def invoke_with_progress(
        self,
        state: ScoutState,
        progress_callback: Callable[[dict[str, str]], None] | None = None,
    ) -> ScoutState:
        if progress_callback is not None:
            progress_callback({"node": "workflow", "status": "started"})
        for node_name, node_fn in self._SEQUENCE:
            if progress_callback is not None:
                progress_callback({"node": node_name, "status": "running"})
            update = node_fn(state)
            state = self._merge_update(state, update)
            if progress_callback is not None:
                progress_callback({"node": node_name, "status": "completed"})
        if progress_callback is not None:
            progress_callback({"node": "workflow", "status": "completed"})
        return state


def get_scout_graph() -> Any:
    if StateGraph is None:
        return SimpleScoutGraph()

    workflow = StateGraph(ScoutState)

    workflow.add_node("lead_delegator", lead_delegator_node)
    workflow.add_node("cfbd_analyst", cfbd_analyst_node)
    workflow.add_node("parallel_web_scout", parallel_web_scout_node)
    workflow.add_node("lead_synthesizer", lead_synthesizer_node)

    workflow.set_entry_point("lead_delegator")

    workflow.add_edge("lead_delegator", "cfbd_analyst")
    workflow.add_edge("lead_delegator", "parallel_web_scout")

    workflow.add_edge("cfbd_analyst", "lead_synthesizer")
    workflow.add_edge("parallel_web_scout", "lead_synthesizer")
    workflow.add_edge("lead_synthesizer", END)

    return workflow.compile()


def get_structured_web_graph() -> Any:
    if StateGraph is None:
        return SimpleStructuredWebGraph()

    workflow = StateGraph(ScoutState)

    workflow.add_node("parallel_web_scout", parallel_web_scout_node)

    workflow.set_entry_point("parallel_web_scout")
    workflow.add_edge("parallel_web_scout", END)

    return workflow.compile()

class SimpleTransferChatGraph:
    """Fallback sequential runner for transfer chat when langgraph is not available."""

    _SEQUENCE = [
        ("transfer_delegator", transfer_delegator_node),
        ("transfer_web_scout", transfer_web_scout_node),
        ("transfer_synthesizer", transfer_synthesizer_node),
    ]

    @staticmethod
    def _merge_update(state: ScoutState, update: ScoutState) -> ScoutState:
        merged = dict(state)
        for key, value in dict(update or {}).items():
            if key in {"citations", "errors", "trace_log"}:
                merged[key] = list(merged.get(key, [])) + list(value or [])
            elif key == "telemetry":
                previous = dict(merged.get("telemetry") or {})
                incoming = dict(value or {})
                previous_rows = list(previous.get("model_telemetry") or [])
                incoming_rows = list(incoming.get("model_telemetry") or [])
                merged_rows = previous_rows + incoming_rows
                
                # Rollup is done at orchestration layer if needed, or we just collect rows
                merged[key] = {
                    "model_telemetry": merged_rows,
                }
            else:
                merged[key] = value
        return merged

    def invoke(self, state: ScoutState) -> ScoutState:
        for _, node_fn in self._SEQUENCE:
            update = node_fn(state)
            state = self._merge_update(state, update)
        return state

    def invoke_with_progress(
        self,
        state: ScoutState,
        progress_callback: Callable[[dict[str, str]], None] | None = None,
    ) -> ScoutState:
        if progress_callback is not None:
            progress_callback({"node": "workflow", "status": "started"})
        for node_name, node_fn in self._SEQUENCE:
            if progress_callback is not None:
                progress_callback({"node": node_name, "status": "running"})
            update = node_fn(state)
            state = self._merge_update(state, update)
            if progress_callback is not None:
                progress_callback({"node": node_name, "status": "completed"})
        if progress_callback is not None:
            progress_callback({"node": "workflow", "status": "completed"})
        return state


def get_transfer_chat_graph() -> Any:
    if StateGraph is None:
        return SimpleTransferChatGraph()

    workflow = StateGraph(ScoutState)

    workflow.add_node("transfer_delegator", transfer_delegator_node)
    workflow.add_node("transfer_web_scout", transfer_web_scout_node)
    workflow.add_node("transfer_synthesizer", transfer_synthesizer_node)

    workflow.set_entry_point("transfer_delegator")
    
    # Conditional logic
    def should_refresh_web(state: ScoutState) -> str:
        plan = state.get("transfer_delegator_plan") or {}
        if plan.get("should_refresh_web"):
            return "transfer_web_scout"
        return "transfer_synthesizer"

    workflow.add_conditional_edges(
        "transfer_delegator",
        should_refresh_web,
        {
            "transfer_web_scout": "transfer_web_scout",
            "transfer_synthesizer": "transfer_synthesizer",
        }
    )
    
    workflow.add_edge("transfer_web_scout", "transfer_synthesizer")
    workflow.add_edge("transfer_synthesizer", END)

    return workflow.compile()

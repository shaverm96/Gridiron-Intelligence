from __future__ import annotations

from typing import Callable
from typing import Any

from .agents import (
    cfbd_analyst_node,
    lead_delegator_node,
    lead_synthesizer_node,
    recruiting_scout_node,
    team_scout_node,
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
        ("recruiting_scout", recruiting_scout_node),
        ("team_scout", team_scout_node),
        ("lead_synthesizer", lead_synthesizer_node),
    ]

    @staticmethod
    def _merge_update(state: ScoutState, update: ScoutState) -> ScoutState:
        merged = dict(state)
        for key, value in dict(update or {}).items():
            if key in {"citations", "errors", "trace_log"}:
                merged[key] = list(merged.get(key, [])) + list(value or [])
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
    workflow.add_node("recruiting_scout", recruiting_scout_node)
    workflow.add_node("team_scout", team_scout_node)
    workflow.add_node("lead_synthesizer", lead_synthesizer_node)

    workflow.set_entry_point("lead_delegator")

    workflow.add_edge("lead_delegator", "cfbd_analyst")
    workflow.add_edge("lead_delegator", "recruiting_scout")
    workflow.add_edge("lead_delegator", "team_scout")

    workflow.add_edge("cfbd_analyst", "lead_synthesizer")
    workflow.add_edge("recruiting_scout", "lead_synthesizer")
    workflow.add_edge("team_scout", "lead_synthesizer")
    workflow.add_edge("lead_synthesizer", END)

    return workflow.compile()


def get_structured_web_graph() -> Any:
    if StateGraph is None:
        return SimpleStructuredWebGraph()

    workflow = StateGraph(ScoutState)

    workflow.add_node("recruiting_scout", recruiting_scout_node)
    workflow.add_node("team_scout", team_scout_node)

    workflow.set_entry_point("recruiting_scout")
    workflow.add_edge("recruiting_scout", "team_scout")
    workflow.add_edge("team_scout", END)

    return workflow.compile()

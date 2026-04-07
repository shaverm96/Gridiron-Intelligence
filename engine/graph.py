from __future__ import annotations

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

    def invoke(self, state: ScoutState) -> ScoutState:
        state = lead_delegator_node(state)
        state = cfbd_analyst_node(state)
        state = recruiting_scout_node(state)
        state = team_scout_node(state)
        state = lead_synthesizer_node(state)
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

    # Run worker nodes sequentially to avoid parallel state merge conflicts
    # on keys like citations/errors/trace_log when reducers are not defined.
    workflow.add_edge("lead_delegator", "cfbd_analyst")
    workflow.add_edge("cfbd_analyst", "recruiting_scout")
    workflow.add_edge("recruiting_scout", "team_scout")
    workflow.add_edge("team_scout", "lead_synthesizer")
    workflow.add_edge("lead_synthesizer", END)

    return workflow.compile()

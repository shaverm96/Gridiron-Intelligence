from __future__ import annotations

from typing import Any

from .agents import (
    chat_followup_node,
    comparables_node,
    sql_analyst_node,
    supervisor_node,
    synthesizer_node,
    vector_analyst_node,
    web_scout_node,
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
        state = supervisor_node(state)
        safety_counter = 0
        while state.get("next_step") != "end" and safety_counter < 12:
            safety_counter += 1
            step = state.get("next_step", "synthesizer")
            if step == "sql_analyst":
                state = sql_analyst_node(state)
            elif step == "web_scout":
                state = web_scout_node(state)
            elif step == "vector_analyst":
                state = vector_analyst_node(state)
            elif step == "comparables":
                state = comparables_node(state)
            elif step == "chat_followup":
                state = chat_followup_node(state)
            elif step == "synthesizer":
                state = synthesizer_node(state)
            else:
                state = synthesizer_node(state)

            if state.get("next_step") != "end":
                state = supervisor_node(state)

        return state


def _route_next_step(state: ScoutState) -> str:
    return state.get("next_step", "synthesizer")


def get_scout_graph() -> Any:
    if StateGraph is None:
        return SimpleScoutGraph()

    workflow = StateGraph(ScoutState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("sql_analyst", sql_analyst_node)
    workflow.add_node("web_scout", web_scout_node)
    workflow.add_node("vector_analyst", vector_analyst_node)
    workflow.add_node("comparables", comparables_node)
    workflow.add_node("chat_followup", chat_followup_node)
    workflow.add_node("synthesizer", synthesizer_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        _route_next_step,
        {
            "sql_analyst": "sql_analyst",
            "web_scout": "web_scout",
            "vector_analyst": "vector_analyst",
            "comparables": "comparables",
            "chat_followup": "chat_followup",
            "synthesizer": "synthesizer",
            "end": END,
        },
    )

    workflow.add_edge("sql_analyst", "supervisor")
    workflow.add_edge("web_scout", "supervisor")
    workflow.add_edge("vector_analyst", "supervisor")
    workflow.add_edge("comparables", "supervisor")
    workflow.add_edge("chat_followup", END)
    workflow.add_edge("synthesizer", END)

    return workflow.compile()

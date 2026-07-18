"""
graph.py
--------
Wires the nodes together into an actual LangGraph StateGraph.

Flow:

    planner -> search -> [conditional] -> synthesizer -> END
                              |
                              +--> (if search failed & retries left) -> search again
                              |
                              +--> (if search failed & no retries left) -> failure -> END

This conditional edge is the key thing that makes this a "graph" rather than
a straight-line script: the agent can react to a bad search result by
retrying, instead of just plowing ahead with empty data.
"""

from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import planner_node, search_node, synthesizer_node, failure_node, MAX_RETRIES


def route_after_search(state: AgentState) -> str:
    """
    Decides what happens after the search node runs.
    Returns the name of the next node to visit.
    """
    if not state.get("search_failed", False):
        return "synthesizer"

    if state.get("retry_count", 0) < MAX_RETRIES:
        return "search"  # try again

    return "failure"  # give up gracefully


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("search", search_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("failure", failure_node)

    graph.set_entry_point("planner")

    graph.add_edge("planner", "search")

    graph.add_conditional_edges(
        "search",
        route_after_search,
        {
            "synthesizer": "synthesizer",
            "search": "search",
            "failure": "failure",
        },
    )

    graph.add_edge("synthesizer", END)
    graph.add_edge("failure", END)

    return graph.compile()

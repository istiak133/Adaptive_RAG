"""Assemble the prep-flow state machine.

Flow:
    START
      → detect_mode          (cold-start? load mastery? pick difficulty?)
      → create_session       (DB row)
      → generate             (per-section allocation + per-seed LLM w/ retry)
      → record               (persist questions + topics)
      → simulate_and_score   (answers → mastery updates → score)
      → complete             (finalize session + metadata)
      → END
"""

from __future__ import annotations

from functools import partial
from typing import Callable

from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from src.graph import nodes
from src.graph.state import PrepState


def build_prep_graph(session: Session) -> Callable:
    """Construct and compile the prep-flow graph bound to a DB session.

    The session is closure-captured into each node via functools.partial so
    nodes stay pure (state in → state out) at the graph level.
    """
    builder: StateGraph = StateGraph(PrepState)

    builder.add_node("detect_mode",
                     partial(nodes.detect_mode_node, session=session))
    builder.add_node("create_session",
                     partial(nodes.create_session_node, session=session))
    builder.add_node("generate",
                     partial(nodes.generate_node, session=session))
    builder.add_node("record",
                     partial(nodes.record_node, session=session))
    builder.add_node("simulate_and_score",
                     partial(nodes.simulate_and_score_node, session=session))
    builder.add_node("complete",
                     partial(nodes.complete_node, session=session))

    builder.add_edge(START, "detect_mode")
    builder.add_conditional_edges(
        "detect_mode",
        nodes.route_after_detect,
        {"create_session": "create_session"},
    )
    builder.add_edge("create_session", "generate")
    builder.add_edge("generate", "record")
    builder.add_edge("record", "simulate_and_score")
    builder.add_edge("simulate_and_score", "complete")
    builder.add_edge("complete", END)

    return builder.compile()


def get_graph_diagram() -> str:
    """Return a Mermaid diagram of the compiled graph (for docs/viva)."""
    from langgraph.graph import StateGraph

    builder: StateGraph = StateGraph(PrepState)
    builder.add_node("detect_mode", lambda s: s)
    builder.add_node("create_session", lambda s: s)
    builder.add_node("generate", lambda s: s)
    builder.add_node("record", lambda s: s)
    builder.add_node("simulate_and_score", lambda s: s)
    builder.add_node("complete", lambda s: s)
    builder.add_edge(START, "detect_mode")
    builder.add_edge("detect_mode", "create_session")
    builder.add_edge("create_session", "generate")
    builder.add_edge("generate", "record")
    builder.add_edge("record", "simulate_and_score")
    builder.add_edge("simulate_and_score", "complete")
    builder.add_edge("complete", END)
    return builder.compile().get_graph().draw_mermaid()

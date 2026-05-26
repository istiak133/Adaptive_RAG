"""Assemble the prep-flow state machine.

Real conditional edges:
  • After detect_mode    → cold-start skips load_adaptive_context
  • After validate       → retry generate, or proceed to record

                    ┌───────────┐
                    │   START   │
                    └─────┬─────┘
                          ▼
                  ┌─────────────────┐
                  │   detect_mode   │
                  └────────┬────────┘
                           │
              is_cold_start? (conditional)
              ┌────────────┴────────────┐
              ▼                         ▼
       (yes / cold)              (no / adaptive)
              │                         │
              │              ┌──────────────────────┐
              │              │ load_adaptive_context│
              │              └──────────┬───────────┘
              ▼                         ▼
                  ┌─────────────────┐
                  │ create_session  │
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
            ┌────▶│    generate     │
            │     └────────┬────────┘
            │              ▼
            │     ┌─────────────────┐
            │     │    validate     │
            │     └────────┬────────┘
            │              │
        retry?  (conditional)
            │              │
            └──── yes ◀────┤ no
                           ▼
                  ┌─────────────────┐
                  │     record      │
                  └────────┬────────┘
                           ▼
                  ┌──────────────────────┐
                  │ simulate_and_score   │
                  └──────────┬───────────┘
                             ▼
                  ┌─────────────────┐
                  │    complete     │
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │      END        │
                  └─────────────────┘
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
    builder.add_node("load_adaptive_context",
                     partial(nodes.load_adaptive_context_node, session=session))
    builder.add_node("create_session",
                     partial(nodes.create_session_node, session=session))
    builder.add_node("generate",
                     partial(nodes.generate_node, session=session))
    builder.add_node("validate",
                     partial(nodes.validate_generation_node, session=session))
    builder.add_node("record",
                     partial(nodes.record_node, session=session))
    builder.add_node("simulate_and_score",
                     partial(nodes.simulate_and_score_node, session=session))
    builder.add_node("complete",
                     partial(nodes.complete_node, session=session))

    builder.add_edge(START, "detect_mode")

    # Real cold-vs-adaptive conditional branch
    builder.add_conditional_edges(
        "detect_mode",
        nodes.route_after_detect,
        {
            "create_session": "create_session",
            "load_adaptive_context": "load_adaptive_context",
        },
    )
    builder.add_edge("load_adaptive_context", "create_session")
    builder.add_edge("create_session", "generate")
    builder.add_edge("generate", "validate")

    # Real retry conditional branch
    builder.add_conditional_edges(
        "validate",
        nodes.route_after_validation,
        {
            "generate": "generate",  # loop back on missing MCQs
            "record": "record",
        },
    )
    builder.add_edge("record", "simulate_and_score")
    builder.add_edge("simulate_and_score", "complete")
    builder.add_edge("complete", END)

    return builder.compile()


def get_graph_diagram() -> str:
    """Mermaid diagram for docs/viva.

    Uses a parallel skeleton (without DB) so it can be rendered without
    a live SQLAlchemy session.
    """
    builder: StateGraph = StateGraph(PrepState)
    for n in (
        "detect_mode", "load_adaptive_context", "create_session",
        "generate", "validate", "record", "simulate_and_score", "complete",
    ):
        builder.add_node(n, lambda s: s)

    builder.add_edge(START, "detect_mode")
    builder.add_conditional_edges(
        "detect_mode",
        nodes.route_after_detect,
        {
            "create_session": "create_session",
            "load_adaptive_context": "load_adaptive_context",
        },
    )
    builder.add_edge("load_adaptive_context", "create_session")
    builder.add_edge("create_session", "generate")
    builder.add_edge("generate", "validate")
    builder.add_conditional_edges(
        "validate",
        nodes.route_after_validation,
        {"generate": "generate", "record": "record"},
    )
    builder.add_edge("record", "simulate_and_score")
    builder.add_edge("simulate_and_score", "complete")
    builder.add_edge("complete", END)

    return builder.compile().get_graph().draw_mermaid()

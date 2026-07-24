"""Layer 1 - node-level unit testing via checkpointer isolation.

The idea: seed the graph's state to look as though the upstream nodes have
*already* run (using ``update_state(..., as_node=...)``), then execute exactly
one more node and assert on its output. This is a true unit test - neither test
below depends on the upstream node's real logic being correct, only on the one
node under test.

To run a single node we compile the graph with an ``interrupt_after`` on the
node we want, so execution pauses the moment that node finishes and never
reaches the following node. In current LangGraph, ``interrupt_before`` /
``interrupt_after`` are static and passed to ``compile()`` (not to ``invoke``),
which is the form used here.
"""

from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from src.graph import builder


def _fresh_config():
    """A config with a unique thread_id so every test is fully isolated."""
    return {"configurable": {"thread_id": str(uuid4())}}


def test_fetch_account_context_in_isolation():
    # Compile a private graph that pauses right after fetch_account_context.
    graph = builder.compile(
        checkpointer=InMemorySaver(),
        interrupt_after=["fetch_account_context"],
    )
    config = _fresh_config()

    # Seed state as if classify_ticket already produced a classification. We do
    # NOT run classify_ticket's real logic - we assert the values directly.
    graph.update_state(
        config,
        values={
            "ticket_text": "seeded ticket",
            "account_id": "ACC-1001",
            "category": "billing_dispute",
            "priority": "high",
        },
        as_node="classify_ticket",
    )

    # Resume: only fetch_account_context runs, then execution interrupts.
    graph.invoke(None, config)

    values = graph.get_state(config).values
    # fetch_account_context ran and populated account_context...
    assert "account_context" in values
    assert values["account_context"]["account_id"] == "ACC-1001"
    assert values["account_context"]["plan"] == "enterprise"
    # ...and draft_response never ran, so there is no draft yet.
    assert "draft" not in values


def test_draft_response_in_isolation():
    # Same pattern, one node further downstream: pause after draft_response.
    graph = builder.compile(
        checkpointer=InMemorySaver(),
        interrupt_after=["draft_response"],
    )
    config = _fresh_config()

    # Seed state as if fetch_account_context already ran, with a distinctive
    # account_context we can look for in the generated draft.
    graph.update_state(
        config,
        values={
            "ticket_text": "seeded ticket",
            "category": "billing_dispute",
            "priority": "high",
            "account_context": {"account_id": "ACC-1001", "plan": "enterprise"},
        },
        as_node="fetch_account_context",
    )

    # Resume: only draft_response runs, then execution interrupts (the terminal
    # escalate/resolve node never runs).
    graph.invoke(None, config)

    values = graph.get_state(config).values
    assert "draft" in values
    # The draft references the seeded account_context, proving draft_response
    # consumed exactly the state we injected.
    assert "enterprise" in values["draft"]
    # The terminal node has not run, so status is still unset.
    assert "status" not in values

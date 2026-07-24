"""Layer 2 - trajectory / route assertions via get_state_history.

Instead of only checking the final answer, we assert on the *path* the graph
took: which nodes ran, in what order. We reconstruct that path from the
checkpoint history.

Note on the current LangGraph API: ``get_state_history`` yields one
``StateSnapshot`` per super-step in reverse-chronological order. In this
version the per-node ``writes`` map is no longer carried in snapshot metadata,
so we reconstruct the executed order from each snapshot's ``.next`` field - the
node that was about to run at that checkpoint. Reading those in chronological
order (skipping the initial ``__start__`` input snapshot and the final snapshot
whose ``.next`` is empty) yields the exact sequence of executed nodes.

Because the nodes in src/graph.py are deterministic stand-ins (no real model
call), this trajectory is stable across repeated runs, which is what makes an
exact list-equality assertion appropriate here.
"""

from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver

from src.graph import builder


def _fresh_config():
    return {"configurable": {"thread_id": str(uuid4())}}


def _executed_trajectory(graph, config):
    """Reconstruct the chronological list of executed nodes from history."""
    history = list(graph.get_state_history(config))  # reverse-chronological
    trajectory = []
    for snapshot in reversed(history):  # walk chronologically
        # Each snapshot's `.next` is the node about to run at that checkpoint.
        # Skip the input snapshot (next == ('__start__',)) and the final
        # snapshot (next == ()).
        if len(snapshot.next) == 1 and snapshot.next[0] != "__start__":
            trajectory.append(snapshot.next[0])
    return trajectory


def test_billing_dispute_trajectory_escalates():
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _fresh_config()

    graph.invoke(
        {
            "ticket_text": "I was double charged, please refund my invoice ASAP.",
            "account_id": "ACC-1001",
        },
        config,
    )

    assert _executed_trajectory(graph, config) == [
        "classify_ticket",
        "fetch_account_context",
        "draft_response",
        "escalate",
    ]


def test_routine_ticket_trajectory_resolves():
    graph = builder.compile(checkpointer=InMemorySaver())
    config = _fresh_config()

    graph.invoke(
        {
            "ticket_text": "How do I change the email address on my account?",
            "account_id": "ACC-1002",
        },
        config,
    )

    assert _executed_trajectory(graph, config) == [
        "classify_ticket",
        "fetch_account_context",
        "draft_response",
        "resolve",
    ]

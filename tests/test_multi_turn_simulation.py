"""Layer 3 - multi-turn simulation over a single thread_id.

A support ticket is a conversation, not a one-shot call. Here we reuse ONE
config (one ``thread_id``) across three sequential ``invoke`` calls so the
checkpointer threads state from turn to turn, exactly as a real multi-turn
session would.

We assert on *invariants* rather than exact wording. Free text (the ``draft``)
is intentionally not asserted verbatim: if a scenario ever needed to check the
semantic meaning of generated prose, an LLM-as-judge would be the right tool -
reserved for that case only. And if these nodes called a real LLM instead of
the deterministic stand-ins used here, pinning ``temperature=0`` would be the
equivalent move to keep the structured fields stable enough to assert on.
"""

from langgraph.checkpoint.memory import InMemorySaver

from src.graph import builder


def test_multi_turn_ticket_thread_invariants():
    graph = builder.compile(checkpointer=InMemorySaver())
    # ONE config reused across every turn: the whole point of the test.
    config = {"configurable": {"thread_id": "support-ticket-42"}}

    # Turn 1: an ordinary question with no account attached.
    graph.invoke(
        {"ticket_text": "I have a general question about my account settings."},
        config,
    )

    # Turn 2: the customer introduces an account_id and escalating urgency.
    account_id = "ACC-1001"
    graph.invoke(
        {
            "ticket_text": "Actually I was overcharged and need a refund urgently!",
            "account_id": account_id,
        },
        config,
    )

    # Turn 3: a follow-up nudge on the same thread.
    graph.invoke(
        {"ticket_text": "Any update on this? I'll cancel otherwise."},
        config,
    )

    final = graph.get_state(config).values

    # Invariant 1: the ticket reaches a valid terminal status.
    assert final["status"] in {"escalated", "resolved"}

    # Invariant 2: the escalate path is counted at most once per thread, even
    # though routing chose "escalate" on more than one turn. The idempotent
    # guard in the escalate node enforces this bound.
    assert final.get("escalation_count", 0) <= 1

    # Invariant 3: the account_id introduced in turn 2 survives unchanged into
    # the final state - state persists verbatim across turns on one thread.
    assert final["account_id"] == account_id

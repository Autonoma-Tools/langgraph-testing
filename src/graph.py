"""Support-ticket triage graph built with LangGraph.

The graph classifies an incoming support ticket, fetches deterministic
"account context", drafts a response, and then either escalates or resolves
the ticket. Every node is a *deterministic stand-in* for what would, in a real
deployment, be a model call. Because the nodes never touch a network or an LLM,
the whole graph runs with zero API keys and produces byte-stable trajectories,
which is exactly what makes the accompanying pytest suite able to assert on
exact node sequences.

Run it directly to see a full triage in action:

    python src/graph.py
"""

from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class TicketState(TypedDict, total=False):
    """State channels threaded through the triage graph.

    ``total=False`` lets tests seed a partial state (e.g. pretend only
    ``classify_ticket`` has run) without having to populate every key.
    """

    ticket_text: str
    account_id: str
    category: str
    priority: str
    account_context: dict
    draft: str
    status: str
    escalation_count: int


# A tiny deterministic "database" of accounts. In production this would be a
# real lookup against a CRM or billing system.
MOCK_ACCOUNTS = {
    "ACC-1001": {"plan": "enterprise", "mrr": 4200, "tenure_months": 26},
    "ACC-1002": {"plan": "starter", "mrr": 49, "tenure_months": 3},
}
DEFAULT_ACCOUNT = {"plan": "unknown", "mrr": 0, "tenure_months": 0}

# Keyword heuristics standing in for a real classifier.
BILLING_KEYWORDS = ("refund", "charge", "invoice", "billing", "overcharged", "double charged")
URGENT_KEYWORDS = ("urgent", "asap", "immediately", "escalate", "angry", "cancel")


def classify_ticket(state: TicketState) -> dict:
    """Assign a category and priority from the ticket text.

    A real deployment would replace this body with an LLM call pinned to
    ``temperature=0`` (e.g. a structured-output classification prompt). The
    keyword heuristic here is a deterministic stand-in so the trajectory is
    stable and no API key is required.
    """
    text = state.get("ticket_text", "").lower()

    if any(word in text for word in BILLING_KEYWORDS):
        category = "billing_dispute"
    else:
        category = "general"

    if category == "billing_dispute" or any(word in text for word in URGENT_KEYWORDS):
        priority = "high"
    else:
        priority = "normal"

    return {"category": category, "priority": priority}


def fetch_account_context(state: TicketState) -> dict:
    """Deterministically look up mock account data for the ticket's account."""
    account_id = state.get("account_id", "")
    context = dict(MOCK_ACCOUNTS.get(account_id, DEFAULT_ACCOUNT))
    context["account_id"] = account_id
    return {"account_context": context}


def draft_response(state: TicketState) -> dict:
    """Build a draft reply that references the category and account context."""
    category = state.get("category", "general")
    context = state.get("account_context", {}) or {}
    plan = context.get("plan", "unknown")

    if category == "billing_dispute":
        body = (
            f"Thanks for reaching out about your billing concern. As a {plan} "
            "customer, your case is being reviewed by our billing specialists."
        )
    else:
        body = (
            f"Thanks for contacting support. As a {plan} customer, here is some "
            "guidance to help resolve your question."
        )

    return {"draft": body}


def route_after_draft(state: TicketState) -> str:
    """Decide whether the drafted ticket should escalate or resolve."""
    if state.get("category") == "billing_dispute" or state.get("priority") == "high":
        return "escalate"
    return "resolve"


def escalate(state: TicketState) -> dict:
    """Terminal node: mark the ticket escalated.

    Idempotent by design: a ticket thread can be invoked multiple times (a
    multi-turn conversation reuses one ``thread_id``), but the escalate path
    must never be counted more than once for the same thread. If the ticket is
    already escalated we leave ``escalation_count`` untouched.
    """
    if state.get("status") == "escalated":
        return {"status": "escalated"}
    return {
        "status": "escalated",
        "escalation_count": state.get("escalation_count", 0) + 1,
    }


def resolve(state: TicketState) -> dict:
    """Terminal node: mark the ticket resolved."""
    return {"status": "resolved"}


def build_graph() -> StateGraph:
    """Assemble and return the *uncompiled* graph builder.

    Tests import this so they can compile a fresh graph with their own
    checkpointer (and their own interrupt configuration) per test, keeping
    every test fully isolated.
    """
    builder = StateGraph(TicketState)

    builder.add_node("classify_ticket", classify_ticket)
    builder.add_node("fetch_account_context", fetch_account_context)
    builder.add_node("draft_response", draft_response)
    builder.add_node("escalate", escalate)
    builder.add_node("resolve", resolve)

    builder.add_edge(START, "classify_ticket")
    builder.add_edge("classify_ticket", "fetch_account_context")
    builder.add_edge("fetch_account_context", "draft_response")
    builder.add_conditional_edges(
        "draft_response",
        route_after_draft,
        {"escalate": "escalate", "resolve": "resolve"},
    )
    builder.add_edge("escalate", END)
    builder.add_edge("resolve", END)

    return builder


# Uncompiled builder, exported for tests.
builder = build_graph()

# Default compiled graph with an in-memory checkpointer, ready to run.
graph = builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    import uuid

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    sample_ticket = {
        "ticket_text": "I was double charged on my last invoice, please refund ASAP.",
        "account_id": "ACC-1001",
    }

    final_state = graph.invoke(sample_ticket, config)

    print("=== Support-ticket triage: final state ===")
    for key in (
        "ticket_text",
        "account_id",
        "category",
        "priority",
        "account_context",
        "draft",
        "status",
        "escalation_count",
    ):
        print(f"{key:>17}: {final_state.get(key)}")

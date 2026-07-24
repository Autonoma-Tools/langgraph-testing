"""Layer 4 - time-travel debugging via checkpoint replay.

This is a standalone script (not a pytest file). It shows how to:

  1. Run a ticket to completion and inspect the final state.
  2. Walk the full checkpoint history.
  3. Rewind to an earlier checkpoint, change one field to fork the timeline,
     and replay forward from that point - watching the outcome change.

Run it directly (no API keys required):

    python scripts/debug_replay.py
"""

import os
import sys
from uuid import uuid4

# Make the repo root importable so `from src.graph import ...` works when this
# script is run directly as `python scripts/debug_replay.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.checkpoint.memory import InMemorySaver

from src.graph import builder


def main():
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid4())}}

    # 1. Run a routine ticket that ends up resolved.
    graph.invoke(
        {
            "ticket_text": "How do I change the email address on my account?",
            "account_id": "ACC-1002",
        },
        config,
    )

    current = graph.get_state(config)
    print("=== Original run ===")
    print("final status :", current.values["status"])
    print("state .next  :", current.next)  # empty -> the run is complete
    print()

    # 2. Walk the checkpoint history, newest first, noting which node was about
    #    to run at each checkpoint and its checkpoint_id.
    print("=== Checkpoint history (newest first) ===")
    target_config = None
    for snapshot in graph.get_state_history(config):
        checkpoint_id = snapshot.config["configurable"]["checkpoint_id"]
        about_to_run = snapshot.next or ("<end>",)
        print(f"checkpoint {checkpoint_id}  next={about_to_run}")
        # 3. Find the checkpoint right before draft_response ran.
        if "draft_response" in snapshot.next:
            target_config = snapshot.config
    print()

    if target_config is None:
        raise RuntimeError("Could not find a checkpoint before draft_response")

    # 4. Fork the timeline: rewind to that checkpoint and reclassify the ticket
    #    as a billing dispute, then replay forward from that point.
    forked_config = graph.update_state(
        target_config,
        values={"category": "billing_dispute"},
    )
    forked_final = graph.invoke(None, forked_config)

    print("=== Forked replay (category changed to billing_dispute) ===")
    print("forked status:", forked_final["status"])
    print(f"Original resolved -> forked {forked_final['status']} by changing one")
    print("field at an earlier checkpoint and replaying forward.")

    # If any node here were itself a compiled subgraph, graph.get_state(config,
    # subgraphs=True) is the call to reach for to inspect nested state. This
    # example uses no subgraphs, so we don't need it.


if __name__ == "__main__":
    main()

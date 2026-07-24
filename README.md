# LangGraph Testing and Debugging: A Practical Guide

A runnable LangGraph agent (a support-ticket triage graph with `classify`, `account-context`, and `draft-response` nodes) plus a full pytest suite demonstrating node-level unit testing via checkpointer isolation, trajectory/route assertions via `get_state_history`, multi-turn simulation over a single `thread_id`, and a standalone time-travel debugging script that forks a replay from an earlier checkpoint. Everything runs with **zero API keys**: the graph's nodes are deterministic stand-ins for real model calls, so trajectories are stable and safe to assert on exactly.

> Companion code for the Autonoma blog post: **[LangGraph Testing and Debugging: A Practical Guide](https://getautonoma.com/blog/langgraph-testing)**

## Requirements

- Python 3.10+
- No API keys. Every node is a deterministic stand-in for a real model call, so the whole suite runs offline.

Install dependencies:

```bash
git clone https://github.com/Autonoma-Tools/langgraph-testing.git
cd langgraph-testing
pip install -r requirements.txt
```

## The graph under test

`src/graph.py` defines a support-ticket triage graph:

```
START -> classify_ticket -> fetch_account_context -> draft_response -> (route)
                                                                        |-> escalate -> END
                                                                        |-> resolve  -> END
```

- **classify_ticket** — sets `category` (`billing_dispute` / `general`) and `priority` (`high` / `normal`) from keyword heuristics. In a real deployment this node would call an LLM pinned to `temperature=0`; the heuristic is a deterministic stand-in.
- **fetch_account_context** — looks up a mock account dict and populates `account_context`.
- **draft_response** — builds a draft that references the category and account context.
- **route_after_draft** — routes to `escalate` if the ticket is a billing dispute or high priority, otherwise `resolve`.
- **escalate / resolve** — terminal nodes that set the final `status`. `escalate` is idempotent, so re-invoking the same thread never double-counts an escalation.

The graph is compiled with an `InMemorySaver` checkpointer, and the uncompiled `builder` is exported so tests can compile a fresh, isolated graph with their own checkpointer.

Run it to see a full triage:

```bash
python src/graph.py
```

## The four testing layers

Each layer answers a different question about an agent graph, and each has its own run command.

### 1. Node-level unit testing via checkpointer isolation

**File:** `tests/test_node_isolation.py`

Seed the graph's state to look as though the upstream nodes already ran (`update_state(..., as_node=...)`), then execute exactly one more node using an `interrupt_after` compiled into the graph, and assert on that single node's output. It's a true unit test: the node under test is exercised without depending on the correctness of any upstream node.

### 2. Trajectory / route assertions via `get_state_history`

**File:** `tests/test_trajectory.py`

Assert on the *path* the graph took, not just the final answer. The executed node sequence is reconstructed from the checkpoint history (`get_state_history`) and compared for exact equality (e.g. `['classify_ticket', 'fetch_account_context', 'draft_response', 'escalate']`). Exact-equality assertions are appropriate precisely because the deterministic nodes produce a stable trajectory.

### 3. Multi-turn simulation over a single `thread_id`

**File:** `tests/test_multi_turn_simulation.py`

Reuse one config (one `thread_id`) across three sequential `invoke` calls so the checkpointer threads state from turn to turn. Assert on invariants (valid terminal status, escalation counted at most once, a fact introduced mid-conversation survives verbatim) rather than exact free-text wording.

Run all three test layers:

```bash
pytest tests/
```

### 4. Time-travel debugging via checkpoint replay

**File:** `scripts/debug_replay.py`

A standalone script (not a pytest file). It runs a ticket to completion, walks the checkpoint history, rewinds to the checkpoint right before `draft_response` ran, changes one field to fork the timeline, and replays forward, showing the outcome flip from `resolved` to `escalated`.

```bash
python scripts/debug_replay.py
```

## Project structure

```
langgraph-testing/
├── src/
│   └── graph.py                        # the triage graph (deterministic nodes)
├── tests/
│   ├── test_node_isolation.py          # layer 1: node unit tests
│   ├── test_trajectory.py              # layer 2: trajectory/route assertions
│   └── test_multi_turn_simulation.py   # layer 3: multi-turn simulation
├── scripts/
│   └── debug_replay.py                 # layer 4: time-travel debugging
├── conftest.py                         # puts repo root on sys.path for pytest
└── requirements.txt
```

## About

This repository is maintained by [Autonoma](https://getautonoma.com) as reference material for the linked blog post. Autonoma builds autonomous AI agents that plan, execute, and maintain end-to-end tests directly from your codebase.

If something here is wrong, out of date, or unclear, please [open an issue](https://github.com/Autonoma-Tools/langgraph-testing/issues/new).

## License

Released under the [MIT License](./LICENSE) © 2026 Autonoma Labs.

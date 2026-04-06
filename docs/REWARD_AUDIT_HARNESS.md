# Reward Audit Harness

This project now includes a lightweight reward-audit script:

- [`reward_audit.py`](../reward_audit.py)

Its purpose is to evaluate reward quality *before* running LLM or RL training.

## Why this exists

The grader defines what "winning" means at the end of an episode. The dense
reward should make progress toward that visible throughout the trajectory.

Before tuning prompts or training agents, we want to answer:

- does cumulative reward rank policies similarly to the grader?
- can obviously bad policies still earn decent reward?
- are we over-rewarding passive survival or under-rewarding decisive action?

## Probe Policies

The harness runs a small set of fixed strategies:

- `no_op`
- `aggressive_all_in`
- `ground_only`
- `structure_first`
- `cost_aware`

These are not meant to be strong agents. They are intentionally different so we
can see whether reward and grader prefer the same kinds of behavior.

## What it measures

For each `(task, seed, policy)` episode it records:

- cumulative dense reward
- final grader score
- steps taken
- containment outcome
- structures lost
- burned cells
- reward breakdown by category

It also computes:

- overall reward / grader correlation
- pairwise ranking agreement between reward order and grader order
- scenarios where reward and grader disagree on the top policy

## Usage

From the repo root:

```bash
.\.venv\Scripts\python.exe reward_audit.py
```

Useful variants:

```bash
.\.venv\Scripts\python.exe reward_audit.py --seed-sweep 1
.\.venv\Scripts\python.exe reward_audit.py --tasks easy medium
.\.venv\Scripts\python.exe reward_audit.py --policies no_op structure_first cost_aware
.\.venv\Scripts\python.exe reward_audit.py --json-out -
```

By default the script writes a machine-readable report to:

- `reward_audit_report.json`

## How to use the results

The reward is in a good place when:

- `no_op` is reliably near the bottom
- structure-saving strategies outrank land-only preservation when assets are at risk
- reward and grader mostly agree on the top policy
- no cheap policy exploits a shaping term to look better than it should

If reward and grader disagree often, reward shaping needs adjustment before
doing LLM or RL experiments.

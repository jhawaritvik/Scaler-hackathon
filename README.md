---
title: Wildfire Resource Allocation Environment
emoji: "\U0001F525"
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# Wildfire Resource Allocation Environment

This repository contains an OpenEnv-compatible wildfire incident-command
environment. An AI agent acts as an incident commander and learns to dispatch
limited firefighting resources across a terrain grid to contain wildfire spread
and protect structures.

The actual environment package lives in `wildfire_env/`. The detailed package
README is at `wildfire_env/README.md`.

## Hackathon Theme Fit

**Primary theme: #3.1 World Modeling / Professional Tasks**

This environment is best framed as a professional incident-command simulator:

- the agent operates inside a dynamic partially observable world
- actions have delayed physical consequences through logistics, travel, and fire spread
- the agent must maintain consistent internal state across multiple turns
- success depends on tool-like orchestration of heterogeneous resources, not one-shot text answers

**Secondary theme: #2 Long-Horizon Planning & Instruction Following**

The environment also fits long-horizon planning because the policy has to:

- commit scarce assets early without knowing the full fire picture
- recover from weak opening moves after delayed ignitions and warmup spread
- track unit availability, return-to-service timing, and structure priorities over 15-25 steps

This submission is a single-agent world-modeling environment with long-horizon planning — not a multi-agent or self-play arena.

## Submission Links

- **Live Space:** [Chunchunmaru-101/wildfire-env](https://huggingface.co/spaces/Chunchunmaru-101/wildfire-env)
- **Live app:** [chunchunmaru-101-wildfire-env.hf.space](https://chunchunmaru-101-wildfire-env.hf.space)
- **Training pipeline:** [`train_grpo.py`](./train_grpo.py) (also runnable via [`notebooks/wildfire_grpo_minimal_colab.ipynb`](./notebooks/wildfire_grpo_minimal_colab.ipynb))
- **Reward-hacking audit:** [`reward_audit.py`](./reward_audit.py) + [`reward_audit.json`](./reward_audit.json) (84 fixed-seed episodes, no exploit-like policies flagged)
- **Submission artifact helpers:** [`plot_training_curves.py`](./plot_training_curves.py), [`submission_check.py`](./submission_check.py), and [`submission_artifacts/README.md`](./submission_artifacts/README.md)
- **Writeup / demo video / slides:** _to be linked after training completes_
- **Training reward & loss plots:** _to be added after training completes_
- **Trained-vs-baseline comparison:** _to be added after training completes_

## What's in this repo

- OpenEnv-compliant environment with a valid [`openenv.yaml`](./openenv.yaml) manifest and `openenv-core[core]>=0.2.3` pin
- FastAPI server exposing `/reset`, `/step`, `/state`, `/grader`, `/baseline`, `/tasks`, `/ws`, `/demo`
- Deterministic heuristic baseline and deterministic grader for reproducible evaluation
- Hand-rolled multi-turn GRPO training pipeline ([`train_grpo.py`](./train_grpo.py)) — Qwen3-4B-Instruct-2507 + 4-bit QLoRA (Unsloth) + XGrammar-constrained decoding
- GPU smoke test ([`smoke_test.py`](./smoke_test.py)) with NaN/Inf guards and preflight dependency checks
- Reward-hacking audit harness ([`reward_audit.py`](./reward_audit.py)) running a 7-policy bank against the dense reward and grader, with rank-correlation reporting and exploit-like-policy flagging
- Training plot/export helper ([`plot_training_curves.py`](./plot_training_curves.py)) that turns `log.jsonl` into judge-friendly SVG reward/loss curves with no extra plotting dependency
- Submission readiness checker ([`submission_check.py`](./submission_check.py)) for the final hackathon packaging pass
- Regression test suite ([`tests/test_regressions.py`](./tests/test_regressions.py))
- Colab-friendly notebook scaffold ([`notebooks/wildfire_grpo_minimal_colab.ipynb`](./notebooks/wildfire_grpo_minimal_colab.ipynb))

## Why this environment

This is a real-world resource allocation task, not a toy game. The agent must
make sequential decisions under uncertainty and delay:

- choose which unit to send
- choose where to send it
- choose the mission geometry and commitment
- account for limited fleet size, travel time, and return-to-service timing

## Action Space

The action is a batch of assignments for the current decision interval.

- `plan`: optional one-sentence tactical scratchpad, capped at 160 chars
- `assignments`: list of resource assignments
- each assignment includes `unit_id`, `mission_type`, `target`,
  `commitment_steps`, and optional `drop_configuration`

Supported mission types:

- `direct_attack`
- `line_construction`
- `wet_line`
- `water_drop`
- `retardant_drop`
- `point_protection`
- `backfire`
- `staging`

Supported target types:

- `point`
- `line`
- `area`
- `polygon`
- `structure`

## Observation Space

Each observation includes:

- task goal and episode progress, plus a natural-language `action_guide`
- fire cells and heat warnings — **filtered by fog-of-war** sensor range
- structures with priority and status
- current weather (temp, humidity, wind) and a **2-step Spot Forecast**
  (NWCG PMS 425) with diurnal temperature/humidity and wind trend
- full fleet status, active missions, and outposts
- `visible_cell_count` and `fog_of_war_active` flags
- dense step reward plus action/error summaries

### Partial observability (fog of war)

Only cells within sensor range of a deployed resource are fully visible;
the rest show as `?` in the grid string. Per-resource sensor radii
(Manhattan cells) are calibrated from PMC 2017 FLIR reconnaissance data:

| resource | radius | radius (standby) |
|---|---:|---:|
| helicopter | 6 | 2 |
| airtanker | 5 | 2 |
| crew / smokejumper | 4 | 2 |
| engine / dozer | 3 | 2 |

`burning_cells` (total count) remains unmasked — the agent knows the fire
exists but must deploy air assets to localise it. This is the core
incident-commander tension the environment models.

### LCES safety doctrine

Ground-resource assignments (`direct_attack`, `line_construction`,
`backfire`, `wet_line`) are checked against NWCG LCES (Lookouts /
Communications / Escape Routes / Safety Zones). If a burning cell is
immediately adjacent to the target and no escape cell (firebreak, water,
burned, suppressed) is reachable within 2 cells, a `-0.03` penalty is
applied per Butler & Cohen (1998) safety-zone criterion.

## Tasks

Three graded difficulty levels with cellular-automaton fire spread
(Alexandridis 2008, Rothermel 1972):

| task | grid | max steps | ignitions | structures | crews | helicopter | airtanker | wind (km/h) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| easy | 15×15 | 20 | 1 | 2-3 (P1) | 3-5 | 1-2 | 0-1 | 5-12 |
| medium | 15×15 | 15 | 2 + 1 delayed | 3-4 (≤P2) | 2-4 | 1-2 | 0-1 | 10-20 |
| hard | 25×25 | 25 | 2 + 1-2 delayed | 3-5 (≤P3) | 2-3 | 1 | 0-1 | 15-25 |

Each task has a deterministic grader that returns `0.0` to `1.0` using:

- `60%` structure protection (priority-weighted)
- `30%` area preservation
- `10%` containment efficiency

## Reward

Reward signals are grid-size normalised on structural penalties (factor
`gn = min(1, (15/size)^1.5)` — 1.00 on 15×15, ~0.46 on 25×25) so GRPO
group variance stays meaningful across difficulty levels.

| Signal | Amount | Trigger |
|---|---:|---|
| `structure_burning` | `-0.12 × priority × gn` | Each step a structure cell is burning |
| `structure_lost` | `-0.50 × priority × gn` | Once when a structure transitions to burned |
| `structure_safe` | `+0.003 × priority` | Each step an intact structure remains under nearby threat |
| `cells_suppressed` | `+0.06` per cell | Each burning cell extinguished that step |
| `cells_protected` | `+0.0025` per cell | Each newly protected threatened unburned cell, capped at 12 cells per step |
| `containment` | `+0.018 × net cells` | Net burning-cell reduction, **gated on active suppression** (prevents passive-burnout exploit) |
| `active_fire_pressure` | `-0.005 × min(burning, 8)` | Each step while fire remains active |
| `fire_extinguished` | `+0.30` to `+0.70` | Once when all burning cells are out, scaled by containment speed |
| Mission dispatch cost | `-0.0008` to `-0.0112` | Per assignment, based on mission type |
| Resource dispatch surcharge | `-0.0004` to `-0.0030` | Additional per assignment, based on resource type |
| `idle_penalty` | `-0.005 × min(avail, 4)` | Empty `assignments[]` while fire is burning and units are available |
| LCES violation | `-0.03` | Hazardous ground assignment without a reachable safety zone |
| Invalid action | `-0.05` | Rejected assignment |
| Low-impact action | `-0.02` | Wasteful assignment with negligible effect |

### Anti-reward-hacking design

Per the hackathon guide on reward-hacking (§8), multiple independent
checks protect the training signal:

- **Format enforcement**: XGrammar-constrained decoding + Pydantic
  validation — malformed actions can't reach the environment.
- **Causal containment gate**: `containment` only pays when the agent
  actively suppressed at least one cell that tick — natural fuel
  burnout earns nothing.
- **No free no-ops**: `idle_penalty` scales with available units when
  the agent skips action while fire burns.
- **Single-use penalties**: `structure_lost` fires once per structure;
  `fire_extinguished` fires once per episode.
- **Duplicate-dispatch block**: assigning the same unit twice in a step
  triggers `INVALID_ACTION_PENALTY`.
- **Min-step guard**: episodes cannot terminate before 3-5 steps
  (`min_steps_before_early_end`) to prevent early-exit reward farming.

### Reward audit (`reward_audit.py` / `reward_audit.json`)

To empirically verify the dense shaped reward ranks policies the same way
the grader does — and that no exploit-class policy is over-rewarded — we run
a fixed 7-policy bank (`noop`, `heuristic`, `heuristic_ground_only`,
`heuristic_aerial_only`, `stage_all`, `invalid_duplicate`, `random_valid`)
on each difficulty and report rank-correlation metrics.

Results on the current HEAD (84 episodes across the fixed curriculum seed bank):

| Task | Policy Spearman | Policy Kendall | Episode Spearman | Exploit flags |
|---|---:|---:|---:|---|
| easy | 0.964 | 0.926 | 0.528 | none |
| medium | 0.927 | 0.823 | 0.898 | none |
| hard | 0.852 | 0.720 | 0.792 | none |

Exploit policies (`noop`, `stage_all`, `invalid_duplicate`) consistently
rank below the sane policies on every task and never outrank the heuristic on
the final grader, confirming the gating and duplicate-dispatch blocks work in
practice. Episode-level correlation is noisier because seed difficulty changes
absolute return scale; the more relevant signal here is policy-level ranking.
Full per-policy breakdown is in `reward_audit.json`.

Rerun with:

```bash
.\.venv\Scripts\python.exe reward_audit.py --json-out reward_audit.json
```

## Setup

Python:

```bash
.\.venv\Scripts\python.exe -m wildfire_env.test_env_smoke
.\.venv\Scripts\openenv.exe validate .
```

Run server locally:

```bash
cd wildfire_env
..\.venv\Scripts\python.exe -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Docker from repo root:

```bash
docker build -t wildfire-env .
docker run -p 8000:8000 wildfire-env
```

### Live viewer

A websocket-streaming demo at `/demo` renders episodes in real time:

```
http://localhost:8000/demo
```

The grid resizes automatically for hard-task 25×25 episodes.

## Baselines

Deterministic heuristic baseline via `/baseline` on the default seeded tasks:

| task | heuristic score | seed |
|---|---:|---:|
| easy | 0.3196 | 42 |
| medium | 0.3549 | 67 |
| hard | 0.0982 | 12 |

These values come from the current fog-of-war environment and reflect the
default single-seed `/baseline` endpoint, not an older multi-seed evaluation
run from before the environment changes.

LLM baseline script:

- root script: `inference.py`
- uses OpenAI client
- reads `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME`
- runs explicit seeded task episodes for reproducibility

## Training

A multi-turn GRPO pipeline lives in `train_grpo.py`. Stack:
Qwen3-4B-Instruct-2507 (4-bit QLoRA via Unsloth) + XGrammar-constrained decoding +
hand-rolled GRPO loop (TRL's GRPOTrainer is single-turn only; multi-turn
trajectory advantages require a custom loop).

Colab notebook scaffold:

- `notebooks/wildfire_grpo_minimal_colab.ipynb`

Install training extras with:

```bash
.\.venv\Scripts\python.exe -m pip install -e .[train]
```

The training path requires a CUDA GPU. On a CPU-only machine, `smoke_test.py`
now fails fast with a clear preflight error instead of a long stack trace.

## Submission Workflow

Once you have GPU access, this is the clean path to a final submission package:

```bash
.\.venv\Scripts\python.exe reward_audit.py --json-out reward_audit.json
.\.venv\Scripts\python.exe train_grpo.py
.\.venv\Scripts\python.exe plot_training_curves.py --log grpo_wildfire/log.jsonl --out-dir submission_artifacts
.\.venv\Scripts\python.exe eval_policy.py --untrained --output submission_artifacts/eval_untrained.json
.\.venv\Scripts\python.exe eval_policy.py --output submission_artifacts/eval_trained.json
.\.venv\Scripts\python.exe submission_check.py --strict
```

The helper outputs land in [`submission_artifacts/`](./submission_artifacts/),
which is where the final reward/loss plots and eval JSONs should be committed
before the final hackathon push.

## Results

_Populated after the main training run. Expected artifacts:_

- `submission_artifacts/training_reward_curve.svg`
- `submission_artifacts/training_loss_curve.svg`
- `submission_artifacts/training_summary.md`
- `submission_artifacts/eval_untrained.json`
- `submission_artifacts/eval_trained.json`
- short qualitative walkthrough: one hard-task episode where the trained
  policy saves a structure the heuristic loses

## Repository layout

- `inference.py` — OpenAI-client baseline agent script
- `Dockerfile` — Space container
- `openenv.yaml` — OpenEnv manifest
- `train_grpo.py` — multi-turn GRPO training pipeline
- `eval_policy.py` — deterministic held-out evaluation of a trained adapter
- `smoke_test.py` — 1-iter GRPO preflight
- `reward_audit.py` — reward-hacking audit harness
- `plot_training_curves.py` — SVG reward/loss plot generator for `log.jsonl`
- `submission_check.py` — final packaging checker for hackathon submission
- `tests/test_regressions.py` — regression tests
- `submission_artifacts/` — generated training plots, eval JSONs, and final evidence
- `notebooks/wildfire_grpo_minimal_colab.ipynb` — Colab scaffold
- `wildfire_env/` — environment package (see `wildfire_env/README.md` for
  internals)

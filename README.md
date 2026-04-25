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
- **Writeup / demo video / slides:** [`ENV_REVIEW.md`](./ENV_REVIEW.md) (interim technical writeup; replace with final public blog/video/slides URL)
- **Training reward & loss plots:** [`submission_artifacts/training_reward_curve.png`](./submission_artifacts/training_reward_curve.png) and [`submission_artifacts/training_loss_curve.png`](./submission_artifacts/training_loss_curve.png)
- **Trained-vs-baseline comparison:** [`submission_artifacts/eval_untrained.json`](./submission_artifacts/eval_untrained.json) and [`submission_artifacts/eval_trained.json`](./submission_artifacts/eval_trained.json)

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
| easy | 15×15 | 20 | 1-2 + 0-1 delayed | 3-4 (P1) | 2-4 | 1 | 0 | 8-15 |
| medium | 20×20 | 20 | 2 + 1-2 delayed | 4-5 (≤P2) | 3-4 | 1-2 | 0-1 | 12-22 |
| hard | 25×25 | 25 | 2 + 1-2 delayed | 3-5 (≤P3) | 3-4 | 1 | 0-1 | 15-24 |

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

A websocket-streaming demo at `/viewer` renders episodes in real time —
animated grid, structure priority labels, unit movement dots, wind compass,
2-step weather forecast, and a per-step action summary. The grid resizes
automatically for the 20×20 medium and 25×25 hard tasks.

```
http://localhost:8000/viewer                            # live heuristic stream
http://localhost:8000/viewer?replay=replays/foo.json    # replay a captured episode
```

For the submission demo video, use `capture_replay.py` to record a single
trained-policy episode (or heuristic baseline on the same seed for side-by-side
comparison):

```bash
# Trained LLM (GPU required) — captures the model's plan field per step too
python capture_replay.py --task hard --seed 9 --policy llm \
    --adapter grpo_wildfire/best_adapter_hard \
    --output replays/trained_hard_9.json

# Heuristic baseline on the same seed (no GPU)
python capture_replay.py --task hard --seed 9 --policy heuristic \
    --output replays/heuristic_hard_9.json
```

`GET /replays` returns an index of every JSON visible to the server under
`replays/`, `submission_artifacts/`, or `artifacts/`.

## Baselines

Deterministic heuristic baseline measured on five **held-out** seeds per task
(no overlap with the 16-seed training pool in `train_grpo.py`). All seeds were
selected from a 0-79 sweep and filtered to heuristic grader in 0.2-0.85
(signal-rich range; excludes dead-zone and trivially-won scenarios).

| task | mean | min | max | held-out eval seeds |
|---|---:|---:|---:|---|
| easy | 0.6970 | 0.58 | 0.82 | 11, 18, 25, 56, 76 |
| medium | 0.5500 | 0.35 | 0.66 | 1, 7, 28, 61, 69 |
| hard | 0.5800 | 0.41 | 0.65 | 9, 28, 32, 61, 75 |

These numbers are produced by the structure-aware heuristic in
`wildfire_env/server/app.py` (`_heuristic_action`). The heuristic ranks
fires by structure proximity and matches resources to missions greedily; it
plateaus on multi-front incidents (medium, hard) where the trained policy
must pre-position units against the forecast and split coverage across
priority structures.

The trained-policy evaluation in `eval_policy.py` runs the same five seeds
per task, so the trained-vs-heuristic delta in `submission_artifacts/eval_*.json`
is a clean held-out generalization signal.

### Why a 16-seed training pool, not thousands

Standard RL setups (PPO on Atari, MuJoCo) train on millions of frames across
thousands of randomly sampled environment seeds. The reasoning is that each
seed contributes one tiny gradient nudge, and diversity beats repetition for
generalization. That math depends on having enough total samples — typically
10⁶–10⁸ frames — for the law of large numbers to kick in.

This run does not. The compute budget is 60 GRPO iterations × 4 trajectories
= **240 total episodes**. Stretching 240 episodes across 1000 unique seeds means
each scenario is seen ~0.24 times on average — far below what GRPO needs for
meaningful within-group advantage estimation (which compares trajectories on
the *same task* and computes `(r - mean_r) / std_r`). With unique-seed-per-iter
sampling the std collapses on lucky/unlucky scenarios, the gradient becomes
noisy, and `inner_epochs=4` re-uses data the model has already adapted to.

The 16-seed pool is the deliberate middle ground:

- **Per-scenario sample count.** With `seeds_per_iter=2` × `group_size=4` and
  `vary_env_seed_in_group=True`, each iteration visits ~4 unique scenarios.
  Over 60 iterations the policy sees roughly 120-200 distinct env states with
  meaningful repetition (~5-10 per scenario), enough for stable advantages.
- **Difficulty filtering.** Seeds were screened against a 0-79 sweep of the
  heuristic. Seeds outside the 0.2-0.85 grader range were dropped: dead-zone
  seeds (heuristic ≈ 0) collapse within-group variance, and trivially-won
  seeds (heuristic ≈ 1) leave no headroom for the policy to learn from.
- **Difficulty spread.** The 16 seeds per task are picked evenly across the
  surviving 0.2-0.85 range, so the curriculum sees a representative slice of
  scenarios at each difficulty tier rather than clustering on one variant.
- **Held-out generalization check.** The eval seeds (5 per task) are pulled
  from the same sweep but disjoint from training. If the trained policy beats
  the heuristic on training seeds but not eval seeds, that is overfitting to
  the pool — the eval JSONs are designed to surface that failure mode.

The honest tradeoff: a wider seed pool would generalize better *if* the
compute existed to support it. Within a $30 hackathon budget, concentrating
240 episodes on 16 representative scenarios produces a more reliable
gradient than spreading them thin.

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

Current artifact snapshot (generated from the fixture training log to validate packaging paths):

- `submission_artifacts/training_reward_curve.png`
- `submission_artifacts/training_loss_curve.png`
- `submission_artifacts/training_summary.md`
- `submission_artifacts/eval_untrained.json`
- `submission_artifacts/eval_trained.json`

Replace this snapshot with real run artifacts before final submission.

![Training Reward Curve](./submission_artifacts/training_reward_curve.png)
![Training Loss Curve](./submission_artifacts/training_loss_curve.png)

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

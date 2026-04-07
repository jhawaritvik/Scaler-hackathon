---
title: Wildfire Resource Allocation Environment
emoji: "\U0001F525"
colorFrom: red
colorTo: orange
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

## Why this environment

This is a real-world resource allocation task, not a toy game. The agent must
make sequential decisions under uncertainty and delay:

- choose which unit to send
- choose where to send it
- choose the mission geometry and commitment
- account for limited fleet size, travel time, and return-to-service timing

## Action Space

The action is a batch of assignments for the current decision interval.

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

- task goal and episode progress
- current fire cells and heat warnings
- structures with priority and status
- weather and time fields
- full fleet status, active missions, and outposts
- dense step reward plus action/error summaries

## Tasks

Three graded tasks are included:

- `easy`: single ignition, better moisture, more water, lower-value assets
- `medium`: hotter, drier, delayed re-ignition, and a tighter response window
- `hard`: multi-ignition, wind-driven, scarce resources, highest-value assets

Each task has a deterministic grader that returns `0.0` to `1.0` using:

- `60%` structure protection
- `30%` area preservation
- `10%` containment efficiency

## Reward

The reward gives partial progress signals over the trajectory:

| Signal | Amount | Trigger |
|---|---:|---|
| `structure_burning` | `-0.12 × priority` | Each step a structure cell is burning |
| `structure_lost` | `-0.50 × priority` | Once when a structure transitions to burned |
| `structure_safe` | `+0.003 × priority` | Each step an intact structure remains under nearby threat |
| `cells_suppressed` | `+0.04` per cell | Each burning cell extinguished that step |
| `cells_protected` | `+0.0025` per cell | Each newly protected threatened unburned cell, capped at 12 cells per step |
| `active_fire_pressure` | `-0.008 × min(burning_cells, 10)` | Each step while fire remains active |
| `fire_extinguished` | `+0.30` to `+0.70` | Once when all burning cells are out, scaled by containment speed |
| Mission dispatch cost | `-0.001` to `-0.015` | Per assignment, based on mission type |
| Resource dispatch surcharge | `-0.0005` to `-0.0040` | Additional per assignment, based on resource type |
| Invalid action | `-0.05` | Rejected assignment |
| Low-impact action | `-0.02` | Wasteful assignment with negligible effect |

The reward has been audited against the grader using `reward_audit.py`.

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

## Baselines

Deterministic heuristic baseline via `/baseline`:

- `easy`: `0.9950`
- `medium`: `0.9384`
- `hard`: `0.0148`
- average: `0.6494`

LLM baseline script:

- root script: `inference.py`
- uses OpenAI client
- reads `HF_TOKEN`, `API_BASE_URL`, and `MODEL_NAME`
- runs explicit seeded task episodes for reproducibility

## Submission Files

- root inference script: `inference.py`
- root Dockerfile: `Dockerfile`
- environment manifest: `openenv.yaml`
- detailed environment docs: `wildfire_env/README.md`

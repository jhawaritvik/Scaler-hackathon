# Hackathon Requirements Context

This file consolidates the round-1 environment requirements and the extra
submission constraints that are easy to miss while implementing the repo.
Use it as the source-of-truth checklist for the `wildfire_env` environment.

## Goal

Build a complete, real-world OpenEnv environment that an AI agent can learn
from through the standard `step()` / `reset()` / `state()` API.

## Required Deliverables

- A real-world environment, not a game or toy.
- Full OpenEnv spec support:
  - Typed `Action`, `Observation`, and `Reward` models.
  - `step(action)`
  - `reset()`
  - `state()`
  - `openenv.yaml`
- At least 3 tasks with deterministic programmatic graders:
  - `easy`
  - `medium`
  - `hard`
- Scores must remain in the range `0.0` to `1.0`.
- Dense reward shaping with partial-progress signals and penalties for bad
  behavior.
- A reproducible baseline inference script.
- A working Dockerfile and HF Space deployment path.
- A README with:
  - environment description and motivation
  - action space definition
  - observation space definition
  - task descriptions and difficulty
  - setup instructions
  - usage instructions
  - baseline scores

## Functional Requirements

### Real-World Utility

The environment must simulate a task humans actually do. For this repo that
means wildfire incident command and resource allocation, not a synthetic puzzle.

### OpenEnv Spec Compliance

The environment must expose the full OpenEnv interface:

- `reset()` returns the initial observation
- `step(action)` returns observation, reward, done, and info
- `state()` returns the current episode state
- models must be typed and Pydantic-backed
- `openenv validate` must pass

### Tasks and Graders

Each task must:

- define a concrete objective
- have a deterministic grader
- produce a score from `0.0` to `1.0`

The repo must expose at least 3 tasks with clear difficulty progression.

### Reward Design

Rewards must:

- provide signal across the trajectory, not just at the end
- reward partial progress
- penalize undesirable behavior such as:
  - repeated useless actions
  - destructive actions
  - wasting limited resources
  - stalling or looping

### Baseline Inference

The inference script must:

- use the OpenAI client for all LLM calls
- read credentials and model settings from environment variables
- run on all 3 tasks
- produce reproducible scores
- complete within 20 minutes

## Additional Submission Instructions

These constraints were not fully captured in the current repo docs and must be
kept in scope during implementation.

### Required Environment Variables

- `API_BASE_URL`: API endpoint for the LLM
- `MODEL_NAME`: model identifier used for inference
- `HF_TOKEN`: Hugging Face or API key used for deployment/integration
- `OPENAI_API_KEY`: required by the OpenAI client

### Inference Script Rules

- The inference script must be named `inference.py`.
- It must live in the repository root.
- It must use the OpenAI client even when `API_BASE_URL` points to a compatible
  endpoint.
- It must be reproducible and produce scores for all required tasks.

### Infra Constraints

- Inference runtime must stay under 20 minutes.
- The environment and inference flow must run on:
  - `vcpu=2`
  - `memory=8gb`

### Validation Gate

Before submission, all of these must pass:

- HF Space deploys and responds
- `openenv validate` passes
- `docker build` succeeds
- `docker run` starts the environment cleanly
- `inference.py` runs without error and returns scores
- all task graders run and stay within `0.0` to `1.0`

## Environment-Specific Interpretation For This Repo

To stay aligned with the wildfire domain, the environment should model a real
incident commander workflow:

- inspect terrain, structures, weather, and fire state
- deploy limited resources such as crews, dozers, and bombers
- manage tradeoffs between containment, structure protection, and resource use
- operate under task-specific constraints and episode limits

## Repo Checklist

Use this checklist while implementing:

- Replace scaffold echo models with wildfire action/observation/reward models
- Wire `WildfireEnvironment` to terrain generation and `FireSimulation`
- Implement resource actions and resource accounting
- Define `easy`, `medium`, and `hard` tasks as explicit environment tasks
- Add deterministic graders
- Add `/tasks`, `/grader`, and `/baseline` endpoints
- Add root-level `inference.py`
- Make README submission-ready
- Make Docker build and runtime submission-ready
- Run validator before submission


# 🔥 Wildfire Resource Allocation — Project Context & Reference

> This file stores all links, resources, and contextual information for the
> Scaler x Meta PyTorch OpenEnv Hackathon (Round 1). Refer here whenever you
> need background, API references, or architectural guidance.

---

## Hackathon Info

| Item | Detail |
|------|--------|
| **Hackathon** | Scaler School of Technology × Meta PyTorch OpenEnv Hackathon |
| **Round** | Round 1 — Build an OpenEnv Environment |
| **Problem Statement** | 🔥 Wildfire Resource Allocation |
| **Team Repo** | `c:\Python\Scaler-hackathon` |
| **Environment Dir** | `c:\Python\Scaler-hackathon\wildfire_env` (scaffolded via `openenv init`) |
| **Python** | 3.13.7 (in `.venv`) |
| **Docker** | 29.1.5 |

---

## Problem Statement (Selected)

**🔥 Wildfire Resource Allocation**
Build an OpenEnv environment where an AI agent acts as a wildfire incident
commander, making sequential resource allocation decisions to contain and
suppress wildfires across a terrain grid. The environment uses cellular
automata for fire spread and provides rich, dense rewards for training RL
agents.

**Local copy:** [docs/round-1-problem-statement.md](./round-1-problem-statement.md)

**Consolidated implementation checklist:** [docs/HACKATHON_REQUIREMENTS_CONTEXT.md](./HACKATHON_REQUIREMENTS_CONTEXT.md)

**Execution plan:** [docs/ENVIRONMENT_IMPLEMENTATION_PLAN.md](./ENVIRONMENT_IMPLEMENTATION_PLAN.md)

**Environment contract freeze:** [docs/ENVIRONMENT_CONTRACT_FREEZE.md](./ENVIRONMENT_CONTRACT_FREEZE.md)

**Environment dynamics spec:** [docs/ENVIRONMENT_DYNAMICS_SPEC.md](./ENVIRONMENT_DYNAMICS_SPEC.md)

**Factor validation and calibration log:** [docs/FIRE_FACTOR_VALIDATION.md](./FIRE_FACTOR_VALIDATION.md)

**Emergence-first modeling spec:** [docs/EMERGENCE_FIRST_MODELING_SPEC.md](./EMERGENCE_FIRST_MODELING_SPEC.md)

**Reward audit harness:** [docs/REWARD_AUDIT_HARNESS.md](./REWARD_AUDIT_HARNESS.md)

---

## OpenEnv Framework — Key Links

### GitHub Repository
- **Main Repo:** https://github.com/meta-pytorch/OpenEnv

### Tutorials (Official)
| # | Topic | URL |
|---|-------|-----|
| 01 | Environments (fundamentals, patterns) | https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/01-environments.md |
| 02 | Deployment (local, Docker, HF Spaces) | https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/02-deployment.md |
| 03 | Scaling (WebSocket, concurrency, HPC) | https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/03-scaling.md |
| 04 | Training (TRL + GRPO, Wordle example) | https://github.com/meta-pytorch/OpenEnv/blob/main/tutorial/04-training.md |

### Reference Implementation (echo_env)
| File | URL |
|------|-----|
| `echo_environment.py` (MCPEnvironment) | https://github.com/meta-pytorch/OpenEnv/blob/main/envs/echo_env/server/echo_environment.py |
| `app.py` (FastAPI + create_app) | https://github.com/meta-pytorch/OpenEnv/blob/main/envs/echo_env/server/app.py |
| `client.py` (EnvClient) | https://github.com/meta-pytorch/OpenEnv/blob/main/envs/echo_env/client.py |
| `models.py` (Action/Observation) | https://github.com/meta-pytorch/OpenEnv/blob/main/envs/echo_env/models.py |
| `openenv.yaml` (manifest) | https://github.com/meta-pytorch/OpenEnv/blob/main/envs/echo_env/openenv.yaml |
| `pyproject.toml` | https://github.com/meta-pytorch/OpenEnv/blob/main/envs/echo_env/pyproject.toml |
| `Dockerfile` | https://github.com/meta-pytorch/OpenEnv/blob/main/envs/echo_env/server/Dockerfile |
| `__init__.py` | https://github.com/meta-pytorch/OpenEnv/blob/main/envs/echo_env/__init__.py |

### Core Source Code
| File | URL |
|------|-----|
| types.py (Action, Observation, State, etc.) | https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/types.py |
| http_server.py (HTTPEnvServer, create_app) | https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/http_server.py |
| interfaces.py (Environment ABC) | https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/interfaces.py |
| mcp_environment.py (MCPEnvironment) | https://github.com/meta-pytorch/OpenEnv/blob/main/src/openenv/core/env_server/mcp_environment.py |

---

## Architecture Summary

### How OpenEnv Works
```
┌────────────────────────────────────────────────────────┐
│  YOUR TRAINING CODE / BASELINE AGENT                   │
│                                                        │
│  env = WildfireEnv(base_url="http://localhost:8000")   │
│  result = env.reset()                                  │
│  result = env.step(action)                             │
│                                                        │
└───────────────────┬────────────────────────────────────┘
                    │  WebSocket /ws (primary)
                    │  HTTP /reset, /step, /state (debug)
┌───────────────────▼────────────────────────────────────┐
│  DOCKER CONTAINER / LOCAL UVICORN                      │
│                                                        │
│  ┌──────────────────────────────────────────┐          │
│  │  FastAPI Server (app.py)                 │          │
│  │  └─ create_app(WildfireEnvironment,      │          │
│  │       WildfireAction, WildfireObservation)│         │
│  │     └─ WildfireEnvironment               │          │
│  │        ├── reset() → Observation         │          │
│  │        ├── step(action) → Observation    │          │
│  │        └── state → State                 │          │
│  └──────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────┘
```

### Two Patterns Available
| Pattern | Base Class | Client | Used By |
|---------|-----------|--------|---------|
| **Classic** (our scaffold) | `Environment` | `EnvClient` | echo_env, openspiel_env |
| **MCP-based** | `MCPEnvironment` | `MCPToolClient` | More advanced envs |

Our scaffold uses the **Classic** pattern (`Environment` + `EnvClient`), which is simpler and perfectly valid for the hackathon.

---

## Scaffolded Project Structure

```
wildfire_env/
├── __init__.py                          # Exports: WildfireEnv, WildfireAction, WildfireObservation
├── client.py                            # WildfireEnv(EnvClient) — WebSocket client
├── models.py                            # WildfireAction, WildfireObservation (Pydantic)
├── openenv.yaml                         # Environment manifest
├── pyproject.toml                       # Dependencies, package config
├── README.md                            # Auto-generated docs
├── uv.lock                              # Lock file
└── server/
    ├── __init__.py
    ├── app.py                           # FastAPI app via create_app()
    ├── wildfire_env_environment.py       # WildfireEnvironment(Environment) — OUR MAIN FILE
    ├── Dockerfile                       # Multi-stage Docker build
    └── requirements.txt
```

### Files We Need to Create (not in scaffold)
```
wildfire_env/
├── server/
│   ├── fire_simulation.py               # Cellular automata engine
│   ├── terrain.py                       # Terrain/fuel/elevation generation
│   ├── resources.py                     # Resource types and constraints
│   └── grader.py                        # Deterministic graders for 3 tasks
├── baseline.py                          # Baseline inference script
└── outputs/                             # Logs and eval results
```

---

## Key Technical Decisions

### Fire Simulation
- **No simfire** — incompatible with Python 3.13 (requires 3.9.x)
- Custom cellular automata engine using NumPy (~200 lines)
- Spread formula: `P_spread = P_base × F_fuel × F_wind × F_slope × F_moisture`
- 8-cell Moore neighborhood, seeded RNG for determinism

### Hackathon Requirements Checklist
- [ ] Real-world task simulation (wildfire resource allocation)
- [ ] Full OpenEnv spec: `step()` / `reset()` / `state()`
- [ ] Typed Pydantic models (Action, Observation)
- [ ] Minimum 3 tasks (easy → medium → hard)
- [ ] Deterministic graders (scores 0.0–1.0)
- [ ] Dense reward function with partial progress signals
- [ ] Baseline inference script with reproducible scores
- [ ] `openenv.yaml` manifest
- [ ] Working Dockerfile
- [ ] Deploy to HF Spaces
- [ ] README with full documentation

---

## Commands Reference

```bash
# Activate virtual environment (ALWAYS do this first)
.venv\Scripts\activate

# Install deps (from wildfire_env directory)
cd wildfire_env
pip install -e .

# Run locally with hot reload
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

# Test health
curl http://localhost:8000/health

# Docker build
docker build -t wildfire-env:latest -f server/Dockerfile .

# Docker run
docker run -d -p 8000:8000 wildfire-env:latest

# Deploy to HF Spaces
openenv push --repo-id USERNAME/wildfire-env

# Validate environment
openenv validate
```

---

## Dependencies

```toml
dependencies = [
    "openenv-core[core]>=0.2.2",
    "numpy>=1.24.0",       # Grid operations
    "openai>=1.0.0",        # Baseline script
]
```

---

## Cellular Automata Research

- **Spread probability**: P_spread = P_base × F_fuel × F_wind × F_slope × F_moisture
- **P_base**: 0.3 (calm, flat conditions)
- **F_fuel**: 0.5 (sparse) / 1.0 (moderate) / 1.5 (dense)
- **F_wind**: `1 + V_wind × cos(θ)` — downwind cells get higher probability
- **F_slope**: `exp(a × tan(slope))` where a ≈ 0.078 — fire spreads faster uphill
- **F_moisture**: 0.3–1.0 — drier = faster spread
- **Neighborhood**: 8-cell Moore, diagonal spread × 0.707

### References
- Copernicus.org — CA wildfire spread models
- Rothermel fire spread model
- INPE — Forest fire simulation with CA
- SBC.org.br — Wind-driven fire spread modeling

---

## HF Spaces Deployment Notes (from Tutorial 02)

| Component | What it provides |
|-----------|-----------------|
| **Server** | Running endpoint at `https://USERNAME-wildfire-env.hf.space` |
| **Repository** | `pip install git+https://huggingface.co/spaces/USERNAME/wildfire-env` |
| **Registry** | `docker pull registry.hf.space/USERNAME-wildfire-env:latest` |

### Scaling (from Tutorial 03)
- HF Spaces free tier: max 128 concurrent WebSocket sessions
- Use `MAX_CONCURRENT_ENVS` env var
- WebSocket `/ws` is primary protocol

### Training Integration (from Tutorial 04)
- TRL + GRPO pattern: `rollout_func` → `reset()` → loop(`step()`) → rewards
- Dense per-step rewards are essential for GRPO training
- Multiple reward channels enable fine-grained learning

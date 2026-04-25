---
title: Wildfire Resource Allocation Environment
emoji: 🔥
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Wildfire Resource Allocation Environment

An [OpenEnv](https://github.com/meta-pytorch/openenv)-compatible environment that simulates wildfire incident command on seeded terrain grids: `15×15` for easy, `20×20` for medium, and `25×25` for hard. An AI agent acts as an Incident Commander, dispatching real firefighting resources — hand crews, engines, helicopters, air tankers, dozers, and smokejumpers — to contain spreading fires and protect structures.

Fire spread is driven by coupled terrain, fuel, moisture, weather, and fire-intensity physics grounded in published wildfire science (Rothermel 1972, Alexandridis et al. 2008, NWCG operational data). Tactics modelled include direct attack, wet lines, retardant drops, dozer firebreaks, and intentional backfires with anchor-point validation.

Live deployment:

- Space: [Chunchunmaru-101/wildfire-env](https://huggingface.co/spaces/Chunchunmaru-101/wildfire-env)
- App: [chunchunmaru-101-wildfire-env.hf.space](https://chunchunmaru-101-wildfire-env.hf.space)

For the canonical hackathon submission story, training workflow, and reward
audit summary, see the repository root [README](../README.md).

---

## Quick Start

### Direct Python

```python
from wildfire_env.server.wildfire_env_environment import WildfireEnvironment
from wildfire_env.models import GridPoint, ResourceAssignment, TargetSpec, WildfireAction

env = WildfireEnvironment()
obs = env.reset()

while not obs.done:
    # Check obs.action_guide for available units and valid missions each step
    print(obs.action_guide)

    # Example: dispatch the first available crew to the hottest burning cell
    available_crews = [u for u in obs.fleet_units if u.resource_type == "crews" and u.status == "available"]
    if available_crews and obs.fire_details:
        fire = max(obs.fire_details, key=lambda f: f.intensity)
        action = WildfireAction(assignments=[
            ResourceAssignment(
                unit_id=available_crews[0].unit_id,
                mission_type="direct_attack",
                target=TargetSpec(target_kind="point", point=GridPoint(row=fire.row, col=fire.col)),
            )
        ])
    else:
        action = WildfireAction()

    obs = env.step(action)
    print(f"Step {obs.step}: reward={obs.reward:+.3f}  burning={obs.burning_cells}")
```

### Server

```bash
cd wildfire_env
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t wildfire-env:latest ..
docker run -p 8000:8000 wildfire-env:latest
```

### Smoke Tests

```bash
python -m wildfire_env.test_env_smoke
openenv validate .
```

---

## Action Space

`WildfireAction` is a batch of resource assignments issued in one decision interval.

```python
class WildfireAction(Action):
    plan: str = ""                        # optional short tactical note, ignored by env
    assignments: list[ResourceAssignment]   # can be empty (no-op)

class ResourceAssignment(BaseModel):
    unit_id: str                            # e.g. "crew_1", "helicopter_2"
    mission_type: MissionType
    target: TargetSpec
    commitment_steps: int = 1              # 1–12; how many steps to stay on mission
    drop_configuration: "salvo"|"trail"|None = None  # aerial drops only
```

### Mission Types

| Mission | Valid Resources | Effect |
|---------|----------------|--------|
| `direct_attack` | crews, engines, smokejumpers | Suppress burning cells — reduce heat, intensity, add moisture |
| `line_construction` | crews, dozers, smokejumpers | Build permanent `FIREBREAK` cells (mineral soil) |
| `wet_line` | engines | Spray water/foam along a line — high moisture that decays over several steps |
| `water_drop` | helicopters | Aerial water bucket — moderate area, fast turnaround near water |
| `retardant_drop` | airtankers | Drop long-term retardant — permanent ignition threshold increase over wide area |
| `point_protection` | crews, engines, helicopters, dozers, smokejumpers | Guard a location continuously |
| `backfire` | crews, engines | Deliberately ignite cells to consume fuel ahead of the main fire — high risk/reward |
| `staging` | all | Reposition standby location — no combat effect |

### Target Kinds

| Kind | Schema | Use Case |
|------|--------|----------|
| `point` | `{"point": {"row": R, "col": C}}` | Single cell target |
| `line` | `{"waypoints": [{"row":R1,"col":C1}, ...]}` | Multi-cell line (Bresenham drawn) |
| `area` | `{"center": {"row":R,"col":C}, "radius": N}` | Circular area |
| `polygon` | `{"vertices": [...]}` | Arbitrary polygon (≤ 8 vertices) |
| `structure` | `{"structure_id": "structure_1"}` | Named structure |

`plan` is an optional one-sentence scratchpad field capped at 160 characters.
It is preserved in the JSON schema so grammar-constrained models can emit a
short tactical rationale without breaking parse rate, but the simulator itself
ignores it and only executes `assignments`.

---

## Observation Space

`WildfireObservation` provides a complete incident picture each step:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | str | Current task: `easy`, `medium`, or `hard` |
| `goal` | str | Natural-language task objective |
| `action_guide` | str | **Per-step LLM guide**: available units + valid missions + tactical summary |
| `step` / `max_steps` | int | Current step and episode limit |
| `elapsed_minutes` | float | Incident time elapsed (20 min per step) |
| `wind_speed` / `wind_direction` | float | Current atmosphere |
| `temperature` / `humidity` | float | Current atmosphere |
| `burning_cells` / `burned_cells` | int | Fire extent |
| `fire_details` | list | Per-cell `{row, col, intensity, timer}` for burning cells |
| `heat_warnings` | list | Cells approaching ignition threshold |
| `structures` | list | Per-structure `{structure_id, row, col, priority, status}` |
| `structures_remaining` / `structures_lost` | int | Structure tally |
| `fleet_units` | list | Per-unit `{unit_id, resource_type, status, position, missions_completed, available_in_steps, ...}` |
| `fleet_status` | list | Per-resource-type status counts |
| `active_missions` | list | Non-available units with ETAs |
| `resources_remaining` | dict | Available unit count per type |
| `elevation` | list[list[int]] | `grid_size×grid_size` elevation map (15×15 easy, 20×20 medium, 25×25 hard) |
| `fuel_types` | list[list[int]] | `grid_size×grid_size` fuel map (0=none, 1=grass, 2=brush, 3=forest) |
| `last_action_summary` | str | What happened last step |
| `last_action_error` | str\|None | Validation error, if any |
| `reward` | float | Dense per-step reward |

### `action_guide` (Key Feature)

Every observation includes a natural-language `action_guide` that lists which units are available right now, what missions they can perform, which units are committed and their ETAs, and a tactical summary. LLM agents can use this directly without memorising the resource–mission compatibility matrix.

Example:
```
AVAILABLE (can dispatch now):
  crew_1 [crews]: backfire | direct_attack | line_construction | point_protection | staging
  engine_1 [engines]: backfire | direct_attack | point_protection | staging | wet_line
  helicopter_1 [helicopters]: point_protection | staging | water_drop
COMMITTED:
  dozer_1 operating → line_construction, done in 1 step(s)
  crew_2 returning, available in 2 step(s)

FIRE: 4 burning | 7 burned. Wind: 15 km/h from NE (45°)
STRUCTURES: structure_1(p1,safe) | structure_2(p2,safe)
```

---

## Fleet System

Six resource types with realistic dispatch timing (NWCG-sourced):

| Resource | Prep | Travel | Return | Notable |
|----------|------|--------|--------|---------|
| **Crews** | 10 min | 2.5 min/cell | 15 min | Versatile; can set backfires |
| **Engines** | 5 min | 1.5 min/cell | 10 min | Fast mobile attack; wet lines |
| **Helicopters** | 5 min | 0.6 min/cell | 8 min (scoop) / 20 min (base) | **Water-scoop mechanic**: refills in 8 min if a water body is within 5 cells of the drop target |
| **Air Tankers** | 8 min | 0.4 min/cell | 30 min | Permanent retardant; widest area coverage |
| **Dozers** | 15 min | 2.0 min/cell | 20 min | Permanent firebreaks; rate varies by fuel type |
| **Smokejumpers** | 2 min | 0.3 min/cell | 40 min | Near-instant deployment anywhere; rare and slow return |

Each `FleetUnit` tracks: `status` (available / en_route / operating / returning), `missions_completed` (cumulative episode count), `available_in_steps`, and current `position`.

---

## Tasks

| | easy | medium | hard |
|---|---|---|---|
| Seed | 7 | 6 | 9 |
| Ignitions | 1 at step 0 + 1 at step 7 | 2 at step 0 + 1 at step 5 + 1 at step 6 | 2 at step 0 + 2 at step 4 |
| Temperature | 29.4C | 32.5C | 38.1C |
| Humidity | 42.5% | 30.2% | 38.5% |
| Wind | 10.5 km/h | 19.2 km/h | 21.5 km/h |
| Water bodies | 2 | 1 | 1 |
| Structures | 4 x priority 1 | 2 x p1 + 3 x p2 | 2 x p1 + 3 x p2 |
| Default seeded resources | 2c 1e 1h 1d | 3c 3e 2h 1d | 5c 3e 1h 1a 2d 1s |
| Warmup before first observation | 0 steps | 0 steps | 2 steps (40 min) |
| Max steps | 20 | 20 | 25 |

**Grader formula (0.0–1.0):**
- 60% — Structure protection: saved priority / total priority, weighted by structure priority values
- 30% — Area preservation: fraction of burnable terrain not burned at episode end
- 10% — Efficiency bonus: awarded only when fire fully contained; faster is better

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Reset to the next task; returns initial observation |
| `POST` | `/step` | Submit action, advance simulation one tick; returns observation |
| `GET` | `/state` | Return current environment state |
| `GET` | `/schema` | Return action and observation JSON schemas |
| `WS` | `/ws` | Persistent WebSocket session for multi-step episodes |
| `GET` | `/tasks` | List all tasks, action schema, resource–mission compatibility |
| `POST` | `/grader` | Grade a completed episode (pass final observation data) |
| `GET` | `/baseline` | Run deterministic heuristic agent; return reproducible scores |

### `/grader` example

```bash
curl -X POST http://localhost:8000/grader \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "easy",
    "step": 12,
    "max_steps": 20,
    "structures": [{"priority": 1, "status": "safe"}, {"priority": 1, "status": "safe"}],
    "burned_cells": 9,
    "burning_cells": 0
  }'
# → {"score": 0.891, "components": {...}, ...}
```

---

## Reward Function

Dense per-step reward is composed of the following current signals:

| Signal | Value | Trigger |
|--------|-------|---------|
| Structure burning | `-0.12 × priority × gn` | Each step a structure is burning (`gn = min(1, (15/size)^1.5)`) |
| Structure lost | `-0.50 × priority × gn` | **Once** when a structure cell transitions to burned |
| Structure safe | `+0.003 × priority` | Each step an intact structure remains under nearby threat |
| Cells suppressed | `+0.06` per cell | Each burning cell extinguished by active suppression that step |
| Cells protected | `+0.0025` per cell | Each newly protected threatened unburned cell, capped at 12 cells per step |
| Containment | `+0.018 × net cells` | Net burning-cell reduction, gated on active suppression |
| Fire extinguished | `+0.30` to `+0.70` | **Once** when all burning cells reach zero, scaled by containment speed |
| Active fire pressure | `-0.005 × min(burning_cells, 8)` | Each step while fire remains active |
| Mission dispatch cost | `-0.0008` to `-0.0112` | Per dispatch, based on mission type |
| Resource dispatch surcharge | `-0.0004` to `-0.0030` | Additional per dispatch, based on resource type |
| Idle penalty | `-0.005 × min(available_units, 4)` | Empty action while fire is active and units are available |
| LCES violation | `-0.03` | Unsafe ground assignment without a reachable safety zone |
| Invalid action | `-0.05` | Rejected assignment |
| Low-impact action | `-0.02` | Wasteful mission |

Weather conditions (temperature, humidity, wind) are observable but do not directly generate reward signals. The agent should learn that adverse weather increases fire spread risk and adjust tactics accordingly.

---

## Baseline Scores

Run the LLM agent:
```bash
HF_TOKEN=hf_or_provider_token API_BASE_URL=https://your-endpoint/v1 MODEL_NAME=your-model python inference.py
```

Run the deterministic heuristic baseline (no API key needed):
```bash
curl http://localhost:8000/baseline
```

| Task | Heuristic baseline |
|------|--------------------|
| easy | 0.7697 |
| medium | 0.6437 |
| hard | 0.6246 |

---

## Environment Design

### Emergence-First Modeling

Primary state variables and their direct couplings are modelled explicitly. Named wildfire phenomena are not scripted as event switches. Complex behaviours (spotting, crossover, crown fire transition, wind-driven runs) emerge naturally from coupled dynamics.

### Physics Sources

| Coupling | Source |
|----------|--------|
| Wind spread loading | Alexandridis et al. (2008) |
| Slope spread factor | Rothermel (1972) via Alexandridis |
| Moisture damping | Rothermel (1972) polynomial |
| Spotting | Albini (1983) compact form |
| Fireline production rates | NWCG 2021 San Dimas Tech Tip |
| Dispatch timings | NWCG / USDA AFUE study |
| Aspect modifiers | NWCG S-190 |
| Fuel extinction moisture | Scott & Burgan (2005) |


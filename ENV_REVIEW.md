# Environment Review & Submittability Assessment

Strict review of the wildfire env relative to the round-2 problem statement.
Scope: env structure and realism (reward function deferred until 40-iter
training results land). Baseline for comparison: an alternative submission
such as an email-triage agent.

---

## Strengths (worth keeping intact)

- 6 resource × 8 mission × 5 target-geometry action space — combinatorially
  huge and genuinely hard to master.
- Physics constants in `wildfire_env/server/fire_simulation.py:58-143` match
  their cited sources (Alexandridis 2008 CA; Rothermel 1972 moisture damping;
  Scott & Burgan fuel moisture extinction; NWCG time-lag classes).
- NWCG-sourced dispatch timings in `wildfire_env/server/resources.py:69-155`
  (hand-crew 55 m/min hike, Type 3 engine 10-15 mph, LAT 200 kt cruise,
  helicopter scoop vs. base-reload auto-selection, smokejumper
  helicopter-extraction return logic).
- Deterministic seeded terrain with difficulty bands drawn from parameter
  ranges (`wildfire_env/server/terrain.py:815-894`).
- Outpost ground/air split with round-robin resource distribution.
- Fire-driven airflow-potential field that perturbs ambient wind (reduced-order
  plume coupling).
- Ember spotting past 28.8 km/h threshold (Freire & DaCamara 2019).
- Resource–mission compatibility matrix mirrors real NWCG resource typing.
- Retardant permanent ignition-threshold bonus (needs decay — see below).
- Structure priority model with grader weighting.
- OpenEnv compliance: `Dockerfile`, `openenv.yaml`, `/reset`, `/step`,
  `/grader`, `/baseline`, `/tasks`, `/ws`, `baseline_scores.json`,
  `pyproject.toml` + `uv.lock` all present and correct.

---

## Concrete improvements (non-reward)

Ordered by judge-impact relative to implementation cost.

### 1. Grid is too small, episode is too short — HIGH impact, LOW cost

- `terrain.py:26` hardcodes `GRID_SIZE = 15` → 225 cells.
- `max_steps` is 15–20 per difficulty → ~5–7 sim hours (wildfires evolve over
  days in reality).
- Hard task gives the agent ~12 active decisions before timeout — not enough
  room to separate skill from luck; easy baseline 0.995 confirms the ceiling
  is hit immediately.
- **Fix:** Bump hard → 25×25 grid, 25 steps. 1-line spec change per difficulty.
  Will visibly grow the strategy space and give the trained model actual
  headroom above the heuristic baseline.

### 2. No weather forecast in observation — HIGH impact, LOW cost

- A real incident commander receives RAWS forecast pulls.
  `WildfireObservation` currently only exposes *current* atmosphere
  (`wind_speed`, `humidity`, `temperature`).
- **Fix:** Add 2–3 step lookahead fields (`forecast_wind_speed_next`,
  `forecast_wind_direction_next`, `forecast_humidity_next`). ~20 lines in
  `wildfire_env_environment.py`. Clean innovation bullet for the writeup,
  and the agent can actually learn anticipatory pre-positioning.

### 3. No partial observability — HIGH impact, MEDIUM cost

- Agent currently sees the full grid.
- Real ICs operate with fog-of-war; situational awareness is the central
  operational challenge.
- **Fix:** Add `visibility_radius` per active unit (drone/lookout coverage)
  and mask `fire_details`/`heat_warnings` outside that union. Biggest
  single realism upgrade available and storytelling gold ("the agent learned
  to deploy a lookout to a ridge before committing crews").

### 4. No crew fatigue / LCES safety model — HIGH innovation, MEDIUM cost

- LCES (Lookouts / Comms / Escape routes / Safety zones) is *the* NWCG
  wildland safety doctrine. Not modeled at all currently.
- **Fix:** Penalise assignments that put crews downwind of high-intensity
  fire with no escape path to a firebreak/water/burned cell. Uniquely
  fire-domain — judges cannot find this feature in any other RL env.

### 5. No retardant decay — LOW cost

- `retardant_bonus` in `fire_simulation.py:303` is permanent, capped at
  0.40. NWCG models retardant degrading with rain/UV exposure.
- **Fix:** One subtraction per tick (e.g., 0.02 decay/step), floored at 0.
  Small change but physically honest.

### 6. Episode-end condition is too easy to trigger — MEDIUM impact, LOW cost

- `fire_simulation.py:471-481`: suppressing the first ignition and waiting
  past the future-ignition schedule ends the episode.
- Easy baseline at 0.995 is the ceiling-clip symptom.
- **Fix:** Require either all burning + future ignitions consumed AND a
  minimum elapsed step count, OR require structures defensible score > X,
  so "put one fire out fast" is not a perfect score.

### 7. `GRID_SIZE` hardcoded in three places — LOW cost

- `terrain.py:26` constant.
- `wildfire_env/server/app.py:300` heuristic uses `min(14, ...)`.
- `models.py` has validators referencing the size.
- **Fix:** Thread `grid_size` through from the resolved `TerrainConfig` so
  scaling to 25×25 or 30×30 is a config change, not a search-and-replace.

### 8. `/ws` websocket is largely cosmetic — MEDIUM storytelling impact

- Exists on the server but no published client demo.
- **Fix (option A):** Drop it from advertised endpoints.
- **Fix (option B, better):** Wire up a minimal live-step HTML page that
  streams grid + unit positions each tick. If you are making a demo video,
  this gives you the visual. Likely the highest storytelling-ROI change in
  the repo.

### 9. Aspect / fuel / water generation are pure-Python double loops — LOW
   urgency at current scale

- `terrain.py:_compute_aspect`, `_generate_fuel`, `_place_water` all
  O(N²) Python.
- Fine at 15×15, noticeable at 30×30.
- **Fix:** Vectorise with numpy if you scale the grid up per #1.

### 10. No evacuation mission — OPTIONAL

- Structures have priority but agents cannot move/evacuate occupants.
- Adding an `evacuation` mission that consumes crew-steps to declare a
  structure "evacuated" (counts as saved even if later burned) adds a
  human-stakes dimension that plays well in storytelling.

---

## Submittability — strict assessment

**Verdict:** submittable, but not a lock. Lives or dies on env-innovation
polish and storytelling, not the training pipeline.

### By judging criterion

| Criterion | Weight | Read |
|---|---:|---|
| Env Innovation | 40% | **Strong-but-shallow.** Genuinely novel domain with sourced physics, but at 15×15 / 20 steps it reads like a tech demo. Vs. an email-triage agent: wildfire wins on theme novelty and #3.1 World-Modeling alignment, loses on "agent that could ship Monday." Improvements 1–4 above directly attack this. |
| Storytelling | 30% | **Weakest link.** No demo video, no scenario walkthrough, no narrative about what the agent learned. README is operationally clear but emotionally flat. A 2-minute screen recording of a trained agent saving a high-priority structure while the heuristic loses it would close most of this gap. Improvement #8 enables this. |
| Reward Improvement | 20% | **At risk.** Heuristic baselines: easy=0.995, medium=0.795, hard=0.103. Training currently matches medium and barely moves hard. Either (a) beat medium clearly, or (b) make hard reachable (per #1 / #6) and argue "we moved hard from 0.10 → 0.X." Reward fn review deferred until 40-iter results arrive. |
| Pipeline | 10% | **Strong.** Hand-rolled multi-turn GRPO (not trl boilerplate), curriculum, cosine LR + warmup, multi-seed advantage normalisation, per-task best checkpointing, resume-from-disconnect. Most defensible part of the submission. |

### Competitive framing

The honest risk vs. an email-triage submission:

- Judges can pick up an email-triage agent and immediately see "yes this
  would be useful at my job."
- Wildfire requires them to *first* care about wildfires.
- The mitigation is storytelling — make them care in the first 30 seconds.

The 40% Env Innovation weight and the explicit #3.1 World-Modeling
(Professional Tasks) theme alignment in the problem statement do favor a
deeply-modeled domain env over a thin SaaS-style agent — but only if the
depth is legible to a judge skimming the repo in 5 minutes.

### Not a structural blocker

All of these are present and correct:
- `Dockerfile` (builds from `ghcr.io/meta-pytorch/openenv-base:latest`)
- `openenv.yaml` (spec v1, fastapi runtime, port 8000)
- `/reset`, `/step`, `/grader`, `/baseline`, `/tasks` endpoints
- `baseline_scores.json`
- Deterministic seeds per difficulty
- `pyproject.toml` + `uv.lock`

---

## Suggested next steps

1. **First:** improvements #1 and #6 together — scale hard to 25×25 / 25
   steps with a stricter end-of-episode condition. This single change should
   materially widen the gap between trained model and heuristic.
2. **Then:** #2 (forecast in observation) and #5 (retardant decay) — small,
   cheap, and each one is a bullet in the writeup.
3. **Then:** #3 (partial observability) or #4 (LCES) — pick one as the
   headline innovation for the 40% criterion.
4. **In parallel with training runs:** #8 option B — live websocket viewer
   for the demo video. This is the highest-ROI storytelling change.
5. **Revisit reward function** once the 40-iter output is in.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Wildfire Env Environment.

Exposes the standard OpenEnv endpoints plus three submission-required extras:

    GET  /tasks     — list tasks, action schema, resource–mission compatibility
    POST /grader    — grade a completed episode (returns 0.0–1.0 score)
    GET  /baseline  — run a deterministic heuristic agent and return scores

Standard OpenEnv endpoints (via create_app):
    POST /reset
    POST /step
    GET  /state
    GET  /schema
    WS   /ws
"""

import asyncio

import numpy as np
from fastapi import Body, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import (
        GridPoint,
        ResourceAssignment,
        TargetSpec,
        WildfireAction,
        WildfireObservation,
    )
    from .wildfire_env_environment import WildfireEnvironment, RESOURCE_MISSION_COMPATIBILITY, TASK_GOALS
    from .terrain import DIFFICULTY_SPECS, DEFAULT_SEEDS, get_task_config, generate_terrain, FUEL_NONE
except ImportError:
    from models import (
        GridPoint,
        ResourceAssignment,
        TargetSpec,
        WildfireAction,
        WildfireObservation,
    )
    from server.wildfire_env_environment import WildfireEnvironment, RESOURCE_MISSION_COMPATIBILITY, TASK_GOALS
    from server.terrain import DIFFICULTY_SPECS, DEFAULT_SEEDS, get_task_config, generate_terrain, FUEL_NONE


# ---------------------------------------------------------------------------
# Core OpenEnv app
# ---------------------------------------------------------------------------

app = create_app(
    WildfireEnvironment,
    WildfireAction,
    WildfireObservation,
    env_name="wildfire_env",
    max_concurrent_envs=1,
)


@app.get("/")
def root():
    """Health / landing endpoint."""
    return {"status": "ok", "environment": "wildfire_env"}


# ---------------------------------------------------------------------------
# Static task catalogue
# ---------------------------------------------------------------------------

def _build_task_definitions() -> dict[str, dict]:
    """Build task catalogue from difficulty specs + default seeds."""
    defs = {}
    for task_id, spec in DIFFICULTY_SPECS.items():
        config = get_task_config(task_id)  # default seed
        defs[task_id] = {
            "task_id": task_id,
            "difficulty": task_id,
            "description": (
                f"Parameterized {task_id} scenario (seed={DEFAULT_SEEDS[task_id]}). "
                f"Temperature {spec.temperature[0]:.0f}-{spec.temperature[1]:.0f}C, "
                f"humidity {spec.humidity[0]*100:.0f}-{spec.humidity[1]*100:.0f}%, "
                f"wind {spec.wind_speed[0]:.0f}-{spec.wind_speed[1]:.0f} km/h. "
                f"Ignitions: {spec.ignitions_step0[0]}-{spec.ignitions_step0[1]} at step 0"
                + (f" + {spec.delayed_ignitions[0]}-{spec.delayed_ignitions[1]} delayed"
                   if spec.delayed_ignitions[1] > 0 else "")
                + (f". Warmup: {spec.warmup_steps[0]}-{spec.warmup_steps[1]} step(s)"
                   if spec.warmup_steps[1] > 0 else "")
                + f". Structures: {spec.num_structures[0]}-{spec.num_structures[1]} (max priority {spec.max_priority})."
            ),
            "goal": TASK_GOALS.get(task_id, ""),
            "default_seed": DEFAULT_SEEDS[task_id],
            "parameter_ranges": {
                "temperature_c": list(spec.temperature),
                "humidity": list(spec.humidity),
                "wind_speed_kmh": list(spec.wind_speed),
                "ignitions_step0": list(spec.ignitions_step0),
                "delayed_ignitions": list(spec.delayed_ignitions),
                "warmup_steps": list(spec.warmup_steps),
                "num_structures": list(spec.num_structures),
                "max_priority": spec.max_priority,
                "crews": list(spec.crews),
                "engines": list(spec.engines),
                "helicopters": list(spec.helicopters),
                "airtankers": list(spec.airtankers),
                "dozers": list(spec.dozers),
                "smokejumpers": list(spec.smokejumpers),
                "num_water_bodies": list(spec.num_water_bodies),
                "num_ground_outposts": list(spec.num_ground_outposts),
                "num_air_bases": list(spec.num_air_bases),
            },
            "max_steps": spec.max_steps,
        }
    return defs


TASK_DEFINITIONS = _build_task_definitions()


# ---------------------------------------------------------------------------
# Grader models and logic
# ---------------------------------------------------------------------------

class GraderRequest(BaseModel):
    """Episode outcome data needed to compute a grader score."""

    task_id: str = Field(..., description="Task identifier: easy, medium, or hard")
    seed: int = Field(default=0, description="Seed used for the episode (0 = default seed)")
    step: int = Field(..., ge=0, description="Step at which the episode ended")
    max_steps: int = Field(..., ge=1)
    structures: list[dict] = Field(
        default_factory=list,
        description=(
            "List of structure dicts from WildfireObservation.structures. "
            "Each entry must have 'priority' (int) and 'status' (str) fields. "
            "Valid status values that count as saved: anything other than "
            "'burning', 'burned', or 'lost'."
        ),
    )
    burned_cells: int = Field(..., ge=0, description="Number of fully burned cells at episode end")
    burning_cells: int = Field(default=0, ge=0, description="Number of still-burning cells at episode end")


class GraderResponse(BaseModel):
    score: float = Field(..., gt=0.0, lt=1.0, description="Final episode score strictly within (0, 1)")
    task_id: str
    components: dict = Field(default_factory=dict)
    description: str = ""


_SCORE_EPS = 0.01


def _strict_open_unit_interval(value: float) -> float:
    """Clamp a value strictly inside (0, 1) for validator compatibility."""
    return max(_SCORE_EPS, min(1.0 - _SCORE_EPS, float(value)))


def _grade_episode(req: GraderRequest) -> GraderResponse:
    """Compute a normalised 0.0–1.0 grader score from episode outcome data.

    Scoring weights:
      - Structure protection (60%): weighted sum of saved structure priorities
        divided by total structure priority in the task.
      - Area preservation (30%): fraction of burnable terrain cells that
        were NOT burned or burning at episode end.
      - Efficiency bonus (10%): awarded only when fire is fully contained;
        a faster containment gives a higher bonus.
    """
    if req.task_id not in DIFFICULTY_SPECS:
        return GraderResponse(
            score=_SCORE_EPS,
            task_id=req.task_id,
            components={
                "structure_component": _SCORE_EPS,
                "area_component": _SCORE_EPS,
                "efficiency_component": _SCORE_EPS,
                "weights": {"structure": 0.60, "area": 0.30, "efficiency": 0.10},
                "saved_priority": 0,
                "total_priority": 1,
                "cells_damaged": 0,
                "total_burnable": 1,
            },
            description=f"unknown task_id '{req.task_id}'",
        )

    # Compute total burnable cells from the seeded terrain
    seed = req.seed if req.seed else None
    config = get_task_config(req.task_id, seed)
    terrain = generate_terrain(config)
    total_burnable = int(np.sum(
        (terrain.fuel_type != FUEL_NONE) & ~terrain.is_water
    ))

    # ── Structure score (60%) ──
    lost_statuses = {"burning", "burned", "lost", "BURNING", "BURNED", "LOST"}
    total_priority = sum(s.get("priority", 1) for s in req.structures) if req.structures else 1
    saved_priority = sum(
        s.get("priority", 1) for s in req.structures
        if s.get("status", "safe") not in lost_statuses
    )
    structure_score = saved_priority / max(1, total_priority)

    # ── Area preservation score (30%) ──
    cells_damaged = req.burned_cells + req.burning_cells
    area_score = max(0.0, 1.0 - cells_damaged / max(1, total_burnable))

    # ── Efficiency score (10%) ── only awarded when fire is fully out
    if req.burning_cells == 0:
        efficiency_score = 1.0 - (req.step / max(1, req.max_steps))
    else:
        efficiency_score = 0.0

    # Clamp strictly within (0, 1) — validator requires score ∈ (0.0, 1.0)
    score = _strict_open_unit_interval(
        structure_score * 0.60
        + area_score * 0.30
        + efficiency_score * 0.10
    )
    structure_component = _strict_open_unit_interval(structure_score)
    area_component = _strict_open_unit_interval(area_score)
    efficiency_component = _strict_open_unit_interval(efficiency_score)

    return GraderResponse(
        score=round(score, 4),
        task_id=req.task_id,
        components={
            "structure_component": round(structure_component, 4),
            "area_component": round(area_component, 4),
            "efficiency_component": round(efficiency_component, 4),
            "weights": {"structure": 0.60, "area": 0.30, "efficiency": 0.10},
            "saved_priority": saved_priority,
            "total_priority": total_priority,
            "cells_damaged": cells_damaged,
            "total_burnable": total_burnable,
        },
        description=(
            f"Structure: {saved_priority}/{total_priority} priority saved. "
            f"Area: {total_burnable - cells_damaged}/{total_burnable} burnable cells intact."
        ),
    )


# ---------------------------------------------------------------------------
# Heuristic baseline agent (used by /baseline endpoint)
# ---------------------------------------------------------------------------

def _heuristic_action(obs: WildfireObservation) -> WildfireAction:
    """Rule-based heuristic: attack the highest-intensity burning cell."""
    assignments = []
    grid_max = (len(obs.fuel_types[0]) - 1) if obs.fuel_types else 14

    target_point: GridPoint | None = None
    if obs.fire_details:
        best = max(obs.fire_details, key=lambda f: f.intensity)
        target_point = GridPoint(row=best.row, col=best.col)

    for unit in obs.fleet_units:
        if unit.status != "available" or target_point is None:
            break

        rtype = unit.resource_type
        if rtype in ("crews", "engines", "smokejumpers"):
            assignments.append(ResourceAssignment(
                unit_id=unit.unit_id,
                mission_type="direct_attack",
                target=TargetSpec(target_kind="point", point=target_point),
            ))
        elif rtype == "helicopters":
            assignments.append(ResourceAssignment(
                unit_id=unit.unit_id,
                mission_type="water_drop",
                target=TargetSpec(target_kind="area", center=target_point, radius=1),
                drop_configuration="salvo",
            ))
        elif rtype == "airtankers":
            assignments.append(ResourceAssignment(
                unit_id=unit.unit_id,
                mission_type="retardant_drop",
                target=TargetSpec(target_kind="area", center=target_point, radius=2),
                drop_configuration="salvo",
            ))
        elif rtype == "dozers":
            ahead_row = max(0, target_point.row - 3)
            assignments.append(ResourceAssignment(
                unit_id=unit.unit_id,
                mission_type="line_construction",
                target=TargetSpec(
                    target_kind="line",
                    waypoints=[
                        GridPoint(row=ahead_row, col=max(0, target_point.col - 2)),
                        GridPoint(row=ahead_row, col=min(grid_max, target_point.col + 2)),
                    ],
                ),
            ))

    return WildfireAction(assignments=assignments)


# ---------------------------------------------------------------------------
# Additional endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks")
def get_tasks() -> dict:
    """List all available tasks, the action schema, and resource–mission compatibility."""
    return {
        "tasks": list(TASK_DEFINITIONS.values()),
        "task_count": len(TASK_DEFINITIONS),
        "action_schema": WildfireAction.model_json_schema(),
        "resource_mission_compatibility": {
            resource: sorted(missions)
            for resource, missions in RESOURCE_MISSION_COMPATIBILITY.items()
        },
    }


@app.post("/grader", response_model=GraderResponse)
def compute_grader(request: GraderRequest = Body(...)) -> GraderResponse:
    """Grade a completed episode and return a 0.0–1.0 score.

    POST the final WildfireObservation.structures list together with
    burned_cells and burning_cells counts from the terminal observation.

    Example body::

        {
          "task_id": "easy",
          "step": 12,
          "max_steps": 20,
          "structures": [
            {"priority": 1, "status": "safe"},
            {"priority": 1, "status": "safe"}
          ],
          "burned_cells": 8,
          "burning_cells": 0
        }
    """
    return _grade_episode(request)


@app.get("/baseline")
def run_baseline() -> dict:
    """Run a deterministic heuristic agent against all three tasks and return scores.

    Uses a rule-based agent — no LLM or API key required.
    Results are reproducible across runs because terrain and fire simulation
    are seeded deterministically per task.
    """
    results: dict[str, dict] = {}

    for task_id in DIFFICULTY_SPECS:
        seed = DEFAULT_SEEDS[task_id]
        env = WildfireEnvironment()
        obs = env.reset(seed=seed, task_id=task_id)

        while not obs.done:
            action = _heuristic_action(obs)
            obs = env.step(action)

        req = GraderRequest(
            task_id=task_id,
            seed=seed,
            step=obs.step,
            max_steps=obs.max_steps,
            structures=[s.model_dump() for s in obs.structures],
            burned_cells=obs.burned_cells,
            burning_cells=obs.burning_cells,
        )
        grade = _grade_episode(req)
        results[task_id] = {
            "score": grade.score,
            "seed": seed,
            "components": grade.components,
            "description": grade.description,
            "steps_taken": obs.step,
            "structures_remaining": obs.structures_remaining,
            "structures_lost": obs.structures_lost,
            "burned_cells": obs.burned_cells,
        }

    return {
        "agent": "heuristic (rule-based, deterministic)",
        "baseline_scores": results,
        "average_score": round(
            sum(r["score"] for r in results.values()) / max(1, len(results)), 4
        ),
    }


# ---------------------------------------------------------------------------
# Live viewer — WebSocket demo stream + HTML page
# ---------------------------------------------------------------------------

_CELL_COLORS = {
    0: "#3a7d44",   # unburned  — forest green
    1: "#ff4500",   # burning   — orange-red (intensity applied client-side)
    2: "#4a3728",   # burned    — dark brown
    3: "#e8c84a",   # firebreak — golden
    4: "#1e90ff",   # water     — dodger blue
    5: "#9b59b6",   # structure — purple
    6: "#20b2aa",   # suppressed — teal
}

_RESOURCE_COLORS = {
    "crews":        "#3498db",
    "engines":      "#e74c3c",
    "helicopters":  "#00bcd4",
    "airtankers":   "#ff9800",
    "dozers":       "#f1c40f",
    "smokejumpers": "#8e44ad",
}

_VIEWER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Wildfire Env — Live Viewer</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0f0f1a;color:#e0e0e0;font-family:'Courier New',monospace;
       display:flex;flex-direction:column;min-height:100vh}
  h1{text-align:center;padding:12px;background:#1a1a2e;font-size:1.1rem;
     letter-spacing:2px;color:#ff6b35;border-bottom:1px solid #333}
  #main{display:flex;flex:1;gap:12px;padding:12px;flex-wrap:wrap}
  #grid-panel{flex:0 0 auto}
  #side-panel{flex:1;min-width:260px;display:flex;flex-direction:column;gap:10px}
  svg#grid{display:block;border:1px solid #333;background:#1a1a2e}
  .panel{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:4px;padding:10px}
  .panel h3{font-size:.75rem;color:#888;letter-spacing:1px;margin-bottom:8px;
             text-transform:uppercase}
  #controls{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  select,button{background:#2a2a4a;color:#e0e0e0;border:1px solid #444;
                 padding:4px 8px;border-radius:3px;cursor:pointer;font-size:.8rem}
  button{background:#ff6b35;border-color:#ff6b35;color:#fff;font-weight:bold}
  button:hover{background:#e55a28}
  label{font-size:.75rem;color:#aaa}
  input[type=range]{width:100px}
  #step-bar{background:#1a1a2e;padding:8px 12px;border-top:1px solid #333}
  #progress-track{background:#2a2a4a;height:6px;border-radius:3px;overflow:hidden;margin:4px 0}
  #progress-fill{height:100%;background:#ff6b35;width:0%;transition:width .3s}
  #step-label{font-size:.75rem;color:#aaa}
  #action-text{font-size:.68rem;color:#888;margin-top:4px;white-space:nowrap;
                overflow:hidden;text-overflow:ellipsis;max-width:100%}
  .stat-row{display:flex;justify-content:space-between;font-size:.75rem;
             margin:3px 0}
  .stat-val{color:#ff6b35;font-weight:bold}
  #wind-canvas{display:block;margin:0 auto}
  #forecast-table{font-size:.68rem;width:100%;border-collapse:collapse}
  #forecast-table th{color:#666;text-align:left;padding:2px 4px;
                      border-bottom:1px solid #333}
  #forecast-table td{padding:2px 4px;color:#aaa}
  .legend-row{display:flex;flex-wrap:wrap;gap:6px;margin-top:4px}
  .legend-item{display:flex;align-items:center;gap:4px;font-size:.65rem}
  .legend-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0}
  #score-overlay{display:none;position:fixed;top:50%;left:50%;
                  transform:translate(-50%,-50%);background:#1a1a2e;
                  border:2px solid #ff6b35;border-radius:8px;
                  padding:24px 36px;text-align:center;z-index:10}
  #score-overlay h2{color:#ff6b35;font-size:2rem}
  #score-overlay p{color:#aaa;font-size:.85rem;margin-top:6px}
  #score-overlay button{margin-top:14px}
</style>
</head>
<body>
<h1>&#128293; WILDFIRE INCIDENT COMMAND — LIVE VIEWER</h1>
<div id="main">
  <div id="grid-panel">
    <svg id="grid" width="450" height="450"></svg>
  </div>
  <div id="side-panel">
    <div class="panel">
      <h3>Controls</h3>
      <div id="controls">
        <select id="task-sel">
          <option value="easy">Easy</option>
          <option value="medium" selected>Medium</option>
          <option value="hard">Hard</option>
        </select>
        <button id="play-btn" onclick="startEpisode()">&#9654; Play</button>
        <label>Speed <input type="range" id="speed-sl" min="100" max="2000" step="100" value="700"></label>
        <span id="speed-label" style="font-size:.7rem;color:#888">700ms</span>
      </div>
    </div>
    <div class="panel">
      <h3>Fire Status</h3>
      <div class="stat-row"><span>Burning cells</span><span class="stat-val" id="st-burning">—</span></div>
      <div class="stat-row"><span>Burned cells</span><span class="stat-val" id="st-burned">—</span></div>
      <div class="stat-row"><span>Structures safe</span><span class="stat-val" id="st-safe">—</span></div>
      <div class="stat-row"><span>Structures lost</span><span class="stat-val" id="st-lost">—</span></div>
    </div>
    <div class="panel">
      <h3>Weather</h3>
      <canvas id="wind-canvas" width="80" height="80"></canvas>
      <div class="stat-row" style="margin-top:6px">
        <span>Wind</span><span class="stat-val" id="st-wind">—</span>
      </div>
      <div class="stat-row"><span>Temp</span><span class="stat-val" id="st-temp">—</span></div>
      <div class="stat-row"><span>Humidity</span><span class="stat-val" id="st-hum">—</span></div>
    </div>
    <div class="panel">
      <h3>Forecast (+20 min / +40 min)</h3>
      <table id="forecast-table">
        <tr><th></th><th>Temp</th><th>Hum</th><th>Wind</th></tr>
        <tr><td>+20m</td><td id="fc1-t">—</td><td id="fc1-h">—</td><td id="fc1-w">—</td></tr>
        <tr><td>+40m</td><td id="fc2-t">—</td><td id="fc2-h">—</td><td id="fc2-w">—</td></tr>
      </table>
    </div>
    <div class="panel">
      <h3>Legend</h3>
      <div class="legend-row">
        <div class="legend-item"><div class="legend-dot" style="background:#3a7d44"></div>Unburned</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ff4500"></div>Burning</div>
        <div class="legend-item"><div class="legend-dot" style="background:#4a3728"></div>Burned</div>
        <div class="legend-item"><div class="legend-dot" style="background:#e8c84a"></div>Firebreak</div>
        <div class="legend-item"><div class="legend-dot" style="background:#1e90ff"></div>Water</div>
        <div class="legend-item"><div class="legend-dot" style="background:#9b59b6"></div>Structure</div>
        <div class="legend-item"><div class="legend-dot" style="background:#20b2aa"></div>Suppressed</div>
      </div>
      <div class="legend-row" style="margin-top:8px">
        <div class="legend-item"><div class="legend-dot" style="background:#3498db;border-radius:50%"></div>Crews</div>
        <div class="legend-item"><div class="legend-dot" style="background:#e74c3c;border-radius:50%"></div>Engines</div>
        <div class="legend-item"><div class="legend-dot" style="background:#00bcd4;border-radius:50%"></div>Helicopters</div>
        <div class="legend-item"><div class="legend-dot" style="background:#ff9800;border-radius:50%"></div>Airtankers</div>
        <div class="legend-item"><div class="legend-dot" style="background:#f1c40f;border-radius:50%"></div>Dozers</div>
        <div class="legend-item"><div class="legend-dot" style="background:#8e44ad;border-radius:50%"></div>Smokejumpers</div>
      </div>
    </div>
  </div>
</div>
<div id="step-bar">
  <div id="step-label">Step — / —</div>
  <div id="progress-track"><div id="progress-fill"></div></div>
  <div id="action-text"></div>
</div>
<div id="score-overlay">
  <h2 id="score-val">—</h2>
  <p id="score-desc"></p>
  <button onclick="document.getElementById('score-overlay').style.display='none'">Close</button>
</div>

<script>
const CELL_COLORS = {0:"#3a7d44",1:"#ff4500",2:"#4a3728",3:"#e8c84a",
                     4:"#1e90ff",5:"#9b59b6",6:"#20b2aa"};
const RES_COLORS  = {crews:"#3498db",engines:"#e74c3c",helicopters:"#00bcd4",
                     airtankers:"#ff9800",dozers:"#f1c40f",smokejumpers:"#8e44ad"};
// GRID_N and CELL_SZ are derived from the first frame so the viewer adapts to
// any grid size (easy/medium = 15×15, hard = 25×25).
let CELL_SZ = 30, GRID_N = 15;
let ws = null, svgEl, gridCells = [], unitDots = {};

function initGrid(n) {
  GRID_N = n || 15;
  // Fit inside a 450px canvas regardless of grid size
  CELL_SZ = Math.floor(450 / GRID_N);
  const sz = CELL_SZ * GRID_N;
  svgEl = document.getElementById('grid');
  svgEl.setAttribute('width',  sz);
  svgEl.setAttribute('height', sz);
  svgEl.innerHTML = '';
  gridCells = [];
  for (let r = 0; r < GRID_N; r++) {
    gridCells.push([]);
    for (let c = 0; c < GRID_N; c++) {
      const rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
      rect.setAttribute('x', c * CELL_SZ);
      rect.setAttribute('y', r * CELL_SZ);
      rect.setAttribute('width',  CELL_SZ - 1);
      rect.setAttribute('height', CELL_SZ - 1);
      rect.setAttribute('fill', CELL_COLORS[0]);
      rect.setAttribute('rx','1');
      svgEl.appendChild(rect);
      gridCells[r].push(rect);
    }
  }
  unitDots = {};
}

function renderFrame(data) {
  const states = data.cell_states, intensity = data.cell_intensity;
  // Re-initialise grid if server reports a different size
  if (states && states.length !== GRID_N) initGrid(states.length);
  for (let r = 0; r < GRID_N; r++) {
    for (let c = 0; c < GRID_N; c++) {
      const s = states[r][c];
      let fill = CELL_COLORS[s] || '#333';
      if (s === 1) {
        // Burning: interpolate yellow→red by intensity
        const iv = Math.min(1, (intensity[r][c] || 0) / 1.5);
        fill = intensityColor(iv);
      }
      gridCells[r][c].setAttribute('fill', fill);
    }
  }

  // Structure priority labels (drawn once, updated)
  svgEl.querySelectorAll('.struct-label').forEach(e => e.remove());
  for (const s of (data.structures || [])) {
    if (s.row < GRID_N && s.col < GRID_N) {
      const t = document.createElementNS('http://www.w3.org/2000/svg','text');
      t.setAttribute('x', s.col * CELL_SZ + 2);
      t.setAttribute('y', s.row * CELL_SZ + 11);
      t.setAttribute('font-size', '9');
      t.setAttribute('fill', s.status === 'burning' || s.status === 'burned' ? '#f00' : '#fff');
      t.setAttribute('class','struct-label');
      t.textContent = 'P' + s.priority;
      svgEl.appendChild(t);
    }
  }

  // Units
  svgEl.querySelectorAll('.unit-dot').forEach(e => e.remove());
  for (const u of (data.units || [])) {
    const cx = (u.current_col + 0.5) * CELL_SZ;
    const cy = (u.current_row + 0.5) * CELL_SZ;
    const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx', cx.toFixed(1));
    c.setAttribute('cy', cy.toFixed(1));
    c.setAttribute('r', u.status === 'available' ? 4 : 6);
    c.setAttribute('fill', RES_COLORS[u.resource_type] || '#fff');
    c.setAttribute('opacity', u.status === 'available' ? '0.5' : '0.95');
    c.setAttribute('stroke', '#000');
    c.setAttribute('stroke-width', '0.5');
    c.setAttribute('class','unit-dot');
    c.setAttribute('title', u.unit_id + ' ' + u.status);
    svgEl.appendChild(c);
  }

  // Stats
  document.getElementById('st-burning').textContent = data.burning_cells;
  document.getElementById('st-burned').textContent  = data.burned_cells;
  const safe = (data.structures||[]).filter(s=>s.status!=='burned'&&s.status!=='burning').length;
  const lost = (data.structures||[]).filter(s=>s.status==='burned').length;
  document.getElementById('st-safe').textContent = safe;
  document.getElementById('st-lost').textContent = lost;
  document.getElementById('st-wind').textContent =
    data.wind_speed.toFixed(1) + ' km/h ' + degToCompass(data.wind_direction);
  document.getElementById('st-temp').textContent = data.temperature.toFixed(1) + ' °C';
  document.getElementById('st-hum').textContent  = (data.humidity*100).toFixed(0) + '%';
  drawWindArrow(data.wind_speed, data.wind_direction);

  // Forecast
  const fc = data.weather_forecast || [];
  if (fc[0]) {
    document.getElementById('fc1-t').textContent = fc[0].temperature + '°C';
    document.getElementById('fc1-h').textContent = (fc[0].humidity*100).toFixed(0)+'%';
    document.getElementById('fc1-w').textContent = fc[0].wind_speed_expected+' km/h';
  }
  if (fc[1]) {
    document.getElementById('fc2-t').textContent = fc[1].temperature + '°C';
    document.getElementById('fc2-h').textContent = (fc[1].humidity*100).toFixed(0)+'%';
    document.getElementById('fc2-w').textContent = fc[1].wind_speed_expected+' km/h';
  }

  // Progress
  const pct = data.max_steps > 0 ? (data.step / data.max_steps * 100) : 0;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('step-label').textContent =
    'Step ' + data.step + ' / ' + data.max_steps +
    '  |  Task: ' + data.task_id.toUpperCase();
  const act = data.last_action_summary || '';
  document.getElementById('action-text').textContent = act.slice(0, 200);

  if (data.done && data.score != null) {
    const pct = (data.score * 100).toFixed(1);
    document.getElementById('score-val').textContent = pct + '%';
    const comp = data.components || {};
    document.getElementById('score-desc').textContent =
      'Structures: ' + ((comp.structure_component||0)*100).toFixed(1) + '%  |  ' +
      'Area: ' + ((comp.area_component||0)*100).toFixed(1) + '%  |  ' +
      'Efficiency: ' + ((comp.efficiency_component||0)*100).toFixed(1) + '%';
    document.getElementById('score-overlay').style.display = 'block';
  }
}

function intensityColor(t) {
  // t in [0,1]: yellow (low) → deep red (high)
  const r = 255;
  const g = Math.round(200 * (1 - t));
  const b = 0;
  return 'rgb('+r+','+g+','+b+')';
}

function degToCompass(deg) {
  const dirs = ['N','NE','E','SE','S','SW','W','NW'];
  return dirs[Math.round(((deg % 360) + 360) % 360 / 45) % 8];
}

function drawWindArrow(speed, dirDeg) {
  const cv = document.getElementById('wind-canvas');
  const ctx = cv.getContext('2d');
  ctx.clearRect(0, 0, 80, 80);
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, 80, 80);
  const cx = 40, cy = 40, len = 25;
  const rad = (dirDeg - 180) * Math.PI / 180; // FROM direction → arrow points toward
  const ex = cx + len * Math.sin(rad), ey = cy - len * Math.cos(rad);
  ctx.strokeStyle = '#ff6b35';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
  ctx.fillStyle = '#ff6b35';
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI*2); ctx.fill();
  const intensity = Math.min(1, speed / 40);
  ctx.fillStyle = 'rgba(255,107,53,' + (0.15 + 0.35*intensity) + ')';
  ctx.beginPath(); ctx.arc(cx, cy, 20, 0, Math.PI*2); ctx.fill();
  ctx.fillStyle = '#888';
  ctx.font = '9px monospace';
  ctx.textAlign = 'center';
  ctx.fillText(speed.toFixed(0)+' km/h', cx, 74);
}

function startEpisode() {
  if (ws) { ws.close(); ws = null; }
  document.getElementById('score-overlay').style.display = 'none';
  initGrid(GRID_N);
  const task = document.getElementById('task-sel').value;
  const delay = document.getElementById('speed-sl').value;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const url = proto+'://'+location.host+'/demo?task_id='+task+'&delay_ms='+delay;
  ws = new WebSocket(url);
  ws.onmessage = e => {
    try { renderFrame(JSON.parse(e.data)); } catch(ex) { console.error(ex); }
  };
  ws.onerror = e => console.error('WS error', e);
  ws.onclose = () => { ws = null; };
}

document.getElementById('speed-sl').oninput = function() {
  document.getElementById('speed-label').textContent = this.value + 'ms';
};

// Auto-start on load — initGrid() with default size; first frame resizes if needed
window.addEventListener('load', () => {
  initGrid(15);
  startEpisode();
});
</script>
</body>
</html>"""


@app.websocket("/demo")
async def demo_stream(
    websocket: WebSocket,
    task_id: str = "easy",
    seed: int | None = None,
    delay_ms: int = 700,
) -> None:
    """Stream a heuristic-agent episode for real-time visualization.

    Query params:
      task_id   — easy | medium | hard  (default: easy)
      seed      — integer seed; default is the reproducible baseline seed
      delay_ms  — milliseconds between steps (default: 700)

    Each WebSocket message is a JSON object containing cell_states (N×N
    int array), cell_intensity (N×N float array), fleet unit positions,
    weather, forecast, and step metadata.  The final message includes
    ``score`` and ``components`` from the /grader endpoint.
    """
    await websocket.accept()
    try:
        env = WildfireEnvironment()
        seed_val = seed if seed is not None else DEFAULT_SEEDS.get(task_id, 42)
        obs = env.reset(seed=seed_val, task_id=task_id)
        final_score_sent = False

        while True:
            # Extract raw grid arrays directly from sim for client rendering
            cell_states: list = []
            cell_intensity: list = []
            if env._sim is not None and env._sim.state is not None:
                cell_states = env._sim.state.cell_state.tolist()
                cell_intensity = [
                    [round(float(v), 3) for v in row]
                    for row in env._sim.state.intensity.tolist()
                ]

            frame: dict = {
                "step": obs.step,
                "max_steps": obs.max_steps,
                "done": obs.done,
                "task_id": obs.task_id,
                "cell_states": cell_states,
                "cell_intensity": cell_intensity,
                "wind_speed": round(obs.wind_speed, 1),
                "wind_direction": round(obs.wind_direction, 1),
                "temperature": round(obs.temperature, 1),
                "humidity": round(obs.humidity, 2),
                "burning_cells": obs.burning_cells,
                "burned_cells": obs.burned_cells,
                "structures": [s.model_dump() for s in obs.structures],
                "units": [u.model_dump() for u in obs.fleet_units],
                "last_action_summary": (obs.last_action_summary or "")[:200],
                "weather_forecast": obs.weather_forecast,
                "score": None,
                "components": None,
            }

            if obs.done and not final_score_sent:
                req = GraderRequest(
                    task_id=obs.task_id,
                    seed=seed_val,
                    step=obs.step,
                    max_steps=obs.max_steps,
                    structures=[s.model_dump() for s in obs.structures],
                    burned_cells=obs.burned_cells,
                    burning_cells=obs.burning_cells,
                )
                grade = _grade_episode(req)
                frame["score"] = grade.score
                frame["components"] = grade.components
                final_score_sent = True

            await websocket.send_json(frame)

            if obs.done:
                break

            await asyncio.sleep(delay_ms / 1000.0)
            action = _heuristic_action(obs)
            obs = env.step(action)

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass


@app.get("/viewer", response_class=HTMLResponse)
async def viewer() -> HTMLResponse:
    """Serve the live incident-command viewer.

    Open in a browser while the server is running:
        http://localhost:8000/viewer

    The page connects automatically to /demo and replays a heuristic-agent
    episode with an animated grid, weather panel, and spot-forecast display.
    Task and playback speed are adjustable without reloading.
    """
    return HTMLResponse(content=_VIEWER_HTML)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int | None = None):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    if port is None:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=8000)
        args = parser.parse_args()
        port = args.port

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

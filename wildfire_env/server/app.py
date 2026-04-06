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

import numpy as np
from fastapi import Body
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
    score: float = Field(..., ge=0.0, le=1.0, description="Final episode score 0.0–1.0")
    task_id: str
    components: dict = Field(default_factory=dict)
    description: str = ""


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
            score=0.0,
            task_id=req.task_id,
            components={},
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

    score = min(1.0, max(0.0,
        structure_score * 0.60
        + area_score * 0.30
        + efficiency_score * 0.10
    ))

    return GraderResponse(
        score=round(score, 4),
        task_id=req.task_id,
        components={
            "structure_score": round(structure_score, 4),
            "area_score": round(area_score, 4),
            "efficiency_score": round(efficiency_score, 4),
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
                        GridPoint(row=ahead_row, col=min(14, target_point.col + 2)),
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

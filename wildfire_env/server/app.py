# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Wildfire Env Environment.

Exposes the standard OpenEnv endpoints plus three submission-required extras:

    GET  /tasks     — list tasks, action schema, resource–mission compatibility
    POST /grader    — grade a completed episode (returns a score strictly within (0, 1))
    GET  /baseline  — run a deterministic heuristic agent and return scores

Standard OpenEnv endpoints (via create_app):
    POST /reset
    POST /step
    GET  /state
    GET  /schema
    WS   /ws
"""

import asyncio
import math

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
GRADER_WEIGHTS = {
    "structure": 0.45,
    "area": 0.20,
    "containment": 0.15,
    "spread_limit": 0.10,
    "efficiency": 0.10,
}
SPREAD_BUDGET_RATIOS = {
    "easy": 0.08,
    "medium": 0.14,
    "hard": 0.22,
}
ACTIVE_FIRE_BUDGET_RATIO = 0.08


def _strict_open_unit_interval(value: float) -> float:
    """Clamp a value strictly inside (0, 1) for validator compatibility."""
    return max(_SCORE_EPS, min(1.0 - _SCORE_EPS, float(value)))


def _grade_episode(req: GraderRequest) -> GraderResponse:
    """Compute a normalised grader score strictly within (0, 1).

    Scoring weights:
      - Structure protection (45%): weighted sum of saved structure priorities
        divided by total structure priority in the task.
      - Area preservation (20%): fraction of burnable terrain cells that
        were NOT burned or burning at episode end.
      - Active containment (15%): rewards ending the episode with little or
        no active fire remaining.
      - Spread limit (10%): rewards keeping damaged area below a
        difficulty-scaled incident budget.
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
                "containment_component": _SCORE_EPS,
                "spread_limit_component": _SCORE_EPS,
                "efficiency_component": _SCORE_EPS,
                "weights": GRADER_WEIGHTS,
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

    # ── Structure score (45%) ──
    lost_statuses = {"burning", "burned", "lost"}
    expected_structures: list[dict] = []
    for index, structure in enumerate(terrain.structures, start=1):
        expected_structures.append(
            {
                "structure_id": f"structure_{index}",
                "row": int(structure["row"]),
                "col": int(structure["col"]),
                "priority": int(structure["priority"]),
            }
        )

    reported_by_id: dict[str, dict] = {}
    reported_by_pos: dict[tuple[int, int], dict] = {}
    for structure in req.structures:
        structure_id = structure.get("structure_id")
        if isinstance(structure_id, str) and structure_id:
            reported_by_id[structure_id] = structure

        row = structure.get("row")
        col = structure.get("col")
        if isinstance(row, int) and isinstance(col, int):
            reported_by_pos[(row, col)] = structure

    use_order_fallback = len(req.structures) == len(expected_structures)
    total_priority = sum(item["priority"] for item in expected_structures) or 1
    saved_priority = 0
    for index, expected in enumerate(expected_structures):
        reported = reported_by_id.get(expected["structure_id"])
        if reported is None:
            reported = reported_by_pos.get((expected["row"], expected["col"]))
        if reported is None and use_order_fallback:
            reported = req.structures[index]

        status = str((reported or {}).get("status", "lost")).lower()
        if status not in lost_statuses:
            saved_priority += expected["priority"]
    structure_score = saved_priority / max(1, total_priority)

    # ── Area preservation score (20%) ──
    cells_damaged = req.burned_cells + req.burning_cells
    area_score = max(0.0, 1.0 - cells_damaged / max(1, total_burnable))

    # ── Efficiency score (10%) ── only awarded when fire is fully out
    if req.burning_cells == 0:
        efficiency_score = 1.0 - (req.step / max(1, req.max_steps))
    else:
        efficiency_score = 0.0

    # ── Active containment score (15%) ──
    # This is separate from area preservation: a policy can save structures
    # and limit burned acreage while still leaving an expanding active front.
    active_fire_budget = max(1, int(total_burnable * ACTIVE_FIRE_BUDGET_RATIO))
    containment_score = max(0.0, 1.0 - req.burning_cells / active_fire_budget)

    # ── Spread-limit score (10%) ──
    # Difficulty-scaled damage budget: easy incidents should stay tight,
    # while hard incidents get a larger acceptable footprint.
    spread_budget_cells = max(1, int(
        total_burnable * SPREAD_BUDGET_RATIOS.get(req.task_id, SPREAD_BUDGET_RATIOS["medium"])
    ))
    spread_limit_score = max(0.0, 1.0 - cells_damaged / spread_budget_cells)

    # Clamp strictly within (0, 1) — validator requires score ∈ (0.0, 1.0)
    score = _strict_open_unit_interval(
        structure_score * GRADER_WEIGHTS["structure"]
        + area_score * GRADER_WEIGHTS["area"]
        + containment_score * GRADER_WEIGHTS["containment"]
        + spread_limit_score * GRADER_WEIGHTS["spread_limit"]
        + efficiency_score * GRADER_WEIGHTS["efficiency"]
    )
    structure_component = _strict_open_unit_interval(structure_score)
    area_component = _strict_open_unit_interval(area_score)
    containment_component = _strict_open_unit_interval(containment_score)
    spread_limit_component = _strict_open_unit_interval(spread_limit_score)
    efficiency_component = _strict_open_unit_interval(efficiency_score)

    return GraderResponse(
        score=round(score, 4),
        task_id=req.task_id,
        components={
            "structure_component": round(structure_component, 4),
            "area_component": round(area_component, 4),
            "containment_component": round(containment_component, 4),
            "spread_limit_component": round(spread_limit_component, 4),
            "efficiency_component": round(efficiency_component, 4),
            "weights": GRADER_WEIGHTS,
            "saved_priority": saved_priority,
            "total_priority": total_priority,
            "reported_structures": len(req.structures),
            "expected_structures": len(expected_structures),
            "cells_damaged": cells_damaged,
            "total_burnable": total_burnable,
            "active_fire_budget": active_fire_budget,
            "spread_budget_cells": spread_budget_cells,
        },
        description=(
            f"Structure: {saved_priority}/{total_priority} priority saved. "
            f"Area: {total_burnable - cells_damaged}/{total_burnable} burnable cells intact. "
            f"Active fire: {req.burning_cells}/{active_fire_budget} containment budget. "
            f"Spread: {cells_damaged}/{spread_budget_cells} damaged-cell budget."
        ),
    )


# ---------------------------------------------------------------------------
# Heuristic baseline agent (used by /baseline endpoint)
# ---------------------------------------------------------------------------

def _clamp_cell(value: int, grid_max: int) -> int:
    return max(0, min(grid_max, value))


def _manhattan(row_a: float, col_a: float, row_b: float, col_b: float) -> float:
    return abs(float(row_a) - float(row_b)) + abs(float(col_a) - float(col_b))


def _build_line_between_fire_and_structure(
    structure,
    fire,
    grid_max: int,
    *,
    offset: int,
    half_span: int,
) -> list[GridPoint]:
    """Construct a short defensive line perpendicular to the fire-structure axis."""
    row_gap = structure.row - fire.row
    col_gap = structure.col - fire.col

    if abs(row_gap) >= abs(col_gap):
        line_row = _clamp_cell(structure.row - (offset if row_gap > 0 else -offset), grid_max)
        start = GridPoint(row=line_row, col=_clamp_cell(structure.col - half_span, grid_max))
        end = GridPoint(row=line_row, col=_clamp_cell(structure.col + half_span, grid_max))
    else:
        line_col = _clamp_cell(structure.col - (offset if col_gap > 0 else -offset), grid_max)
        start = GridPoint(row=_clamp_cell(structure.row - half_span, grid_max), col=line_col)
        end = GridPoint(row=_clamp_cell(structure.row + half_span, grid_max), col=line_col)

    return [start, end]


def _midpoint_target(row_a: int, col_a: int, row_b: int, col_b: int, grid_max: int) -> GridPoint:
    return GridPoint(
        row=_clamp_cell(int(round((row_a + row_b) / 2.0)), grid_max),
        col=_clamp_cell(int(round((col_a + col_b) / 2.0)), grid_max),
    )


def _staging_point_for_structure(structure, grid_max: int) -> GridPoint:
    return GridPoint(
        row=_clamp_cell(structure.row, grid_max),
        col=_clamp_cell(structure.col, grid_max),
    )


def _heuristic_action(obs: WildfireObservation) -> WildfireAction:
    """Structure-aware deterministic baseline for the `/baseline` endpoint."""
    grid_max = (len(obs.fuel_types[0]) - 1) if obs.fuel_types else 14
    available_units = [unit for unit in obs.fleet_units if unit.status == "available"]
    if not available_units:
        return WildfireAction()

    live_structures = [s for s in obs.structures if s.status.lower() != "burned"]
    if live_structures:
        structure_rank = []
        for structure in live_structures:
            nearest_fire = min(
                (_manhattan(structure.row, structure.col, fire.row, fire.col) for fire in obs.fire_details),
                default=math.inf,
            )
            nearest_heat = min(
                (_manhattan(structure.row, structure.col, heat.row, heat.col) for heat in obs.heat_warnings),
                default=math.inf,
            )
            status = structure.status.lower()
            threat_score = float(structure.priority) * 10.0
            if status == "burning":
                threat_score += 20.0
            if nearest_fire < math.inf:
                threat_score += max(0.0, 9.0 - nearest_fire) * (2.0 + 0.6 * structure.priority)
            if nearest_heat < math.inf:
                threat_score += max(0.0, 7.0 - nearest_heat) * (1.0 + 0.4 * structure.priority)
            structure_rank.append((threat_score, nearest_fire, structure))
        structure_rank.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        primary_structure = structure_rank[0][2]
        primary_structure_fire_dist = structure_rank[0][1]
        structure_by_id = {item[2].structure_id: item[2] for item in structure_rank}
    else:
        primary_structure = None
        primary_structure_fire_dist = math.inf
        structure_by_id = {}

    fire_rank = []
    for fire in obs.fire_details:
        fire_score = float(fire.intensity) * 3.0
        best_structure = None
        best_structure_score = -1.0
        best_structure_dist = math.inf
        for structure in live_structures:
            dist = _manhattan(fire.row, fire.col, structure.row, structure.col)
            structure_score = max(0.0, 10.0 - dist) * (1.5 + 0.7 * structure.priority)
            if structure.status.lower() == "burning":
                structure_score += 6.0
            if structure_score > best_structure_score:
                best_structure_score = structure_score
                best_structure = structure
                best_structure_dist = dist
        fire_score += max(0.0, best_structure_score)
        fire_rank.append((fire_score, best_structure_dist, fire, best_structure))
    fire_rank.sort(key=lambda item: (item[0], -item[1], item[2].intensity), reverse=True)

    attack_targets = [
        GridPoint(row=item[2].row, col=item[2].col)
        for item in fire_rank[: max(1, min(3, len(fire_rank)))]
    ]
    primary_fire = fire_rank[0][2] if fire_rank else None
    primary_fire_structure = fire_rank[0][3] if fire_rank else None

    def _attack_target(index: int) -> GridPoint:
        return attack_targets[index % len(attack_targets)]

    assignments = []
    attack_idx = 0
    if primary_fire is None:
        if primary_structure is None:
            center = GridPoint(row=grid_max // 2, col=grid_max // 2)
            return WildfireAction(assignments=[
                ResourceAssignment(
                    unit_id=unit.unit_id,
                    mission_type="staging",
                    target=TargetSpec(target_kind="point", point=center),
                )
                for unit in available_units
            ])

        stage_point = _staging_point_for_structure(primary_structure, grid_max)
        for unit in available_units:
            rtype = unit.resource_type
            if rtype in {"engines", "crews", "dozers", "smokejumpers"}:
                assignments.append(
                    ResourceAssignment(
                        unit_id=unit.unit_id,
                        mission_type="staging",
                        target=TargetSpec(target_kind="point", point=stage_point),
                    )
                )
            elif rtype == "helicopters":
                assignments.append(
                    ResourceAssignment(
                        unit_id=unit.unit_id,
                        mission_type="point_protection",
                        target=TargetSpec(target_kind="structure", structure_id=primary_structure.structure_id),
                        commitment_steps=2,
                    )
                )
            elif rtype == "airtankers":
                assignments.append(
                    ResourceAssignment(
                        unit_id=unit.unit_id,
                        mission_type="staging",
                        target=TargetSpec(target_kind="point", point=stage_point),
                    )
                )
        return WildfireAction(assignments=assignments)

    threat_structure = primary_fire_structure or primary_structure
    high_threat_structure = (
        threat_structure is not None
        and (
            threat_structure.status.lower() == "burning"
            or _manhattan(primary_fire.row, primary_fire.col, threat_structure.row, threat_structure.col) <= 4
            or primary_structure_fire_dist <= 4
        )
    )

    for unit in available_units:
        rtype = unit.resource_type
        if rtype in {"crews", "smokejumpers"}:
            assignments.append(
                ResourceAssignment(
                    unit_id=unit.unit_id,
                    mission_type="direct_attack",
                    target=TargetSpec(target_kind="point", point=_attack_target(attack_idx)),
                    commitment_steps=2 if high_threat_structure else 1,
                )
            )
            attack_idx += 1
            continue

        if rtype == "engines":
            if high_threat_structure and threat_structure is not None:
                assignments.append(
                    ResourceAssignment(
                        unit_id=unit.unit_id,
                        mission_type="point_protection",
                        target=TargetSpec(target_kind="structure", structure_id=threat_structure.structure_id),
                        commitment_steps=2,
                    )
                )
            else:
                assignments.append(
                    ResourceAssignment(
                        unit_id=unit.unit_id,
                        mission_type="wet_line",
                        target=TargetSpec(
                            target_kind="line",
                            waypoints=_build_line_between_fire_and_structure(
                                threat_structure or primary_structure or list(structure_by_id.values())[0],
                                primary_fire,
                                grid_max,
                                offset=1,
                                half_span=2,
                            ) if (threat_structure or primary_structure) is not None else [
                                GridPoint(row=_clamp_cell(primary_fire.row - 1, grid_max), col=_clamp_cell(primary_fire.col - 2, grid_max)),
                                GridPoint(row=_clamp_cell(primary_fire.row - 1, grid_max), col=_clamp_cell(primary_fire.col + 2, grid_max)),
                            ],
                        ),
                    )
                )
            continue

        if rtype == "helicopters":
            area_center = (
                _midpoint_target(primary_fire.row, primary_fire.col, threat_structure.row, threat_structure.col, grid_max)
                if high_threat_structure and threat_structure is not None
                else GridPoint(row=primary_fire.row, col=primary_fire.col)
            )
            assignments.append(
                ResourceAssignment(
                    unit_id=unit.unit_id,
                    mission_type="water_drop",
                    target=TargetSpec(target_kind="area", center=area_center, radius=1),
                    drop_configuration="salvo",
                )
            )
            continue

        if rtype == "airtankers":
            area_center = (
                _midpoint_target(primary_fire.row, primary_fire.col, threat_structure.row, threat_structure.col, grid_max)
                if threat_structure is not None
                else GridPoint(row=primary_fire.row, col=primary_fire.col)
            )
            assignments.append(
                ResourceAssignment(
                    unit_id=unit.unit_id,
                    mission_type="retardant_drop",
                    target=TargetSpec(target_kind="area", center=area_center, radius=2),
                    drop_configuration="trail" if high_threat_structure else "salvo",
                )
            )
            continue

        if rtype == "dozers":
            if threat_structure is not None:
                line = _build_line_between_fire_and_structure(
                    threat_structure,
                    primary_fire,
                    grid_max,
                    offset=2,
                    half_span=3,
                )
            else:
                line = [
                    GridPoint(row=_clamp_cell(primary_fire.row - 2, grid_max), col=_clamp_cell(primary_fire.col - 3, grid_max)),
                    GridPoint(row=_clamp_cell(primary_fire.row - 2, grid_max), col=_clamp_cell(primary_fire.col + 3, grid_max)),
                ]
            assignments.append(
                ResourceAssignment(
                    unit_id=unit.unit_id,
                    mission_type="line_construction",
                    target=TargetSpec(target_kind="line", waypoints=line),
                    commitment_steps=2,
                )
            )

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
    """Grade a completed episode and return a score strictly within (0, 1).

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
          "burned_cells": 9,
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
// any grid size (easy = 15×15, medium = 20×20, hard = 25×25).
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
  const plan = data.model_plan || '';
  // When a replay frame includes the model's plan field, show it inline
  // before the env's action summary so viewers see the agent's reasoning.
  const combined = plan ? ('PLAN: ' + plan + '  |  ' + act) : act;
  document.getElementById('action-text').textContent = combined.slice(0, 280);

  if (data.done && data.score != null) {
    const pct = (data.score * 100).toFixed(1);
    document.getElementById('score-val').textContent = pct + '%';
    const comp = data.components || {};
    document.getElementById('score-desc').textContent =
      'Structures: ' + ((comp.structure_component||0)*100).toFixed(1) + '%  |  ' +
      'Area: ' + ((comp.area_component||0)*100).toFixed(1) + '%  |  ' +
      'Active: ' + ((comp.containment_component||0)*100).toFixed(1) + '%  |  ' +
      'Spread: ' + ((comp.spread_limit_component||0)*100).toFixed(1) + '%  |  ' +
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
  // ?replay=<path> switches the viewer from live heuristic streaming to
  // replaying a JSON file captured by capture_replay.py.
  const params = new URLSearchParams(location.search);
  const replay = params.get('replay');
  let url;
  if (replay) {
    url = proto+'://'+location.host+'/demo_replay?file='+encodeURIComponent(replay)+'&delay_ms='+delay;
  } else {
    url = proto+'://'+location.host+'/demo?task_id='+task+'&delay_ms='+delay;
  }
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


_REPLAY_SEARCH_DIRS = ("submission_artifacts", "replays", "artifacts")


def _resolve_replay_path(file_arg: str) -> str | None:
    """Resolve a replay-file query string to an absolute path on disk.

    Tries (in order):
      1. as-given (relative to cwd or absolute)
      2. each entry in _REPLAY_SEARCH_DIRS

    Returns None if no resolution exists or the resolved path escapes the
    cwd (basic path-traversal guard — replays are static JSON, but the
    endpoint is publicly reachable on a deployed Space).
    """
    import os as _os
    cwd = _os.path.abspath(_os.getcwd())
    candidates = [file_arg] + [_os.path.join(d, file_arg) for d in _REPLAY_SEARCH_DIRS]
    for cand in candidates:
        full = _os.path.abspath(cand)
        if not full.startswith(cwd):
            continue
        if _os.path.isfile(full):
            return full
    return None


@app.websocket("/demo_replay")
async def demo_replay(
    websocket: WebSocket,
    file: str,
    delay_ms: int = 700,
) -> None:
    """Stream a previously-captured episode JSON for visualization.

    Captured by ``capture_replay.py`` — see that script for usage. The frame
    schema is identical to /demo so the viewer renders without changes.

    Query params:
      file      — relative path to the captured frames JSON
      delay_ms  — milliseconds between frames (default: 700)
    """
    import json as _json
    await websocket.accept()
    try:
        full_path = _resolve_replay_path(file)
        if full_path is None:
            await websocket.send_json({"error": f"replay file not found: {file}"})
            return
        with open(full_path, "r", encoding="utf-8") as fh:
            payload = _json.load(fh)
        frames = payload.get("frames", [])
        if not frames:
            await websocket.send_json({"error": "replay payload contains no frames"})
            return
        for idx, frame in enumerate(frames):
            await websocket.send_json(frame)
            if frame.get("done"):
                break
            if idx + 1 < len(frames):
                await asyncio.sleep(delay_ms / 1000.0)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await websocket.send_json({"error": str(exc)})
        except Exception:
            pass


@app.get("/replays")
async def list_replays() -> dict:
    """Index any captured-episode JSONs reachable under the search dirs.

    Helpful for the demo video workflow: lets you confirm a replay file
    is visible to the server before opening /viewer?replay=...
    """
    import os as _os
    found: list[dict] = []
    cwd = _os.path.abspath(_os.getcwd())
    for d in _REPLAY_SEARCH_DIRS:
        dir_path = _os.path.join(cwd, d)
        if not _os.path.isdir(dir_path):
            continue
        for name in sorted(_os.listdir(dir_path)):
            if name.endswith(".json"):
                rel = _os.path.relpath(_os.path.join(dir_path, name), cwd).replace(_os.sep, "/")
                found.append({"file": rel, "size": _os.path.getsize(_os.path.join(dir_path, name))})
    return {"replays": found, "search_dirs": list(_REPLAY_SEARCH_DIRS)}


@app.get("/viewer", response_class=HTMLResponse)
async def viewer() -> HTMLResponse:
    """Serve the live incident-command viewer.

    Open in a browser while the server is running:
        http://localhost:8000/viewer                           (live heuristic)
        http://localhost:8000/viewer?replay=replays/foo.json   (captured replay)

    The page connects to /demo (live heuristic) by default. When a
    ?replay=<path> query param is present, it streams via /demo_replay
    instead, replaying a JSON captured by capture_replay.py.
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

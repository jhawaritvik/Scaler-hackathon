# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenEnv environment wrapper for the wildfire simulator."""

from __future__ import annotations

import math
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        FleetMissionObservation,
        FleetStatusObservation,
        FleetUnitObservation,
        FireCellObservation,
        HeatWarningObservation,
        OutpostObservation,
        ResourceAssignment,
        StructureObservation,
        TargetSpec,
        WildfireAction,
        WildfireObservation,
    )
    from .fire_simulation import (
        FUEL_BRUSH,
        FUEL_FOREST,
        FUEL_GRASS,
        FUEL_NONE,
        STATE_BURNING,
        STATE_FIREBREAK,
        STATE_WATER,
        STATE_BURNED,
        STATE_SUPPRESSED,
        FireSimulation,
        SIMULATION_STEP_MINUTES,
    )
    from .resources import (
        DISPATCH_MODEL,
        FleetUnit,
        build_initial_fleet,
        cell_distance,
        minutes_to_steps,
    )
    from .terrain import DIFFICULTY_SPECS, DEFAULT_SEEDS, get_task_config, generate_terrain
except ImportError:
    from models import (
        FleetMissionObservation,
        FleetStatusObservation,
        FleetUnitObservation,
        FireCellObservation,
        HeatWarningObservation,
        OutpostObservation,
        ResourceAssignment,
        StructureObservation,
        TargetSpec,
        WildfireAction,
        WildfireObservation,
    )
    from server.fire_simulation import (
        FUEL_BRUSH,
        FUEL_FOREST,
        FUEL_GRASS,
        FUEL_NONE,
        STATE_BURNING,
        STATE_FIREBREAK,
        STATE_WATER,
        STATE_BURNED,
        STATE_SUPPRESSED,
        FireSimulation,
        SIMULATION_STEP_MINUTES,
    )
    from server.resources import (
        DISPATCH_MODEL,
        FleetUnit,
        build_initial_fleet,
        cell_distance,
        minutes_to_steps,
    )
    from server.terrain import DIFFICULTY_SPECS, DEFAULT_SEEDS, get_task_config, generate_terrain


TASK_GOALS = {
    "easy": "Keep both low-priority structures safe and contain the initial fire early.",
    "medium": "Protect the mixed-priority structures while balancing suppression and containment.",
    "hard": "Survive a multi-ignition wind-driven incident with scarce resources and high-value assets.",
}

# Dispatch costs — reflects real resource expense differences.
# Aerial assets cost more (fuel, maintenance, flight hours).
# Backfire carries highest ground cost (reflects risk, IC authorization overhead).
MISSION_COST = {
    "direct_attack": -0.003,
    "line_construction": -0.005,
    "wet_line": -0.004,
    "water_drop": -0.008,
    "retardant_drop": -0.012,
    "point_protection": -0.004,
    "backfire": -0.015,
    "staging": -0.001,
}

# Per-resource dispatch surcharge. This keeps the reward aware of which
# assets were committed, not just which mission label was chosen.
RESOURCE_DISPATCH_COST = {
    "crews": -0.0005,
    "engines": -0.0010,
    "helicopters": -0.0025,
    "airtankers": -0.0040,
    "dozers": -0.0015,
    "smokejumpers": -0.0025,
}

# Resource → allowed missions.  Based on NWCG resource typing.
# Hand crews: versatile ground force, can set backfires (drip torch ops)
# Engines: pump-and-roll, foam spray, mobile attack
# Helicopters: water drops only (bucket ops)
# Air tankers: retardant drops only (fixed-wing, can't do water scoop)
# Dozers: firebreak construction only
# Smokejumpers: rapid initial attack, hand line construction
RESOURCE_MISSION_COMPATIBILITY = {
    "crews":        {"direct_attack", "line_construction", "point_protection", "backfire", "staging"},
    "engines":      {"direct_attack", "wet_line", "point_protection", "backfire", "staging"},
    "helicopters":  {"water_drop", "point_protection", "staging"},
    "airtankers":   {"retardant_drop", "staging"},
    "dozers":       {"line_construction", "point_protection", "staging"},
    "smokejumpers": {"direct_attack", "line_construction", "point_protection", "staging"},
}

# Resource types that use drop_configuration (salvo/trail)
AERIAL_DROP_RESOURCES = {"helicopters", "airtankers"}

INVALID_ACTION_PENALTY = -0.05
LOW_IMPACT_ACTION_PENALTY = -0.02

# ── LCES safety check ──
# LCES (Lookouts / Communications / Escape Routes / Safety Zones) is the
# foundational safety doctrine for wildland firefighters.
# Source: NWCG 10 Standard Firefighting Orders #6 ("Maintain prompt
# communications with your forces, your supervisor, and adjoining forces")
# and #8 ("Give terrain and fuels the upper hand"), plus the LCES
# framework from the NWCG Incident Response Pocket Guide (PMS 461).
#
# Quantitative safety zone criterion from Butler & Cohen (1998):
#   "Calculated Safety Zone Dimensions and Application to Firefighter Safety"
#   USDA FS Int. Res. Station. Key result: safety zone diameter ≥ 4× flame
#   height (radiant heat only, flat terrain, no wind). With wind the
#   required clearance grows substantially.
#
# At our 100 m cell pitch and the typical surface fire flame heights this
# model produces (grass 1-3 m, brush 3-8 m, forest 5-15 m), a single
# unburning non-fuel cell technically satisfies the geometric threshold.
# The practical LCES requirement we enforce is therefore: when a burning
# cell is IMMEDIATELY adjacent (≤ 1 cell) to the assignment target, an
# escape cell (FIREBREAK / WATER / BURNED / SUPPRESSED) must exist within
# 2 cells of the target — giving crews a reachable safety zone before the
# fire cuts off their route.  Grid edge counts as open terrain escape.
LCES_GROUND_RESOURCES = {"crews", "engines", "smokejumpers", "dozers"}
LCES_HAZARDOUS_MISSIONS = {"direct_attack", "line_construction", "backfire", "wet_line"}
LCES_THREAT_RADIUS = 1    # fire this many cells away = active threat
LCES_ESCAPE_RADIUS = 2    # safety zone must be within this radius
LCES_VIOLATION_PENALTY = -0.03  # per NWCG 10-Orders #6; not a hard block

POINT_PROTECTION_RADIUS = 1
MAX_POLYGON_VERTICES = 8

# Cells treated per work step, by resource type.
# Source: NWCG fireline production rates (chains/hr), scaled to our
# ~30-min step and ~100 m cell pitch.
CREW_CELLS_PER_STEP = 2
ENGINE_CELLS_PER_STEP = 3         # engines are faster than hand crews
SMOKEJUMPER_CELLS_PER_STEP = 2    # same as crew (hand tools)
HELICOPTER_TRAIL_CELLS_PER_STEP = 2
AIRTANKER_TRAIL_CELLS_PER_STEP = 4  # LAT covers ~1/4 mile line per run

# ── Dozer firebreak production rates ──
# Source: NWCG 2021 Fireline Production Rates (San Dimas Tech Tip).
# Values represent sustained single-pass rates for a Type II (D6-class)
# dozer in chains/hour, by Anderson fuel model family.
DOZER_LINE_RATE_CHAINS_PER_HOUR = {
    FUEL_NONE: 120.0,    # cleared/mineral soil — fastest
    FUEL_GRASS: 90.0,    # fuel models 1-3
    FUEL_BRUSH: 55.0,    # fuel models 4-7 (chaparral/brush)
    FUEL_FOREST: 40.0,   # fuel models 8-10 (timber litter)
}
DOZER_CHAIN_LENGTH_PER_CELL = 5.0

# ── Backfire anchor validation ──
# Backfire lines must originate from a secure anchor point to prevent
# flanking (NWCG Fireline Handbook PMS 410-1, Section 8).
BACKFIRE_ANCHOR_STATES = {STATE_FIREBREAK, STATE_WATER, STATE_BURNED, STATE_SUPPRESSED}

# All resource type keys for iteration
ALL_RESOURCE_TYPES = (
    "crews", "engines", "helicopters", "airtankers", "dozers", "smokejumpers",
)


class WildfireEnvironment(Environment):
    """OpenEnv wrapper around the wildfire simulation core."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_order = list(DIFFICULTY_SPECS.keys())
        self._task_cursor = 0
        self._episode_counter: dict[str, int] = {t: 0 for t in self._task_order}

        self._task_id = "easy"
        self._task_seed: int = DEFAULT_SEEDS["easy"]
        self._sim: FireSimulation | None = None
        self._fleet_units: list[FleetUnit] = []
        self._resource_totals: dict[str, int] = self._empty_resource_counts()
        self._structures_by_id: dict[str, dict] = {}
        self._outposts: list[dict] = []
        self._last_action_summary = ""
        self._last_action_error: str | None = None
        self._total_reward = 0.0

    def close(self) -> None:
        """Release simulation resources."""
        self._sim = None
        self._fleet_units.clear()
        self._structures_by_id.clear()
        self._outposts.clear()

    def _select_task_id(self, task_id: str | None = None) -> str:
        if task_id and task_id in DIFFICULTY_SPECS:
            return task_id
        tid = self._task_order[self._task_cursor % len(self._task_order)]
        self._task_cursor += 1
        return tid

    def _empty_resource_counts(self) -> dict[str, int]:
        return {rt: 0 for rt in ALL_RESOURCE_TYPES}

    def _available_resource_counts(self) -> dict[str, int]:
        counts = self._empty_resource_counts()
        for unit in self._fleet_units:
            if unit.status == "available":
                counts[unit.resource_type] += 1
        return counts

    def _fleet_status(self) -> list[FleetStatusObservation]:
        statuses: list[FleetStatusObservation] = []
        for resource_type in ALL_RESOURCE_TYPES:
            total_units = self._resource_totals.get(resource_type, 0)
            available_units = 0
            en_route_units = 0
            operating_units = 0
            returning_units = 0

            for unit in self._fleet_units:
                if unit.resource_type != resource_type:
                    continue
                if unit.status == "available":
                    available_units += 1
                elif unit.status == "en_route":
                    en_route_units += 1
                elif unit.status == "operating":
                    operating_units += 1
                elif unit.status == "returning":
                    returning_units += 1

            statuses.append(
                FleetStatusObservation(
                    resource_type=resource_type,
                    total_units=total_units,
                    available_units=available_units,
                    en_route_units=en_route_units,
                    operating_units=operating_units,
                    returning_units=returning_units,
                )
            )

        return statuses

    def _active_missions(self) -> list[FleetMissionObservation]:
        missions: list[FleetMissionObservation] = []

        for unit in self._fleet_units:
            if unit.status == "available":
                continue

            if unit.status == "en_route":
                summary = f"{unit.unit_id} en route to {unit.target_summary}"
                eta_steps = unit.eta_steps
            elif unit.status == "operating":
                summary = f"{unit.unit_id} operating on {unit.target_summary}"
                eta_steps = unit.work_steps_remaining
            else:
                summary = f"{unit.unit_id} returning to standby"
                eta_steps = unit.eta_steps

            missions.append(
                FleetMissionObservation(
                    unit_id=unit.unit_id,
                    resource_type=unit.resource_type,
                    status=unit.status,
                    mission_type=unit.mission_type,
                    target_kind=unit.target_kind,
                    target_summary=unit.target_summary,
                    eta_steps=max(0, eta_steps),
                    eta_minutes=max(0.0, eta_steps * SIMULATION_STEP_MINUTES),
                    summary=summary,
                )
            )

        missions.sort(key=lambda mission: (mission.resource_type, mission.unit_id))
        return missions

    def _fleet_units_observation(self) -> list[FleetUnitObservation]:
        units: list[FleetUnitObservation] = []

        for unit in self._fleet_units:
            if unit.status == "available":
                available_in_steps = 0
            elif unit.status == "en_route":
                available_in_steps = unit.eta_steps + unit.work_steps_remaining
                if unit.return_after_mission:
                    available_in_steps += unit.return_steps_remaining
            elif unit.status == "operating":
                available_in_steps = unit.work_steps_remaining
                if unit.return_after_mission:
                    available_in_steps += unit.return_steps_remaining
            else:
                available_in_steps = unit.eta_steps

            units.append(
                FleetUnitObservation(
                    unit_id=unit.unit_id,
                    resource_type=unit.resource_type,
                    status=unit.status,
                    current_row=round(unit.position_row, 2),
                    current_col=round(unit.position_col, 2),
                    standby_row=round(unit.standby_row, 2),
                    standby_col=round(unit.standby_col, 2),
                    base_row=round(unit.base_row, 2),
                    base_col=round(unit.base_col, 2),
                    mission_type=unit.mission_type,
                    target_kind=unit.target_kind,
                    target_summary=unit.target_summary,
                    available_in_steps=max(0, available_in_steps),
                    available_in_minutes=max(0.0, available_in_steps * SIMULATION_STEP_MINUTES),
                    outpost_id=unit.outpost_id,
                    missions_completed=unit.missions_completed,
                )
            )

        units.sort(key=lambda item: (item.resource_type, item.unit_id))
        return units

    def _outpost_observations(self) -> list[OutpostObservation]:
        obs: list[OutpostObservation] = []
        for op in self._outposts:
            unit_ids = [
                u.unit_id for u in self._fleet_units
                if u.outpost_id == op["outpost_id"]
            ]
            obs.append(OutpostObservation(
                outpost_id=op["outpost_id"],
                row=op["row"],
                col=op["col"],
                is_airbase=op.get("is_airbase", False),
                unit_ids=unit_ids,
            ))
        return obs

    def _build_action_guide(
        self,
        visible_cells: set[tuple[int, int]] | None = None,
    ) -> str:
        """Build a natural-language per-step action guide for LLM agents.

        Summarises which units are ready to dispatch (and what missions they
        support), which are committed (with ETAs), and the current tactical
        situation.  This lives in the observation so the agent doesn't have
        to memorise the resource–mission compatibility matrix.
        """
        lines: list[str] = []

        # ── Outposts ──
        if self._outposts:
            lines.append("OUTPOSTS:")
            for op in self._outposts:
                kind = "airbase" if op.get("is_airbase") else "ground"
                unit_ids = [
                    u.unit_id for u in self._fleet_units
                    if u.outpost_id == op["outpost_id"]
                ]
                lines.append(
                    f"  {op['outpost_id']} ({op['row']:.0f},{op['col']:.0f}) "
                    f"[{kind}]: {', '.join(unit_ids) if unit_ids else 'empty'}"
                )

        # ── Available units ──
        available = sorted(
            [u for u in self._fleet_units if u.status == "available"],
            key=lambda u: (u.resource_type, u.unit_id),
        )
        if available:
            lines.append("AVAILABLE (can dispatch now):")
            for unit in available:
                missions = " | ".join(
                    sorted(RESOURCE_MISSION_COMPATIBILITY[unit.resource_type])
                )
                lines.append(f"  {unit.unit_id} [{unit.resource_type}]: {missions}")
        else:
            lines.append("AVAILABLE: none — all units committed")

        # ── Committed units ──
        busy = sorted(
            [u for u in self._fleet_units if u.status != "available"],
            key=lambda u: (u.resource_type, u.unit_id),
        )
        if busy:
            lines.append("COMMITTED:")
            for unit in busy:
                if unit.status == "en_route":
                    lines.append(
                        f"  {unit.unit_id} en_route → {unit.mission_type} "
                        f"at ({unit.target_row},{unit.target_col}), "
                        f"arrives in {unit.eta_steps} step(s)"
                    )
                elif unit.status == "operating":
                    lines.append(
                        f"  {unit.unit_id} operating → {unit.mission_type}, "
                        f"done in {unit.work_steps_remaining} step(s)"
                    )
                else:  # returning
                    lines.append(
                        f"  {unit.unit_id} returning, "
                        f"available in {unit.eta_steps} step(s)"
                    )

        # ── Tactical situation ──
        lines.append("")
        if self._sim is not None and self._sim.state is not None:
            st = self._sim.state
            wind_dir = st.wind_direction
            compass = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][
                int((wind_dir + 22.5) // 45) % 8
            ]
            lines.append(
                f"FIRE: {st.total_burning} burning | {st.total_burned} burned. "
                f"Wind: {st.wind_speed:.0f} km/h from {compass} ({wind_dir:.0f}°)"
            )
            if self._sim.terrain.structures:
                struct_parts: list[str] = []
                for s in self._sim.terrain.structures:
                    r, c = s["row"], s["col"]
                    sid = s.get("structure_id", f"s_{r}_{c}")
                    cell = int(st.cell_state[r, c])
                    if cell == STATE_BURNING:
                        label = "BURNING"
                    elif cell == STATE_BURNED:
                        label = "LOST"
                    else:
                        label = "safe"
                    struct_parts.append(f"{sid}(p{s['priority']},{label})")
                lines.append(f"STRUCTURES: {' | '.join(struct_parts)}")
        else:
            lines.append("FIRE: simulation not started")

        # ── Visibility / fog-of-war ──
        if visible_cells is not None and self._sim is not None:
            grid_size = self._sim.size
            total_cells = grid_size * grid_size
            pct = round(100.0 * len(visible_cells) / total_cells)
            lines.append(
                f"VISIBILITY: {len(visible_cells)}/{total_cells} cells ({pct}%) — "
                "cells outside sensor range shown as '?' in grid."
            )

        return "\n".join(lines)

    def _build_observation(
        self,
        reward: float = 0.0,
        done: bool = False,
    ) -> WildfireObservation:
        available_resources = self._available_resource_counts()
        fleet_status = self._fleet_status()
        active_missions = self._active_missions()
        fleet_units = self._fleet_units_observation()
        outposts = self._outpost_observations()

        if self._sim is None:
            return WildfireObservation(
                task_id=self._task_id,
                goal=TASK_GOALS.get(self._task_id, ""),
                grid="",
                step=0,
                max_steps=1,
                step_minutes=SIMULATION_STEP_MINUTES,
                elapsed_minutes=0.0,
                time_of_day=0.0,
                wind_speed=0.0,
                wind_direction=0.0,
                temperature=0.0,
                humidity=0.0,
                atmospheric_dryness_index=0.0,
                airflow_potential_peak=0.0,
                burning_cells=0,
                burned_cells=0,
                structures_remaining=0,
                structures_lost=0,
                resources_remaining=available_resources,
                resource_totals=self._resource_totals.copy(),
                fleet_status=fleet_status,
                fleet_units=fleet_units,
                active_missions=active_missions,
                outposts=outposts,
                last_action_summary=self._last_action_summary,
                last_action_error=self._last_action_error,
                action_guide=self._build_action_guide(),
                weather_forecast=[],
                visible_cell_count=0,
                fog_of_war_active=False,
                done=done,
                reward=reward,
            )

        visible_cells = self._compute_visible_cells()
        obs = self._sim.get_observation_dict(visible_cells=visible_cells)
        fire_details = [FireCellObservation(**item) for item in obs.get("fire_details", [])]
        structures = [StructureObservation(**item) for item in obs.get("structures", [])]
        heat_warnings = [
            HeatWarningObservation(**item) for item in obs.get("heat_warnings", [])
        ]

        return WildfireObservation(
            task_id=self._task_id,
            goal=TASK_GOALS[self._task_id],
            grid=obs.get("grid", ""),
            step=obs.get("step", 0),
            max_steps=obs.get("max_steps", 1),
            step_minutes=obs.get("step_minutes", SIMULATION_STEP_MINUTES),
            elapsed_minutes=obs.get(
                "elapsed_minutes",
                obs.get("step", 0) * SIMULATION_STEP_MINUTES,
            ),
            time_of_day=obs.get("time_of_day", 0.0),
            wind_speed=obs.get("wind_speed", 0.0),
            wind_direction=obs.get("wind_direction", 0.0),
            temperature=obs.get("temperature", 0.0),
            humidity=obs.get("humidity", 0.0),
            atmospheric_dryness_index=obs.get("atmospheric_dryness_index", 0.0),
            airflow_potential_peak=obs.get("airflow_potential_peak", 0.0),
            burning_cells=obs.get("burning_cells", 0),
            burned_cells=obs.get("burned_cells", 0),
            structures_remaining=obs.get("structures_remaining", 0),
            structures_lost=obs.get("structures_lost", 0),
            resources_remaining=available_resources,
            resource_totals=self._resource_totals.copy(),
            fleet_status=fleet_status,
            fleet_units=fleet_units,
            active_missions=active_missions,
            last_action_summary=self._last_action_summary,
            last_action_error=self._last_action_error,
            outposts=outposts,
            fire_details=fire_details,
            structures=structures,
            heat_warnings=heat_warnings,
            elevation=obs.get("elevation", []),
            fuel_types=obs.get("fuel_types", []),
            action_guide=self._build_action_guide(visible_cells=visible_cells),
            weather_forecast=obs.get("weather_forecast", []),
            visible_cell_count=len(visible_cells),
            fog_of_war_active=True,
            done=done,
            reward=reward,
            metadata={
                "task_id": self._task_id,
                "seed": self._task_seed,
                "goal": TASK_GOALS[self._task_id],
                "total_reward": round(self._total_reward, 4),
            },
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> WildfireObservation:
        """Reset to a new episode.

        Args:
            seed: Deterministic seed.  If ``None``, uses the default seed for
                  the selected task so that baselines are reproducible.
            episode_id: Optional caller-provided episode identifier.
            **kwargs: May contain ``task_id`` to select a specific difficulty
                      (``"easy"``, ``"medium"``, ``"hard"``).
        """
        task_id_arg = kwargs.get("task_id", None)
        self._task_id = self._select_task_id(task_id_arg)

        if seed is None:
            seed = DEFAULT_SEEDS.get(self._task_id, 42)
        self._task_seed = seed

        config = get_task_config(self._task_id, seed)
        terrain = generate_terrain(config)
        for index, structure in enumerate(terrain.structures, start=1):
            structure["structure_id"] = structure.get("structure_id", f"structure_{index}")

        self._structures_by_id = {
            structure["structure_id"]: structure for structure in terrain.structures
        }

        self._outposts = terrain.outposts

        self._sim = FireSimulation(terrain)
        sim_state = self._sim.reset()

        warmup_steps = max(0, int(getattr(config, "warmup_steps", 0)))
        executed_warmup = 0
        for _ in range(warmup_steps):
            if sim_state.done:
                break
            sim_state = self._sim.tick()
            executed_warmup += 1

        eid = episode_id or str(uuid4())
        self._state = State(episode_id=eid, step_count=sim_state.step)
        self._resource_totals = {**self._empty_resource_counts(), **dict(config.resources)}
        self._fleet_units = build_initial_fleet(
            config.resources, config.grid_size, outposts=config.outposts,
        )
        if executed_warmup > 0:
            self._last_action_summary = (
                f"started task '{self._task_id}' (seed={seed}) after "
                f"{executed_warmup} warmup step(s) "
                f"({executed_warmup * SIMULATION_STEP_MINUTES:.0f} incident minutes) "
                f"with a {SIMULATION_STEP_MINUTES:.0f}-minute incident timestep"
            )
        else:
            self._last_action_summary = (
                f"started task '{self._task_id}' (seed={seed}) with a "
                f"{SIMULATION_STEP_MINUTES:.0f}-minute incident timestep"
            )
        self._last_action_error = None
        self._total_reward = 0.0

        return self._build_observation(reward=0.0, done=sim_state.done)

    def _get_unit_by_id(self, unit_id: str) -> FleetUnit | None:
        for unit in self._fleet_units:
            if unit.unit_id == unit_id:
                return unit
        return None

    def _get_structure_by_id(self, structure_id: str) -> dict | None:
        return self._structures_by_id.get(structure_id)

    def _require_sim(self) -> FireSimulation:
        if self._sim is None:
            self.reset()
        assert self._sim is not None
        return self._sim

    def _validate_cell(self, row: int, col: int) -> None:
        sim = self._require_sim()
        if not (0 <= row < sim.size and 0 <= col < sim.size):
            raise ValueError(f"target cell ({row}, {col}) is out of bounds")

    def _bresenham_segment(self, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        row0, col0 = start
        row1, col1 = end
        cells: list[tuple[int, int]] = []

        dcol = abs(col1 - col0)
        drow = abs(row1 - row0)
        col = col0
        row = row0
        scol = 1 if col0 < col1 else -1
        srow = 1 if row0 < row1 else -1

        if dcol > drow:
            err = dcol / 2.0
            while col != col1:
                cells.append((row, col))
                err -= drow
                if err < 0:
                    row += srow
                    err += dcol
                col += scol
        else:
            err = drow / 2.0
            while row != row1:
                cells.append((row, col))
                err -= dcol
                if err < 0:
                    col += scol
                    err += drow
                row += srow

        cells.append((row1, col1))
        return cells

    def _dedupe_cells(self, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
        seen: set[tuple[int, int]] = set()
        ordered: list[tuple[int, int]] = []
        for cell in cells:
            if cell not in seen:
                seen.add(cell)
                ordered.append(cell)
        return ordered

    def _cells_for_area(self, center_row: int, center_col: int, radius: int) -> list[tuple[int, int]]:
        sim = self._require_sim()
        cells: list[tuple[int, int]] = []
        for row in range(max(0, center_row - radius), min(sim.size, center_row + radius + 1)):
            for col in range(max(0, center_col - radius), min(sim.size, center_col + radius + 1)):
                if math.hypot(row - center_row, col - center_col) <= radius + 0.01:
                    cells.append((row, col))
        return cells

    def _cells_for_ring(self, center_row: int, center_col: int, radius: int) -> list[tuple[int, int]]:
        sim = self._require_sim()
        cells: list[tuple[int, int]] = []
        inner = max(0.0, radius - 0.75)
        outer = radius + 0.75
        for row in range(max(0, center_row - radius - 1), min(sim.size, center_row + radius + 2)):
            for col in range(max(0, center_col - radius - 1), min(sim.size, center_col + radius + 2)):
                dist = math.hypot(row - center_row, col - center_col)
                if inner < dist <= outer:
                    cells.append((row, col))
        return cells

    def _point_in_polygon(
        self,
        point_row: float,
        point_col: float,
        vertices: list[tuple[int, int]],
    ) -> bool:
        inside = False
        count = len(vertices)
        for idx in range(count):
            row1, col1 = vertices[idx]
            row2, col2 = vertices[(idx + 1) % count]
            intersects = ((col1 > point_col) != (col2 > point_col)) and (
                point_row
                < (row2 - row1) * (point_col - col1) / max(1e-9, col2 - col1) + row1
            )
            if intersects:
                inside = not inside
        return inside

    def _cells_for_polygon_fill(self, vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
        sim = self._require_sim()
        rows = [row for row, _ in vertices]
        cols = [col for _, col in vertices]
        cells: list[tuple[int, int]] = []
        for row in range(max(0, min(rows)), min(sim.size, max(rows) + 1)):
            for col in range(max(0, min(cols)), min(sim.size, max(cols) + 1)):
                if self._point_in_polygon(row + 0.5, col + 0.5, vertices):
                    cells.append((row, col))
        return cells

    def _polygon_perimeter_cells(self, vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
        perimeter: list[tuple[int, int]] = []
        for index in range(len(vertices)):
            start = vertices[index]
            end = vertices[(index + 1) % len(vertices)]
            perimeter.extend(self._bresenham_segment(start, end))
        return self._dedupe_cells(perimeter)

    def _centroid_from_cells(self, cells: list[tuple[int, int]]) -> tuple[int, int]:
        if not cells:
            raise ValueError("target geometry produced no cells")
        row = round(sum(cell[0] for cell in cells) / len(cells))
        col = round(sum(cell[1] for cell in cells) / len(cells))
        self._validate_cell(row, col)
        return row, col

    def _sort_cells_by_anchor(
        self,
        cells: list[tuple[int, int]],
        anchor_row: int,
        anchor_col: int,
    ) -> list[tuple[int, int]]:
        return sorted(
            self._dedupe_cells(cells),
            key=lambda cell: (math.hypot(cell[0] - anchor_row, cell[1] - anchor_col), cell[0], cell[1]),
        )

    def _resolve_target(
        self,
        target: TargetSpec,
        mission_type: str,
    ) -> tuple[str, tuple[int, int], list[tuple[int, int]], str]:
        if target.target_kind == "point":
            if target.point is None:
                raise ValueError("point target requires point")
            row, col = target.point.row, target.point.col
            self._validate_cell(row, col)
            if mission_type == "point_protection":
                cells = self._cells_for_area(row, col, POINT_PROTECTION_RADIUS)
            elif mission_type == "line_construction":
                cells = self._cells_for_ring(row, col, POINT_PROTECTION_RADIUS)
            elif mission_type == "staging":
                cells = []
            else:
                cells = [(row, col)]
            return "point", (row, col), cells, f"point ({row}, {col})"

        if target.target_kind == "line":
            if len(target.waypoints) < 2:
                raise ValueError("line target requires at least two waypoints")
            waypoint_cells = [(point.row, point.col) for point in target.waypoints]
            for row, col in waypoint_cells:
                self._validate_cell(row, col)
            line_cells: list[tuple[int, int]] = []
            for index in range(len(waypoint_cells) - 1):
                line_cells.extend(self._bresenham_segment(waypoint_cells[index], waypoint_cells[index + 1]))
            line_cells = self._dedupe_cells(line_cells)
            anchor = waypoint_cells[0]
            cells = [] if mission_type == "staging" else line_cells
            return "line", anchor, cells, f"line with {len(waypoint_cells)} waypoints"

        if target.target_kind == "area":
            if target.center is None or target.radius is None:
                raise ValueError("area target requires center and radius")
            row, col = target.center.row, target.center.col
            self._validate_cell(row, col)
            area_cells = self._cells_for_area(row, col, target.radius)
            if mission_type == "line_construction":
                cells = self._cells_for_ring(row, col, max(1, target.radius))
            elif mission_type == "staging":
                cells = []
            else:
                cells = area_cells
            return "area", (row, col), cells, f"area center=({row}, {col}) radius={target.radius}"

        if target.target_kind == "polygon":
            if len(target.vertices) < 3:
                raise ValueError("polygon target requires at least three vertices")
            if len(target.vertices) > MAX_POLYGON_VERTICES:
                raise ValueError(f"polygon target supports at most {MAX_POLYGON_VERTICES} vertices")
            vertices = [(vertex.row, vertex.col) for vertex in target.vertices]
            for row, col in vertices:
                self._validate_cell(row, col)
            filled = self._cells_for_polygon_fill(vertices)
            perimeter = self._polygon_perimeter_cells(vertices)
            anchor = self._centroid_from_cells(filled or perimeter)
            if mission_type == "line_construction":
                cells = perimeter
            elif mission_type == "staging":
                cells = []
            else:
                cells = filled
            return "polygon", anchor, cells, f"polygon with {len(vertices)} vertices"

        if target.target_kind == "structure":
            if not target.structure_id:
                raise ValueError("structure target requires structure_id")
            structure = self._get_structure_by_id(target.structure_id)
            if structure is None:
                raise ValueError(f"unknown structure_id '{target.structure_id}'")
            row, col = int(structure["row"]), int(structure["col"])
            if mission_type == "point_protection":
                cells = self._cells_for_area(row, col, POINT_PROTECTION_RADIUS)
            elif mission_type == "line_construction":
                cells = self._cells_for_ring(row, col, POINT_PROTECTION_RADIUS + 1)
            elif mission_type == "staging":
                cells = []
            else:
                cells = [(row, col)]
            return "structure", (row, col), cells, f"structure {target.structure_id}"

        raise ValueError(f"unsupported target_kind '{target.target_kind}'")

    def _estimate_dispatch_steps(self, unit: FleetUnit, row: int, col: int) -> int:
        model = DISPATCH_MODEL[unit.resource_type]
        distance_cells = cell_distance(unit.standby_row, unit.standby_col, row, col)
        minutes = model["prep_minutes"] + distance_cells * model["travel_minutes_per_cell"]
        return minutes_to_steps(
            minutes,
            SIMULATION_STEP_MINUTES,
            minimum_steps=model["minimum_steps"],
        )

    def _estimate_return_steps(self, unit: FleetUnit, row: int, col: int) -> int:
        """Estimate how many steps this unit needs to return to standby after its mission.

        Ground and aerial assets (other than helicopters and smokejumpers) use
        a straight distance-based formula:
            return_overhead + dist(target -> standby) x travel_rate

        Helicopters and smokejumpers have dedicated methods because their return
        option is position-dependent: helicopters choose between scooping from
        the nearest water body or flying back to helibase; smokejumpers are
        extracted by helicopter from wherever they landed.
        """
        if unit.resource_type == "helicopters":
            return self._compute_helicopter_return_steps(unit, row, col)

        if unit.resource_type == "smokejumpers":
            return self._compute_smokejumper_return_steps(unit, row, col)

        model = DISPATCH_MODEL[unit.resource_type]
        distance_cells = cell_distance(unit.standby_row, unit.standby_col, row, col)
        minutes = model["return_base_minutes"] + distance_cells * model["travel_minutes_per_cell"]
        return minutes_to_steps(minutes, SIMULATION_STEP_MINUTES, minimum_steps=1)

    def _compute_helicopter_return_steps(
        self, unit: FleetUnit, target_row: int, target_col: int,
    ) -> int:
        """Helicopter return time — fully calculation-based, no hardcoded constants.

        Two options are compared and the faster is used:

        Scoop option (water body on map):
            fill_overhead + 2 × dist(target → nearest_water) × travel_rate
            fill_overhead = water_scoop_fill_minutes (~2.5 min, bucket submersion only)

        Base option (no water, or water is far):
            reload_overhead + dist(target → standby) × travel_rate

        The map's water-body geometry determines which option wins.  A
        helicopter assigned to a sector with a lake will automatically cycle
        faster than one over dry forest — the agent can discover this pattern
        without it being explicitly taught.

        Sources: SEI PowerFill fill time; NWCG/AFUE helicopter ops data.
        """
        model = DISPATCH_MODEL["helicopters"]
        fill_minutes = model["water_scoop_fill_minutes"]
        base_overhead = model["return_base_minutes"]
        travel_rate = model["travel_minutes_per_cell"]

        sim = self._require_sim()
        terrain = sim.terrain

        # Scan the entire map for the nearest water body.
        # No arbitrary range cap — the comparison below decides whether
        # scooping is actually faster than flying to base.
        nearest_water_dist: float | None = None
        for r in range(sim.size):
            for c in range(sim.size):
                if terrain.is_water[r, c]:
                    d = cell_distance(float(target_row), float(target_col), float(r), float(c))
                    if nearest_water_dist is None or d < nearest_water_dist:
                        nearest_water_dist = d

        # Option A — scoop: fly to water, fill bucket, fly back to drop zone
        if nearest_water_dist is not None:
            scoop_minutes = fill_minutes + nearest_water_dist * travel_rate * 2.0
        else:
            scoop_minutes = float("inf")  # no water on map

        # Option B — base reload: fly back to current standby, overhead, done
        dist_to_standby = cell_distance(
            unit.standby_row, unit.standby_col, float(target_row), float(target_col)
        )
        base_minutes = base_overhead + dist_to_standby * travel_rate

        minutes = min(scoop_minutes, base_minutes)
        return minutes_to_steps(minutes, SIMULATION_STEP_MINUTES, minimum_steps=1)

    def _compute_smokejumper_return_steps(
        self, unit: FleetUnit, target_row: int, target_col: int,
    ) -> int:
        """Smokejumper extraction time — fully calculation-based.

        Smokejumpers are extracted by a dedicated helicopter that departs
        from the jump base, flies to the target location, picks up the team
        (hover extraction or short landing), then returns to base.

        Formula:
            extraction_overhead + 2 × dist(jump_base → target) × extraction_transit_rate

        The unit's base_row/base_col is the jump aircraft's origin (set when
        the fleet is initialised from RESOURCE_BASES["smokejumpers"]).
        Extraction uses helicopter cruise speed, not jump-aircraft speed.

        This makes return time emerge from the deployment location: a jumper
        dropped near the base comes back quickly; one deployed in the far
        corner of the grid takes proportionally longer.
        """
        model = DISPATCH_MODEL["smokejumpers"]
        overhead = model["return_extraction_overhead_minutes"]
        transit_rate = model["return_extraction_transit_rate"]

        dist = cell_distance(
            unit.base_row, unit.base_col, float(target_row), float(target_col)
        )
        minutes = overhead + dist * transit_rate * 2.0
        return minutes_to_steps(minutes, SIMULATION_STEP_MINUTES, minimum_steps=1)

    def _estimate_dozer_work_steps(self, planned_cells: list[tuple[int, int]]) -> int:
        sim = self._require_sim()
        if not planned_cells:
            return 1

        terrain = sim.terrain
        minutes_required = 0.0
        for row, col in planned_cells:
            fuel_type = int(terrain.fuel_type[row, col])
            line_rate = DOZER_LINE_RATE_CHAINS_PER_HOUR.get(
                fuel_type,
                DOZER_LINE_RATE_CHAINS_PER_HOUR[FUEL_BRUSH],
            )
            local_relief = 0.0
            for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = row + drow, col + dcol
                if 0 <= nr < sim.size and 0 <= nc < sim.size:
                    local_relief = max(
                        local_relief,
                        abs(int(terrain.elevation[row, col]) - int(terrain.elevation[nr, nc])),
                    )
            slope_multiplier = 1.0 + 0.15 * local_relief
            minutes_required += (
                (DOZER_CHAIN_LENGTH_PER_CELL / max(1.0, line_rate))
                * 60.0
                * slope_multiplier
            )
        return minutes_to_steps(minutes_required, SIMULATION_STEP_MINUTES, minimum_steps=1)

    def _mission_plan(
        self,
        unit: FleetUnit,
        assignment: ResourceAssignment,
        anchor: tuple[int, int],
        target_cells: list[tuple[int, int]],
    ) -> tuple[int, list[tuple[int, int]], int, bool, bool]:
        if assignment.mission_type == "staging":
            return 0, [], 0, False, False

        ordered_cells = self._sort_cells_by_anchor(target_cells, anchor[0], anchor[1])
        mission = assignment.mission_type
        rtype = unit.resource_type

        # ── Ground crews (hand tools, drip torch for backfire) ──
        if rtype == "crews":
            cells_per_step = CREW_CELLS_PER_STEP
            if mission == "backfire":
                cells_per_step = 1  # deliberate, careful ignition
            work_steps = max(
                assignment.commitment_steps,
                math.ceil(max(1, len(ordered_cells)) / cells_per_step),
            )
            repeat_pattern = mission == "point_protection"
            return work_steps, ordered_cells, cells_per_step, repeat_pattern, True

        # ── Engines (pump-and-roll, foam spray) ──
        if rtype == "engines":
            cells_per_step = ENGINE_CELLS_PER_STEP
            if mission == "backfire":
                cells_per_step = 1
            work_steps = max(
                assignment.commitment_steps,
                math.ceil(max(1, len(ordered_cells)) / cells_per_step),
            )
            repeat_pattern = mission == "point_protection"
            return work_steps, ordered_cells, cells_per_step, repeat_pattern, True

        # ── Helicopters (bucket drop — single operating step) ──
        if rtype == "helicopters":
            cells_per_step = (
                1 if assignment.drop_configuration != "trail"
                else HELICOPTER_TRAIL_CELLS_PER_STEP
            )
            return 1, ordered_cells, cells_per_step, False, True

        # ── Air tankers (retardant line — single operating step) ──
        if rtype == "airtankers":
            cells_per_step = (
                1 if assignment.drop_configuration != "trail"
                else AIRTANKER_TRAIL_CELLS_PER_STEP
            )
            return 1, ordered_cells, cells_per_step, False, True

        # ── Smokejumpers (hand tools, same pace as crew) ──
        if rtype == "smokejumpers":
            cells_per_step = SMOKEJUMPER_CELLS_PER_STEP
            work_steps = max(
                assignment.commitment_steps,
                math.ceil(max(1, len(ordered_cells)) / cells_per_step),
            )
            repeat_pattern = mission == "point_protection"
            return work_steps, ordered_cells, cells_per_step, repeat_pattern, True

        # ── Dozers (terrain-dependent production rate) ──
        work_steps = max(self._estimate_dozer_work_steps(ordered_cells), assignment.commitment_steps)
        cells_per_step = max(1, math.ceil(max(1, len(ordered_cells)) / max(1, work_steps)))
        return work_steps, ordered_cells, cells_per_step, False, True

    # ── Fog-of-War visibility radii (cells) ──────────────────────────────────
    # Based on:
    #   PMC (2017): FLIR helicopter swath 5-8 km at 300-750m AGL → ~50-80 cells
    #   at 100 m/cell, but bounded to 6 for a 15×15 grid (practical coverage).
    #   Ground crew line-of-sight in smoke-limited terrain: ~400 m → 4 cells.
    #   Available-at-outpost units conduct local patrols: ~200 m → 2 cells.
    _FOW_RADII: dict[str, int] = {
        "helicopters":   6,
        "airtankers":    5,
        "crews":         4,
        "smokejumpers":  4,
        "engines":       3,
        "dozers":        3,
    }
    _FOW_OUTPOST_RADIUS: int = 2   # standby units observe immediate outpost area
    _FOW_STRUCTURE_RADIUS: int = 1  # structures always report their immediate neighbourhood

    def _compute_visible_cells(self) -> set[tuple[int, int]]:
        """Return the set of (row, col) cells observable by current resources.

        Each deployed unit reveals a disc of cells centred on its current
        position.  Units waiting at an outpost reveal a smaller disc around
        the outpost.  Each structure always reveals its immediate neighbours
        (local lookout / sensor presence).
        """
        if self._sim is None:
            return set()

        grid_size = self._sim.size
        visible: set[tuple[int, int]] = set()

        def _add_disc(row: float, col: float, radius: int) -> None:
            r0, c0 = int(round(row)), int(round(col))
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr * dr + dc * dc <= radius * radius:
                        nr, nc = r0 + dr, c0 + dc
                        if 0 <= nr < grid_size and 0 <= nc < grid_size:
                            visible.add((nr, nc))

        for unit in self._fleet_units:
            radius = self._FOW_RADII.get(unit.resource_type, 3)
            if unit.status == "available":
                # Standby at outpost — smaller local radius
                _add_disc(unit.standby_row, unit.standby_col, self._FOW_OUTPOST_RADIUS)
            else:
                # En route, operating, or returning — full sensor footprint
                _add_disc(unit.current_row, unit.current_col, radius)

        # Structures always reveal their immediate neighbourhood
        for s in self._sim.terrain.structures:
            _add_disc(s["row"], s["col"], self._FOW_STRUCTURE_RADIUS)

        return visible

    def _check_lces(
        self,
        unit_id: str,
        target_row: int,
        target_col: int,
    ) -> tuple[bool, str]:
        """Check LCES compliance for a ground-resource assignment.

        Violation condition: a burning cell is immediately adjacent to the
        target (≤ LCES_THREAT_RADIUS cells) AND no escape cell
        (FIREBREAK / WATER / BURNED / SUPPRESSED) exists within
        LCES_ESCAPE_RADIUS cells of the target.

        Grid edge is treated as open-terrain escape (firefighters can
        exit the map boundary on foot or by air).

        Returns (is_compliant, warning_text).

        Sources:
          - NWCG 10 Standard Firefighting Orders #6 and #8 (PMS 110)
          - NWCG LCES framework, Incident Response Pocket Guide PMS 461
          - Butler & Cohen (1998): safety zone diameter ≥ 4× flame height,
            USDA FS Intermountain Research Station
        """
        sim = self._require_sim()
        st = sim.state

        # Step 1 — Is there active fire within threat radius?
        fire_adjacent = False
        for dr in range(-LCES_THREAT_RADIUS, LCES_THREAT_RADIUS + 1):
            for dc in range(-LCES_THREAT_RADIUS, LCES_THREAT_RADIUS + 1):
                nr, nc = target_row + dr, target_col + dc
                if 0 <= nr < sim.size and 0 <= nc < sim.size:
                    if st.cell_state[nr, nc] == STATE_BURNING:
                        fire_adjacent = True
                        break
            if fire_adjacent:
                break

        if not fire_adjacent:
            return True, ""  # no active threat; LCES not triggered

        # Step 2 — Is there a reachable escape cell within escape radius?
        _ESCAPE = {STATE_FIREBREAK, STATE_WATER, STATE_BURNED, STATE_SUPPRESSED}
        for dr in range(-LCES_ESCAPE_RADIUS, LCES_ESCAPE_RADIUS + 1):
            for dc in range(-LCES_ESCAPE_RADIUS, LCES_ESCAPE_RADIUS + 1):
                nr, nc = target_row + dr, target_col + dc
                if 0 <= nr < sim.size and 0 <= nc < sim.size:
                    if st.cell_state[nr, nc] in _ESCAPE:
                        return True, ""

        # Step 3 — Grid edge is also a valid escape route
        if (
            target_row <= LCES_ESCAPE_RADIUS
            or target_row >= sim.size - 1 - LCES_ESCAPE_RADIUS
            or target_col <= LCES_ESCAPE_RADIUS
            or target_col >= sim.size - 1 - LCES_ESCAPE_RADIUS
        ):
            return True, ""

        # LCES violation — active fire with no escape route
        return False, (
            f"LCES: {unit_id} at ({target_row},{target_col}) has active fire "
            f"within {LCES_THREAT_RADIUS} cell(s) and no escape route within "
            f"{LCES_ESCAPE_RADIUS} cell(s). "
            f"NWCG 10-Orders #8: establish a firebreak, water, or burned "
            f"anchor before committing ground forces to the fire line."
        )

    def _schedule_assignment(
        self,
        assignment: ResourceAssignment,
    ) -> tuple[float, str | None, str | None]:
        unit = self._get_unit_by_id(assignment.unit_id)
        if unit is None:
            return INVALID_ACTION_PENALTY, None, f"unknown unit_id '{assignment.unit_id}'"
        if unit.status != "available":
            return (
                INVALID_ACTION_PENALTY,
                None,
                f"unit '{assignment.unit_id}' is currently {unit.status} and unavailable",
            )
        if assignment.mission_type not in RESOURCE_MISSION_COMPATIBILITY[unit.resource_type]:
            return (
                INVALID_ACTION_PENALTY,
                None,
                f"unit '{assignment.unit_id}' cannot perform mission '{assignment.mission_type}'",
            )
        if unit.resource_type not in AERIAL_DROP_RESOURCES and assignment.drop_configuration is not None:
            return INVALID_ACTION_PENALTY, None, "drop_configuration is only valid for helicopters and airtankers"

        try:
            target_kind, anchor, target_cells, target_summary = self._resolve_target(
                assignment.target,
                assignment.mission_type,
            )
            work_steps, mission_cells, cells_per_step, repeat_pattern, return_after = self._mission_plan(
                unit,
                assignment,
                anchor,
                target_cells,
            )
        except ValueError as exc:
            return INVALID_ACTION_PENALTY, None, str(exc)

        # Backfire anchor-point validation (NWCG PMS 410-1)
        if assignment.mission_type == "backfire" and mission_cells:
            anchor_valid, anchor_msg = self._validate_backfire_anchor(mission_cells)
            if not anchor_valid:
                return INVALID_ACTION_PENALTY, None, anchor_msg

        # LCES safety check for ground resources on hazardous missions.
        # Not a hard block — NWCG doctrine allows firefighter judgment — but
        # the penalty (-0.03) creates training pressure to pre-establish escape
        # routes before committing crews to the fire line.
        lces_penalty = 0.0
        lces_warning = ""
        if (
            unit.resource_type in LCES_GROUND_RESOURCES
            and assignment.mission_type in LCES_HAZARDOUS_MISSIONS
        ):
            lces_ok, lces_msg = self._check_lces(unit.unit_id, anchor[0], anchor[1])
            if not lces_ok:
                lces_penalty = LCES_VIOLATION_PENALTY
                lces_warning = lces_msg

        dispatch_steps = self._estimate_dispatch_steps(unit, anchor[0], anchor[1])
        return_steps = 0 if not return_after else self._estimate_return_steps(unit, anchor[0], anchor[1])

        unit.status = "en_route"
        unit.mission_type = assignment.mission_type
        unit.target_kind = target_kind  # type: ignore[assignment]
        unit.target_row = anchor[0]
        unit.target_col = anchor[1]
        unit.target_summary = target_summary
        unit.eta_steps = dispatch_steps
        unit.work_steps_remaining = work_steps
        unit.return_steps_remaining = return_steps
        unit.commitment_steps = assignment.commitment_steps
        unit.drop_configuration = assignment.drop_configuration
        unit.repeat_pattern = repeat_pattern
        unit.return_after_mission = return_after
        unit.mission_cells = list(mission_cells)
        unit.pending_cells = list(mission_cells)
        unit.cells_per_step = cells_per_step

        summary = (
            f"{unit.unit_id} assigned to {assignment.mission_type} on {target_summary} "
            f"(ETA {dispatch_steps} step(s))"
        )
        if lces_warning:
            summary = f"{summary} | {lces_warning}"

        return (
            MISSION_COST[assignment.mission_type]
            + RESOURCE_DISPATCH_COST[unit.resource_type]
            + lces_penalty,
            summary,
            None,
        )

    def _schedule_assignments(self, action: WildfireAction) -> float:
        self._last_action_error = None
        self._last_action_summary = ""

        if not action.assignments:
            self._last_action_summary = "advanced simulation and held current fleet assignments"
            return 0.0

        reward_delta = 0.0
        summaries: list[str] = []
        errors: list[str] = []
        seen_units: set[str] = set()

        for assignment in action.assignments:
            if assignment.unit_id in seen_units:
                reward_delta += INVALID_ACTION_PENALTY
                errors.append(f"unit '{assignment.unit_id}' was assigned more than once in the same step")
                continue
            seen_units.add(assignment.unit_id)

            delta, summary, error = self._schedule_assignment(assignment)
            reward_delta += delta
            if summary:
                summaries.append(summary)
            if error:
                errors.append(error)

        self._last_action_summary = " | ".join(summaries) if summaries else "all assignments rejected"
        if errors:
            self._last_action_error = " | ".join(errors)

        return reward_delta

    def _take_cells_for_step(self, unit: FleetUnit) -> list[tuple[int, int]]:
        if unit.mission_type == "staging":
            return []
        if not unit.pending_cells and unit.repeat_pattern and unit.mission_cells:
            unit.pending_cells = list(unit.mission_cells)

        cells: list[tuple[int, int]] = []
        while unit.pending_cells and len(cells) < max(1, unit.cells_per_step):
            cells.append(unit.pending_cells.pop(0))
        return cells

    def _validate_backfire_anchor(
        self, cells: list[tuple[int, int]],
    ) -> tuple[bool, str]:
        """Validate that a backfire line has at least one anchor point.

        Per NWCG Fireline Handbook PMS 410-1 Section 8, all backfire
        operations must originate from a secure anchor such as a road,
        river, rock, or previously burned area to prevent flanking.

        Returns (is_valid, message).
        """
        if not cells:
            return False, "backfire has no target cells"

        sim = self._require_sim()
        st = sim.state
        if st is None:
            return False, "simulation not initialized"

        endpoints = [cells[0], cells[-1]]
        for er, ec in endpoints:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = er + dr, ec + dc
                    if 0 <= nr < sim.size and 0 <= nc < sim.size:
                        if st.cell_state[nr, nc] in BACKFIRE_ANCHOR_STATES:
                            return True, "backfire anchored"
                    else:
                        # Grid edge counts as an anchor (natural boundary)
                        return True, "backfire anchored at grid edge"

        # Also accept FUEL_NONE cells and ROCK terrain as anchors
        terrain = sim.terrain
        for er, ec in endpoints:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = er + dr, ec + dc
                    if 0 <= nr < sim.size and 0 <= nc < sim.size:
                        if terrain.fuel_type[nr, nc] == FUEL_NONE:
                            return True, "backfire anchored at rock/bare ground"

        return False, (
            "backfire rejected: no anchor point found adjacent to line endpoints. "
            "Backfire lines must start from a FIREBREAK, WATER, BURNED, SUPPRESSED, "
            "or bare-ground cell to prevent flanking."
        )

    def _execute_unit_work(self, unit: FleetUnit) -> tuple[bool, str]:
        """Execute one work-step for an operating unit.

        Dispatches to the appropriate fire_simulation method based on the
        unit's resource_type and mission_type combination.
        """
        sim = self._require_sim()
        if unit.mission_type is None:
            return False, "unit has no active mission"

        cells = self._take_cells_for_step(unit)
        if unit.mission_type != "staging" and not cells:
            return False, "mission had no actionable target cells left"

        mission = unit.mission_type
        rtype = unit.resource_type

        # ── STAGING (all resource types) ──
        if mission == "staging":
            return True, f"{unit.unit_id} holding at staging position"

        # ── DIRECT ATTACK (crews, engines, smokejumpers) ──
        if mission == "direct_attack":
            if rtype == "engines":
                successful = 0
                for row, col in cells:
                    success, _ = sim.apply_engine(row, col)
                    if success:
                        successful += 1
                return successful > 0, f"engine treated {successful}/{len(cells)} target cells"
            else:
                # crews and smokejumpers both use hand-tool suppression
                successful = 0
                for row, col in cells:
                    success, _ = sim.apply_crew(row, col)
                    if success:
                        successful += 1
                label = "smokejumper" if rtype == "smokejumpers" else "crew"
                return successful > 0, f"{label} treated {successful}/{len(cells)} target cells"

        # ── LINE CONSTRUCTION (dozers, crews, smokejumpers) ──
        if mission == "line_construction":
            built = 0
            blocked = 0
            for row, col in cells:
                if rtype == "dozers":
                    success, _ = sim.apply_dozer_segment(row, col)
                else:
                    # Hand crews and smokejumpers build hand lines at slower rate
                    # Hand line is modeled as a firebreak segment
                    success, _ = sim.apply_dozer_segment(row, col)
                if success:
                    built += 1
                else:
                    blocked += 1
            label = {"dozers": "dozer", "crews": "crew", "smokejumpers": "smokejumper"}.get(rtype, rtype)
            if built == 0:
                return False, f"{label} hit {blocked} blocked cells"
            return True, f"{label} built {built} segment(s) ({blocked} blocked)"

        # ── WET LINE (engines only) ──
        if mission == "wet_line":
            built = 0
            blocked = 0
            for row, col in cells:
                success, _ = sim.apply_wetline(row, col)
                if success:
                    built += 1
                else:
                    blocked += 1
            if built == 0:
                return False, f"engine wet-line hit {blocked} blocked cells"
            return True, f"engine sprayed {built} wet-line segment(s) ({blocked} blocked)"

        # ── WATER DROP (helicopters) ──
        if mission == "water_drop":
            successful = 0
            for row, col in cells:
                success, _ = sim.apply_helicopter(row, col)
                if success:
                    successful += 1
            return successful > 0, f"helicopter drop affected {successful}/{len(cells)} target cells"

        # ── RETARDANT DROP (air tankers) ──
        if mission == "retardant_drop":
            successful = 0
            for row, col in cells:
                success, _ = sim.apply_airtanker(row, col)
                if success:
                    successful += 1
            return successful > 0, f"retardant drop affected {successful}/{len(cells)} target cells"

        # ── POINT PROTECTION (all except air tankers) ──
        if mission == "point_protection":
            if rtype == "engines":
                successful = 0
                for row, col in cells:
                    success, _ = sim.apply_engine(row, col)
                    if success:
                        successful += 1
                return successful > 0, f"engine protecting {successful}/{len(cells)} cells"
            elif rtype == "helicopters":
                successful = 0
                for row, col in cells:
                    success, _ = sim.apply_helicopter(row, col)
                    if success:
                        successful += 1
                return successful > 0, f"helicopter protecting {successful}/{len(cells)} cells"
            else:
                # crews, dozers, smokejumpers — ground-based protection
                successful = 0
                for row, col in cells:
                    success, _ = sim.apply_crew(row, col)
                    if success:
                        successful += 1
                label = {"dozers": "dozer", "smokejumpers": "smokejumper"}.get(rtype, "crew")
                return successful > 0, f"{label} protecting {successful}/{len(cells)} cells"

        # ── BACKFIRE (crews, engines) ──
        if mission == "backfire":
            # Anchor validation happens at schedule time, but double-check
            ignited = 0
            failed = 0
            for row, col in cells:
                success, _ = sim.apply_backfire(row, col)
                if success:
                    ignited += 1
                else:
                    failed += 1
            if ignited == 0:
                return False, f"backfire failed to ignite any cells ({failed} blocked)"
            return True, f"backfire ignited {ignited} cells ({failed} blocked)"

        return False, f"unhandled mission type '{mission}' for resource '{rtype}'"

    def _progress_fleet(self) -> tuple[list[str], float]:
        events: list[str] = []
        reward_delta = 0.0

        for unit in self._fleet_units:
            if unit.status in {"en_route", "returning"} and unit.eta_steps > 0:
                unit.eta_steps -= 1

        for unit in self._fleet_units:
            if unit.status != "en_route" or unit.eta_steps > 0:
                continue

            if unit.target_row is None or unit.target_col is None:
                reward_delta += LOW_IMPACT_ACTION_PENALTY
                unit.clear_mission()
                events.append(f"{unit.unit_id} lost its mission target and returned to standby")
                continue

            unit.assign_position(unit.target_row, unit.target_col)
            if unit.mission_type == "staging":
                unit.set_standby_position(unit.target_row, unit.target_col)
                unit.clear_mission()
                events.append(f"{unit.unit_id} staged at ({unit.target_row}, {unit.target_col})")
            else:
                unit.status = "operating"
                events.append(f"{unit.unit_id} arrived to start {unit.mission_type}")

        for unit in self._fleet_units:
            if unit.status != "operating":
                continue

            success, summary = self._execute_unit_work(unit)
            events.append(f"{unit.unit_id}: {summary}")
            if not success:
                reward_delta += LOW_IMPACT_ACTION_PENALTY

            unit.work_steps_remaining = max(0, unit.work_steps_remaining - 1)
            if unit.work_steps_remaining <= 0:
                unit.missions_completed += 1
                if unit.return_after_mission:
                    unit.status = "returning"
                    unit.eta_steps = max(1, unit.return_steps_remaining)
                else:
                    unit.clear_mission()

        for unit in self._fleet_units:
            if unit.status == "returning" and unit.eta_steps <= 0:
                unit.clear_mission()
                events.append(f"{unit.unit_id} is back in service")

        return events, reward_delta

    def step(self, action: WildfireAction, timeout_s: float | None = None, **kwargs) -> WildfireObservation:
        """
        Queue one or more resource assignments, advance logistics and fire
        dynamics by one tick, and return the resulting observation.
        """
        self._require_sim()

        reward_delta = self._schedule_assignments(action)
        dispatch_summary = self._last_action_summary
        fleet_events, fleet_reward_delta = self._progress_fleet()

        if fleet_events:
            event_summary = " | ".join(fleet_events)
            if dispatch_summary:
                self._last_action_summary = f"{dispatch_summary} | {event_summary}"
            else:
                self._last_action_summary = event_summary

        sim = self._require_sim()
        sim_state = sim.tick()
        env_rewards = sim.compute_environmental_rewards()
        total_step_reward = reward_delta + fleet_reward_delta + sum(env_rewards.values())

        self._total_reward += total_step_reward
        self._state.step_count = sim_state.step

        observation = self._build_observation(
            reward=total_step_reward,
            done=sim_state.done,
        )
        observation.metadata["reward_components"] = env_rewards
        observation.metadata["action_cost"] = reward_delta
        observation.metadata["fleet_effect_penalty"] = fleet_reward_delta
        return observation

    @property
    def state(self) -> State:
        """Return the current environment state with useful extra fields."""
        extras = {
            "task_id": self._task_id,
            "resources_remaining": self._available_resource_counts(),
            "resource_totals": self._resource_totals.copy(),
            "fleet_status": [item.model_dump() for item in self._fleet_status()],
            "fleet_units": [item.model_dump() for item in self._fleet_units_observation()],
            "active_missions": [item.model_dump() for item in self._active_missions()],
            "last_action_summary": self._last_action_summary,
            "last_action_error": self._last_action_error,
            "total_reward": round(self._total_reward, 4),
        }
        if self._sim is not None and self._sim.state is not None:
            extras.update(
                {
                    "burning_cells": self._sim.state.total_burning,
                    "burned_cells": self._sim.state.total_burned,
                    "structures_lost": self._sim.state.structures_lost,
                    "structures_saved": self._sim.state.structures_saved,
                    "elapsed_minutes": round(
                        self._sim.state.step * SIMULATION_STEP_MINUTES,
                        1,
                    ),
                    "time_of_day": round(self._sim.state.time_of_day, 2),
                    "done": self._sim.state.done,
                }
            )
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            **extras,
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the wildfire environment."""

from typing import Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


class GridPoint(BaseModel):
    """One grid coordinate."""

    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)


class TargetSpec(BaseModel):
    """Flexible geometric target for resource assignments."""

    target_kind: Literal["point", "line", "area", "polygon", "structure"]
    point: GridPoint | None = None
    waypoints: list[GridPoint] = Field(default_factory=list)
    center: GridPoint | None = None
    radius: int | None = Field(default=None, ge=0)
    vertices: list[GridPoint] = Field(default_factory=list)
    structure_id: str | None = None


class ResourceAssignment(BaseModel):
    """One resource mission assigned during the current decision interval."""

    unit_id: str
    mission_type: Literal[
        "direct_attack",
        "line_construction",
        "wet_line",
        "water_drop",
        "retardant_drop",
        "point_protection",
        "backfire",
        "staging",
    ]
    target: TargetSpec
    commitment_steps: int = Field(default=1, ge=1, le=12)
    drop_configuration: Literal["salvo", "trail"] | None = None


class FireCellObservation(BaseModel):
    """Observation for a currently burning cell."""

    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    intensity: float = Field(..., ge=0.0)
    timer: int = Field(..., ge=0)


class StructureObservation(BaseModel):
    """Observation for a structure or asset cell."""

    structure_id: str
    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    priority: int = Field(..., ge=1)
    status: str = Field(..., description="Current structure status label")


class HeatWarningObservation(BaseModel):
    """Observation for cells approaching ignition."""

    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    heat: float = Field(..., ge=0.0)
    threshold: float = Field(..., ge=0.0)


class OutpostObservation(BaseModel):
    """Observation for a resource outpost / staging area."""

    outpost_id: str
    row: float
    col: float
    is_airbase: bool = False
    unit_ids: list[str] = Field(default_factory=list)


class FleetStatusObservation(BaseModel):
    """Aggregate status for one resource type."""

    resource_type: Literal[
        "crews", "engines", "helicopters", "airtankers", "dozers", "smokejumpers",
    ]
    total_units: int = Field(..., ge=0)
    available_units: int = Field(..., ge=0)
    en_route_units: int = Field(..., ge=0)
    operating_units: int = Field(..., ge=0)
    returning_units: int = Field(..., ge=0)


class FleetMissionObservation(BaseModel):
    """One active or pending fleet mission."""

    unit_id: str
    resource_type: Literal[
        "crews", "engines", "helicopters", "airtankers", "dozers", "smokejumpers",
    ]
    status: Literal["en_route", "operating", "returning"]
    mission_type: Literal[
        "direct_attack",
        "line_construction",
        "wet_line",
        "water_drop",
        "retardant_drop",
        "point_protection",
        "backfire",
        "staging",
    ] | None = None
    target_kind: Literal["point", "line", "area", "polygon", "structure"] | None = None
    target_summary: str = ""
    eta_steps: int = Field(..., ge=0)
    eta_minutes: float = Field(..., ge=0.0)
    summary: str = Field(..., description="Readable status of the mission")


class FleetUnitObservation(BaseModel):
    """Observation for an individual fleet unit."""

    unit_id: str
    resource_type: Literal[
        "crews", "engines", "helicopters", "airtankers", "dozers", "smokejumpers",
    ]
    status: Literal["available", "en_route", "operating", "returning"]
    current_row: float
    current_col: float
    standby_row: float
    standby_col: float
    base_row: float
    base_col: float
    mission_type: Literal[
        "direct_attack",
        "line_construction",
        "wet_line",
        "water_drop",
        "retardant_drop",
        "point_protection",
        "backfire",
        "staging",
    ] | None = None
    target_kind: Literal["point", "line", "area", "polygon", "structure"] | None = None
    target_summary: str = ""
    available_in_steps: int = Field(..., ge=0)
    available_in_minutes: float = Field(..., ge=0.0)
    outpost_id: str = Field(default="", description="Home outpost for this unit")
    missions_completed: int = Field(default=0, ge=0, description="Cumulative missions this unit has completed this episode")


class WildfireAction(Action):
    """Structured action for the wildfire environment."""

    assignments: list[ResourceAssignment] = Field(
        default_factory=list,
        description="Assignments to issue during this decision interval",
    )


class WildfireObservation(Observation):
    """Observation from the wildfire incident-command environment."""

    task_id: str = Field(..., description="Current task identifier")
    goal: str = Field(..., description="Task objective for the current episode")
    grid: str = Field(..., description="Human-readable grid rendering")
    step: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)
    step_minutes: float = Field(..., gt=0.0)
    elapsed_minutes: float = Field(..., ge=0.0)
    time_of_day: float = Field(..., ge=0.0, le=1.0)

    wind_speed: float = Field(..., ge=0.0)
    wind_direction: float = Field(..., ge=0.0, le=360.0)
    temperature: float = Field(...)
    humidity: float = Field(..., ge=0.0, le=1.0)
    atmospheric_dryness_index: float = Field(...)
    airflow_potential_peak: float = Field(..., ge=0.0)

    burning_cells: int = Field(..., ge=0)
    burned_cells: int = Field(..., ge=0)
    structures_remaining: int = Field(..., ge=0)
    structures_lost: int = Field(..., ge=0)

    resources_remaining: dict[str, int] = Field(
        default_factory=dict,
        description="Resource units currently available to dispatch now",
    )
    resource_totals: dict[str, int] = Field(
        default_factory=dict,
        description="Total fleet units assigned to the episode by resource type",
    )
    fleet_status: list[FleetStatusObservation] = Field(default_factory=list)
    fleet_units: list[FleetUnitObservation] = Field(default_factory=list)
    active_missions: list[FleetMissionObservation] = Field(default_factory=list)
    last_action_summary: str = Field(
        default="",
        description="Short summary of how the last action affected the world",
    )
    last_action_error: str | None = Field(
        default=None,
        description="Validation or execution error for the last action, if any",
    )

    outposts: list[OutpostObservation] = Field(default_factory=list)
    fire_details: list[FireCellObservation] = Field(default_factory=list)
    structures: list[StructureObservation] = Field(default_factory=list)
    heat_warnings: list[HeatWarningObservation] = Field(default_factory=list)
    elevation: list[list[int]] = Field(default_factory=list)
    fuel_types: list[list[int]] = Field(default_factory=list)
    action_guide: str = Field(
        default="",
        description=(
            "Natural-language per-step guide listing available units and their valid missions, "
            "committed unit ETAs, and a brief tactical situation summary. "
            "Designed to help LLM agents produce valid actions without memorising the compatibility matrix."
        ),
    )
    weather_forecast: list[dict] = Field(
        default_factory=list,
        description=(
            "2-step NWS Spot Forecast style look-ahead (NWCG PMS 425). "
            "Each entry: {step, minutes_ahead, temperature, humidity, "
            "wind_speed_expected, wind_direction_current}. "
            "Temperature and humidity are deterministic (diurnal cycle). "
            "wind_speed_expected is the diurnal trend without stochastic noise."
        ),
    )
    visible_cell_count: int = Field(
        default=0,
        ge=0,
        description="Number of grid cells currently observable by deployed resources (fog-of-war).",
    )
    fog_of_war_active: bool = Field(
        default=True,
        description=(
            "When True, fire_details and heat_warnings only cover cells within "
            "sensor range of deployed units. Cells outside sensor range are "
            "shown as '?' in the grid string."
        ),
    )

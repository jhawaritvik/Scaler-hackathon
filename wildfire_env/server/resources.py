"""Fleet resource models for wildfire dispatch and recovery timing.

All dispatch-model numbers are derived from or calibrated against published
operational data:

- **Hand crew travel**: NWCG-sourced hiking rates of ~55 m/min on gentle
  terrain (with 45-lb packs), mapping to ~2.5 min/cell at 100 m cell pitch.
  (NWCG Firefighter Travel Rates, Butler et al. 2020)
- **Engine travel**: Type 3 wildland engines travel off-road at ~10-15 mph
  (NWCG PMS 200 resource typing), mapping to ~1.5 min/cell.
- **Helicopter transit**: 100-130 kt cruise, mapping to ~0.6 min/cell.
  Water-scoop refill ~5 min turnaround near water body (SEI PowerFill data).
  Base-reload ~20 min (bucketless reload at helibase).
- **Air tanker transit**: LAT 200+ kt cruise, mapping to ~0.4 min/cell.
  Reload 12-20 min at retardant base (10 Tanker Air Carrier ops data;
  USDA AFUE study reload cycle analysis).
- **Dozer mobilization**: 15 min prep per NWCG incident mgmt guidelines,
  on-trailer road speed ~25 mph equating to ~2.0 min/cell.
- **Smokejumper deployment**: 15-30 min from dispatch to jump per USFS
  smokejumper program data (NIFC Bureau of Land Management); self-sufficient
  for 48-72 hours; return requires helicopter extraction (~40 min average).

Fireline production rates source:
    NWCG 2021 Fireline Production Rates
    (San Dimas Technology & Development Center, Tech Tip 1151-1805P)
    https://www.frames.gov/documents/behaveplus/publications/
    NWCG_2021_FireLineProductionRates.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ResourceType = Literal[
    "crews",           # Type 2 hand crews (20-person)
    "engines",         # Type 3-6 wildland fire engines
    "helicopters",     # Type 2-3 helicopters with Bambi bucket
    "airtankers",      # LAT fixed-wing air tankers
    "dozers",          # D6-class bulldozers
    "smokejumpers",    # Smokejumper sticks (2-8 person)
]

MissionType = Literal[
    "direct_attack",
    "line_construction",
    "wet_line",
    "water_drop",
    "retardant_drop",
    "point_protection",
    "backfire",
    "staging",
]

TargetKind = Literal["point", "line", "area", "polygon", "structure"]
UnitStatus = Literal["available", "en_route", "operating", "returning"]


# ---------------------------------------------------------------------------
# Dispatch model — realistic timing parameters
# ---------------------------------------------------------------------------

DISPATCH_MODEL = {
    # ── Hand Crews (Type 2, 20-person) ──
    # Pack Test baseline: 3 mi / 45 min w/ 45 lb pack = 4 mph on flat.
    # Cross-country with gear on moderate slope: ~55 m/min (NWCG Butler 2020).
    # At 100 m cell pitch → ~1.8 min/cell. We use 2.5 to account for
    # tool-carrying and fireline terrain.
    "crews": {
        "prep_minutes": 10.0,           # gear-up, briefing
        "travel_minutes_per_cell": 2.5,  # foot movement with pack/tools
        "return_base_minutes": 15.0,     # pack-up and hike out
        "minimum_steps": 1,
    },

    # ── Wildland Engines (Type 3-6) ──
    # Off-road speed ~10-15 mph on fire roads/trails.
    # At 100 m cell pitch → ~1.2-1.5 min/cell.
    # Prep is faster (already on wheels, crew stays on engine).
    "engines": {
        "prep_minutes": 5.0,            # start engine, verify water/foam
        "travel_minutes_per_cell": 1.5,  # road/trail vehicle movement
        "return_base_minutes": 10.0,     # drive back to staging
        "minimum_steps": 1,
    },

    # ── Helicopters (Type 2-3, Bambi bucket) ──
    # Cruise 100-130 kt → ~0.5-0.8 min/cell.
    # Return time is fully calculated — not hardcoded:
    #   scoop option : water_scoop_fill_minutes + 2 × dist_to_nearest_water × travel_rate
    #   base option  : return_base_minutes       + dist_to_standby           × travel_rate
    # The cheaper option is chosen automatically, so map water-body placement
    # directly determines how fast helicopters cycle.  No range cap — the
    # comparison itself decides whether scooping is worth it.
    #
    # water_scoop_fill_minutes is the physical fill overhead only (bucket
    # submerge + fill + ascent, SEI PowerFill: ~60-90 s actual; we use 2.5 min
    # to include the slow-down / positioning pass).  Transit to/from the water
    # body is already in the distance term.
    "helicopters": {
        "prep_minutes": 5.0,                 # rotor spin-up, briefing
        "travel_minutes_per_cell": 0.6,      # ~120 kt cruise
        "return_base_minutes": 20.0,         # helibase reload overhead (no transit)
        "water_scoop_fill_minutes": 2.5,     # bucket-fill overhead only
        "minimum_steps": 1,
    },

    # ── Air Tankers (LAT, 2000-4000 gal) ──
    # Cruise 200+ kt → ~0.3-0.4 min/cell.
    # Reload 12-20 min at retardant base (10 Tanker ops data).
    # Total turnaround (land, reload, taxi, takeoff) 30-45 min for VLAT,
    # 15-25 min for standard LAT.
    "airtankers": {
        "prep_minutes": 8.0,            # taxi, load, pre-flight
        "travel_minutes_per_cell": 0.4,  # ~250 kt cruise
        "return_base_minutes": 30.0,     # base turnaround (reload retardant)
        "minimum_steps": 1,
    },

    # ── Dozers (D6-class, Type II) ──
    # On-trailer road speed ~25 mph → ~2 min/cell.
    # 15 min prep: unload from trailer, warm up, operator brief.
    "dozers": {
        "prep_minutes": 15.0,           # unload, warm-up
        "travel_minutes_per_cell": 2.0,  # trailer transport + unload
        "return_base_minutes": 20.0,     # load on trailer and return
        "minimum_steps": 1,
    },

    # ── Smokejumpers (2-8 person sticks) ──
    # Dispatch-to-jump: 15-30 min (USFS smokejumper ops).
    # Jump aircraft cruise ~150 kt → ~0.3 min/cell transit.
    # Return: helicopter extraction — extraction aircraft flies from jump base
    # to target, lands (or hover-extracts), then returns to base.
    # Extraction transit rate uses helicopter cruise (~120 kt = 0.6 min/cell)
    # since a dedicated Type 1/2 helicopter performs the extraction.
    # Return time is fully calculated:
    #   return_extraction_overhead_minutes + 2 × dist(base→target) × return_extraction_transit_rate
    # This makes extraction time emerge from where on the map jumpers were
    # deployed — a target near the jump base is extracted quickly; a far
    # corner target takes much longer.
    "smokejumpers": {
        "prep_minutes": 2.0,                         # already suited, minimal in-air prep
        "travel_minutes_per_cell": 0.3,              # jump aircraft transit (~150 kt)
        "return_extraction_overhead_minutes": 15.0,  # land/hover, load, debrief, lift-off
        "return_extraction_transit_rate": 0.6,       # extraction helicopter (~120 kt)
        "minimum_steps": 1,
    },
}


# ---------------------------------------------------------------------------
# Legacy base positions — used as fallback when no outposts are defined.
# New scenarios use outpost-based positioning (see build_initial_fleet).
# ---------------------------------------------------------------------------

_LEGACY_BASES = {
    "crews": lambda size: [
        (0.0, 0.0), (0.0, float(size - 1)),
        (float(size - 1), 0.0), (float(size - 1), float(size - 1)),
    ],
    "engines": lambda size: [
        (0.0, float(size // 2)), (float(size // 2), 0.0),
        (float(size // 2), float(size - 1)), (float(size - 1), float(size // 2)),
    ],
    "helicopters": lambda size: [
        (-4.0, float(size // 2)), (float(size + 3), float(size // 2)),
    ],
    "airtankers": lambda size: [(-8.0, float(size // 2))],
    "dozers": lambda size: [
        (2.0, 2.0), (2.0, float(size - 3)),
        (float(size - 3), 2.0), (float(size - 3), float(size - 3)),
    ],
    "smokejumpers": lambda size: [(-8.0, float(size // 2))],
}


# ---------------------------------------------------------------------------
# Fleet unit data class
# ---------------------------------------------------------------------------

@dataclass
class FleetUnit:
    """A reusable incident resource with mission timing state."""

    unit_id: str
    resource_type: ResourceType
    base_row: float
    base_col: float
    standby_row: float
    standby_col: float
    position_row: float
    position_col: float
    outpost_id: str = ""
    status: UnitStatus = "available"
    mission_type: MissionType | None = None
    target_kind: TargetKind | None = None
    target_row: int | None = None
    target_col: int | None = None
    target_summary: str = ""
    eta_steps: int = 0
    work_steps_remaining: int = 0
    return_steps_remaining: int = 0
    commitment_steps: int = 1
    drop_configuration: str | None = None
    repeat_pattern: bool = False
    return_after_mission: bool = True
    mission_cells: list[tuple[int, int]] = field(default_factory=list)
    pending_cells: list[tuple[int, int]] = field(default_factory=list)
    cells_per_step: int = 1
    missions_completed: int = 0  # cumulative count — NOT reset by clear_mission()

    def assign_position(self, row: float, col: float) -> None:
        """Update the unit's notional location."""
        self.position_row = float(row)
        self.position_col = float(col)

    def set_standby_position(self, row: float, col: float) -> None:
        """Update where the unit dispatches from when available."""
        self.standby_row = float(row)
        self.standby_col = float(col)
        self.assign_position(row, col)

    def clear_mission(self) -> None:
        """Return the unit to an available standby state.

        missions_completed is intentionally NOT reset — it's a cumulative
        episode-level counter that persists across individual assignments.
        """
        self.status = "available"
        self.mission_type = None
        self.target_kind = None
        self.target_row = None
        self.target_col = None
        self.target_summary = ""
        self.eta_steps = 0
        self.work_steps_remaining = 0
        self.return_steps_remaining = 0
        self.commitment_steps = 1
        self.drop_configuration = None
        self.repeat_pattern = False
        self.return_after_mission = True
        self.mission_cells.clear()
        self.pending_cells.clear()
        self.cells_per_step = 1
        self.assign_position(self.standby_row, self.standby_col)


# ---------------------------------------------------------------------------
# Unit ID naming convention for each resource type
# ---------------------------------------------------------------------------

_RESOURCE_UNIT_PREFIX = {
    "crews": "crew",
    "engines": "engine",
    "helicopters": "helicopter",
    "airtankers": "airtanker",
    "dozers": "dozer",
    "smokejumpers": "smokejumper",
}


def build_initial_fleet(
    resource_counts: dict[str, int],
    grid_size: int,
    outposts: list[dict] | None = None,
) -> list[FleetUnit]:
    """Create deterministic initial fleet units from scenario counts.

    If *outposts* is provided, each unit is assigned to its outpost's
    position.  Otherwise falls back to legacy fixed base positions.
    """
    fleet: list[FleetUnit] = []

    if outposts:
        return _build_fleet_from_outposts(outposts)

    # Legacy fallback — fixed base positions per resource type
    for resource_type in (
        "crews", "engines", "helicopters", "airtankers", "dozers", "smokejumpers",
    ):
        count = int(resource_counts.get(resource_type, 0))
        if count == 0:
            continue
        bases = _LEGACY_BASES[resource_type](grid_size)
        prefix = _RESOURCE_UNIT_PREFIX[resource_type]

        for idx in range(count):
            base_row, base_col = bases[idx % len(bases)]
            fleet.append(
                FleetUnit(
                    unit_id=f"{prefix}_{idx + 1}",
                    resource_type=resource_type,
                    base_row=base_row,
                    base_col=base_col,
                    standby_row=base_row,
                    standby_col=base_col,
                    position_row=base_row,
                    position_col=base_col,
                )
            )

    return fleet


def _build_fleet_from_outposts(outposts: list[dict]) -> list[FleetUnit]:
    """Build fleet units assigned to their respective outposts."""
    fleet: list[FleetUnit] = []
    # Track unit counts per resource type for unique IDs
    type_counter: dict[str, int] = {}

    for outpost in outposts:
        row = float(outpost["row"])
        col = float(outpost["col"])
        outpost_id = outpost["outpost_id"]

        for resource_type, count in outpost.get("resources", {}).items():
            prefix = _RESOURCE_UNIT_PREFIX.get(resource_type, resource_type)
            for _ in range(int(count)):
                type_counter[resource_type] = type_counter.get(resource_type, 0) + 1
                idx = type_counter[resource_type]
                fleet.append(
                    FleetUnit(
                        unit_id=f"{prefix}_{idx}",
                        resource_type=resource_type,
                        base_row=row,
                        base_col=col,
                        standby_row=row,
                        standby_col=col,
                        position_row=row,
                        position_col=col,
                        outpost_id=outpost_id,
                    )
                )

    return fleet


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def cell_distance(row_a: float, col_a: float, row_b: float, col_b: float) -> float:
    """Euclidean distance in grid cells."""
    return math.hypot(float(row_a) - float(row_b), float(col_a) - float(col_b))


def minutes_to_steps(minutes: float, step_minutes: float, minimum_steps: int = 1) -> int:
    """Convert elapsed minutes into one or more simulation steps."""
    if minutes <= 0.0:
        return max(0, minimum_steps)
    return max(minimum_steps, int(math.ceil(minutes / step_minutes)))

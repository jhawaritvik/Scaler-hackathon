"""
Terrain Generator for Wildfire Simulation.

Generates a 15x15 grid with:
  - elevation (0-9): height map using diamond-square-like noise
  - fuel_type (0-3): 0=none/rock, 1=grass, 2=brush, 3=forest
  - aspect (0-7): N,NE,E,SE,S,SW,W,NW — derived from elevation gradient
  - is_water: boolean mask for water bodies
  - structures: list of {row, col, priority} dicts
  - outposts: list of {outpost_id, row, col, is_airbase, resources} dicts

All generation is seeded for deterministic replay.  Each difficulty level
defines *ranges* for every scenario parameter; the seed draws a specific
value from each range.  Same seed + same difficulty = identical episode.
"""

import math
import zlib
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# --- Constants ---

GRID_SIZE = 15

# Fuel types
FUEL_NONE = 0    # rock / road / bare
FUEL_GRASS = 1
FUEL_BRUSH = 2
FUEL_FOREST = 3

# Aspect directions (8-way compass)
ASPECT_N = 0
ASPECT_NE = 1
ASPECT_E = 2
ASPECT_SE = 3
ASPECT_S = 4
ASPECT_SW = 5
ASPECT_W = 6
ASPECT_NW = 7

ASPECT_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Aspect modifiers.
# These are reduced-order solar exposure proxies anchored to NWCG S-190 guidance:
# north-facing slopes stay cooler/wetter, south-facing slopes stay hotter/drier,
# and fuels in full sun can hold roughly 8 percentage points less moisture than
# comparable shaded fuels during the day. We keep the air-temperature effect
# modest and push most of the aspect signal into fuel moisture and drying.
# Source basis:
# - NWCG S-190: north slopes -> lower temps / higher humidity / higher moisture
# - NWCG S-190: south slopes -> hotter / lower humidity / rapid moisture loss
# - NWCG S-190: fuels in full sun may contain up to ~8% less moisture than shade

# Temperature modifier per aspect (°C) — south-facing is somewhat hotter
ASPECT_TEMP_MOD = np.array([
    -1.0,  # N
    -0.5,  # NE
     0.0,  # E
    +0.5,  # SE
    +1.0,  # S
    +0.5,  # SW
     0.0,  # W
    -0.5,  # NW
])

# Initial fuel moisture modifier per aspect — south-facing starts drier
ASPECT_MOISTURE_MOD = np.array([
    +0.04,  # N
    +0.02,  # NE
     0.00,  # E
    -0.02,  # SE
    -0.04,  # S
    -0.02,  # SW
     0.00,  # W
    +0.02,  # NW
])

# Fuel moisture drying rate multiplier per aspect
ASPECT_DRYING_MULT = np.array([
    0.85,  # N    — shaded, dries slowly
    0.90,  # NE
    1.0,  # E
    1.10,  # SE
    1.15,  # S   — sun-exposed, dries fastest
    1.10,  # SW
    1.0,  # W
    0.90,  # NW
])


# ---------------------------------------------------------------------------
# Outpost position candidates — road-accessible grid-edge locations for
# ground staging, and off-grid positions for airbases.
# ---------------------------------------------------------------------------

# Ground outposts sit on the grid edge (corners and midpoints of each side).
# These represent road junctions where crews, engines, and dozers stage.
_GROUND_OUTPOST_CANDIDATES = [
    (0.0, 0.0),       # NW corner
    (0.0, 7.0),       # N midpoint
    (0.0, 14.0),      # NE corner
    (7.0, 0.0),       # W midpoint
    (7.0, 14.0),      # E midpoint
    (14.0, 0.0),      # SW corner
    (14.0, 7.0),      # S midpoint
    (14.0, 14.0),     # SE corner
]

# Air bases sit off the grid (helicopters/airtankers need helipads/runways).
_AIRBASE_CANDIDATES = [
    (-4.0, 7.0),      # North helibase
    (18.0, 7.0),      # South helibase
    (-8.0, 7.0),      # Far north airfield
    (7.0, -5.0),      # West airfield
    (7.0, 19.0),      # East airfield
]

# Outpost naming: NATO phonetic alphabet
_OUTPOST_NAMES = [
    "alpha", "bravo", "charlie", "delta", "echo",
    "foxtrot", "golf", "hotel", "india", "juliet",
]


# ---------------------------------------------------------------------------
# TerrainConfig — a fully resolved scenario (all values concrete, no ranges)
# ---------------------------------------------------------------------------

@dataclass
class TerrainConfig:
    """Configuration for terrain generation — all values are concrete."""
    grid_size: int = GRID_SIZE
    seed: int = 42

    # Elevation
    elevation_roughness: float = 0.5  # 0=flat, 1=very rough
    elevation_max: int = 9

    # Fuel distribution probabilities
    fuel_probs: tuple = (0.05, 0.35, 0.35, 0.25)  # none, grass, brush, forest

    # Water
    num_water_bodies: int = 2
    water_body_size: int = 2  # cells per water body cluster

    # Structures
    structures: list = field(default_factory=lambda: [
        {"priority": 1},  # house (low)
        {"priority": 2},  # school (medium)
    ])

    # Ignition
    ignition_points: list = field(default_factory=lambda: [
        {"step": 0},  # immediate ignition
    ])

    # Atmosphere initial conditions
    initial_temperature: float = 28.0   # °C
    initial_humidity: float = 0.45      # 0-1
    initial_wind_speed: float = 12.0    # km/h
    initial_wind_direction: float = 45.0  # degrees (0=N, 90=E)

    # Diurnal amplitude
    temp_amplitude: float = 8.0    # °C swing over the episode
    humidity_swing: float = 0.15   # humidity swing over the episode

    # Resources (for the agent — stored in config for task defs)
    resources: dict = field(default_factory=lambda: {
        "crews": 4, "engines": 3, "helicopters": 2,
        "airtankers": 1, "dozers": 2, "smokejumpers": 1,
    })

    # Outposts — list of {outpost_id, row, col, is_airbase, resources: dict}
    outposts: list = field(default_factory=list)

    # Episode
    max_steps: int = 20


# ---------------------------------------------------------------------------
# DifficultySpec — parameter ranges for a difficulty level
# ---------------------------------------------------------------------------

@dataclass
class DifficultySpec:
    """Parameter ranges for a difficulty level.

    Each tuple field is (min, max).  ``draw_scenario()`` uses the seed to
    draw a specific concrete value from each range, producing a
    ``TerrainConfig`` that is fully deterministic for that seed.
    """
    # Terrain
    elevation_roughness: tuple[float, float] = (0.2, 0.4)
    num_water_bodies: tuple[int, int] = (2, 3)
    fuel_grass: tuple[float, float] = (0.30, 0.45)
    fuel_brush: tuple[float, float] = (0.25, 0.35)
    fuel_forest: tuple[float, float] = (0.15, 0.25)

    # Weather
    temperature: tuple[float, float] = (20.0, 28.0)
    humidity: tuple[float, float] = (0.45, 0.65)
    wind_speed: tuple[float, float] = (5.0, 12.0)
    temp_amplitude: tuple[float, float] = (5.0, 8.0)
    humidity_swing: tuple[float, float] = (0.08, 0.15)

    # Fire
    ignitions_step0: tuple[int, int] = (1, 1)
    delayed_ignitions: tuple[int, int] = (0, 0)
    delayed_step_range: tuple[int, int] = (3, 8)

    # Structures
    num_structures: tuple[int, int] = (2, 2)
    max_priority: int = 1

    # Resource budget caps (total across all outposts)
    crews: tuple[int, int] = (3, 5)
    engines: tuple[int, int] = (2, 4)
    helicopters: tuple[int, int] = (1, 2)
    airtankers: tuple[int, int] = (0, 1)
    dozers: tuple[int, int] = (1, 2)
    smokejumpers: tuple[int, int] = (0, 1)

    # Outposts
    num_ground_outposts: tuple[int, int] = (2, 3)
    num_air_bases: tuple[int, int] = (1, 1)

    # Episode
    max_steps: int = 20
    grid_size: int = GRID_SIZE


def _draw_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    """Inclusive random int from [lo, hi]."""
    if lo >= hi:
        return lo
    return int(rng.integers(lo, hi + 1))


def _draw_float(rng: np.random.Generator, lo: float, hi: float) -> float:
    """Uniform random float from [lo, hi]."""
    if lo >= hi:
        return lo
    return float(rng.uniform(lo, hi))


def _rng_for(seed: int, stream: str) -> np.random.Generator:
    """Create a deterministic substream RNG for one scenario component.

    This keeps difficulty-axis changes isolated. For example, increasing the
    number of ignitions should not also silently change the sampled resources
    or structure priorities for the same task seed.
    """
    stream_id = zlib.crc32(stream.encode("utf-8")) & 0xFFFFFFFF
    return np.random.default_rng(np.random.SeedSequence([seed, stream_id]))


def draw_scenario(spec: DifficultySpec, seed: int) -> TerrainConfig:
    """Draw a concrete ``TerrainConfig`` from a difficulty spec using *seed*.

    The same (spec, seed) pair always produces the identical config.
    """
    size = spec.grid_size
    terrain_rng = _rng_for(seed, "terrain")
    weather_rng = _rng_for(seed, "weather")
    fire_rng = _rng_for(seed, "fire")
    structures_rng = _rng_for(seed, "structures")
    resources_rng = _rng_for(seed, "resources")
    outposts_rng = _rng_for(seed, "outposts")

    # --- Terrain ---
    roughness = _draw_float(terrain_rng, *spec.elevation_roughness)
    n_water = _draw_int(terrain_rng, *spec.num_water_bodies)

    grass = _draw_float(terrain_rng, *spec.fuel_grass)
    brush = _draw_float(terrain_rng, *spec.fuel_brush)
    forest = _draw_float(terrain_rng, *spec.fuel_forest)
    none_frac = max(0.02, 1.0 - grass - brush - forest)
    total = none_frac + grass + brush + forest
    fuel_probs = (none_frac / total, grass / total, brush / total, forest / total)

    # --- Weather ---
    temperature = _draw_float(weather_rng, *spec.temperature)
    humidity = _draw_float(weather_rng, *spec.humidity)
    wind_speed = _draw_float(weather_rng, *spec.wind_speed)
    wind_direction = _draw_float(weather_rng, 0.0, 360.0)
    temp_amp = _draw_float(weather_rng, *spec.temp_amplitude)
    hum_swing = _draw_float(weather_rng, *spec.humidity_swing)

    # --- Fire (ignitions) ---
    n_step0 = _draw_int(fire_rng, *spec.ignitions_step0)
    n_delayed = _draw_int(fire_rng, *spec.delayed_ignitions)

    ignition_points: list[dict] = [{"step": 0} for _ in range(n_step0)]
    for _ in range(n_delayed):
        step = _draw_int(fire_rng, *spec.delayed_step_range)
        ignition_points.append({"step": step})

    # --- Structures ---
    n_structures = _draw_int(structures_rng, *spec.num_structures)
    structures: list[dict] = []
    for i in range(n_structures):
        # Priority distribution: mostly low, some high
        if spec.max_priority == 1:
            p = 1
        else:
            p = _draw_int(structures_rng, 1, spec.max_priority)
        structures.append({"priority": p})

    # --- Resources ---
    resource_counts = {
        "crews": _draw_int(resources_rng, *spec.crews),
        "engines": _draw_int(resources_rng, *spec.engines),
        "helicopters": _draw_int(resources_rng, *spec.helicopters),
        "airtankers": _draw_int(resources_rng, *spec.airtankers),
        "dozers": _draw_int(resources_rng, *spec.dozers),
        "smokejumpers": _draw_int(resources_rng, *spec.smokejumpers),
    }

    # --- Outposts ---
    outposts = _generate_outposts(outposts_rng, size, resource_counts, spec)

    return TerrainConfig(
        grid_size=size,
        seed=seed,
        elevation_roughness=roughness,
        num_water_bodies=n_water,
        fuel_probs=fuel_probs,
        structures=structures,
        ignition_points=ignition_points,
        initial_temperature=temperature,
        initial_humidity=humidity,
        initial_wind_speed=wind_speed,
        initial_wind_direction=wind_direction,
        temp_amplitude=temp_amp,
        humidity_swing=hum_swing,
        resources=resource_counts,
        outposts=outposts,
        max_steps=spec.max_steps,
    )


def _generate_outposts(
    rng: np.random.Generator,
    size: int,
    resource_counts: dict[str, int],
    spec: DifficultySpec,
) -> list[dict]:
    """Generate outpost positions and distribute resources across them.

    Ground outposts (edges/corners) receive crews, engines, dozers.
    Air bases (off-grid) receive helicopters, airtankers, smokejumpers.
    """
    n_ground = _draw_int(rng, *spec.num_ground_outposts)
    n_air = _draw_int(rng, *spec.num_air_bases)

    # Pick ground positions (shuffle candidates, take first n)
    ground_pool = list(_GROUND_OUTPOST_CANDIDATES)
    rng.shuffle(ground_pool)
    ground_positions = ground_pool[:n_ground]

    # Pick air positions
    air_pool = list(_AIRBASE_CANDIDATES)
    rng.shuffle(air_pool)
    air_positions = air_pool[:n_air]

    outposts: list[dict] = []
    name_idx = 0

    # Create ground outposts
    for row, col in ground_positions:
        outposts.append({
            "outpost_id": _OUTPOST_NAMES[name_idx % len(_OUTPOST_NAMES)],
            "row": row,
            "col": col,
            "is_airbase": False,
            "resources": {},
        })
        name_idx += 1

    # Create air bases
    for row, col in air_positions:
        outposts.append({
            "outpost_id": _OUTPOST_NAMES[name_idx % len(_OUTPOST_NAMES)],
            "row": row,
            "col": col,
            "is_airbase": True,
            "resources": {},
        })
        name_idx += 1

    if not outposts:
        # Fallback: at least one ground outpost at (0, 0)
        outposts.append({
            "outpost_id": "alpha",
            "row": 0.0,
            "col": 0.0,
            "is_airbase": False,
            "resources": {},
        })

    # Distribute resources across outposts
    ground_outposts = [o for o in outposts if not o["is_airbase"]]
    air_outposts = [o for o in outposts if o["is_airbase"]]

    # Ground resources go to ground outposts (round-robin with random start)
    for rtype in ("crews", "engines", "dozers"):
        count = resource_counts.get(rtype, 0)
        targets = ground_outposts if ground_outposts else outposts
        for i in range(count):
            target = targets[i % len(targets)]
            target["resources"][rtype] = target["resources"].get(rtype, 0) + 1

    # Aerial resources go to air bases
    for rtype in ("helicopters", "airtankers", "smokejumpers"):
        count = resource_counts.get(rtype, 0)
        targets = air_outposts if air_outposts else outposts
        for i in range(count):
            target = targets[i % len(targets)]
            target["resources"][rtype] = target["resources"].get(rtype, 0) + 1

    return outposts


# ---------------------------------------------------------------------------
# Terrain data class
# ---------------------------------------------------------------------------

@dataclass
class Terrain:
    """Generated terrain with all static layers."""
    elevation: np.ndarray        # (H, W) int 0-9
    fuel_type: np.ndarray        # (H, W) int 0-3
    aspect: np.ndarray           # (H, W) int 0-7
    is_water: np.ndarray         # (H, W) bool
    structures: list             # [{row, col, priority}, ...]
    ignition_points: list        # [{row, col, step}, ...]
    initial_fuel_moisture: np.ndarray  # (H, W) float 0-1
    config: TerrainConfig
    outposts: list = field(default_factory=list)  # [{outpost_id, row, col, is_airbase, resources}]


# ---------------------------------------------------------------------------
# Internal generation helpers (unchanged)
# ---------------------------------------------------------------------------

def _generate_elevation(size: int, roughness: float, max_elev: int,
                        rng: np.random.Generator) -> np.ndarray:
    """
    Generate a smooth elevation map using multi-octave noise.

    Uses layered random noise with Gaussian smoothing to create
    natural-looking terrain.
    """
    elevation = np.zeros((size, size), dtype=np.float64)

    # Layer multiple octaves of noise
    for octave in range(4):
        scale = 2 ** octave
        # Generate random noise at this scale
        noise_size = max(2, size // scale)
        noise = rng.random((noise_size, noise_size))

        # Upscale with interpolation (simple bilinear via repeat + smooth)
        if noise_size < size:
            # Repeat to fill grid
            repeat_factor = (size + noise_size - 1) // noise_size
            noise = np.tile(noise, (repeat_factor, repeat_factor))[:size, :size]

        # Weight by octave (lower octaves = broader features)
        weight = roughness ** octave
        elevation += noise * weight

    # Normalize to 0-max_elev
    elevation -= elevation.min()
    if elevation.max() > 0:
        elevation = elevation / elevation.max() * max_elev

    # Apply gentle smoothing for natural look when SciPy is available.
    try:
        from scipy.ndimage import uniform_filter
        elevation = uniform_filter(elevation, size=3, mode='reflect')
    except ImportError:
        # Fallback: simple averaging without scipy
        padded = np.pad(elevation, 1, mode='reflect')
        elevation = np.zeros((size, size))
        for di in range(-1, 2):
            for dj in range(-1, 2):
                elevation += padded[1+di:size+1+di, 1+dj:size+1+dj]
        elevation /= 9.0

    return np.round(elevation).astype(np.int32).clip(0, max_elev)


def _compute_aspect(elevation: np.ndarray) -> np.ndarray:
    """
    Compute aspect (slope direction) from elevation using gradient.

    Returns 8-way compass direction (0=N, 1=NE, ... 7=NW).
    For flat areas (no gradient), defaults to South (most conservative for fire).
    """
    size = elevation.shape[0]
    aspect = np.full((size, size), ASPECT_S, dtype=np.int32)

    # Compute gradient (dy=north-south, dx=east-west)
    # Pad for edge handling
    padded = np.pad(elevation.astype(np.float64), 1, mode='reflect')

    for r in range(size):
        for c in range(size):
            pr, pc = r + 1, c + 1  # padded coordinates

            # Gradient: which direction does the slope FACE?
            # dx > 0 means slope faces east, dy > 0 means slope faces south
            dx = float(padded[pr, pc + 1] - padded[pr, pc - 1])
            dy = float(padded[pr + 1, pc] - padded[pr - 1, pc])

            if abs(dx) < 0.01 and abs(dy) < 0.01:
                continue  # flat -> keep default (S)

            # Angle in degrees (0=N=up, 90=E=right)
            angle = np.degrees(np.arctan2(dx, -dy)) % 360

            # Quantize to 8 directions
            aspect[r, c] = int((angle + 22.5) // 45) % 8

    return aspect


def _generate_fuel(size: int, elevation: np.ndarray,
                   fuel_probs: tuple, rng: np.random.Generator) -> np.ndarray:
    """
    Generate fuel type map correlated with elevation.

    Lower elevations tend toward grass/brush.
    Higher elevations tend toward forest.
    Rocky peaks have no fuel.
    """
    fuel = np.zeros((size, size), dtype=np.int32)
    max_elev = elevation.max()

    for r in range(size):
        for c in range(size):
            elev = elevation[r, c]
            elev_ratio = elev / max(max_elev, 1)

            # Adjust probabilities by elevation
            probs = list(fuel_probs)

            # High elevation: more forest, less grass
            if elev_ratio > 0.7:
                probs[FUEL_NONE] += 0.10   # rocky peaks
                probs[FUEL_GRASS] -= 0.10
                probs[FUEL_FOREST] += 0.10
                probs[FUEL_BRUSH] -= 0.10
            # Low elevation: more grass
            elif elev_ratio < 0.3:
                probs[FUEL_GRASS] += 0.15
                probs[FUEL_FOREST] -= 0.10
                probs[FUEL_BRUSH] -= 0.05

            # Normalize
            probs = np.array(probs).clip(0)
            probs = probs / probs.sum()

            fuel[r, c] = rng.choice(4, p=probs)

    return fuel


def _place_water(size: int, num_bodies: int, body_size: int,
                 elevation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Place water bodies at low-elevation areas.

    Water naturally collects in valleys (low elevation).
    """
    is_water = np.zeros((size, size), dtype=bool)

    if num_bodies <= 0:
        return is_water

    # Find lowest elevation cells as candidates
    flat_elev = elevation.flatten()
    low_indices = np.argsort(flat_elev)

    # Pick random low-elevation cells as water centers
    candidates = low_indices[:max(len(low_indices) // 3, num_bodies)]
    centers = rng.choice(candidates, size=min(num_bodies, len(candidates)),
                         replace=False)

    for center_flat in centers:
        cr, cc = divmod(int(center_flat), size)
        # Place a cluster around the center
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if rng.random() < 0.6 or (dr == 0 and dc == 0):
                        is_water[nr, nc] = True
                        body_size -= 1
                        if body_size <= 0:
                            break

    return is_water


def _place_structures(size: int, structure_defs: list,
                      elevation: np.ndarray, is_water: np.ndarray,
                      fuel_type: np.ndarray,
                      rng: np.random.Generator) -> list:
    """
    Place structures on the map.

    Structures go on moderate-elevation, non-water cells with fuel.
    Returns list of {row, col, priority} dicts.
    """
    structures = []
    used = set()

    for sdef in structure_defs:
        priority = sdef.get("priority", 1)

        for _ in range(100):  # max attempts
            r = rng.integers(1, size - 1)
            c = rng.integers(1, size - 1)

            if (r, c) in used:
                continue
            if is_water[r, c]:
                continue
            if fuel_type[r, c] == FUEL_NONE:
                continue

            # Prefer moderate elevation (not too high, not too low)
            elev = elevation[r, c]
            max_elev = elevation.max()
            if elev > max_elev * 0.8:
                continue

            structures.append({"row": int(r), "col": int(c), "priority": priority})
            used.add((r, c))
            break

    return structures


def _place_ignitions(size: int, ignition_defs: list,
                     elevation: np.ndarray, is_water: np.ndarray,
                     fuel_type: np.ndarray, structures: list,
                     rng: np.random.Generator) -> list:
    """
    Place ignition points away from structures and water.
    """
    ignitions = []
    structure_positions = {(s["row"], s["col"]) for s in structures}
    used = set()

    for idef in ignition_defs:
        step = idef.get("step", 0)

        for _ in range(200):  # max attempts
            r = rng.integers(2, size - 2)
            c = rng.integers(2, size - 2)

            if (r, c) in used:
                continue
            if is_water[r, c]:
                continue
            if fuel_type[r, c] == FUEL_NONE:
                continue
            if (r, c) in structure_positions:
                continue

            # Ensure minimum distance from structures
            too_close = False
            for s in structures:
                dist = abs(r - s["row"]) + abs(c - s["col"])
                if dist < 4:
                    too_close = True
                    break
            if too_close:
                continue

            ignitions.append({"row": int(r), "col": int(c), "step": step})
            used.add((r, c))
            break

    return ignitions


def _compute_initial_moisture(fuel_type: np.ndarray, elevation: np.ndarray,
                              aspect: np.ndarray, is_water: np.ndarray,
                              humidity: float) -> np.ndarray:
    """
    Compute initial fuel moisture based on terrain properties.

    Moisture is influenced by:
    - Equilibrium moisture content (EMC) from humidity: EMC ~ humidity / 5
    - Elevation (higher = cooler = wetter)
    - Aspect (south-facing = drier)
    - Proximity to water (wetter)
    """
    size = fuel_type.shape[0]

    # Base: equilibrium moisture content (Rule of 5)
    emc = humidity / 5.0  # e.g., humidity 0.45 -> EMC = 0.09

    # Start at a moderate moisture level above EMC
    moisture = np.full((size, size), emc + 0.15, dtype=np.float64)

    # Elevation modifier: +0.02 per level (higher = cooler = wetter)
    max_elev = max(elevation.max(), 1)
    moisture += (elevation / max_elev) * 0.06

    # Aspect modifier
    for r in range(size):
        for c in range(size):
            moisture[r, c] += ASPECT_MOISTURE_MOD[aspect[r, c]]

    # Water proximity: cells adjacent to water get +0.05
    padded_water = np.pad(is_water, 1, mode='constant', constant_values=False)
    for r in range(size):
        for c in range(size):
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if padded_water[r + 1 + dr, c + 1 + dc]:
                        moisture[r, c] += 0.03
                        break
                else:
                    continue
                break

    # No fuel = no moisture
    moisture[fuel_type == FUEL_NONE] = 0.0
    moisture[is_water] = 1.0  # water cells are "fully wet"

    return moisture.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# Main terrain generation entry point
# ---------------------------------------------------------------------------

def generate_terrain(config: Optional[TerrainConfig] = None) -> Terrain:
    """
    Generate a complete terrain from configuration.

    Args:
        config: TerrainConfig with all generation parameters.
                Defaults to easy task configuration with default seed.

    Returns:
        Terrain object with all static layers.
    """
    if config is None:
        config = get_task_config("easy")

    rng = np.random.default_rng(config.seed)
    size = config.grid_size

    # Generate layers in dependency order
    elevation = _generate_elevation(
        size, config.elevation_roughness, config.elevation_max, rng
    )

    aspect = _compute_aspect(elevation)

    fuel_type = _generate_fuel(size, elevation, config.fuel_probs, rng)

    is_water = _place_water(
        size, config.num_water_bodies, config.water_body_size, elevation, rng
    )

    # Water cells have no fuel
    fuel_type[is_water] = FUEL_NONE

    structures = _place_structures(
        size, config.structures, elevation, is_water, fuel_type, rng
    )

    ignition_points = _place_ignitions(
        size, config.ignition_points, elevation, is_water, fuel_type,
        structures, rng
    )

    initial_moisture = _compute_initial_moisture(
        fuel_type, elevation, aspect, is_water, config.initial_humidity
    )

    return Terrain(
        elevation=elevation,
        fuel_type=fuel_type,
        aspect=aspect,
        is_water=is_water,
        structures=structures,
        ignition_points=ignition_points,
        initial_fuel_moisture=initial_moisture,
        config=config,
        outposts=config.outposts,
    )


# ---------------------------------------------------------------------------
# Difficulty specs and task configuration helpers
# ---------------------------------------------------------------------------

DIFFICULTY_SPECS: dict[str, DifficultySpec] = {
    "easy": DifficultySpec(
        elevation_roughness=(0.2, 0.4),
        num_water_bodies=(2, 3),
        fuel_grass=(0.30, 0.45),
        fuel_brush=(0.25, 0.35),
        fuel_forest=(0.10, 0.25),
        temperature=(20.0, 28.0),
        humidity=(0.45, 0.65),
        wind_speed=(5.0, 12.0),
        temp_amplitude=(5.0, 8.0),
        humidity_swing=(0.08, 0.12),
        ignitions_step0=(1, 1),
        delayed_ignitions=(0, 0),
        num_structures=(2, 3),
        max_priority=1,
        crews=(3, 5),
        engines=(2, 4),
        helicopters=(1, 2),
        airtankers=(0, 1),
        dozers=(1, 2),
        smokejumpers=(0, 1),
        num_ground_outposts=(2, 3),
        num_air_bases=(1, 1),
    ),
    "medium": DifficultySpec(
        elevation_roughness=(0.4, 0.6),
        num_water_bodies=(1, 2),
        fuel_grass=(0.20, 0.35),
        fuel_brush=(0.30, 0.40),
        fuel_forest=(0.20, 0.35),
        temperature=(26.0, 34.0),
        humidity=(0.30, 0.50),
        wind_speed=(10.0, 20.0),
        temp_amplitude=(8.0, 12.0),
        humidity_swing=(0.12, 0.20),
        ignitions_step0=(2, 2),
        delayed_ignitions=(1, 1),
        delayed_step_range=(3, 8),
        num_structures=(3, 4),
        max_priority=2,
        crews=(2, 4),
        engines=(1, 3),
        helicopters=(1, 2),
        airtankers=(0, 1),
        dozers=(1, 2),
        smokejumpers=(0, 1),
        num_ground_outposts=(2, 3),
        num_air_bases=(1, 2),
        max_steps=15,
    ),
    "hard": DifficultySpec(
        elevation_roughness=(0.5, 0.8),
        num_water_bodies=(0, 1),
        fuel_grass=(0.15, 0.25),
        fuel_brush=(0.30, 0.40),
        fuel_forest=(0.30, 0.45),
        temperature=(30.0, 40.0),
        humidity=(0.20, 0.35),
        wind_speed=(18.0, 28.0),
        temp_amplitude=(10.0, 15.0),
        humidity_swing=(0.18, 0.25),
        ignitions_step0=(2, 3),
        delayed_ignitions=(1, 3),
        delayed_step_range=(3, 12),
        num_structures=(4, 6),
        max_priority=3,
        crews=(1, 3),
        engines=(1, 2),
        helicopters=(0, 1),
        airtankers=(0, 0),
        dozers=(0, 1),
        smokejumpers=(0, 0),
        num_ground_outposts=(1, 2),
        num_air_bases=(1, 1),
    ),
}

# Default seeds for reproducible baselines.  The baseline endpoint and
# inference script use these so that ``openenv validate`` and the hackathon
# evaluator always see the same episodes.
DEFAULT_SEEDS: dict[str, int] = {
    "easy": 42,
    "medium": 34,
    "hard": 12,
}


def get_task_config(task_id: str, seed: int | None = None) -> TerrainConfig:
    """Return a fully resolved ``TerrainConfig`` for a task + seed.

    If *seed* is ``None``, uses the default reproducible seed for that task.
    """
    spec = DIFFICULTY_SPECS[task_id]
    if seed is None:
        seed = DEFAULT_SEEDS[task_id]
    return draw_scenario(spec, seed)


# Backwards-compatible alias used by grader — maps task_id to the default
# config.  Lazy-evaluated to avoid circular imports.
class _TaskConfigProxy(dict):
    """Lazily generates TerrainConfig on first access per task_id."""

    def __missing__(self, task_id: str) -> TerrainConfig:
        config = get_task_config(task_id)
        self[task_id] = config
        return config

    def get(self, task_id, default=None):  # type: ignore[override]
        try:
            return self[task_id]
        except KeyError:
            return default


TASK_CONFIGS = _TaskConfigProxy()

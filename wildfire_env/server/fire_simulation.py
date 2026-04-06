"""
Fire Simulation Engine — Cellular Automata with Full Physics.

The core wildfire spread model implementing all 15 parameters:
  Static: fuel_type, elevation, aspect, is_water, structure
  Dynamic per-cell: state, fuel_moisture, burn_timer, heat, intensity
  Global: wind_speed, wind_direction, temperature, humidity, time_of_day

Based on:
  - Alexandridis et al. (2008) CA spread model (p0=0.58, c1=0.045, c2=0.131)
  - Rothermel (1972) moisture damping polynomial
  - Standard atmospheric physics (lapse rate, EMC Rule-of-5)
  - Diurnal cycle research
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

try:
    from .terrain import (
        Terrain, TerrainConfig,
        FUEL_NONE, FUEL_GRASS, FUEL_BRUSH, FUEL_FOREST,
        ASPECT_TEMP_MOD, ASPECT_DRYING_MULT, GRID_SIZE,
    )
except ImportError:
    from terrain import (
        Terrain, TerrainConfig,
        FUEL_NONE, FUEL_GRASS, FUEL_BRUSH, FUEL_FOREST,
        ASPECT_TEMP_MOD, ASPECT_DRYING_MULT, GRID_SIZE,
    )


# ──────────────────────────────────────────────────────────
# Cell States
# ──────────────────────────────────────────────────────────

STATE_UNBURNED = 0
STATE_BURNING = 1
STATE_BURNED = 2
STATE_FIREBREAK = 3
STATE_WATER = 4
STATE_STRUCTURE = 5
STATE_SUPPRESSED = 6

STATE_NAMES = {
    0: "unburned", 1: "burning", 2: "burned",
    3: "firebreak", 4: "water", 5: "structure", 6: "suppressed",
}


# ──────────────────────────────────────────────────────────
# Physics Constants (from published research)
# ──────────────────────────────────────────────────────────

# Alexandridis (2008) base probability
P0 = 0.58

# Wind coefficients (Alexandridis et al. 2008)
# The published wind-loading formulation is:
#   p_w = exp(V * (c1 + c2 * (cos(theta) - 1)))
# where V is wind speed and theta is the angle between wind direction and the
# direction of spread. The calibration in Alexandridis/Freire uses 100 m cells
# and an effective time step of about 20 minutes, so we use that as the base
# temporal/spatial interpretation for this reduced-order environment.
C_WIND = 0.045
C_WIND_DIRECTIONAL = 0.131

# Synthetic grid scale used to turn abstract elevation levels into slope angles.
# This is an explicit modeling assumption for our abstract 15x15 terrain, not a
# directly observed site measurement.
CELL_SIZE_METERS = 100.0
ELEVATION_LEVEL_METERS = 100.0
SIMULATION_STEP_MINUTES = 20.0
SIMULATION_STEP_HOURS = SIMULATION_STEP_MINUTES / 60.0

# Slope coefficient (Alexandridis, from Rothermel)
C_SLOPE = 0.078

# Fuel type spread modifier (p_veg, Alexandridis Table 2)
FUEL_SPREAD_MOD = {
    FUEL_NONE: -1.0,    # can't burn
    FUEL_GRASS: -0.3,
    FUEL_BRUSH: 0.0,
    FUEL_FOREST: 0.4,
}

# Fuel burn duration (ticks)
FUEL_MAX_BURN = {
    FUEL_NONE: 0,
    FUEL_GRASS: 2,
    FUEL_BRUSH: 4,
    FUEL_FOREST: 6,
}

# Fuel heat absorption rate
FUEL_HEAT_ABSORB = {
    FUEL_NONE: 0.0,
    FUEL_GRASS: 0.8,
    FUEL_BRUSH: 1.0,
    FUEL_FOREST: 1.2,
}

# Fuel peak intensity (Byram I=Hwr proportional)
FUEL_PEAK_INTENSITY = {
    FUEL_NONE: 0.0,
    FUEL_GRASS: 0.7,
    FUEL_BRUSH: 1.0,
    FUEL_FOREST: 1.3,
}

# Fuel moisture response rate per fuel type.
# We model the effective fine dead fuel response using NWCG time-lag classes and
# an exponential approach to equilibrium:
#   alpha = 1 - exp(-dt / tau)
# The abstract fuel groups are mapped to the dominant fine-fuel response they
# present to surface spread:
# - grass: 1 h fuels
# - brush: 10 h fuels
# - forest litter / understory: 10 h fuels
FUEL_MOISTURE_TIMELAG_HOURS = {
    FUEL_NONE: 0.0,
    FUEL_GRASS: 1.0,
    FUEL_BRUSH: 10.0,
    FUEL_FOREST: 10.0,
}

FUEL_MOISTURE_RATE = {
    fuel: (
        0.0 if tau <= 0.0 else 1.0 - math.exp(-SIMULATION_STEP_HOURS / tau)
    )
    for fuel, tau in FUEL_MOISTURE_TIMELAG_HOURS.items()
}

# Representative moisture of extinction by broad fuel class.
# Scott & Burgan fuel models span a range of extinction moistures; these are
# broad-category representatives used with the Rothermel damping polynomial.
FUEL_MOISTURE_EXTINCTION = {
    FUEL_NONE: 1.0,
    FUEL_GRASS: 0.15,
    FUEL_BRUSH: 0.25,
    FUEL_FOREST: 0.30,
}

# Structure fire intensity multiplier
STRUCTURE_INTENSITY_MULT = 1.5

# Ignition threshold (heat accumulation)
IGNITION_THRESHOLD = 0.4

# Temperature lapse rate per elevation unit
LAPSE_RATE = 0.65  # °C per level

# Fire radiant heat contribution to local temperature
FIRE_TEMP_CONTRIBUTION = 15.0  # °C per adjacent fully-intense cell

# Fuel moisture fire drying rate
FIRE_DRYING_RATE = 0.05  # per adjacent burning cell at full intensity

# Water recharge rate for adjacent cells
WATER_RECHARGE_RATE = 0.02

# Water humidity bonus for adjacent cells
WATER_HUMIDITY_BONUS = 0.05

# Spotting parameters
# Freire and DaCamara (2019), following the Alexandridis family of CA models,
# introduce downwind nonlocal propagation once wind exceeds 8 m/s.
SPOT_WIND_THRESHOLD = 28.8  # km/h
SPOT_INTENSITY_THRESHOLD = 3.0
SPOT_MAX_PROB = 0.25
SPOT_DISTANCE_CAP = 8

# Fire-atmosphere coupling via a hidden airflow-potential field.
# This is a reduced-order stand-in for plume-driven entrainment: fire intensity
# raises local buoyancy/pressure-deficit potential, the field diffuses and
# decays, and its spatial gradient perturbs the ambient wind.
AIRFLOW_POTENTIAL_DECAY = 0.72
AIRFLOW_POTENTIAL_DIFFUSION = 0.35
AIRFLOW_POTENTIAL_SOURCE_GAIN = 0.9
AIRFLOW_WIND_SCALE = 4.0  # km/h per unit potential gradient

# 8-direction neighbor offsets (dr, dc) and distances
NEIGHBORS_8 = [
    (-1, 0, 1.0),     # N
    (-1, 1, 1.414),   # NE
    (0, 1, 1.0),      # E
    (1, 1, 1.414),    # SE
    (1, 0, 1.0),      # S
    (1, -1, 1.414),   # SW
    (0, -1, 1.0),     # W
    (-1, -1, 1.414),  # NW
]

# Direction angles (rad) for each neighbor (direction FROM source TO target)
NEIGHBOR_ANGLES = [
    0.0,                # N → 0°
    math.pi / 4,        # NE → 45°
    math.pi / 2,        # E → 90°
    3 * math.pi / 4,    # SE → 135°
    math.pi,            # S → 180°
    5 * math.pi / 4,    # SW → 225°
    3 * math.pi / 2,    # W → 270°
    7 * math.pi / 4,    # NW → 315°
]


# Saturation vapor pressure coefficients over water from NASA POWER.
# Valid over the temperature range used by this environment.
SAT_VAP_A1 = 6.11213476
SAT_VAP_A2 = 0.444007856
SAT_VAP_A3 = 0.0143064234
SAT_VAP_A4 = 0.000264461437
SAT_VAP_A5 = 0.00000305903558
SAT_VAP_A6 = 0.000000196237241
SAT_VAP_A7 = 0.000000000892344772
SAT_VAP_A8 = -0.00000000000037320841
SAT_VAP_A9 = 0.000000000000000209339997


def _saturation_vapor_pressure_hpa(temp_c: float) -> float:
    """Approximate saturation vapor pressure (hPa) over water."""
    t = float(temp_c)
    esat = (
        SAT_VAP_A1
        + SAT_VAP_A2 * t
        + SAT_VAP_A3 * (t ** 2)
        + SAT_VAP_A4 * (t ** 3)
        + SAT_VAP_A5 * (t ** 4)
        + SAT_VAP_A6 * (t ** 5)
        + SAT_VAP_A7 * (t ** 6)
        + SAT_VAP_A8 * (t ** 7)
        + SAT_VAP_A9 * (t ** 8)
    )
    return max(0.01, esat)


def _rothermel_moisture_damping(moisture: float, moisture_extinction: float) -> float:
    """
    Rothermel (1972) moisture damping coefficient.

    eta_M = 1 - 2.59 r + 5.11 r^2 - 3.52 r^3, where r = Mf / Mx.
    """
    if moisture_extinction <= 0.0:
        return 0.0

    ratio = max(0.0, moisture / moisture_extinction)
    if ratio >= 1.0:
        return 0.0

    damping = 1.0 - 2.59 * ratio + 5.11 * (ratio ** 2) - 3.52 * (ratio ** 3)
    return max(0.0, min(1.0, damping))


# ──────────────────────────────────────────────────────────
# Simulation State
# ──────────────────────────────────────────────────────────

@dataclass
class SimulationState:
    """Complete mutable state of the fire simulation."""

    # Per-cell dynamic grids
    cell_state: np.ndarray       # (H, W) int — STATE_*
    fuel_moisture: np.ndarray    # (H, W) float 0-1
    burn_timer: np.ndarray       # (H, W) int — ticks burning
    heat: np.ndarray             # (H, W) float — accumulated heat
    intensity: np.ndarray        # (H, W) float — current burn intensity
    airflow_potential: np.ndarray  # (H, W) float — fire-driven flow potential

    # Global atmosphere
    wind_speed: float            # km/h
    wind_direction: float        # degrees (0=N, 90=E)
    temperature: float           # °C
    humidity: float              # 0-1
    time_of_day: float           # 0-1 (fraction of episode)

    # Tracking
    step: int = 0
    done: bool = False

    # Statistics for each step
    total_burned: int = 0
    total_burning: int = 0
    structures_lost: int = 0
    structures_saved: int = 0

    # Cells suppressed this tick — incremented by apply_crew(), apply_bomber(),
    # apply_retardant_drop() when a BURNING cell transitions to SUPPRESSED.
    # Reset to 0 at the start of each tick().  Read by compute_environmental_rewards().
    cells_suppressed_this_step: int = 0

    # Retardant permanent ignition-threshold bonus (per-cell, cumulative,
    # capped at 0.40).  Populated by apply_retardant_drop(); read during
    # ignition checks in tick().  Initialized to None so we only allocate
    # when retardant is actually used.
    retardant_bonus: Optional[np.ndarray] = field(default=None, repr=False)

    # RNG state for determinism
    rng_state: Optional[dict] = field(default=None, repr=False)


class FireSimulation:
    """
    Core wildfire cellular automata simulation engine.
    
    Implements the complete 15-parameter interwoven system:
    - Fuel properties determine burn characteristics
    - Elevation creates slope effects and temperature lapse
    - Aspect modifies heating and moisture dynamics
    - Water bodies recharge moisture and block fire
    - Wind drives directional spread and ember spotting
    - Temperature and humidity follow diurnal cycles
    - Fire creates feedback loops (heating, drying, airflow perturbation)
    
    Usage:
        terrain = generate_terrain(config)
        sim = FireSimulation(terrain)
        state = sim.reset()
        while not state.done:
            state = sim.tick()
    """

    def __init__(self, terrain: Terrain):
        """Initialize simulation with generated terrain."""
        self.terrain = terrain
        self.config = terrain.config
        self.size = terrain.config.grid_size
        self._rng = np.random.default_rng(terrain.config.seed + 1000)
        self.state: Optional[SimulationState] = None

        # Reward tracking — resets per episode in reset(), persists across ticks.
        self._structures_lost_penalized: set[tuple[int, int]] = set()
        self._fire_extinguished_rewarded: bool = False

    def reset(self) -> SimulationState:
        """
        Reset the simulation to initial conditions.
        
        Returns the initial SimulationState.
        """
        self._rng = np.random.default_rng(self.config.seed + 1000)
        self._structures_lost_penalized = set()
        self._fire_extinguished_rewarded = False
        s = self.size

        # Initialize cell states
        cell_state = np.full((s, s), STATE_UNBURNED, dtype=np.int32)
        cell_state[self.terrain.is_water] = STATE_WATER

        # Mark structure cells
        for st in self.terrain.structures:
            cell_state[st["row"], st["col"]] = STATE_STRUCTURE

        # No-fuel cells
        cell_state[self.terrain.fuel_type == FUEL_NONE] = STATE_FIREBREAK

        # Restore water (in case fuel_type was set to NONE for water)
        cell_state[self.terrain.is_water] = STATE_WATER

        self.state = SimulationState(
            cell_state=cell_state,
            fuel_moisture=self.terrain.initial_fuel_moisture.copy(),
            burn_timer=np.zeros((s, s), dtype=np.int32),
            heat=np.zeros((s, s), dtype=np.float64),
            intensity=np.zeros((s, s), dtype=np.float64),
            airflow_potential=np.zeros((s, s), dtype=np.float64),
            wind_speed=self.config.initial_wind_speed,
            wind_direction=self.config.initial_wind_direction,
            temperature=self.config.initial_temperature,
            humidity=self.config.initial_humidity,
            time_of_day=0.0,
            step=0,
            done=False,
        )

        # Apply immediate ignitions (step=0)
        self._apply_ignitions()

        # Compute initial statistics
        self._update_statistics()

        return self.state

    def tick(self) -> SimulationState:
        """
        Advance the simulation by one step.
        
        Implements the complete physics update in the correct order:
        1. Advance time
        2. Update ambient atmosphere
        3. Update local conditions per cell (temp, humidity, moisture)
        4. Advance burn timers and update intensity
        5. Update fire-driven airflow potential
        6. Radiate heat (preheating model)
        7. Check ignitions
        8. Check ember spotting
        9. Apply delayed ignitions
        10. Update statistics
        
        Returns the updated SimulationState.
        """
        if self.state is None:
            raise RuntimeError("Call reset() before tick()")
        if self.state.done:
            return self.state

        st = self.state
        st.step += 1

        # 1. Advance time
        st.time_of_day = st.step / self.config.max_steps

        # 2. Update atmosphere
        self._update_atmosphere()

        # 3. Update local conditions (fuel moisture per cell)
        self._update_fuel_moisture()

        # 4. Update burning cells (timers + intensity)
        self._update_burning_cells()

        # 5. Update fire-driven airflow field
        self._update_airflow_potential()

        # 6. Radiate heat and check ignition
        self._radiate_heat_and_ignite()

        # 7. Ember spotting
        self._check_spotting()

        # 8. Apply delayed ignitions
        self._apply_ignitions()

        # 9. Update statistics
        self._update_statistics()

        # 10. Check episode end
        if st.step >= self.config.max_steps:
            st.done = True
        elif st.total_burning == 0 and st.step > 0:
            # Only end early when no future scheduled ignitions remain.
            # Without this guard the hard task's cascading ignitions (steps 5
            # and 10) would never fire if the agent suppresses the first fire
            # quickly — making the task trivially easy.
            future_ignitions = [
                ig for ig in self.terrain.ignition_points
                if ig["step"] > st.step
            ]
            if not future_ignitions:
                st.done = True  # fire is out and no more ignitions incoming

        return st

    # ──────────────────────────────────────────────────
    # Internal update methods
    # ──────────────────────────────────────────────────

    def _diurnal_factor(self) -> float:
        """
        Compute diurnal cycle factor (0-1).
        
        Peaks in late afternoon (~70% through the episode).
        Uses a sine curve shifted to peak at the right time.
        """
        t = self.state.time_of_day
        return max(0.0, math.sin(math.pi * t * 0.85 + 0.2))

    def _update_atmosphere(self):
        """
        Update global atmospheric conditions.
        
        Applies:
        - Diurnal temperature and humidity curves
        - Smooth ambient wind evolution with persistence and noise
        """
        st = self.state
        cfg = self.config
        df = self._diurnal_factor()

        # Temperature: base + diurnal swing
        st.temperature = cfg.initial_temperature + cfg.temp_amplitude * df

        # Humidity follows the inverse daily cycle supplied by the scenario.
        # Local RH/temperature coupling is handled later via vapor pressure.
        st.humidity = cfg.initial_humidity - cfg.humidity_swing * df
        st.humidity = max(0.05, min(1.0, st.humidity))

        # Ambient wind evolves smoothly around a diurnal target rather than
        # through discrete "gust event" switches.
        wind_diurnal = 0.6 + 0.4 * df
        target_speed = cfg.initial_wind_speed * wind_diurnal
        st.wind_speed = 0.75 * st.wind_speed + 0.25 * target_speed
        st.wind_speed += self._rng.normal(0, 0.8)
        st.wind_speed = max(0.0, min(40.0, st.wind_speed))

        # Direction retains persistence with moderate stochastic drift.
        st.wind_direction += self._rng.normal(0, 4.0)
        st.wind_direction = st.wind_direction % 360.0

    def _compute_local_microclimate(self, r: int, c: int) -> tuple[float, float]:
        """
        Estimate local temperature and humidity at a cell.

        This folds in factors that were already declared in the environment
        constants but were not yet influencing the simulation:
        - elevation lapse rate
        - aspect heating/cooling
        - radiant heating from nearby fires
        - humidity recovery from adjacent water

        Returns:
            (local_temperature_celsius, local_relative_humidity_0_to_1)
        """
        st = self.state
        t = self.terrain

        local_temp = st.temperature
        local_temp -= t.elevation[r, c] * LAPSE_RATE
        local_temp += ASPECT_TEMP_MOD[t.aspect[r, c]]

        fire_temp_bonus = 0.0
        water_neighbors = 0

        for dr, dc, dist in NEIGHBORS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                continue

            if st.cell_state[nr, nc] == STATE_BURNING:
                # Nearby flames strongly heat local fuels, but the effect drops
                # with distance and is softened slightly to keep the system
                # numerically stable.
                fire_temp_bonus += (
                    FIRE_TEMP_CONTRIBUTION * st.intensity[nr, nc] / (dist * 2.0)
                )

            if t.is_water[nr, nc]:
                water_neighbors += 1

        local_temp += fire_temp_bonus

        # Hold actual vapor pressure fixed while local temperature changes.
        # This yields the expected inverse temperature-RH relationship without
        # needing an arbitrary "RH per degree" constant.
        ambient_esat = _saturation_vapor_pressure_hpa(st.temperature)
        ambient_vapor_pressure = st.humidity * ambient_esat
        local_esat = _saturation_vapor_pressure_hpa(local_temp)
        local_humidity = ambient_vapor_pressure / local_esat

        if water_neighbors:
            local_humidity += min(
                water_neighbors * WATER_HUMIDITY_BONUS,
                WATER_HUMIDITY_BONUS * 3,
            )

        local_humidity = max(0.05, min(1.0, local_humidity))

        return local_temp, local_humidity

    def _update_fuel_moisture(self):
        """
        Update fuel moisture for all cells.
        
        Moisture drifts toward Equilibrium Moisture Content (EMC)
        at a rate determined by:
        - Fuel type (time-lag class)
        - Aspect (south dries faster)
        - Fire proximity (radiant drying)
        - Water proximity (recharge)
        - Wind (evaporation boost)
        """
        st = self.state
        t = self.terrain

        for r in range(self.size):
            for c in range(self.size):
                if st.cell_state[r, c] in (STATE_WATER, STATE_FIREBREAK,
                                            STATE_BURNED, STATE_BURNING):
                    continue

                fuel = t.fuel_type[r, c]
                if fuel == FUEL_NONE:
                    continue

                _, local_humidity = self._compute_local_microclimate(r, c)

                # A compact dead-fuel EMC proxy used operationally in fire
                # behavior training is that fine dead fuel moisture roughly
                # tracks one-fifth of RH. RH is in [0, 1] here.
                emc = local_humidity / 5.0

                # Time-lag response rate
                response_rate = FUEL_MOISTURE_RATE[fuel]

                # Aspect modifier
                aspect_mult = ASPECT_DRYING_MULT[t.aspect[r, c]]

                # Atmospheric drift toward EMC
                moisture_diff = emc - st.fuel_moisture[r, c]
                atm_change = moisture_diff * response_rate * aspect_mult

                # Diurnal aspect bonus (aspect matters more during peak sun)
                df = self._diurnal_factor()
                atm_change *= (0.5 + 0.5 * df)  # weaker at night

                # Wind evaporation boost
                wind_boost = 1.0 + st.wind_speed / 30.0
                atm_change *= wind_boost

                # Fire radiant drying
                fire_drying = 0.0
                for dr, dc, dist in NEIGHBORS_8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if st.cell_state[nr, nc] == STATE_BURNING:
                            fire_drying += st.intensity[nr, nc] * FIRE_DRYING_RATE / dist

                # 2-cell radius fire drying (for intense nearby fires)
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if abs(dr) <= 1 and abs(dc) <= 1:
                            continue  # already counted above
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            if st.cell_state[nr, nc] == STATE_BURNING:
                                dist2 = math.sqrt(dr * dr + dc * dc)
                                fire_drying += (
                                    st.intensity[nr, nc] * FIRE_DRYING_RATE
                                    * 0.3 / dist2
                                )

                # Water proximity recharge
                water_recharge = 0.0
                for dr, dc, _ in NEIGHBORS_8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        if t.is_water[nr, nc]:
                            water_recharge = WATER_RECHARGE_RATE
                            break

                # Apply changes
                st.fuel_moisture[r, c] += atm_change
                st.fuel_moisture[r, c] -= fire_drying
                st.fuel_moisture[r, c] += water_recharge
                st.fuel_moisture[r, c] = max(0.0, min(1.0,
                                             st.fuel_moisture[r, c]))

    def _update_burning_cells(self):
        """
        Update all burning cells: advance timers, compute intensity.
        
        Intensity follows a bell curve:
        - 0-25% of burn duration: ramping up
        - 25-60%: peak intensity
        - 60-100%: decaying
        
        When timer exceeds max burn duration → cell becomes BURNED_OUT.
        """
        st = self.state
        t = self.terrain

        for r in range(self.size):
            for c in range(self.size):
                if st.cell_state[r, c] != STATE_BURNING:
                    continue

                st.burn_timer[r, c] += 1
                fuel = t.fuel_type[r, c]
                max_burn = FUEL_MAX_BURN.get(fuel, 3)

                # Check for structures (which might have been set on fire
                # but terrain fuel_type might be something else)
                is_structure = any(
                    s["row"] == r and s["col"] == c
                    for s in t.structures
                )

                if max_burn == 0:
                    max_burn = 3  # default

                timer = st.burn_timer[r, c]

                if timer >= max_burn:
                    # Burned out
                    st.cell_state[r, c] = STATE_BURNED
                    st.intensity[r, c] = 0.0
                    continue

                # Intensity bell curve
                t_norm = timer / max_burn
                peak_intensity = FUEL_PEAK_INTENSITY.get(fuel, 1.0)

                if is_structure:
                    peak_intensity *= STRUCTURE_INTENSITY_MULT

                if t_norm < 0.25:
                    intensity = (t_norm / 0.25) * peak_intensity
                elif t_norm < 0.6:
                    intensity = peak_intensity
                else:
                    intensity = ((1.0 - t_norm) / 0.4) * peak_intensity

                st.intensity[r, c] = max(0.0, intensity)

    def _update_airflow_potential(self):
        """
        Update the hidden fire-driven airflow potential field.

        This is the minimum extra state we keep to allow fire-atmosphere
        feedback without hardcoding named behaviors. Burning intensity acts as
        a source term, the field diffuses to nearby cells, and it decays over
        time when fire weakens.
        """
        st = self.state
        prev = st.airflow_potential
        next_field = np.zeros_like(prev)

        for r in range(self.size):
            for c in range(self.size):
                neighbor_sum = 0.0
                neighbor_count = 0
                for dr, dc, _ in NEIGHBORS_8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.size and 0 <= nc < self.size:
                        neighbor_sum += prev[nr, nc]
                        neighbor_count += 1

                neighbor_mean = (
                    neighbor_sum / neighbor_count if neighbor_count else prev[r, c]
                )
                source = st.intensity[r, c] if st.cell_state[r, c] == STATE_BURNING else 0.0

                next_field[r, c] = (
                    AIRFLOW_POTENTIAL_DECAY * prev[r, c]
                    + AIRFLOW_POTENTIAL_DIFFUSION * (neighbor_mean - prev[r, c])
                    + AIRFLOW_POTENTIAL_SOURCE_GAIN * source
                )

        st.airflow_potential = np.clip(next_field, 0.0, 10.0)

    def _airflow_gradient(self, r: int, c: int) -> tuple[float, float]:
        """
        Compute the local gradient of the airflow potential field.

        Returns:
            (eastward_gradient, southward_gradient)
        """
        field = self.state.airflow_potential

        west = field[r, c - 1] if c > 0 else field[r, c]
        east = field[r, c + 1] if c < self.size - 1 else field[r, c]
        north = field[r - 1, c] if r > 0 else field[r, c]
        south = field[r + 1, c] if r < self.size - 1 else field[r, c]

        grad_x = (east - west) * 0.5
        grad_y = (south - north) * 0.5
        return grad_x, grad_y

    def _compute_local_wind(self, r: int, c: int) -> tuple:
        """
        Compute local wind speed and direction at cell (r, c).
        
        Modifies ambient wind based on:
        - the ambient wind vector
        - gradients in the fire-driven airflow potential field
        
        Returns (local_wind_speed, local_wind_direction_rad).
        """
        st = self.state

        local_speed = st.wind_speed
        local_dir = math.radians(st.wind_direction)

        ambient_vx = local_speed * math.sin(local_dir)
        ambient_vy = -local_speed * math.cos(local_dir)

        grad_x, grad_y = self._airflow_gradient(r, c)
        perturb_vx = AIRFLOW_WIND_SCALE * grad_x
        perturb_vy = AIRFLOW_WIND_SCALE * grad_y

        total_vx = ambient_vx + perturb_vx
        total_vy = ambient_vy + perturb_vy

        local_speed = math.sqrt(total_vx ** 2 + total_vy ** 2)
        local_dir = math.atan2(total_vx, -total_vy)

        local_speed = max(0.0, min(40.0, local_speed))
        return local_speed, local_dir

    def _radiate_heat_and_ignite(self):
        """
        For each unburned/structure cell, accumulate heat from nearby
        burning cells and check for ignition.
        
        Heat transfer depends on:
        - Source intensity
        - Target fuel absorption rate
        - Target moisture (dry absorbs more)
        - Wind (pushes heat downwind)
        - Slope (uphill gets more heat)  
        - Distance (inverse for ortho/diag)
        
        When accumulated heat >= ignition threshold → cell ignites.
        """
        st = self.state
        t = self.terrain

        # Pre-identify burning cells for efficiency
        burning_cells = []
        for r in range(self.size):
            for c in range(self.size):
                if st.cell_state[r, c] == STATE_BURNING:
                    burning_cells.append((r, c))

        if not burning_cells:
            return

        # For each potentially ignitable cell
        new_ignitions = []

        for r in range(self.size):
            for c in range(self.size):
                cs = st.cell_state[r, c]
                if cs not in (STATE_UNBURNED, STATE_STRUCTURE, STATE_SUPPRESSED):
                    continue

                fuel = t.fuel_type[r, c]
                if fuel == FUEL_NONE:
                    continue

                moisture_extinction = FUEL_MOISTURE_EXTINCTION.get(
                    fuel, FUEL_MOISTURE_EXTINCTION[FUEL_FOREST]
                )
                moisture_damping = _rothermel_moisture_damping(
                    st.fuel_moisture[r, c], moisture_extinction
                )
                if moisture_damping <= 0.0:
                    continue

                # Compute local wind at this cell
                local_wind_speed, local_wind_dir = self._compute_local_wind(r, c)
                local_wind_speed_mps = local_wind_speed / 3.6

                # Accumulate heat from burning neighbors
                heat_received = 0.0

                fuel_spread_factor = max(0.05, 1.0 + FUEL_SPREAD_MOD.get(fuel, 0.0))

                for idx, (dr, dc, dist) in enumerate(NEIGHBORS_8):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self.size and 0 <= nc < self.size):
                        continue
                    if st.cell_state[nr, nc] != STATE_BURNING:
                        continue

                    source_intensity = st.intensity[nr, nc]
                    if source_intensity <= 0:
                        continue

                    # Direction from source to target
                    # (dr, dc) points from source neighbor (nr,nc) to target (r,c)
                    # We need direction from source to target
                    spread_angle = NEIGHBOR_ANGLES[idx]

                    # Wind alignment: cos(angle between wind dir and spread dir)
                    wind_alignment = math.cos(local_wind_dir - spread_angle)
                    wind_factor = math.exp(
                        local_wind_speed_mps
                        * (
                            C_WIND
                            + C_WIND_DIRECTIONAL * (wind_alignment - 1.0)
                        )
                    )

                    # Slope factor from geometric slope angle.
                    elev_diff_m = (
                        float(t.elevation[r, c] - t.elevation[nr, nc])
                        * ELEVATION_LEVEL_METERS
                    )
                    slope_angle = math.atan2(
                        elev_diff_m,
                        CELL_SIZE_METERS * dist,
                    )
                    slope_factor = math.exp(C_SLOPE * slope_angle)

                    # Fuel absorption
                    fuel_absorb = FUEL_HEAT_ABSORB.get(fuel, 1.0)

                    # Total heat from this neighbor
                    h = (
                        P0
                        * fuel_spread_factor
                        * source_intensity
                        * fuel_absorb
                        * moisture_damping
                        * wind_factor
                        * slope_factor
                        / dist
                    )

                    heat_received += h

                # Also check 2-cell distant burning cells (reduced effect)
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if abs(dr) <= 1 and abs(dc) <= 1:
                            continue  # already counted
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < self.size and 0 <= nc < self.size):
                            continue
                        if st.cell_state[nr, nc] != STATE_BURNING:
                            continue
                        d = math.sqrt(dr * dr + dc * dc)
                        heat_received += (
                            st.intensity[nr, nc] * 0.15 / d
                        )

                # Accumulate
                st.heat[r, c] += heat_received

                # Ignition check
                threshold = IGNITION_THRESHOLD

                # Long-term retardant bonus (permanent threshold increase)
                if st.retardant_bonus is not None:
                    threshold += st.retardant_bonus[r, c]

                # Defensible space for structures
                if cs == STATE_STRUCTURE:
                    defended_count = 0
                    for dr2, dc2, _ in NEIGHBORS_8:
                        nr2, nc2 = r + dr2, c + dc2
                        if 0 <= nr2 < self.size and 0 <= nc2 < self.size:
                            ns = st.cell_state[nr2, nc2]
                            if ns in (STATE_FIREBREAK, STATE_WATER,
                                      STATE_SUPPRESSED):
                                defended_count += 1
                    # Increase threshold per defended neighbor
                    threshold += defended_count * 0.05

                # Suppressed cells are harder to re-ignite
                if cs == STATE_SUPPRESSED:
                    threshold = 0.8

                if st.heat[r, c] >= threshold:
                    new_ignitions.append((r, c))

        # Apply ignitions
        for r, c in new_ignitions:
            if st.cell_state[r, c] in (STATE_UNBURNED, STATE_STRUCTURE,
                                        STATE_SUPPRESSED):
                st.cell_state[r, c] = STATE_BURNING
                st.burn_timer[r, c] = 0
                st.heat[r, c] = 0.0

    def _check_spotting(self):
        """
        Check for ember spotting (long-range ignition).
        
        Spotting emerges from:
        - available ember source intensity
        - ambient wind transport
        - target fuel receptivity

        This intentionally does not target special assets such as structures.
        """
        st = self.state
        t = self.terrain

        # Compute total fire intensity
        total_intensity = float(st.intensity[st.cell_state == STATE_BURNING].sum())

        if st.wind_speed < SPOT_WIND_THRESHOLD:
            return
        if total_intensity < SPOT_INTENSITY_THRESHOLD:
            return

        # Spotting probability
        spot_prob = (
            (st.wind_speed / 50.0)
            * (total_intensity / 15.0)
            * 0.08
        )
        spot_prob = min(spot_prob, SPOT_MAX_PROB)

        if self._rng.random() >= spot_prob:
            return

        burning_positions = np.argwhere(st.cell_state == STATE_BURNING)
        if len(burning_positions) == 0:
            return

        # Pick a source weighted by ember-producing intensity.
        source_weights = np.array(
            [st.intensity[r, c] for r, c in burning_positions],
            dtype=np.float64,
        )
        weight_sum = float(source_weights.sum())
        if weight_sum <= 0.0:
            return
        source_weights /= weight_sum

        src_idx = int(self._rng.choice(len(burning_positions), p=source_weights))
        src_r, src_c = burning_positions[src_idx]
        source_intensity = float(st.intensity[src_r, src_c])
        wind_mps = st.wind_speed / 3.6

        # Ember transport distance grows with ambient wind and source intensity.
        # This is a reduced-order transport rule, not a target-specific event.
        expected_distance = 1.5 + 0.35 * wind_mps + 1.1 * source_intensity
        distance = int(round(expected_distance + self._rng.normal(0.0, 0.8)))
        distance = max(2, min(SPOT_DISTANCE_CAP, distance))
        wind_rad = math.radians(st.wind_direction)

        tr = int(src_r + distance * math.cos(wind_rad))
        tc = int(src_c + distance * math.sin(wind_rad))

        # Crosswind spread increases with travel distance.
        lateral_sigma = max(0.4, 0.15 * distance)
        tr += int(round(self._rng.normal(0.0, lateral_sigma)))
        tc += int(round(self._rng.normal(0.0, lateral_sigma)))

        if 0 <= tr < self.size and 0 <= tc < self.size:
            cs = st.cell_state[tr, tc]
            if cs in (STATE_UNBURNED, STATE_STRUCTURE):
                target_fuel = t.fuel_type[tr, tc]
                moisture_extinction = FUEL_MOISTURE_EXTINCTION.get(
                    target_fuel, FUEL_MOISTURE_EXTINCTION[FUEL_FOREST]
                )
                receptivity = _rothermel_moisture_damping(
                    st.fuel_moisture[tr, tc], moisture_extinction
                )
                if receptivity > 0.0:
                    spot_heat = (0.18 + 0.12 * source_intensity) * receptivity
                    st.heat[tr, tc] += spot_heat

    def _apply_ignitions(self):
        """Apply scheduled ignition points for this step."""
        st = self.state
        t = self.terrain

        for ig in t.ignition_points:
            if ig["step"] == st.step:
                r, c = ig["row"], ig["col"]
                if 0 <= r < self.size and 0 <= c < self.size:
                    if st.cell_state[r, c] in (STATE_UNBURNED, STATE_STRUCTURE):
                        st.cell_state[r, c] = STATE_BURNING
                        st.burn_timer[r, c] = 0
                        st.intensity[r, c] = 0.3  # initial flame
                        st.heat[r, c] = 0.0

    def _update_statistics(self):
        """Update aggregate statistics."""
        st = self.state

        st.total_burning = int(np.sum(st.cell_state == STATE_BURNING))
        st.total_burned = int(np.sum(st.cell_state == STATE_BURNED))

        # Count structures
        st.structures_lost = 0
        st.structures_saved = 0
        for s in self.terrain.structures:
            r, c = s["row"], s["col"]
            cs = st.cell_state[r, c]
            if cs in (STATE_BURNED, STATE_BURNING):
                st.structures_lost += 1
            elif cs in (STATE_STRUCTURE, STATE_SUPPRESSED, STATE_UNBURNED):
                st.structures_saved += 1

    # ──────────────────────────────────────────────────
    # Resource actions
    # ──────────────────────────────────────────────────

    def _in_bounds(self, row: int, col: int) -> bool:
        """Return whether the coordinates fall inside the grid."""
        return 0 <= row < self.size and 0 <= col < self.size

    def apply_crew(self, row: int, col: int) -> tuple[bool, str]:
        """
        Apply a ground-crew action around a target cell.

        Crews provide local suppression and pre-wetting:
        - reduce heat and intensity for nearby burning cells
        - harden nearby unburned cells by adding moisture
        """
        if self.state is None:
            return False, "simulation not initialized"
        if not self._in_bounds(row, col):
            return False, "crew target is out of bounds"

        st = self.state
        affected = 0
        extinguished = 0
        hardened = 0

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = row + dr, col + dc
                if not self._in_bounds(nr, nc):
                    continue

                cs = st.cell_state[nr, nc]
                if cs in (STATE_WATER, STATE_FIREBREAK, STATE_BURNED):
                    continue

                affected += 1
                if cs == STATE_BURNING:
                    st.heat[nr, nc] *= 0.35
                    st.intensity[nr, nc] *= 0.45
                    if st.intensity[nr, nc] < 0.45:
                        st.cell_state[nr, nc] = STATE_SUPPRESSED
                        st.intensity[nr, nc] = 0.0
                        st.heat[nr, nc] = 0.0
                        extinguished += 1
                        st.cells_suppressed_this_step += 1
                else:
                    # Pre-wetting / hardening: crew treats non-burning cells to
                    # raise their ignition resistance.  SUPPRESSED cells have a
                    # higher re-ignition threshold (0.8 vs normal), giving the
                    # crew's preventive work lasting value even after moisture
                    # evaporates.
                    st.fuel_moisture[nr, nc] = min(1.0, st.fuel_moisture[nr, nc] + 0.10)
                    st.heat[nr, nc] *= 0.60
                    if cs == STATE_UNBURNED:
                        st.cell_state[nr, nc] = STATE_SUPPRESSED
                    hardened += 1

        if affected == 0:
            return False, "crew action had no reachable effect"

        return True, (
            f"crew treated {affected} cells, suppressed {extinguished} burning cells, "
            f"hardened {hardened} cells"
        )

    def apply_bomber(self, row: int, col: int) -> tuple[bool, str]:
        """
        Apply an aerial water drop over a small radius.

        Bombers are stronger than crews and mainly work by rapid cooling and
        heavy moisture addition over a wider area.
        """
        if self.state is None:
            return False, "simulation not initialized"
        if not self._in_bounds(row, col):
            return False, "water-drop target is out of bounds"

        st = self.state
        affected = 0
        extinguished = 0

        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = row + dr, col + dc
                if not self._in_bounds(nr, nc):
                    continue
                if dr * dr + dc * dc > 4:
                    continue

                cs = st.cell_state[nr, nc]
                if cs in (STATE_WATER, STATE_FIREBREAK, STATE_BURNED):
                    continue

                affected += 1
                st.fuel_moisture[nr, nc] = min(1.0, st.fuel_moisture[nr, nc] + 0.18)
                st.heat[nr, nc] *= 0.20

                if cs == STATE_BURNING:
                    st.intensity[nr, nc] *= 0.30
                    if st.intensity[nr, nc] < 0.75:
                        st.cell_state[nr, nc] = STATE_SUPPRESSED
                        st.intensity[nr, nc] = 0.0
                        st.heat[nr, nc] = 0.0
                        extinguished += 1
                        st.cells_suppressed_this_step += 1
                elif cs == STATE_UNBURNED:
                    st.cell_state[nr, nc] = STATE_SUPPRESSED

        if affected == 0:
            return False, "water drop had no reachable effect"

        return True, (
            f"water drop treated {affected} cells and suppressed {extinguished} burning cells"
        )

    def apply_dozer_segment(self, row: int, col: int) -> tuple[bool, str]:
        """Build a single firebreak cell for time-staggered dozer work."""
        if self.state is None:
            return False, "simulation not initialized"
        if not self._in_bounds(row, col):
            return False, "firebreak segment is out of bounds"

        st = self.state
        cs = st.cell_state[row, col]
        if cs in (STATE_WATER, STATE_STRUCTURE, STATE_BURNING, STATE_BURNED):
            return False, "firebreak segment blocked by current cell state"

        st.cell_state[row, col] = STATE_FIREBREAK
        st.heat[row, col] = 0.0
        st.intensity[row, col] = 0.0
        return True, "firebreak segment completed"

    def apply_dozer(self, row: int, col: int, orientation: str) -> tuple[bool, str]:
        """
        Cut a short firebreak line centered on a target cell.

        Orientation options:
        - horizontal
        - vertical
        - diag_down
        - diag_up
        """
        if self.state is None:
            return False, "simulation not initialized"
        if not self._in_bounds(row, col):
            return False, "firebreak target is out of bounds"

        direction_map = {
            "horizontal": (0, 1),
            "vertical": (1, 0),
            "diag_down": (1, 1),
            "diag_up": (-1, 1),
        }
        if orientation not in direction_map:
            return False, "invalid firebreak orientation"

        dr, dc = direction_map[orientation]
        st = self.state
        built = 0
        blocked = 0

        for offset in (-1, 0, 1):
            nr = row + offset * dr
            nc = col + offset * dc
            if not self._in_bounds(nr, nc):
                continue
            success, _ = self.apply_dozer_segment(nr, nc)
            if success:
                built += 1
            else:
                blocked += 1

        if built == 0:
            return False, "dozer could not cut a firebreak at the target"

        return True, f"dozer built {built} firebreak cells (blocked on {blocked})"

    def apply_engine(self, row: int, col: int) -> tuple[bool, str]:
        """
        Apply engine pump-and-roll action.

        Engines are faster than hand crews and carry water/foam, providing
        a mobile suppression option. They treat a 3×3 area like crews but
        with stronger moisture boost (Class A foam effect per NWCG S-420)
        and faster knockdown.

        Effect calibration:
        - Foam reduces surface tension → better wetting → moisture +0.14
          (vs crew +0.10). Based on Class A foam performance data from
          Perimeter Solutions / NWCG foam application guidelines.
        - Heat reduction 0.30× (vs crew 0.35×) — pressurized hose is more
          efficient than hand tools.
        - Intensity knockdown to 0.40× (vs crew 0.45×).
        """
        if self.state is None:
            return False, "simulation not initialized"
        if not self._in_bounds(row, col):
            return False, "engine target is out of bounds"

        st = self.state
        affected = 0
        extinguished = 0
        hardened = 0

        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = row + dr, col + dc
                if not self._in_bounds(nr, nc):
                    continue

                cs = st.cell_state[nr, nc]
                if cs in (STATE_WATER, STATE_FIREBREAK, STATE_BURNED):
                    continue

                affected += 1
                if cs == STATE_BURNING:
                    st.heat[nr, nc] *= 0.30
                    st.intensity[nr, nc] *= 0.40
                    if st.intensity[nr, nc] < 0.40:
                        st.cell_state[nr, nc] = STATE_SUPPRESSED
                        st.intensity[nr, nc] = 0.0
                        st.heat[nr, nc] = 0.0
                        extinguished += 1
                        st.cells_suppressed_this_step += 1
                else:
                    # Class A foam provides better wetting than hand tools
                    st.fuel_moisture[nr, nc] = min(
                        1.0, st.fuel_moisture[nr, nc] + 0.14
                    )
                    st.heat[nr, nc] *= 0.50
                    if cs == STATE_UNBURNED:
                        st.cell_state[nr, nc] = STATE_SUPPRESSED
                    hardened += 1

        if affected == 0:
            return False, "engine action had no reachable effect"

        return True, (
            f"engine treated {affected} cells, suppressed {extinguished} "
            f"burning cells, hardened {hardened} cells"
        )

    def apply_helicopter(self, row: int, col: int) -> tuple[bool, str]:
        """
        Apply a helicopter water drop.

        Helicopters use Bambi Buckets (300–2600 gal per NWCG/SEI) for
        precision drops over a smaller radius than air tankers but with
        faster turnaround when a water source is nearby.

        Effect calibration (AFUE study, USDA 2018):
        - AFUE found helicopter drops' primary role is "reducing fire
          intensity" rather than halting spread outright.
        - Radius-1 area (smaller than air tanker radius-2) reflects
          Type 2-3 helicopter bucket volumes (~300-700 gal).
        - Moisture +0.15, intensity ×0.35 — less powerful than the old
          "bomber" (which mapped to a LAT-class air tanker).
        """
        if self.state is None:
            return False, "simulation not initialized"
        if not self._in_bounds(row, col):
            return False, "helicopter drop target is out of bounds"

        st = self.state
        affected = 0
        extinguished = 0

        # Radius-1 precision drop (helicopter bucket is smaller than tanker)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = row + dr, col + dc
                if not self._in_bounds(nr, nc):
                    continue

                cs = st.cell_state[nr, nc]
                if cs in (STATE_WATER, STATE_FIREBREAK, STATE_BURNED):
                    continue

                affected += 1
                st.fuel_moisture[nr, nc] = min(
                    1.0, st.fuel_moisture[nr, nc] + 0.15
                )
                st.heat[nr, nc] *= 0.25

                if cs == STATE_BURNING:
                    st.intensity[nr, nc] *= 0.35
                    if st.intensity[nr, nc] < 0.55:
                        st.cell_state[nr, nc] = STATE_SUPPRESSED
                        st.intensity[nr, nc] = 0.0
                        st.heat[nr, nc] = 0.0
                        extinguished += 1
                        st.cells_suppressed_this_step += 1
                elif cs == STATE_UNBURNED:
                    st.cell_state[nr, nc] = STATE_SUPPRESSED

        if affected == 0:
            return False, "helicopter drop had no reachable effect"

        return True, (
            f"helicopter drop treated {affected} cells and suppressed "
            f"{extinguished} burning cells"
        )

    def apply_airtanker(self, row: int, col: int) -> tuple[bool, str]:
        """
        Apply a fixed-wing air tanker retardant drop.

        Air tankers drop long-term retardant (ammonium phosphate, e.g.
        Phos-Chek LC-95) that provides PERMANENT fire resistance.

        Key difference from water drops:
        - Retardant chemically alters fuel combustion (Perimeter Solutions
          technical docs). Effective for weeks until washed by rain.
        - AFUE study: retardant-protected land takes 4-5 hours longer to
          burn than unprotected areas.

        Effect calibration:
        - Radius-2 area (large tanker coverage — LAT covers ~1/4 mile line).
        - Moisture boost is modest (+0.10) since retardant works via
          chemistry, not wetting.
        - PERMANENT ignition threshold increase: +0.20 per treated cell.
          This models the long-term retardant persistence.
        - Intensity reduction strong (×0.25) for any currently burning cells.
        """
        if self.state is None:
            return False, "simulation not initialized"
        if not self._in_bounds(row, col):
            return False, "retardant drop target is out of bounds"

        st = self.state
        affected = 0
        extinguished = 0

        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = row + dr, col + dc
                if not self._in_bounds(nr, nc):
                    continue
                if dr * dr + dc * dc > 4:
                    continue

                cs = st.cell_state[nr, nc]
                if cs in (STATE_WATER, STATE_FIREBREAK, STATE_BURNED):
                    continue

                affected += 1
                # Retardant provides modest moisture but permanent threshold
                st.fuel_moisture[nr, nc] = min(
                    1.0, st.fuel_moisture[nr, nc] + 0.10
                )
                st.heat[nr, nc] *= 0.20

                # Permanent ignition-threshold increase — makes retardant-treated
                # cells harder to ignite even after moisture evaporates.
                if st.retardant_bonus is None:
                    st.retardant_bonus = np.zeros(
                        (self.size, self.size), dtype=np.float64
                    )
                st.retardant_bonus[nr, nc] = min(
                    0.40, st.retardant_bonus[nr, nc] + 0.20
                )

                if cs == STATE_BURNING:
                    st.intensity[nr, nc] *= 0.25
                    if st.intensity[nr, nc] < 0.65:
                        st.cell_state[nr, nc] = STATE_SUPPRESSED
                        st.intensity[nr, nc] = 0.0
                        st.heat[nr, nc] = 0.0
                        extinguished += 1
                        st.cells_suppressed_this_step += 1
                elif cs == STATE_UNBURNED:
                    st.cell_state[nr, nc] = STATE_SUPPRESSED

        if affected == 0:
            return False, "retardant drop had no reachable effect"

        return True, (
            f"retardant drop treated {affected} cells and suppressed "
            f"{extinguished} burning cells (permanent threshold increase applied)"
        )

    def apply_wetline(self, row: int, col: int) -> tuple[bool, str]:
        """
        Apply a wet-line segment via engine foam spray.

        Wet lines are temporary fire barriers created by spraying water/foam
        on vegetation. Faster to create than dozer firebreaks but degrade
        as moisture evaporates (per NWCG S-290 fire behavior curriculum).

        Effect:
        - Sets cell moisture to 0.90 (near-saturated)
        - Does NOT change cell state to FIREBREAK (temporary, not permanent)
        - Moisture will naturally decay over ~3-5 steps via the simulation's
          EMC drift mechanics, making this a time-limited barrier.
        """
        if self.state is None:
            return False, "simulation not initialized"
        if not self._in_bounds(row, col):
            return False, "wet-line segment is out of bounds"

        st = self.state
        cs = st.cell_state[row, col]
        if cs in (STATE_WATER, STATE_BURNING, STATE_BURNED):
            return False, "wet-line blocked by current cell state"

        # Saturate with water/foam — temporary but very effective barrier
        st.fuel_moisture[row, col] = min(1.0, 0.90)
        st.heat[row, col] = 0.0
        if cs == STATE_UNBURNED:
            st.cell_state[row, col] = STATE_SUPPRESSED
        return True, "wet-line segment applied"

    def apply_backfire(self, row: int, col: int) -> tuple[bool, str]:
        """
        Deliberately ignite a cell as part of a backfire operation.

        Backfiring is the highest-stakes tactic in wildfire suppression:
        intentionally setting fire to consume fuel before the main fire
        arrives (NWCG S-290, Fireline Handbook PMS 410-1).

        The cell is ignited at low initial intensity (0.2). From there,
        normal fire physics governs its spread — wind, fuel, slope all
        apply. If the backfire burns toward the main fire, it creates a
        fuel-free zone. If wind shifts, it can escape and worsen the
        incident.

        Validation requires anchor-point adjacency — enforced by the
        environment wrapper, not here.
        """
        if self.state is None:
            return False, "simulation not initialized"
        if not self._in_bounds(row, col):
            return False, "backfire target is out of bounds"

        st = self.state
        cs = st.cell_state[row, col]

        # Can only backfire unburned or suppressed cells with fuel
        if cs not in (STATE_UNBURNED, STATE_SUPPRESSED):
            return False, f"cannot ignite cell in state {cs}"

        fuel = self.terrain.fuel_type[row, col]
        if fuel == FUEL_NONE:
            return False, "no fuel at backfire target"

        # Intentional ignition
        st.cell_state[row, col] = STATE_BURNING
        st.burn_timer[row, col] = 0
        st.intensity[row, col] = 0.2  # starts low, physics ramps it up
        st.heat[row, col] = 0.0

        return True, f"backfire ignited at ({row}, {col})"

    # ──────────────────────────────────────────────────
    # Public utility methods
    # ──────────────────────────────────────────────────


    def get_grid_string(self) -> str:
        """
        Render the grid as a human-readable string.
        
        Uses symbols:
        . = unburned grass/brush/forest
        F = burning
        B = burned out
        X = firebreak
        W = water
        S = structure
        ~ = suppressed
        """
        if self.state is None:
            return ""

        symbols = {
            STATE_UNBURNED: ".",
            STATE_BURNING: "F",
            STATE_BURNED: "B",
            STATE_FIREBREAK: "X",
            STATE_WATER: "W",
            STATE_STRUCTURE: "S",
            STATE_SUPPRESSED: "~",
        }

        lines = []
        # Header
        header = "   " + " ".join(f"{c:X}" for c in range(self.size))
        lines.append(header)

        for r in range(self.size):
            row_str = f"{r:2d} "
            for c in range(self.size):
                cs = self.state.cell_state[r, c]
                row_str += symbols.get(cs, "?") + " "
            lines.append(row_str)

        return "\n".join(lines)

    def get_observation_dict(self) -> dict:
        """
        Get the full observation as a dictionary for the agent.
        
        Includes all information the agent needs to make decisions.
        """
        if self.state is None:
            return {}

        st = self.state
        t = self.terrain

        # Fire positions
        fire_cells = []
        for r in range(self.size):
            for c in range(self.size):
                if st.cell_state[r, c] == STATE_BURNING:
                    fire_cells.append({
                        "row": int(r), "col": int(c),
                        "intensity": round(float(st.intensity[r, c]), 2),
                        "timer": int(st.burn_timer[r, c]),
                    })

        # Structure status
        struct_status = []
        for s in t.structures:
            r, c = s["row"], s["col"]
            struct_status.append({
                "structure_id": s.get("structure_id", f"structure_{len(struct_status) + 1}"),
                "row": r, "col": c,
                "priority": s["priority"],
                "status": STATE_NAMES.get(st.cell_state[r, c], "unknown"),
            })

        # Heat map for cells approaching ignition
        heat_warnings = []
        for r in range(self.size):
            for c in range(self.size):
                if st.heat[r, c] > IGNITION_THRESHOLD * 0.5:
                    if st.cell_state[r, c] in (STATE_UNBURNED, STATE_STRUCTURE):
                        heat_warnings.append({
                            "row": int(r), "col": int(c),
                            "heat": round(float(st.heat[r, c]), 2),
                            "threshold": round(IGNITION_THRESHOLD, 2),
                        })

        return {
            "grid": self.get_grid_string(),
            "step": st.step,
            "max_steps": self.config.max_steps,
            "step_minutes": SIMULATION_STEP_MINUTES,
            "elapsed_minutes": round(st.step * SIMULATION_STEP_MINUTES, 1),
            "time_of_day": round(st.time_of_day, 2),

            # Atmosphere
            "wind_speed": round(st.wind_speed, 1),
            "wind_direction": round(st.wind_direction, 1),
            "temperature": round(st.temperature, 1),
            "humidity": round(st.humidity, 2),
            "atmospheric_dryness_index": round(st.temperature - st.humidity * 100.0, 1),
            "airflow_potential_peak": round(float(st.airflow_potential.max()), 2),

            # Fire status
            "burning_cells": len(fire_cells),
            "burned_cells": st.total_burned,
            "fire_details": fire_cells,

            # Structures
            "structures": struct_status,
            "structures_remaining": st.structures_saved,
            "structures_lost": st.structures_lost,

            # Heat warnings (approaching ignition)
            "heat_warnings": heat_warnings,

            # Terrain info (static, but useful for agent)
            "elevation": t.elevation.tolist(),
            "fuel_types": t.fuel_type.tolist(),
        }

    def compute_environmental_rewards(self) -> dict:
        """
        Compute per-step reward signals from environment state.

        Design principles:
        - Reward TRANSITIONS (events the agent caused), not uncontrollable states.
        - Weather is information, not punishment — the agent should learn that
          bad weather means fire spreads faster, not that it gets penalized for
          weather it cannot change.
        - One-time events (structure lost, fire extinguished) fire once, not
          every subsequent step.
        - Continuous signals (structure safe, area preserved) provide shaping
          throughout the episode.
        """
        if self.state is None:
            return {}

        st = self.state
        rewards = {}

        # ── Structure damage ──
        # Burning: per-step urgency signal — act NOW to save the structure.
        # Lost: one-time penalty when a structure transitions to BURNED.
        for s in self.terrain.structures:
            r, c = s["row"], s["col"]
            if st.cell_state[r, c] == STATE_BURNING:
                rewards[f"structure_burning_{r}_{c}"] = -0.20 * s["priority"]
            elif st.cell_state[r, c] == STATE_BURNED:
                if (r, c) not in self._structures_lost_penalized:
                    rewards[f"structure_lost_{r}_{c}"] = -0.50 * s["priority"]
                    self._structures_lost_penalized.add((r, c))

        # ── Structure safe ── continuous positive signal for each intact structure.
        for s in self.terrain.structures:
            r, c = s["row"], s["col"]
            if st.cell_state[r, c] in (STATE_STRUCTURE, STATE_SUPPRESSED):
                rewards[f"structure_safe_{r}_{c}"] = 0.01 * s["priority"]

        # ── Cells suppressed ── reward each BURNING → SUPPRESSED transition
        # since the last reward computation.  The agent cannot game this by
        # re-burning cells because it does not control ignition — fire
        # spreads via physics.
        if st.cells_suppressed_this_step > 0:
            rewards["cells_suppressed"] = 0.03 * st.cells_suppressed_this_step
            st.cells_suppressed_this_step = 0

        # ── Fire extinguished ── one-time bonus when all fires are out.
        if st.total_burning == 0 and st.step > 0 and not self._fire_extinguished_rewarded:
            rewards["fire_extinguished"] = 0.50
            self._fire_extinguished_rewarded = True

        # ── Area preserved ── continuous shaping proportional to unburned area.
        total_burnable = int(np.sum(
            (self.terrain.fuel_type != FUEL_NONE) & ~self.terrain.is_water
        ))
        if total_burnable > 0:
            preserved_ratio = 1.0 - (st.total_burned + st.total_burning) / total_burnable
            rewards["area_preserved"] = preserved_ratio * 0.02

        return rewards

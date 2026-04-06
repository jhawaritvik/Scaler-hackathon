# Wildfire Environment Dynamics Spec

This document defines the environment-side dynamics for the wildfire simulator:
what state exists, how the world updates each step, and which parameters affect
which outcomes.

It is intended to make the environment mechanics explicit before we finish the
agent-facing resource action layer.

For source validation, coefficient provenance, and the latest calibrated
numbers, treat [FIRE_FACTOR_VALIDATION.md](./FIRE_FACTOR_VALIDATION.md) as the
canonical companion document.

For the target simulator philosophy, especially what should count as a base
variable versus a phenomenon-specific shortcut, treat
[EMERGENCE_FIRST_MODELING_SPEC.md](./EMERGENCE_FIRST_MODELING_SPEC.md) as the
design standard.

Important:

- this document describes the current world-update mechanics
- it does not mean every current mechanic belongs in the final emergence-first
  simulator
- some currently documented heuristics are temporary and are candidates for
  removal or replacement

## Scope

This spec covers the world simulation itself:

- terrain generation
- atmospheric evolution
- fuel moisture dynamics
- ignition and spread
- structure risk
- step rewards already available from the simulator

This spec does not yet claim that the OpenEnv wrapper is complete. It is the
environment-core contract we will wire into `reset()` / `step()` / `state()`.

## State Variables

Important:

- this section lists all relevant fields currently present in the simulator
- not all of them should be treated as core modeled parameters
- for design review, use the following buckets:
  - `essential state`
  - `derived quantities`
  - `setup-only inputs`
  - `bookkeeping`

### Essential State

These are the evolving variables that matter most to the core simulator:

- `elevation`
- `fuel_type`
- `structures`
- `cell_state`
- `fuel_moisture`
- `burn_timer`
- `intensity`
- `heat`
- `wind_speed`
- `wind_direction`
- `temperature`
- `humidity`
- `time_of_day`
- `airflow_potential`

### Derived Quantities

These are computed from the essential state and should not be treated as
independent base parameters:

- `aspect`
- slope angle
- local RH
- equilibrium moisture content
- ignition damping
- atmospheric dryness index
- ember transport distance

### Setup-Only Inputs

These shape the scenario but are not ongoing environment physics:

- `seed`
- `ignition_points`
- `initial_fuel_moisture`
- initial weather values
- resource inventory
- `max_steps`

### Bookkeeping

These are counters or episode outputs:

- `step`
- `done`
- `total_burning`
- `total_burned`
- `structures_lost`
- `structures_saved`

### Static Terrain State

Defined in [terrain.py](../wildfire_env/server/terrain.py):

- `elevation`
- `fuel_type`
- `aspect`
- `is_water`
- `structures`
- `ignition_points`
- `initial_fuel_moisture`
- task config including:
  - `seed`
  - `resources`
  - `max_steps`
  - initial weather

### Dynamic Simulation State

Defined in [fire_simulation.py](../wildfire_env/server/fire_simulation.py):

- `cell_state`
- `fuel_moisture`
- `burn_timer`
- `heat`
- `intensity`
- `wind_speed`
- `wind_direction`
- `temperature`
- `humidity`
- `time_of_day`
- `airflow_potential`
- `step`
- `done`
- `total_burning`
- `total_burned`
- `structures_lost`
- `structures_saved`

## Cell States

- `STATE_UNBURNED = 0`
- `STATE_BURNING = 1`
- `STATE_BURNED = 2`
- `STATE_FIREBREAK = 3`
- `STATE_WATER = 4`
- `STATE_STRUCTURE = 5`
- `STATE_SUPPRESSED = 6`

## Per-Step Update Order

Each world tick currently applies updates in this order:

1. advance time
2. update atmosphere
3. update per-cell fuel moisture
4. advance burning cells and their intensity curves
5. update the fire-driven airflow-potential field
6. radiate heat and ignite susceptible cells
7. apply ember spotting
8. apply scheduled ignition points
9. update aggregate statistics
10. end episode when max steps reached or active fire reaches zero

## Causal Map

This is the main factor graph of the environment.

### Elevation

Elevation affects:

- terrain shape and local slopes
- local temperature through lapse rate
- fuel moisture through cooler/wetter higher ground
- ignition pressure through slope-driven heat transfer

Quantitative effects:

- local temperature shift:
  - `local_temp -= elevation * 0.65`
- initial moisture bonus:
  - `+ 0.06 * elevation / max_elevation`
- slope ignition multiplier:
  - `exp(0.078 * slope_angle)`
  - `slope_angle = atan(elevation_diff_m / distance_m)`
  - current synthetic terrain scale uses `100 m` horizontal cells and `100 m`
    per elevation level

### Aspect

Aspect affects:

- local heating/cooling
- initial fuel moisture
- moisture drying rate

Quantitative effects:

- temperature modifier by aspect:
  - north: `-1.0 C`
  - northeast: `-0.5 C`
  - east: `0.0 C`
  - southeast: `+0.5 C`
  - south: `+1.0 C`
  - southwest: `+0.5 C`
  - west: `0.0 C`
  - northwest: `-0.5 C`
- initial moisture modifier by aspect:
  - north: `+0.04`
  - northeast: `+0.02`
  - east: `0.00`
  - southeast: `-0.02`
  - south: `-0.04`
  - southwest: `-0.02`
  - west: `0.00`
  - northwest: `+0.02`
- drying multiplier by aspect:
  - north: `0.85x`
  - northeast: `0.90x`
  - east: `1.0x`
  - southeast: `1.10x`
  - south: `1.15x`
  - southwest: `1.10x`
  - west: `1.0x`
  - northwest: `0.90x`

### Fuel Type

Fuel type affects:

- spread susceptibility
- burn duration
- heat absorption
- peak intensity
- moisture response rate

Quantitative effects:

- spread factor:
  - none: `0.0` effective
  - grass: `1.0 - 0.3 = 0.7x`
  - brush: `1.0x`
  - forest: `1.0 + 0.4 = 1.4x`
- burn duration:
  - grass: `2`
  - brush: `4`
  - forest: `6`
- heat absorption:
  - grass: `0.8`
  - brush: `1.0`
  - forest: `1.2`
- peak intensity:
  - grass: `0.7`
  - brush: `1.0`
  - forest: `1.3`
- moisture response rate:
  - derived from `alpha = 1 - exp(-dt / tau)`
  - current mapping uses:
    - grass: `1 h`
    - brush: `10 h`
    - forest surface litter / understory: `10 h`
  - with `dt = 20 minutes`, this yields:
    - grass: `~0.283`
    - brush: `~0.033`
    - forest: `~0.033`
- representative moisture of extinction:
  - grass: `0.15`
  - brush: `0.25`
  - forest: `0.30`

### Water

Water affects:

- terrain blocking
- fuel removal
- local humidity support
- local moisture recharge
- defensible space around structures

Quantitative effects:

- water cells are forced to `STATE_WATER`
- adjacent water can raise local humidity by up to `0.15`
  - `0.05` per adjacent water cell, capped at 3 neighbors
- adjacent water recharge during moisture update:
  - `+0.02`
- initial adjacent-water moisture bump:
  - `+0.03`

### Wind

Wind affects:

- directional heat transfer
- spotting probability
- moisture evaporation
- local spread asymmetry
- interaction with the fire-driven airflow-potential field

Quantitative effects:

- wind factor in spread:
  - `exp(V * (0.045 + 0.131 * (cos(theta) - 1)))`
  - `V` is converted to `m/s` before applying the published wind-loading form
- evaporation boost in moisture update:
  - multiplied by `1 + wind_speed / 30`
- spotting only activates when:
  - `wind_speed >= 28.8 km/h` (`8 m/s`)
- ambient wind evolves with:
  - diurnal target forcing
  - persistence from the previous step
  - moderate stochastic drift

### Temperature And Humidity

Temperature and humidity affect:

- equilibrium moisture content
- drying pressure
- atmospheric risk diagnostics

Quantitative effects:

- humidity-temperature coupling:
  - local RH is recomputed from ambient vapor pressure and local saturation
    vapor pressure rather than from a fixed linear coefficient
- EMC approximation:
  - `emc = humidity / 5`
- atmospheric dryness index:
  - `temperature - humidity * 100`
  - used as a continuous diagnostic rather than a behavior switch

### Fire Proximity

Nearby fire affects:

- local temperature
- local humidity through heating
- fuel moisture through radiant drying
- the airflow-potential field through heat-release forcing
- heat accumulation for ignition
- spotting probability through total fire intensity

Quantitative effects:

- local temperature bonus from nearby fire:
  - `15.0 * intensity / (distance * 2.0)` per burning neighbor
- moisture drying from burning neighbors:
  - `0.05 * intensity / distance`
- second-ring drying:
  - reduced to `30%` of the first-ring effect
- airflow-potential source term:
  - proportional to local burning intensity
  - diffuses and decays over time
- local wind perturbation:
  - computed from the gradient of `airflow_potential`
- ignition heat threshold:
  - `0.4` base

### Structures

Structures affect:

- task utility
- fire intensity if they ignite
- defensibility threshold
- reward and grader signals

Quantitative effects:

- structure burning intensity multiplier:
  - `1.5x`
- defended structure ignition threshold bonus:
  - `+0.05` per neighboring defended cell
  - defended cell means `firebreak`, `water`, or `suppressed`

## Ignition Model

A cell can ignite when all of these are true:

- it is in an ignitable state
  - `UNBURNED`
  - `STRUCTURE`
  - `SUPPRESSED`
- it has burnable fuel
- its moisture is below the moisture-of-extinction threshold
  - fuel-class-specific moisture-of-extinction is used
- accumulated heat reaches threshold

Heat received from a burning neighbor is currently proportional to:

`0.58`
`* fuel_spread_factor`
`* source_intensity`
`* fuel_absorption`
`* moisture_damping`
`* wind_factor`
`* slope_factor`
`/ distance`

Where:

- `moisture_damping` uses the Rothermel polynomial
- `fuel_spread_factor` depends on target vegetation

## Burn Intensity Model

Burning cells follow a simple bell curve:

- 0% to 25% of burn duration: ramp up
- 25% to 60%: peak intensity
- 60% to 100%: decay
- after max burn duration: transition to `STATE_BURNED`

## Spotting Model

Spotting is a long-range ignition mechanism.

It only activates when:

- `wind_speed >= 28.8 km/h`
- total active-fire intensity `>= 3`

Spotting probability:

- `min((wind_speed / 50) * (total_intensity / 15) * 0.08, 0.25)`

Transport behavior:

- source burning cells are sampled with probability proportional to intensity
- target distance grows with:
  - ambient wind speed
  - source intensity
- lateral spread grows with transport distance
- target cells receive additional heat based on:
  - ember source intensity
  - target fuel receptivity

## Current Reward Signals

The simulator already computes dense environmental reward components:

- atmospheric dryness: continuous negative signal when `temperature > humidity * 100`
- atmospheric dryness approaching: mild negative signal near the threshold
- moisture crisis: `-0.03`
- wind-dryness risk: `-0.04`
- structure burning: `-0.20 * priority`
- structure lost: `-0.15 * priority`
- structure safe: `+0.01 * priority`
- fire extinguished: `+0.50`
- area preserved: `preserved_ratio * 0.02`

These are good building blocks for the final environment reward, but the
agent-action penalties and task-specific scoring still need to be added.

## Task Presets Already Defined

The environment already has seeded preset configurations for:

- `easy`
- `medium`
- `hard`

They differ in:

- terrain roughness
- water availability
- structure count and priorities
- ignition timing and count
- initial weather severity
- resource inventory
- fuel distribution

## Fleet Dispatch Timing

The environment now treats suppression resources as reusable fleet units rather
than consumable tokens.

Each OpenEnv step now accepts a batch of resource assignments for the incident
interval:

- crews, bombers, and dozers are dispatched from deterministic staging bases
- the agent selects the specific `unit_id` to send
- the agent can issue multiple assignments in the same step
- each assignment can target operational geometry, not just a single cell
- travel time is converted into discrete simulation steps using the incident
  timestep (`20` minutes per step)
- a unit transitions through:
  - `en_route`
  - `operating`
  - `returning`
  - `available`

This matters because the wildfire physics still advance every step. A delayed
arrival means the agent must plan ahead instead of assuming instantaneous
suppression.

### Operational Timing Model

- crews:
  - short dispatch preparation delay
  - cell-distance travel estimate
  - one-step suppression effect on arrival
  - nonzero return-to-service delay
- bombers:
  - fast arrival relative to ground units
  - one-step drop effect on arrival
  - longer return / reload delay before reuse
- dozers:
  - slower dispatch than bombers
  - firebreak line may be built across multiple operating steps
  - return-to-service delay after line construction

### Time Exposure In The Observation

The observation now exposes incident time explicitly:

- `step_minutes`
- `elapsed_minutes`
- `time_of_day`
- active fleet missions with ETAs in steps and minutes

This avoids a hidden-clock problem for the agent and makes resource timing a
first-class planning variable.

The same principle applies to dispatch choice: the environment exposes fleet
state, but does not silently choose the "best" unit on behalf of the agent.

### Assignment Geometry

Assignments currently support these target kinds:

- `point`
- `line`
- `area`
- `polygon`
- `structure`

This keeps the action space closer to incident-command planning than a
single-cell action API, while still remaining grounded in a finite grid.

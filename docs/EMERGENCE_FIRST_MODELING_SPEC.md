# Emergence-First Modeling Spec

This document defines the target modeling philosophy for the wildfire
environment.

The core idea is simple:

- we model primary state variables
- we model direct couplings between those variables
- we do **not** model named wildfire phenomena as explicit rules

If the base variables and couplings are correct, the larger behaviors should
appear in rollout trajectories without us scripting them one by one.

This is the design we want the simulator to converge toward. It is more
important than preserving any existing heuristic if that heuristic violates the
emergence-first principle.

Related docs:

- [ENVIRONMENT_DYNAMICS_SPEC.md](./ENVIRONMENT_DYNAMICS_SPEC.md)
- [FIRE_FACTOR_VALIDATION.md](./FIRE_FACTOR_VALIDATION.md)
- [ENVIRONMENT_IMPLEMENTATION_PLAN.md](./ENVIRONMENT_IMPLEMENTATION_PLAN.md)

## Core Principle

Do this:

- encode direct physical dependencies
- let indirect chains happen through state updates
- use repeated simulation runs to reveal behaviors

Do not do this:

- add special-case rules for named behaviors
- add event flags that directly boost spread because a phenomenon is "active"
- target specific outcomes such as structures or flare-ups with handcrafted
  logic

Good:

- `temperature -> local RH`
- `local RH -> fuel moisture`
- `fuel moisture -> ignition damping`
- `wind vector -> directional spread loading`
- `fire intensity -> local heat release`

Bad:

- `if crossover: multiply spread`
- `if large fire: inject bonus wind`
- `30% chance to target a structure with spotting`

## Why This Matters

Wildfire behavior is full of chained effects:

- heating changes humidity
- humidity changes fuel moisture
- fuel moisture changes ignition resistance
- burning intensity changes local heating
- terrain changes wind exposure and slope loading

Those chains are exactly what we want the environment to discover during
simulation.

If we directly encode the named phenomenon instead, we:

- hide the real mechanism
- make validation harder
- double-count effects
- reduce the chance of genuinely unexpected behaviors appearing

## Base Variables We Want

These are the variables the environment is allowed to model directly.

Important:

- not every field in code is a core modeled parameter
- we distinguish between:
  - `essential state`
  - `derived quantities`
  - `setup-only inputs`
  - `bookkeeping`

This keeps the simulator lean and prevents us from mistaking diagnostics or
task metadata for physical state.

### Static Terrain / Fuel

- `fuel_type`
- `fuel_load` or a coarse proxy if we later need it
- `elevation`
- enough terrain information to derive slope and exposure
- `nonburnable_mask`
- `water_mask`
- `structure_locations`

### Atmospheric State

- `ambient_temperature`
- `ambient_atmospheric_moisture`
- `ambient_wind_vector`
- optional `solar_forcing` or `time_of_day` as a driver of the above

### Fire / Fuel State

- `cell_state`
- `fuel_moisture`
- `cell_heat`
- `fire_intensity`
- `burn_progress`

### Minimum Additional Hidden State If Needed

If we want true fire-atmosphere feedback instead of fake behavior switches, the
most justifiable extra field is:

- `local_airflow_perturbation` or a pressure / buoyancy-like field

Why:

- fire-driven inflow cannot emerge from ambient wind alone
- if we want fire heat to alter local flow, the model needs somewhere to store
  that perturbation
- without a state like this, any fire-driven wind effect becomes a handcrafted
  shortcut

## Lean Taxonomy For This Project

This is the recommended way to talk about the simulator in reviews and docs.

### Essential State

These are the fields that actually define the evolving environment.

- terrain elevation
- surface / fuel class
- structure locations and priorities
- fuel moisture
- fire state
- burn progress
- fire intensity
- ambient wind
- ambient temperature
- ambient humidity
- time-of-day forcing
- optional `airflow_potential`

### Derived Quantities

These are useful, but they are not independent modeled parameters.

- aspect
- slope angle
- local RH
- EMC
- ignition damping
- dryness index
- ember transport distance
- local wind perturbation from `airflow_potential`

### Setup-Only Inputs

These initialize a scenario but are not ongoing physics state.

- seed
- ignition schedule
- initial weather
- initial fuel moisture
- resource inventory
- max steps

### Bookkeeping

These are outputs or counters, not environment physics.

- current step
- done flag
- total burning cells
- total burned cells
- structures lost
- structures saved

## Derived Variables We Are Allowed To Compute

These are not independent modeled parameters. They are derived each step from
the base state.

- local RH
- equilibrium moisture content
- slope angle
- wind alignment with spread direction
- ignition damping
- ember transport direction
- local temperature anomalies
- local drying pressure

These are good because they are intermediate physics quantities, not named
wildfire "events".

## Direct Couplings We Should Encode

Only direct edges should become governing equations in the simulator.

### Terrain -> Microclimate

- elevation -> temperature
- aspect / exposure -> net drying tendency
- terrain shape -> wind exposure or shelter
- water / nonburnable surfaces -> local moisture support or spread blocking

### Atmosphere -> Fuel State

- ambient moisture -> fuel moisture equilibrium
- ambient temperature -> local RH through vapor pressure
- ambient wind -> directional spread loading
- solar forcing / time of day -> ambient temperature and moisture

### Fire State -> Local Environment

- fire intensity -> local heat release
- local heat release -> local temperature anomaly
- local temperature anomaly -> local RH
- local RH + local heat -> fuel moisture change
- fire intensity -> ember availability
- ember availability + wind -> spotting transport potential

### Fuel / Terrain / Weather -> Spread

- fuel type -> spread susceptibility
- fuel moisture -> ignition damping
- slope angle -> spread loading
- wind alignment -> spread loading
- intensity -> outgoing heat flux proxy

## Things We Should Not Encode Explicitly

These may be real wildfire phenomena, but they should not exist as direct rules
in the target simulator.

- crossover as a simulator switch
- gust as a special random event flag unless gusts are part of the ambient wind
  process itself
- structure-targeted spotting
- chimney effect switch
- blow-up behavior switch
- crown-run mode switch
- "large fire attracts wind" as a direct special-case rule

If one of these matters, it should appear because the underlying state and
couplings permit it.

## Current Simulator Patterns That Conflict With This Goal

The original simulator contained several phenomenon-first or shortcut-style
rules:

- explicit `crossover_active` spread multiplier
- explicit `gust_active` event
- structure-targeted spotting probability
- hardcoded valley amplification / ridge blocking multipliers
- direct fire-updraft wind blending without an explicit perturbation field
- fixed water humidity / recharge bonuses

The current implementation has already removed the first five of those and
replaced them with a cleaner structure:

- ambient wind now evolves continuously instead of through gust flags
- spread no longer receives an explicit crossover multiplier
- spotting is now pure ember transport and does not target structures
- valley / ridge wind shortcuts have been removed
- fire-atmosphere feedback now goes through a hidden `airflow_potential` field
  instead of a direct updraft shortcut

The remaining water-moisture coupling is still a reduced-order approximation,
but it is a base-state coupling rather than a named phenomenon rule.

## Interaction Matrix Philosophy

A dense `15 x 15` matrix is useful as documentation, but not as the simulator.

Why:

- many variable pairs have no direct physical interaction
- some pairs only interact through intermediates
- forcing direct coefficients between everything creates false dependencies
- it encourages double-counting the same mechanism

The right use of the matrix is:

- mark each pair as `governing`, `diagnostic`, or `none`
- only `governing` edges become simulator equations
- `diagnostic` relationships are used to validate the simulator output

Example:

- `temperature -> spread rate` is often diagnostic
- `temperature -> local RH` is governing
- `local RH -> fuel moisture` is governing
- `fuel moisture -> spread damping` is governing

That keeps the model sparse and causal.

## What Emergence Means Here

The simulator can produce behaviors we did not explicitly name if:

- the state is sufficient
- the couplings are valid
- the dynamics are nonlinear enough

But emergence still has limits:

- a missing state variable cannot produce its missing mechanism
- a missing coupling cannot appear just by running more episodes

So repeated runs are useful for revealing behavior, but they do not replace
proper model design.

## Design Rule For New Mechanics

Before adding any new wildfire mechanic, ask:

1. Is this a primary state variable?
2. Is this a derived intermediate quantity?
3. Is this a direct physical dependency from literature?
4. Or am I trying to inject a named phenomenon by hand?

If the answer is the fourth one, it should usually not be added.

## Practical Refactor Direction

The target refactor path for the simulator is:

1. keep the validated governing couplings from
   [FIRE_FACTOR_VALIDATION.md](./FIRE_FACTOR_VALIDATION.md)
2. remove explicit phenomenon switches where possible
3. replace shortcut behavior rules with base-state couplings
4. introduce a minimal local airflow / buoyancy perturbation field only if we
   decide fire-atmosphere feedback is worth modeling directly
5. treat named wildfire phenomena only as analysis labels, not as simulator
   logic

Implementation note:

- steps `2`, `3`, and `4` have now started in the live simulator
- the remaining work is to keep tightening the base couplings and to avoid
  reintroducing phenomenon-specific shortcuts while building the action layer

## Decision Record

Current decision:

- the wildfire environment should be built as an emergence-first simulator
- the simulator should prefer sparse causal couplings over dense pairwise
  coefficient tables
- named wildfire phenomena should not be authored directly unless they are
  unavoidable stand-ins for a missing state variable during development

This is the standard we should use going forward when reviewing simulator
changes.

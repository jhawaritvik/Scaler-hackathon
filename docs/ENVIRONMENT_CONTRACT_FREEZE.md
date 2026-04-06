# Environment Contract Freeze

This file records the current wildfire environment contract as the frozen
environment-only milestone.

The goal of this milestone is:

- the environment world model is implemented
- the OpenEnv contract is implemented
- the action and observation schema are stable enough to build against
- agent policy, graders, baseline inference, and training are explicitly out of
  scope for this freeze

## Scope Of This Freeze

Included:

- terrain generation
- wildfire dynamics
- fleet dispatch timing
- resource effects
- reward shaping from environment state
- OpenEnv `reset()`, `step()`, and `state()`
- typed action and observation models

Excluded for now:

- trained policy / controller
- baseline inference script
- task graders
- submission-only scoring endpoints
- model-training logic

## Frozen Action Contract

`WildfireAction` is a batch assignment object:

- `assignments: list[ResourceAssignment]`

Each assignment contains:

- `unit_id`
- `mission_type`
- `target`
- optional `commitment_steps`
- optional `drop_configuration`

Supported mission types:

- `direct_attack`
- `line_construction`
- `wet_line`
- `water_drop`
- `retardant_drop`
- `point_protection`
- `backfire`
- `staging`

Supported target kinds:

- `point`
- `line`
- `area`
- `polygon`
- `structure`

Design intent:

- the environment does not choose the fleet unit on behalf of the controller
- the controller chooses `unit_id`
- the controller can issue multiple assignments in one decision interval
- the environment enforces timing, availability, and physical consequences

## Frozen Observation Contract

`WildfireObservation` currently exposes:

- task and goal:
  - `task_id`
  - `goal`
- incident clock:
  - `step`
  - `max_steps`
  - `step_minutes`
  - `elapsed_minutes`
  - `time_of_day`
- atmosphere:
  - `wind_speed`
  - `wind_direction`
  - `temperature`
  - `humidity`
  - `atmospheric_dryness_index`
  - `airflow_potential_peak`
- fire and terrain:
  - `grid`
  - `burning_cells`
  - `burned_cells`
  - `fire_details`
  - `heat_warnings`
  - `elevation`
  - `fuel_types`
- structures:
  - `structures`
  - `structures_remaining`
  - `structures_lost`
- fleet:
  - `resources_remaining`
  - `resource_totals`
  - `fleet_status`
  - `fleet_units`
  - `active_missions`
- last action feedback:
  - `last_action_summary`
  - `last_action_error`

## Frozen Environment Behavior

The environment currently assumes:

- one simulation tick equals `20` incident minutes
- fire evolves every step whether or not the controller acts
- fleet resources are reusable units, not consumable tokens
- units can be:
  - `available`
  - `en_route`
  - `operating`
  - `returning`
- crews and bombers act after arrival
- dozers may require multiple operating steps to complete a line

## Reward Scope In This Freeze

The environment reward is environment-driven and includes:

- structure burning (per-step urgency) and structure lost (one-time penalty)
- structure safe (continuous positive shaping)
- cells suppressed (per-cell reward for extinguishing fires)
- fire-extinguished bonus (one-time when all fires out)
- area-preservation shaping (continuous)
- action cost / invalid-action / low-impact penalties

Weather conditions (temperature, humidity, wind) are observable state only — they
do not generate reward signals.  The agent should learn that adverse weather
increases fire spread risk from the fire dynamics, not from explicit penalties.

This freeze does not attempt to finalize benchmark-grade task scoring yet.

## Validation State

At the time of this freeze:

- `python -m wildfire_env.test_env_smoke` passes
- `openenv validate wildfire_env` passes

## Change Policy

Until the freeze is intentionally lifted:

- do not change the action schema casually
- do not change the observation schema casually
- do not add non-environment features into the core environment milestone
- prefer documentation-only clarifications unless a bug is found in the
  environment itself

# Wildfire Environment Implementation Plan

This plan translates the hackathon requirements into concrete work for the
current repository.

Related context:

- [round-1-problem-statement.md](./round-1-problem-statement.md)
- [HACKATHON_REQUIREMENTS_CONTEXT.md](./HACKATHON_REQUIREMENTS_CONTEXT.md)
- [PROJECT_CONTEXT.md](./PROJECT_CONTEXT.md)
- [EMERGENCE_FIRST_MODELING_SPEC.md](./EMERGENCE_FIRST_MODELING_SPEC.md)
- [FIRE_FACTOR_VALIDATION.md](./FIRE_FACTOR_VALIDATION.md)

## Modeling Rule

The simulator should follow an emergence-first approach:

- model base variables and direct couplings
- avoid scripting named wildfire phenomena as explicit logic
- use validation-backed governing equations where possible
- treat dense pairwise factor tables as documentation, not as the simulator

This matters for every phase below, especially environment-core work and
resource effects.

## Current Status

Already present:

- terrain generation and seeded task presets
- a detailed wildfire simulation engine
- an OpenEnv scaffold with server, client, Dockerfile, and manifest
- the OpenEnv wrapper now runs the real wildfire simulator instead of the echo
  scaffold
- structured wildfire actions now exist for:
  - `advance`
  - `deploy_crew`
  - `drop_water`
  - `cut_firebreak`
- direct smoke tests now exist for:
  - simulator-only runs
  - OpenEnv wrapper runs

Not yet connected:

- no task graders
- no baseline inference script
- no custom `/tasks`, `/grader`, or `/baseline` endpoints
- README still needs final submission-grade polish and benchmark reporting

## Phase 0: Validator And Runtime Blockers

Goal: make the project import and validate cleanly before deeper feature work.

Tasks:

- fix `server.app` import behavior so validator import paths work
- fix terrain generation fallback so it does not require SciPy unless declared
- align dependencies in `pyproject.toml` and runtime requirements
- ensure package entrypoints work from both package and local execution modes

Done when:

- `openenv validate` passes the basic app import checks
- terrain generation runs in the current environment
- local server startup works from the documented commands

## Phase 1: Replace Echo Environment With Wildfire Environment

Goal: make the OpenEnv API expose the actual wildfire simulation.

Tasks:

- replace `WildfireAction` with structured resource-allocation actions
- replace `WildfireObservation` with a wildfire command observation
- add a typed reward model if the chosen OpenEnv pattern requires it explicitly
- update `WildfireEnvironment.reset()` to:
  - select a task
  - generate terrain
  - initialize simulation
  - initialize resource inventory and episode tracking
- update `WildfireEnvironment.step()` to:
  - validate the action
  - apply the command
  - advance the simulation
  - compute reward
  - set `done`
  - attach useful `info`
- update `state()` to expose current episode/task/resource state
- update `client.py` to serialize and parse the new action/observation shape

Done when:

- `reset()`, `step()`, and `state()` all operate on the wildfire environment
- an agent can complete a full episode through the OpenEnv client
- the environment no longer references message echo semantics anywhere

## Phase 1.5: Align The Core Simulator With Emergence-First Design

Goal: make sure the wildfire core models only base state and direct couplings.

Tasks:

- audit current phenomenon-specific heuristics in the simulator
- classify each current rule as:
  - governing
  - temporary approximation
  - remove
- remove explicit behavior switches where possible
- keep validated governing equations from the factor-validation doc
- decide whether a minimal local airflow / buoyancy perturbation field is
  needed for genuine fire-atmosphere feedback

Done when:

- the core simulator matches the emergence-first modeling spec closely enough
  that higher-level wildfire behavior is coming from coupled variables rather
  than named special cases

## Phase 2: Resource System

Goal: model a real incident-command decision loop rather than passive
observation.

Tasks:

- create `server/resources.py`
- define deployable actions for:
  - crews
  - bombers
  - dozers
  - optional no-op or observe action
- define resource constraints:
  - limited counts by task
  - valid target cells or target areas
  - cooldowns, per-step limits, or travel assumptions
  - invalid action handling
- map actions into simulation effects:
  - suppression turns cells into `STATE_SUPPRESSED`
  - dozers create firebreaks
  - bombers reduce heat and/or increase moisture
- track resource usage and waste for grading and reward shaping

Done when:

- each resource has clear mechanics
- invalid or wasteful actions have deterministic consequences
- the action space reflects a real commander workflow

## Phase 3: Tasks And Graders

Goal: satisfy the hackathon requirement for 3 graded tasks.

Tasks:

- define explicit task metadata for `easy`, `medium`, and `hard`
- make task objectives concrete, for example:
  - easy: protect low-priority structures with ample resources
  - medium: balance containment and structure protection with tighter limits
  - hard: multi-ignition event with scarce resources and higher-value assets
- create `server/grader.py`
- implement deterministic graders that score:
  - structures saved
  - area preserved
  - containment success
  - resource efficiency
  - policy violations or waste
- clamp final grader scores to `0.0` to `1.0`
- ensure the hard task is materially harder, not just longer

Done when:

- all three tasks are discoverable
- graders are reproducible for fixed seeds
- each completed episode yields a deterministic score

## Phase 4: Reward Design

Goal: give dense training signal while keeping final grading objective clean.

Tasks:

- separate step reward shaping from final grader score
- reuse the simulation's environmental reward signals where appropriate
- add action-based signals for:
  - protecting threatened structures
  - reducing spread
  - timely containment
  - efficient resource use
- add penalties for:
  - invalid actions
  - repeated no-impact actions
  - overspending scarce resources
  - failing to respond to imminent threats

Done when:

- rewards vary meaningfully across a trajectory
- an agent can improve before full task success
- the reward does not collapse into a single terminal binary

## Phase 5: API Surface And Submission Endpoints

Goal: expose the endpoints expected by the hackathon and evaluators.

Tasks:

- keep standard OpenEnv endpoints working:
  - `/reset`
  - `/step`
  - `/state`
- add custom endpoints:
  - `/tasks`
  - `/grader`
  - `/baseline`
- make `/tasks` return:
  - task list
  - action schema
  - required step fields
- make `/grader` return the final task score and grading breakdown
- make `/baseline` execute the baseline routine and return scores for all tasks

Done when:

- evaluator tooling can inspect tasks and retrieve scores without manual steps

## Phase 6: Baseline Inference

Goal: produce a reproducible baseline agent consistent with the submission rules.

Tasks:

- add root-level `inference.py`
- use the OpenAI client for every LLM request
- read:
  - `OPENAI_API_KEY`
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN` if needed by the deployment flow
- run across `easy`, `medium`, and `hard`
- keep runtime under 20 minutes on 2 vCPU / 8 GB
- use deterministic settings where possible:
  - fixed seed
  - low or zero temperature
  - bounded max steps
- emit final per-task and aggregate scores

Done when:

- `inference.py` runs end-to-end from the repo root
- the output is reproducible and submission-ready

## Phase 7: Documentation And Packaging

Goal: make the project ready for validation and review.

Tasks:

- rewrite `wildfire_env/README.md`
- update `openenv.yaml` metadata as needed
- ensure Dockerfile builds from a clean environment
- document local run, Docker run, validation, and inference usage
- document action/observation spaces and task difficulty
- include baseline scores in the README once the baseline exists

Done when:

- the README reads like a finished environment, not a scaffold
- all documented commands work

## Phase 8: Validation Pass

Goal: verify the submission against the hackathon gate.

Tasks:

- run `openenv validate`
- run local smoke tests for reset/step/state
- run graders for all tasks
- run `inference.py`
- build and run Docker locally
- verify endpoint health and basic responses

Done when:

- the repo satisfies the pre-submission checklist in the hackathon context file

## Recommended Implementation Order

1. Phase 0: validator and dependency blockers
2. Phase 1: real environment wiring
3. Phase 1.5: emergence-first simulator cleanup
4. Phase 2: resource mechanics
5. Phase 3: tasks and graders
6. Phase 4: reward shaping
7. Phase 5: custom endpoints
8. Phase 6: inference baseline
9. Phase 7: docs and packaging
10. Phase 8: validation pass

## Immediate Next Steps

- Fix the validator/runtime blockers first.
- Define the wildfire action model before touching the client and endpoints.
- Keep the core simulator aligned with the emergence-first design while adding
  resource mechanics.
- Wire one complete task end-to-end, then generalize to all three difficulties.
- Add graders before the baseline so the baseline can score against the final
  contract.

## March 31 Status Update

Resource mechanics are no longer instantaneous-token actions.

Implemented:

- reusable fleet units for crews, bombers, and dozers
- deterministic staging bases per resource type
- dispatch ETA, on-incident operating state, and return-to-service timing
- explicit incident clock in the observation (`step_minutes`,
  `elapsed_minutes`, `time_of_day`)
- active mission visibility so the agent can reason about what is already in
  motion

Environment-only freeze decision:

- freeze the current environment contract
- keep non-environment files in the repo, but treat them as deferred work
- do not mix trained-policy or baseline work into the current environment
  milestone

Deferred beyond this freeze:

- richer control policies on top of the fleet timing model
- graders that score strategic use of delayed resources
- baseline inference that learns to plan around ETAs instead of assuming
  immediate suppression

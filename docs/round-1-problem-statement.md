# Round 1 Problem Statement

## The Task

Build a complete, real-world OpenEnv environment that an AI agent can learn from through the standard `step()` / `reset()` / `state()` API.

## Key Requirements at a Glance

- Simulate a real-world task, not a game or toy.
- Implement the full OpenEnv spec:
  - Typed models
  - `step()`
  - `reset()`
  - `state()`
  - `openenv.yaml`
- Include at least 3 tasks with agent graders:
  - Easy
  - Medium
  - Hard
  - Scores must be in the range `0.0` to `1.0`
- Design a meaningful reward function with partial progress signals.
- Provide a baseline inference script with reproducible scores.
- Deploy to Hugging Face Spaces with a working `Dockerfile`.
- Add a `README` covering:
  - Environment description
  - Action and observation spaces
  - Setup instructions

## Functional Requirements

### 1. Real-World Task Simulation

The environment must simulate a task that humans actually do in the real world.

Not allowed:

- Games
- Toy problems
- Artificial tasks with no practical use

Example domains:

- Email triage
- Code review
- Data cleaning
- Scheduling
- Customer support
- Content moderation

### 2. OpenEnv Spec Compliance

The environment must implement the full OpenEnv interface.

Required components:

- Typed `Observation`, `Action`, and `Reward` Pydantic models
- `step(action)` returning:
  - Observation
  - Reward
  - Done
  - Info
- `reset()` returning the initial observation
- `state()` returning the current state
- `openenv.yaml` with metadata

Validation requirement:

- Must pass `openenv validate`

### 3. Minimum 3 Tasks with Agent Graders

Each task must:

- Define a concrete objective
- Include a programmatic grader
- Return a score between `0.0` and `1.0`

Task progression must include:

- Easy
- Medium
- Hard

Grader requirements:

- Clear
- Deterministic
- Reproducible

### 4. Meaningful Reward Function

The reward should provide signal across the full trajectory, not only at the end.

It should:

- Reward partial progress toward success
- Penalize undesirable behavior

Examples of undesirable behavior:

- Infinite loops
- Destructive actions
- Repeated useless actions

### 5. Baseline Inference Script

Provide a script that:

- Uses the OpenAI API client
- Runs a model against the environment
- Reads credentials from environment variables

Required environment variable:

- `OPENAI_API_KEY`

The script must:

- Run across all 3 tasks
- Produce reproducible baseline scores

## Non-Functional Requirements

### 1. Deploys to a Hugging Face Space

The environment must run as a containerized Hugging Face Space tagged with `openenv`.

### 2. Containerized Execution

The repository must include a working `Dockerfile`.

The environment should start cleanly with:

```bash
docker build .
docker run ...
```

### 3. Documentation

The `README` must include:

- Environment description and motivation
- Action space definition
- Observation space definition
- Task descriptions with expected difficulty
- Setup instructions
- Usage instructions
- Baseline scores

## Judging Weights

| Parameter | Weight | Description |
| --- | ---: | --- |
| Real-world utility | 30% | Does the environment model a genuine task? Would someone actually use it to train or evaluate agents? |
| Task and grader quality | 25% | Are tasks well-defined, fairly graded, and meaningfully progressive in difficulty? |
| Environment design | 20% | Is state management clean? Are the spaces sensible? Is reward shaping useful? |
| Code quality and spec compliance | 15% | Does it follow OpenEnv spec, stay well-structured, and ship with working infra? |
| Creativity and novelty | 10% | Is the domain original, interesting, and thoughtfully designed? |

## Scoring Breakdown

### Real-World Utility (30%)

- `0–5`: Toy or artificial problem with no practical application
- `6–15`: Valid domain, but shallow real-world modeling
- `16–25`: Good domain modeling, useful for agent evaluation
- `26–30`: Excellent real-world value and fills a meaningful gap

### Task and Grader Quality (25%)

Evaluation questions:

- Are there 3 or more tasks with a clear difficulty range?
- Do graders produce scores between `0.0` and `1.0`?
- Are graders deterministic and reproducible?
- Does the hard task genuinely challenge strong models?

### Environment Design (20%)

Evaluation questions:

- Does `reset()` produce a clean initial state?
- Are action and observation types well-designed and documented?
- Does the reward function provide useful varying signal?
- Are episode boundaries sensible?

### Code Quality and Spec Compliance (15%)

Evaluation questions:

- Does `openenv validate` pass?
- Does `docker build && docker run` work?
- Does the Hugging Face Space deploy and respond?
- Does the baseline script run and reproduce scores?

### Creativity and Novelty (10%)

Evaluation questions:

- Is the domain new or uncommon in OpenEnv?
- Does the reward design have interesting properties?
- Are there clever mechanics that improve realism or challenge?

## Evaluation Process

### Phase 1: Automated Validation

Pass/fail gate. The submission must satisfy:

- Hugging Face Space deploys
- OpenEnv spec compliance passes
- `Dockerfile` builds successfully
- Baseline script reproduces results
- 3 or more tasks with graders are present

### Phase 2: Agentic Evaluation

Scored phase. Evaluators will:

- Re-run the submitted baseline agent
- Run a standard open LLM agent against all environments
- Check score variance and consistency

### Phase 3: Human Review

Top submissions will be reviewed by Meta and Hugging Face engineers for:

- Real-world utility
- Creativity
- Exploit resistance

## Disqualification Criteria

- Environment does not deploy or respond
- Submission is plagiarized or only trivially modified
- Graders always return the same score
- No baseline inference script is included

## How Judging Works

### Pre-Submission Checklist

All of the following must pass or the submission is disqualified:

#### HF Space Deploys

- Automated ping to the Space URL
- Must return `200`
- Must respond to `reset()`

#### OpenEnv Spec Compliance

Validator checks:

- `openenv.yaml`
- Typed models
- `step()`
- `reset()`
- `state()` endpoints

#### Dockerfile Builds

- Automated `docker build` runs on the submitted repository

#### Baseline Reproduces

- Submitted inference script is executed
- Must complete without error
- Must produce scores

#### 3 or More Tasks with Graders

- Tasks are enumerated
- Each grader is run
- Scores are verified to stay within `0.0` to `1.0`

## Additional Endpoints to Expose

### `/baseline`

Triggers the inference script and returns the baseline score for all 3 tasks.

### `/grader`

Returns the grader score after an episode is completed.

### `/tasks`

Returns:

- The list of tasks
- The action schema
- The required fields for an action in a step

## Validator

Run the pre-submission validation script before submitting.

## Recommended Build Checklist

Use this as the working implementation checklist:

- Choose a genuinely useful real-world workflow
- Define observation, action, and reward models with Pydantic
- Implement `reset()`, `step()`, and `state()`
- Create `openenv.yaml`
- Build at least 3 progressively harder tasks
- Implement deterministic graders returning `0.0` to `1.0`
- Design dense reward shaping with penalties for bad behavior
- Write a reproducible baseline inference script
- Add `/baseline`, `/grader`, and `/tasks` endpoints
- Containerize with a working `Dockerfile`
- Deploy to Hugging Face Spaces
- Write a complete `README`
- Run validation before submission

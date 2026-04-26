# An Environment to Save the Environment

## TL;DR

This project is an OpenEnv-compatible wildfire incident-command environment hosted on Hugging Face Spaces. An AI agent acts as an incident commander: it observes a partially visible wildfire, dispatches limited firefighting resources, protects structures, and is graded through a deterministic `/grader` endpoint.

The core idea is to test whether language-model agents can make useful long-horizon decisions in a dynamic world where actions have delayed consequences. The agent is not answering a one-shot question. It is commanding an incident through changing weather, terrain, fire spread, resource logistics, structure priorities, and safety constraints.

- Live Space: https://huggingface.co/spaces/Chunchunmaru-101/wildfire-env
- Live app: https://chunchunmaru-101-wildfire-env.hf.space
- OpenEnv endpoints: `WS /ws` (stateful `EnvClient` session for `reset`/`step`), `POST /grader`, `GET /tasks`, `GET /schema`
- Primary hackathon theme: world modeling / professional tasks
- Secondary theme: long-horizon planning and instruction following

## Why Wildfire?

When I started this project, I thought I was building a wildfire simulator.

That sounds straightforward enough: make a forest, start a fire, give an AI agent a few firefighting tools, and ask it to stop the spread. In the first version, that is more or less what I did. I prompted an AI, accepted a lot of generated code, and watched something that looked like an environment come together.

Then I read through what had been produced.

It had the shape of a simulator, but not the soul of one. The environment was too rigid. Fires spread because a rule said they should spread. Resources worked because a rule said they should work. The agent was not really operating inside a living system. It was mostly solving a scripted puzzle.

That matters, because wildfire is not a scripted puzzle.

Wildfire is weather, terrain, fuel, moisture, timing, visibility, logistics, and human decision-making colliding under pressure. A decision that looks reasonable at minute 0 can become useless by minute 40 if the wind shifts, a crew is still returning, a structure becomes exposed, or the fire creates its own local conditions.

The moment that changed the project for me was one physical loop I remembered from school-level science: a forest catches fire, the air gets hotter, hot air rises, pressure changes near the fire, and wind can be pulled toward the burn area, making the fire worse. That was just one phenomenon. Once I started thinking that way, I realized there would be dozens of patterns like it. I could not hand-script all of them.

So I changed the goal.

Instead of trying to model every wildfire phenomenon directly, I tried to model the base variables that allow those phenomena to emerge: fuel type, elevation, slope, aspect, moisture, temperature, humidity, wind speed, wind direction, heat, fire intensity, structures, and resource logistics.

AI helped most when I used it as a research assistant rather than a code generator: finding wildfire modeling references, pointing me toward Rothermel-style fuel and moisture effects, Alexandridis-style cellular automata, spotting behavior, and operational firefighting concepts like LCES safety and dispatch timing.

The result is a grid-based OpenEnv environment where the agent is not asked to answer a wildfire question. It is asked to command an incident.

## What the Environment Tests

I did not want to build a generic benchmark where an agent moves around a toy grid and collects points. I wanted an environment where planning actually meant something.

The agent must make a sequence of decisions under uncertainty:

- Which fire front matters most right now?
- Which structure is worth protecting first?
- Should scarce resources attack active fire, build a line ahead of it, stage for a delayed ignition, or protect a structure?
- Is a ground assignment tactically useful but unsafe?
- Should aircraft be spent now, or saved for a wind-driven run later?

This is why the environment fits the world-modeling and professional-task theme. The task is not solved by producing fluent text. It is solved by maintaining state across turns and choosing structured actions that change a simulated world.

## The World the Agent Sees

Each episode is a wildfire incident on a seeded terrain grid:

- Easy: `15x15`
- Medium: `20x20`
- Hard: `25x25`

The world contains fuel types, elevation, slope, aspect, water, structures with priorities, ignition points, delayed ignitions, and changing weather.

The fire does not simply expand one square at a time. Spread is affected by wind, fuel, moisture, slope, heat, and intensity. Weather changes over time. Moisture dries. Retardant decays. Wind can push spread directionally. High wind can create spotting behavior. The environment also tracks local airflow potential, a simplified way of capturing the idea that a large fire can influence the air around it.

The agent receives a structured observation each step:

- visible burning cells and heat warnings
- structure status and priorities
- current weather and a short forecast
- terrain information
- fleet status
- active missions and return times
- available resources
- last action summary
- dense step reward
- a natural-language `action_guide`

The environment also includes fog of war. The agent may know that a fire exists, but it does not automatically know every important detail unless resources are positioned to observe it. This turns the problem from "put water on the red squares" into something closer to incident command: where should I send scarce resources, what should I protect first, and what am I willing to leave uncertain?

## The Fleet

The agent has six types of firefighting resources:

- Hand crews are versatile and can perform direct attack, build lines, protect points, and set backfires.
- Engines are faster ground units that can attack fire and build wet lines.
- Helicopters can perform water drops and point protection, with refill behavior tied to water access.
- Air tankers can drop long-term retardant over larger areas, but have longer return times.
- Dozers can build durable firebreaks, especially useful for shaping the future fire perimeter.
- Smokejumpers deploy quickly into remote areas but are scarce and slow to return.

These resources are not magic buttons. Each unit has dispatch timing, travel time, return time, status, position, and mission constraints. If the agent sends everything to one flame front, it may have nothing left when a delayed ignition threatens a structure somewhere else. If it keeps units idle, the fire keeps gaining ground. If it sends ground crews into unsafe positions without escape routes or safety zones, the environment penalizes that through an LCES-inspired safety check.

The actions are tactical, not just directional. A model can issue:

- direct attack
- line construction
- wet line
- water drop
- retardant drop
- point protection
- backfire
- staging

Targets can be points, lines, areas, polygons, or specific structures. A valid action is structured JSON, but the reasoning behind it is strategic: protect the highest-priority structure, cut off spread before wind makes it worse, pre-position resources before they are needed, and avoid wasting expensive missions on low-impact moves.

## OpenEnv API Surface

The deployed Space exposes the environment through OpenEnv's standard surface:

- `WS /ws` runs a stateful session via the official `EnvClient`. One persistent connection drives `reset(task_id, seed)` then `step(action)` calls until the episode ends. This is the canonical OpenEnv transport for multi-turn rollouts.
- `POST /grader` scores the final episode state. Wildfire-specific endpoint, layered on top of the OpenEnv core.
- `GET /tasks` lists task definitions and resource compatibility.
- `GET /schema` exposes action and observation schemas.

A small but important detail: OpenEnv core's `POST /reset` and `POST /step` HTTP handlers are intentionally stateless — they construct a fresh environment per request and tear it down. They're useful for one-shot probes but cannot drive multi-turn rollouts. We discovered this the hard way when our first evaluation pass returned `obs.step == 1` after every step regardless of how many turns we ran. Switching to the official `EnvClient` WebSocket session — which keeps one environment alive for the lifetime of the connection — fixed it immediately. This is the contract every OpenEnv example documents, and it's what real RL evaluations need.

This matters for judging because the final showcase evaluation is not a private in-process shortcut. Both baseline and trained-policy evaluation are driven through the live Hugging Face Space using the official OpenEnv `EnvClient` plus the wildfire-specific `/grader`.

## Reward and Grader Design

The reward function had to do two things at once.

First, it had to teach the agent what matters: protect structures, preserve area, suppress active fire, contain spread, and extinguish the incident quickly. Structures are priority-weighted. Suppression earns reward when it actually reduces burning cells. Containment reward is gated on active suppression so the agent cannot get paid for simply waiting until fuel burns out.

Second, it had to avoid obvious reward hacks:

- No-op behavior is penalized when units are available and fire is burning.
- Duplicate dispatches are invalid.
- Wasteful low-impact actions are penalized.
- Structure-loss penalties are single-use.
- Episodes cannot terminate immediately just to farm early rewards.
- The final episode grader is separate from the dense training reward.

The final grader returns a score from `0.0` to `1.0` using:

- `60%` structure protection
- `30%` area preservation
- `10%` containment efficiency

That separation matters. The dense reward helps GRPO training learn from every step, but the grader is the cleaner judge-facing measure of whether the incident was actually handled well.

## Reward-Hacking Audit

The repository includes a reward audit harness that compares dense reward against final grader behavior. It runs a bank of fixed policies, including sane baselines and exploit-like policies:

- `noop`
- deterministic heuristic
- ground-only heuristic
- aerial-only heuristic
- stage-all
- invalid duplicate dispatch
- random valid policy

On the current audit summarized in the Space README, exploit-like policies do not outrank the sane heuristic policies on the final grader. The policy-level Spearman correlations are high across difficulties:

- Easy: `0.964`
- Medium: `0.927`
- Hard: `0.852`

This is not proof that the reward is perfect, but it is evidence that the obvious failure modes were tested instead of ignored.

## Training the Commander

The training setup uses:

- Qwen3-4B-Instruct
- LoRA / QLoRA
- GRPO
- structured JSON actions
- grammar-constrained decoding to reduce malformed actions
- a curriculum from easy to medium to hard incidents

The training loop is multi-turn. The model does not receive a static prompt and produce a single answer. It observes the environment, emits an action, receives the next state, and continues until the incident ends or the episode limit is reached.

The final evaluation path uses the official OpenEnv transport. The evaluator opens a persistent `EnvClient` WebSocket session against the deployed Space, runs `reset` then `step` per episode (preserving server-side state across turns the way OpenEnv intends), and calls `POST /grader` for the final score. Untrained and trained policies interact with the same deployed environment surface using the same flags so the delta is a clean apples-to-apples comparison.

## Baseline and Evaluation Plan

The environment includes a deterministic heuristic baseline that ranks fires by structure proximity and assigns resources greedily. This baseline is useful because it gives judges a non-LLM reference point and helps catch broken environment behavior.

The final trained-vs-untrained comparison is designed to use the same settings for both runs:

- same live Space URL
- same held-out seeds
- same max episode steps
- same max generation tokens
- same grader endpoint

The expected final artifacts are:

- `submission_artifacts/eval_untrained.json`
- `submission_artifacts/eval_trained.json`
- `submission_artifacts/training_reward_curve.png`
- `submission_artifacts/training_loss_curve.png`
- `submission_artifacts/training_summary.md`
- `reward_audit.json`

If the trained model does not dramatically beat the baseline, the environment is still meaningful: the project is primarily an environment submission. The important claim is that the world is rich enough to expose planning behavior, delayed consequences, partial observability, resource scarcity, safety tradeoffs, and weather-driven dynamics.

## What I Learned

The biggest lesson was that AI is very good at producing something that looks complete before it is actually correct.

At the start, I was prompting the AI to "do things" and accepting the output too quickly. The real progress began when I stopped treating AI as an autopilot and started treating it as a collaborator I had to audit. The first generated environment gave me a scaffold. The second pass made me ask whether the scaffold matched the real problem.

For a world-modeling environment, rigidity is the enemy. If the agent is only learning your hand-written script, it is not learning the world. The environment needs enough base mechanics for surprising but plausible behavior to emerge.

That is what I tried to build here: not a perfect wildfire simulator, but a serious planning environment where an AI incident commander has to reason under uncertainty, act with limited resources, and live with the consequences of its decisions.

## Why This Belongs in the HF Space

The judge's guidance was clear: the blog should be a separate Markdown file pushed into the Hugging Face Space, separate from the README. That is why this writeup is provided as `Blog.MD`.

Keeping the environment, README, blog, training scripts, eval scripts, and artifacts together in the HF Space makes the submission easier to judge. The Space is not only a demo link. It is the deployed environment, the API surface, the documentation, and the evidence trail for the final run.

## References and Influences

- Rothermel, R. C. (1972), wildfire spread modeling
- Alexandridis et al., cellular automata wildfire spread modeling
- Albini-style spotting approximations
- NWCG operational concepts, including LCES and dispatch timing
- Australia 2019-2020 Black Summer
- California 2020 wildfire season

## Links

- Live Space: https://huggingface.co/spaces/Chunchunmaru-101/wildfire-env
- Live app: https://chunchunmaru-101-wildfire-env.hf.space
- Space README: ./README.md
- Environment package docs: ./wildfire_env/README.md
- Training pipeline: ./train_grpo.py
- OpenEnv evaluator (WebSocket via official `EnvClient`, plus `POST /grader`): ./eval_policy_http.py

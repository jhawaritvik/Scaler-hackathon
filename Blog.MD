# An Environment to Save the Environment

I started this project with a small thought:

What if an AI agent could practice making decisions in a wildfire before those decisions ever mattered in the real world?

That sentence slowly became the heart of the project:

**An environment to save the environment.**

At first, I thought I was simply building a wildfire simulator. Make a grid, add trees, start a fire, give the agent water drops and fire crews, and see if it can stop the spread. It sounded clean and manageable.

The first version even looked convincing. There was a map. There was fire. There were actions. The agent could send resources and the fire would react.

But when I sat with it for a while, something felt wrong.

It was not really a wildfire. It was a puzzle wearing wildfire clothes.

The fire spread because I told it to spread. The resources worked because I told them to work. The agent was not learning to survive a changing situation. It was learning my script.

That bothered me, because wildfire is the opposite of scripted.

Wildfire is messy. Wind changes. Moisture drops. Heat builds. Fire moves uphill faster. A crew can be busy when you suddenly need it somewhere else. A structure that looked safe ten minutes ago can become the most important thing on the map.

The more I thought about it, the more I realized that the environment should not be a list of hand-written wildfire events. It should be a world with enough basic pieces that interesting behavior can happen on its own.

So I changed direction.

Instead of scripting "wildfire behavior", I started modeling the ingredients behind it: fuel, elevation, slope, water, moisture, temperature, humidity, wind, heat, intensity, structures, and resource travel time. I wanted the agent to feel pressure from the world, not just from a scoring rule.

That is where the project became real for me.

## The Moment It Clicked

There was one simple science idea that stuck in my head: fire heats the air, hot air rises, pressure changes around the fire, and wind can be pulled toward the burn area. That can make the fire worse.

It is such a small loop, but it changed how I looked at the project.

If even one part of wildfire had that kind of feedback, then how many other loops was I missing? Wind affects fire. Fire affects heat. Heat affects nearby cells. Moisture changes over time. Terrain changes spread. Delayed ignitions change the plan. Resources take time to move and come back.

I could not script all of that one by one.

So the goal became different: build a place where those pressures can exist together, and let the agent deal with the consequences.

## What the Agent Is Really Doing

The agent is not just putting water on red squares.

It is acting like an incident commander.

Every episode starts as a seeded wildfire incident. Easy tasks use a `15x15` grid, medium uses `20x20`, and hard uses `25x25`. The map has terrain, fuel, water, structures, ignition points, delayed ignitions, and weather.

The agent gets observations over time. It sees burning cells, heat warnings, structure priorities, weather, short forecasts, available units, active missions, and return times. It also gets an `action_guide` in normal language so it knows what resources are available and what they can do.

But it does not get a perfect world.

There is fog of war. The agent may know a fire exists, but it may not know every detail unless resources are positioned to observe the area. That one choice made the environment feel much more alive to me. Suddenly, scouting mattered. Staging mattered. Not knowing mattered.

The agent has to make decisions like:

- Do I protect the closest structure or the highest-priority one?
- Do I attack the active fire or build a line ahead of it?
- Do I send the helicopter now or save it for a worse front later?
- Do I move a unit closer before I need it?
- Is this ground crew assignment useful but unsafe?

Those are the kinds of questions I wanted the environment to create.

## The Resources Are Not Magic

The agent controls hand crews, engines, helicopters, air tankers, dozers, and smokejumpers.

I tried to make them feel limited.

A helicopter can move fast, but it still has refill and return behavior. An air tanker can cover more area, but it takes longer to come back. A dozer can build a strong firebreak, but it is not instant. Smokejumpers can deploy quickly, but they are scarce. Ground crews can do a lot, but they can also be sent into unsafe places.

That last part mattered to me. In a real wildfire, "effective" and "safe" are not always the same thing. So the environment includes an LCES-inspired safety check. If a ground unit is sent too close to active fire without a reachable escape or safety zone, the agent is penalized.

This is where the environment starts to become more than a game. The best action is not always the action with the biggest immediate effect.

## A Bug That Taught Me Something

One of the most useful mistakes happened during evaluation.

I first tried to run episodes through raw HTTP calls: `POST /reset`, then `POST /step`, then another `POST /step`.

The numbers looked strange. The observation kept coming back like the episode was not really moving forward. Eventually I realized why: the OpenEnv HTTP handlers are stateless. Each `POST /step` creates a fresh environment, steps once, and then throws it away.

That is fine for quick checks, but it is not a real rollout.

For a real multi-turn episode, I had to use the official OpenEnv `EnvClient` through `WS /ws`. That keeps the environment alive across the whole episode.

It sounds like a technical detail, but it mattered a lot. A long-horizon planning environment is only meaningful if the world remembers what happened.

So the final evaluation uses the WebSocket path for `reset` and `step`, then calls `POST /grader` at the end. Both the untrained and trained policies are judged through the live Hugging Face Space.

## Teaching Without Letting the Agent Cheat

Designing the reward was harder than I expected.

If the reward is too simple, the model can learn weird shortcuts. If it gets points for containment, maybe it waits for the fire to burn itself out. If actions cost something, maybe it learns to do nothing. If structure loss is punished every step, maybe one bad event dominates the whole episode.

So I split the problem into two parts.

The dense reward helps training step by step. It rewards useful suppression, structure protection, area preservation, and faster containment. It also punishes no-ops, duplicate dispatches, invalid actions, unsafe ground assignments, and low-impact moves.

The final grader is separate. It gives a score strictly inside `(0, 1)` using five parts:

- `45%` structure protection
- `20%` area preservation
- `15%` active containment
- `10%` spread limit
- `10%` efficiency

I also wrote a reward audit. It runs normal policies and obviously bad policies like no-op, stage-all, invalid duplicate dispatch, and random valid actions. The goal was to make sure the bad policies do not accidentally look good.

On the current audit, the exploit-like policies stay below the sane heuristic policies. That does not make the reward perfect, but it made me more confident that the environment was not rewarding the most obvious wrong behavior.

## Training the Commander

For training, I used `Qwen/Qwen3-4B-Instruct-2507` with LoRA / QLoRA and GRPO. The
path I used is the
[Hugging Face training notebook](https://huggingface.co/spaces/Chunchunmaru-101/wildfire-env/blob/main/notebooks/wildfire_grpo_train_hf.ipynb)
(four code cells). **Cell 4** runs a **20-iteration** `deadline_v2_a10g`
`Config` (compressed curriculum, tuned for a single A10G), not the **60-iteration**
default in `train_grpo.py`. See the notebook and `grpo_wildfire/config.json`
for exact hyperparameters.

The model has to output structured JSON actions. To make that less fragile, I used grammar-constrained decoding. The training curriculum moves from easy to medium to hard incidents.

The important part is that the model acts over many turns. It observes the incident, sends an action, receives the next state, and keeps going until the episode ends.

The final comparison is meant to be fair:

- same live Hugging Face Space
- same held-out seeds
- same generation settings
- same episode limits
- same final grader

The final evidence lives in files like:

- `submission_artifacts/eval_untrained.json`
- `submission_artifacts/eval_trained.json`
- `submission_artifacts/training_reward_curve.png`
- `submission_artifacts/training_loss_curve.png`
- `submission_artifacts/training_summary.md`
- `reward_audit.json`

Even if the trained model is not perfect, I think the environment still matters. The real goal is to create a world where planning can be tested: uncertainty, delays, scarce resources, safety tradeoffs, and weather-driven consequences.

## The Choices Behind the Numbers

I want to be honest about the tradeoffs that shaped what this submission can and cannot show.

### Why a 4B model and not a 1B one

The first version of this project tried `Qwen2.5-1.5B`. On paper it was the obvious pick: faster training, more iterations per hour, and a bigger relative jump from baseline to trained.

In practice it broke on the action structure.

Each action is a nested JSON object: a unit ID, a mission type, a target shape with waypoints, and a commitment length. Not every mission type is valid for every resource type. Not every target shape is valid for every mission type. The unit IDs change every observation. Grammar-constrained decoding can keep the JSON well-formed, but it cannot tell the model that a dozer cannot do a water drop or that a smokejumper cannot build a polygon. Smaller models kept producing valid-looking actions that the environment had to reject.

When the environment rejects too many actions, the GRPO reward signal becomes noise. You are no longer training a wildfire policy. You are training a JSON validator that occasionally also fights fires.

So I moved to `Qwen3-4B-Instruct-2507`. The base model already understands the action structure well enough that even untrained, the parse rate stays close to perfect through every observation. That lets GRPO actually focus on tactics: which unit, where, for how long, and why now.

The cost is real. A 4B model on a single A10G is slower per training step. I made the call anyway, because a small jump on a fair signal seemed more honest than a large jump on a broken signal.

### How much training actually happened

I want to be specific about this, because it shapes what you should and should not read into the eval numbers.

The Hugging Face notebook run uses **Cell 4** = **`deadline_v2_a10g`**: **20
GRPO iterations** with a compressed curriculum (easy / medium / hard stage
lengths in the notebook) and the knobs there (`group_size=4`, `seeds_per_iter=2`,
`micro_batch_size=2`, etc.). That is a shorter schedule than the **60-iteration**
default in `train_grpo.py` (roughly 240 episode slots in the default math).

The checked-in `train_grpo.py` default is still the longer “full” configuration
I would run with more wall-clock. For the hackathon run, the source of truth is
`grpo_wildfire/config.json` and `grpo_wildfire/log.jsonl` on disk, not this blog.

### What that training budget buys you

Twenty iterations of GRPO on a 4B model is a real slice of training, not a
one-step smoke test, but it is still not “solve wildfire” scale.

The trained model is better than the untrained model. It uses more of the fleet, plans further ahead, and avoids more of the obviously bad moves. It is still beaten on some seeds by the hand-written heuristic baseline that has years of operational logic written into it line by line.

I think that comparison is the right one to look at. The interesting question for an environment like this is not "did the agent beat a number". It is "does training in this environment teach the agent something that transfers across seeds it never saw". The numbers answer that, even when the absolute scores are not flashy.

### What I would do with more compute

If I had another two days of A10G time, the order of priority would be:

1. Run the default **60-iteration** `train_grpo.py` configuration end to end, or continue past 20 iters, so the hard stage gets more of the total budget. A larger held-out seed bank would also stress-test generalization more honestly.
2. Behavior cloning the heuristic baseline first, then GRPO on top. That gives the model a strong prior on the constraint structure and lets RL focus purely on tactics.
3. Curriculum stages with explicit fog-of-war and weather-shift mini-episodes, so the model is forced to handle each pressure in isolation before they all combine.
4. A larger held-out set, ideally 25-50 seeds per task, so the trained-versus-baseline gap can be reported with proper confidence intervals.

None of these are research stretches. They are time stretches.

### What I would not change

The environment design. The two-stage reward and grader. The OpenEnv WebSocket transport. The grammar-constrained action format. The reward audit. The held-out seeds. Those are the parts I want the next agent to inherit.

The agent is the variable. The environment is the contribution.

## What I Learned

The biggest lesson was that AI can make something look finished before it is actually correct.

At the beginning, I was asking Claude to build things and accepting too much too quickly. It felt productive, but I was not always checking whether the result matched the problem.

The project improved when I stopped treating AI like an autopilot and started treating it like a collaborator I had to challenge.

The first version gave me a scaffold. The second version came from asking harder questions:

Is this really dynamic?

Can the agent make a bad early decision and suffer for it later?

Does the world keep state?

Can the reward be hacked?

Does the agent have to plan, or is it just reacting?

That process changed how I think about building environments. A good environment should not just look complex. It should create pressure. It should make the agent choose between imperfect options. It should make consequences arrive later.

That is what I tried to build here.

Not a perfect wildfire simulator. Not a real firefighting tool. But a serious planning environment where an AI incident commander has to reason under uncertainty and live with its decisions.

An environment to save the environment.

## Why This File Is Here

The judges asked for the blog as a separate Markdown file in the Hugging Face Space, separate from the README. That is why this file is named `Blog.MD`.

The Space is where the environment lives, so it makes sense for the story, code, README, eval scripts, and artifacts to live there too.

## References and Influences

- Rothermel, R. C. (1972), wildfire spread modeling
- Alexandridis et al., cellular automata wildfire modeling
- Albini-style spotting approximations
- NWCG operational concepts, including LCES and dispatch timing
- Australia 2019-2020 Black Summer
- California 2020 wildfire season

## Links

- Live Space: https://huggingface.co/spaces/Chunchunmaru-101/wildfire-env
- Live app: https://chunchunmaru-101-wildfire-env.hf.space
- Training notebook (Space): https://huggingface.co/spaces/Chunchunmaru-101/wildfire-env/blob/main/notebooks/wildfire_grpo_train_hf.ipynb
- Repository README (full technical writeup, references, repo layout): ./README.md
- Training pipeline: ./train_grpo.py
- OpenEnv evaluator: ./eval_policy_http.py

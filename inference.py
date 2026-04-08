#!/usr/bin/env python3
"""
Wildfire Environment - LLM baseline inference script.

Uses the OpenAI API client (OpenAI-compatible transport) to run an LLM agent
against all three wildfire tasks (easy, medium, hard) and reports final
grader scores.

Required environment variable:
    HF_TOKEN

Optional environment variables:
    API_BASE_URL     (default: https://router.huggingface.co/v1/)
    MODEL_NAME       (default: meta-llama/Llama-3.1-8B-Instruct)
"""

from __future__ import annotations

import json
import os
import sys

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
SCORE_EPS = 0.01

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ---------------------------------------------------------------------------
# Local imports - direct Python integration, no HTTP server needed
# ---------------------------------------------------------------------------

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from wildfire_env.server.terrain import DEFAULT_SEEDS
from wildfire_env.server.wildfire_env_environment import WildfireEnvironment
from wildfire_env.server.app import GraderRequest, _grade_episode
from wildfire_env.models import (
    GridPoint,
    ResourceAssignment,
    TargetSpec,
    WildfireAction,
    WildfireObservation,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI wildfire incident commander. Dispatch firefighting resources to contain fires and protect structures on a 15x15 terrain grid.

Each step you receive a JSON observation and must respond with a JSON action.

ACTION FORMAT (respond with valid JSON only, no markdown):
{
  "assignments": [
    {
      "unit_id": "crew_1",
      "mission_type": "direct_attack",
      "target": {
        "target_kind": "point",
        "point": {"row": 7, "col": 8}
      },
      "commitment_steps": 1
    }
  ]
}

TARGET KINDS:
  point     -> {"point": {"row": R, "col": C}}
  line      -> {"waypoints": [{"row": R1, "col": C1}, {"row": R2, "col": C2}]}
  area      -> {"center": {"row": R, "col": C}, "radius": N}
  structure -> {"structure_id": "structure_1"}

RESOURCE CAPABILITIES (from action_guide in observation):
  crews        -> direct_attack | line_construction | point_protection | backfire | staging
  engines      -> direct_attack | wet_line | point_protection | backfire | staging
  helicopters  -> water_drop | point_protection | staging (+drop_configuration: salvo/trail)
  airtankers   -> retardant_drop | staging (+drop_configuration: salvo/trail)
  dozers       -> line_construction | point_protection | staging
  smokejumpers -> direct_attack | line_construction | point_protection | staging

STRATEGY:
- Fire spreads every step - act immediately on available resources.
- Use action_guide to see which units are available and their valid missions.
- Protect high-priority structures (priority 3 > 2 > 1) first.
- Dozers build permanent firebreaks; engines create faster but temporary wet lines.
- Helicopters near water bodies reload faster (water_scoop mechanic).
- Backfire requires a firebreak, water, burned, or bare-ground anchor cell.
- If no fire is present, stage resources near likely ignition zones.
- You may issue multiple assignments per step.

Return ONLY valid JSON. No explanation, no code blocks."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_user_prompt(obs: WildfireObservation) -> str:
    """Serialize the relevant parts of an observation for the LLM."""
    return json.dumps(
        {
            "step": obs.step,
            "max_steps": obs.max_steps,
            "goal": obs.goal,
            "action_guide": obs.action_guide,
            "fire": {
                "burning_cells": obs.burning_cells,
                "burned_cells": obs.burned_cells,
                "details": [
                    {"row": f.row, "col": f.col, "intensity": round(f.intensity, 2)}
                    for f in sorted(obs.fire_details, key=lambda f: -f.intensity)[:8]
                ],
                "heat_warnings": [
                    {"row": h.row, "col": h.col}
                    for h in obs.heat_warnings[:5]
                ],
            },
            "atmosphere": {
                "wind_speed_kmh": round(obs.wind_speed, 1),
                "wind_direction_deg": round(obs.wind_direction, 1),
                "temperature_c": round(obs.temperature, 1),
                "humidity": round(obs.humidity, 2),
            },
            "structures": [
                {
                    "id": s.structure_id,
                    "row": s.row,
                    "col": s.col,
                    "priority": s.priority,
                    "status": s.status,
                }
                for s in obs.structures
            ],
        },
        indent=2,
    )


def _single_line(value: str | None, default: str = "null") -> str:
    """Keep structured log fields on one line."""
    if value is None:
        return default
    compact = " ".join(str(value).split())
    return compact if compact else default


def _llm_action(client, obs: WildfireObservation) -> WildfireAction:
    """Ask the LLM for an action and parse the JSON response."""
    user_content = (
        f"Current observation:\n{_build_user_prompt(obs)}\n\n"
        "Respond with your action JSON:"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=800,
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return WildfireAction(**data)
    except Exception:
        return WildfireAction()


def _clamp_reward(value: float) -> float:
    """Clamp a value strictly inside (0, 1) surviving 2-dp rounding."""
    return max(0.01, min(0.99, float(value)))


def _compute_score(obs: WildfireObservation) -> float:
    """Compute the grader score from the final observation."""
    req = GraderRequest(
        task_id=obs.task_id,
        step=obs.step,
        max_steps=obs.max_steps,
        structures=[s.model_dump() for s in obs.structures],
        burned_cells=obs.burned_cells,
        burning_cells=obs.burning_cells,
    )
    return _grade_episode(req).score


def _action_str(action: WildfireAction) -> str:
    """Compact string representation of an action for [STEP] output."""
    if not action.assignments:
        return "noop()"
    return ",".join(f"{a.mission_type}({a.unit_id})" for a in action.assignments)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

TASK_RUNS = [
    ("easy", DEFAULT_SEEDS["easy"]),
    ("medium", DEFAULT_SEEDS["medium"]),
    ("hard", DEFAULT_SEEDS["hard"]),
]

ENV_NAME = "wildfire_env"


def run_inference() -> dict[str, float]:
    """Run the LLM agent over all three tasks and return {task_id: score}."""
    from openai import OpenAI

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    scores: dict[str, float] = {}

    for task_id, seed in TASK_RUNS:
        env = WildfireEnvironment()
        obs: WildfireObservation | None = None
        step = 0
        rewards: list[float] = []
        success = False

        print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

        try:
            obs = env.reset(task_id=task_id, seed=seed)
            score = SCORE_EPS
            while not obs.done:
                action = _llm_action(client, obs)
                obs = env.step(action)
                step += 1

                if obs.done:
                    score = _compute_score(obs)
                    reward = round(_clamp_reward(score), 2)
                else:
                    reward = round(_clamp_reward(obs.reward), 2)

                rewards.append(reward)
                error = _single_line(obs.last_action_error)

                print(
                    f"[STEP] step={step} "
                    f"action={_single_line(_action_str(action), default='noop()')} "
                    f"reward={reward:.2f} "
                    f"done={'true' if obs.done else 'false'} "
                    f"error={error}"
                )

            scores[task_id] = score
            success = score > SCORE_EPS
        except Exception:
            scores[task_id] = SCORE_EPS
        finally:
            env.close()
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(
                f"[END] success={'true' if success else 'false'} "
                f"steps={step} "
                f"rewards={rewards_str}"
            )

    avg = sum(scores.values()) / max(1, len(scores))

    with open("baseline_scores.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": MODEL_NAME,
                "task_runs": [
                    {"task_id": task_id, "seed": seed}
                    for task_id, seed in TASK_RUNS
                ],
                "scores": scores,
                "average": round(avg, 4),
            },
            fh,
            indent=2,
        )

    return scores


if __name__ == "__main__":
    run_inference()

#!/usr/bin/env python3
"""
Wildfire Environment — LLM Baseline Inference Script

Uses the OpenAI API client (OpenAI-compatible transport) to run an LLM agent
against all three wildfire tasks (easy, medium, hard) and reports final
grader scores.

Required environment variable:
    HF_TOKEN         — Hugging Face API token

Optional environment variables:
    API_BASE_URL     — HF inference endpoint (default: https://router.huggingface.co/v1/)
    MODEL_NAME       — Model identifier (default: meta-llama/Llama-3.1-8B-Instruct)

Usage:
    python inference.py

The script uses direct Python integration — no running server required.
Scores are written to stdout and also saved to baseline_scores.json.
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ---------------------------------------------------------------------------
# Local imports — direct Python integration, no HTTP server needed
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
  point    -> {"point": {"row": R, "col": C}}
  line     -> {"waypoints": [{"row": R1, "col": C1}, {"row": R2, "col": C2}]}
  area     -> {"center": {"row": R, "col": C}, "radius": N}
  structure -> {"structure_id": "structure_1"}

RESOURCE CAPABILITIES (from action_guide in observation):
  crews        -> direct_attack | line_construction | point_protection | backfire | staging
  engines      -> direct_attack | wet_line | point_protection | backfire | staging
  helicopters  -> water_drop | point_protection | staging  (+drop_configuration: salvo/trail)
  airtankers   -> retardant_drop | staging                 (+drop_configuration: salvo/trail)
  dozers       -> line_construction | point_protection | staging
  smokejumpers -> direct_attack | line_construction | point_protection | staging

STRATEGY:
- Fire spreads every step — act immediately on available resources.
- Use action_guide to see which units are available and their valid missions.
- Protect high-priority structures (priority 3 > 2 > 1) first.
- Dozers build permanent firebreaks; engines create faster but temporary wet lines.
- Helicopters near water bodies reload faster (water_scoop mechanic).
- Backfire requires a firebreak, water, burned, or bare-ground anchor cell.
- If no fire is present, stage resources near likely ignition zones.
- You may issue multiple assignments per step.

Return ONLY valid JSON. No explanation, no code blocks."""


# ---------------------------------------------------------------------------
# LLM interaction helpers
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
    except Exception as exc:
        print(f"    [LLM error: {exc}] — submitting no-op action", file=sys.stderr)
        return WildfireAction()


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
    parts = []
    for a in action.assignments:
        parts.append(f"{a.mission_type}({a.unit_id})")
    return ",".join(parts)


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
            while not obs.done:
                action = _llm_action(client, obs)
                obs = env.step(action)
                step += 1
                reward = round(obs.reward, 2)
                rewards.append(reward)
                error = obs.last_action_error or "null"

                # [STEP] line
                print(
                    f"[STEP] step={step} "
                    f"action={_action_str(action)} "
                    f"reward={reward:.2f} "
                    f"done={'true' if obs.done else 'false'} "
                    f"error={error}"
                )

            score = _compute_score(obs)
            scores[task_id] = score
            success = score > 0.0
        except Exception as exc:
            print(f"    [Episode error: {exc}]", file=sys.stderr)
            scores[task_id] = 0.0
        finally:
            env.close()
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(
                f"[END] success={'true' if success else 'false'} "
                f"steps={step} "
                f"rewards={rewards_str}"
            )

    # Summary to stderr (not part of required format)
    print("\n" + "=" * 60, file=sys.stderr)
    print("BASELINE SCORES:", file=sys.stderr)
    for tid, sc in scores.items():
        print(f"  {tid:6s}: {sc:.4f}", file=sys.stderr)
    avg = sum(scores.values()) / max(1, len(scores))
    print(f"  {'avg':6s}: {avg:.4f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    # Write to file for CI/evaluator consumption
    with open("baseline_scores.json", "w") as fh:
        json.dump(
            {
                "model": MODEL_NAME,
                "task_runs": [{"task_id": task_id, "seed": seed} for task_id, seed in TASK_RUNS],
                "scores": scores,
                "average": round(avg, 4),
            },
            fh,
            indent=2,
        )
    print(f"\nScores saved to baseline_scores.json", file=sys.stderr)

    return scores


if __name__ == "__main__":
    run_inference()

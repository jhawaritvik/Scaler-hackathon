#!/usr/bin/env python3
"""
Wildfire Environment — LLM Baseline Inference Script

Uses the OpenAI API client to run an LLM agent against all three wildfire
tasks (easy, medium, hard) and reports final grader scores.

Required environment variable:
    OPENAI_API_KEY   — API key for the OpenAI client

Optional environment variables:
    API_BASE_URL     — Override base URL (e.g., for compatible endpoints)
    MODEL_NAME       — Model identifier (default: gpt-4o-mini)

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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
API_BASE_URL = os.environ.get("API_BASE_URL", None)
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Local imports — direct Python integration, no HTTP server needed
# ---------------------------------------------------------------------------

_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

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

SYSTEM_PROMPT = """You are an AI wildfire incident commander. Dispatch firefighting resources to contain fires and protect structures on a 15×15 terrain grid.

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
  point    → {"point": {"row": R, "col": C}}
  line     → {"waypoints": [{"row": R1, "col": C1}, {"row": R2, "col": C2}]}
  area     → {"center": {"row": R, "col": C}, "radius": N}
  structure → {"structure_id": "structure_1"}

RESOURCE CAPABILITIES (from action_guide in observation):
  crews        → direct_attack | line_construction | point_protection | backfire | staging
  engines      → direct_attack | wet_line | point_protection | backfire | staging
  helicopters  → water_drop | point_protection | staging  (+drop_configuration: salvo/trail)
  airtankers   → retardant_drop | staging                 (+drop_configuration: salvo/trail)
  dozers       → line_construction | point_protection | staging
  smokejumpers → direct_attack | line_construction | point_protection | staging

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


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference() -> dict[str, float]:
    """Run the LLM agent over all three tasks and return {task_id: score}."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI

    client_kwargs: dict = {"api_key": OPENAI_API_KEY}
    if API_BASE_URL:
        client_kwargs["base_url"] = API_BASE_URL

    client = OpenAI(**client_kwargs)
    env = WildfireEnvironment()
    scores: dict[str, float] = {}

    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    for episode in range(3):
        obs = env.reset()
        task_id = obs.task_id
        print(f"\nTask [{episode + 1}/3]: {task_id}")
        print(f"  Goal: {obs.goal}")
        print(f"  Resources: {obs.resources_remaining}")

        step = 0
        while not obs.done:
            action = _llm_action(client, obs)
            obs = env.step(action)
            step += 1
            print(
                f"  Step {obs.step:2d}: "
                f"reward={obs.reward:+.3f}  "
                f"burning={obs.burning_cells}  "
                f"lost={obs.structures_lost}  "
                f"assignments={len(action.assignments)}  "
                f"error={obs.last_action_error or '-'}"
            )

        score = _compute_score(obs)
        scores[task_id] = score
        print(
            f"  → Score: {score:.4f}  "
            f"({step} steps, "
            f"{obs.structures_remaining}/{obs.structures_remaining + obs.structures_lost} "
            f"structures saved)"
        )

    print("\n" + "=" * 60)
    print("BASELINE SCORES:")
    for tid, sc in scores.items():
        print(f"  {tid:6s}: {sc:.4f}")
    avg = sum(scores.values()) / max(1, len(scores))
    print(f"  {'avg':6s}: {avg:.4f}")
    print("=" * 60)

    # Write to file for CI/evaluator consumption
    with open("baseline_scores.json", "w") as fh:
        json.dump(
            {
                "model": MODEL_NAME,
                "scores": scores,
                "average": round(avg, 4),
            },
            fh,
            indent=2,
        )
    print(f"\nScores saved to baseline_scores.json")

    return scores


if __name__ == "__main__":
    run_inference()

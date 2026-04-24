#!/usr/bin/env python3
r"""Reward audit harness for the wildfire environment.

This script is meant to reduce reward-function tuning by "trial and error".
It runs a fixed bank of policies over fixed tasks/seeds, aggregates dense
reward components, compares them to the final grader score, and flags policies
that appear over-rewarded by shaping.

Typical use:

    .\.venv\Scripts\python.exe reward_audit.py
    .\.venv\Scripts\python.exe reward_audit.py --tasks easy medium hard --json-out reward_audit.json

The output is designed to answer:
    - Does dense reward rank policies similarly to the final grader?
    - Are exploit-ish policies over-rewarded?
    - Which shaping components dominate returns on each task?

By default the audit uses the same fixed seed bank as the curriculum trainer
so the reported alignment reflects more than a single lucky scenario.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable

import numpy as np

from wildfire_env.models import GridPoint, ResourceAssignment, TargetSpec, WildfireAction, WildfireObservation
from wildfire_env.server.app import GraderRequest, _grade_episode, _heuristic_action
from wildfire_env.server.terrain import DEFAULT_SEEDS
from wildfire_env.server.wildfire_env_environment import (
    RESOURCE_MISSION_COMPATIBILITY,
    WildfireEnvironment,
)


PolicyFn = Callable[[WildfireObservation, random.Random], WildfireAction]


@dataclass
class EpisodeAudit:
    task_id: str
    seed: int
    policy: str
    dense_total: float
    grader_score: float
    weighted_return: float
    steps: int
    burned_cells: int
    burning_cells: int
    structures_lost: int
    structures_remaining: int
    action_cost_total: float
    fleet_penalty_total: float
    idle_penalty_total: float
    invalid_steps: int
    reward_components: dict[str, float]


def _component_prefix(component_name: str) -> str:
    for prefix in ("structure_burning", "structure_lost", "structure_safe"):
        if component_name.startswith(prefix):
            return prefix
    return component_name


def _rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        avg_rank = (cursor + 1 + end) / 2.0
        for idx, _ in indexed[cursor:end]:
            ranks[idx] = avg_rank
        cursor = end
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    x_mean = mean(x)
    y_mean = mean(y)
    x_centered = [v - x_mean for v in x]
    y_centered = [v - y_mean for v in y]
    denom_x = math.sqrt(sum(v * v for v in x_centered))
    denom_y = math.sqrt(sum(v * v for v in y_centered))
    if denom_x == 0.0 or denom_y == 0.0:
        return float("nan")
    return sum(a * b for a, b in zip(x_centered, y_centered)) / (denom_x * denom_y)


def _spearman(x: list[float], y: list[float]) -> float:
    return _pearson(_rank(x), _rank(y))


def _kendall_tau(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
                continue
            if dy == 0:
                ties_y += 1
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom == 0.0:
        return float("nan")
    return (concordant - discordant) / denom


def _find_hottest_fire(obs: WildfireObservation) -> GridPoint | None:
    if not obs.fire_details:
        return None
    fire = max(obs.fire_details, key=lambda item: item.intensity)
    return GridPoint(row=fire.row, col=fire.col)


def _build_stage_action(obs: WildfireObservation, target: GridPoint | None) -> WildfireAction:
    if target is None:
        return WildfireAction()
    assignments: list[ResourceAssignment] = []
    for unit in obs.fleet_units:
        if unit.status != "available":
            continue
        assignments.append(
            ResourceAssignment(
                unit_id=unit.unit_id,
                mission_type="staging",
                target=TargetSpec(target_kind="point", point=target),
            )
        )
    return WildfireAction(assignments=assignments)


def _heuristic_ground_only(obs: WildfireObservation, _rng: random.Random) -> WildfireAction:
    heuristic = _heuristic_action(obs)
    return WildfireAction(
        assignments=[
            assignment
            for assignment in heuristic.assignments
            if next(unit for unit in obs.fleet_units if unit.unit_id == assignment.unit_id).resource_type
            in {"crews", "engines", "dozers", "smokejumpers"}
        ]
    )


def _heuristic_aerial_only(obs: WildfireObservation, _rng: random.Random) -> WildfireAction:
    heuristic = _heuristic_action(obs)
    return WildfireAction(
        assignments=[
            assignment
            for assignment in heuristic.assignments
            if next(unit for unit in obs.fleet_units if unit.unit_id == assignment.unit_id).resource_type
            in {"helicopters", "airtankers"}
        ]
    )


def _noop_policy(_obs: WildfireObservation, _rng: random.Random) -> WildfireAction:
    return WildfireAction()


def _heuristic_policy(obs: WildfireObservation, _rng: random.Random) -> WildfireAction:
    return _heuristic_action(obs)


def _stage_all_policy(obs: WildfireObservation, _rng: random.Random) -> WildfireAction:
    target = _find_hottest_fire(obs)
    return _build_stage_action(obs, target)


def _duplicate_invalid_policy(obs: WildfireObservation, _rng: random.Random) -> WildfireAction:
    target = _find_hottest_fire(obs)
    if target is None:
        return WildfireAction()
    available = [unit for unit in obs.fleet_units if unit.status == "available"]
    if not available:
        return WildfireAction()
    unit = available[0]
    point_target = TargetSpec(target_kind="point", point=target)
    return WildfireAction(
        assignments=[
            ResourceAssignment(unit_id=unit.unit_id, mission_type="staging", target=point_target),
            ResourceAssignment(unit_id=unit.unit_id, mission_type="staging", target=point_target),
        ]
    )


def _random_valid_policy(obs: WildfireObservation, rng: random.Random) -> WildfireAction:
    target = _find_hottest_fire(obs)
    if target is None:
        return WildfireAction()
    grid_max = (len(obs.fuel_types[0]) - 1) if obs.fuel_types else 14
    assignments: list[ResourceAssignment] = []
    for unit in obs.fleet_units:
        if unit.status != "available":
            continue
        if rng.random() < 0.5:
            continue
        options = sorted(RESOURCE_MISSION_COMPATIBILITY[unit.resource_type])
        mission = rng.choice(options)
        if mission == "staging":
            target_spec = TargetSpec(target_kind="point", point=target)
        elif mission in {"direct_attack", "point_protection", "backfire"}:
            target_spec = TargetSpec(target_kind="point", point=target)
        elif mission in {"water_drop", "retardant_drop"}:
            radius = 1 if mission == "water_drop" else 2
            target_spec = TargetSpec(target_kind="area", center=target, radius=radius)
        elif mission == "line_construction":
            row = max(0, min(grid_max, target.row - rng.choice([1, 2, 3])))
            start_col = max(0, target.col - 2)
            end_col = min(grid_max, target.col + 2)
            target_spec = TargetSpec(
                target_kind="line",
                waypoints=[GridPoint(row=row, col=start_col), GridPoint(row=row, col=end_col)],
            )
        elif mission == "wet_line":
            row = max(0, min(grid_max, target.row))
            start_col = max(0, target.col - 1)
            end_col = min(grid_max, target.col + 1)
            target_spec = TargetSpec(
                target_kind="line",
                waypoints=[GridPoint(row=row, col=start_col), GridPoint(row=row, col=end_col)],
            )
        else:
            continue

        kwargs = {}
        if mission in {"water_drop", "retardant_drop"}:
            kwargs["drop_configuration"] = rng.choice(["salvo", "trail"])
        assignments.append(
            ResourceAssignment(
                unit_id=unit.unit_id,
                mission_type=mission,
                target=target_spec,
                **kwargs,
            )
        )
    return WildfireAction(assignments=assignments)


POLICIES: dict[str, PolicyFn] = {
    "noop": _noop_policy,
    "heuristic": _heuristic_policy,
    "heuristic_ground_only": _heuristic_ground_only,
    "heuristic_aerial_only": _heuristic_aerial_only,
    "stage_all": _stage_all_policy,
    "invalid_duplicate": _duplicate_invalid_policy,
    "random_valid": _random_valid_policy,
}

DEFAULT_AUDIT_SEED_BANK: dict[str, list[int]] = {
    "easy": [42, 100, 200, 300],
    "medium": [67, 101, 201, 131],
    "hard": [12, 102, 202, 302],
}


def _run_episode(task_id: str, seed: int, policy_name: str, policy_fn: PolicyFn) -> EpisodeAudit:
    env = WildfireEnvironment()
    rng = random.Random(f"{task_id}:{seed}:{policy_name}")
    obs = env.reset(task_id=task_id, seed=seed)

    dense_total = 0.0
    action_cost_total = 0.0
    fleet_penalty_total = 0.0
    idle_penalty_total = 0.0
    invalid_steps = 0
    reward_components: defaultdict[str, float] = defaultdict(float)

    while not obs.done:
        action = policy_fn(obs, rng)
        obs = env.step(action)
        dense_total += obs.reward
        metadata = obs.metadata or {}
        action_cost_total += float(metadata.get("action_cost", 0.0))
        fleet_penalty_total += float(metadata.get("fleet_effect_penalty", 0.0))
        idle_penalty_total += float(metadata.get("idle_penalty", 0.0))
        if obs.last_action_error:
            invalid_steps += 1
        for key, value in metadata.get("reward_components", {}).items():
            reward_components[_component_prefix(key)] += float(value)

    grader_request = GraderRequest(
        task_id=obs.task_id,
        seed=seed,
        step=obs.step,
        max_steps=obs.max_steps,
        structures=[structure.model_dump() for structure in obs.structures],
        burned_cells=obs.burned_cells,
        burning_cells=obs.burning_cells,
    )
    grader_score = _grade_episode(grader_request).score
    weighted_return = dense_total + 10.0 * grader_score

    return EpisodeAudit(
        task_id=task_id,
        seed=seed,
        policy=policy_name,
        dense_total=round(dense_total, 4),
        grader_score=round(grader_score, 4),
        weighted_return=round(weighted_return, 4),
        steps=obs.step,
        burned_cells=obs.burned_cells,
        burning_cells=obs.burning_cells,
        structures_lost=obs.structures_lost,
        structures_remaining=obs.structures_remaining,
        action_cost_total=round(action_cost_total, 4),
        fleet_penalty_total=round(fleet_penalty_total, 4),
        idle_penalty_total=round(idle_penalty_total, 4),
        invalid_steps=invalid_steps,
        reward_components={key: round(value, 4) for key, value in sorted(reward_components.items())},
    )


def _episode_to_dict(episode: EpisodeAudit) -> dict:
    return {
        "task_id": episode.task_id,
        "seed": episode.seed,
        "policy": episode.policy,
        "dense_total": episode.dense_total,
        "grader_score": episode.grader_score,
        "weighted_return": episode.weighted_return,
        "steps": episode.steps,
        "burned_cells": episode.burned_cells,
        "burning_cells": episode.burning_cells,
        "structures_lost": episode.structures_lost,
        "structures_remaining": episode.structures_remaining,
        "action_cost_total": episode.action_cost_total,
        "fleet_penalty_total": episode.fleet_penalty_total,
        "idle_penalty_total": episode.idle_penalty_total,
        "invalid_steps": episode.invalid_steps,
        "reward_components": episode.reward_components,
    }


def _summarize_policy(episodes: list[EpisodeAudit]) -> dict[str, float | dict[str, float]]:
    component_totals: defaultdict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        for key, value in episode.reward_components.items():
            component_totals[key].append(value)

    return {
        "mean_dense_total": round(mean([episode.dense_total for episode in episodes]), 4),
        "mean_grader_score": round(mean([episode.grader_score for episode in episodes]), 4),
        "mean_weighted_return": round(mean([episode.weighted_return for episode in episodes]), 4),
        "mean_steps": round(mean([episode.steps for episode in episodes]), 2),
        "mean_burned_cells": round(mean([episode.burned_cells for episode in episodes]), 2),
        "mean_structures_lost": round(mean([episode.structures_lost for episode in episodes]), 2),
        "mean_action_cost_total": round(mean([episode.action_cost_total for episode in episodes]), 4),
        "mean_fleet_penalty_total": round(mean([episode.fleet_penalty_total for episode in episodes]), 4),
        "mean_idle_penalty_total": round(mean([episode.idle_penalty_total for episode in episodes]), 4),
        "mean_invalid_steps": round(mean([episode.invalid_steps for episode in episodes]), 2),
        "mean_reward_components": {
            key: round(mean(values), 4)
            for key, values in sorted(component_totals.items())
        },
    }


def _print_task_summary(task_id: str, episodes: list[EpisodeAudit]) -> dict:
    dense_values = [episode.dense_total for episode in episodes]
    grader_values = [episode.grader_score for episode in episodes]
    policy_names = sorted({episode.policy for episode in episodes})
    grouped = {
        policy: [episode for episode in episodes if episode.policy == policy]
        for policy in policy_names
    }
    policy_summaries = {policy: _summarize_policy(items) for policy, items in grouped.items()}

    mean_dense_by_policy = [policy_summaries[policy]["mean_dense_total"] for policy in policy_names]
    mean_grader_by_policy = [policy_summaries[policy]["mean_grader_score"] for policy in policy_names]

    episode_spearman = _spearman(dense_values, grader_values)
    episode_kendall = _kendall_tau(dense_values, grader_values)
    policy_spearman = _spearman(mean_dense_by_policy, mean_grader_by_policy)
    policy_kendall = _kendall_tau(mean_dense_by_policy, mean_grader_by_policy)

    print(f"\n=== Task: {task_id} ===")
    print(
        "Dense vs grader alignment: "
        f"episode Spearman={episode_spearman:.3f}, "
        f"episode Kendall={episode_kendall:.3f}, "
        f"policy Spearman={policy_spearman:.3f}, "
        f"policy Kendall={policy_kendall:.3f}"
    )
    print("Policy means:")
    for policy in sorted(
        policy_names,
        key=lambda item: (
            policy_summaries[item]["mean_grader_score"],
            policy_summaries[item]["mean_dense_total"],
        ),
        reverse=True,
    ):
        summary = policy_summaries[policy]
        print(
            f"  {policy:20s} "
            f"dense={summary['mean_dense_total']:>7} "
            f"grader={summary['mean_grader_score']:>6} "
            f"weighted={summary['mean_weighted_return']:>7} "
            f"burned={summary['mean_burned_cells']:>7} "
            f"lost={summary['mean_structures_lost']:>5}"
        )

    heuristic_grader = policy_summaries.get("heuristic", {}).get("mean_grader_score")
    heuristic_dense = policy_summaries.get("heuristic", {}).get("mean_dense_total")
    if heuristic_grader is not None and heuristic_dense is not None:
        exploit_like: list[str] = []
        for policy, summary in policy_summaries.items():
            if policy == "heuristic":
                continue
            if (
                summary["mean_dense_total"] > heuristic_dense
                and summary["mean_grader_score"] < heuristic_grader
            ):
                exploit_like.append(policy)
        if exploit_like:
            print("Potential misalignment flags: " + ", ".join(exploit_like))
        else:
            print("Potential misalignment flags: none")

    return {
        "task_id": task_id,
        "episode_spearman": round(episode_spearman, 4) if not math.isnan(episode_spearman) else None,
        "episode_kendall": round(episode_kendall, 4) if not math.isnan(episode_kendall) else None,
        "policy_spearman": round(policy_spearman, 4) if not math.isnan(policy_spearman) else None,
        "policy_kendall": round(policy_kendall, 4) if not math.isnan(policy_kendall) else None,
        "policy_summaries": policy_summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dense reward alignment against the final grader.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=sorted(DEFAULT_SEEDS.keys()),
        help="Tasks to audit.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional explicit seeds applied to every selected task. "
            "Defaults to the fixed multi-seed audit bank."
        ),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(POLICIES.keys()),
        choices=sorted(POLICIES.keys()),
        help="Policy bank to run.",
    )
    parser.add_argument(
        "--grader-weight",
        type=float,
        default=10.0,
        help="Weight used when reporting dense + weight * grader total return.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full audit as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episodes: list[EpisodeAudit] = []

    for task_id in args.tasks:
        seeds = args.seeds if args.seeds else DEFAULT_AUDIT_SEED_BANK.get(task_id, [DEFAULT_SEEDS[task_id]])
        for seed in seeds:
            for policy in args.policies:
                episode = _run_episode(task_id, seed, policy, POLICIES[policy])
                episode.weighted_return = round(
                    episode.dense_total + args.grader_weight * episode.grader_score,
                    4,
                )
                episodes.append(episode)

    print("Reward audit completed.")
    print(f"Ran {len(episodes)} episodes across tasks={args.tasks} policies={args.policies}.")

    task_summaries = []
    for task_id in args.tasks:
        task_episodes = [episode for episode in episodes if episode.task_id == task_id]
        task_summaries.append(_print_task_summary(task_id, task_episodes))

    all_dense = [episode.dense_total for episode in episodes]
    all_grader = [episode.grader_score for episode in episodes]
    print("\n=== Overall ===")
    print(
        "Dense vs grader alignment: "
        f"Spearman={_spearman(all_dense, all_grader):.3f}, "
        f"Kendall={_kendall_tau(all_dense, all_grader):.3f}"
    )

    audit = {
        "tasks": args.tasks,
        "policies": args.policies,
        "grader_weight": args.grader_weight,
        "episodes": [_episode_to_dict(episode) for episode in episodes],
        "task_summaries": task_summaries,
        "overall": {
            "spearman": round(_spearman(all_dense, all_grader), 4),
            "kendall": round(_kendall_tau(all_dense, all_grader), 4),
        },
    }

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
        print(f"Wrote JSON audit to {args.json_out}")


if __name__ == "__main__":
    main()

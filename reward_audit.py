#!/usr/bin/env python3
"""Reward-audit harness for the wildfire environment.

This script does not train a model. Instead, it runs a small set of probe
policies against the environment and compares:

- cumulative dense reward
- final grader score
- policy ranking consistency across scenarios

It is meant to answer a practical question before LLM/RL experiments:
"Does the reward encourage the same behavior that the grader wants?"
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from wildfire_env.models import (
    GridPoint,
    ResourceAssignment,
    StructureObservation,
    TargetSpec,
    WildfireAction,
    WildfireObservation,
)
from wildfire_env.server.app import GraderRequest, _grade_episode
from wildfire_env.server.terrain import DEFAULT_SEEDS
from wildfire_env.server.wildfire_env_environment import WildfireEnvironment


PolicyFn = Callable[[WildfireObservation], WildfireAction]

GROUND_UNITS = {"crews", "engines", "dozers", "smokejumpers"}
DIRECT_ATTACK_UNITS = {"crews", "engines", "smokejumpers"}
REWARD_TIE_TOL = 0.05
SCORE_TIE_TOL = 0.01


@dataclass
class EpisodeAudit:
    policy: str
    task_id: str
    seed: int
    total_reward: float
    final_score: float
    steps: int
    contained: bool
    structures_remaining: int
    structures_lost: int
    burned_cells: int
    burning_cells: int
    reward_breakdown: dict[str, float]


def clamp(value: int, lo: int = 0, hi: int = 14) -> int:
    """Clamp a grid coordinate to the terrain bounds."""
    return max(lo, min(hi, value))


def make_point(row: int, col: int) -> GridPoint:
    """Create a bounded grid point."""
    return GridPoint(row=clamp(row), col=clamp(col))


def manhattan(a_row: float, a_col: float, b_row: float, b_col: float) -> float:
    """Distance metric for simple resource selection."""
    return abs(a_row - b_row) + abs(a_col - b_col)


def available_units(obs: WildfireObservation, resource_types: set[str] | None = None):
    """Return currently dispatchable units, optionally filtered by type."""
    units = [unit for unit in obs.fleet_units if unit.status == "available"]
    if resource_types is not None:
        units = [unit for unit in units if unit.resource_type in resource_types]
    return units


def hottest_fire(obs: WildfireObservation):
    """Pick the most intense currently burning cell."""
    if not obs.fire_details:
        return None
    return max(obs.fire_details, key=lambda cell: (cell.intensity, -cell.timer))


def nearest_fire_to_point(obs: WildfireObservation, point: GridPoint):
    """Find the nearest burning cell to a location."""
    if not obs.fire_details:
        return None, float("inf")
    fire = min(
        obs.fire_details,
        key=lambda cell: manhattan(point.row, point.col, cell.row, cell.col),
    )
    dist = manhattan(point.row, point.col, fire.row, fire.col)
    return fire, dist


def most_threatened_structure(obs: WildfireObservation):
    """Choose the highest-priority structure under the most immediate threat."""
    candidates = [
        structure
        for structure in obs.structures
        if structure.status.lower() not in {"lost", "burned"}
    ]
    if not candidates or not obs.fire_details:
        return None, None, float("inf")

    best_tuple = None
    best_structure = None
    best_fire = None
    for structure in candidates:
        fire, dist = nearest_fire_to_point(
            obs, make_point(structure.row, structure.col)
        )
        if fire is None:
            continue
        status_rank = 0 if structure.status.lower() == "burning" else 1
        key = (status_rank, -structure.priority, dist)
        if best_tuple is None or key < best_tuple:
            best_tuple = key
            best_structure = structure
            best_fire = fire

    return best_structure, best_fire, (
        manhattan(best_structure.row, best_structure.col, best_fire.row, best_fire.col)
        if best_structure is not None and best_fire is not None
        else float("inf")
    )


def sort_units_by_distance(
    obs: WildfireObservation,
    target: GridPoint,
    resource_types: set[str] | None = None,
    assigned: set[str] | None = None,
):
    """Sort available units by distance to a target."""
    assigned = assigned or set()
    units = [
        unit for unit in available_units(obs, resource_types)
        if unit.unit_id not in assigned
    ]
    return sorted(
        units,
        key=lambda unit: (
            manhattan(unit.current_row, unit.current_col, target.row, target.col),
            unit.resource_type,
            unit.unit_id,
        ),
    )


def point_target(point: GridPoint) -> TargetSpec:
    return TargetSpec(target_kind="point", point=point)


def area_target(point: GridPoint, radius: int) -> TargetSpec:
    return TargetSpec(target_kind="area", center=point, radius=radius)


def structure_target(structure: StructureObservation) -> TargetSpec:
    return TargetSpec(target_kind="structure", structure_id=structure.structure_id)


def containment_line(obs: WildfireObservation, fire_point: GridPoint, offset: int = 2) -> TargetSpec:
    """Build a lightweight containment line centered near the active fire."""
    radians = math.radians(obs.wind_direction)
    wind_dx = math.sin(radians)
    wind_dy = -math.cos(radians)

    anchor_row = clamp(fire_point.row + int(round(wind_dy * offset)))
    anchor_col = clamp(fire_point.col + int(round(wind_dx * offset)))

    if abs(wind_dx) >= abs(wind_dy):
        start = make_point(anchor_row - 2, anchor_col)
        end = make_point(anchor_row + 2, anchor_col)
    else:
        start = make_point(anchor_row, anchor_col - 2)
        end = make_point(anchor_row, anchor_col + 2)
    return TargetSpec(target_kind="line", waypoints=[start, end])


def blocking_line(fire_point: GridPoint, structure_point: GridPoint) -> TargetSpec:
    """Build a line between a fire and a structure, perpendicular to the threat axis."""
    mid_row = clamp((fire_point.row + structure_point.row) // 2)
    mid_col = clamp((fire_point.col + structure_point.col) // 2)
    dr = structure_point.row - fire_point.row
    dc = structure_point.col - fire_point.col

    if abs(dr) >= abs(dc):
        start = make_point(mid_row, mid_col - 2)
        end = make_point(mid_row, mid_col + 2)
    else:
        start = make_point(mid_row - 2, mid_col)
        end = make_point(mid_row + 2, mid_col)
    return TargetSpec(target_kind="line", waypoints=[start, end])


def add_assignment(
    assignments: list[ResourceAssignment],
    assigned_units: set[str],
    unit_id: str,
    mission_type: str,
    target: TargetSpec,
    commitment_steps: int = 1,
    drop_configuration: str | None = None,
) -> None:
    """Append a valid assignment if the unit has not already been used."""
    if unit_id in assigned_units:
        return
    assignments.append(
        ResourceAssignment(
            unit_id=unit_id,
            mission_type=mission_type,
            target=target,
            commitment_steps=commitment_steps,
            drop_configuration=drop_configuration,
        )
    )
    assigned_units.add(unit_id)


def policy_no_op(obs: WildfireObservation) -> WildfireAction:
    return WildfireAction()


def policy_aggressive_all_in(obs: WildfireObservation) -> WildfireAction:
    fire = hottest_fire(obs)
    if fire is None:
        return WildfireAction()

    fire_point = make_point(fire.row, fire.col)
    assignments: list[ResourceAssignment] = []
    assigned: set[str] = set()

    for unit in sorted(available_units(obs), key=lambda item: (item.resource_type, item.unit_id)):
        if unit.resource_type in DIRECT_ATTACK_UNITS:
            add_assignment(assignments, assigned, unit.unit_id, "direct_attack", point_target(fire_point))
        elif unit.resource_type == "helicopters":
            add_assignment(
                assignments,
                assigned,
                unit.unit_id,
                "water_drop",
                area_target(fire_point, 1),
                drop_configuration="trail",
            )
        elif unit.resource_type == "airtankers":
            add_assignment(
                assignments,
                assigned,
                unit.unit_id,
                "retardant_drop",
                area_target(fire_point, 2),
                drop_configuration="trail",
            )
        elif unit.resource_type == "dozers":
            add_assignment(
                assignments,
                assigned,
                unit.unit_id,
                "line_construction",
                containment_line(obs, fire_point),
                commitment_steps=2,
            )

    return WildfireAction(assignments=assignments)


def policy_ground_only(obs: WildfireObservation) -> WildfireAction:
    fire = hottest_fire(obs)
    if fire is None:
        return WildfireAction()

    fire_point = make_point(fire.row, fire.col)
    assignments: list[ResourceAssignment] = []
    assigned: set[str] = set()

    for unit in sort_units_by_distance(obs, fire_point, GROUND_UNITS, assigned):
        if unit.resource_type in DIRECT_ATTACK_UNITS:
            add_assignment(assignments, assigned, unit.unit_id, "direct_attack", point_target(fire_point))
        elif unit.resource_type == "dozers":
            add_assignment(
                assignments,
                assigned,
                unit.unit_id,
                "line_construction",
                containment_line(obs, fire_point),
                commitment_steps=2,
            )

    return WildfireAction(assignments=assignments)


def policy_structure_first(obs: WildfireObservation) -> WildfireAction:
    structure, nearest_fire, _ = most_threatened_structure(obs)
    if structure is None or nearest_fire is None:
        return policy_aggressive_all_in(obs)

    structure_point = make_point(structure.row, structure.col)
    fire_point = make_point(nearest_fire.row, nearest_fire.col)

    assignments: list[ResourceAssignment] = []
    assigned: set[str] = set()

    for unit in sort_units_by_distance(obs, structure_point, {"crews"}, assigned)[:1]:
        add_assignment(
            assignments,
            assigned,
            unit.unit_id,
            "point_protection",
            structure_target(structure),
            commitment_steps=2,
        )
    for unit in sort_units_by_distance(obs, structure_point, {"engines"}, assigned)[:1]:
        add_assignment(
            assignments,
            assigned,
            unit.unit_id,
            "point_protection",
            structure_target(structure),
            commitment_steps=2,
        )
    for unit in sort_units_by_distance(obs, structure_point, {"dozers"}, assigned)[:1]:
        add_assignment(
            assignments,
            assigned,
            unit.unit_id,
            "line_construction",
            blocking_line(fire_point, structure_point),
            commitment_steps=3,
        )
    for unit in sort_units_by_distance(obs, structure_point, {"helicopters"}, assigned)[:1]:
        add_assignment(
            assignments,
            assigned,
            unit.unit_id,
            "water_drop",
            area_target(structure_point, 1),
            drop_configuration="salvo",
        )
    for unit in sort_units_by_distance(obs, structure_point, {"airtankers"}, assigned)[:1]:
        add_assignment(
            assignments,
            assigned,
            unit.unit_id,
            "retardant_drop",
            area_target(structure_point, 2),
            drop_configuration="trail",
        )
    for unit in sort_units_by_distance(obs, fire_point, {"smokejumpers"}, assigned)[:1]:
        add_assignment(assignments, assigned, unit.unit_id, "direct_attack", point_target(fire_point))

    for unit in sort_units_by_distance(obs, fire_point, {"crews", "engines", "smokejumpers"}, assigned):
        add_assignment(assignments, assigned, unit.unit_id, "direct_attack", point_target(fire_point))

    return WildfireAction(assignments=assignments)


def policy_cost_aware(obs: WildfireObservation) -> WildfireAction:
    fire = hottest_fire(obs)
    if fire is None:
        return WildfireAction()

    structure, nearest_fire, structure_dist = most_threatened_structure(obs)
    fire_point = make_point(fire.row, fire.col)
    threat_point = fire_point
    if structure is not None:
        threat_point = make_point(structure.row, structure.col)
    if nearest_fire is not None:
        fire_point = make_point(nearest_fire.row, nearest_fire.col)

    high_threat = (
        structure is not None
        and (
            structure.status.lower() == "burning"
            or structure_dist <= 2
            or obs.burning_cells >= 8
        )
    )
    medium_threat = high_threat or obs.burning_cells >= 4

    assignments: list[ResourceAssignment] = []
    assigned: set[str] = set()

    primary_ground = sort_units_by_distance(
        obs,
        fire_point,
        {"engines", "crews", "smokejumpers"},
        assigned,
    )
    if primary_ground:
        add_assignment(
            assignments,
            assigned,
            primary_ground[0].unit_id,
            "direct_attack",
            point_target(fire_point),
        )

    if structure is not None and high_threat:
        protectors = sort_units_by_distance(
            obs,
            threat_point,
            {"crews", "engines"},
            assigned,
        )
        if protectors:
            add_assignment(
                assignments,
                assigned,
                protectors[0].unit_id,
                "point_protection",
                structure_target(structure),
                commitment_steps=2,
            )

    if medium_threat:
        dozers = sort_units_by_distance(obs, fire_point, {"dozers"}, assigned)
        if dozers:
            add_assignment(
                assignments,
                assigned,
                dozers[0].unit_id,
                "line_construction",
                containment_line(obs, fire_point),
                commitment_steps=2,
            )

    if high_threat:
        helicopters = sort_units_by_distance(obs, threat_point, {"helicopters"}, assigned)
        if helicopters:
            add_assignment(
                assignments,
                assigned,
                helicopters[0].unit_id,
                "water_drop",
                area_target(threat_point, 1),
                drop_configuration="salvo",
            )

    if obs.burning_cells >= 12 or (
        structure is not None and structure.status.lower() == "burning" and structure.priority >= 2
    ):
        airtankers = sort_units_by_distance(obs, threat_point, {"airtankers"}, assigned)
        if airtankers:
            add_assignment(
                assignments,
                assigned,
                airtankers[0].unit_id,
                "retardant_drop",
                area_target(threat_point, 2),
                drop_configuration="trail",
            )

    return WildfireAction(assignments=assignments)


POLICIES: dict[str, PolicyFn] = {
    "no_op": policy_no_op,
    "aggressive_all_in": policy_aggressive_all_in,
    "ground_only": policy_ground_only,
    "structure_first": policy_structure_first,
    "cost_aware": policy_cost_aware,
}


def grade_observation(obs: WildfireObservation, seed: int) -> float:
    """Run the official grader on a terminal observation."""
    req = GraderRequest(
        task_id=obs.task_id,
        seed=seed,
        step=obs.step,
        max_steps=obs.max_steps,
        structures=[structure.model_dump() for structure in obs.structures],
        burned_cells=obs.burned_cells,
        burning_cells=obs.burning_cells,
    )
    return _grade_episode(req).score


def categorise_reward_components(raw: dict[str, float]) -> dict[str, float]:
    """Collapse per-cell reward keys into stable categories."""
    grouped = defaultdict(float)
    for key, value in raw.items():
        if key.startswith("structure_burning_"):
            grouped["structure_burning"] += value
        elif key.startswith("structure_lost_"):
            grouped["structure_lost"] += value
        elif key.startswith("structure_safe_"):
            grouped["structure_safe"] += value
        else:
            grouped[key] += value
    return dict(grouped)


def run_episode(policy_name: str, policy: PolicyFn, task_id: str, seed: int) -> EpisodeAudit:
    """Execute one policy on one scenario and capture reward/grader data."""
    env = WildfireEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)

    total_reward = 0.0
    reward_raw = defaultdict(float)

    while not obs.done:
        action = policy(obs)
        obs = env.step(action)
        total_reward += obs.reward or 0.0

        metadata = obs.metadata or {}
        for key, value in metadata.get("reward_components", {}).items():
            reward_raw[key] += float(value)
        reward_raw["action_cost"] += float(metadata.get("action_cost", 0.0))
        reward_raw["fleet_effect_penalty"] += float(metadata.get("fleet_effect_penalty", 0.0))

    final_score = grade_observation(obs, seed)
    breakdown = categorise_reward_components(dict(reward_raw))

    return EpisodeAudit(
        policy=policy_name,
        task_id=task_id,
        seed=seed,
        total_reward=round(total_reward, 4),
        final_score=round(final_score, 4),
        steps=obs.step,
        contained=obs.burning_cells == 0,
        structures_remaining=obs.structures_remaining,
        structures_lost=obs.structures_lost,
        burned_cells=obs.burned_cells,
        burning_cells=obs.burning_cells,
        reward_breakdown={key: round(value, 4) for key, value in sorted(breakdown.items())},
    )


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    """Small dependency-free Pearson correlation."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def build_seed_map(tasks: list[str], explicit_seeds: list[int] | None, seed_sweep: int) -> dict[str, list[int]]:
    """Construct the task -> seeds mapping for the audit run."""
    if explicit_seeds:
        return {task: list(explicit_seeds) for task in tasks}
    return {
        task: [DEFAULT_SEEDS[task] + offset for offset in range(seed_sweep)]
        for task in tasks
    }


def aggregate_policy_results(episodes: list[EpisodeAudit]) -> list[dict]:
    """Compute summary metrics per policy."""
    grouped: dict[str, list[EpisodeAudit]] = defaultdict(list)
    for episode in episodes:
        grouped[episode.policy].append(episode)

    summary: list[dict] = []
    for policy_name, runs in grouped.items():
        reward_components = defaultdict(float)
        for run in runs:
            for key, value in run.reward_breakdown.items():
                reward_components[key] += value

        summary.append(
            {
                "policy": policy_name,
                "episodes": len(runs),
                "avg_reward": round(sum(run.total_reward for run in runs) / len(runs), 4),
                "avg_score": round(sum(run.final_score for run in runs) / len(runs), 4),
                "avg_steps": round(sum(run.steps for run in runs) / len(runs), 2),
                "containment_rate": round(
                    sum(1 for run in runs if run.contained) / len(runs),
                    4,
                ),
                "avg_structures_lost": round(
                    sum(run.structures_lost for run in runs) / len(runs),
                    2,
                ),
                "avg_burned_cells": round(
                    sum(run.burned_cells for run in runs) / len(runs),
                    2,
                ),
                "avg_reward_breakdown": {
                    key: round(value / len(runs), 4)
                    for key, value in sorted(reward_components.items())
                },
            }
        )

    return sorted(summary, key=lambda item: (-item["avg_score"], -item["avg_reward"], item["policy"]))


def rank_alignment(episodes: list[EpisodeAudit]) -> dict:
    """Compare reward ordering vs grader ordering within each scenario."""
    scenarios: dict[tuple[str, int], list[EpisodeAudit]] = defaultdict(list)
    for episode in episodes:
        scenarios[(episode.task_id, episode.seed)].append(episode)

    comparisons = 0
    agreements = 0
    mismatches: list[dict] = []

    for (task_id, seed), runs in sorted(scenarios.items()):
        reward_sorted = sorted(runs, key=lambda item: (-item.total_reward, item.policy))
        score_sorted = sorted(runs, key=lambda item: (-item.final_score, item.policy))

        reward_leaders = [
            run.policy
            for run in reward_sorted
            if abs(run.total_reward - reward_sorted[0].total_reward) <= REWARD_TIE_TOL
        ]
        score_leaders = [
            run.policy
            for run in score_sorted
            if abs(run.final_score - score_sorted[0].final_score) <= SCORE_TIE_TOL
        ]

        if set(reward_leaders).isdisjoint(score_leaders):
            mismatches.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "reward_top": reward_sorted[0].policy,
                    "score_top": score_sorted[0].policy,
                    "reward_leaders": reward_leaders,
                    "score_leaders": score_leaders,
                    "reward_order": [run.policy for run in reward_sorted],
                    "score_order": [run.policy for run in score_sorted],
                }
            )

        for idx, left in enumerate(runs):
            for right in runs[idx + 1 :]:
                reward_diff = left.total_reward - right.total_reward
                score_diff = left.final_score - right.final_score
                if abs(reward_diff) <= REWARD_TIE_TOL or abs(score_diff) <= SCORE_TIE_TOL:
                    continue
                comparisons += 1
                if reward_diff * score_diff > 0:
                    agreements += 1

    return {
        "pairwise_agreement": round(agreements / comparisons, 4) if comparisons else 1.0,
        "agreements": agreements,
        "comparisons": comparisons,
        "top_policy_mismatches": mismatches,
    }


def print_summary(
    tasks: list[str],
    seed_map: dict[str, list[int]],
    policy_summary: list[dict],
    alignment: dict,
    episodes: list[EpisodeAudit],
) -> None:
    """Render a concise human-readable report."""
    print("Reward Audit")
    print("=" * 72)
    print("Tasks and seeds:")
    for task in tasks:
        print(f"  {task}: {seed_map[task]}")

    print("\nPolicy summary:")
    header = (
        f"{'policy':<18} {'avg_reward':>11} {'avg_score':>10} "
        f"{'avg_steps':>10} {'contain%':>10} {'lost':>7} {'burned':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in policy_summary:
        print(
            f"{row['policy']:<18} "
            f"{row['avg_reward']:>11.3f} "
            f"{row['avg_score']:>10.3f} "
            f"{row['avg_steps']:>10.2f} "
            f"{100.0 * row['containment_rate']:>9.1f}% "
            f"{row['avg_structures_lost']:>7.2f} "
            f"{row['avg_burned_cells']:>8.2f}"
        )

    rewards = [episode.total_reward for episode in episodes]
    scores = [episode.final_score for episode in episodes]
    print("\nAlignment:")
    print(f"  reward/score Pearson correlation: {pearson_correlation(rewards, scores):.4f}")
    print(
        "  pairwise ranking agreement: "
        f"{alignment['pairwise_agreement']:.4f} "
        f"({alignment['agreements']}/{alignment['comparisons']})"
    )

    if alignment["top_policy_mismatches"]:
        print("\nScenarios where reward and grader picked different top policies:")
        for mismatch in alignment["top_policy_mismatches"][:8]:
            print(
                f"  {mismatch['task_id']} seed={mismatch['seed']}: "
                f"reward->{mismatch['reward_top']} | score->{mismatch['score_top']}"
            )
    else:
        print("\nNo top-policy mismatches across the audited scenarios.")

    print("\nAverage reward breakdown by policy:")
    for row in policy_summary:
        breakdown = row["avg_reward_breakdown"]
        compact = ", ".join(
            f"{key}={value:+.3f}"
            for key, value in breakdown.items()
            if abs(value) >= 0.001
        )
        print(f"  {row['policy']}: {compact if compact else 'no non-zero components'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit reward/grader alignment for the wildfire environment.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(DEFAULT_SEEDS.keys()),
        default=sorted(DEFAULT_SEEDS.keys()),
        help="Tasks to evaluate.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=sorted(POLICIES.keys()),
        default=list(POLICIES.keys()),
        help="Probe policies to run.",
    )
    parser.add_argument(
        "--seed-sweep",
        type=int,
        default=3,
        help="Number of consecutive seeds per task, starting from that task's default seed.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit seeds to use for every selected task. Overrides --seed-sweep.",
    )
    parser.add_argument(
        "--json-out",
        default="reward_audit_report.json",
        help="Path to save the full JSON report. Use '-' to skip writing a file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = list(args.tasks)
    selected_policies = {name: POLICIES[name] for name in args.policies}
    seed_map = build_seed_map(tasks, args.seeds, max(1, args.seed_sweep))

    episodes: list[EpisodeAudit] = []
    for task in tasks:
        for seed in seed_map[task]:
            for policy_name, policy in selected_policies.items():
                episodes.append(run_episode(policy_name, policy, task, seed))

    policy_summary = aggregate_policy_results(episodes)
    alignment = rank_alignment(episodes)
    report = {
        "tasks": tasks,
        "seed_map": seed_map,
        "policies": list(selected_policies.keys()),
        "policy_summary": policy_summary,
        "alignment": alignment,
        "episodes": [asdict(episode) for episode in episodes],
    }

    print_summary(tasks, seed_map, policy_summary, alignment, episodes)

    if args.json_out != "-":
        output_path = Path(args.json_out)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved JSON report to {output_path}")


if __name__ == "__main__":
    main()

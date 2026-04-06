"""Smoke tests for the OpenEnv-facing wildfire environment wrapper.

Exercises all 6 resource types and the new mission types added in the
resource & mission overhaul.
"""

import pytest

from wildfire_env.models import GridPoint, ResourceAssignment, TargetSpec, WildfireAction
from wildfire_env.server.wildfire_env_environment import WildfireEnvironment


# ---------------------------------------------------------------------------
# Helper: pick a simple deterministic action for a smoke-test step
# ---------------------------------------------------------------------------

def _first_available(observation, resource_type: str):
    """Return the first available unit of a resource type, or None."""
    for unit in observation.fleet_units:
        if unit.resource_type == resource_type and unit.status == "available":
            return unit
    return None


def _fire_target(observation) -> GridPoint | None:
    """Return the position of the first burning cell, if any."""
    if observation.fire_details:
        fire = observation.fire_details[0]
        return GridPoint(row=fire.row, col=fire.col)
    return None


def _heat_target(observation) -> GridPoint | None:
    """Return the position of the first heat-warning cell, if any."""
    if observation.heat_warnings:
        hw = observation.heat_warnings[0]
        return GridPoint(row=hw.row, col=hw.col)
    return None


def pick_action(observation) -> WildfireAction:
    """Choose a simple action, cycling through resource types."""
    assignments: list[ResourceAssignment] = []
    fire = _fire_target(observation)
    heat = _heat_target(observation)

    # 1. Crews → direct_attack on fire
    crew = _first_available(observation, "crews")
    if crew and fire:
        assignments.append(ResourceAssignment(
            unit_id=crew.unit_id,
            mission_type="direct_attack",
            target=TargetSpec(target_kind="point", point=fire),
        ))

    # 2. Engines → direct_attack on fire (or wet_line if no fire)
    engine = _first_available(observation, "engines")
    if engine and fire:
        assignments.append(ResourceAssignment(
            unit_id=engine.unit_id,
            mission_type="direct_attack",
            target=TargetSpec(target_kind="point", point=fire),
        ))

    # 3. Helicopters → water_drop on fire/heat
    heli = _first_available(observation, "helicopters")
    target = fire or heat
    if heli and target:
        assignments.append(ResourceAssignment(
            unit_id=heli.unit_id,
            mission_type="water_drop",
            target=TargetSpec(target_kind="area", center=target, radius=1),
            drop_configuration="salvo",
        ))

    # 4. Air tankers → retardant_drop
    tanker = _first_available(observation, "airtankers")
    if tanker and target:
        assignments.append(ResourceAssignment(
            unit_id=tanker.unit_id,
            mission_type="retardant_drop",
            target=TargetSpec(target_kind="area", center=target, radius=1),
            drop_configuration="salvo",
        ))

    # 5. Dozers → line_construction
    dozer = _first_available(observation, "dozers")
    if dozer and fire:
        assignments.append(ResourceAssignment(
            unit_id=dozer.unit_id,
            mission_type="line_construction",
            target=TargetSpec(
                target_kind="line",
                waypoints=[
                    GridPoint(row=max(0, fire.row - 2), col=fire.col),
                    GridPoint(row=max(0, fire.row - 2), col=min(14, fire.col + 3)),
                ],
            ),
        ))

    # 6. Smokejumpers → direct_attack
    sj = _first_available(observation, "smokejumpers")
    if sj and fire:
        assignments.append(ResourceAssignment(
            unit_id=sj.unit_id,
            mission_type="direct_attack",
            target=TargetSpec(target_kind="point", point=fire),
        ))

    return WildfireAction(assignments=assignments)


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return WildfireEnvironment()


def test_reset_returns_observation(env):
    obs = env.reset()
    assert obs.task_id in ("easy", "medium", "hard")
    assert obs.step == 0
    assert obs.max_steps > 0
    assert len(obs.fleet_units) > 0
    assert obs.done is False


def test_step_returns_observation(env):
    obs = env.reset()
    action = WildfireAction()  # no-op
    obs = env.step(action)
    assert obs.step == 1
    assert obs.reward is not None


def test_all_resource_types_present_easy(env):
    obs = env.reset()  # first reset is "easy"
    resource_types = {u.resource_type for u in obs.fleet_units}
    assert "crews" in resource_types
    assert "engines" in resource_types
    assert "helicopters" in resource_types
    assert "airtankers" in resource_types
    assert "dozers" in resource_types
    assert "smokejumpers" in resource_types


def test_dispatch_all_resource_types(env):
    obs = env.reset()
    action = pick_action(obs)
    obs = env.step(action)
    # Should have active missions if fire was present
    assert obs.step == 1
    # Run a few more steps to ensure units progress through lifecycle
    for _ in range(5):
        action = pick_action(obs)
        obs = env.step(action)


def test_invalid_mission_rejected(env):
    obs = env.reset()
    # Helicopters can't do direct_attack
    heli = _first_available(obs, "helicopters")
    if heli:
        action = WildfireAction(assignments=[
            ResourceAssignment(
                unit_id=heli.unit_id,
                mission_type="direct_attack",
                target=TargetSpec(target_kind="point", point=GridPoint(row=7, col=7)),
            )
        ])
        obs = env.step(action)
        assert obs.last_action_error is not None
        assert "cannot perform" in obs.last_action_error


def test_full_episode_easy(env):
    obs = env.reset()
    steps = 0
    while not obs.done and steps < obs.max_steps:
        action = pick_action(obs)
        obs = env.step(action)
        steps += 1
    assert steps > 0


def test_medium_and_hard_tasks(env):
    """Run through easy/medium/hard to verify all configs load."""
    for _ in range(3):
        obs = env.reset()
        assert obs.task_id in ("easy", "medium", "hard")
        for _ in range(3):
            obs = env.step(pick_action(obs))
            if obs.done:
                break


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    env = WildfireEnvironment()

    for episode_idx in range(3):
        obs = env.reset()
        print(f"\nEpisode {episode_idx + 1}: task={obs.task_id} goal={obs.goal}")
        print(
            f"  Start -> burning={obs.burning_cells}, "
            f"structures_remaining={obs.structures_remaining}, "
            f"resources={obs.resources_remaining}"
        )

        for step_idx in range(obs.max_steps):
            action = pick_action(obs)
            obs = env.step(action)
            n_assignments = len(action.assignments)
            print(
                f"  Step {obs.step:2d}: assignments={n_assignments}, "
                f"reward={obs.reward:+.3f}, burning={obs.burning_cells}, "
                f"lost={obs.structures_lost}, missions={len(obs.active_missions)}, "
                f"error={obs.last_action_error}"
            )
            if obs.done:
                break

        total = env.state.total_reward
        print(f"  Episode done={obs.done} total_reward={total}")


if __name__ == "__main__":
    main()

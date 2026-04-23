import sys
import types
from pathlib import Path

import pytest

import smoke_test
from wildfire_env.models import GridPoint, ResourceAssignment, TargetSpec, WildfireAction
from wildfire_env.server.app import GraderRequest, _grade_episode, _heuristic_action
from wildfire_env.server.terrain import (
    _airbase_candidates,
    _ground_outpost_candidates,
    get_task_config,
)
from wildfire_env.server.wildfire_env_environment import WildfireEnvironment


def test_grader_uses_seeded_structure_inventory() -> None:
    response = _grade_episode(
        GraderRequest(
            task_id="hard",
            step=25,
            max_steps=25,
            structures=[{"priority": 3, "status": "safe"}],
            burned_cells=0,
            burning_cells=0,
        )
    )

    assert response.score == 0.3
    assert response.components["saved_priority"] == 0
    assert response.components["total_priority"] == 7
    assert response.components["reported_structures"] == 1
    assert response.components["expected_structures"] == 4


def test_busy_unit_visibility_handles_dispatched_units() -> None:
    env = WildfireEnvironment()
    try:
        obs = env.reset(task_id="easy")
        unit = next(fleet_unit for fleet_unit in obs.fleet_units if fleet_unit.resource_type == "crews")
        action = WildfireAction(
            assignments=[
                ResourceAssignment(
                    unit_id=unit.unit_id,
                    mission_type="staging",
                    target=TargetSpec(
                        target_kind="point",
                        point=GridPoint(row=7, col=7),
                    ),
                )
            ]
        )

        next_obs = env.step(action)

        assert next_obs.step == 1
        assert any(
            fleet_unit.unit_id == unit.unit_id and fleet_unit.status != "available"
            for fleet_unit in next_obs.fleet_units
        )
    finally:
        env.close()


def test_heuristic_uses_later_available_units_after_busy_units() -> None:
    env = WildfireEnvironment()
    try:
        obs = env.reset(task_id="medium")
        for _ in range(20):
            obs = env.step(_heuristic_action(obs))

            later_available: list[str] = []
            saw_busy_unit = False
            for fleet_unit in obs.fleet_units:
                if fleet_unit.status != "available":
                    saw_busy_unit = True
                elif saw_busy_unit:
                    later_available.append(fleet_unit.unit_id)

            if later_available:
                heuristic = _heuristic_action(obs)
                assigned_units = {assignment.unit_id for assignment in heuristic.assignments}

                assert assigned_units
                assert assigned_units.intersection(later_available)
                return
    finally:
        env.close()

    pytest.fail("Did not encounter a mixed-availability state on the seeded medium task")


def test_hard_outpost_candidates_scale_with_grid_size() -> None:
    hard_config = get_task_config("hard")
    ground_candidates = set(_ground_outpost_candidates(hard_config.grid_size))
    air_candidates = set(_airbase_candidates(hard_config.grid_size))

    assert hard_config.grid_size == 25
    assert (12.0, 0.0) in ground_candidates
    assert (12.0, -5.0) in air_candidates
    assert (7.0, 0.0) not in ground_candidates

    for outpost in hard_config.outposts:
        point = (outpost["row"], outpost["col"])
        if outpost["is_airbase"]:
            assert point in air_candidates
        else:
            assert point in ground_candidates


def test_pyproject_declares_train_extra_dependencies() -> None:
    pyproject_text = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "train = [" in pyproject_text
    for dependency in ("torch", "xgrammar", "unsloth", "peft", "bitsandbytes"):
        assert f'"{dependency}' in pyproject_text


def test_smoke_test_reports_failures_with_cp1252_safe_output(monkeypatch, capsys) -> None:
    monkeypatch.setattr(smoke_test, "_validate_training_modules", lambda: None)

    fake_train_grpo = types.ModuleType("train_grpo")

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def train(_config):
        raise RuntimeError("boom")

    fake_train_grpo.Config = Config
    fake_train_grpo.train = train
    monkeypatch.setitem(sys.modules, "train_grpo", fake_train_grpo)

    with pytest.raises(RuntimeError, match="boom"):
        smoke_test.main()

    stdout = capsys.readouterr().out
    assert "SMOKE TEST FAILED: boom" in stdout
    stdout.encode("cp1252")

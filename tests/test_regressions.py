import sys
import types
from pathlib import Path

import pytest

from eval_policy import find_latest_adapter
import smoke_test
from plot_training_curves import (
    MetricSpec,
    build_summary_markdown,
    load_training_log,
    render_svg_dashboard,
)
from submission_check import collect_checks
from wildfire_env.models import GridPoint, ResourceAssignment, TargetSpec, WildfireAction
from wildfire_env.server.app import GRADER_WEIGHTS, GraderRequest, _grade_episode, _heuristic_action
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

    assert response.score == 0.45
    assert response.components["saved_priority"] == 0
    assert response.components["total_priority"] == 8
    assert response.components["reported_structures"] == 1
    assert response.components["expected_structures"] == 5
    assert response.components["weights"] == GRADER_WEIGHTS


def test_grader_rewards_containment_and_spread_control() -> None:
    structures = [
        {"structure_id": f"structure_{idx}", "priority": 1, "status": "structure"}
        for idx in range(1, 5)
    ]
    contained = _grade_episode(
        GraderRequest(
            task_id="easy",
            seed=7,
            step=8,
            max_steps=20,
            structures=structures,
            burned_cells=4,
            burning_cells=0,
        )
    )
    still_spreading = _grade_episode(
        GraderRequest(
            task_id="easy",
            seed=7,
            step=8,
            max_steps=20,
            structures=structures,
            burned_cells=4,
            burning_cells=8,
        )
    )
    high_spread = _grade_episode(
        GraderRequest(
            task_id="easy",
            seed=7,
            step=8,
            max_steps=20,
            structures=structures,
            burned_cells=25,
            burning_cells=8,
        )
    )

    assert 0.0 < high_spread.score < still_spreading.score < contained.score < 1.0
    assert contained.components["containment_component"] > still_spreading.components["containment_component"]
    assert still_spreading.components["spread_limit_component"] > high_spread.components["spread_limit_component"]


def test_grader_exposes_curriculum_scaled_spread_budgets() -> None:
    common = {
        "seed": 7,
        "step": 10,
        "max_steps": 20,
        "structures": [],
        "burned_cells": 10,
        "burning_cells": 1,
    }

    easy = _grade_episode(GraderRequest(task_id="easy", **common))
    medium = _grade_episode(GraderRequest(task_id="medium", **common))
    hard = _grade_episode(GraderRequest(task_id="hard", **common))

    assert easy.components["spread_budget_cells"] < medium.components["spread_budget_cells"]
    assert medium.components["spread_budget_cells"] < hard.components["spread_budget_cells"]


def test_busy_unit_visibility_handles_dispatched_units() -> None:
    env = WildfireEnvironment()
    try:
        obs = env.reset(task_id="easy")
        unit = next(fleet_unit for fleet_unit in obs.fleet_units if fleet_unit.resource_type == "crews")
        action = WildfireAction(
            plan="Stage the nearest crew toward the center for future attack.",
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


def test_action_plan_field_is_optional_and_length_limited() -> None:
    action = WildfireAction(plan="Protect the highest-priority structure first.")

    assert action.plan.startswith("Protect")
    with pytest.raises(Exception):
        WildfireAction(plan="x" * 161)


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


def test_plot_training_curves_generates_svg_and_summary() -> None:
    records = load_training_log(Path("tests/fixtures/sample_train_log.jsonl"))
    svg_text = render_svg_dashboard(
        "Wildfire GRPO Reward Curves",
        records,
        [MetricSpec("mean_return", "Mean Trajectory Return", "#2563eb")],
    )
    summary_text = build_summary_markdown(records)

    assert "<svg" in svg_text
    assert "Mean Trajectory Return" in svg_text
    assert "Training Summary" in summary_text
    assert "| easy | 0 | 0.200 | 1.000 | 90.0% |" in summary_text


def test_submission_check_reports_missing_story_link() -> None:
    fixture_root = Path("tests/fixtures/submission_repo")
    checks = collect_checks(fixture_root, fixture_root / "submission_artifacts")
    by_label = {check.label: check for check in checks}

    assert by_label["README has public Space link"].ok is True
    assert by_label["README links a writeup/video/slides"].ok is False
    assert by_label["README includes training plots/results"].ok is False


def test_find_latest_adapter_prefers_final_then_latest_then_snapshot(monkeypatch) -> None:
    root = Path("grpo_wildfire")
    snapshot = str(root / "adapter_iter0010")
    latest = str(root / "latest")
    final = str(root / "final_adapter")

    monkeypatch.setattr("eval_policy.glob.glob", lambda pattern: [snapshot])
    monkeypatch.setattr("eval_policy.os.path.isdir", lambda path: path in {latest, final})

    assert find_latest_adapter(str(root)) == final

    monkeypatch.setattr("eval_policy.os.path.isdir", lambda path: path == latest)
    assert find_latest_adapter(str(root)) == latest

    monkeypatch.setattr("eval_policy.os.path.isdir", lambda path: False)
    assert find_latest_adapter(str(root)) == snapshot

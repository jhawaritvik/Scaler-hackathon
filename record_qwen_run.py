#!/usr/bin/env python3
"""Export plots, eval outputs, and a run record from a GRPO training directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval_policy import Config, eval_policy, find_latest_adapter
from plot_training_curves import main as plot_training_curves_main


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _resolve_adapter(run_dir: Path, explicit_adapter: str | None) -> str | None:
    if explicit_adapter:
        return explicit_adapter

    final_dir = run_dir / "final_adapter"
    if final_dir.is_dir():
        return str(final_dir)

    latest_dir = run_dir / "latest"
    if latest_dir.is_dir():
        return str(latest_dir)

    return find_latest_adapter(str(run_dir))


def _call_plotter(log_path: Path, artifacts_dir: Path) -> None:
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "plot_training_curves.py",
            "--log",
            str(log_path),
            "--out-dir",
            str(artifacts_dir),
        ]
        plot_training_curves_main()
    finally:
        sys.argv = original_argv


def _build_record(
    *,
    run_dir: Path,
    artifacts_dir: Path,
    adapter_path: str | None,
    trained_eval: dict | None,
    untrained_eval: dict | None,
) -> dict:
    config_json = _read_json(run_dir / "config.json")
    task_catalog = _read_json(run_dir / "task_catalog.json")
    checkpoint_index = _read_json(run_dir / "checkpoint_index.json")
    run_status = _read_json(run_dir / "run_status.json")
    final_summary = _read_json(run_dir / "final_summary.json")

    return {
        "run_dir": str(run_dir.resolve()),
        "artifacts_dir": str(artifacts_dir.resolve()),
        "adapter_evaluated": adapter_path,
        "log_path": str((run_dir / "log.jsonl").resolve()),
        "config": config_json,
        "task_catalog": task_catalog,
        "checkpoint_index": checkpoint_index,
        "run_status": run_status,
        "final_summary": final_summary,
        "artifacts": {
            "reward_curve_png": str((artifacts_dir / "training_reward_curve.png").resolve()),
            "loss_curve_png": str((artifacts_dir / "training_loss_curve.png").resolve()),
            "training_summary_md": str((artifacts_dir / "training_summary.md").resolve()),
            "eval_untrained_json": str((artifacts_dir / "eval_untrained.json").resolve())
            if untrained_eval is not None
            else None,
            "eval_trained_json": str((artifacts_dir / "eval_trained.json").resolve())
            if trained_eval is not None
            else None,
        },
        "evaluation": {
            "untrained": untrained_eval,
            "trained": trained_eval,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="grpo_wildfire", help="Training output directory.")
    parser.add_argument(
        "--artifacts-dir",
        default="submission_artifacts",
        help="Directory for plots, eval JSONs, and the run record.",
    )
    parser.add_argument("--adapter", default=None, help="Explicit adapter path to evaluate.")
    parser.add_argument("--skip-untrained", action="store_true", help="Skip base-model evaluation.")
    parser.add_argument("--skip-trained", action="store_true", help="Skip adapter evaluation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    log_path = run_dir / "log.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(f"Training log not found at {log_path}")

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    _call_plotter(log_path, artifacts_dir)

    config = Config()
    config_json = _read_json(run_dir / "config.json")
    if config_json:
        config = Config(**{**vars(config), **config_json})

    untrained_eval = None
    if not args.skip_untrained:
        untrained_path = artifacts_dir / "eval_untrained.json"
        untrained_eval = eval_policy(
            adapter_path=None,
            output_json=str(untrained_path),
            config=config,
        )

    trained_eval = None
    adapter_path = _resolve_adapter(run_dir, args.adapter)
    if not args.skip_trained:
        if adapter_path is None:
            raise FileNotFoundError(
                "No adapter found. Expected final_adapter/, latest/, or adapter_iterXXXX in the run dir."
            )
        trained_path = artifacts_dir / "eval_trained.json"
        trained_eval = eval_policy(
            adapter_path=adapter_path,
            output_json=str(trained_path),
            config=config,
        )

    record = _build_record(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        adapter_path=adapter_path,
        trained_eval=trained_eval,
        untrained_eval=untrained_eval,
    )
    _write_json(artifacts_dir / "run_record.json", record)
    print(f"Wrote {artifacts_dir / 'run_record.json'}")


if __name__ == "__main__":
    main()

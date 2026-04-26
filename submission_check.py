#!/usr/bin/env python3
"""Check whether the repo has the artifacts needed for hackathon submission.

The checker is intentionally lightweight: it verifies the presence of the
environment, training, evaluation, audit, and presentation assets that judges
will look for. Use ``--strict`` before the final push so missing items cause a
non-zero exit code.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    label: str
    ok: bool
    detail: str


SPACE_PATTERN = re.compile(r"https://huggingface\.co/spaces/[^\s)]+")
TRAINING_NOTEBOOK_PATTERN = re.compile(r"wildfire_grpo_train_hf\.ipynb")
EVAL_NOTEBOOK_PATTERN = re.compile(r"wildfire_eval_plots_hf\.ipynb")
PLOT_IMAGE_PATTERNS = {
    "reward": re.compile(r"!\[[^\]]*\]\([^)]+training_reward_curve\.(png|jpg|jpeg)\)", re.IGNORECASE),
    "loss": re.compile(r"!\[[^\]]*\]\([^)]+training_loss_curve\.(png|jpg|jpeg)\)", re.IGNORECASE),
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def collect_checks(repo_root: Path, artifacts_dir: Path) -> list[CheckResult]:
    readme = _read_text(repo_root / "README.md")

    required_files = [
        ("OpenEnv manifest", repo_root / "openenv.yaml"),
        ("Training script", repo_root / "train_grpo.py"),
        ("OpenEnv showcase evaluation script", repo_root / "eval_policy_http.py"),
        ("Reward audit", repo_root / "reward_audit.py"),
        ("Separate Blog.MD writeup", repo_root / "Blog.MD"),
        ("HF training notebook (GRPO only)", repo_root / "notebooks" / "wildfire_grpo_train_hf.ipynb"),
        ("HF eval + plots notebook", repo_root / "notebooks" / "wildfire_eval_plots_hf.ipynb"),
    ]

    checks = [
        CheckResult(label, path.exists(), str(path.relative_to(repo_root)) if path.exists() else f"missing: {path.name}")
        for label, path in required_files
    ]

    checks.append(
        CheckResult(
            "README has public Space link",
            bool(SPACE_PATTERN.search(readme)),
            "found" if SPACE_PATTERN.search(readme) else "add the HF Space URL to README.md",
        )
    )
    checks.append(
        CheckResult(
            "README links the training notebook",
            bool(TRAINING_NOTEBOOK_PATTERN.search(readme)),
            "found" if TRAINING_NOTEBOOK_PATTERN.search(readme) else "link notebooks/wildfire_grpo_train_hf.ipynb from README.md",
        )
    )
    checks.append(
        CheckResult(
            "README links the eval + plots notebook",
            bool(EVAL_NOTEBOOK_PATTERN.search(readme)),
            "found" if EVAL_NOTEBOOK_PATTERN.search(readme) else "link notebooks/wildfire_eval_plots_hf.ipynb from README.md",
        )
    )
    checks.append(
        CheckResult(
            "README links a writeup/demo",
            "blog.md" in readme.lower(),
            "found" if "blog.md" in readme.lower() else "link Blog.MD from README.md",
        )
    )
    checks.append(
        CheckResult(
            "README includes training plots/results",
            "to be added after training completes" not in readme.lower(),
            "found" if "to be added after training completes" not in readme.lower() else "replace the placeholder with generated plots/results",
        )
    )
    reward_curve = artifacts_dir / "training_reward_curve.png"
    loss_curve = artifacts_dir / "training_loss_curve.png"
    if reward_curve.exists() and loss_curve.exists():
        checks.append(
            CheckResult(
                "README embeds reward/loss plot images",
                all(pattern.search(readme) for pattern in PLOT_IMAGE_PATTERNS.values()),
                "found"
                if all(pattern.search(readme) for pattern in PLOT_IMAGE_PATTERNS.values())
                else "embed training_reward_curve.png and training_loss_curve.png inline in README.md",
            )
        )
    else:
        checks.append(
            CheckResult(
                "README embeds reward/loss plot images",
                True,
                "deferred until training plots are generated",
            )
        )

    artifact_files = [
        ("Reward audit JSON", repo_root / "reward_audit.json"),
        ("Training reward curve image", artifacts_dir / "training_reward_curve.png"),
        ("Training loss curve image", artifacts_dir / "training_loss_curve.png"),
        ("Training summary", artifacts_dir / "training_summary.md"),
        ("Untrained eval JSON", artifacts_dir / "eval_untrained.json"),
        ("Trained eval JSON", artifacts_dir / "eval_trained.json"),
    ]
    checks.extend(
        CheckResult(label, path.exists(), str(path.relative_to(repo_root)) if path.exists() else f"missing: {path}")
        for label, path in artifact_files
    )

    return checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify hackathon submission readiness.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root to inspect.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("submission_artifacts"),
        help="Directory that should contain generated training/eval artifacts.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any check fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifacts_dir = (repo_root / args.artifacts_dir).resolve()
    checks = collect_checks(repo_root, artifacts_dir)

    print("Hackathon submission readiness")
    print("=" * 32)
    failures = 0
    for check in checks:
        status = "OK" if check.ok else "MISSING"
        print(f"[{status:<7}] {check.label}: {check.detail}")
        failures += int(not check.ok)

    if failures:
        print(f"\n{failures} check(s) still need attention.")
    else:
        print("\nAll tracked submission checks passed.")

    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

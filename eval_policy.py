#!/usr/bin/env python3
"""
Evaluate a trained LoRA adapter on the wildfire environment.

Runs 5 eval episodes per task (easy / medium / hard) with fixed seeds,
reports per-task grader scores, and saves results.json.

Usage:
    python eval_policy.py                          # auto-finds latest adapter in ./grpo_wildfire/
    python eval_policy.py --adapter ./grpo_wildfire/adapter_iter0030
    python eval_policy.py --untrained              # base model, no adapter (zero-shot baseline)
    python eval_policy.py --output my_results.json

Verification:
    Run --untrained first and compare to inference.py scores.
    If they differ by > 0.1 absolute, the eval harness has a bug — fix before
    claiming training worked.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wildfire_env.models import WildfireAction
from train_grpo import (
    Config,
    Trajectory,
    build_grammar_compiler,
    rollout_episode,
)

import xgrammar as xgr


# Fixed seeds: same set across all runs so eval numbers are reproducible.
# Fully held-out from train_grpo.py:Config.seeds_per_task — no overlap. All
# seeds verified to produce heuristic grader in 0.2-0.85 (meaningful signal).
EVAL_TASKS = {
    "easy":   [11, 18, 25, 56, 76],
    "medium": [1, 7, 28, 61, 69],
    "hard":   [9, 28, 32, 61, 75],
}


def find_latest_adapter(output_dir: str) -> str | None:
    checkpoints = sorted(glob.glob(os.path.join(output_dir, "adapter_iter*")))
    return checkpoints[-1] if checkpoints else None


def load_model_for_eval(adapter_path: str | None, config: Config, device: torch.device):
    """
    Load base model and optionally attach a trained LoRA adapter.

    If adapter_path is None, evaluates the unmodified base model
    (zero-shot baseline — should match inference.py scores within ~0.05).
    """
    import unsloth  # noqa: F401, PLC0415
    from unsloth import FastLanguageModel  # noqa: PLC0415

    print(f"Loading base model {config.model_name} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        # Match train_grpo.py: 25x25 hard observations + action_guide approach
        # 2k tokens; 2048 silently truncates and tanks the eval grader score.
        max_seq_length=3072,
        dtype=None,
        load_in_4bit=True,
        fast_inference=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        print(f"Loading adapter from {adapter_path} …")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print("  Adapter loaded.")
    else:
        print("  No adapter — evaluating base model (zero-shot).")

    model.eval()
    return model, tokenizer


def eval_policy(
    adapter_path: str | None = None,
    output_json: str = "results.json",
    config: Config | None = None,
) -> dict:
    if config is None:
        config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model_for_eval(adapter_path, config, device)
    compiler = build_grammar_compiler(tokenizer, model)
    compiled_grammar = compiler.compile_json_schema(WildfireAction.model_json_schema())

    results: dict = {}
    all_scores: list[float] = []

    for task_id, seeds in EVAL_TASKS.items():
        print(f"\n── {task_id} ──")
        task_scores: list[float] = []

        for seed in seeds:
            # Use a deterministic but separate sampling_seed so rollouts are
            # reproducible and distinct from training seeds.
            sampling_seed = seed + 777_000
            traj: Trajectory = rollout_episode(
                model, tokenizer, compiled_grammar,
                task_id=task_id,
                seed=seed,
                sampling_seed=sampling_seed,
                config=config,
                device=device,
            )
            score = traj.grader_score
            task_scores.append(score)
            print(
                f"  seed={seed:4d}  grader={score:.4f}  "
                f"steps={traj.episode_steps:2d}  "
                f"return={traj.total_return:.3f}  "
                f"parse={traj.action_parse_successes}/{traj.episode_steps}"
            )

        task_mean = float(np.mean(task_scores))
        task_std  = float(np.std(task_scores))
        results[task_id] = {
            "scores": task_scores,
            "mean":   task_mean,
            "std":    task_std,
            "min":    float(np.min(task_scores)),
            "max":    float(np.max(task_scores)),
        }
        all_scores.extend(task_scores)
        print(f"  {task_id}: mean={task_mean:.4f} ± {task_std:.4f}")

    results["overall_mean"] = float(np.mean(all_scores))
    results["adapter_path"] = adapter_path or "base_model (zero-shot)"

    with open(output_json, "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nOverall mean grader score: {results['overall_mean']:.4f}")
    print(f"Results saved → {output_json}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a GRPO-trained wildfire policy")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to saved LoRA adapter directory")
    parser.add_argument("--untrained", action="store_true",
                        help="Evaluate base model without any adapter (zero-shot baseline)")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Path for output JSON (default: results.json)")
    args = parser.parse_args()

    if args.untrained:
        adapter = None
    elif args.adapter:
        adapter = args.adapter
    else:
        adapter = find_latest_adapter(Config().output_dir)
        if adapter:
            print(f"Auto-selected latest adapter: {adapter}")
        else:
            print("No adapter found in ./grpo_wildfire/ — evaluating base model")
            adapter = None

    eval_policy(adapter_path=adapter, output_json=args.output)

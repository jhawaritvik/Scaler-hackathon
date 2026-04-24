#!/usr/bin/env python3
"""
Smoke test: one GRPO iteration with tiny config (~5 minutes on T4).

Run this FIRST before committing to a full curriculum training run.
A clean exit (no crash, no NaN/inf in loss) means the pipeline is functional.

Usage:
    python smoke_test.py
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import time

# Must be set before torch is imported - reduces CUDA memory fragmentation.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_REQUIRED_TRAINING_MODULES = ("torch", "xgrammar", "peft", "unsloth")


def _validate_training_modules() -> None:
    missing: list[str] = []
    for module_name in _REQUIRED_TRAINING_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)

    if missing:
        joined = ", ".join(missing)
        raise ModuleNotFoundError(
            "Missing training dependencies: "
            f"{joined}. Install them with `pip install -e .[train]` "
            "or `uv pip install -e .[train]`."
        )

    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GRPO smoke test requires a CUDA GPU. This environment is CPU-only, "
            "so the training stack cannot be validated end to end here."
        )


def main():
    _validate_training_modules()

    from train_grpo import Config, train

    config = Config(
        group_size=2,          # 2 trajectories (instead of 8)
        max_episode_steps=5,   # 5 steps per episode (instead of 20)
        total_iterations=1,    # 1 iteration only
        inner_epochs=2,        # 2 inner epochs (instead of 4)
        micro_batch_size=2,    # smaller micro-batch
        output_dir="./smoke_test_output",
    )

    print("=" * 60)
    print("GRPO Wildfire Smoke Test")
    print(f"  group_size={config.group_size}  max_steps={config.max_episode_steps}")
    print(f"  iterations={config.total_iterations}  inner_epochs={config.inner_epochs}")
    print("=" * 60)

    t0 = time.time()
    try:
        train(config)
    except Exception as exc:
        print(f"\nSMOKE TEST FAILED: {exc}")
        raise

    elapsed = time.time() - t0

    # Verify log.jsonl was written with a valid entry
    log_path = os.path.join(config.output_dir, "log.jsonl")
    assert os.path.exists(log_path), "log.jsonl not created"
    with open(log_path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    assert lines, "log.jsonl is empty"
    entry = None
    for ln in lines:
        try:
            entry = __import__("json").loads(ln)
            break
        except Exception:
            continue
    assert entry is not None, "no valid JSON entry in log.jsonl"

    # Check for NaN/inf in key metrics
    for key in ("policy_loss", "kl_divergence", "mean_return"):
        val = entry.get(key, None)
        if val is not None:
            import math
            assert not math.isnan(val), f"{key} is NaN"
            assert not math.isinf(val), f"{key} is inf"

    print(f"\n{'=' * 60}")
    print(f"Smoke test PASSED in {elapsed:.1f}s")
    print("Pipeline is functional. Run train_grpo.py for full training.")
    print("=" * 60)


if __name__ == "__main__":
    main()

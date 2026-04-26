#!/usr/bin/env python3
"""Evaluate a wildfire policy through the official OpenEnv API.

Despite the legacy filename (``eval_policy_http.py`` is referenced in the
README, notebooks, and ``submission_check.py``), this evaluator uses the
official OpenEnv ``EnvClient`` over a *persistent WebSocket session*:

    POST  /grader   (raw HTTP — wildfire-specific endpoint, not part of OpenEnv)
    WS    /ws       (reset / step / state — the canonical OpenEnv transport)

Why WebSocket and not raw POST /reset, /step?  The HTTP transport in
``openenv.core.env_server.http_server`` is intentionally *stateless* — every
``POST /step`` constructs a fresh ``Environment`` instance, calls ``step``
once, and tears it down (see ``HTTPEnvServer.register_routes``).  That makes
it impossible to run multi-turn rollouts over raw HTTP, which is exactly
what an RL evaluation needs.  The WebSocket session keeps the environment
alive across ``reset → step → step → ...`` — the contract every OpenEnv
example documents and the contract our ``WildfireEnv`` client already
implements.

Episode termination is governed by the environment itself
(``observation.done`` or ``observation.step >= observation.max_steps``)
rather than an arbitrary CLI cap, so delayed ignitions actually have time to
fire.  ``--max-episode-steps`` is preserved as an *upper-bound* override for
deadline-driven smoke runs only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train_grpo import Config, build_grammar_compiler, _build_messages
from wildfire_env.client import WildfireEnv
from wildfire_env.models import WildfireAction, WildfireObservation


# Fixed held-out seeds. These do not overlap with train_grpo.py:Config.seeds_per_task.
EVAL_TASKS = {
    "easy": [11, 18, 25, 56, 76],
    "medium": [1, 7, 28, 61, 69],
    "hard": [9, 28, 32, 61, 75],
}


def find_latest_adapter(output_dir: str) -> str | None:
    final_dir = os.path.join(output_dir, "final_adapter")
    if os.path.isdir(final_dir):
        return final_dir
    latest_dir = os.path.join(output_dir, "latest")
    if os.path.isdir(latest_dir):
        return latest_dir
    snapshots = sorted(
        os.path.join(output_dir, name)
        for name in os.listdir(output_dir)
        if name.startswith("adapter_iter") and os.path.isdir(os.path.join(output_dir, name))
    ) if os.path.isdir(output_dir) else []
    return snapshots[-1] if snapshots else None


def load_model_for_eval(adapter_path: str | None, config: Config, device: torch.device):
    import unsloth  # noqa: F401, PLC0415
    from unsloth import FastLanguageModel  # noqa: PLC0415

    print(f"Loading base model {config.model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=4096,
        dtype=None,
        load_in_4bit=True,
        fast_inference=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path:
        print(f"Loading adapter from {adapter_path} ...")
        from peft import PeftModel  # noqa: PLC0415

        model = PeftModel.from_pretrained(model, adapter_path)
        print("  Adapter loaded.")
    else:
        print("  No adapter - evaluating base model over OpenEnv WebSocket.")

    model.eval()
    return model, tokenizer


def _http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Raw HTTP helper, used only for the wildfire-specific /grader endpoint."""
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {details}") from exc


def _generate_action(
    model,
    tokenizer,
    compiled_grammar: xgr.CompiledGrammar,
    obs: WildfireObservation,
    config: Config,
    device: torch.device,
) -> tuple[WildfireAction, bool, int, str]:
    messages = _build_messages(obs)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids_t = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    prompt_attn_t = torch.ones_like(prompt_ids_t, device=device)
    prompt_len = prompt_ids_t.shape[1]

    xgr_proc = XGrammarLogitsProcessor(compiled_grammar)
    generate_fn = getattr(model, "_old_generate", None)
    if generate_fn is None and hasattr(model, "base_model"):
        generate_fn = getattr(model.base_model, "_old_generate", None)
    if generate_fn is None:
        generate_fn = model.generate

    model.eval()
    with torch.no_grad():
        gen_out = generate_fn(
            prompt_ids_t,
            attention_mask=prompt_attn_t,
            max_new_tokens=config.max_new_tokens,
            temperature=config.rollout_temperature,
            top_p=config.rollout_top_p,
            top_k=config.rollout_top_k,
            do_sample=True,
            use_cache=True,
            logits_processor=[xgr_proc],
            pad_token_id=tokenizer.eos_token_id,
        )

    completion_ids = gen_out[0, prompt_len:].tolist()
    raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    try:
        return WildfireAction.model_validate_json(raw_text), True, len(completion_ids), raw_text
    except Exception:
        return WildfireAction(), False, len(completion_ids), raw_text


def _grade_final_observation(
    base_url: str,
    obs: WildfireObservation,
    seed: int,
    *,
    timeout: float,
) -> dict[str, Any]:
    return _http_json(
        "POST",
        f"{base_url}/grader",
        {
            "task_id": obs.task_id,
            "seed": seed,
            "step": obs.step,
            "max_steps": obs.max_steps,
            "structures": [structure.model_dump() for structure in obs.structures],
            "burned_cells": obs.burned_cells,
            "burning_cells": obs.burning_cells,
        },
        timeout=timeout,
    )


def _episode_step_cap(env_max_steps: int, override: int | None) -> int:
    """Cap = env's own max_steps, optionally tightened (never widened) by user override."""
    if override is None or override <= 0:
        return env_max_steps
    return min(env_max_steps, override)


def eval_policy_http(
    *,
    base_url: str,
    adapter_path: str | None,
    output_json: str,
    config: Config | None = None,
    seeds_per_task: int | None = None,
    num_shards: int = 1,
    shard_index: int = 0,
    timeout: float = 120.0,
    max_episode_steps_override: int | None = None,
) -> dict[str, Any]:
    if config is None:
        config = Config()
    base_url = base_url.rstrip("/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fail fast if the server is unavailable. /grader is wildfire-specific, but
    # /schema is OpenEnv standard and confirms the server is up.
    _http_json("GET", f"{base_url}/", timeout=timeout)
    _http_json("GET", f"{base_url}/schema", timeout=timeout)

    model, tokenizer = load_model_for_eval(adapter_path, config, device)
    compiler = build_grammar_compiler(tokenizer, model)
    compiled_grammar = compiler.compile_json_schema(WildfireAction.model_json_schema())

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    limited_tasks = {
        task_id: seeds[:seeds_per_task] if seeds_per_task is not None else seeds
        for task_id, seeds in EVAL_TASKS.items()
    }
    flat_pairs = [
        (task_id, seed)
        for task_id, seeds in limited_tasks.items()
        for seed in seeds
    ]
    selected_pairs = [
        pair for index, pair in enumerate(flat_pairs)
        if index % num_shards == shard_index
    ]
    eval_tasks: dict[str, list[int]] = {task_id: [] for task_id in limited_tasks}
    for task_id, seed in selected_pairs:
        eval_tasks[task_id].append(seed)

    results: dict[str, Any] = {
        "base_url": base_url,
        "adapter_path": adapter_path or "base_model (zero-shot)",
        "model_name": config.model_name,
        "eval_tasks": eval_tasks,
        "full_eval_tasks": EVAL_TASKS,
        "seeds_per_task": seeds_per_task or "all",
        "num_shards": num_shards,
        "shard_index": shard_index,
        "transport": "openenv_websocket",
        "max_episode_steps_override": max_episode_steps_override,
    }
    all_scores: list[float] = []

    # Single persistent WebSocket session for the entire eval. The server
    # keeps one Environment instance alive for the lifetime of this session,
    # so reset() starts a new episode and step() advances the same episode.
    with WildfireEnv(base_url=base_url, message_timeout_s=timeout).sync() as client:
        for task_id, seeds in eval_tasks.items():
            print(f"\n── {task_id} over OpenEnv WebSocket ──")
            task_scores: list[float] = []
            task_episodes: list[dict[str, Any]] = []

            for seed in seeds:
                torch.manual_seed(seed + 777_000)
                reset_result = client.reset(task_id=task_id, seed=seed)
                obs = reset_result.observation
                done = bool(reset_result.done)

                step_cap = _episode_step_cap(obs.max_steps, max_episode_steps_override)

                parse_successes = 0
                token_counts: list[int] = []
                raw_failures: list[str] = []
                started = time.time()
                policy_calls = 0

                while not done and obs.step < step_cap:
                    action, parse_ok, token_count, raw_text = _generate_action(
                        model, tokenizer, compiled_grammar, obs, config, device
                    )
                    parse_successes += int(parse_ok)
                    token_counts.append(token_count)
                    if not parse_ok and len(raw_failures) < 3:
                        raw_failures.append(raw_text[:500])

                    step_result = client.step(action)
                    obs = step_result.observation
                    done = bool(step_result.done)
                    policy_calls += 1

                grade = _grade_final_observation(base_url, obs, seed, timeout=timeout)
                score = float(grade["score"])
                task_scores.append(score)
                all_scores.append(score)

                episode = {
                    "seed": seed,
                    "score": score,
                    "steps": obs.step,
                    "step_cap": step_cap,
                    "env_max_steps": obs.max_steps,
                    "policy_calls": policy_calls,
                    "done": done,
                    "parse_successes": parse_successes,
                    "parse_total": len(token_counts),
                    "parse_success_rate": parse_successes / max(1, len(token_counts)),
                    "mean_completion_tokens": float(np.mean(token_counts)) if token_counts else 0.0,
                    "burned_cells": obs.burned_cells,
                    "burning_cells": obs.burning_cells,
                    "structures_remaining": obs.structures_remaining,
                    "structures_lost": obs.structures_lost,
                    "grader_components": grade.get("components", {}),
                    "duration_seconds": time.time() - started,
                    "raw_parse_failures": raw_failures,
                }
                task_episodes.append(episode)
                print(
                    f"  seed={seed:4d}  grader={score:.4f}  "
                    f"steps={obs.step:2d}/{step_cap}  "
                    f"parse={parse_successes}/{max(1, len(token_counts))}  "
                    f"burning={obs.burning_cells}  burned={obs.burned_cells}"
                )

            if task_scores:
                results[task_id] = {
                    "scores": task_scores,
                    "episodes": task_episodes,
                    "mean": float(np.mean(task_scores)),
                    "std": float(np.std(task_scores)),
                    "min": float(np.min(task_scores)),
                    "max": float(np.max(task_scores)),
                }
                print(f"  {task_id}: mean={results[task_id]['mean']:.4f} ± {results[task_id]['std']:.4f}")

    results["overall_mean"] = float(np.mean(all_scores)) if all_scores else 0.0
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
        fh.write("\n")

    print(f"\nOverall OpenEnv mean grader score: {results['overall_mean']:.4f}")
    print(f"Results saved -> {output_json}")
    return results


def merge_result_files(paths: list[str], output_json: str) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "transport": "openenv_websocket",
        "merged_from": paths,
        "eval_tasks": {},
    }
    all_scores: list[float] = []

    for path in paths:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for key in ("base_url", "adapter_path", "model_name", "full_eval_tasks", "seeds_per_task"):
            if key in data and key not in merged:
                merged[key] = data[key]
        for task_id in EVAL_TASKS:
            task_data = data.get(task_id)
            if not task_data:
                continue
            bucket = merged.setdefault(task_id, {"episodes": []})
            bucket["episodes"].extend(task_data.get("episodes", []))

    for task_id in EVAL_TASKS:
        task_data = merged.get(task_id)
        if not task_data:
            continue
        task_data["episodes"].sort(key=lambda item: item["seed"])
        task_data["scores"] = [float(item["score"]) for item in task_data["episodes"]]
        task_data["mean"] = float(np.mean(task_data["scores"]))
        task_data["std"] = float(np.std(task_data["scores"]))
        task_data["min"] = float(np.min(task_data["scores"]))
        task_data["max"] = float(np.max(task_data["scores"]))
        merged["eval_tasks"][task_id] = [int(item["seed"]) for item in task_data["episodes"]]
        all_scores.extend(task_data["scores"])

    merged["overall_mean"] = float(np.mean(all_scores)) if all_scores else 0.0
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
        fh.write("\n")
    print(f"Merged {len(paths)} shard files -> {output_json}")
    return merged


def run_parallel_eval(args: argparse.Namespace) -> None:
    devices = [item.strip() for item in args.parallel_devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("--parallel-devices must list at least one GPU id")

    output_root, output_ext = os.path.splitext(args.output)
    shard_paths = [
        f"{output_root}.shard{index}{output_ext or '.json'}"
        for index in range(len(devices))
    ]
    processes = []
    for index, device_id in enumerate(devices):
        command = [
            sys.executable,
            os.path.abspath(__file__),
            "--base-url", args.base_url,
            "--output", shard_paths[index],
            "--num-shards", str(len(devices)),
            "--shard-index", str(index),
            "--timeout", str(args.timeout),
        ]
        if args.untrained:
            command.append("--untrained")
        elif args.adapter:
            command.extend(["--adapter", args.adapter])
        if args.seeds_per_task is not None:
            command.extend(["--seeds-per-task", str(args.seeds_per_task)])
        if args.max_new_tokens is not None:
            command.extend(["--max-new-tokens", str(args.max_new_tokens)])
        if args.max_episode_steps is not None:
            command.extend(["--max-episode-steps", str(args.max_episode_steps)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device_id
        print(f"Launching shard {index}/{len(devices)} on CUDA_VISIBLE_DEVICES={device_id}")
        processes.append(subprocess.Popen(command, env=env))

    failures = 0
    for process in processes:
        failures += int(process.wait() != 0)
    if failures:
        raise SystemExit(f"{failures} shard process(es) failed")
    merge_result_files(shard_paths, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the wildfire policy through the OpenEnv WebSocket API."
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:8000"),
        help="Base URL for the OpenEnv server (http:// is auto-converted to ws://).",
    )
    parser.add_argument("--adapter", default=None, help="Path to saved LoRA adapter directory.")
    parser.add_argument("--untrained", action="store_true", help="Evaluate base model without an adapter.")
    parser.add_argument(
        "--seeds-per-task",
        type=int,
        default=None,
        help="Limit evaluation to the first N held-out seeds per task for deadline demos.",
    )
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of eval shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index.")
    parser.add_argument(
        "--parallel-devices",
        default=None,
        help="Comma-separated GPU ids. Launches one eval shard per GPU and merges results.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help=(
            "Generation length for evaluation. Default 1024 — required to fit "
            "the trained policy's action JSON (plan + 1-3 assignments). The "
            "trained adapter naturally produces ~240-300 tokens; lower budgets "
            "truncate mid-JSON and fail to parse, biasing scores against the "
            "trained policy."
        ),
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help=(
            "OPTIONAL upper bound on per-episode policy calls. Defaults to the "
            "env's own max_steps (20 for easy/medium, 25 for hard). Use this only "
            "to time-box a smoke run; smaller values cut off delayed ignitions."
        ),
    )
    parser.add_argument(
        "--output",
        default="submission_artifacts/eval_trained.json",
        help="Path for output JSON.",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-message timeout in seconds.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.parallel_devices:
        run_parallel_eval(args)
        raise SystemExit(0)

    if args.untrained:
        selected_adapter = None
    elif args.adapter:
        selected_adapter = args.adapter
    else:
        selected_adapter = find_latest_adapter(Config().output_dir)
        if selected_adapter:
            print(f"Auto-selected latest adapter: {selected_adapter}")
        else:
            print("No adapter found in ./grpo_wildfire/ - evaluating base model")
    config = Config()
    if args.max_new_tokens is not None:
        config.max_new_tokens = args.max_new_tokens
    eval_policy_http(
        base_url=args.base_url,
        adapter_path=selected_adapter,
        output_json=args.output,
        config=config,
        seeds_per_task=args.seeds_per_task,
        timeout=args.timeout,
        max_episode_steps_override=args.max_episode_steps,
    )

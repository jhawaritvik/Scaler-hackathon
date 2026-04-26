#!/usr/bin/env python3
"""Capture an episode's per-step frame data into a JSON file for replay.

Use this to record a single episode (heuristic or trained-LLM policy) for
the submission demo video. The captured JSON is streamable through the
/demo_replay WebSocket and rendered by the /viewer page exactly the same
way as a live /demo run.

Usage:
    # Heuristic baseline capture (no GPU required)
    python capture_replay.py --task hard --seed 9 --policy heuristic \
        --output replays/heuristic_hard_9.json

    # Trained-LLM capture (GPU required)
    python capture_replay.py --task hard --seed 9 --policy llm \
        --adapter grpo_wildfire/best_adapter_hard \
        --output replays/trained_hard_9.json

    # Side-by-side comparison: capture both with the same seed,
    # then play back at the same rate in two browser tabs.

After capture, view in a browser:
    http://localhost:8000/viewer?replay=replays/trained_hard_9.json
"""
from __future__ import annotations

# Must be the very first import so Unsloth can patch transformers before they load.
# Wrapped in try/except so the heuristic capture path (--policy heuristic) stays
# runnable on CPU-only machines that don't have unsloth installed.
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wildfire_env.server.app import _heuristic_action, _grade_episode, GraderRequest
from wildfire_env.server.wildfire_env_environment import WildfireEnvironment
from wildfire_env.models import WildfireAction


def _build_frame(env, obs, plan_text: str = "") -> dict:
    """Match the JSON shape streamed by /demo so /demo_replay can reuse the
    same WebSocket contract and the existing viewer renders without changes."""
    cell_states: list = []
    cell_intensity: list = []
    if env._sim is not None and env._sim.state is not None:
        cell_states = env._sim.state.cell_state.tolist()
        cell_intensity = [
            [round(float(v), 3) for v in row]
            for row in env._sim.state.intensity.tolist()
        ]
    return {
        "step": obs.step,
        "max_steps": obs.max_steps,
        "done": obs.done,
        "task_id": obs.task_id,
        "cell_states": cell_states,
        "cell_intensity": cell_intensity,
        "wind_speed": round(obs.wind_speed, 1),
        "wind_direction": round(obs.wind_direction, 1),
        "temperature": round(obs.temperature, 1),
        "humidity": round(obs.humidity, 2),
        "burning_cells": obs.burning_cells,
        "burned_cells": obs.burned_cells,
        "structures": [s.model_dump() for s in obs.structures],
        "units": [u.model_dump() for u in obs.fleet_units],
        "last_action_summary": (obs.last_action_summary or "")[:200],
        "model_plan": plan_text[:200],
        "weather_forecast": obs.weather_forecast,
        "score": None,
        "components": None,
    }


def _attach_final_score(frames: list[dict], obs, seed: int) -> None:
    if not frames:
        return
    req = GraderRequest(
        task_id=obs.task_id,
        seed=seed,
        step=obs.step,
        max_steps=obs.max_steps,
        structures=[s.model_dump() for s in obs.structures],
        burned_cells=obs.burned_cells,
        burning_cells=obs.burning_cells,
    )
    grade = _grade_episode(req)
    frames[-1]["score"] = grade.score
    frames[-1]["components"] = grade.components


def capture_heuristic(task_id: str, seed: int) -> list[dict]:
    env = WildfireEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    frames = [_build_frame(env, obs)]
    while not obs.done:
        action = _heuristic_action(obs)
        obs = env.step(action)
        frames.append(_build_frame(env, obs, plan_text=action.plan or ""))
    _attach_final_score(frames, obs, seed)
    env.close()
    return frames


def capture_llm(
    task_id: str,
    seed: int,
    adapter_path: str | None,
    model_name: str,
    temperature: float,
    max_new_tokens: int,
) -> list[dict]:
    # Remaining heavy imports deferred until --policy llm is actually used.
    import torch
    import xgrammar as xgr
    from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

    from unsloth import FastLanguageModel
    from train_grpo import build_grammar_compiler, _build_messages

    print(f"Loading {model_name} (4-bit) …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
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

    model.eval()
    device = next(model.parameters()).device

    compiler = build_grammar_compiler(tokenizer, model)
    compiled_grammar = compiler.compile_json_schema(WildfireAction.model_json_schema())

    env = WildfireEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)
    frames = [_build_frame(env, obs)]

    step_idx = 0
    while not obs.done and step_idx < obs.max_steps + 5:
        messages = _build_messages(obs)
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False,
        ).input_ids.to(device)
        prompt_attn = torch.ones_like(prompt_ids, device=device)

        xgr_proc = XGrammarLogitsProcessor(compiled_grammar)
        model.eval()
        generate_fn = getattr(model, "_old_generate", None)
        if generate_fn is None and hasattr(model, "base_model"):
            generate_fn = getattr(model.base_model, "_old_generate", None)
        if generate_fn is None:
            generate_fn = model.generate

        # use_cache=True — same generation path as eval_policy_http.py.
        # 5-10× faster than use_cache=False; necessary for an untrained capture
        # to finish in minutes instead of an hour.
        gen_out = generate_fn(
            prompt_ids,
            attention_mask=prompt_attn,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            do_sample=True,
            use_cache=True,
            logits_processor=[xgr_proc],
            pad_token_id=tokenizer.eos_token_id,
        )
        model.train()
        completion_ids = gen_out[0, prompt_ids.shape[1]:].tolist()
        raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        try:
            action = WildfireAction.model_validate_json(raw_text)
        except Exception:
            action = WildfireAction()

        plan_text = (action.plan or "").strip()
        obs = env.step(action)
        frames.append(_build_frame(env, obs, plan_text=plan_text))
        step_idx += 1

    _attach_final_score(frames, obs, seed)
    env.close()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--policy",
        choices=["heuristic", "llm"],
        default="heuristic",
        help="heuristic = deterministic baseline (no GPU); llm = trained model (GPU required)",
    )
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter (--policy llm only)")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Lower than training (1.1) for more deterministic demo behavior.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", required=True, help="Path for the captured frames JSON")
    args = parser.parse_args()

    if args.policy == "heuristic":
        print(f"Capturing heuristic on task={args.task} seed={args.seed} …")
        frames = capture_heuristic(args.task, args.seed)
    else:
        print(f"Capturing LLM on task={args.task} seed={args.seed} adapter={args.adapter or '(zero-shot)'} …")
        frames = capture_llm(
            args.task,
            args.seed,
            args.adapter,
            args.model,
            args.temperature,
            args.max_new_tokens,
        )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "task": args.task,
        "seed": args.seed,
        "policy": args.policy,
        "adapter": args.adapter,
        "model": args.model if args.policy == "llm" else None,
        "frames": frames,
    }
    with open(args.output, "w") as fh:
        json.dump(payload, fh)

    final = frames[-1]
    print(f"\nCaptured {len(frames)} frames -> {args.output}")
    print(f"  steps: {final['step']}/{final['max_steps']}")
    print(f"  burned: {final['burned_cells']}  burning: {final['burning_cells']}")
    if final.get("score") is not None:
        print(f"  grader score: {final['score']:.4f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Smoke test: validate use_cache=True with the LoRA-adapted Qwen3 across
diverse step depths and difficulty tasks before flipping the production flag.

Stages a real episode by stepping the env with empty actions and capturing
observations at multiple step depths (0, 2, 4, ...). For each captured obs,
runs cache=True generation N times and reports parse rate. If parse rate
holds across all step depths and both easy and hard tasks, the cache path
is safe to enable in eval_policy_http.py.

Run on the HF runtime (or any CUDA box with the trained adapter present):

    python _tmp_cache_smoke.py \
        --base-url https://chunchunmaru-101-wildfire-env.hf.space \
        --adapter grpo_wildfire/final_adapter \
        --num-prompts 2 --max-new-tokens 256

Decision rule: enable use_cache=True only if EVERY step depth on EVERY task
returns parse rate == num_prompts. Any failure (truncation, malformed JSON,
runtime exception) means the cache path is not safe — fall back to cache=False.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from train_grpo import Config, build_grammar_compiler, _build_messages
from wildfire_env.client import WildfireEnv
from wildfire_env.models import WildfireAction, WildfireObservation


def _generate(
    model,
    tokenizer,
    compiled_grammar,
    obs: WildfireObservation,
    config: Config,
    device: torch.device,
    *,
    use_cache: bool,
) -> tuple[bool, int, float, str]:
    messages = _build_messages(obs)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)
    prompt_attn = torch.ones_like(prompt_ids, device=device)
    prompt_len = prompt_ids.shape[1]

    generate_fn = getattr(model, "_old_generate", None)
    if generate_fn is None and hasattr(model, "base_model"):
        generate_fn = getattr(model.base_model, "_old_generate", None)
    if generate_fn is None:
        generate_fn = model.generate

    xgr_proc = XGrammarLogitsProcessor(compiled_grammar)
    t0 = time.time()
    with torch.no_grad():
        gen_out = generate_fn(
            prompt_ids,
            attention_mask=prompt_attn,
            max_new_tokens=config.max_new_tokens,
            temperature=config.rollout_temperature,
            top_p=config.rollout_top_p,
            top_k=config.rollout_top_k,
            do_sample=True,
            use_cache=use_cache,
            logits_processor=[xgr_proc],
            pad_token_id=tokenizer.eos_token_id,
        )
    dt = time.time() - t0

    completion_ids = gen_out[0, prompt_len:].tolist()
    raw = tokenizer.decode(completion_ids, skip_special_tokens=True)
    try:
        WildfireAction.model_validate_json(raw)
        ok = True
    except Exception:
        ok = False
    return ok, len(completion_ids), dt, raw


def _capture_obs_trajectory(
    base_url: str,
    task_id: str,
    seed: int,
    *,
    capture_steps: list[int],
    timeout: float,
) -> list[WildfireObservation]:
    """Step env with empty actions; return obs at each requested step depth."""
    captured: dict[int, WildfireObservation] = {}
    max_step = max(capture_steps)
    with WildfireEnv(base_url=base_url, message_timeout_s=timeout).sync() as client:
        reset_result = client.reset(task_id=task_id, seed=seed)
        obs = reset_result.observation
        if 0 in capture_steps:
            captured[0] = obs
        empty_action = WildfireAction()
        for _ in range(max_step):
            step_result = client.step(empty_action)
            obs = step_result.observation
            if obs.step in capture_steps and obs.step not in captured:
                captured[obs.step] = obs
            if step_result.done:
                break
    return [captured[s] for s in sorted(captured.keys())]


def _run_at_obs(
    label: str,
    *,
    obs: WildfireObservation,
    model,
    tokenizer,
    compiled_grammar,
    config: Config,
    device: torch.device,
    num_prompts: int,
) -> tuple[int, int, float, list[str]]:
    """Run cache=True N times against one obs. Returns (ok_count, total_tokens, total_time, raw_outputs)."""
    print(f"\n  --- {label} (obs.step={obs.step}, "
          f"burning={obs.burning_cells}, fires={len(obs.fire_details)}, "
          f"missions={len(obs.active_missions)}) ---")
    ok_count = 0
    total_tokens = 0
    total_time = 0.0
    raws: list[str] = []
    for i in range(num_prompts):
        try:
            ok, tokens, dt, raw = _generate(
                model, tokenizer, compiled_grammar, obs, config, device,
                use_cache=True,
            )
        except Exception as exc:
            print(f"    [{i+1}/{num_prompts}] EXCEPTION: {type(exc).__name__}: {exc}")
            raws.append(f"<EXCEPTION: {exc}>")
            continue
        ok_count += int(ok)
        total_tokens += tokens
        total_time += dt
        flag = "ok " if ok else "BAD"
        snippet = raw.replace("\n", " ")[:100]
        print(f"    [{i+1}/{num_prompts}] {flag} {tokens:3d}tok in {dt:.1f}s  {snippet!r}")
        raws.append(raw)
    return ok_count, total_tokens, total_time, raws


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--adapter", required=True,
                        help="Path or HF Hub repo id of the trained adapter")
    parser.add_argument("--num-prompts", type=int, default=2,
                        help="Generations per (task, step) cell")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    config = Config()
    config.max_new_tokens = args.max_new_tokens
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_plan = [
        ("easy",  11, [0, 2, 4, 6]),
        ("hard",  9,  [0, 3, 6, 10]),
    ]

    print(f"=== Stage 1: capture obs trajectories from {args.base_url} ===")
    obs_by_task: dict[str, list[WildfireObservation]] = {}
    for task_id, seed, depths in test_plan:
        print(f"  {task_id}/seed={seed} capturing at steps {depths} ...")
        try:
            captured = _capture_obs_trajectory(
                args.base_url, task_id, seed,
                capture_steps=depths, timeout=args.timeout,
            )
        except Exception as exc:
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            captured = []
        obs_by_task[task_id] = captured
        print(f"    captured {len(captured)} obs (steps={[o.step for o in captured]}, "
              f"max_steps={captured[0].max_steps if captured else '?'})")

    if not any(obs_by_task.values()):
        print("\nNo observations captured — cannot proceed.")
        return 1

    print("\n=== Stage 2: load model + adapter ===")
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=3072,
        dtype=None,
        load_in_4bit=True,
        fast_inference=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    compiler = build_grammar_compiler(tokenizer, model)
    compiled_grammar = compiler.compile_json_schema(WildfireAction.model_json_schema())

    print("\n=== Stage 3: run cache=True on each captured obs ===")
    grand_ok = 0
    grand_total = 0
    grand_time = 0.0
    grand_tokens = 0
    per_cell_results: list[dict] = []
    for task_id, observations in obs_by_task.items():
        print(f"\n--- task={task_id} ---")
        for obs in observations:
            label = f"{task_id} step={obs.step}"
            ok, tokens, dt, raws = _run_at_obs(
                label,
                obs=obs, model=model, tokenizer=tokenizer,
                compiled_grammar=compiled_grammar, config=config, device=device,
                num_prompts=args.num_prompts,
            )
            grand_ok += ok
            grand_total += args.num_prompts
            grand_time += dt
            grand_tokens += tokens
            per_cell_results.append({
                "task": task_id,
                "step": obs.step,
                "ok": ok,
                "n": args.num_prompts,
                "avg_s": dt / max(args.num_prompts, 1),
            })

    print("\n=========== FINAL VERDICT ===========")
    print(f"{'task':<8} {'step':<6} {'parse':<8} {'avg_s':<8}")
    print("-" * 36)
    for row in per_cell_results:
        flag = "OK" if row["ok"] == row["n"] else "FAIL"
        print(f"{row['task']:<8} {row['step']:<6} "
              f"{row['ok']}/{row['n']:<5} {row['avg_s']:<8.1f} {flag}")
    print("-" * 36)
    overall = "PASS" if grand_ok == grand_total else "FAIL"
    tok_per_s = grand_tokens / grand_time if grand_time > 0 else 0.0
    print(f"OVERALL: {grand_ok}/{grand_total} parse, {tok_per_s:.1f} tok/s avg  ==> {overall}")
    if overall == "PASS":
        print("\n>>> Safe to enable use_cache=True in eval_policy_http.py")
    else:
        print("\n>>> DO NOT enable use_cache=True. Cache pathway is unreliable on at least one cell.")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

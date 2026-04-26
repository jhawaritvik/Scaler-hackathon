#!/usr/bin/env python3
"""Smoke test: try to make use_cache=True work with the LoRA-adapted Qwen3.

Tests several inference paths and reports parse rate + tokens-per-second
for each. Goal: find a config that matches use_cache=False's parse rate
while being substantially faster.

Hits the env once at the start to capture a real observation, then runs
all inference configs offline against that captured obs — so it only
holds the env's concurrency slot for ~1 second total.

Run on the HF runtime (or any CUDA box with the trained adapter present):

    python _tmp_cache_smoke.py \
        --base-url https://chunchunmaru-101-wildfire-env.hf.space \
        --adapter outputs/grpo_wildfire/final_adapter \
        --task easy --seed 11 --num-prompts 3

Parse rate < num_prompts on a config => that config is broken; do not ship.
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
    use_unsloth_generate: bool,
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

    if use_unsloth_generate:
        generate_fn = model.generate
    else:
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


def _run_config(
    label: str,
    *,
    obs: WildfireObservation,
    model,
    tokenizer,
    compiled_grammar,
    config: Config,
    device: torch.device,
    num_prompts: int,
    use_cache: bool,
    use_unsloth_generate: bool,
) -> dict:
    print(f"\n=== {label} (use_cache={use_cache}, unsloth_gen={use_unsloth_generate}) ===")
    ok_count = 0
    total_tokens = 0
    total_time = 0.0
    for i in range(num_prompts):
        try:
            ok, tokens, dt, raw = _generate(
                model, tokenizer, compiled_grammar, obs, config, device,
                use_cache=use_cache, use_unsloth_generate=use_unsloth_generate,
            )
        except Exception as exc:
            print(f"  [{i+1}/{num_prompts}] EXCEPTION: {type(exc).__name__}: {exc}")
            return {"label": label, "ok_count": 0, "num_prompts": num_prompts,
                    "tok_per_s": 0.0, "error": str(exc)}
        ok_count += int(ok)
        total_tokens += tokens
        total_time += dt
        flag = "ok" if ok else "BAD"
        snippet = raw.replace("\n", " ")[:120]
        print(f"  [{i+1}/{num_prompts}] {flag} {tokens}tok in {dt:.1f}s  raw={snippet!r}")
    tok_per_s = total_tokens / total_time if total_time > 0 else 0.0
    print(f"  >>> parse {ok_count}/{num_prompts}, {tok_per_s:.1f} tok/s, "
          f"{total_time/num_prompts:.1f}s/call avg")
    return {"label": label, "ok_count": ok_count, "num_prompts": num_prompts,
            "tok_per_s": tok_per_s, "avg_s": total_time / num_prompts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--adapter", default="outputs/grpo_wildfire/final_adapter")
    parser.add_argument("--task", default="easy")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-prompts", type=int, default=3)
    args = parser.parse_args()

    config = Config()
    config.max_new_tokens = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Pulling one observation from {args.base_url} task={args.task} seed={args.seed} ...")
    with WildfireEnv(base_url=args.base_url, message_timeout_s=60.0).sync() as client:
        reset_result = client.reset(task_id=args.task, seed=args.seed)
        obs = reset_result.observation
    print(f"  obs.step={obs.step} obs.max_steps={obs.max_steps} burning={obs.burning_cells}")

    print("\nLoading base model + adapter ...")
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

    summary: list[dict] = []

    summary.append(_run_config(
        "A: _old_generate + cache=False (current baseline)",
        obs=obs, model=model, tokenizer=tokenizer, compiled_grammar=compiled_grammar,
        config=config, device=device, num_prompts=args.num_prompts,
        use_cache=False, use_unsloth_generate=False,
    ))

    summary.append(_run_config(
        "B: _old_generate + cache=True (known broken)",
        obs=obs, model=model, tokenizer=tokenizer, compiled_grammar=compiled_grammar,
        config=config, device=device, num_prompts=args.num_prompts,
        use_cache=True, use_unsloth_generate=False,
    ))

    print("\nFix C: PEFT merge_adapter() then _old_generate + cache=True")
    try:
        model.merge_adapter()
        summary.append(_run_config(
            "C: merge_adapter + _old_generate + cache=True",
            obs=obs, model=model, tokenizer=tokenizer, compiled_grammar=compiled_grammar,
            config=config, device=device, num_prompts=args.num_prompts,
            use_cache=True, use_unsloth_generate=False,
        ))
    except Exception as exc:
        print(f"  merge_adapter() failed: {type(exc).__name__}: {exc}")
        summary.append({"label": "C: merge_adapter", "ok_count": 0,
                        "num_prompts": args.num_prompts, "error": str(exc)})
    finally:
        try:
            model.unmerge_adapter()
        except Exception:
            pass

    print("\nFix D: FastLanguageModel.for_inference + Unsloth model.generate + cache=True")
    try:
        FastLanguageModel.for_inference(model)
        summary.append(_run_config(
            "D: for_inference + Unsloth generate + cache=True",
            obs=obs, model=model, tokenizer=tokenizer, compiled_grammar=compiled_grammar,
            config=config, device=device, num_prompts=args.num_prompts,
            use_cache=True, use_unsloth_generate=True,
        ))
    except Exception as exc:
        print(f"  for_inference path failed: {type(exc).__name__}: {exc}")
        summary.append({"label": "D: for_inference", "ok_count": 0,
                        "num_prompts": args.num_prompts, "error": str(exc)})

    print("\n=========== SUMMARY ===========")
    for row in summary:
        if "error" in row:
            print(f"  {row['label']}: ERROR — {row['error']}")
            continue
        print(f"  {row['label']}: parse {row['ok_count']}/{row['num_prompts']}, "
              f"{row['tok_per_s']:.1f} tok/s, {row['avg_s']:.1f}s/call avg")
    print("\nWinner = highest tok/s with parse == num_prompts.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

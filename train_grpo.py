#!/usr/bin/env python3
"""
Multi-turn GRPO training pipeline for the wildfire incident-command environment.

Target hardware:
  - T4 (16 GB, Colab free tier): use Qwen/Qwen3-1.7B  ← default model_name
  - On-site / A100 (40+ GB):     use Qwen/Qwen3-4B (change model_name in Config)

Expected peak VRAM for Qwen3-1.7B on T4:
  - Base model (4-bit NF4 QLoRA):              ~1.1 GB
  - LoRA adapters (rank=16, attn only):         ~0.05 GB
  - Gradient-checkpointed backward pass
    (one seq at a time in _logprobs_for_batch): ~2-4 GB
  - AdamW 8-bit optimizer states:               ~0.2 GB
  - XGrammar compiled grammar + bitmask cache:  ~0.2 GB
  Total (conservative):                         ~5-7 GB  ← comfortable on T4

Expected peak VRAM for Qwen3-4B on A100 (40 GB):
  - Base model (4-bit NF4 QLoRA):              ~2.5 GB
  - Same breakdown above scaled up:            ~10-14 GB total

Architecture notes:
  - fast_inference=False is required so model.generate() accepts HF LogitsProcessors.
    XGrammarLogitsProcessor is a transformers.LogitsProcessor; vLLM ignores it.
    Speed cost vs fast_inference=True: ~3-5× slower rollout on T4 — still viable.
  - GRPO loss is hand-rolled.  TRL's GRPOTrainer assumes single-turn rollouts;
    multi-turn trajectory-level advantages require a custom loop.
  - Reference policy = base model with LoRA disabled (model.disable_adapter()).
    This avoids loading a second copy of the weights.
  - old_logprobs are computed by a no-grad forward pass after generation
    (not from output_scores) to get unconstrained logprobs consistent with
    the forward pass used during the update.
"""
from __future__ import annotations

import json
import os

# Reduce CUDA memory fragmentation — must be set before torch is imported.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from wildfire_env.server.wildfire_env_environment import WildfireEnvironment
from wildfire_env.server.app import GraderRequest, _grade_episode
from wildfire_env.models import WildfireAction, WildfireObservation

# Third-party — install: pip install unsloth xgrammar bitsandbytes
# xgrammar is imported at module level (lightweight, no CUDA required).
# unsloth and bitsandbytes are imported lazily inside load_model() so that
# the rest of the module (env, advantages, loss math) is importable on CPU
# environments without a GPU/unsloth installation.
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-1.7B"
    task_curriculum: tuple = (("medium", 0, 30), ("hard", 30, 60))
    seeds_per_task: dict = field(default_factory=lambda: {
        "easy":   [42, 100, 200, 300],
        "medium": [67, 101, 201, 131],   # replaced 301 (dead-zone); 131 heuristic ≈ 0.38, aligned with pool
        "hard":   [12, 102, 202, 302],
    })
    group_size: int = 4
    seeds_per_iter: int = 2         # base seeds sampled per iter; group_size split across them
    inner_epochs: int = 4
    micro_batch_size: int = 1       # sequences per optimizer.step(); each seq backprops alone
    max_episode_steps: int = 25       # match hard task horizon (medium still stops at its own max_steps=20)
    max_new_tokens: int = 256       # max tokens generated per action
    learning_rate: float = 3e-5
    lr_min: float = 5e-6            # cosine schedule end LR
    warmup_iters: int = 5           # linear warmup before cosine — kills iter-0 loss spike
    weight_decay: float = 0.01      # AdamW weight decay (mild regularizer on LoRA params)
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05      # dropout on the LoRA path — primary overfit defense
    # Qwen3 uses standard transformer attention: q/k/v/o projections are correct targets.
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    kl_coef: float = 0.08           # raised from 0.04 — stronger anchor to ref policy
    clip_range: float = 0.2
    total_iterations: int = 60
    rollout_temperature: float = 1.1
    rollout_top_p: float = 0.95
    rollout_top_k: int = 50
    grader_return_weight: float = 10.0     # grader contribution to total_return
    vary_env_seed_in_group: bool = True    # inject env variance across group members
    output_dir: str = "./grpo_wildfire"


# ---------------------------------------------------------------------------
# System prompt (mirrors inference.py; redefined here to avoid that
# module's HF_TOKEN check at import time)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI wildfire incident commander. Dispatch firefighting resources to contain fires and protect structures on a terrain grid (easy/medium: 15×15, hard: 25×25).

Each step you receive a JSON observation and must respond with a JSON action.

ACTION FORMAT (respond with valid JSON only, no markdown):
{
  "assignments": [
    {
      "unit_id": "crew_1",
      "mission_type": "direct_attack",
      "target": {
        "target_kind": "point",
        "point": {"row": 7, "col": 8}
      },
      "commitment_steps": 1
    }
  ]
}

TARGET KINDS:
  point     -> {"point": {"row": R, "col": C}}
  line      -> {"waypoints": [{"row": R1, "col": C1}, {"row": R2, "col": C2}]}
  area      -> {"center": {"row": R, "col": C}, "radius": N}
  structure -> {"structure_id": "structure_1"}

RESOURCE CAPABILITIES (from action_guide in observation):
  crews        -> direct_attack | line_construction | point_protection | backfire | staging
  engines      -> direct_attack | wet_line | point_protection | backfire | staging
  helicopters  -> water_drop | point_protection | staging (+drop_configuration: salvo/trail)
  airtankers   -> retardant_drop | staging (+drop_configuration: salvo/trail)
  dozers       -> line_construction | point_protection | staging
  smokejumpers -> direct_attack | line_construction | point_protection | staging

STRATEGY:
- Fire spreads every step - act immediately on available resources.
- Use action_guide to see which units are available and their valid missions.
- Protect high-priority structures (priority 3 > 2 > 1) first.
- Dozers build permanent firebreaks; engines create faster but temporary wet lines.
- Helicopters near water bodies reload faster (water_scoop mechanic).
- Backfire requires a firebreak, water, burned, or bare-ground anchor cell.
- If no fire is present, stage resources near likely ignition zones.
- You may issue multiple assignments per step.

OBSERVATION SIGNALS:
- forecast: next 1-2 steps of wind/temp/humidity — position units upwind of projected spread.
- visibility.fog_of_war_active: when true, fire_details/heat_warnings only cover cells within
  sensor range of deployed units. burning_cells (total count) is still unmasked.
  Deploy helicopters/airtankers early to scout unseen terrain (they have the largest sensor radius).

Return ONLY valid JSON. No explanation, no code blocks."""


def _build_user_prompt(obs: WildfireObservation) -> str:
    """Compact prompt that omits elevation/fuel_type grids (large, low-signal)."""
    return json.dumps(
        {
            "step": obs.step,
            "max_steps": obs.max_steps,
            "goal": obs.goal,
            "action_guide": obs.action_guide,
            "fire": {
                "burning_cells": obs.burning_cells,
                "burned_cells": obs.burned_cells,
                "details": [
                    {"row": f.row, "col": f.col, "intensity": round(f.intensity, 2)}
                    for f in sorted(obs.fire_details, key=lambda f: -f.intensity)[:5]
                ],
                "heat_warnings": [
                    {"row": h.row, "col": h.col}
                    for h in obs.heat_warnings[:3]
                ],
            },
            "atmosphere": {
                "wind_speed_kmh": round(obs.wind_speed, 1),
                "wind_direction_deg": round(obs.wind_direction, 1),
                "temperature_c": round(obs.temperature, 1),
                "humidity": round(obs.humidity, 2),
            },
            "forecast": [
                {
                    "step": fc.get("step"),
                    "minutes_ahead": fc.get("minutes_ahead"),
                    "temp_c": round(float(fc.get("temperature", 0.0)), 1),
                    "humidity": round(float(fc.get("humidity", 0.0)), 2),
                    "wind_speed_kmh": round(float(fc.get("wind_speed_expected", 0.0)), 1),
                    "wind_direction_deg": round(float(fc.get("wind_direction_current", 0.0)), 1),
                }
                for fc in (obs.weather_forecast or [])[:2]
            ],
            "visibility": {
                "fog_of_war_active": obs.fog_of_war_active,
                "visible_cell_count": obs.visible_cell_count,
            },
            "structures": [
                {
                    "id": s.structure_id,
                    "row": s.row,
                    "col": s.col,
                    "priority": s.priority,
                    "status": s.status,
                }
                for s in obs.structures
            ],
        },
        indent=2,
    )


def _build_messages(obs: WildfireObservation) -> list[dict]:
    user_content = (
        f"Current observation:\n{_build_user_prompt(obs)}\n\nRespond with your action JSON:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Trajectory dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StepData:
    """Data for one (obs → action) step in a trajectory."""
    full_seq: list[int]        # prompt_ids + completion_ids concatenated
    prompt_len: int
    comp_len: int
    old_logprobs: list[float]  # unconstrained logprobs, shape (comp_len,)
    step_reward: float
    parse_success: bool


@dataclass
class Trajectory:
    steps: list[StepData]
    task_id: str
    seed: int
    sampling_seed: int
    total_return: float        # sum(step_rewards) + 2.0 * grader_score
    grader_score: float
    episode_steps: int
    action_parse_successes: int


# ---------------------------------------------------------------------------
# XGrammar helpers
# ---------------------------------------------------------------------------

def build_grammar_compiler(tokenizer, model) -> xgr.GrammarCompiler:
    """Build a GrammarCompiler from the loaded tokenizer and model config."""
    token_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=model.config.vocab_size
    )
    return xgr.GrammarCompiler(token_info)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_old_logprobs(
    model,
    full_seq: list[int],
    prompt_len: int,
    comp_len: int,
    device: torch.device,
) -> list[float]:
    """
    Unconstrained token logprobs for the completion via one forward pass.

    Using a forward pass (not output_scores from generate) ensures the
    logprob normalization is identical to the forward pass used during the
    policy update, making the importance ratio r = exp(new_lp - old_lp)
    well-defined regardless of XGrammar's token masking during generation.
    """
    ids = torch.tensor([full_seq], dtype=torch.long, device=device)
    logits = model(ids).logits[0]                       # (L, V)
    log_probs = F.log_softmax(logits, dim=-1)           # (L, V)
    # Causal shift: position p-1 predicts token at position p.
    comp_ids = torch.tensor(
        full_seq[prompt_len : prompt_len + comp_len], dtype=torch.long, device=device
    )
    comp_lp = log_probs[prompt_len - 1 : prompt_len - 1 + comp_len]   # (C, V)
    lps = comp_lp[torch.arange(comp_len, device=device), comp_ids]    # (C,)
    return lps.float().cpu().tolist()


def rollout_episode(
    model,
    tokenizer,
    compiled_grammar: xgr.CompiledGrammar,
    task_id: str,
    seed: int,
    sampling_seed: int,
    config: Config,
    device: torch.device,
) -> Trajectory:
    """
    Run one full episode; return a Trajectory.

    A fresh WildfireEnvironment is created per call (stateful — never reuse).
    """
    torch.manual_seed(sampling_seed)

    env = WildfireEnvironment()
    obs = env.reset(task_id=task_id, seed=seed)

    steps: list[StepData] = []
    parse_successes = 0

    model.eval()
    while not obs.done and len(steps) < config.max_episode_steps:
        messages = _build_messages(obs)
        # enable_thinking=False: suppresses <think> blocks that eat generation
        # budget and break structured decoding.
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids_t = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(device)
        prompt_len = prompt_ids_t.shape[1]

        # New XGrammarLogitsProcessor instance required per generate() call.
        xgr_proc = XGrammarLogitsProcessor(compiled_grammar)
        gen_out = model.generate(
            prompt_ids_t,
            max_new_tokens=config.max_new_tokens,
            temperature=config.rollout_temperature,
            top_p=config.rollout_top_p,
            top_k=config.rollout_top_k,
            do_sample=True,
            logits_processor=[xgr_proc],
            pad_token_id=tokenizer.eos_token_id,
        )
        completion_ids = gen_out[0, prompt_len:].tolist()
        comp_len = len(completion_ids)
        del gen_out  # free KV-cache storage held by the output tensor

        # Parse action (should be ~100% with XGrammar; track for diagnostics)
        raw_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
        try:
            action = WildfireAction.model_validate_json(raw_text)
            parse_ok = True
        except Exception:
            action = WildfireAction()   # no-op fallback
            parse_ok = False
        parse_successes += int(parse_ok)

        # Step the environment
        obs = env.step(action)

        # Compute unconstrained old_logprobs for the importance ratio later.
        # Skip steps where generation produced nothing (degenerate edge case).
        if comp_len > 0:
            full_seq = prompt_ids_t[0].tolist() + completion_ids
            del prompt_ids_t  # free prompt tensor before the extra forward pass
            old_lps = _compute_old_logprobs(model, full_seq, prompt_len, comp_len, device)
            steps.append(StepData(
                full_seq=full_seq,
                prompt_len=prompt_len,
                comp_len=comp_len,
                old_logprobs=old_lps,
                step_reward=obs.reward,
                parse_success=parse_ok,
            ))

    env.close()

    # Final grader score
    req = GraderRequest(
        task_id=obs.task_id,
        seed=seed,
        step=obs.step,
        max_steps=obs.max_steps,
        structures=[s.model_dump() for s in obs.structures],
        burned_cells=obs.burned_cells,
        burning_cells=obs.burning_cells,
    )
    grader_score = _grade_episode(req).score
    total_return = sum(s.step_reward for s in steps) + config.grader_return_weight * grader_score

    return Trajectory(
        steps=steps,
        task_id=task_id,
        seed=seed,
        sampling_seed=sampling_seed,
        total_return=total_return,
        grader_score=grader_score,
        episode_steps=len(steps),
        action_parse_successes=parse_successes,
    )


def rollout_group(
    model,
    tokenizer,
    compiled_grammar: xgr.CompiledGrammar,
    task_id: str,
    seed: int,
    config: Config,
    device: torch.device,
    base_sampling_seed: int = 0,
) -> list[Trajectory]:
    """
    Collect G trajectories for GRPO advantage estimation.

    GRPO needs within-group reward variance to compute meaningful advantages.
    With a fully-deterministic env (fire spread seeded by `seed`) and XGrammar
    constraining token choice, fixing `seed` across the group often produces
    near-identical trajectories → std_return ≈ 0 → zero gradient.

    When `vary_env_seed_in_group=True` we bump the env seed per group member,
    trading "same problem, different policy samples" purity for real variance
    in fire layouts. The task_id (difficulty tier) is still held fixed, so
    the advantage signal is still tier-calibrated.
    """
    trajectories = []
    for g in range(config.group_size):
        sampling_seed = base_sampling_seed + g * 1000 + seed
        env_seed = seed + g if config.vary_env_seed_in_group else seed
        traj = rollout_episode(
            model, tokenizer, compiled_grammar,
            task_id, env_seed, sampling_seed, config, device,
        )
        trajectories.append(traj)
    return trajectories


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------

def compute_advantages(trajectories: list[Trajectory]) -> list[float]:
    """
    GRPO group-relative advantage: (r - mean(r)) / (std(r) + eps).

    Trajectory-level returns are normalised within the group.  Every token
    in trajectory i receives the same advantage (broadcast).

    If std == 0 (all G trajectories produced identical return), advantages
    are all zero and the gradient is zero — wasted compute but not a crash.
    """
    returns = np.array([t.total_return for t in trajectories], dtype=np.float64)
    mean_r = returns.mean()
    std_r = returns.std()
    if std_r < 1e-8:
        return [0.0] * len(trajectories)
    return ((returns - mean_r) / (std_r + 1e-8)).tolist()


# ---------------------------------------------------------------------------
# GRPO loss and update
# ---------------------------------------------------------------------------

class _UnslothHiddenStatesMode:
    """
    Scoped env-var toggle — while active, Unsloth's patched CausalLM.forward
    returns hidden states (B, L, H) in the .logits field of
    CausalLMOutputWithPast instead of running the internal lm_head.

    Checked by Unsloth per forward call (not cached), so it's safe to flip
    around specific forward passes.  Must be scoped: globally setting it
    would break model.generate() and any caller reading real .logits.
    """
    _VAR = "UNSLOTH_RETURN_HIDDEN_STATES"

    def __enter__(self):
        self._prev = os.environ.get(self._VAR)
        os.environ[self._VAR] = "1"
        return self

    def __exit__(self, *_):
        if self._prev is None:
            os.environ.pop(self._VAR, None)
        else:
            os.environ[self._VAR] = self._prev


def _logprobs_for_batch(
    model,
    sequences: list[list[int]],
    prompt_lens: list[int],
    comp_lens: list[int],
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Per-token completion logprobs, processed one sequence at a time.

    Memory strategy: under _UnslothHiddenStatesMode, the full model forward
    skips Unsloth's internal lm_head and returns hidden states (B, L, H) in
    the .logits field.  We then apply lm_head ourselves to the c completion
    positions only, avoiding the full (L, V) logits allocation.

    Forwarding `model` (the PEFT wrapper) directly — rather than reaching into
    `get_base_model().model` — keeps Unsloth's fast forward path, gradient
    checkpointing, and disable_adapter() all working without any assumptions
    about internal module nesting (the source of the earlier
    "mat1 and mat2 shapes cannot be multiplied" crash, where `model.model`
    was actually Qwen3ForCausalLM and returned logits instead of hidden states).

    For Qwen3-1.7B (V=151936, H=2048) at L≈600, c≈100 in fp16:
      Without: L × V × 2 B ≈ 183 MB per sequence (full logits)
      With:    c × V × 2 B ≈  30 MB per sequence (~6× reduction)
    """
    # lm_head lives on the causal LM two PEFT wrappers in.  get_base_model()
    # returns Qwen3ForCausalLM whose .lm_head is the vocab projection.
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    lm_head   = causal_lm.lm_head  # Linear(hidden_dim → vocab_size)
    hidden_dim = lm_head.in_features

    result = []
    with _UnslothHiddenStatesMode():
        for seq, p, c in zip(sequences, prompt_lens, comp_lens):
            ids  = torch.tensor([seq], dtype=torch.long, device=device)
            attn = torch.ones(1, len(seq), dtype=torch.long, device=device)

            # ── Step 1: full model forward — .logits holds hidden states ────
            out = model(input_ids=ids, attention_mask=attn, use_cache=False)
            hidden = out.logits                                      # (1, L, H)

            # Guard: if the Unsloth version in use doesn't honour the env var
            # the last dim will be vocab_size rather than hidden_dim.  Fail
            # loudly rather than letting lm_head crash with a cryptic matmul
            # shape error downstream.
            if hidden.shape[-1] != hidden_dim:
                raise RuntimeError(
                    f"Expected hidden dim {hidden_dim} from model forward, got "
                    f"{hidden.shape[-1]}. Unsloth did not honour "
                    "UNSLOTH_RETURN_HIDDEN_STATES=1 — check unsloth version."
                )

            # ── Step 2: slice to completion positions only ─────────────────
            # Causal shift: hidden[p-1] predicts the token at position p.
            comp_hidden = hidden[0, p - 1 : p - 1 + c]              # (c, H)
            del hidden, out                                          # free (1, L, H)

            # ── Step 3: apply lm_head to the small slice only ──────────────
            comp_logits = lm_head(comp_hidden)                       # (c, V)

            comp_ids = torch.tensor(seq[p : p + c], dtype=torch.long, device=device)
            lps = F.log_softmax(comp_logits, dim=-1)[
                torch.arange(c, device=device), comp_ids
            ]                                                        # (c,)
            result.append(lps)
    return result


def grpo_update(
    model,
    optimizer: torch.optim.Optimizer,
    trajectories: list[Trajectory],
    advantages: list[float],
    config: Config,
    device: torch.device,
) -> dict:
    """
    K inner epochs of GRPO clipped-surrogate updates with optional KL penalty.

    Loss per token:
        L = -min(r*A, clip(r,1-ε,1+ε)*A) + β * KL(π_θ || π_ref)
    where r = exp(new_lp - old_lp) and KL uses Schulman (2020) estimator.

    Memory strategy: each sequence is forward→loss→backward→discard. Only one
    autograd graph is alive at a time; gradients accumulate in param.grad across
    sequences within an epoch. The old "micro_batch_size" flag is kept as a
    cosmetic grouping — the peak memory is 1 seq either way. optimizer.step()
    is called once per inner epoch (not per micro-batch) so the effective batch
    size is the full epoch. Loss is scaled by 1/len(all_steps) at each backward
    so accumulated gradients equal the mean-loss gradient of the old path.
    """
    # Flatten (traj, step) → list of (StepData, advantage)
    all_steps: list[tuple[StepData, float]] = []
    for traj_idx, traj in enumerate(trajectories):
        adv = advantages[traj_idx]
        for step in traj.steps:
            if step.comp_len > 0:
                all_steps.append((step, adv))

    if not all_steps:
        return {"policy_loss": 0.0, "kl_divergence": 0.0, "clip_fraction": 0.0}

    total_policy_loss = 0.0
    total_kl = 0.0
    total_clip_frac = 0.0
    n_samples = 0
    inv_n = 1.0 / len(all_steps)

    for _epoch in range(config.inner_epochs):
        random.shuffle(all_steps)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()  # free fragments from previous epoch / rollout
        epoch_loss = 0.0

        for step, adv in all_steps:
            model.train()

            # ── Reference logprobs first (no grad) so the PEFT adapter can be
            # disabled cleanly without interleaving with the grad forward. ──
            if config.kl_coef > 0:
                with model.disable_adapter():
                    with torch.no_grad():
                        ref_lp = _logprobs_for_batch(
                            model, [step.full_seq], [step.prompt_len],
                            [step.comp_len], device,
                        )[0].detach()
            else:
                ref_lp = None

            # ── Policy forward WITH grad ──────────────────────────────────
            new_lp = _logprobs_for_batch(
                model, [step.full_seq], [step.prompt_len],
                [step.comp_len], device,
            )[0]                                                        # (C,) with grad

            old_lp = torch.tensor(step.old_logprobs, device=device)     # (C,) detached
            adv_t  = torch.tensor(adv, dtype=torch.float32, device=device)

            ratio     = torch.exp(new_lp - old_lp)
            clipped   = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
            surrogate = torch.min(ratio * adv_t, clipped * adv_t)
            loss_i    = -surrogate.mean()

            cf = ((ratio - clipped).abs() > 1e-6).float().mean().item()
            total_clip_frac += cf

            if ref_lp is not None:
                # Schulman (2020) unbiased KL estimator:
                #   KL(π_θ || π_ref) ≈ e^(log π_ref - log π_θ) - 1 - (log π_ref - log π_θ)
                log_r = ref_lp - new_lp
                kl_i  = (torch.exp(log_r) - 1.0 - log_r).mean()
                total_kl += kl_i.item()
                loss_i = loss_i + config.kl_coef * kl_i

            (loss_i * inv_n).backward()
            epoch_loss += loss_i.item()
            n_samples += 1

            # Free the graph eagerly — matters on T4 where the next forward
            # would otherwise trip OOM before Python GC runs.
            del new_lp, ratio, clipped, surrogate, loss_i
            if ref_lp is not None:
                del ref_lp

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_policy_loss += epoch_loss

    n = max(1, n_samples)
    return {
        "policy_loss":   total_policy_loss / config.inner_epochs,
        "kl_divergence": total_kl / n,
        "clip_fraction": total_clip_frac / n,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def make_logger(config: Config):
    """Returns a log(dict) callable. wandb if WANDB_API_KEY is set, else JSONL."""
    os.makedirs(config.output_dir, exist_ok=True)
    _wandb_ok = False
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            wandb.init(project="grpo-wildfire", config=vars(config))
            _wandb_ok = True
        except Exception as exc:
            print(f"[warn] wandb init failed ({exc}); falling back to JSONL")

    log_path = os.path.join(config.output_dir, "log.jsonl")

    def _log(metrics: dict):
        if _wandb_ok:
            import wandb as _w
            _w.log(metrics)
        with open(log_path, "a") as fh:
            fh.write(json.dumps(metrics) + "\n")

    return _log


# ---------------------------------------------------------------------------
# Curriculum helper
# ---------------------------------------------------------------------------

def get_task_for_iter(iteration: int, config: Config) -> str:
    for task_id, start, end in config.task_curriculum:
        if start <= iteration < end:
            return task_id
    return config.task_curriculum[-1][0]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.output_dir, exist_ok=True)
    log = make_logger(config)

    # ── Load model ──────────────────────────────────────────────────────────
    # Unsloth must be imported before transformers to apply all kernel patches.
    # We import it here (lazily) so the module is importable on CPU-only machines
    # for testing, but on GPU it patches transformers correctly at load time.
    import unsloth  # noqa: F401, PLC0415 — patches transformers before model load
    from unsloth import FastLanguageModel  # noqa: PLC0415

    print(f"Loading {config.model_name} (4-bit QLoRA) …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=3072,           # raised from 2048 — 25×25 hard obs + action_guide approach 2k tokens
        dtype=None,             # auto: bf16 on Ampere+, fp16 otherwise
        load_in_4bit=True,      # Unsloth handles BitsAndBytesConfig internally
        fast_inference=False,   # REQUIRED: enables HF model.generate() + LogitsProcessors
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA ────────────────────────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=list(config.lora_target_modules),
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",   # Unsloth's memory-efficient checkpointing
        random_state=42,
    )
    print(f"  LoRA: rank={config.lora_rank} alpha={config.lora_alpha} "
          f"dropout={config.lora_dropout} modules={config.lora_target_modules}")

    # ── Resume from previous run if latest/ exists ──────────────────────────
    # Saved every iter → at most one iter is lost on disconnect.
    start_iter = 0
    best_grader_per_task: dict[str, float] = {}
    resume_dir    = os.path.join(config.output_dir, "latest")
    resume_state  = os.path.join(config.output_dir, "resume_state.json")
    if os.path.isdir(resume_dir) and os.path.isfile(resume_state):
        from peft import set_peft_model_state_dict
        try:
            from safetensors.torch import load_file as _load_st
            adapter_file = os.path.join(resume_dir, "adapter_model.safetensors")
            if os.path.isfile(adapter_file):
                state = _load_st(adapter_file, device="cpu")
            else:  # older PEFT saved .bin
                state = torch.load(
                    os.path.join(resume_dir, "adapter_model.bin"),
                    map_location="cpu",
                )
            set_peft_model_state_dict(model, state)
            with open(resume_state) as fh:
                meta = json.load(fh)
            start_iter = int(meta["next_iter"])
            best_grader_per_task = {k: float(v) for k, v in meta.get("best_grader_per_task", {}).items()}
            print(f"  ↻ Resumed from {resume_dir}  → start_iter={start_iter}  "
                  f"best={best_grader_per_task}")
        except Exception as exc:
            print(f"  ⚠  resume failed ({exc}); starting from scratch")
            start_iter = 0
            best_grader_per_task = {}

    # ── XGrammar ────────────────────────────────────────────────────────────
    print("Compiling WildfireAction JSON schema grammar …")
    compiler = build_grammar_compiler(tokenizer, model)
    compiled_grammar = compiler.compile_json_schema(WildfireAction.model_json_schema())
    print("  Grammar compiled.")

    # ── Optimiser ───────────────────────────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer: torch.optim.Optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        print(f"  Using AdamW8bit optimizer (bitsandbytes), wd={config.weight_decay}")
    except (ImportError, AttributeError):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        print(f"  Using standard AdamW optimizer, wd={config.weight_decay}")

    # ── LR schedule: linear warmup → cosine decay to lr_min ────────────────
    # Warmup fights the iter-0 loss spike (prior run hit +17 on iter 0 because
    # the full 3e-5 LR met uncalibrated advantages). Cosine then handles the
    # late-training drift that plateaued seed-67 gains after iter 8.
    import math as _math
    def _lr_at(iter_idx: int) -> float:
        if iter_idx < config.warmup_iters:
            # Linear ramp from lr_min → learning_rate over warmup_iters
            frac = (iter_idx + 1) / max(1, config.warmup_iters)
            return config.lr_min + (config.learning_rate - config.lr_min) * frac
        post_warmup_total = max(1, config.total_iterations - config.warmup_iters)
        progress = (iter_idx - config.warmup_iters) / post_warmup_total
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + _math.cos(_math.pi * progress))
        return config.lr_min + (config.learning_rate - config.lr_min) * cos

    # ── Training iterations ─────────────────────────────────────────────────
    # (best_grader_per_task is initialized above, possibly from resume state)
    for iteration in range(start_iter, config.total_iterations):
        task_id    = get_task_for_iter(iteration, config)
        seed_pool  = config.seeds_per_task[task_id]

        # Sample K=seeds_per_iter base seeds without replacement, then split
        # group_size rollouts evenly across them. Prevents the policy from
        # specializing to one easy seed (the iter-0/4/8 pattern from the prior run).
        K = max(1, min(config.seeds_per_iter, len(seed_pool)))
        if config.group_size % K != 0:
            raise ValueError(
                f"group_size ({config.group_size}) must be divisible by "
                f"seeds_per_iter ({K}) — adjust Config."
            )
        rng = random.Random(iteration)
        chosen_seeds = rng.sample(seed_pool, K)
        rollouts_per_seed = config.group_size // K

        # Set LR for this iteration
        cur_lr = _lr_at(iteration)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        print(f"\n── iter {iteration:3d}/{config.total_iterations}  "
              f"task={task_id}  seeds={chosen_seeds}  lr={cur_lr:.2e} ──")

        # ── Rollout phase: K independent groups, advantages computed PER GROUP ─
        t0 = time.time()
        all_trajectories: list[Trajectory] = []
        all_advantages:   list[float] = []
        per_seed_meta = []     # (seed, group_mean_grader, group_std_return)

        # Use a small saved-config trick to override group_size inside rollout_group
        # without further refactoring — copy config, set group_size = rollouts_per_seed.
        sub_cfg = type(config)(**{**vars(config), "group_size": rollouts_per_seed})

        for k_idx, base_seed in enumerate(chosen_seeds):
            sub_trajs = rollout_group(
                model, tokenizer, compiled_grammar,
                task_id, base_seed, sub_cfg, device,
                base_sampling_seed=iteration * 10_000 + k_idx * 100_000,
            )
            sub_adv = compute_advantages(sub_trajs)
            all_trajectories.extend(sub_trajs)
            all_advantages.extend(sub_adv)
            per_seed_meta.append((
                base_seed,
                float(np.mean([t.grader_score   for t in sub_trajs])),
                float(np.std ([t.total_return  for t in sub_trajs])),
            ))
        rollout_secs = time.time() - t0

        # ── Aggregate diagnostics ──────────────────────────────────────────
        returns     = np.array([t.total_return for t in all_trajectories])
        g_scores    = np.array([t.grader_score for t in all_trajectories])
        ep_steps    = np.array([t.episode_steps for t in all_trajectories])
        parse_ok    = sum(t.action_parse_successes for t in all_trajectories)
        parse_total = sum(t.episode_steps          for t in all_trajectories)
        ret_std     = float(returns.std())

        # ── Update phase ────────────────────────────────────────────────────
        torch.cuda.empty_cache()
        t1 = time.time()
        upd = grpo_update(
            model, optimizer, all_trajectories, all_advantages, config, device,
        )
        update_secs = time.time() - t1

        # ── Log ─────────────────────────────────────────────────────────────
        metrics = {
            "iter":                      iteration,
            "task_id":                   task_id,
            "seeds":                     chosen_seeds,
            "lr":                        cur_lr,
            "mean_return":               float(returns.mean()),
            "std_return":                ret_std,
            "min_return":                float(returns.min()),
            "max_return":                float(returns.max()),
            "mean_grader_score":         float(g_scores.mean()),
            "max_grader_score":          float(g_scores.max()),
            "per_seed_grader":           {s: g for s, g, _ in per_seed_meta},
            "action_parse_success_rate": parse_ok / max(1, parse_total),
            "mean_episode_steps":        float(ep_steps.mean()),
            "policy_loss":               upd["policy_loss"],
            "kl_divergence":             upd["kl_divergence"],
            "clip_fraction":             upd["clip_fraction"],
            "rollout_seconds":           rollout_secs,
            "update_seconds":            update_secs,
        }
        log(metrics)
        per_seed_str = "  ".join(f"s{s}={g:.2f}" for s, g, _ in per_seed_meta)
        print(
            f"  loss={metrics['policy_loss']:+.4f}  "
            f"grader={metrics['mean_grader_score']:.3f} ({per_seed_str})  "
            f"ret={metrics['mean_return']:+.2f}±{ret_std:.2f}  "
            f"parse={metrics['action_parse_success_rate']:.1%}  "
            f"kl={metrics['kl_divergence']:.4f}  "
            f"clip={metrics['clip_fraction']:.2%}"
        )

        if metrics["action_parse_success_rate"] < 0.95:
            print("  ⚠  parse_success_rate < 0.95 — check XGrammar integration")

        # ── Save best-grader adapter per task ──────────────────────────────
        cur_grader = metrics["mean_grader_score"]
        prev_best  = best_grader_per_task.get(task_id, -1.0)
        if cur_grader > prev_best:
            best_grader_per_task[task_id] = cur_grader
            best_dir = os.path.join(config.output_dir, f"best_adapter_{task_id}")
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"  ★ new best on {task_id}: {cur_grader:.3f} (prev {prev_best:.3f}) → {best_dir}")

        # ── Resume checkpoint every iter (overwrites) ──────────────────────
        # Tiny overhead (~1-2 s for rank-16 LoRA) but caps disconnect loss to
        # at most one iteration. The companion resume_state.json carries the
        # iter pointer and per-task best-grader map.
        latest_dir = os.path.join(config.output_dir, "latest")
        model.save_pretrained(latest_dir)
        with open(os.path.join(config.output_dir, "resume_state.json"), "w") as fh:
            json.dump({
                "next_iter":             iteration + 1,
                "best_grader_per_task":  best_grader_per_task,
                "total_iterations":      config.total_iterations,
            }, fh, indent=2)

        # ── Periodic snapshot every 10 iters ───────────────────────────────
        if (iteration + 1) % 10 == 0 or iteration == config.total_iterations - 1:
            ckpt = os.path.join(config.output_dir, f"adapter_iter{iteration + 1:04d}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  Saved → {ckpt}")

    print("\nTraining complete.")
    print("Best grader per task:", best_grader_per_task)


if __name__ == "__main__":
    train(Config())

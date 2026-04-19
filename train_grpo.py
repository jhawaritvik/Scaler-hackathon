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
    task_curriculum: tuple = (("medium", 0, 20), ("hard", 20, 30))
    seeds_per_task: dict = field(default_factory=lambda: {
        "easy":   [42, 100, 200, 300],
        "medium": [67, 101, 201, 301],
        "hard":   [12, 102, 202, 302],
    })
    group_size: int = 8
    inner_epochs: int = 4
    micro_batch_size: int = 4       # sequences per gradient accumulation step
    max_episode_steps: int = 20
    max_new_tokens: int = 512       # max tokens generated per action
    learning_rate: float = 5e-6
    lora_rank: int = 16
    lora_alpha: int = 32
    # Qwen3 uses standard transformer attention: q/k/v/o projections are correct targets.
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    kl_coef: float = 0.04
    clip_range: float = 0.2
    total_iterations: int = 30
    rollout_temperature: float = 0.9
    rollout_top_p: float = 0.8
    rollout_top_k: int = 20
    output_dir: str = "./grpo_wildfire"


# ---------------------------------------------------------------------------
# System prompt (mirrors inference.py; redefined here to avoid that
# module's HF_TOKEN check at import time)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI wildfire incident commander. Dispatch firefighting resources to contain fires and protect structures on a 15x15 terrain grid.

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
                    for f in sorted(obs.fire_details, key=lambda f: -f.intensity)[:8]
                ],
                "heat_warnings": [
                    {"row": h.row, "col": h.col}
                    for h in obs.heat_warnings[:5]
                ],
            },
            "atmosphere": {
                "wind_speed_kmh": round(obs.wind_speed, 1),
                "wind_direction_deg": round(obs.wind_direction, 1),
                "temperature_c": round(obs.temperature, 1),
                "humidity": round(obs.humidity, 2),
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
            "wind": {
                "speed_kmh": round(obs.wind_speed, 1),
                "direction_deg": round(obs.wind_direction, 1),
            },
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
    total_return = sum(s.step_reward for s in steps) + 2.0 * grader_score

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
    Collect G trajectories for the same task+seed with different sampling seeds.

    GRPO needs within-group reward variance to compute meaningful advantages.
    Varying only sampling_seed while holding task+seed fixed gives variance
    from stochastic sampling while keeping the problem identical.
    """
    trajectories = []
    for g in range(config.group_size):
        sampling_seed = base_sampling_seed + g * 1000 + seed
        traj = rollout_episode(
            model, tokenizer, compiled_grammar,
            task_id, seed, sampling_seed, config, device,
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

def _logprobs_for_batch(
    model,
    sequences: list[list[int]],
    prompt_lens: list[int],
    comp_lens: list[int],
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Per-token completion logprobs, processed one sequence at a time.

    Memory strategy: run the base transformer (model.model) WITHOUT the
    lm_head to get hidden states (1, L, hidden_dim), then slice to only the
    c completion positions, THEN apply lm_head to the tiny (c, hidden_dim)
    slice.  This avoids ever materialising the full (L, vocab_size) logits
    tensor — only (c, vocab_size) is created.

    For Qwen3-1.7B (V=151936, hidden=2048) at L≈600, c≈100 in fp16:
      - Old peak: L × V × 2 B  ≈  183 MB per sequence
      - New peak: c × V × 2 B  ≈   31 MB per sequence  (~6× reduction)

    Both the policy pass (with grad) and the reference pass
    (model.disable_adapter() + torch.no_grad()) go through this function,
    so both benefit from the same reduction.
    """
    result = []
    for seq, p, c in zip(sequences, prompt_lens, comp_lens):
        ids  = torch.tensor([seq], dtype=torch.long, device=device)
        attn = torch.ones(1, len(seq), dtype=torch.long, device=device)

        # ── Step 1: base transformer forward — no lm_head ──────────────────
        # model.model is the bare transformer stack; lm_head is a separate
        # Linear layer applied afterwards.  Skipping it here keeps the full
        # (1, L, hidden_dim) hidden state in memory rather than the much
        # larger (1, L, vocab_size) logit tensor.
        hidden = model.model(ids, attention_mask=attn)[0]        # (1, L, H)

        # ── Step 2: slice to completion positions only ──────────────────────
        # Causal shift: hidden[p-1] is the state that predicts token at p.
        comp_hidden = hidden[0, p - 1 : p - 1 + c]              # (c, H)
        del hidden                                               # free (1, L, H)

        # ── Step 3: apply lm_head to the small slice ────────────────────────
        comp_logits = model.lm_head(comp_hidden)                 # (c, V)

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

    Gradient is accumulated across micro-batches before each optimizer step.
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

    for _epoch in range(config.inner_epochs):
        random.shuffle(all_steps)
        optimizer.zero_grad()
        torch.cuda.empty_cache()  # free fragments from previous epoch / rollout
        epoch_loss = 0.0

        for chunk_start in range(0, len(all_steps), config.micro_batch_size):
            chunk = all_steps[chunk_start : chunk_start + config.micro_batch_size]
            seqs    = [s.full_seq     for s, _ in chunk]
            plens   = [s.prompt_len   for s, _ in chunk]
            clens   = [s.comp_len     for s, _ in chunk]
            old_lps = [s.old_logprobs for s, _ in chunk]
            advs    = [a              for _, a in chunk]

            model.train()
            new_lps_list = _logprobs_for_batch(model, seqs, plens, clens, device)

            # Reference logprobs for KL (base model, no gradient needed)
            if config.kl_coef > 0:
                with model.disable_adapter():
                    with torch.no_grad():
                        ref_lps_list = _logprobs_for_batch(model, seqs, plens, clens, device)
            else:
                ref_lps_list = [None] * len(chunk)

            loss_terms = []
            for i in range(len(chunk)):
                new_lp = new_lps_list[i]                                         # (C,) with grad
                old_lp = torch.tensor(old_lps[i], device=device)                 # (C,) detached
                adv_t  = torch.tensor(advs[i], dtype=torch.float32, device=device)

                ratio   = torch.exp(new_lp - old_lp)
                clipped = torch.clamp(ratio, 1.0 - config.clip_range, 1.0 + config.clip_range)
                surrogate = torch.min(ratio * adv_t, clipped * adv_t)
                loss_i    = -surrogate.mean()

                cf = ((ratio - clipped).abs() > 1e-6).float().mean().item()
                total_clip_frac += cf

                if config.kl_coef > 0 and ref_lps_list[i] is not None:
                    ref_lp = ref_lps_list[i].detach()
                    # Schulman (2020) unbiased KL estimator:
                    #   KL(π_θ || π_ref) ≈ e^(log π_ref - log π_θ) - 1 - (log π_ref - log π_θ)
                    log_r = ref_lp - new_lp
                    kl_i  = (torch.exp(log_r) - 1.0 - log_r).mean()
                    total_kl += kl_i.item()
                    loss_i = loss_i + config.kl_coef * kl_i

                loss_terms.append(loss_i)
                n_samples += 1

            # Normalise by total step count so the gradient magnitude is
            # consistent regardless of how many steps are in this chunk.
            chunk_loss = sum(loss_terms) / len(all_steps)
            chunk_loss.backward()
            epoch_loss += chunk_loss.item() * len(chunk)

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
        max_seq_length=2048,
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
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",   # Unsloth's memory-efficient checkpointing
        random_state=42,
    )
    print(f"  LoRA: rank={config.lora_rank} alpha={config.lora_alpha} "
          f"modules={config.lora_target_modules}")

    # ── XGrammar ────────────────────────────────────────────────────────────
    print("Compiling WildfireAction JSON schema grammar …")
    compiler = build_grammar_compiler(tokenizer, model)
    compiled_grammar = compiler.compile_json_schema(WildfireAction.model_json_schema())
    print("  Grammar compiled.")

    # ── Optimiser ───────────────────────────────────────────────────────────
    try:
        import bitsandbytes as bnb
        optimizer: torch.optim.Optimizer = bnb.optim.AdamW8bit(
            model.parameters(), lr=config.learning_rate
        )
        print("  Using AdamW8bit optimizer (bitsandbytes)")
    except (ImportError, AttributeError):
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        print("  Using standard AdamW optimizer")

    # ── Training iterations ─────────────────────────────────────────────────
    for iteration in range(config.total_iterations):
        task_id = get_task_for_iter(iteration, config)
        seeds   = config.seeds_per_task[task_id]
        seed    = seeds[iteration % len(seeds)]

        print(f"\n── iter {iteration:3d}/{config.total_iterations}  "
              f"task={task_id}  seed={seed} ──")

        # ── Rollout phase ───────────────────────────────────────────────────
        t0 = time.time()
        trajectories = rollout_group(
            model, tokenizer, compiled_grammar,
            task_id, seed, config, device,
            base_sampling_seed=iteration * 10_000,
        )
        rollout_secs = time.time() - t0

        # ── Advantages ─────────────────────────────────────────────────────
        advantages = compute_advantages(trajectories)
        returns     = np.array([t.total_return        for t in trajectories])
        g_scores    = np.array([t.grader_score        for t in trajectories])
        ep_steps    = np.array([t.episode_steps       for t in trajectories])
        parse_ok    = sum(t.action_parse_successes     for t in trajectories)
        parse_total = sum(t.episode_steps              for t in trajectories)

        ret_std = float(returns.std())
        if ret_std < 1e-6:
            print("  ⚠  reward std=0 — all trajectories tied; gradient will be zero")

        # ── Update phase ────────────────────────────────────────────────────
        # Release cached GPU memory that accumulated during rollout (KV-cache
        # fragments, intermediate tensors from generate()) before starting the
        # backward-pass-heavy update step.
        torch.cuda.empty_cache()

        t1 = time.time()
        upd = grpo_update(model, optimizer, trajectories, advantages, config, device)
        update_secs = time.time() - t1

        # ── Log ─────────────────────────────────────────────────────────────
        metrics = {
            "iter":                      iteration,
            "task_id":                   task_id,
            "seed":                      seed,
            "mean_return":               float(returns.mean()),
            "std_return":                ret_std,
            "min_return":                float(returns.min()),
            "max_return":                float(returns.max()),
            "mean_grader_score":         float(g_scores.mean()),
            "max_grader_score":          float(g_scores.max()),
            "action_parse_success_rate": parse_ok / max(1, parse_total),
            "mean_episode_steps":        float(ep_steps.mean()),
            "policy_loss":               upd["policy_loss"],
            "kl_divergence":             upd["kl_divergence"],
            "clip_fraction":             upd["clip_fraction"],
            "rollout_seconds":           rollout_secs,
            "update_seconds":            update_secs,
        }
        log(metrics)
        print(
            f"  loss={metrics['policy_loss']:+.4f}  "
            f"grader={metrics['mean_grader_score']:.3f}  "
            f"ret={metrics['mean_return']:.2f}±{ret_std:.2f}  "
            f"parse={metrics['action_parse_success_rate']:.1%}  "
            f"kl={metrics['kl_divergence']:.4f}  "
            f"clip={metrics['clip_fraction']:.2%}"
        )

        # Warn if parse rate drops below threshold (should be ~1.0 with XGrammar)
        if metrics["action_parse_success_rate"] < 0.95:
            print("  ⚠  parse_success_rate < 0.95 — check XGrammar integration")

        # ── Save adapter checkpoint every 10 iters ─────────────────────────
        if (iteration + 1) % 10 == 0 or iteration == config.total_iterations - 1:
            ckpt = os.path.join(config.output_dir, f"adapter_iter{iteration + 1:04d}")
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"  Saved → {ckpt}")

    print("\nTraining complete.")


if __name__ == "__main__":
    train(Config())

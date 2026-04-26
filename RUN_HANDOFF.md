# Run Handoff

This file captures the key decisions and current operating plan from the live
training/evaluation session so another assistant/session can continue without
reconstructing the whole chat.

## Current Goal

Submit the Wildfire OpenEnv hackathon project before the deadline with:

- a deployed OpenEnv Hugging Face Space
- real GRPO training evidence
- baseline and trained-model evaluation through the live Space using the
  official OpenEnv `EnvClient` WebSocket session (so multi-turn rollouts
  actually maintain server-side state)
- reward/loss plots and final JSON artifacts
- a lean repo containing only the final submission path

`eval_policy_http.py` (legacy filename, now WebSocket-based) is the single
evaluator. The filename is preserved so README/notebook/`submission_check.py`
references keep working.

## Live Space

- HF Space repo: `Chunchunmaru-101/wildfire-env`
- Live app: `https://chunchunmaru-101-wildfire-env.hf.space`
- Verified endpoints:
  - `GET /`
  - `GET /tasks`
  - `GET /schema`
  - `WS  /ws`        (OpenEnv `EnvClient` session — used for `reset`/`step`)
  - `POST /grader`   (wildfire-specific scoring endpoint)

The evaluator drives:

```text
GET /, GET /schema                 # smoke / health check
WS  /ws  -> reset(task_id, seed)   # opens a stateful session
WS  /ws  -> step(action)           # repeats until done or max_steps
POST /grader                       # final score
```

Model inference happens locally on the GPU runtime; environment interaction
goes through the deployed Space.

> Why WebSocket? OpenEnv's HTTP `POST /reset` and `POST /step` handlers
> construct a fresh `Environment` per request and tear it down — they're
> stateless by design and only useful for one-shot probes. We initially
> tried raw HTTP and saw `obs.step == 1` after every step regardless of how
> many we sent (the server kept building a new env each call). The
> WebSocket session keeps one env alive, which is what RL rollouts need.

## Repo Cleanup Already Done

Committed and pushed:

- `3a62d30 Use OpenEnv evaluator for final showcase`

This commit:

- added `eval_policy_http.py`
- added `notebooks/wildfire_http_eval_hf.ipynb`
- made `notebooks/wildfire_training_eval_hf.ipynb` training-only
- removed old direct `eval_policy.py`
- removed `record_qwen_run.py`
- removed old Kaggle notebook
- removed tests/fixtures
- updated `README.md`, `submission_check.py`, and docs for OpenEnv eval

Later local-only cleanup was performed but not necessarily pushed yet:

- deleted `ENV_REVIEW.md`
- deleted `inference.py`
- deleted `replays/heuristic_hard_9.json`
- updated stale references in `README.md`, `wildfire_env/README.md`, and `train_grpo.py`

Before pushing any additional cleanup, check `git status`.

## Training Runtime

Training is running on a Hugging Face JupyterLab Space using **Nvidia A10G large**
at about `$1.50/hour`.

Estimated cost as of around 5:48 AM was roughly `$17-18`; continuing another
5-6 hours would bring total near `$25-27`. Stop/pause the Space immediately
after final artifacts are produced.

The original 60-iteration run was too slow. We switched to a deadline run:

```python
from train_grpo import Config, train

train(Config(
    total_iterations=20,
    task_curriculum=(("easy", 0, 5), ("medium", 5, 12), ("hard", 12, 20)),
    group_size=2,
    seeds_per_iter=1,
    max_episode_steps=20,
    lora_dropout=0.0,
    warmup_iters=3,
    save_every=6,
    run_name="deadline_v2_a10g",
))
```

It resumed from `grpo_wildfire/latest` at `start_iter=2`.

Important: do **not** `git pull` inside the HF training runtime while Cell 4 is
actively training. Pull only after training stops/finishes and checkpoints are
uploaded or safely saved.

## Training Status Seen So Far

By ~5:46 AM, training had reached hard:

- `iter 12`, hard seed `[80]`: grader `0.3133`, parse `0.825`
- `iter 13`, hard seed `[52]`: grader `0.24685`, parse `0.60`

Recommendation from the chat:

- keep running through at least one more hard iteration (`iter 14`)
- if time/cost is tight, stop after iter 14 or 15
- upload checkpoints
- run final artifact generation and OpenEnv eval

Do not wait so long that there is no time for final eval. The project is judged
heavily on environment/story and observable training evidence, not perfect model
quality.

## Final Eval Plan

Use `eval_policy_http.py` for both untrained and trained. Keep settings identical
for fair comparison.

Recommended settings (5 held-out seeds × 3 tasks = 15 episodes; lets each
episode run to the env's own `max_steps` so delayed ignitions actually fire):

```bash
--seeds-per-task 5
```

`--max-new-tokens` now defaults to **1024** in the eval CLI. The trained policy
naturally produces ~240-300 tokens of action JSON; lower budgets truncate
mid-`assignments` and bias scores against the trained model. Do not override
unless you know what you're doing.

`use_cache=True` is enabled in `eval_policy_http.py:_generate_action`, validated
by `_tmp_cache_smoke.py` to give 16/16 parse rate across diverse step depths
(easy seed=11 at steps 0/2/4/6, hard seed=9 at steps 2/3/6/10) at ~16.6 tok/s
— roughly 5× faster than `use_cache=False` while producing equivalent JSON.

> Do **not** pass `--max-episode-steps 10`. That cap was a previous-session
> mistake; with delayed ignitions firing at obs-steps 5-9, a 10-step cap
> truncates fights mid-cascade and squashes the trained-vs-baseline signal.
> The default behaviour now uses each task's own `max_steps` (20 easy/medium,
> 25 hard) and exits early when the env returns `done=True`.

If time is short, the smallest defensible run is `--seeds-per-task 3`. Going
down to 1 seed/task gives only 3 episodes total — too few for a meaningful
mean.

### Untrained eval (OpenEnv WebSocket)

```bash
python eval_policy_http.py --untrained \
  --base-url https://chunchunmaru-101-wildfire-env.hf.space \
  --seeds-per-task 5 \
  --output submission_artifacts/eval_untrained.json
```

### Trained eval (OpenEnv WebSocket)

```bash
python eval_policy_http.py \
  --base-url https://chunchunmaru-101-wildfire-env.hf.space \
  --seeds-per-task 5 \
  --output submission_artifacts/eval_trained.json
```

The JSON output should show:

- `transport: openenv_websocket`
- `model_name: Qwen/Qwen3-4B-Instruct-2507`
- untrained uses `adapter_path: base_model (zero-shot)`
- trained auto-selects `grpo_wildfire/final_adapter`, then `latest`, then snapshots
- per-episode `steps` should equal `step_cap` (i.e. the env's `max_steps`) for
  long-burning seeds, or terminate earlier with `done: true`. If `steps == 1`
  on every episode, the WebSocket session broke — fall back to a fresh `git
  pull` and rerun.
- `valid_action_rate` should be ≥ 0.95 on both arms with the new 1024-token
  default. If you see < 0.9, re-check `--max-new-tokens` was not overridden.

### Wall-clock estimate

With `use_cache=True` and 1024-token budget, generation runs at ~16.6 tok/s
on A10G (~16 s/call avg for 240-280 tok outputs). 15 episodes × ~16-20 steps
(many end early on `done=True`) × ~16 s/step ≈ **60-95 min per run** on A10G.
Untrained and trained back-to-back: budget **~2 to 3 hours**.

## Kaggle Baseline Attempt

Kaggle was used for the untrained OpenEnv baseline. The old direct evaluator was stopped.

Kaggle dual-GPU command supports two T4s by episode sharding, not model sharding:

```bash
python eval_policy_http.py --untrained \
  --base-url https://chunchunmaru-101-wildfire-env.hf.space \
  --parallel-devices 0,1 \
  --seeds-per-task 5 \
  --output submission_artifacts/eval_untrained.json
```

`--parallel-devices 0,1` launches two workers pinned with `CUDA_VISIBLE_DEVICES`.

If a Kaggle run is interrupted, check:

```python
import glob, os
for p in glob.glob("submission_artifacts/eval_untrained*.json"):
    print(p, os.path.getsize(p), "bytes")
```

If GPU usage remains high after interrupt:

```python
import subprocess
subprocess.run(["pkill", "-f", "eval_policy_http.py"], check=False)
subprocess.run(["nvidia-smi"], check=False)
```

## After Training Finishes

Suggested order:

1. Upload `grpo_wildfire/` checkpoints to HF model repo.
2. `git pull --rebase origin main` in HF runtime.
3. Run `plot_training_curves.py`.
4. Run untrained OpenEnv eval if not already available from Kaggle.
5. Run trained OpenEnv eval.
6. Inspect artifacts.
7. Commit/push artifacts.
8. Stop/pause HF Space runtime.

Checkpoint upload snippet:

```python
import os
from huggingface_hub import HfApi, create_repo

HF_REPO_ID = os.environ.get("HF_REPO_ID", "Chunchunmaru-101/wildfire-grpo-checkpoints")
HF_TOKEN = os.environ.get("HF_TOKEN")

api = HfApi(token=HF_TOKEN)
create_repo(HF_REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)
api.upload_folder(
    folder_path="grpo_wildfire",
    repo_id=HF_REPO_ID,
    repo_type="model",
    commit_message="Upload wildfire GRPO checkpoints",
    ignore_patterns=["*.pyc", "__pycache__/*"],
)
```

## Final Required Artifacts

`submission_check.py` currently expects:

- `reward_audit.json`
- `submission_artifacts/training_reward_curve.png`
- `submission_artifacts/training_loss_curve.png`
- `submission_artifacts/training_summary.md`
- `submission_artifacts/eval_untrained.json`
- `submission_artifacts/eval_trained.json`

Generate plots/summary:

```bash
python plot_training_curves.py
```

Run reward audit locally if not already done:

```bash
.\.venv\Scripts\python.exe reward_audit.py --json-out reward_audit.json
```

Final check:

```bash
python submission_check.py --strict
```

## Final Submission Additions

Before the final hackathon submission, add or verify these items:

1. Public writeup/demo link in `README.md`
   - Verify the README links the separate `Blog.MD` file requested for the
     Hugging Face Space.
   - If a YouTube demo is created, link it directly from the README so judges
     do not have to ask.

2. Results table in `README.md`
   - Add a concise trained-vs-untrained OpenEnv eval table after both JSON files exist.
   - Use `submission_artifacts/eval_untrained.json` and
     `submission_artifacts/eval_trained.json`.
   - Suggested table columns:
     - model/run
     - transport
     - seeds per task
     - max tokens / max steps
     - easy / medium / hard score
     - overall mean

3. Training curves embedded in `README.md`
   - Commit:
     - `submission_artifacts/training_reward_curve.png`
     - `submission_artifacts/training_loss_curve.png`
   - Embed both images in the README with one-line captions.
   - The checker only enforces embedding after the files exist.

4. Reward audit artifact
   - Generate and commit `reward_audit.json`.
   - Mention briefly that exploit-like policies (`noop`, `stage_all`,
     duplicate dispatch) do not outrank the sane heuristic policies.

5. OpenEnv evaluation evidence
   - Commit both:
     - `submission_artifacts/eval_untrained.json`
     - `submission_artifacts/eval_trained.json`
   - Ensure both use the same flags. Recommended:
     `--seeds-per-task 5` (CLI defaults `--max-new-tokens 1024` and
     `use_cache=True` after the cache-fix patch — do not override either).
     No `--max-episode-steps` override; let the env's own `max_steps` win.
   - In the README, be explicit that the final showcase evaluation uses the
     OpenEnv `EnvClient` WebSocket session on the live Space, plus the
     wildfire-specific `POST /grader` for scoring.

6. Checkpoint backup
   - Upload `grpo_wildfire/` to the HF model repo:
     `Chunchunmaru-101/wildfire-grpo-checkpoints`.
   - Do not commit `grpo_wildfire/` to this repo.

7. Hugging Face Space sanity
   - Verify before submission:
     - `https://chunchunmaru-101-wildfire-env.hf.space/`
     - `/tasks`
     - `/schema`
     - a `WildfireEnv(...).sync()` round-trip with `reset` + a few `step`s
       (the smoke test loop in this repo's history is the reference)
     - `POST /grader`
   - The final submitted URL should be the HF Space URL.

8. Final repository state
   - Run:
     ```bash
     python submission_check.py --strict
     git status --short
     ```
   - Commit only final source/docs/artifacts.
   - Avoid committing caches, `grpo_wildfire/`, `.venv/`, notebooks with huge
     cell outputs, or large video files.

9. Cost/runtime cleanup
   - Stop or pause the A10G Hugging Face runtime immediately after all artifacts
     and checkpoints are uploaded.

## Important Notes

- Do not commit `grpo_wildfire/`; upload checkpoints to HF model repo instead.
- Commit generated JSON/PNG/summary artifacts.
- Verify the README links `Blog.MD`; add a YouTube link only if a demo video exists.
- The model may not obviously improve in noisy training logs; final proof is
  same-seed OpenEnv WebSocket eval comparing untrained vs trained under
  identical flags.
- If trained eval underperforms, still submit: environment innovation and story are the
  highest-weighted criteria.

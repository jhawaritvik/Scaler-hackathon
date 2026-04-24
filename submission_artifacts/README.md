# Submission Artifacts

Populate this folder after the main GPU training run so the final repository
contains the evidence judges look for.

Recommended workflow from the repo root:

```bash
.\.venv\Scripts\python.exe reward_audit.py --json-out reward_audit.json
.\.venv\Scripts\python.exe train_grpo.py
.\.venv\Scripts\python.exe plot_training_curves.py --log grpo_wildfire/log.jsonl --out-dir submission_artifacts
.\.venv\Scripts\python.exe eval_policy.py --untrained --output submission_artifacts/eval_untrained.json
.\.venv\Scripts\python.exe eval_policy.py --output submission_artifacts/eval_trained.json
.\.venv\Scripts\python.exe submission_check.py --strict
```

Expected files:

- `training_reward_curve.svg`
- `training_loss_curve.svg`
- `training_summary.md`
- `eval_untrained.json`
- `eval_trained.json`

Final manual additions before submission:

- replace the README placeholder with your public HF blog / YouTube demo / slide deck link
- embed `training_reward_curve.svg` and `training_loss_curve.svg` into the root README
- add a short trained-vs-baseline table using the two eval JSON files

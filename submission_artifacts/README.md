# Submission Artifacts

Populate this folder after the main GPU training run so the final repository
contains the evidence judges look for.

Recommended workflow from the repo root:

```bash
.\.venv\Scripts\python.exe reward_audit.py --json-out reward_audit.json
.\.venv\Scripts\python.exe train_grpo.py
.\.venv\Scripts\python.exe plot_training_curves.py
.\.venv\Scripts\python.exe eval_policy_http.py --untrained --base-url https://chunchunmaru-101-wildfire-env.hf.space --output submission_artifacts/eval_untrained.json
.\.venv\Scripts\python.exe eval_policy_http.py --base-url https://chunchunmaru-101-wildfire-env.hf.space --output submission_artifacts/eval_trained.json
.\.venv\Scripts\python.exe submission_check.py --strict
```

Expected files:

- `training_reward_curve.png`
- `training_loss_curve.png`
- `training_summary.md`
- `eval_untrained.json`
- `eval_trained.json`

Final manual additions before submission:

- replace the README placeholder with your public HF blog / YouTube demo / slide deck link
- embed `training_reward_curve.png` and `training_loss_curve.png` into the root README
- add a short trained-vs-baseline table using the two eval JSON files

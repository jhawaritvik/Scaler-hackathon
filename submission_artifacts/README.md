# Submission Artifacts

Populate this folder after the main GPU training run so the final repository
contains the evidence judges look for.

**Training length (submission).** The adapter evaluated in `eval_trained.json`
is from a **20-GRPO-iteration** `deadline_v2_a10g` run (see
`notebooks/wildfire_training_eval_hf.ipynb` and `Blog.MD`), not the 60-iter
default of `python train_grpo.py` with no args.

Recommended workflow from the repo root:

```bash
.\.venv\Scripts\python.exe reward_audit.py --json-out reward_audit.json
.\.venv\Scripts\python.exe train_grpo.py
.\.venv\Scripts\python.exe plot_training_curves.py
.\.venv\Scripts\python.exe eval_policy_http.py --untrained --base-url https://chunchunmaru-101-wildfire-env.hf.space --seeds-per-task 5 --output submission_artifacts/eval_untrained.json
.\.venv\Scripts\python.exe eval_policy_http.py --base-url https://chunchunmaru-101-wildfire-env.hf.space --seeds-per-task 5 --output submission_artifacts/eval_trained.json
.\.venv\Scripts\python.exe submission_check.py --strict
```

Expected files:

- `training_reward_curve.png`
- `training_loss_curve.png`
- `training_summary.md`
- `eval_untrained.json`
- `eval_trained.json`

Final manual additions before submission:

- confirm the root README links the separate `Blog.MD` file in the HF Space
- add the YouTube demo link to the root README if a video is created
- embed `training_reward_curve.png` and `training_loss_curve.png` into the root README
- add a short trained-vs-baseline table using the two eval JSON files

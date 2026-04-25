# Environment Review

This repository packages a wildfire incident-command environment plus the
training, evaluation, and demo tooling needed to measure how a 4B Qwen policy
behaves on the current task set.

## Current state

- The environment exposes three seeded tasks: `easy`, `medium`, and `hard`.
- `train_grpo.py` is the main GRPO fine-tuning entrypoint for
  `Qwen/Qwen3-4B-Instruct-2507`.
- `eval_policy_http.py` runs the held-out seed bank through the live OpenEnv
  HTTP endpoints for reproducible trained vs. untrained comparisons.
- `capture_replay.py` and `/viewer` provide the visual demo path for recorded
  or live episodes.

## Run artifacts

Each training run now writes these files into its output directory:

- `config.json`: resolved training config used for the run
- `task_catalog.json`: task curriculum, difficulty specs, and seed banks
- `run_status.json`: current state of the run for restart/debug visibility
- `checkpoint_index.json`: latest, final, per-task best, and snapshot pointers
- `latest_metrics.json`: the newest iteration metrics
- `log.jsonl`: full iteration-by-iteration metrics log
- `latest/`: rolling restart checkpoint
- `best_adapter_easy/`, `best_adapter_medium/`, `best_adapter_hard/`
- `final_adapter/`

This layout is meant to be friendly to Hugging Face GPU jobs,
where a run may need to be resumed or inspected after an interrupted session.

## Post-run recording

After training, use `plot_training_curves.py` and `eval_policy_http.py` to export:

- training reward and optimization plots
- `eval_untrained.json`
- `eval_trained.json`

The output lands in `submission_artifacts/` by default and is the cleanest way
to produce a submission-ready performance bundle for the current HEAD.

## Demo path

The visual demo workflow is:

1. Train or pick an adapter.
2. Capture a replay with `capture_replay.py`.
3. Open `/viewer?replay=...` or stream `/demo` live.

This keeps the qualitative review path aligned with the exact task and seed
configuration used by the environment.

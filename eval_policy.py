"""Compatibility CLI for OpenEnv policy evaluation.

The implementation lives in :mod:`eval_policy_http` (WebSocket session + grader).
This module exists so older docs and
``notebooks/wildfire_train_eval_hf.ipynb`` (or legacy stubs) can run ``eval_policy.py`` via subprocess
unchanged.
"""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("eval_policy_http.py")), run_name="__main__")

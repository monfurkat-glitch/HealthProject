"""Experiment tracking: every run is appended as one row to experiments/runs.csv.

Each row records when the run happened, the code version (git commit), the model and its
settings, which features were used, and the metrics on the validation set. The file is
committed with the code, so every number in the README can be traced back to a run.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.preprocess import REPO_ROOT

RUNS_PATH = REPO_ROOT / "experiments" / "runs.csv"
# Read these as text: a commit hash like "1913e50" would otherwise be parsed as a number.
TEXT_COLUMNS = {"git_commit": str, "run_name": str, "params": str, "notes": str}


def git_commit() -> str:
    """Short hash of the current commit, with '+dirty' if there are uncommitted changes."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        # The log itself changes with every run, so it does not count as a code change
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--",
                                ".", ":!experiments/runs.csv"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_run(experiment: str, run_name: str, model: str, params: dict, include_sms: bool,
            metrics: dict[str, float], split: str = "validation", notes: str = "",
            path: Path = RUNS_PATH) -> dict:
    """Append one run to the log and return the logged row."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "experiment": experiment,
        "run_name": run_name,
        "model": model,
        "params": json.dumps(params, sort_keys=True),
        "include_sms": include_sms,
        "split": split,
        **{k: round(float(v), 4) for k, v in metrics.items()},
        "notes": notes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        frame = pd.concat([load_runs(path), frame], ignore_index=True)
    frame.to_csv(path, index=False)
    return row


def load_runs(path: Path = RUNS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, dtype=TEXT_COLUMNS, keep_default_na=False) if path.exists() else pd.DataFrame()

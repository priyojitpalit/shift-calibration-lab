from __future__ import annotations

from pathlib import Path

from scalab.config import load_config
from scalab.experiment.runner import ExperimentRunner


def run_experiment(config_path: Path) -> dict:
    config = load_config(config_path)
    runner = ExperimentRunner(config=config, config_path=config_path)
    return runner.run()

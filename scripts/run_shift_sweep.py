from pathlib import Path
from scalab.runtime import run_experiment

if __name__ == "__main__":
    run_experiment(Path("configs/digits_shift_sweep.yaml"))

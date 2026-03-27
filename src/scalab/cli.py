from __future__ import annotations

import argparse
from pathlib import Path

from scalab.runtime import run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scalab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run an experiment from a YAML config")
    run_parser.add_argument("--config", required=True, type=Path, help="Path to YAML configuration")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_experiment(args.config)

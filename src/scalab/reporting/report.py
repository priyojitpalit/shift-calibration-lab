from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def build_markdown_report(
    output_dir: Path,
    experiment_name: str,
    clean_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    scalar_temperature: float | None,
    adaptive_summary: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# {experiment_name}")
    lines.append("")
    lines.append(f"Generated: {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append("## Clean Test Performance")
    lines.append("")
    lines.append(clean_df.to_markdown(index=False))
    lines.append("")
    if scalar_temperature is not None:
        lines.append(f"Scalar temperature selected on calibration split: {scalar_temperature:.4f}")
        lines.append("")
    lines.append(f"Adaptive calibration summary: {adaptive_summary}")
    lines.append("")
    if not shift_df.empty:
        lines.append("## Shift Stress Test")
        lines.append("")
        grouped = shift_df.groupby(["shift_family", "method"])[["accuracy", "ece", "nll"]].mean().reset_index()
        lines.append(grouped.to_markdown(index=False))
        lines.append("")
        worst = shift_df.sort_values("ece", ascending=False).head(10)
        lines.append("## Highest ECE Cases")
        lines.append("")
        lines.append(worst.to_markdown(index=False))
        lines.append("")
    lines.append("## Output Files")
    lines.append("")
    for artifact in sorted(output_dir.rglob("*")):
        if artifact.is_file() and artifact.name != "report.md":
            lines.append(f"- `{artifact.relative_to(output_dir)}`")
    lines.append("")
    return "\n".join(lines)

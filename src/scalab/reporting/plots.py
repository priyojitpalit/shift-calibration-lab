from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scalab.metrics.calibration import reliability_bins


def plot_reliability(probabilities, y_true, bins: int, title: str, path: Path) -> None:
    rows = reliability_bins(probabilities, y_true, bins)
    df = pd.DataFrame(rows)
    centers = (df["bin_lower"] + df["bin_upper"]) / 2.0

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.bar(centers, df["accuracy"], width=1.0 / bins, alpha=0.8, align="center")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_confidence_histogram(probabilities, title: str, path: Path) -> None:
    confidence = probabilities.max(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(confidence, bins=15)
    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric_by_severity(df: pd.DataFrame, metric: str, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in df.groupby("method"):
        ordered = group.sort_values("severity")
        ax.plot(ordered["severity"], ordered[metric], marker="o", label=method)
    ax.set_xlabel("Severity")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

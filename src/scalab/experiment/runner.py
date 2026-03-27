from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from scalab.calibration.adaptive import AdaptiveTemperatureScaler
from scalab.calibration.temperature import ScalarTemperatureScaler
from scalab.config import AppConfig
from scalab.datasets.registry import load_dataset
from scalab.experiment.records import EvaluationRecord
from scalab.metrics.summary import summarize_probabilities
from scalab.models.registry import build_model
from scalab.reporting.plots import plot_confidence_histogram, plot_metric_by_severity, plot_reliability
from scalab.reporting.report import build_markdown_report
from scalab.shifts.registry import apply_shift
from scalab.utils.io import ensure_dir, write_csv, write_json, write_text
from scalab.utils.randomness import seed_everything


class ExperimentRunner:
    def __init__(self, config: AppConfig, config_path: Path) -> None:
        self.config = config
        self.config_path = config_path
        self.output_dir = ensure_dir(Path(config.experiment.output_dir))
        self.artifacts_dir = ensure_dir(self.output_dir / "artifacts")
        self.plots_dir = ensure_dir(self.output_dir / "plots")
        self.tables_dir = ensure_dir(self.output_dir / "tables")

    def run(self) -> dict:
        seed_everything(self.config.experiment.seed)
        dataset = load_dataset(self.config.dataset, seed=self.config.experiment.seed)
        model = build_model(self.config.model, seed=self.config.experiment.seed)
        model.fit(dataset.x_train, dataset.y_train)

        val_bundle = model.predict_bundle(dataset.x_validation)
        cal_bundle = model.predict_bundle(dataset.x_calibration)
        test_bundle = model.predict_bundle(dataset.x_test)

        scalar_scaler = None
        if self.config.calibration.scalar_temperature.enabled:
            scalar_scaler = ScalarTemperatureScaler().fit(
                logits=cal_bundle.logits,
                y_true=dataset.y_calibration,
                search_bounds=self.config.calibration.scalar_temperature.search_bounds,
                grid_points=self.config.calibration.scalar_temperature.grid_points,
            )

        adaptive_scaler = None
        if self.config.calibration.adaptive_temperature.enabled:
            adaptive_scaler = AdaptiveTemperatureScaler(
                learning_rate=self.config.calibration.adaptive_temperature.learning_rate,
                max_iter=self.config.calibration.adaptive_temperature.max_iter,
                l2=self.config.calibration.adaptive_temperature.l2,
                include_severity=self.config.calibration.adaptive_temperature.include_severity,
            ).fit(
                logits=cal_bundle.logits,
                probabilities=cal_bundle.probabilities,
                x=dataset.x_calibration,
                y_true=dataset.y_calibration,
                severity=0.0,
            )

        clean_records: list[EvaluationRecord] = []
        shift_records: list[EvaluationRecord] = []

        clean_records.extend(
            self._evaluate_condition(
                partition="test",
                condition="clean",
                shift_family="none",
                severity=0.0,
                x=dataset.x_test,
                y=dataset.y_test,
                logits=test_bundle.logits,
                base_probabilities=test_bundle.probabilities,
                scalar_scaler=scalar_scaler,
                adaptive_scaler=adaptive_scaler,
            )
        )

        if self.config.shift.families:
            for family_index, family in enumerate(self.config.shift.families):
                for sev_index, severity in enumerate(family.severities):
                    shifted_x = apply_shift(
                        name=family.name,
                        x=dataset.x_test,
                        severity=severity,
                        image_shape=dataset.image_shape,
                        seed=self.config.experiment.seed + family_index * 100 + sev_index,
                    )
                    shifted_bundle = model.predict_bundle(shifted_x)
                    shift_records.extend(
                        self._evaluate_condition(
                            partition="test",
                            condition="shifted",
                            shift_family=family.name,
                            severity=float(severity),
                            x=shifted_x,
                            y=dataset.y_test,
                            logits=shifted_bundle.logits,
                            base_probabilities=shifted_bundle.probabilities,
                            scalar_scaler=scalar_scaler,
                            adaptive_scaler=adaptive_scaler,
                        )
                    )

                    plot_reliability(
                        probabilities=shifted_bundle.probabilities,
                        y_true=dataset.y_test,
                        bins=self.config.reporting.reliability_bins,
                        title=f"Reliability: {family.name} severity={severity} uncalibrated",
                        path=self.plots_dir / f"reliability_{family.name}_{str(severity).replace('.', '_')}_uncalibrated.png",
                    )

        clean_df = pd.DataFrame([r.to_dict() for r in clean_records])
        shift_df = pd.DataFrame([r.to_dict() for r in shift_records])

        write_csv(self.tables_dir / "clean_metrics_summary.csv", clean_df.to_dict(orient="records"))
        write_csv(self.tables_dir / "shift_metrics_summary.csv", shift_df.to_dict(orient="records"))

        metadata = {
            "config_path": str(self.config_path),
            "config": self._serialize_config(),
            "scalar_temperature": None if scalar_scaler is None else scalar_scaler.temperature,
            "adaptive_loss_tail": [] if adaptive_scaler is None else adaptive_scaler.loss_history[-10:],
        }
        write_json(self.artifacts_dir / "run_metadata.json", metadata)

        if self.config.reporting.save_probabilities:
            self._save_probability_artifacts(dataset=dataset, test_bundle=test_bundle, scalar_scaler=scalar_scaler, adaptive_scaler=adaptive_scaler)

        plot_reliability(
            probabilities=test_bundle.probabilities,
            y_true=dataset.y_test,
            bins=self.config.reporting.reliability_bins,
            title="Reliability: clean test uncalibrated",
            path=self.plots_dir / "reliability_clean_uncalibrated.png",
        )
        plot_confidence_histogram(
            probabilities=test_bundle.probabilities,
            title="Confidence histogram: clean test uncalibrated",
            path=self.plots_dir / "confidence_histogram_clean_uncalibrated.png",
        )

        if scalar_scaler is not None:
            scalar_probs = scalar_scaler.transform(test_bundle.logits)
            plot_reliability(
                probabilities=scalar_probs,
                y_true=dataset.y_test,
                bins=self.config.reporting.reliability_bins,
                title="Reliability: clean test scalar temperature",
                path=self.plots_dir / "reliability_clean_scalar_temperature.png",
            )

        if adaptive_scaler is not None:
            adaptive_probs = adaptive_scaler.transform(
                logits=test_bundle.logits,
                probabilities=test_bundle.probabilities,
                x=dataset.x_test,
                severity=0.0,
            )
            plot_reliability(
                probabilities=adaptive_probs,
                y_true=dataset.y_test,
                bins=self.config.reporting.reliability_bins,
                title="Reliability: clean test adaptive temperature",
                path=self.plots_dir / "reliability_clean_adaptive_temperature.png",
            )

        if not shift_df.empty:
            plot_metric_by_severity(
                df=shift_df,
                metric="ece",
                path=self.plots_dir / "ece_by_severity.png",
                title="ECE under distribution shift",
            )
            plot_metric_by_severity(
                df=shift_df,
                metric="nll",
                path=self.plots_dir / "nll_by_severity.png",
                title="NLL under distribution shift",
            )
            plot_metric_by_severity(
                df=shift_df,
                metric="accuracy",
                path=self.plots_dir / "accuracy_by_severity.png",
                title="Accuracy under distribution shift",
            )

        adaptive_summary = "not enabled"
        if adaptive_scaler is not None:
            adaptive_summary = f"{len(adaptive_scaler.loss_history)} optimization steps; final loss={adaptive_scaler.loss_history[-1]:.6f}"

        report = build_markdown_report(
            output_dir=self.output_dir,
            experiment_name=self.config.experiment.name,
            clean_df=clean_df,
            shift_df=shift_df,
            scalar_temperature=None if scalar_scaler is None else scalar_scaler.temperature,
            adaptive_summary=adaptive_summary,
        )
        write_text(self.output_dir / "report.md", report)

        payload = {
            "clean": clean_df.to_dict(orient="records"),
            "shifted": shift_df.to_dict(orient="records"),
        }
        write_json(self.output_dir / "clean_vs_shifted.json", payload)
        return payload

    def _evaluate_condition(
        self,
        partition: str,
        condition: str,
        shift_family: str,
        severity: float,
        x: np.ndarray,
        y: np.ndarray,
        logits: np.ndarray,
        base_probabilities: np.ndarray,
        scalar_scaler: ScalarTemperatureScaler | None,
        adaptive_scaler: AdaptiveTemperatureScaler | None,
    ) -> list[EvaluationRecord]:
        rows: list[EvaluationRecord] = []
        method_map = {"uncalibrated": base_probabilities}
        if scalar_scaler is not None:
            method_map["scalar_temperature"] = scalar_scaler.transform(logits)
        if adaptive_scaler is not None:
            method_map["adaptive_temperature"] = adaptive_scaler.transform(
                logits=logits,
                probabilities=base_probabilities,
                x=x,
                severity=severity,
            )

        for method_name, probabilities in method_map.items():
            metrics = summarize_probabilities(probabilities=probabilities, y_true=y, bins=self.config.reporting.reliability_bins)
            rows.append(
                EvaluationRecord(
                    dataset_partition=partition,
                    condition=condition,
                    shift_family=shift_family,
                    severity=float(severity),
                    method=method_name,
                    accuracy=metrics["accuracy"],
                    nll=metrics["nll"],
                    brier=metrics["brier"],
                    ece=metrics["ece"],
                    mce=metrics["mce"],
                    mean_confidence=metrics["mean_confidence"],
                )
            )
        return rows

    def _serialize_config(self) -> dict:
        config = self.config
        return {
            "experiment": asdict(config.experiment),
            "dataset": {
                "name": config.dataset.name,
                "normalize": config.dataset.normalize,
                "split": asdict(config.dataset.split),
            },
            "model": asdict(config.model),
            "calibration": {
                "scalar_temperature": asdict(config.calibration.scalar_temperature),
                "adaptive_temperature": asdict(config.calibration.adaptive_temperature),
            },
            "shift": {
                "families": [asdict(family) for family in config.shift.families],
            },
            "reporting": asdict(config.reporting),
        }

    def _save_probability_artifacts(self, dataset, test_bundle, scalar_scaler, adaptive_scaler) -> None:
        rows = []
        base_conf = test_bundle.probabilities.max(axis=1)
        for idx, (truth, pred, conf) in enumerate(zip(dataset.y_test, test_bundle.predictions, base_conf)):
            row = {
                "index": idx,
                "true_label": int(truth),
                "uncalibrated_prediction": int(pred),
                "uncalibrated_confidence": float(conf),
            }
            if scalar_scaler is not None:
                scalar_probs = scalar_scaler.transform(test_bundle.logits)
                row["scalar_prediction"] = int(scalar_probs[idx].argmax())
                row["scalar_confidence"] = float(scalar_probs[idx].max())
            if adaptive_scaler is not None:
                adaptive_probs = adaptive_scaler.transform(
                    logits=test_bundle.logits,
                    probabilities=test_bundle.probabilities,
                    x=dataset.x_test,
                    severity=0.0,
                )
                row["adaptive_prediction"] = int(adaptive_probs[idx].argmax())
                row["adaptive_confidence"] = float(adaptive_probs[idx].max())
            rows.append(row)
        write_csv(self.tables_dir / "per_sample_clean_predictions.csv", rows)

# digits_baseline

Generated: 2026-03-26T11:54:56.214282Z

## Clean Test Performance

| dataset_partition   | condition   | shift_family   |   severity | method               |   accuracy |      nll |     brier |       ece |      mce |   mean_confidence |
|:--------------------|:------------|:---------------|-----------:|:---------------------|-----------:|---------:|----------:|----------:|---------:|------------------:|
| test                | clean       | none           |          0 | uncalibrated         |   0.955556 | 0.17625  | 0.077787  | 0.0518254 | 0.67961  |          0.909374 |
| test                | clean       | none           |          0 | scalar_temperature   |   0.955556 | 0.165348 | 0.0717932 | 0.0255363 | 0.566368 |          0.967795 |
| test                | clean       | none           |          0 | adaptive_temperature |   0.955556 | 0.183422 | 0.0799555 | 0.0592639 | 0.694969 |          0.901842 |

Scalar temperature selected on calibration split: 0.5000

Adaptive calibration summary: 700 optimization steps; final loss=0.107777

## Output Files

- `artifacts/run_metadata.json`
- `plots/confidence_histogram_clean_uncalibrated.png`
- `plots/reliability_clean_adaptive_temperature.png`
- `plots/reliability_clean_scalar_temperature.png`
- `plots/reliability_clean_uncalibrated.png`
- `tables/clean_metrics_summary.csv`
- `tables/per_sample_clean_predictions.csv`
- `tables/shift_metrics_summary.csv`

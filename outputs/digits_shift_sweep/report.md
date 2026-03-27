# digits_shift_sweep

Generated: 2026-03-27T03:16:05.778271Z

## Clean Test Performance

| dataset_partition   | condition   | shift_family   |   severity | method               |   accuracy |      nll |     brier |       ece |      mce |   mean_confidence |
|:--------------------|:------------|:---------------|-----------:|:---------------------|-----------:|---------:|----------:|----------:|---------:|------------------:|
| test                | clean       | none           |          0 | uncalibrated         |   0.944444 | 0.161921 | 0.0773138 | 0.0378004 | 0.567404 |          0.948282 |
| test                | clean       | none           |          0 | scalar_temperature   |   0.944444 | 0.165626 | 0.077087  | 0.0336884 | 0.561577 |          0.936588 |
| test                | clean       | none           |          0 | adaptive_temperature |   0.944444 | 0.165847 | 0.0764749 | 0.0353621 | 0.497314 |          0.932587 |

Scalar temperature selected on calibration split: 1.1203

Adaptive calibration summary: 900 optimization steps; final loss=0.148438

## Shift Stress Test

| shift_family    | method               |   accuracy |       ece |      nll |
|:----------------|:---------------------|-----------:|----------:|---------:|
| blur            | adaptive_temperature |   0.837037 | 0.228076  | 0.722927 |
| blur            | scalar_temperature   |   0.837037 | 0.211704  | 0.695204 |
| blur            | uncalibrated         |   0.837037 | 0.171536  | 0.644823 |
| contrast_scale  | adaptive_temperature |   0.942593 | 0.168647  | 0.345194 |
| contrast_scale  | scalar_temperature   |   0.942593 | 0.160253  | 0.331254 |
| contrast_scale  | uncalibrated         |   0.942593 | 0.129368  | 0.288507 |
| gaussian_noise  | adaptive_temperature |   0.941481 | 0.0329939 | 0.181016 |
| gaussian_noise  | scalar_temperature   |   0.941481 | 0.0325888 | 0.18111  |
| gaussian_noise  | uncalibrated         |   0.941481 | 0.0296748 | 0.178325 |
| intensity_shift | adaptive_temperature |   0.947222 | 0.0430346 | 0.168098 |
| intensity_shift | scalar_temperature   |   0.947222 | 0.0423449 | 0.167205 |
| intensity_shift | uncalibrated         |   0.947222 | 0.0338227 | 0.162429 |
| pixel_dropout   | adaptive_temperature |   0.928704 | 0.0593548 | 0.240047 |
| pixel_dropout   | scalar_temperature   |   0.928704 | 0.0519852 | 0.236382 |
| pixel_dropout   | uncalibrated         |   0.928704 | 0.0397404 | 0.226569 |

## Highest ECE Cases

| dataset_partition   | condition   | shift_family   |   severity | method               |   accuracy |      nll |    brier |      ece |      mce |   mean_confidence |
|:--------------------|:------------|:---------------|-----------:|:---------------------|-----------:|---------:|---------:|---------:|---------:|------------------:|
| test                | shifted     | contrast_scale |       0.35 | adaptive_temperature |   0.940741 | 0.600146 | 0.243802 | 0.33778  | 0.512381 |          0.60296  |
| test                | shifted     | contrast_scale |       0.35 | scalar_temperature   |   0.940741 | 0.56492  | 0.22741  | 0.318892 | 0.469921 |          0.621849 |
| test                | shifted     | blur           |       3    | adaptive_temperature |   0.755556 | 1.05279  | 0.478377 | 0.288872 | 0.502759 |          0.472818 |
| test                | shifted     | blur           |       3    | scalar_temperature   |   0.755556 | 1.01302  | 0.463585 | 0.266981 | 0.815021 |          0.494786 |
| test                | shifted     | contrast_scale |       0.35 | uncalibrated         |   0.940741 | 0.484953 | 0.192116 | 0.265985 | 0.423327 |          0.674756 |
| test                | shifted     | blur           |       2    | adaptive_temperature |   0.833333 | 0.734967 | 0.333515 | 0.247189 | 0.383589 |          0.589507 |
| test                | shifted     | blur           |       2    | scalar_temperature   |   0.833333 | 0.703625 | 0.321418 | 0.229206 | 0.422373 |          0.607567 |
| test                | shifted     | blur           |       3    | uncalibrated         |   0.755556 | 0.955855 | 0.440442 | 0.225157 | 0.811973 |          0.53675  |
| test                | shifted     | blur           |       2    | uncalibrated         |   0.833333 | 0.646498 | 0.299146 | 0.184543 | 0.300427 |          0.650769 |
| test                | shifted     | contrast_scale |       0.5  | adaptive_temperature |   0.944444 | 0.349707 | 0.135185 | 0.178465 | 0.417938 |          0.767399 |

## Output Files

- `artifacts\run_metadata.json`
- `plots\accuracy_by_severity.png`
- `plots\confidence_histogram_clean_uncalibrated.png`
- `plots\ece_by_severity.png`
- `plots\nll_by_severity.png`
- `plots\reliability_blur_1_uncalibrated.png`
- `plots\reliability_blur_2_uncalibrated.png`
- `plots\reliability_blur_3_uncalibrated.png`
- `plots\reliability_clean_adaptive_temperature.png`
- `plots\reliability_clean_scalar_temperature.png`
- `plots\reliability_clean_uncalibrated.png`
- `plots\reliability_contrast_scale_0_35_uncalibrated.png`
- `plots\reliability_contrast_scale_0_5_uncalibrated.png`
- `plots\reliability_contrast_scale_0_65_uncalibrated.png`
- `plots\reliability_contrast_scale_0_8_uncalibrated.png`
- `plots\reliability_gaussian_noise_0_05_uncalibrated.png`
- `plots\reliability_gaussian_noise_0_15_uncalibrated.png`
- `plots\reliability_gaussian_noise_0_1_uncalibrated.png`
- `plots\reliability_gaussian_noise_0_2_uncalibrated.png`
- `plots\reliability_gaussian_noise_0_3_uncalibrated.png`
- `plots\reliability_intensity_shift_0_05_uncalibrated.png`
- `plots\reliability_intensity_shift_0_15_uncalibrated.png`
- `plots\reliability_intensity_shift_0_1_uncalibrated.png`
- `plots\reliability_intensity_shift_0_2_uncalibrated.png`
- `plots\reliability_pixel_dropout_0_05_uncalibrated.png`
- `plots\reliability_pixel_dropout_0_15_uncalibrated.png`
- `plots\reliability_pixel_dropout_0_1_uncalibrated.png`
- `plots\reliability_pixel_dropout_0_25_uncalibrated.png`
- `tables\clean_metrics_summary.csv`
- `tables\per_sample_clean_predictions.csv`
- `tables\shift_metrics_summary.csv`

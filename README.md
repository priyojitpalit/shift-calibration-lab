# Shift Calibration Lab

Shift Calibration Lab is a research-oriented experimentation framework for studying confidence calibration under controlled distribution shift. The project reproduces the central temperature-scaling result from Guo et al. and extends it with adaptive temperature scaling driven by input-level uncertainty features.

---

## Research Objective

This project addresses a focused and practically relevant question in trustworthy machine learning:

> How stable is post-hoc calibration when a classifier is deployed on inputs whose distribution has shifted in controlled, structured ways?

While modern classifiers can achieve high accuracy, their confidence estimates often degrade under distribution shift. This repository provides a controlled experimental environment to systematically evaluate that phenomenon.

---

## Key Findings

Across multiple controlled perturbation families (e.g., Gaussian noise, contrast scaling) and increasing severity levels, we observe:

- **Accuracy remains relatively stable** under moderate distribution shift  
- **Calibration error (ECE, MCE) increases and becomes unstable** as shift severity increases  
- **Standard temperature scaling does not reliably correct miscalibration under shift**, and in some cases worsens it  
- **Adaptive temperature scaling provides limited improvement**, but does not fully resolve shift-induced miscalibration  

These results highlight a fundamental gap between predictive performance and confidence reliability in deployed machine learning systems.

---

## Experimental Capabilities

The framework supports three primary experiment classes:

1. Baseline miscalibration measurement on in-distribution data  
2. Post-hoc calibration using scalar temperature scaling  
3. Calibration stress testing under distribution shift, including adaptive temperature models  

Core features include:

- deterministic experiment orchestration  
- train/validation/calibration/test split management  
- configurable image transformations with severity schedules  
- multiclass metrics: accuracy, NLL, Brier score, ECE, MCE  
- reliability diagrams and confidence histograms  
- scalar temperature scaling  
- feature-conditioned adaptive temperature scaling  
- batch experiment execution across shift families and severity levels  
- structured JSON artifact export  
- automated markdown and CSV report generation  

---

## Project Layout

```text
shift-calibration-lab/
├── configs/
│   ├── digits_baseline.yaml
│   └── digits_shift_sweep.yaml
├── scripts/
│   ├── run_baseline.py
│   ├── run_shift_sweep.py
│   └── package_project.py
├── src/scalab/
│   ├── cli.py
│   ├── config.py
│   ├── runtime.py
│   ├── datasets/
│   ├── models/
│   ├── calibration/
│   ├── shifts/
│   ├── metrics/
│   ├── reporting/
│   └── utils/
├── tests/
└── outputs/
```

---

## Quick Start

Create an environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run a baseline calibration experiment:

```bash
python -m scalab.cli run --config configs/digits_baseline.yaml
```

Run a distribution shift experiment:

```bash
python -m scalab.cli run --config configs/digits_shift_sweep.yaml
```

---

## Primary Experimental Flow

1. Load dataset (digits classification task)  
2. Train a probabilistic classifier  
3. Split into training, calibration, and test sets  
4. Fit scalar temperature scaling on calibration data  
5. Fit adaptive temperature scaling using sample-level features  
6. Evaluate all models on clean (in-distribution) test data  
7. Apply structured distribution shifts with increasing severity  
8. Re-evaluate calibration and accuracy under shift  
9. Export metrics, plots, and structured reports  

---

## Adaptive Temperature Model

The adaptive model estimates a sample-specific temperature using a compact feature representation derived from:

- top-1 confidence  
- margin between top-1 and top-2 probabilities  
- normalized logit entropy  
- input intensity statistics  
- shift severity metadata  

The mapping is constrained to produce strictly positive temperatures and is optimized using validation negative log likelihood.

---

## Representative Output Artifacts

Each experiment produces structured outputs including:

- `metrics_summary.csv`  
- `shift_metrics_summary.csv`  
- `clean_vs_shifted.json`  
- `reliability_clean.png`  
- `reliability_shift_<type>.png`  
- `ece_by_severity.png`  
- `report.md`  

---

## Example Results

Representative outputs from a completed run are available in `outputs/example/`.

These include calibration metrics and evaluation results under distribution shift.

---

## Reproducibility

All experiments support deterministic execution via fixed random seeds. Generated reports include:

- random seed  
- dataset splits  
- shift family and severity  
- model hyperparameters  
- calibration parameters  

---

## Citation Context

This work builds on:

Guo et al., *On Calibration of Modern Neural Networks*

and extends the analysis to explicitly study calibration behavior under controlled distribution shift.

---

## Intended Use

This repository is designed for:

- research experiments in calibration and uncertainty  
- instructional modules on trustworthy AI  
- reproducible evaluation of calibration methods under shift  
- benchmarking alternative calibration approaches  

---

## License

This project is released under the MIT License.

---


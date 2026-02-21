Raisin Binary Classification
Logistic Regression From Scratch (NumPy Only)

A production-oriented implementation of binary logistic regression built entirely from scratch using only:

Built-in Python libraries

NumPy

This project demonstrates:

Correct mathematical implementation

Deterministic training

Lightweight experiment tracking (MiniFlow)

Model registry concept

Drift monitoring

Feature-level explainability

Minimal HTTP inference server

Lean containerisation ready for Kubernetes

No sklearn. No pandas. No MLflow. No web frameworks.

Overview

The Raisin dataset contains:

900 samples

7 numerical features

2 classes (Kecimen, Besni)

The objective is not simply classification accuracy, but to demonstrate:

Optimisation from first principles

Reproducibility

Model lifecycle thinking

Monitoring and drift detection

Production-ready design with minimal dependencies

Model

Binary logistic regression:

𝑝
(
𝑦
=
1
∣
𝑥
)
=
𝜎
(
𝑤
𝑇
𝑥
+
𝑏
)
p(y=1∣x)=σ(w
T
x+b)

Where:

𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)=
1+e
−z
1
	​

Training Configuration

Loss: Binary cross-entropy

Optimiser: Batch gradient descent

Feature scaling: Standardisation (train statistics only)

Stable sigmoid via clipped logits

Probability clipping inside loss

Gradients
∇
𝑤
=
1
𝑛
𝑋
𝑇
(
𝑝
−
𝑦
)
∇
w
	​

=
n
1
	​

X
T
(p−y)
∇
𝑏
=
1
𝑛
∑
(
𝑝
−
𝑦
)
∇
b
	​

=
n
1
	​

∑(p−y)

Optional extension: L2 regularisation.

Project Structure
.
├── model.py          # Logistic regression implementation
├── train.py          # Deterministic training pipeline
├── server.py         # Minimal inference server (http.server)
├── miniflow.py       # Lightweight experiment tracker
├── monitoring.py     # Drift detection utilities
├── requirements.txt
├── Dockerfile
└── miniflow/         # Generated experiment runs
Training
Install
pip install -r requirements.txt
Run Training
python train.py

Outputs:

model.npz (weights, bias, scaler, feature order)

MiniFlow run directory

Validation metrics printed to console

Each training run logs parameters, metrics, artifacts, and a training data profile.

MiniFlow — MLflow From Scratch

MiniFlow is a minimal experiment tracking implementation.

Experiment Structure
miniflow/
└── experiments/
    └── raisin_logreg/
        └── runs/
            └── <run_id>/
                ├── meta.json
                ├── params.json
                ├── metrics.jsonl
                └── artifacts/
                    ├── model/
                    │   ├── model.npz
                    │   └── model_card.json
                    └── train_profile.json
Registry

Production model mapping is stored in:

miniflow/registry.json

Example:

{
  "raisin_logreg": {
    "production": "a3f92cbb12e4",
    "updated_ts_ms": 1730000000000
  }
}

Promotion rule can be validation accuracy-based.

What This Demonstrates

Reproducibility

Artifact versioning

Run lineage

Promotion workflow

Lightweight model lifecycle management

Continuous Evaluation

Strategy:

Fixed validation split (deterministic seed)

Metrics logged per run

Compare new runs to production run

Promote only if validation improves

Tracked metrics:

Training loss

Validation loss

Training accuracy

Validation accuracy

Metrics are append-only and timestamped.

Drift Monitoring

During training, we store:

Feature means

Feature standard deviations

Feature ordering

At inference, drift score is computed as:

mean
(
∣
𝑧
𝑠
ℎ
𝑖
𝑓
𝑡
∣
)
mean(∣z
shift
	​

∣)

Where:

𝑧
𝑠
ℎ
𝑖
𝑓
𝑡
=
𝜇
𝑛
𝑒
𝑤
−
𝜇
𝑡
𝑟
𝑎
𝑖
𝑛
𝜎
𝑡
𝑟
𝑎
𝑖
𝑛
z
shift
	​

=
σ
train
	​

μ
new
	​

−μ
train
	​

	​


This provides lightweight covariate shift detection without external dependencies.

Explainability

Logistic regression is inherently interpretable.

For a prediction:

logit
=
𝑤
𝑇
𝑥
+
𝑏
logit=w
T
x+b

Feature contribution:

contribution
𝑖
=
𝑤
𝑖
⋅
𝑥
𝑖
contribution
i
	​

=w
i
	​

⋅x
i
	​


Example response:

{
  "pred": 1,
  "p_class1": 0.87,
  "logit": 1.92,
  "feature_contrib_logit": {
    "Area": 0.52,
    "Perimeter": 0.31
  }
}

This enables per-feature transparency.

Inference Server

Implemented using built-in http.server.

No frameworks.

Start Server
python server.py
Health Check
curl http://localhost:8000/healthz
Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Area":87524,"MajorAxisLength":442.24,"MinorAxisLength":253.29,"Eccentricity":0.8197,"ConvexArea":90546,"Extent":0.7586,"Perimeter":1184.04}'

Server characteristics:

Loads model once at startup

Deterministic feature ordering

Stateless

Horizontally scalable

Minimal attack surface

Containerisation

Runtime dependencies:

Python 3.11 slim

NumPy only

Build
docker build -t raisin-infer .
Run
docker run -p 8000:8000 raisin-infer

Lean image properties:

No pandas

No sklearn

No heavy ML libraries

No web frameworks

Kubernetes Deployment

Container characteristics:

Stateless

CPU-light

Memory-light

Horizontally scalable

Recommended limits:

100m CPU

128Mi memory

Readiness and liveness via /healthz.

Suitable for EKS.

Engineering Considerations
Reproducibility

Fixed random seed

Logged hyperparameters

Versioned artifacts

Numerical Stability

Clipped logits

Probability clipping in loss

Zero-variance feature handling

Production Readiness

Deterministic feature ordering

Lightweight model registry

Drift monitoring

Stateless serving

Minimal dependency footprint

Extensions

L2 regularisation

Early stopping

Mini-batch gradient descent

Calibration metrics

Automated promotion thresholds

Alerting on drift threshold

Summary

This repository contains a complete ML system:

Model

Training

Tracking

Registry

Monitoring

Serving

Containerisation

Built from first principles using only foundational tools.

The emphasis is correctness, clarity, reproducibility, and production realism — not library usage.
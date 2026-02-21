# Raisin Binary Classification

## Logistic Regression From Scratch (NumPy Only)

A production-oriented binary classification system implemented entirely
from first principles using:

-   Built-in Python libraries
-   NumPy

This repository demonstrates:

-   Correct mathematical optimisation
-   Deterministic training
-   Lightweight experiment tracking (MiniFlow)
-   Model registry concept
-   Data drift monitoring
-   Feature-level explainability
-   Minimal HTTP inference server
-   Lean containerisation ready for Kubernetes

No sklearn. No pandas. No MLflow. No web frameworks.

------------------------------------------------------------------------

# Overview

The Raisin dataset contains:

-   900 samples
-   7 numerical features
-   2 classes (Kecimen, Besni)

The objective is not simply classification accuracy, but to demonstrate:

-   Mathematical correctness
-   Reproducibility
-   Model lifecycle thinking
-   Monitoring and drift detection
-   Production-ready design with minimal dependencies

------------------------------------------------------------------------

# Model

Binary logistic regression:

p(y=1 \| x) = sigmoid(w\^T x + b)

Where:

sigmoid(z) = 1 / (1 + exp(-z))

## Training Configuration

-   Loss: Binary cross-entropy
-   Optimiser: Batch gradient descent
-   Feature scaling: Standardisation (training statistics only)
-   Stable sigmoid via clipped logits
-   Probability clipping inside loss

## Gradients

dw = (1/n) X\^T (p − y)\
db = (1/n) sum(p − y)

Optional extension: L2 regularisation.

------------------------------------------------------------------------

# Project Structure

    .
    ├── model.py          # Logistic regression implementation
    ├── train.py          # Deterministic training pipeline
    ├── server.py         # Minimal inference server (http.server)
    ├── miniflow.py       # Lightweight experiment tracker
    ├── monitoring.py     # Drift detection utilities
    ├── requirements.txt
    ├── Dockerfile
    └── miniflow/         # Generated experiment runs

------------------------------------------------------------------------

# Training

## Install Dependencies

pip install -r requirements.txt

## Run Training

python train.py

Outputs:

-   model.npz (weights, bias, scaler statistics, feature order)
-   MiniFlow run directory
-   Validation metrics printed to console

------------------------------------------------------------------------

# MiniFlow --- MLflow From Scratch

MiniFlow is a minimal experiment tracking implementation.

## Experiment Structure

miniflow/experiments/`<experiment>`{=html}/runs/`<run_id>`{=html}/

Each run contains:

-   meta.json --- run metadata
-   params.json --- hyperparameters
-   metrics.jsonl --- append-only metric time series
-   artifacts/
    -   model/model.npz
    -   model/model_card.json
    -   train_profile.json

## Registry

Production model mapping is stored in:

miniflow/registry.json

Example:

{ "raisin_logreg": { "production": "a3f92cbb12e4", "updated_ts_ms":
1730000000000 } }

Promotion rule can be validation accuracy-based.

------------------------------------------------------------------------

# Continuous Evaluation

Strategy:

-   Deterministic validation split
-   Metrics logged per run
-   Compare new runs to production run
-   Promote only if validation improves

Tracked metrics:

-   Training loss
-   Validation loss
-   Training accuracy
-   Validation accuracy

------------------------------------------------------------------------

# Drift Monitoring

During training, the system stores:

-   Feature means
-   Feature standard deviations
-   Feature ordering

At inference, drift score is computed as:

mean(abs(z_shift))

Where:

z_shift = (mu_new − mu_train) / sigma_train

------------------------------------------------------------------------

# Explainability

Logistic regression is inherently interpretable.

For a prediction:

logit = w\^T x + b

Feature contribution:

contribution_i = w_i \* x_i

Example response:

{ "pred": 1, "p_class1": 0.87, "logit": 1.92, "feature_contrib_logit": {
"Area": 0.52, "Perimeter": 0.31 } }

------------------------------------------------------------------------

# Inference Server

Implemented using Python's built-in http.server.

Start server:

python server.py

Health check:

curl http://localhost:8000/healthz

Prediction example:

curl -X POST http://localhost:8000/predict -H "Content-Type:
application/json" -d
'{"Area":87524,"MajorAxisLength":442.24,"MinorAxisLength":253.29,"Eccentricity":0.8197,"ConvexArea":90546,"Extent":0.7586,"Perimeter":1184.04}'

------------------------------------------------------------------------

# Containerisation

Runtime dependencies:

-   Python 3.11 slim
-   NumPy only

Build:

docker build -t raisin-infer .

Run:

docker run -p 8000:8000 raisin-infer

------------------------------------------------------------------------

# Kubernetes Deployment

Container characteristics:

-   Stateless
-   CPU-light
-   Memory-light
-   Horizontally scalable

Recommended limits:

-   100m CPU
-   128Mi memory

------------------------------------------------------------------------

# Engineering Considerations

Reproducibility:

-   Fixed random seed
-   Logged hyperparameters
-   Versioned artifacts

Numerical Stability:

-   Clipped logits
-   Probability clipping in loss
-   Zero-variance feature handling

Production Readiness:

-   Deterministic feature ordering
-   Lightweight model registry
-   Drift monitoring
-   Stateless serving
-   Minimal dependency footprint

------------------------------------------------------------------------

# Summary

This repository contains a complete ML system implementing:

-   Model
-   Training
-   Tracking
-   Registry
-   Monitoring
-   Serving
-   Containerisation

Built from first principles using only foundational tools.

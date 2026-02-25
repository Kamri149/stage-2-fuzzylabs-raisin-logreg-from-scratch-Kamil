# Raisin Binary Classification

## Logistic Regression From Scratch (NumPy Only)

A production-oriented binary classification system implemented entirely
from first principles using:

- Built-in Python libraries
- NumPy

This repository demonstrates:

- Correct mathematical optimisation
- Deterministic training
- Feature-level explainability
- Minimal HTTP inference server
- Lean containerisation ready for Kubernetes

No sklearn. No pandas. No MLflow. No web frameworks.

------------------------------------------------------------------------

# Overview

The Raisin dataset (`data/Raisin_Dataset.csv`) contains:

- 900 samples
- 7 numerical features
- 2 classes (Kecimen, Besni)

The goal is not just classification accuracy, but demonstrating:

- Mathematical correctness
- Reproducibility
- Explicit state handling
- Production-style structure with minimal dependencies

------------------------------------------------------------------------

# Model

Binary logistic regression:

    p(y=1 | x) = sigmoid(w^T x + b)

Where:

    sigmoid(z) = 1 / (1 + exp(-z))

## Training Configuration

- Loss: Binary cross-entropy
- Optimiser: Batch gradient descent
- Feature scaling: Standardisation
- Numerically stable sigmoid (clipped logits)
- Probability clipping in loss

Gradients:

    dw = (1/n) X^T (p − y)
    db = (1/n) sum(p − y)

------------------------------------------------------------------------

# Project Structure

    .
    ├── app/
    │   ├── __init__.py
    │   ├── load.py                 # Data loading utilities
    │   ├── model.py                # Logistic regression implementation
    │   ├── train.py                # Deterministic training pipeline
    │   ├── server.py               # Minimal inference server (http.server)
    │   ├── utils.py                # Additional utilities
    │   └── eval/
    │       ├── __init__.py
    │       ├── cv.py               # Cross-validation utilities
    │       └── functions.py        # Validation functions
    │
    ├── models/                     # Trained models are saved here
    ├── tests/                      # Unit tests 
    │       ├── __init__.py
    │       ├── test_inference.py         # testing server and inference behaviour
    │       └── test_model.py             # testing logistic regression model behaviour
    ├── data/                             # Folder where small .csv data file lives (used as a demo)           
    │       └── Raisin_Dataset.csv        # training data
    │
    ├── Dockerfile
    ├── Makefile
    ├── pyproject.toml
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# Training

## Install Dependencies

    pip install -r requirements.txt

## Run Training (from root)

    python -m app.train --csv data/Raisin_Dataset.csv

Using `-m` ensures correct package import resolution and mirrors production execution.

### Default Behaviour

- Train/validation split
- Feature scaling
- Gradient descent optimisation
- Console metrics (loss + accuracy)
- Model saved to:

    models/lg_model.npz

### Cross Validation

    python -m app.train --csv data/Raisin_Dataset.csv --cv 5

Performs stratified K-fold CV and retrains final model using best hyperparameters.

------------------------------------------------------------------------

# Model Artifact

The saved `.npz` contains:

- Weight vector `w`
- Bias `b`
- Scaler mean and std
- Ordered feature names

This avoids pickle and ensures portability.

------------------------------------------------------------------------

# Inference Server

The inference layer is implemented using Python’s built-in `http.server`.

No external web frameworks are used.

## Architectural Design

The server separates concerns clearly:

1. Domain logic (pure functions)
2. Routing layer
3. Thin HTTP handler

### Domain Layer

- `_vectorize_one` ensures deterministic feature ordering
- `validate_payload` enforces schema correctness
- `predict_single` performs full inference pipeline:
    - Scaling
    - Probability computation
    - Logit computation
    - Per-feature contribution breakdown

### Routing Layer

The `route()` function maps:

- `GET /healthz`
- `GET /model`
- `POST /predict`

to their respective handlers.

This keeps request routing separate from HTTP mechanics.

### HTTP Handler

`Handler` only:

- Parses request path
- Delegates to `route`
- Serialises JSON response
- Writes HTTP headers

This mirrors production server patterns where routing logic is separated from transport logic.

------------------------------------------------------------------------

## Start Server

From repository root:

    python -m app.server

Server binds to:

    http://0.0.0.0:8000

(can also be started using `docker run -p 8000:8000 <container name>`; see Containerisation below)

------------------------------------------------------------------------

## Health Check

    curl -v http://localhost:8000/healthz

Response:

    { "status": "ok" }

------------------------------------------------------------------------

## Model Metadata

    curl http://localhost:8000/model

Returns:

- Model artifact path
- Ordered feature list

------------------------------------------------------------------------

## Prediction Endpoint

Endpoint:

    POST /predict

Payload must be a JSON object containing all required feature names.

Example (PowerShell):

    $body = @{
        Area = 87524
        MajorAxisLength = 442.24
        MinorAxisLength = 253.29
        Eccentricity = 0.8197
        ConvexArea = 90546
        Extent = 0.7586
        Perimeter = 1184.04
    } | ConvertTo-Json

    Invoke-RestMethod `
        -Uri "http://localhost:8000/predict" `
        -Method POST `
        -ContentType "application/json" `
        -Body $body

Example response:

    {
        "pred": 1,
        "p_class1": 0.87,
        "logit": 1.92,
        "feature_contrib_logit": {
            "Area": 0.52,
            "Perimeter": 0.31
        },
        "feature_names": [...]
    }

------------------------------------------------------------------------

# Error Handling

The server returns structured error responses:

- 400 Bad Request for malformed payload
- 404 Not Found for unknown endpoints

Example error:

    {
        "error": "missing_features",
        "missing": ["Perimeter"]
    }

------------------------------------------------------------------------

# Server not starting (Windows / PowerShell)

If test_inference.py times out waiting for /healthz, first check if port 8000 is in use:

Run the following to check what's already running on port 8000

    netstat -ano | findstr :8000

For each PID shown as LISTENING, identify it:
    
    tasklist /FI "PID eq <PID>"

Kill it (if safe to do so):
    
    taskkill /PID <PID> /F

Then rerun:

    pytest


------------------------------------------------------------------------


# Containerisation

Build:

    docker build -t raisin-infer .

Run:

    docker run -p 8000:8000 raisin-infer

------------------------------------------------------------------------

# Engineering Considerations

Reproducibility:

- Deterministic label mapping
- Fixed random seed
- Explicit feature ordering

Numerical Stability:

- Clipped logits
- Probability clipping in loss
- Zero-variance feature handling

Production Readiness:

- Stateless serving
- No pickle
- Minimal dependencies
- Clean separation of concerns
- Deterministic inference path

------------------------------------------------------------------------

# Summary

This repository implements a complete ML system:

- Model training
- Cross-validation
- Model persistence
- Inference server
- Containerisation

Built from first principles using only foundational Python tools.
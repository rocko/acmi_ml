# ACMI ML

Machine-learning pipeline for classifying aircraft maneuver states from flight telemetry.

The project covers the complete workflow from raw telemetry and feature engineering to model training, evaluation, ONNX export, and inference with ONNX Runtime.

This project is part of a larger private telemetry-viewer project. 
The ML pipeline was extracted into a separate repository so that telemetry analysis, model training, evaluation, and ONNX inference can be developed independently from the surrounding recording and visualization application.

The current dataset is generated from flight recordings created in DCS World and stored in ACMI-compatible telemetry files.

## Overview

Telemetry recordings contain time-series information about aircraft position and attitude. `acmi_ml` transforms this telemetry into fixed-duration flight windows and uses derived kinematic features to classify basic maneuver states.

The current pipeline is:

```text
recording
    ↓
telemetry parsing
    ↓
kinematic feature extraction
    ↓
time-based windowing
    ↓
automatic bootstrap labeling
    ↓
PyTorch training
    ↓
recording-based evaluation
    ↓
ONNX export
    ↓
ONNX Runtime inference
```

## Data Source

The current dataset consists of flight telemetry recorded in DCS World.

The recordings contain different types of flight behavior, including:

steady flight
acceleration and deceleration
climbs and descents
sustained turns
combined climbing and descending turns
more irregular maneuvering
approach and landing phases

The recording format is used as the interface between the larger telemetry-viewer project and this standalone ML pipeline.


## Feature Engineering

Aircraft motion is derived from consecutive telemetry samples instead of relying on aircraft-specific airspeed data.

Derived values include:

* ground speed
* three-dimensional speed
* vertical speed
* track heading
* track rate
* acceleration
* roll and pitch information

Samples are aggregated into time-based windows. The current baseline uses 5-second windows with a 1-second stride.

Window-level features include, among others:

* altitude change
* ground-speed change
* mean ground and 3D speed
* mean vertical speed
* peak climb and sink rate
* total track change
* mean and peak track rate
* roll statistics
* pitch statistics
* acceleration statistics

## Bootstrap Labeling

The initial dataset is automatically annotated using transparent heuristic rules.

Current maneuver classes include:

```text
steady
acceleration
deceleration
climb
descent
turn
climbing_turn
descending_turn
```

For example, a window is considered a turn when its accumulated track change or mean track rate exceeds a configured threshold.
Vertical movement and speed changes are classified similarly.

These labels are explicitly treated as **bootstrap labels rather than ground truth**. 
Their purpose is to create an initial reproducible dataset and validate the ML pipeline before introducing manually reviewed or richer domain-specific annotations.

## Training

The current baseline model is a small multilayer perceptron implemented in PyTorch.

```text
16 input features
    ↓
Linear 16 → 32
ReLU
    ↓
Linear 32 → 16
ReLU
    ↓
Linear 16 → maneuver classes
```

Feature normalization is calculated from the training data only.

Weighted cross-entropy is used to reduce the influence of class imbalance.

## Preventing Data Leakage

Consecutive flight windows overlap heavily. 
Randomly splitting individual windows would therefore place nearly identical telemetry in both training and evaluation sets.

Instead, complete flight recordings are assigned to separate partitions:

```text
training recordings
validation recording
test recording
```

No windows from a held-out recording are used during training.

## Initial Results

The first baseline experiment used five independent flight recordings.

Rare classes with insufficient samples were excluded from the initial training run.

Retained classes:

```text
acceleration
climb
climbing_turn
descending_turn
descent
steady
```

Results:

```text
Training (3 recordings)
accuracy:   0.933
macro F1:   0.940

Validation (1 recording)
accuracy:   0.920

Held-out test recording (1 recording)
accuracy:   0.812
macro F1:   0.830
```

Class               Precision   Recall   F1
------------------------------------------------
acceleration          1.000      1.000   1.000
climb                 1.000      0.875   0.933
climbing_turn         0.800      0.667   0.727
descending_turn       0.571      0.800   0.667
descent               0.875      0.778   0.824

Most errors on the held-out flight occurred between closely related maneuver states, particularly:

```text
descent ↔ descending_turn
climb   ↔ climbing_turn
```

The current results demonstrate that the model can approximate the bootstrap maneuver definitions on previously unseen recordings.

They should not be interpreted as production-level flight-behavior classification accuracy.

Results vary considerably depending on which complete recordings are held out due to the currently small dataset. Recording-level cross-validation should be considered.

## ONNX

After training, the PyTorch model is exported to ONNX.

```text
PyTorch checkpoint
        ↓
ONNX export
        ↓
ONNX Runtime
```

The normalization parameters are part of the exported model so that ONNX Runtime can consume the same raw window features used by the PyTorch implementation.



# Quickstart

```bash
python -m venv .venv
Set-ExecutionPolicy Bypass -Scope Process
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Generate dataset

... with bootstrap labeling

```bash
.\build_dataset.bat
```

## MS1 - Pipeline Validation

Before training a real model, a deterministic ONNX graph was generated
manually using fixed weights.

Its purpose was to validate:

ACMI-compatible recording -> feature extraction -> tensor creation -> ONNX Runtime -> result

The model was not trained and its predictions are not intended to represent
meaningful flight analysis.

```bash
python classify_recording.py `
    "raw\sortie_01.acmi" `
    --model models\flight_analyzer_demo.onnx `
    --aircraft-id 1
```


## MS2 - Trained Baseline

dataset -> heuristic labels -> PyTorch -> held-out evaluation -> ONNX

```bash
python train_flight_classifier.py `
    datasets\all_sorties.csv `
    --validation-source sortie_01.acmi `
    --test-source sortie_05.acmi `
    --output-dir models\flight_classifier_retrained
```

```bash
 python classify_recording.py raw/sortie_03.acmi `
    --model models\flight_classifier_v1\flight_classifier.onnx `
    --aircraft-id 1 `
    --show-bootstrap-label
```
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort

from acmi_features import iter_acmi_file
from flight_windows import FlightWindow, iter_flight_windows


FEATURE_NAMES: Tuple[str, ...] = (
    "altitude_delta_m",
    "ground_speed_delta_mps",
    "ground_speed_mean_mps",
    "total_speed_mean_mps",
    "vertical_speed_mean_mps",
    "sink_rate_peak_mps",
    "climb_rate_peak_mps",
    "track_change_deg",
    "track_rate_mean_abs_deg_s",
    "track_rate_peak_abs_deg_s",
    "roll_mean_abs_deg",
    "roll_peak_abs_deg",
    "pitch_mean_abs_deg",
    "pitch_peak_abs_deg",
    "acceleration_mean_mps2",
    "acceleration_peak_abs_mps2",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ACMI ML Classifier",
        description=(
            "Classify overlapping flight windows from an ACMI recording "
            "with the trained ONNX classifier."
        ),
    )

    parser.add_argument("acmi_file", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--aircraft-id", default=None)

    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)

    parser.add_argument("--window-seconds", type=float, default=5.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--min-samples", type=int, default=5)

    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--show-bootstrap-label",
        action="store_true",
        help="Also print the heuristic label for comparison.",
    )

    return parser


def _load_model_metadata(session: ort.InferenceSession) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    metadata = session.get_modelmeta().custom_metadata_map

    try:
        feature_names = tuple(json.loads(metadata["feature_names"]))
        class_labels = tuple(json.loads(metadata["class_labels"]))
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("ONNX model is missing acmi_ml feature/class metadata") from exc

    if feature_names != FEATURE_NAMES:
        raise ValueError(f"ONNX feature contract does not match classify_acmi.py. Model={list(feature_names)}, expected={list(FEATURE_NAMES)}")

    if not class_labels:
        raise ValueError("ONNX model contains no class labels")

    return feature_names, class_labels


def _window_features(window: FlightWindow) -> np.ndarray:
    values = []

    for name in FEATURE_NAMES:
        value = getattr(window, name)

        if value is None:
            raise ValueError(f"Window {window.start_time_s:.3f}-{window.end_time_s:.3f}s has no value for required feature '{name}'")

        values.append(float(value))

    return np.asarray([values], dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.acmi_file.is_file():
        raise FileNotFoundError(f"ACMI file does not exist: {args.acmi_file}")

    if not args.model.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {args.model}")

    if args.window_seconds <= 0.0:
        raise ValueError("--window-seconds must be greater than zero")

    if args.stride_seconds <= 0.0:
        raise ValueError("--stride-seconds must be greater than zero")

    if args.min_samples < 2:
        raise ValueError("--min-samples must be at least two")

    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")

    if args.start_time is not None and args.start_time < 0.0:
        raise ValueError("--start-time must be zero or greater")

    if args.end_time is not None and args.end_time < 0.0:
        raise ValueError("--end-time must be zero or greater")

    if (args.start_time is not None and args.end_time is not None and args.end_time < args.start_time):
        raise ValueError("--end-time must be greater than or equal to --start-time")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])

    _, class_labels = _load_model_metadata(session)

    model_inputs = session.get_inputs()
    model_outputs = session.get_outputs()

    if len(model_inputs) != 1 or len(model_outputs) != 1:
        raise ValueError("Expected ONNX model with one input and one output")

    input_name = model_inputs[0].name
    output_name = model_outputs[0].name

    samples = iter_acmi_file(args.acmi_file, aircraft_id=args.aircraft_id, start_time=args.start_time, end_time=args.end_time    )
    windows = iter_flight_windows(samples, window_seconds=args.window_seconds, stride_seconds=args.stride_seconds, min_samples=args.min_samples)

    window_count = 0
    emitted_count = 0

    for window in windows:
        window_count += 1

        features = _window_features(window)

        logits = session.run(
            [output_name],
            {input_name: features},
        )[0]

        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(f"Unexpected ONNX output shape: {logits.shape}")

        scores = logits[0]

        if len(scores) != len(class_labels):
            raise ValueError("Class count mismatch between ONNX output and metadata: {len(scores)} != {len(class_labels)}")

        probabilities = _softmax(scores)

        predicted_index = int(np.argmax(probabilities))
        label = class_labels[predicted_index]
        confidence = float(probabilities[predicted_index])

        if confidence < args.min_confidence:
            continue

        bootstrap = (f" bootstrap={window.label:18s}" if args.show_bootstrap_label else "")

        ground_speed_delta = (window.ground_speed_delta_mps if window.ground_speed_delta_mps is not None else 0.0)

        print(
            f"{window.start_time_s:9.3f}-"
            f"{window.end_time_s:9.3f}s  "
            f"{label:18s} "
            f"conf={confidence:5.3f}  "
            f"dalt={window.altitude_delta_m:8.1f}m  "
            f"dtrk={window.track_change_deg:7.1f}deg  "
            f"vs={window.vertical_speed_mean_mps:7.2f}m/s  "
            f"dgs={ground_speed_delta:7.2f}m/s"
            f"{bootstrap}",
            flush=True,
        )

        emitted_count += 1

    if window_count == 0:
        raise RuntimeError("No complete flight windows were produced. Check --aircraft-id, time range, and window settings.")

    print(f"\nProcessed {window_count} windows; emitted {emitted_count} predictions.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
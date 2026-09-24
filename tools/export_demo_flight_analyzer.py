from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


FEATURE_NAMES = [
    "altitude_km",
    "ground_speed_100_mps",
    "total_speed_100_mps",
    "roll_abs_90",
    "pitch_abs_45",
    "sink_rate_50",
    "track_rate_abs_30",
    "speed_change_abs_10",
]

CLASS_LABELS = [
    "normal",
    "high_sink",
    "hard_turn",
    "rapid_speed_change",
]


def build_model() -> onnx.ModelProto:
    """Build a tiny deterministic ONNX model for pipeline validation.

    This is deliberately not a trained flight-quality model. The weights only
    provide predictable outputs so the complete ACMI -> feature -> ONNX path can
    be exercised before a real dataset and training loop exist.
    """

    # Rows correspond to input features, columns to output classes.
    # Ground speed and total speed are intentionally neutral in MS1.1. They are
    # part of the stable arbitrary-aircraft feature contract and can be learned
    # properly once MS2 introduces a real training pipeline.
    weights = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-0.4, 0.0, 0.5, 0.0],
            [-0.3, 0.0, 0.0, 0.0],
            [-2.0, 3.0, 0.0, 0.0],
            [-2.0, 0.0, 3.0, 0.0],
            [-1.5, 0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    bias = np.asarray([1.5, -0.5, -0.5, -0.5], dtype=np.float32)

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Gemm",
                inputs=["features", "weights", "bias"],
                outputs=["logits"],
                alpha=1.0,
                beta=1.0,
                transB=0,
            ),
            helper.make_node(
                "Softmax",
                inputs=["logits"],
                outputs=["probabilities"],
                axis=1,
            ),
        ],
        name="ACMI ML Analyzer",
        inputs=[
            helper.make_tensor_value_info(
                "features",
                TensorProto.FLOAT,
                [None, len(FEATURE_NAMES)],
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "probabilities",
                TensorProto.FLOAT,
                [None, len(CLASS_LABELS)],
            )
        ],
        initializer=[
            numpy_helper.from_array(weights, name="weights"),
            numpy_helper.from_array(bias, name="bias"),
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="ACMI ML",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )

    # Keep the model compatible with broadly deployed ONNX Runtime versions.
    model.ir_version = 8

    feature_metadata = model.metadata_props.add()
    feature_metadata.key = "feature_names"
    feature_metadata.value = json.dumps(FEATURE_NAMES)

    label_metadata = model.metadata_props.add()
    label_metadata.key = "class_labels"
    label_metadata.value = json.dumps(CLASS_LABELS)

    note_metadata = model.metadata_props.add()
    note_metadata.key = "purpose"
    note_metadata.value = (
        "MS1.1 pipeline validation only - not a trained flight model"
    )

    onnx.checker.check_model(model)
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the MS1.1 deterministic demo flight analyzer to ONNX."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/flight_analyzer_demo.onnx"),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = build_model()
    onnx.save(model, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

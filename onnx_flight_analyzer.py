from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from acmi_features import FEATURE_NAMES, FlightSample


DEFAULT_LABELS: Tuple[str, ...] = (
    "normal",
    "high_sink",
    "hard_turn",
    "rapid_speed_change",
)


@dataclass(frozen=True)
class AnalysisResult:
    time_s: float
    label: str
    confidence: float
    probabilities: Tuple[float, ...]


class OnnxFlightAnalyzer:
    """Run one fixed-size flight feature vector through an ONNX model."""

    def __init__(self, model_path: Path, *, labels: Optional[Sequence[str]] = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for flight analysis. Install requirements.txt.") from exc

        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input, found {len(inputs)}")

        outputs = self._session.get_outputs()
        if len(outputs) < 1:
            raise ValueError("ONNX model exposes no outputs")

        self._input_name = inputs[0].name
        self._output_name = outputs[0].name

        metadata: Dict[str, str] = dict(self._session.get_modelmeta().custom_metadata_map)

        model_feature_names = self._read_json_list(metadata.get("feature_names"))
        if model_feature_names and tuple(model_feature_names) != FEATURE_NAMES:
            raise ValueError(f"ONNX feature contract does not match extractor. Model={model_feature_names}, extractor={list(FEATURE_NAMES)}")

        model_labels = self._read_json_list(metadata.get("class_labels"))
        selected_labels = labels or model_labels or DEFAULT_LABELS
        self._labels = tuple(str(label) for label in selected_labels)

    @staticmethod
    def _read_json_list(value: Optional[str]) -> Tuple[str, ...]:
        if not value:
            return ()

        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return ()

        if not isinstance(decoded, list):
            return ()

        return tuple(str(item) for item in decoded)

    @staticmethod
    def _probabilities(values: np.ndarray) -> np.ndarray:
        """Accept probabilities directly or normalize logits with softmax."""

        values = np.asarray(values, dtype=np.float32).reshape(-1)
        total = float(values.sum())

        if (np.all(values >= 0.0) and np.all(values <= 1.0) and abs(total - 1.0) <= 1e-3):
            return values

        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        denominator = float(exp_values.sum())
        if denominator <= 0.0:
            raise ValueError("ONNX output cannot be normalized")

        return exp_values / denominator

    def analyze(self, sample: FlightSample) -> AnalysisResult:
        features = np.asarray([sample.features], dtype=np.float32)

        if features.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"Expected {len(FEATURE_NAMES)} features, received {features.shape[1]}")

        output = self._session.run([self._output_name], {self._input_name: features})[0]

        probabilities = self._probabilities(np.asarray(output)[0])
        if len(probabilities) != len(self._labels):
            raise ValueError(f"Class count mismatch between ONNX output and labels: {len(probabilities)} != {len(self._labels)}")

        best_index = int(np.argmax(probabilities))
        return AnalysisResult(
            time_s=sample.time_s,
            label=self._labels[best_index],
            confidence=float(probabilities[best_index]),
            probabilities=tuple(float(value) for value in probabilities)
        )

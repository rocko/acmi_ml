from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


@dataclass(frozen=True)
class DatasetRow:
    source_key: str
    label: str
    features: Tuple[float, ...]


class FlightWindowClassifier(nn.Module):
    def __init__(self, input_size: int, class_count: int, feature_mean: torch.Tensor, feature_std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)
        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, class_count),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std
        return self.network(normalized)


def _source_key(source_file: str) -> str:
    windows_name = PureWindowsPath(source_file).name
    return windows_name or Path(source_file).name or source_file


def _load_rows(path: Path) -> List[DatasetRow]:
    rows: List[DatasetRow] = []

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in ("source_file", "label", *FEATURE_NAMES) if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

        for line_number, raw in enumerate(reader, start=2):
            try:
                features = tuple(float(raw[name]) for name in FEATURE_NAMES)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric feature at CSV line {line_number}") from exc

            rows.append(DatasetRow(source_key=_source_key(raw["source_file"]), label=raw["label"], features=features))

    if not rows:
        raise ValueError(f"Dataset contains no rows: {path}")

    return rows


def _match_source(requested: str, available: Sequence[str]) -> str:
    requested_key = _source_key(requested)
    matches = [source for source in available if source == requested or source == requested_key]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely match source '{requested}'. Available sources: {', '.join(available)}")
    return matches[0]


def _filter_rare_labels(rows: Sequence[DatasetRow], min_class_count: int) -> Tuple[List[DatasetRow], List[str]]:
    counts = Counter(row.label for row in rows)
    retained = sorted(label for label, count in counts.items() if count >= min_class_count)
    retained_set = set(retained)
    return [row for row in rows if row.label in retained_set], retained


def _split_rows(rows: Sequence[DatasetRow], validation_source: str, test_source: str) -> Tuple[List[DatasetRow], List[DatasetRow], List[DatasetRow]]:
    training = [row for row in rows if row.source_key not in {validation_source, test_source}]
    validation = [row for row in rows if row.source_key == validation_source]
    test = [row for row in rows if row.source_key == test_source]

    if not training or not validation or not test:
        raise ValueError("Train/validation/test split contains an empty partition")

    return training, validation, test


def _as_arrays(rows: Sequence[DatasetRow], class_to_index: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray([row.features for row in rows], dtype=np.float32)
    labels = np.asarray([class_to_index[row.label] for row in rows], dtype=np.int64)
    return features, labels


def _confusion_matrix(truth: np.ndarray, predicted: np.ndarray, class_count: int) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for expected, actual in zip(truth, predicted):
        matrix[int(expected), int(actual)] += 1
    return matrix


def _metrics(truth: np.ndarray, predicted: np.ndarray, class_labels: Sequence[str]) -> Dict[str, object]:
    matrix = _confusion_matrix(truth, predicted, len(class_labels))
    per_class: Dict[str, Dict[str, float | int]] = {}
    f1_values: List[float] = []

    for index, label in enumerate(class_labels):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        support = int(matrix[index, :].sum())

        precision = (true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0)
        recall = (true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0)
        f1 = (2.0 * precision * recall / (precision + recall) if precision + recall else 0.0)  #\beta = 2

        if support > 0:
            f1_values.append(f1)

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    accuracy = float((truth == predicted).mean()) if len(truth) else 0.0
    macro_f1 = float(sum(f1_values) / len(f1_values)) if f1_values else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1_present_classes": macro_f1,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def _evaluate(model: nn.Module, features: np.ndarray, labels: np.ndarray, device: torch.device) -> Tuple[float, np.ndarray]:
    model.eval()
    x = torch.from_numpy(features).to(device)
    y = torch.from_numpy(labels).to(device)

    with torch.no_grad():
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        predicted = logits.argmax(dim=1).cpu().numpy()

    return float(loss.item()), predicted


def _print_partition(name: str, rows: Sequence[DatasetRow]) -> None:
    counts = Counter(row.label for row in rows)
    sources = sorted({row.source_key for row in rows})

    print(f"\n{name}: {len(rows)} windows")
    for source in sources:
        print(f"  source: {source}")
    for label, count in sorted(counts.items()):
        print(f"  {label:20s} {count}")


def _print_metrics(name: str, metrics: Dict[str, object], class_labels: Sequence[str]) -> None:
    print(f"\n{name}: accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1_present_classes']:.3f}")

    per_class = metrics["per_class"]
    assert isinstance(per_class, dict)
    for label in class_labels:
        values = per_class[label]
        print(
            f"  {label:20s} "
            f"p={values['precision']:.3f} "
            f"r={values['recall']:.3f} "
            f"f1={values['f1']:.3f} "
            f"n={values['support']}"
        )

    print("  confusion matrix rows=true, columns=predicted")
    print("  labels:", " | ".join(class_labels))
    matrix = metrics["confusion_matrix"]
    assert isinstance(matrix, list)
    for label, row in zip(class_labels, matrix):
        print(f"  {label:20s} " + " ".join(f"{value:4d}" for value in row))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ACMI ML Classifier Trainer",
        description=("Train a small PyTorch MLP on MS2 flight-window CSV data. Entire source recordings are held out for validation and test."),
    )
    parser.add_argument(
        "dataset", 
        type=Path
    )
    parser.add_argument(
        "--validation-source", 
        required=True
        )
    parser.add_argument("--test-source", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/flight_classifier_v1"))
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {args.dataset}")
    if args.min_class_count < 2:
        raise ValueError("--min-class-count must be at least two")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be greater than zero")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = _load_rows(args.dataset)
    all_counts = Counter(row.label for row in rows)
    filtered_rows, class_labels = _filter_rare_labels(rows, min_class_count=args.min_class_count)
    if len(class_labels) < 2:
        raise ValueError("At least two retained classes are required")

    dropped = {
        label: count
        for label, count in sorted(all_counts.items())
        if label not in class_labels
    }
    print("Retained classes:", ", ".join(class_labels))
    if dropped:
        print("Dropped rare classes:",", ".join(f"{label}={count}" for label, count in dropped.items()))

    sources = sorted({row.source_key for row in filtered_rows})
    if len(sources) < 3:
        raise ValueError("At least three independent source recordings are required")

    validation_source = _match_source(args.validation_source, sources)
    test_source = _match_source(args.test_source, sources)
    if validation_source == test_source:
        raise ValueError("Validation and test sources must be different")

    training_rows, validation_rows, test_rows = _split_rows(filtered_rows, validation_source=validation_source, test_source=test_source)

    training_labels = {row.label for row in training_rows}
    missing_from_training = [label for label in class_labels if label not in training_labels]
    if missing_from_training:
        raise ValueError("Retained classes missing from training split: "+ ", ".join(missing_from_training))

    _print_partition("TRAIN", training_rows)
    _print_partition("VALIDATION", validation_rows)
    _print_partition("TEST", test_rows)

    class_to_index = {
        label: index for index, label in enumerate(class_labels)
    }

    train_x, train_y = _as_arrays(training_rows, class_to_index)
    val_x, val_y = _as_arrays(validation_rows, class_to_index)
    test_x, test_y = _as_arrays(test_rows, class_to_index)

    feature_mean_np = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std_np = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std_np[feature_std_np < 1e-6] = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = FlightWindowClassifier(
        input_size=len(FEATURE_NAMES),
        class_count=len(class_labels),
        feature_mean=torch.from_numpy(feature_mean_np),
        feature_std=torch.from_numpy(feature_std_np),
    ).to(device)

    class_counts = np.bincount(train_y, minlength=len(class_labels)).astype(np.float32)
    class_weights_np = len(train_y) / ( len(class_labels) * np.maximum(class_counts, 1.0))
    class_weights = torch.from_numpy(class_weights_np).to(device)

    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train_dataset)), shuffle=True, generator=generator)

    best_validation_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_function(logits, batch_y)
            loss.backward()
            optimizer.step()

        validation_loss, _ = _evaluate(model, val_x, val_y, device)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            train_loss, _ = _evaluate(model, train_x, train_y, device)
            print(f"epoch={epoch:4d} train_loss={train_loss:.4f} val_loss={validation_loss:.4f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a model state")

    model.load_state_dict(best_state)
    model.to(device)

    _, train_pred = _evaluate(model, train_x, train_y, device)
    _, val_pred = _evaluate(model, val_x, val_y, device)
    _, test_pred = _evaluate(model, test_x, test_y, device)

    train_metrics = _metrics(train_y, train_pred, class_labels)
    validation_metrics = _metrics(val_y, val_pred, class_labels)
    test_metrics = _metrics(test_y, test_pred, class_labels)

    print(f"\nBest epoch: {best_epoch}")
    _print_metrics("TRAIN", train_metrics, class_labels)
    _print_metrics("VALIDATION", validation_metrics, class_labels)
    _print_metrics("TEST", test_metrics, class_labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "flight_classifier.pt"
    onnx_path = args.output_dir / "flight_classifier.onnx"
    metadata_path = args.output_dir / "training_metadata.json"

    checkpoint = {
        "state_dict": best_state,
        "feature_names": list(FEATURE_NAMES),
        "class_labels": list(class_labels),
        "feature_mean": feature_mean_np.tolist(),
        "feature_std": feature_std_np.tolist(),
        "validation_source": validation_source,
        "test_source": test_source,
        "training_sources": sorted(
            {row.source_key for row in training_rows}
        ),
        "seed": args.seed,
        "min_class_count": args.min_class_count,
    }
    torch.save(checkpoint, checkpoint_path)

    model_cpu = model.cpu().eval()
    dummy_input = torch.zeros((1, len(FEATURE_NAMES)), dtype=torch.float32)
    torch.onnx.export(
        model_cpu,
        dummy_input,
        onnx_path,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={
            "features": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    onnx_model = onnx.load(onnx_path)
    metadata = {
        "feature_names": list(FEATURE_NAMES),
        "class_labels": list(class_labels),
        "label_source": "heuristic_ms2_v1",
        "normalization": "embedded_train_split_mean_std",
        "purpose": "MS2 trained flight-window baseline",
    }

    for key, value in metadata.items():
        prop = onnx_model.metadata_props.add()
        prop.key = key
        prop.value = json.dumps(value) if isinstance(value, list) else value

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, onnx_path)

    training_metadata = {
        "dataset": str(args.dataset),
        "feature_names": list(FEATURE_NAMES),
        "class_labels": list(class_labels),
        "all_class_counts": dict(sorted(all_counts.items())),
        "dropped_rare_classes": dropped,
        "training_sources": checkpoint["training_sources"],
        "validation_source": validation_source,
        "test_source": test_source,
        "best_epoch": best_epoch,
        "epochs_requested": args.epochs,
        "seed": args.seed,
        "class_weights": {
            label: float(class_weights_np[index])
            for index, label in enumerate(class_labels)
        },
        "feature_mean": feature_mean_np.tolist(),
        "feature_std": feature_std_np.tolist(),
        "metrics": {
            "train": train_metrics,
            "validation": validation_metrics,
            "test": test_metrics,
        },
    }
    metadata_path.write_text(json.dumps(training_metadata, indent=2), encoding="utf-8")

    print(f"\nSaved PyTorch checkpoint: {checkpoint_path}")
    print(f"Saved ONNX model:          {onnx_path}")
    print(f"Saved training metadata:   {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

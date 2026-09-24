from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence


from acmi_features import iter_acmi_file
from flight_windows import DATASET_FIELDS, iter_flight_windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ACMI ML Flight Dataset Builder",
        description=(
            "Convert one aircraft from a loose or ZIP-compressed ACMI recording "
            "into overlapping, training-ready flight windows."
        ),
    )
    parser.add_argument("acmi_file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/flight_windows.csv"),
        help="CSV output path. Default: datasets/flight_windows.csv",
    )
    parser.add_argument(
        "--aircraft-id",
        "--player-id",
        dest="aircraft_id",
        default=None,
        help=(
            "ACMI object ID to extract. If omitted, DNTV.IsPlayer=true is "
            "auto-detected. --player-id is a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Ignore samples before this ACMI time in seconds.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Stop after this ACMI time in seconds.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=5.0,
        help="Window duration in seconds. Default: 5.0",
    )
    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=1.0,
        help="Distance between consecutive window starts. Default: 1.0",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=5,
        help="Minimum samples required inside a window. Default: 5",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not args.acmi_file.is_file():
        raise FileNotFoundError(f"ACMI file does not exist: {args.acmi_file}")
    if args.window_seconds <= 0.0:
        raise ValueError("--window-seconds must be greater than zero")
    if args.stride_seconds <= 0.0:
        raise ValueError("--stride-seconds must be greater than zero")
    if args.min_samples < 2:
        raise ValueError("--min-samples must be at least two")
    if args.start_time is not None and args.start_time < 0.0:
        raise ValueError("--start-time must be zero or greater")
    if args.end_time is not None and args.end_time < 0.0:
        raise ValueError("--end-time must be zero or greater")
    if (args.start_time is not None and args.end_time is not None and args.end_time < args.start_time):
        raise ValueError("--end-time must be greater than or equal to --start-time")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)

    samples = iter_acmi_file(args.acmi_file, aircraft_id=args.aircraft_id, start_time=args.start_time, end_time=args.end_time)
    windows = iter_flight_windows(samples, window_seconds=args.window_seconds, stride_seconds=args.stride_seconds, min_samples=args.min_samples)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    label_counts: Counter[str] = Counter()
    window_count = 0
    fieldnames = list(DATASET_FIELDS)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for window_count, window in enumerate(windows, start=1):
            row = asdict(window)
            row["source_file"] = str(args.acmi_file)
            row["window_index"] = window_count
            writer.writerow({field: row.get(field) for field in fieldnames})
            label_counts[window.label] += 1

    if window_count == 0:
        args.output.unlink(missing_ok=True)
        raise RuntimeError("No complete flight windows were produced. Check --aircraft-id and the selected time range, or reduce --window-seconds/--min-samples.")

    print(f"Wrote {window_count} windows to {args.output}")
    print("Provisional heuristic labels:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label:20s} {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

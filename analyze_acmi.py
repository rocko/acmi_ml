from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

try:
    # Package imports for tests and `python -m app.analyze_flight`.
    from .acmi_features import FlightSample, iter_acmi_file
    from .onnx_flight_analyzer import AnalysisResult, OnnxFlightAnalyzer
except ImportError:
    # Direct-script compatibility for `python app/analyze_flight.py`.
    from acmi_features import FlightSample, iter_acmi_file
    from onnx_flight_analyzer import AnalysisResult, OnnxFlightAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DCS-NotTacView Flight Analyzer",
        description=(
            "Stream one aircraft from a loose or ZIP-compressed ACMI recording "
            "and run its derived telemetry features through an ONNX model."
        ),
    )
    parser.add_argument("acmi_file", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--aircraft-id",
        "--player-id",
        dest="aircraft_id",
        default=None,
        help=(
            "ACMI object ID to analyze. If omitted, DNTV.IsPlayer=true is "
            "auto-detected for ownship recordings. --player-id is retained as "
            "a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Only emit samples at or after this ACMI time in seconds.",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Stop after this ACMI time in seconds.",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help=(
            "Keep waiting for lines appended to a loose text ACMI file. "
            "ZIP-compressed ACMI cannot be followed."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.10,
        help="File polling interval in seconds when --follow is enabled.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=1.0,
        help=(
            "Minimum ACMI-time interval between status lines. Label changes "
            "are printed immediately."
        ),
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Suppress printed results below this confidence.",
    )
    return parser


def _format_optional_speed(value: Optional[float]) -> str:
    if value is None:
        return "    n/a"
    return f"{value:7.1f}m/s"


def _format_optional_heading(value: Optional[float]) -> str:
    if value is None:
        return "   n/a"
    return f"{value:6.1f}deg"


def _format_result(sample: FlightSample, result: AnalysisResult) -> str:
    return (
        f"{sample.time_s:9.3f}s  "
        f"{result.label:18s} "
        f"conf={result.confidence:5.3f}  "
        f"alt={sample.altitude_m:8.1f}m  "
        f"gs={_format_optional_speed(sample.ground_speed_mps)}  "
        f"3d={_format_optional_speed(sample.total_speed_mps)}  "
        f"vs={sample.vertical_speed_mps:7.2f}m/s  "
        f"trk={_format_optional_heading(sample.track_heading_deg)}  "
        f"turn={sample.track_rate_deg_s:7.2f}deg/s  "
        f"accel={sample.acceleration_mps2:7.2f}m/s2"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.acmi_file.is_file():
        raise FileNotFoundError(f"ACMI file does not exist: {args.acmi_file}")

    if not args.model.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {args.model}")

    if args.print_interval < 0.0:
        raise ValueError("--print-interval must be zero or greater")

    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1")

    if args.start_time is not None and args.start_time < 0.0:
        raise ValueError("--start-time must be zero or greater")

    if args.end_time is not None and args.end_time < 0.0:
        raise ValueError("--end-time must be zero or greater")

    if (
        args.start_time is not None
        and args.end_time is not None
        and args.end_time < args.start_time
    ):
        raise ValueError("--end-time must be greater than or equal to --start-time")

    analyzer = OnnxFlightAnalyzer(args.model)
    last_print_time: Optional[float] = None
    last_label: Optional[str] = None
    sample_count = 0

    for sample in iter_acmi_file(
        args.acmi_file,
        aircraft_id=args.aircraft_id,
        follow=args.follow,
        poll_interval=args.poll_interval,
        start_time=args.start_time,
        end_time=args.end_time,
    ):
        sample_count += 1

        # Inference runs for every selected sample. Only console output is throttled.
        result = analyzer.analyze(sample)

        label_changed = result.label != last_label
        interval_elapsed = (
            last_print_time is None
            or sample.time_s - last_print_time >= args.print_interval
        )

        if (
            result.confidence >= args.min_confidence
            and (label_changed or interval_elapsed)
        ):
            print(_format_result(sample, result), flush=True)
            last_print_time = sample.time_s

        last_label = result.label

    if sample_count == 0:
        raise RuntimeError(
            "No aircraft samples were found. For AI aircraft, pass its ACMI "
            "object ID with --aircraft-id. Ownship auto-detection requires "
            "DNTV.IsPlayer=true."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

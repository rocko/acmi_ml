from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, Optional, Sequence, Tuple


from acmi_features import FlightSample


LABEL_VERSION = "heuristic_ms2_v1"

# These are intentionally simple, inspectable bootstrap rules. They are not
# ground truth and should eventually be replaced by reviewed labels.
TURN_TRACK_CHANGE_DEG = 20.0
TURN_MEAN_RATE_DEG_S = 4.0
VERTICAL_DELTA_M = 40.0
SPEED_DELTA_MPS = 12.0


@dataclass(frozen=True)
class FlightWindow:
    aircraft_id: str
    aircraft_name: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    sample_count: int

    altitude_start_m: float
    altitude_end_m: float
    altitude_delta_m: float

    ground_speed_start_mps: Optional[float]
    ground_speed_end_mps: Optional[float]
    ground_speed_delta_mps: Optional[float]
    ground_speed_mean_mps: Optional[float]
    total_speed_mean_mps: Optional[float]

    vertical_speed_mean_mps: float
    sink_rate_peak_mps: float
    climb_rate_peak_mps: float

    track_change_deg: float
    track_rate_mean_abs_deg_s: float
    track_rate_peak_abs_deg_s: float

    roll_mean_abs_deg: float
    roll_peak_abs_deg: float
    pitch_mean_abs_deg: float
    pitch_peak_abs_deg: float

    acceleration_mean_mps2: float
    acceleration_peak_abs_mps2: float

    label: str
    label_version: str = LABEL_VERSION


DATASET_FIELDS: Tuple[str, ...] = (
    "source_file",
    "window_index",
    "aircraft_id",
    "aircraft_name",
    "start_time_s",
    "end_time_s",
    "duration_s",
    "sample_count",
    "altitude_start_m",
    "altitude_end_m",
    "altitude_delta_m",
    "ground_speed_start_mps",
    "ground_speed_end_mps",
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
    "label",
    "label_version",
)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _first_present(values: Sequence[Optional[float]]) -> Optional[float]:
    for value in values:
        if value is not None:
            return value
    return None


def _last_present(values: Sequence[Optional[float]]) -> Optional[float]:
    for value in reversed(values):
        if value is not None:
            return value
    return None


def _angle_delta_deg(current: float, previous: float) -> float:
    return (current - previous + 180.0) % 360.0 - 180.0


def _track_change_deg(samples: Sequence[FlightSample]) -> float:
    headings = [
        sample.track_heading_deg
        for sample in samples
        if sample.track_heading_deg is not None
    ]
    if len(headings) < 2:
        return 0.0

    return sum(_angle_delta_deg(current, previous) for previous, current in zip(headings, headings[1:]))


def _provisional_label(*, altitude_delta_m: float, ground_speed_delta_mps: Optional[float], track_change_deg: float, track_rate_mean_abs_deg_s: float) -> str:
    turning = (abs(track_change_deg) >= TURN_TRACK_CHANGE_DEG or track_rate_mean_abs_deg_s >= TURN_MEAN_RATE_DEG_S)
    descending = altitude_delta_m <= -VERTICAL_DELTA_M
    climbing = altitude_delta_m >= VERTICAL_DELTA_M

    if turning and descending:
        return "descending_turn"
    if turning and climbing:
        return "climbing_turn"
    if turning:
        return "turn"
    if descending:
        return "descent"
    if climbing:
        return "climb"

    if ground_speed_delta_mps is not None:
        if ground_speed_delta_mps <= -SPEED_DELTA_MPS:
            return "deceleration"
        if ground_speed_delta_mps >= SPEED_DELTA_MPS:
            return "acceleration"

    return "steady"


def summarize_window(samples: Sequence[FlightSample]) -> FlightWindow:
    if len(samples) < 2:
        raise ValueError("A flight window requires at least two samples")

    first = samples[0]
    last = samples[-1]
    duration_s = last.time_s - first.time_s
    if duration_s <= 0.0:
        raise ValueError("Flight window duration must be greater than zero")

    ground_speeds = [sample.ground_speed_mps for sample in samples]
    total_speeds = [sample.total_speed_mps for sample in samples]
    vertical_speeds = [sample.vertical_speed_mps for sample in samples]
    track_rates_abs = [abs(sample.track_rate_deg_s) for sample in samples]
    roll_abs = [abs(sample.roll_deg) for sample in samples]
    pitch_abs = [abs(sample.pitch_deg) for sample in samples]
    accelerations = [sample.acceleration_mps2 for sample in samples]

    ground_speed_start = _first_present(ground_speeds)
    ground_speed_end = _last_present(ground_speeds)
    ground_speed_delta = (ground_speed_end - ground_speed_start if ground_speed_start is not None and ground_speed_end is not None else None)
    altitude_delta = last.altitude_m - first.altitude_m
    track_change = _track_change_deg(samples)
    track_rate_mean_abs = _mean(track_rates_abs)

    label = _provisional_label(altitude_delta_m=altitude_delta, ground_speed_delta_mps=ground_speed_delta, track_change_deg=track_change, track_rate_mean_abs_deg_s=track_rate_mean_abs)

    return FlightWindow(
        aircraft_id=last.object_id,
        aircraft_name=last.aircraft_name or first.aircraft_name,
        start_time_s=first.time_s,
        end_time_s=last.time_s,
        duration_s=duration_s,
        sample_count=len(samples),
        altitude_start_m=first.altitude_m,
        altitude_end_m=last.altitude_m,
        altitude_delta_m=altitude_delta,
        ground_speed_start_mps=ground_speed_start,
        ground_speed_end_mps=ground_speed_end,
        ground_speed_delta_mps=ground_speed_delta,
        ground_speed_mean_mps=_mean_optional(ground_speeds),
        total_speed_mean_mps=_mean_optional(total_speeds),
        vertical_speed_mean_mps=_mean(vertical_speeds),
        sink_rate_peak_mps=max(max(-value, 0.0) for value in vertical_speeds),
        climb_rate_peak_mps=max(max(value, 0.0) for value in vertical_speeds),
        track_change_deg=track_change,
        track_rate_mean_abs_deg_s=track_rate_mean_abs,
        track_rate_peak_abs_deg_s=max(track_rates_abs, default=0.0),
        roll_mean_abs_deg=_mean(roll_abs),
        roll_peak_abs_deg=max(roll_abs, default=0.0),
        pitch_mean_abs_deg=_mean(pitch_abs),
        pitch_peak_abs_deg=max(pitch_abs, default=0.0),
        acceleration_mean_mps2=_mean(accelerations),
        acceleration_peak_abs_mps2=max((abs(value) for value in accelerations), default=0.0),
        label=label,
    )


def iter_flight_windows(samples: Iterable[FlightSample], *, window_seconds: float = 5.0, stride_seconds: float = 1.0, min_samples: int = 5, min_coverage_ratio: float = 0.8) -> Iterator[FlightWindow]:
    if window_seconds <= 0.0:
        raise ValueError("window_seconds must be greater than zero")
    if stride_seconds <= 0.0:
        raise ValueError("stride_seconds must be greater than zero")
    if min_samples < 2:
        raise ValueError("min_samples must be at least two")
    if not 0.0 < min_coverage_ratio <= 1.0:
        raise ValueError("min_coverage_ratio must be between zero and one")

    buffer: Deque[FlightSample] = deque()
    next_window_start: Optional[float] = None

    for sample in samples:
        if next_window_start is None:
            next_window_start = sample.time_s

        buffer.append(sample)

        while (next_window_start is not None and sample.time_s >= next_window_start + window_seconds):
            window_end = next_window_start + window_seconds
            selected = [candidate for candidate in buffer if next_window_start <= candidate.time_s <= window_end]

            if len(selected) >= min_samples:
                coverage = selected[-1].time_s - selected[0].time_s
                if coverage >= window_seconds * min_coverage_ratio:
                    yield summarize_window(selected)

            next_window_start += stride_seconds

            while buffer and buffer[0].time_s < next_window_start:
                buffer.popleft()

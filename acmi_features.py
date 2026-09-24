from __future__ import annotations

from dataclasses import dataclass
from io import TextIOWrapper
from math import atan2, cos, degrees, hypot, radians
from pathlib import Path
from time import sleep
from typing import Dict, Iterable, Iterator, List, Optional, TextIO, Tuple
from zipfile import ZipFile, is_zipfile


EARTH_RADIUS_M = 6_371_000.0

FEATURE_NAMES: Tuple[str, ...] = (
    "altitude_km",
    "ground_speed_100_mps",
    "total_speed_100_mps",
    "roll_abs_90",
    "pitch_abs_45",
    "sink_rate_50",
    "track_rate_abs_30",
    "speed_change_abs_10",
)


@dataclass(frozen=True)
class FlightSample:
    """One aircraft sample prepared for model inference."""

    time_s: float
    object_id: str
    aircraft_name: str
    longitude: float
    latitude: float
    altitude_m: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    indicated_airspeed_mps: Optional[float]
    true_airspeed_mps: Optional[float]
    ground_speed_mps: Optional[float]
    total_speed_mps: Optional[float]
    track_heading_deg: Optional[float]
    vertical_speed_mps: float
    turn_rate_deg_s: float
    track_rate_deg_s: float
    acceleration_mps2: float
    features: Tuple[float, ...]


def _split_escaped_fields(line: str) -> List[str]:
    """Split one ACMI CSV-style line while honoring backslash escaping."""

    fields: List[str] = []
    current: List[str] = []
    escaped = False

    for character in line:
        if escaped:
            current.append(character)
            escaped = False
            continue

        if character == "\\":
            escaped = True
            continue

        if character == ",":
            fields.append("".join(current))
            current = []
            continue

        current.append(character)

    if escaped:
        # Preserve a trailing backslash instead of silently dropping malformed data.
        current.append("\\")

    fields.append("".join(current))
    return fields


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None

    try:
        return float(value)
    except ValueError:
        return None


def _angle_delta_deg(current: float, previous: float) -> float:
    """Return the shortest signed angular delta in degrees."""

    return (current - previous + 180.0) % 360.0 - 180.0


def _horizontal_motion(
    *,
    previous_longitude: float,
    previous_latitude: float,
    longitude: float,
    latitude: float,
) -> Tuple[float, float, float]:
    """Return east, north, and horizontal displacement in meters."""

    previous_latitude_rad = radians(previous_latitude)
    latitude_rad = radians(latitude)
    mean_latitude_rad = (previous_latitude_rad + latitude_rad) / 2.0

    north_m = EARTH_RADIUS_M * (latitude_rad - previous_latitude_rad)
    east_m = (EARTH_RADIUS_M * cos(mean_latitude_rad) * radians(longitude - previous_longitude))
    return east_m, north_m, hypot(east_m, north_m)


def _feature_vector(
    *,
    altitude_m: float,
    ground_speed_mps: Optional[float],
    total_speed_mps: Optional[float],
    roll_deg: float,
    pitch_deg: float,
    vertical_speed_mps: float,
    track_rate_deg_s: float,
    acceleration_mps2: float,
) -> Tuple[float, ...]:
    """Create the stable MS1.1 normalized feature vector."""

    ground_speed = ground_speed_mps or 0.0
    total_speed = total_speed_mps or 0.0

    return (
        altitude_m / 1000.0,
        ground_speed / 100.0,
        total_speed / 100.0,
        abs(roll_deg) / 90.0,
        abs(pitch_deg) / 45.0,
        max(-vertical_speed_mps, 0.0) / 50.0,
        abs(track_rate_deg_s) / 30.0,
        abs(acceleration_mps2) / 10.0,
    )


class AcmiAircraftExtractor:
    """Incrementally extract one aircraft from text ACMI telemetry.

    When ``aircraft_id`` is omitted, the extractor still auto-detects ownship through ``DNTV.IsPlayer=true`` for convenience.
    Supplying an explicit object ID allows any recorded aircraft to be analyzed.
    """

    def __init__(self, aircraft_id: Optional[str] = None) -> None:
        self._forced_aircraft_id = aircraft_id
        self._aircraft_id = aircraft_id
        self._time_s: Optional[float] = None
        self._objects: Dict[str, Dict[str, str]] = {}
        self._previous: Optional[FlightSample] = None

    @property
    def aircraft_id(self) -> Optional[str]:
        return self._aircraft_id

    @property
    def player_id(self) -> Optional[str]:
        """Backward-compatible alias for older callers."""

        return self._aircraft_id

    @property
    def time_s(self) -> Optional[float]:
        return self._time_s

    def feed_line(self, line: str) -> Optional[FlightSample]:
        stripped = line.rstrip("\r\n")
        if not stripped:
            return None

        if stripped.startswith("#"):
            try:
                self._time_s = float(stripped[1:])
            except ValueError:
                self._time_s = None
            return None

        if stripped.startswith("-"):
            object_id = stripped[1:]
            self._objects.pop(object_id, None)
            if (self._forced_aircraft_id is None and object_id == self._aircraft_id):
                self._aircraft_id = None
                self._previous = None
            return None

        # Header lines such as FileType=... have no object ID and comma payload.
        if "," not in stripped or self._time_s is None:
            return None

        fields = _split_escaped_fields(stripped)
        if len(fields) < 2:
            return None

        object_id = fields[0]
        properties: Dict[str, str] = {}

        for field in fields[1:]:
            key, separator, value = field.partition("=")
            if separator and key:
                properties[key] = value

        state = self._objects.setdefault(object_id, {})
        state.update(properties)

        if (self._forced_aircraft_id is None and properties.get("DNTV.IsPlayer", "").strip().lower() == "true"):
            self._aircraft_id = object_id

        if object_id != self._aircraft_id or "T" not in properties:
            return None

        transform = state.get("T", "").split("|")
        if len(transform) < 6:
            return None

        try:
            longitude = float(transform[0])
            latitude = float(transform[1])
            altitude_m = float(transform[2])
            roll_deg = float(transform[3])
            pitch_deg = float(transform[4])
            yaw_deg = float(transform[5])
        except ValueError:
            return None

        indicated_airspeed_mps = _safe_float(state.get("IAS"))
        true_airspeed_mps = _safe_float(state.get("TAS"))

        ground_speed_mps: Optional[float] = None
        total_speed_mps: Optional[float] = None
        track_heading_deg: Optional[float] = None
        vertical_speed_mps = 0.0
        turn_rate_deg_s = 0.0
        track_rate_deg_s = 0.0
        acceleration_mps2 = 0.0

        if self._previous is not None:
            dt = self._time_s - self._previous.time_s
            if dt > 0.0:
                vertical_speed_mps = (altitude_m - self._previous.altitude_m) / dt
                turn_rate_deg_s = _angle_delta_deg(yaw_deg, self._previous.yaw_deg) / dt

                east_m, north_m, horizontal_distance_m = _horizontal_motion(previous_longitude=self._previous.longitude, previous_latitude=self._previous.latitude, longitude=longitude, latitude=latitude)
                ground_speed_mps = horizontal_distance_m / dt
                total_speed_mps = hypot(ground_speed_mps, vertical_speed_mps)

                if horizontal_distance_m > 1e-6:
                    track_heading_deg = (degrees(atan2(east_m, north_m)) + 360.0) % 360.0
                else:
                    track_heading_deg = self._previous.track_heading_deg

                if (track_heading_deg is not None and self._previous.track_heading_deg is not None):
                    track_rate_deg_s = _angle_delta_deg(track_heading_deg, self._previous.track_heading_deg) / dt

                if self._previous.total_speed_mps is not None:
                    acceleration_mps2 = (total_speed_mps - self._previous.total_speed_mps) / dt

        features = _feature_vector(
            altitude_m=altitude_m,
            ground_speed_mps=ground_speed_mps,
            total_speed_mps=total_speed_mps,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            vertical_speed_mps=vertical_speed_mps,
            track_rate_deg_s=track_rate_deg_s,
            acceleration_mps2=acceleration_mps2,
        )

        sample = FlightSample(
            time_s=self._time_s,
            object_id=object_id,
            aircraft_name=state.get("Name", ""),
            longitude=longitude,
            latitude=latitude,
            altitude_m=altitude_m,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            indicated_airspeed_mps=indicated_airspeed_mps,
            true_airspeed_mps=true_airspeed_mps,
            ground_speed_mps=ground_speed_mps,
            total_speed_mps=total_speed_mps,
            track_heading_deg=track_heading_deg,
            vertical_speed_mps=vertical_speed_mps,
            turn_rate_deg_s=turn_rate_deg_s,
            track_rate_deg_s=track_rate_deg_s,
            acceleration_mps2=acceleration_mps2,
            features=features,
        )
        self._previous = sample
        return sample


class AcmiPlayerExtractor(AcmiAircraftExtractor):
    """Backward-compatible wrapper for the original MS1 class name."""

    def __init__(self, player_id: Optional[str] = None) -> None:
        super().__init__(aircraft_id=player_id)


def _resolve_aircraft_id(aircraft_id: Optional[str], player_id: Optional[str]) -> Optional[str]:
    if aircraft_id is not None and player_id is not None:
        raise ValueError("Use either aircraft_id or player_id, not both")
    return aircraft_id if aircraft_id is not None else player_id


def iter_acmi_samples(
    lines: Iterable[str],
    *,
    aircraft_id: Optional[str] = None,
    player_id: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Iterator[FlightSample]:
    selected_id = _resolve_aircraft_id(aircraft_id, player_id)
    extractor = AcmiAircraftExtractor(aircraft_id=selected_id)

    for line in lines:
        sample = extractor.feed_line(line)

        if (end_time is not None and extractor.time_s is not None and extractor.time_s > end_time):
            break

        if sample is None:
            continue
        if start_time is not None and sample.time_s < start_time:
            continue
        yield sample


def _iter_text_handle(
    handle: TextIO,
    *,
    extractor: AcmiAircraftExtractor,
    follow: bool,
    poll_interval: float,
    start_time: Optional[float],
    end_time: Optional[float],
) -> Iterator[FlightSample]:
    while True:
        line = handle.readline()
        if line:
            sample = extractor.feed_line(line)

            if ( end_time is not None and extractor.time_s is not None and extractor.time_s > end_time):
                break

            if sample is not None:
                if start_time is None or sample.time_s >= start_time:
                    yield sample
            continue

        if not follow:
            break

        sleep(poll_interval)


def _select_zip_member(archive: ZipFile) -> str:
    members = [name for name in archive.namelist() if not name.endswith("/")]
    if not members:
        raise ValueError("ACMI ZIP archive contains no files")

    preferred = [name for name in members if name.lower().endswith((".txt.acmi", ".acmi"))]
    if preferred:
        return preferred[0]
    if len(members) == 1:
        return members[0]

    raise ValueError("ACMI ZIP archive contains multiple files and no obvious .acmi member")


def iter_acmi_file(
    path: Path,
    *,
    aircraft_id: Optional[str] = None,
    player_id: Optional[str] = None,
    follow: bool = False,
    poll_interval: float = 0.10,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Iterator[FlightSample]:
    """Read aircraft samples from loose or ZIP-compressed text ACMI."""

    if poll_interval <= 0.0:
        raise ValueError("poll_interval must be greater than zero")
    if start_time is not None and start_time < 0.0:
        raise ValueError("start_time must be zero or greater")
    if end_time is not None and end_time < 0.0:
        raise ValueError("end_time must be zero or greater")
    if (start_time is not None and end_time is not None and end_time < start_time):
        raise ValueError("end_time must be greater than or equal to start_time")

    selected_id = _resolve_aircraft_id(aircraft_id, player_id)
    extractor = AcmiAircraftExtractor(aircraft_id=selected_id)

    if is_zipfile(path):
        if follow:
            raise ValueError("--follow is not supported for ZIP-compressed ACMI")
        with ZipFile(path, "r") as archive:
            member = _select_zip_member(archive)
            with archive.open(member, "r") as raw_handle:
                with TextIOWrapper(raw_handle, encoding="utf-8") as handle:
                    yield from _iter_text_handle(
                        handle,
                        extractor=extractor,
                        follow=False,
                        poll_interval=poll_interval,
                        start_time=start_time,
                        end_time=end_time,
                    )
        return

    with path.open("r", encoding="utf-8") as handle:
        yield from _iter_text_handle(
            handle,
            extractor=extractor,
            follow=follow,
            poll_interval=poll_interval,
            start_time=start_time,
            end_time=end_time,
        )

from pathlib import Path
from datetime import datetime
from math import radians, sin, cos, sqrt, asin, atan2, degrees
import csv
import json

import numpy as np


INPUT_FILE = Path(
    r".\data\v8_extreme_precip_grid_forecasts_2025.npz"
)

OUTPUT_JSON = Path(
    r".\data\v8_anomaly_tracking_2025.json"
)

OUTPUT_CSV = Path(
    r".\data\v8_anomaly_tracking_2025.csv"
)

GRID_ROWS = 30
GRID_COLS = 30

MIN_COMPONENT_CELLS = 2

# Used only for event continuity matching.
# This is a tracking heuristic, not a physical wind-speed limit.
BASE_MAX_LINK_SPEED_KMH = 100.0


def parse_time(value):
    text = str(value)

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    return datetime.fromisoformat(text)


def hours_between(a, b):
    seconds = abs(
        (parse_time(b) - parse_time(a)).total_seconds()
    )

    return max(seconds / 3600.0, 1e-6)


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0088

    lat1 = radians(float(lat1))
    lon1 = radians(float(lon1))
    lat2 = radians(float(lat2))
    lon2 = radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2.0) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2.0) ** 2
    )

    return (
        2.0
        * earth_radius_km
        * asin(min(1.0, sqrt(a)))
    )


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1 = radians(float(lat1))
    lat2 = radians(float(lat2))
    dlon = radians(float(lon2) - float(lon1))

    y = sin(dlon) * cos(lat2)

    x = (
        cos(lat1) * sin(lat2)
        - sin(lat1)
        * cos(lat2)
        * cos(dlon)
    )

    bearing = degrees(
        atan2(y, x)
    )

    return (bearing + 360.0) % 360.0


def bearing_to_direction(bearing):
    if bearing is None:
        return None

    directions = [
        "N",
        "NE",
        "E",
        "SE",
        "S",
        "SW",
        "W",
        "NW",
    ]

    index = int(
        round(float(bearing) / 45.0)
    ) % 8

    return directions[index]


def clamp01(value):
    return np.clip(
        value,
        0.0,
        1.0,
    )


def normalized_extreme(
    values,
    p95,
    p99,
):
    denominator = max(
        float(p99) - float(p95),
        1e-6,
    )

    return clamp01(
        (
            values - float(p95)
        ) / denominator
    )


def build_grid_neighbors():
    neighbors = {}

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):

            index = (
                row * GRID_COLS
                + col
            )

            current = []

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):

                    if dr == 0 and dc == 0:
                        continue

                    nr = row + dr
                    nc = col + dc

                    if (
                        0 <= nr < GRID_ROWS
                        and 0 <= nc < GRID_COLS
                    ):
                        current.append(
                            nr * GRID_COLS
                            + nc
                        )

            neighbors[index] = current

    return neighbors


def connected_components(
    active_mask,
    neighbors,
):
    active_nodes = set(
        int(i)
        for i in np.flatnonzero(
            active_mask
        )
    )

    components = []

    while active_nodes:

        start = active_nodes.pop()

        component = [
            start
        ]

        stack = [
            start
        ]

        while stack:

            current = stack.pop()

            for nxt in neighbors[current]:

                if nxt in active_nodes:

                    active_nodes.remove(nxt)
                    component.append(nxt)
                    stack.append(nxt)

        if len(component) >= MIN_COMPONENT_CELLS:
            components.append(
                sorted(component)
            )

    return components


def component_statistics(
    nodes,
    severity,
    temperature,
    humidity,
    wind,
    precipitation,
    occurrence,
    latitudes,
    longitudes,
):
    indices = np.asarray(
        nodes,
        dtype=np.int64,
    )

    weights = np.asarray(
        severity[indices],
        dtype=np.float64,
    )

    if float(weights.sum()) <= 1e-9:
        weights = np.ones_like(
            weights
        )

    weights = weights / weights.sum()

    lat = float(
        np.sum(
            latitudes[indices]
            * weights
        )
    )

    lon = float(
        np.sum(
            longitudes[indices]
            * weights
        )
    )

    return {
        "nodes": [
            int(x)
            for x in indices
        ],
        "cell_count": int(
            len(indices)
        ),
        "centroid": {
            "latitude": lat,
            "longitude": lon,
        },
        "bbox": {
            "lat_min": float(
                np.min(
                    latitudes[indices]
                )
            ),
            "lat_max": float(
                np.max(
                    latitudes[indices]
                )
            ),
            "lon_min": float(
                np.min(
                    longitudes[indices]
                )
            ),
            "lon_max": float(
                np.max(
                    longitudes[indices]
                )
            ),
        },
        "severity": float(
            np.max(
                severity[indices]
            )
        ),
        "mean_severity": float(
            np.mean(
                severity[indices]
            )
        ),
        "temperature_mean": float(
            np.mean(
                temperature[indices]
            )
        ),
        "humidity_mean": float(
            np.mean(
                humidity[indices]
            )
        ),
        "wind_mean": float(
            np.mean(
                wind[indices]
            )
        ),
        "precipitation_mean": float(
            np.mean(
                precipitation[indices]
            )
        ),
        "precipitation_max": float(
            np.max(
                precipitation[indices]
            )
        ),
        "occurrence_probability_mean": float(
            np.mean(
                occurrence[indices]
            )
        ),
    }


def track_events(
    frames,
    hazard_name,
    horizon_hours,
):
    next_track_number = 1

    active_tracks = {}

    all_events = []

    for frame in frames:

        target_timestamp = frame[
            "target_timestamp"
        ]

        issue_timestamp = frame[
            "issue_timestamp"
        ]

        current_events = []

        used_previous = set()

        for event in sorted(
            frame["events"],
            key=lambda item: item["severity"],
            reverse=True,
        ):
            best_track_id = None
            best_cost = float("inf")
            best_distance = None
            best_overlap = 0.0

            for track_id, previous in active_tracks.items():

                previous_time = previous[
                    "target_timestamp"
                ]

                dt_hours = hours_between(
                    previous_time,
                    target_timestamp,
                )

                max_distance_km = max(
                    120.0,
                    BASE_MAX_LINK_SPEED_KMH
                    * dt_hours,
                )

                distance = haversine_km(
                    previous["latitude"],
                    previous["longitude"],
                    event["centroid"]["latitude"],
                    event["centroid"]["longitude"],
                )

                previous_nodes = set(
                    previous["nodes"]
                )

                current_nodes = set(
                    event["nodes"]
                )

                union = (
                    previous_nodes
                    | current_nodes
                )

                intersection = (
                    previous_nodes
                    & current_nodes
                )

                overlap = (
                    len(intersection)
                    / max(
                        1,
                        len(union),
                    )
                )

                if distance > max_distance_km:
                    continue

                # Strong preference for spatial overlap,
                # otherwise nearest-centroid continuity.
                cost = (
                    distance
                    - 250.0 * overlap
                )

                if cost < best_cost:
                    best_cost = cost
                    best_track_id = track_id
                    best_distance = distance
                    best_overlap = overlap

            if (
                best_track_id is None
                or best_track_id in used_previous
            ):
                track_id = (
                    f"{hazard_name}_"
                    f"H{horizon_hours}_"
                    f"E{next_track_number:05d}"
                )

                next_track_number += 1

                speed = None
                bearing = None
                direction = None

                duration_hours = 0.0
                previous_count = 0

            else:
                track_id = best_track_id

                previous = active_tracks[
                    track_id
                ]

                dt_hours = hours_between(
                    previous["target_timestamp"],
                    target_timestamp,
                )

                speed = (
                    best_distance
                    / dt_hours
                )

                bearing = bearing_deg(
                    previous["latitude"],
                    previous["longitude"],
                    event["centroid"]["latitude"],
                    event["centroid"]["longitude"],
                )

                direction = bearing_to_direction(
                    bearing
                )

                duration_hours = (
                    previous["duration_hours"]
                    + dt_hours
                )

                previous_count = (
                    previous["observation_count"]
                    + 1
                )

                used_previous.add(
                    track_id
                )

            event = dict(event)

            event[
                "track_id"
            ] = track_id

            event[
                "hazard"
            ] = hazard_name

            event[
                "horizon_hours"
            ] = int(horizon_hours)

            event[
                "target_timestamp"
            ] = target_timestamp

            event[
                "issue_timestamp"
            ] = issue_timestamp

            event[
                "speed_kmh"
            ] = (
                None
                if speed is None
                else float(speed)
            )

            event[
                "movement_bearing_deg"
            ] = (
                None
                if bearing is None
                else float(bearing)
            )

            event[
                "movement_direction"
            ] = direction

            event[
                "spatial_overlap"
            ] = float(
                best_overlap
            )

            event[
                "duration_hours"
            ] = float(
                duration_hours
            )

            event[
                "observation_count"
            ] = int(
                previous_count
            )

            current_events.append(
                event
            )

            active_tracks[
                track_id
            ] = {
                "target_timestamp": target_timestamp,
                "latitude": event[
                    "centroid"
                ]["latitude"],
                "longitude": event[
                    "centroid"
                ]["longitude"],
                "nodes": event[
                    "nodes"
                ],
                "duration_hours": duration_hours,
                "observation_count": previous_count,
            }

        all_events.extend(
            current_events
        )

    return all_events


def detect_precipitation(
    rain,
    occurrence,
    humidity,
    temperature,
    p95,
    p99,
):
    rain_score = normalized_extreme(
        rain,
        p95,
        p99,
    )

    occurrence_score = clamp01(
        (
            occurrence - 0.50
        ) / 0.50
    )

    moisture_score = clamp01(
        humidity / 100.0
    )

    # Moisture + precipitation occurrence are used
    # as a physics-aware consistency check.
    physics_score = (
        0.60 * occurrence_score
        + 0.40 * moisture_score
    )

    severity = (
        0.65 * rain_score
        + 0.20 * occurrence_score
        + 0.15 * moisture_score
    )

    active = (
        (
            rain >= float(p99)
        )
        & (
            occurrence >= 0.50
        )
        & (
            physics_score >= 0.55
        )
    )

    return active, severity, physics_score


def detect_heat(
    temperature,
    p95,
    p99,
):
    severity = normalized_extreme(
        temperature,
        p95,
        p99,
    )

    active = (
        temperature
        >= float(p99)
    )

    physics_score = np.ones_like(
        severity,
        dtype=np.float32,
    )

    return active, severity, physics_score


def detect_wind(
    wind,
    p95,
    p99,
):
    severity = normalized_extreme(
        wind,
        p95,
        p99,
    )

    active = (
        wind
        >= float(p99)
    )

    physics_score = np.ones_like(
        severity,
        dtype=np.float32,
    )

    return active, severity, physics_score


def make_frames(
    horizon_index,
    horizon_hours,
    timestamps,
    target_timestamps,
    temperature,
    humidity,
    wind,
    precipitation,
    occurrence,
    latitudes,
    longitudes,
    neighbors,
    hazard,
    p95_values,
    p99_values,
):
    frames = []

    p95 = float(
        p95_values
    )

    p99 = float(
        p99_values
    )

    for t in range(
        len(timestamps)
    ):
        target_time = str(
            target_timestamps[
                t,
                horizon_index
            ]
        )

        issue_time = str(
            timestamps[t]
        )

        temp = temperature[
            t,
            horizon_index
        ]

        hum = humidity[
            t,
            horizon_index
        ]

        wnd = wind[
            t,
            horizon_index
        ]

        rain = precipitation[
            t,
            horizon_index
        ]

        occ = occurrence[
            t,
            horizon_index
        ]

        if hazard == "extreme_precipitation":
            active, severity, physics_score = (
                detect_precipitation(
                    rain,
                    occ,
                    hum,
                    temp,
                    p95,
                    p99,
                )
            )

        elif hazard == "heat":
            active, severity, physics_score = (
                detect_heat(
                    temp,
                    p95,
                    p99,
                )
            )

        elif hazard == "high_wind":
            active, severity, physics_score = (
                detect_wind(
                    wnd,
                    p95,
                    p99,
                )
            )

        else:
            raise ValueError(
                f"Unknown hazard: {hazard}"
            )

        components = connected_components(
            active,
            neighbors,
        )

        events = []

        for nodes in components:

            stats = component_statistics(
                nodes,
                severity,
                temp,
                hum,
                wnd,
                rain,
                occ,
                latitudes,
                longitudes,
            )

            stats[
                "physics_consistency"
            ] = float(
                np.mean(
                    physics_score[
                        np.asarray(
                            nodes,
                            dtype=np.int64,
                        )
                    ]
                )
            )

            events.append(
                stats
            )

        frames.append(
            {
                "target_timestamp": target_time,
                "issue_timestamp": issue_time,
                "events": events,
            }
        )

    return frames


def summary_for_events(events):
    if not events:
        return {
            "event_snapshots": 0,
            "unique_tracks": 0,
            "max_severity": 0.0,
            "max_precipitation": 0.0,
            "max_speed_kmh": 0.0,
        }

    track_ids = {
        event["track_id"]
        for event in events
    }

    speeds = [
        event["speed_kmh"]
        for event in events
        if event["speed_kmh"] is not None
    ]

    return {
        "event_snapshots": int(
            len(events)
        ),
        "unique_tracks": int(
            len(track_ids)
        ),
        "max_severity": float(
            max(
                event["severity"]
                for event in events
            )
        ),
        "max_precipitation": float(
            max(
                event["precipitation_max"]
                for event in events
            )
        ),
        "max_speed_kmh": float(
            max(speeds)
            if speeds
            else 0.0
        ),
    }


def json_safe(value):
    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.ndarray,
    ):
        return value.tolist()

    raise TypeError(
        f"Unsupported type: {type(value)}"
    )


def main():
    print("=" * 78)
    print("VATAVARAN - EXTREME WEATHER ANOMALY TRACKING")
    print("=" * 78)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {INPUT_FILE}"
        )

    data = np.load(
        INPUT_FILE,
        allow_pickle=False,
    )

    horizons = data[
        "horizons"
    ].astype(int)

    timestamps = data[
        "timestamps"
    ]

    target_timestamps = data[
        "target_timestamps"
    ]

    node_ids = data[
        "node_ids"
    ]

    latitudes = data[
        "latitudes"
    ]

    longitudes = data[
        "longitudes"
    ]

    temperature = data[
        "temperature"
    ]

    humidity = data[
        "humidity"
    ]

    wind = data[
        "wind"
    ]

    precipitation = data[
        "precipitation"
    ]

    occurrence = data[
        "occurrence_probability"
    ]

    if temperature.shape[2] != 900:
        raise ValueError(
            "Expected 900 grid nodes."
        )

    neighbors = build_grid_neighbors()

    print("")
    print(
        f"Forecast timestamps : {len(timestamps)}"
    )

    print(
        f"Horizons            : {horizons.tolist()}"
    )

    print(
        f"Grid nodes           : {len(node_ids)}"
    )

    results = {
        "model": "VATAVARAN V8",
        "task": (
            "Extreme weather anomaly "
            "detection and spatio-temporal tracking"
        ),
        "source_file": str(
            INPUT_FILE
        ),
        "grid": {
            "rows": GRID_ROWS,
            "columns": GRID_COLS,
            "nodes": int(
                len(node_ids)
            ),
            "graph_type": (
                "8-connected geographic grid"
            ),
        },
        "method": {
            "spatial_detection": (
                "8-connected components"
            ),
            "tracking": (
                "centroid continuity + "
                "spatial overlap"
            ),
            "physics_aware_check": (
                "precipitation occurrence + "
                "relative humidity consistency"
            ),
            "warning": (
                "Thresholds are forecast-distribution "
                "based heuristics, not calibrated "
                "climatological return periods."
            ),
        },
        "thresholds": {},
        "hazards": {},
    }

    flat_rows = []

    hazard_names = [
        "extreme_precipitation",
        "heat",
        "high_wind",
    ]

    for horizon_index, horizon in enumerate(
        horizons
    ):
        horizon_key = f"h{int(horizon)}"

        results[
            "thresholds"
        ][horizon_key] = {}

        results[
            "hazards"
        ][horizon_key] = {}

        print("")
        print(
            "-" * 78
        )
        print(
            f"Horizon {int(horizon)} hours"
        )
        print(
            "-" * 78
        )

        fields = {
            "extreme_precipitation": precipitation[
                :,
                horizon_index,
                :,
            ],
            "heat": temperature[
                :,
                horizon_index,
                :,
            ],
            "high_wind": wind[
                :,
                horizon_index,
                :,
            ],
        }

        for hazard in hazard_names:

            field = fields[
                hazard
            ]

            p95 = float(
                np.percentile(
                    field,
                    95,
                )
            )

            p99 = float(
                np.percentile(
                    field,
                    99,
                )
            )

            results[
                "thresholds"
            ][horizon_key][hazard] = {
                "p95": p95,
                "p99": p99,
            }

            frames = make_frames(
                horizon_index,
                int(horizon),
                timestamps,
                target_timestamps,
                temperature,
                humidity,
                wind,
                precipitation,
                occurrence,
                latitudes,
                longitudes,
                neighbors,
                hazard,
                p95,
                p99,
            )

            events = track_events(
                frames,
                hazard,
                int(horizon),
            )

            results[
                "hazards"
            ][horizon_key][hazard] = {
                "summary": summary_for_events(
                    events
                ),
                "events": events,
            }

            for event in events:
                flat_rows.append(
                    {
                        "horizon_hours": int(
                            horizon
                        ),
                        "hazard": hazard,
                        "track_id": event[
                            "track_id"
                        ],
                        "target_timestamp": event[
                            "target_timestamp"
                        ],
                        "issue_timestamp": event[
                            "issue_timestamp"
                        ],
                        "cell_count": event[
                            "cell_count"
                        ],
                        "severity": event[
                            "severity"
                        ],
                        "mean_severity": event[
                            "mean_severity"
                        ],
                        "latitude": event[
                            "centroid"
                        ]["latitude"],
                        "longitude": event[
                            "centroid"
                        ]["longitude"],
                        "lat_min": event[
                            "bbox"
                        ]["lat_min"],
                        "lat_max": event[
                            "bbox"
                        ]["lat_max"],
                        "lon_min": event[
                            "bbox"
                        ]["lon_min"],
                        "lon_max": event[
                            "bbox"
                        ]["lon_max"],
                        "temperature_mean": event[
                            "temperature_mean"
                        ],
                        "humidity_mean": event[
                            "humidity_mean"
                        ],
                        "wind_mean": event[
                            "wind_mean"
                        ],
                        "precipitation_mean": event[
                            "precipitation_mean"
                        ],
                        "precipitation_max": event[
                            "precipitation_max"
                        ],
                        "occurrence_probability_mean": event[
                            "occurrence_probability_mean"
                        ],
                        "physics_consistency": event[
                            "physics_consistency"
                        ],
                        "speed_kmh": event[
                            "speed_kmh"
                        ],
                        "movement_bearing_deg": event[
                            "movement_bearing_deg"
                        ],
                        "movement_direction": event[
                            "movement_direction"
                        ],
                        "spatial_overlap": event[
                            "spatial_overlap"
                        ],
                        "duration_hours": event[
                            "duration_hours"
                        ],
                    }
                )

            summary = summary_for_events(
                events
            )

            print(
                f"{hazard:24s} "
                f"tracks={summary['unique_tracks']:5d} "
                f"snapshots={summary['event_snapshots']:5d} "
                f"max_severity={summary['max_severity']:.3f}"
            )

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
            default=json_safe,
        )

    fieldnames = list(
        flat_rows[0].keys()
    ) if flat_rows else [
        "horizon_hours",
        "hazard",
        "track_id",
        "target_timestamp",
        "issue_timestamp",
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            flat_rows
        )

    print("")
    print("=" * 78)
    print("ANOMALY TRACKING COMPLETE")
    print("=" * 78)

    print(
        f"JSON : {OUTPUT_JSON}"
    )

    print(
        f"CSV  : {OUTPUT_CSV}"
    )

    print(
        f"Total event snapshots: {len(flat_rows)}"
    )


if __name__ == "__main__":
    main()

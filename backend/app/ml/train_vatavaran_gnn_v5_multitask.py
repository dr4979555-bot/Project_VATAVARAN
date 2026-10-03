from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from app.ml.nationwide_spatial_grid import build_graph_structure
from app.ml.weather_graph_builder import CITY_COORDINATES


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "multi_city_weather_ml_2021_2025.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "vatavaran_gnn_v5_multitask.pt"
)

SEED = 42
TIME_STRIDE = 4

EPOCHS = 8
SAMPLES_PER_EPOCH = 3000

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

HIDDEN_FEATURES = 32

WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]

REGRESSION_TARGETS = [
    "target_temperature",
    "target_humidity",
    "target_wind_speed",
    "target_precipitation_log",
]

TARGET_NAMES = [
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",
]

CITY_NAMES = [
    "Ahmedabad",
    "Bengaluru",
    "Chennai",
    "Delhi",
    "Hyderabad",
    "Kolkata",
    "Mumbai",
    "Patna",
]

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(
    max(1, min(6, torch.get_num_threads()))
)


# ============================================================
# MODEL
# ============================================================

class MaskAwareGraphConv(nn.Module):

    def __init__(
        self,
        in_features,
        out_features
    ):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features
        )

    def forward(
        self,
        x,
        edge_index,
        observation_mask
    ):

        num_nodes = x.size(0)

        source = edge_index[0]
        target = edge_index[1]

        source_observed = (
            observation_mask[source] > 0.5
        )

        valid_source = source[
            source_observed
        ]

        valid_target = target[
            source_observed
        ]

        messages = x[
            valid_source
        ]

        aggregated = torch.zeros_like(x)

        if messages.numel() > 0:

            aggregated.index_add_(
                0,
                valid_target,
                messages
            )

        degree = torch.zeros(
            num_nodes,
            dtype=x.dtype,
            device=x.device
        )

        if valid_target.numel() > 0:

            degree.index_add_(
                0,
                valid_target,
                torch.ones(
                    valid_target.shape[0],
                    dtype=x.dtype,
                    device=x.device
                )
            )

        aggregated = (
            aggregated + x
        )

        degree = (
            degree + 1.0
        )

        aggregated = (
            aggregated
            / degree.unsqueeze(1)
        )

        return self.linear(
            aggregated
        )


class VatavaranGNNV5(nn.Module):

    def __init__(
        self,
        input_features=8,
        hidden_features=32
    ):

        super().__init__()

        self.conv1 = MaskAwareGraphConv(
            input_features,
            hidden_features
        )

        self.conv2 = MaskAwareGraphConv(
            hidden_features,
            hidden_features
        )

        # Separate regression heads.
        self.temperature_head = nn.Linear(
            hidden_features,
            1
        )

        self.humidity_head = nn.Linear(
            hidden_features,
            1
        )

        self.wind_head = nn.Linear(
            hidden_features,
            1
        )

        # Precipitation occurrence:
        # probability of rain > 0 in next hour.
        self.precipitation_occurrence_head = nn.Linear(
            hidden_features,
            1
        )

        # Precipitation amount:
        # normalized log1p(next-hour precipitation).
        self.precipitation_amount_head = nn.Linear(
            hidden_features,
            1
        )

    def forward(
        self,
        x,
        edge_index
    ):

        observation_mask = x[:, -1]

        x = self.conv1(
            x,
            edge_index,
            observation_mask
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index,
            observation_mask
        )

        x = torch.relu(x)

        return {
            "temperature":
                self.temperature_head(x),

            "humidity":
                self.humidity_head(x),

            "wind_speed":
                self.wind_head(x),

            "precipitation_occurrence":
                self.precipitation_occurrence_head(x),

            "precipitation_amount":
                self.precipitation_amount_head(x),
        }


# ============================================================
# GRID MAPPING
# ============================================================

def build_city_grid_mapping(graph):

    mapping = {}

    nodes = graph["nodes"]

    for city, coordinates in (
        CITY_COORDINATES.items()
    ):

        latitude = coordinates[0]
        longitude = coordinates[1]

        best_node = None
        best_distance = float("inf")

        for node in nodes:

            dlat = (
                node.latitude -
                latitude
            )

            dlon = (
                node.longitude -
                longitude
            )

            distance = (
                dlat * dlat +
                dlon * dlon
            )

            if distance < best_distance:

                best_distance = distance
                best_node = node

        mapping[city] = (
            best_node.node_id
        )

    return mapping


def node_id_to_index(node_id):

    parts = node_id.split("_")

    row = int(parts[1])
    column = int(parts[2])

    return row * 30 + column


# ============================================================
# INPUT NORMALIZATION
# ============================================================

def calculate_input_normalization(
    train_df
):

    statistics = {}

    for feature in WEATHER_FEATURES:

        mean = float(
            train_df[feature].mean()
        )

        std = float(
            train_df[feature].std()
        )

        if (
            not np.isfinite(std)
            or std < 1e-6
        ):

            std = 1.0

        statistics[feature] = {
            "mean": mean,
            "std": std,
        }

    return statistics


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def calculate_target_normalization(
    train_df
):

    statistics = {}

    for feature in REGRESSION_TARGETS:

        values = train_df[
            feature
        ].dropna()

        mean = float(
            values.mean()
        )

        std = float(
            values.std()
        )

        if (
            not np.isfinite(std)
            or std < 1e-6
        ):

            std = 1.0

        statistics[feature] = {
            "mean": mean,
            "std": std,
        }

    return statistics


def normalize_value(
    value,
    feature,
    statistics
):

    if (
        value is None
        or not np.isfinite(value)
    ):

        return 0.0

    mean = statistics[
        feature
    ]["mean"]

    std = statistics[
        feature
    ]["std"]

    return (
        float(value) - mean
    ) / std


def denormalize_value(
    value,
    feature,
    statistics
):

    mean = statistics[
        feature
    ]["mean"]

    std = statistics[
        feature
    ]["std"]

    return (
        float(value) * std
    ) + mean


# ============================================================
# TARGET CREATION
# ============================================================

def create_targets(df):

    df = df.copy()

    df = df.sort_values(
        ["location", "time"]
    )

    # Existing next-hour temperature target.
    df["target_temperature"] = (
        df["target_temperature"]
    )

    # Next-hour humidity.
    df["target_humidity"] = (
        df.groupby("location")[
            "relative_humidity_2m"
        ].shift(-1)
    )

    # Next-hour wind speed.
    df["target_wind_speed"] = (
        df.groupby("location")[
            "wind_speed_10m"
        ].shift(-1)
    )

    # Next-hour precipitation.
    df["target_precipitation"] = (
        df.groupby("location")[
            "precipitation"
        ].shift(-1)
    )

    # Rain / no-rain target.
    df["target_precipitation_occurrence"] = (
        df["target_precipitation"] > 0.0
    ).astype(np.float32)

    # Log transform for highly skewed precipitation amount.
    df["target_precipitation_log"] = np.log1p(
        np.maximum(
            df["target_precipitation"],
            0.0
        )
    )

    return df


# ============================================================
# GRAPH SNAPSHOT
# ============================================================

def build_snapshot(
    timestamp,
    city_lookup,
    city_grid_mapping,
    normalization
):

    node_count = 900

    x = np.zeros(
        (node_count, 8),
        dtype=np.float32
    )

    timestamp_rows = city_lookup.get(
        timestamp,
        {}
    )

    for city in CITY_NAMES:

        row = timestamp_rows.get(
            city
        )

        if row is None:
            continue

        node_id = city_grid_mapping.get(
            city
        )

        if node_id is None:
            continue

        node_index = node_id_to_index(
            node_id
        )

        for feature_index, feature in enumerate(
            WEATHER_FEATURES
        ):

            value = row.get(
                feature
            )

            x[
                node_index,
                feature_index
            ] = normalize_value(
                value,
                feature,
                normalization
            )

        x[
            node_index,
            7
        ] = 1.0

    return torch.tensor(
        x,
        dtype=torch.float32
    )


# ============================================================
# CITY LOOKUP
# ============================================================

def build_city_lookup(df):

    lookup = {}

    for row in df.itertuples(
        index=False
    ):

        timestamp = row.time
        city = row.location

        if timestamp not in lookup:
            lookup[timestamp] = {}

        lookup[timestamp][city] = {
            feature: getattr(
                row,
                feature
            )
            for feature in WEATHER_FEATURES
        }

    return lookup


# ============================================================
# TARGET LOOKUP
# ============================================================

def build_target_lookup(df):

    lookup = {}

    for row in df.itertuples(
        index=False
    ):

        timestamp = row.time
        city = row.location

        values = [
            float(
                row.target_temperature
            ),
            float(
                row.target_humidity
            ),
            float(
                row.target_wind_speed
            ),
            float(
                row.target_precipitation_log
            ),
        ]

        occurrence = float(
            row.target_precipitation_occurrence
        )

        if timestamp not in lookup:
            lookup[timestamp] = {}

        lookup[timestamp][city] = {
            "values": values,
            "occurrence": occurrence,
        }

    return lookup


# ============================================================
# TARGET TENSOR
# ============================================================

def build_target_tensor(
    timestamp,
    target_lookup,
    city_grid_mapping,
    target_normalization
):

    continuous_target = torch.zeros(
        (900, 4),
        dtype=torch.float32
    )

    occurrence_target = torch.zeros(
        900,
        dtype=torch.float32
    )

    target_mask = torch.zeros(
        900,
        dtype=torch.bool
    )

    wet_mask = torch.zeros(
        900,
        dtype=torch.bool
    )

    rows = target_lookup.get(
        timestamp,
        {}
    )

    for city in CITY_NAMES:

        if city not in rows:
            continue

        node_id = city_grid_mapping.get(
            city
        )

        if node_id is None:
            continue

        node_index = node_id_to_index(
            node_id
        )

        values = rows[city][
            "values"
        ]

        occurrence = rows[city][
            "occurrence"
        ]

        for index, feature in enumerate(
            REGRESSION_TARGETS
        ):

            continuous_target[
                node_index,
                index
            ] = normalize_value(
                values[index],
                feature,
                target_normalization
            )

        occurrence_target[
            node_index
        ] = occurrence

        target_mask[
            node_index
        ] = True

        wet_mask[
            node_index
        ] = occurrence > 0.5

    return (
        continuous_target,
        occurrence_target,
        target_mask,
        wet_mask,
    )


# ============================================================
# TIMESTAMP SAMPLING
# ============================================================

def sample_timestamps(
    timestamps,
    count
):

    timestamps = list(
        timestamps
    )

    if len(timestamps) <= count:
        return timestamps

    return random.sample(
        timestamps,
        count
    )


# ============================================================
# METRICS
# ============================================================

def regression_metrics(
    predictions,
    actuals
):

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    actuals = np.asarray(
        actuals,
        dtype=float
    )

    errors = (
        predictions -
        actuals
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    ss_res = np.sum(
        errors ** 2
    )

    ss_tot = np.sum(
        (
            actuals -
            actuals.mean()
        ) ** 2
    )

    r2 = (
        1 -
        ss_res / ss_tot
        if ss_tot > 0
        else 0.0
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def binary_metrics(
    probabilities,
    actuals,
    threshold=0.5
):

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    actuals = np.asarray(
        actuals,
        dtype=int
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    true_positive = int(
        np.sum(
            (
                predictions == 1
            )
            &
            (
                actuals == 1
            )
        )
    )

    false_positive = int(
        np.sum(
            (
                predictions == 1
            )
            &
            (
                actuals == 0
            )
        )
    )

    false_negative = int(
        np.sum(
            (
                predictions == 0
            )
            &
            (
                actuals == 1
            )
        )
    )

    true_negative = int(
        np.sum(
            (
                predictions == 0
            )
            &
            (
                actuals == 0
            )
        )
    )

    precision = (
        true_positive /
        (
            true_positive +
            false_positive
        )
        if (
            true_positive +
            false_positive
        ) > 0
        else 0.0
    )

    recall = (
        true_positive /
        (
            true_positive +
            false_negative
        )
        if (
            true_positive +
            false_negative
        ) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (
            precision +
            recall
        )
        if (
            precision +
            recall
        ) > 0
        else 0.0
    )

    accuracy = (
        (
            true_positive +
            true_negative
        )
        /
        max(
            1,
            len(actuals)
        )
    )

    return {
        "Precision": float(
            precision
        ),
        "Recall": float(
            recall
        ),
        "F1": float(f1),
        "Accuracy": float(
            accuracy
        ),
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    timestamps,
    city_lookup,
    target_lookup,
    city_grid_mapping,
    input_normalization,
    target_normalization,
    edge_index,
    occurrence_threshold=0.5
):

    model.eval()

    predictions = {
        name: []
        for name in TARGET_NAMES
    }

    actuals = {
        name: []
        for name in TARGET_NAMES
    }

    occurrence_probabilities = []
    occurrence_actuals = []

    wet_predictions = []
    wet_actuals = []

    normalized_maes = []

    with torch.no_grad():

        for timestamp in timestamps:

            x = build_snapshot(
                timestamp,
                city_lookup,
                city_grid_mapping,
                input_normalization
            ).to(DEVICE)

            output = model(
                x,
                edge_index
            )

            (
                continuous_target,
                occurrence_target,
                target_mask,
                wet_mask
            ) = build_target_tensor(
                timestamp,
                target_lookup,
                city_grid_mapping,
                target_normalization
            )

            timestamp_rows = city_lookup.get(
                timestamp,
                {}
            )

            target_rows = target_lookup.get(
                timestamp,
                {}
            )

            for city in CITY_NAMES:

                if city not in timestamp_rows:
                    continue

                if city not in target_rows:
                    continue

                node_id = (
                    city_grid_mapping.get(
                        city
                    )
                )

                if node_id is None:
                    continue

                node_index = (
                    node_id_to_index(
                        node_id
                    )
                )

                current = (
                    timestamp_rows[city]
                )

                target_data = (
                    target_rows[city]
                )

                raw_target_values = np.asarray(
                    target_data["values"],
                    dtype=float
                )

                predicted_normalized = np.array(
                    [
                        float(
                            output["temperature"][
                                node_index
                            ].item()
                        ),
                        float(
                            output["humidity"][
                                node_index
                            ].item()
                        ),
                        float(
                            output["wind_speed"][
                                node_index
                            ].item()
                        ),
                        float(
                            output["precipitation_amount"][
                                node_index
                            ].item()
                        ),
                    ]
                )

                actual_values = [
                    float(raw_target_values[0]),
                    float(raw_target_values[1]),
                    float(raw_target_values[2]),
                    max(
                        0.0,
                        np.expm1(
                            float(
                                raw_target_values[3]
                            )
                        )
                    ),
                ]

                for index, name in enumerate(
                    TARGET_NAMES
                ):

                    feature = (
                        REGRESSION_TARGETS[
                            index
                        ]
                    )

                    predicted_value = (
                        denormalize_value(
                            predicted_normalized[
                                index
                            ],
                            feature,
                            target_normalization
                        )
                    )

                    actual = (
                        actual_values[index]
                    )

                    if name == "temperature":

                        predicted = float(
                            predicted_value
                        )

                    elif name == "humidity":

                        predicted = float(
                            np.clip(
                                predicted_value,
                                0.0,
                                100.0
                            )
                        )

                    elif name == "wind_speed":

                        predicted = float(
                            max(
                                0.0,
                                predicted_value
                            )
                        )

                    else:

                        predicted_log = float(
                            predicted_value
                        )

                        predicted_log = float(
                            np.clip(
                                predicted_log,
                                0.0,
                                10.0
                            )
                        )

                        predicted = float(
                            np.expm1(
                                predicted_log
                            )
                        )

                    predictions[
                        name
                    ].append(
                        predicted
                    )

                    actuals[
                        name
                    ].append(
                        actual
                    )

                    if index < 3:

                        train_std = max(
                            1e-6,
                            target_normalization[
                                feature
                            ]["std"]
                        )

                        normalized_maes.append(
                            abs(
                                predicted -
                                actual
                            )
                            /
                            train_std
                        )

                    else:

                        log_std = max(
                            1e-6,
                            target_normalization[
                                "target_precipitation_log"
                            ]["std"]
                        )

                        actual_log = float(
                            raw_target_values[3]
                        )

                        predicted_log = float(
                            np.log1p(
                                max(
                                    0.0,
                                    predicted
                                )
                            )
                        )

                        normalized_maes.append(
                            abs(
                                predicted_log -
                                actual_log
                            )
                            /
                            log_std
                        )

                occurrence_logit = float(
                    output[
                        "precipitation_occurrence"
                    ][
                        node_index
                    ].item()
                )

                occurrence_probability = float(
                    torch.sigmoid(
                        torch.tensor(
                            occurrence_logit
                        )
                    ).item()
                )

                actual_occurrence = int(
                    target_data[
                        "occurrence"
                    ]
                )

                occurrence_probabilities.append(
                    occurrence_probability
                )

                occurrence_actuals.append(
                    actual_occurrence
                )

                if actual_occurrence == 1:

                    predicted_amount = (
                        predictions[
                            "precipitation"
                        ][-1]
                    )

                    actual_amount = (
                        actuals[
                            "precipitation"
                        ][-1]
                    )

                    # Evaluate rain amount on
                    # actual wet cases too.
                    wet_predictions.append(
                        predicted_amount
                    )

                    wet_actuals.append(
                        actual_amount
                    )

    metrics = {}

    for name in TARGET_NAMES:

        metrics[name] = regression_metrics(
            predictions[name],
            actuals[name]
        )

    occurrence_results = (
        binary_metrics(
            occurrence_probabilities,
            occurrence_actuals,
            threshold=occurrence_threshold
        )
    )

    metrics[
        "precipitation_occurrence"
    ] = occurrence_results

    if wet_actuals:

        metrics[
            "precipitation_wet_only"
        ] = regression_metrics(
            wet_predictions,
            wet_actuals
        )

    metrics["samples"] = len(
        predictions["temperature"]
    )

    # Model-selection score:
    # normalized errors for the three
    # direct weather variables,
    # normalized precipitation MAE,
    # and rain occurrence penalty.
    precipitation_std = max(
        1e-6,
        target_normalization[
            "target_precipitation_log"
        ]["std"]
    )

    precipitation_mae_normalized = (
        metrics["precipitation"]["MAE"]
        /
        max(
            precipitation_std,
            1e-6
        )
    )

    mean_normalized_regression_error = (
        float(
            np.mean(
                normalized_maes
            )
        )
        if normalized_maes
        else 0.0
    )

    occurrence_penalty = (
        1.0 -
        occurrence_results["F1"]
    )

    metrics[
        "selection_score"
    ] = float(
        (
            mean_normalized_regression_error
            +
            precipitation_mae_normalized
            +
            occurrence_penalty
        )
        / 3.0
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("VATAVARAN GNN V5 - MULTI-TASK WEATHER MODEL")
    print("=" * 70)
    print()

    print(
        "Loading dataset..."
    )

    df = pd.read_csv(
        DATA_PATH
    )

    df["time"] = pd.to_datetime(
        df["time"]
    )

    df = df.sort_values(
        ["location", "time"]
    ).reset_index(
        drop=True
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # TARGET CREATION
    # --------------------------------------------------------

    print()
    print(
        "Creating next-hour targets..."
    )

    df = create_targets(
        df
    )

    print(
        "Targets:"
    )

    print(
        "  ✓ target_temperature"
    )

    print(
        "  ✓ target_humidity"
    )

    print(
        "  ✓ target_wind_speed"
    )

    print(
        "  ✓ target_precipitation"
    )

    print(
        "  ✓ precipitation occurrence"
    )

    print(
        "  ✓ log precipitation amount"
    )

    # --------------------------------------------------------
    # REMOVE INVALID TARGET ROWS
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "target_temperature",
            "target_humidity",
            "target_wind_speed",
            "target_precipitation",
            "target_precipitation_log",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------------

    train_df = df[
        df["time"].dt.year <= 2024
    ].copy()

    test_df = df[
        df["time"].dt.year == 2025
    ].copy()

    training_df = train_df[
        train_df["time"].dt.year <= 2023
    ].copy()

    validation_df = train_df[
        train_df["time"].dt.year == 2024
    ].copy()

    print()
    print(
        f"Training rows:   {len(training_df)}"
    )

    print(
        f"Validation rows: {len(validation_df)}"
    )

    print(
        f"Test rows:       {len(test_df)}"
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    input_normalization = (
        calculate_input_normalization(
            training_df
        )
    )

    target_normalization = (
        calculate_target_normalization(
            training_df
        )
    )

    # --------------------------------------------------------
    # RAIN CLASS BALANCE
    # --------------------------------------------------------

    rain_values = (
        training_df[
            "target_precipitation_occurrence"
        ].values
    )

    positive_count = float(
        np.sum(rain_values == 1)
    )

    negative_count = float(
        np.sum(rain_values == 0)
    )

    occurrence_pos_weight = (
        negative_count /
        max(
            positive_count,
            1.0
        )
    )

    print()
    print(
        "Rain occurrence training distribution:"
    )

    print(
        f"  Dry samples: "
        f"{int(negative_count)}"
    )

    print(
        f"  Wet samples: "
        f"{int(positive_count)}"
    )

    print(
        f"  Positive class weight: "
        f"{occurrence_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    print()
    print(
        "Building nationwide graph..."
    )

    graph = build_graph_structure(
        grid_size=1.0
    )

    city_grid_mapping = (
        build_city_grid_mapping(
            graph
        )
    )

    print(
        f"Graph nodes: "
        f"{len(graph['nodes'])}"
    )

    print(
        f"Graph edges: "
        f"{len(graph['edges'])}"
    )

    # --------------------------------------------------------
    # EDGE INDEX
    # --------------------------------------------------------

    edge_sources = []
    edge_targets = []

    for source, target in graph["edges"]:

        source_index = (
            node_id_to_index(
                source
            )
        )

        target_index = (
            node_id_to_index(
                target
            )
        )

        edge_sources.append(
            source_index
        )

        edge_targets.append(
            target_index
        )

        edge_sources.append(
            target_index
        )

        edge_targets.append(
            source_index
        )

    edge_index = torch.tensor(
        [
            edge_sources,
            edge_targets
        ],
        dtype=torch.long,
        device=DEVICE
    )

    print(
        f"Directed edges: "
        f"{edge_index.shape[1]}"
    )

    # --------------------------------------------------------
    # LOOKUPS
    # --------------------------------------------------------

    training_lookup = (
        build_city_lookup(
            training_df
        )
    )

    validation_lookup = (
        build_city_lookup(
            validation_df
        )
    )

    test_lookup = (
        build_city_lookup(
            test_df
        )
    )

    training_targets = (
        build_target_lookup(
            training_df
        )
    )

    validation_targets = (
        build_target_lookup(
            validation_df
        )
    )

    test_targets = (
        build_target_lookup(
            test_df
        )
    )

    # --------------------------------------------------------
    # TIMESTAMPS
    # --------------------------------------------------------

    training_timestamps = sorted(
        training_lookup.keys()
    )[::TIME_STRIDE]

    validation_timestamps = sorted(
        validation_lookup.keys()
    )[::TIME_STRIDE]

    test_timestamps = sorted(
        test_lookup.keys()
    )[::TIME_STRIDE]

    print()
    print(
        f"Training timestamps: "
        f"{len(training_timestamps)}"
    )

    print(
        f"Validation timestamps: "
        f"{len(validation_timestamps)}"
    )

    print(
        f"Test timestamps: "
        f"{len(test_timestamps)}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = VatavaranGNNV5(
        input_features=8,
        hidden_features=HIDDEN_FEATURES
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    regression_loss = nn.MSELoss()

    occurrence_loss = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                occurrence_pos_weight,
                dtype=torch.float32,
                device=DEVICE
            )
        )
    )

    best_validation_score = float(
        "inf"
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print()
    print(
        "Starting V5 training..."
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        epoch_timestamps = (
            sample_timestamps(
                training_timestamps,
                min(
                    SAMPLES_PER_EPOCH,
                    len(training_timestamps)
                )
            )
        )

        total_loss = 0.0
        valid_batches = 0

        for timestamp in epoch_timestamps:

            x = build_snapshot(
                timestamp,
                training_lookup,
                city_grid_mapping,
                input_normalization
            ).to(DEVICE)

            (
                target,
                occurrence_target,
                target_mask,
                wet_mask
            ) = build_target_tensor(
                timestamp,
                training_targets,
                city_grid_mapping,
                target_normalization
            )

            target = target.to(DEVICE)
            occurrence_target = (
                occurrence_target.to(DEVICE)
            )
            target_mask = (
                target_mask.to(DEVICE)
            )
            wet_mask = (
                wet_mask.to(DEVICE)
            )

            if target_mask.sum() == 0:
                continue

            optimizer.zero_grad()

            output = model(
                x,
                edge_index
            )

            # Three direct regression heads.
            temperature_loss = (
                regression_loss(
                    output["temperature"][
                        target_mask,
                        0
                    ],
                    target[
                        target_mask,
                        0
                    ]
                )
            )

            humidity_loss = (
                regression_loss(
                    output["humidity"][
                        target_mask,
                        0
                    ],
                    target[
                        target_mask,
                        1
                    ]
                )
            )

            wind_loss = (
                regression_loss(
                    output["wind_speed"][
                        target_mask,
                        0
                    ],
                    target[
                        target_mask,
                        2
                    ]
                )
            )

            # Rain occurrence head.
            occurrence_head_loss = (
                occurrence_loss(
                    output[
                        "precipitation_occurrence"
                    ][
                        target_mask,
                        0
                    ],
                    occurrence_target[
                        target_mask
                    ]
                )
            )

            # Rain amount head is trained only
            # on actual wet cases.
            if wet_mask.sum() > 0:

                amount_loss = regression_loss(
                    output[
                        "precipitation_amount"
                    ][
                        wet_mask,
                        0
                    ],
                    target[
                        wet_mask,
                        3
                    ]
                )

            else:

                amount_loss = torch.tensor(
                    0.0,
                    device=DEVICE
                )

            loss = (
                temperature_loss
                +
                humidity_loss
                +
                wind_loss
                +
                occurrence_head_loss
                +
                amount_loss
            ) / 5.0

            loss.backward()

            optimizer.step()

            total_loss += float(
                loss.item()
            )

            valid_batches += 1

        average_loss = (
            total_loss /
            max(
                1,
                valid_batches
            )
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_metrics = (
            evaluate_model(
                model,
                validation_timestamps,
                validation_lookup,
                validation_targets,
                city_grid_mapping,
                input_normalization,
                target_normalization,
                edge_index
            )
        )

        validation_score = (
            validation_metrics.get(
                "selection_score",
                float("inf")
            )
        )

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: "
            f"{average_loss:.4f}"
        )

        for name in TARGET_NAMES:

            metrics = (
                validation_metrics.get(
                    name,
                    {}
                )
            )

            unit = ""

            if name == "temperature":
                unit = " °C"

            elif name == "humidity":
                unit = " %"

            elif name == "wind_speed":
                unit = " km/h"

            elif name == "precipitation":
                unit = " mm"

            print(
                f"  {name:<15} "
                f"MAE={metrics.get('MAE', 0):.4f}{unit} | "
                f"RMSE={metrics.get('RMSE', 0):.4f} | "
                f"R2={metrics.get('R2', 0):.4f}"
            )

        occurrence_metrics = (
            validation_metrics.get(
                "precipitation_occurrence",
                {}
            )
        )

        print(
            "  rain_occurrence "
            f"Precision={occurrence_metrics.get('Precision', 0):.4f} | "
            f"Recall={occurrence_metrics.get('Recall', 0):.4f} | "
            f"F1={occurrence_metrics.get('F1', 0):.4f}"
        )

        if (
            "precipitation_wet_only"
            in validation_metrics
        ):

            wet_metrics = (
                validation_metrics[
                    "precipitation_wet_only"
                ]
            )

            print(
                "  rain_amount_wet  "
                f"MAE={wet_metrics.get('MAE', 0):.4f} | "
                f"RMSE={wet_metrics.get('RMSE', 0):.4f}"
            )

        print(
            f"  Selection score: "
            f"{validation_score:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if (
            validation_score
            <
            best_validation_score
        ):

            best_validation_score = (
                validation_score
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_features":
                        8,

                    "hidden_features":
                        HIDDEN_FEATURES,

                    "output_features":
                        5,

                    "weather_features":
                        WEATHER_FEATURES,

                    "regression_targets":
                        REGRESSION_TARGETS,

                    "target_names":
                        TARGET_NAMES,

                    "city_grid_mapping":
                        city_grid_mapping,

                    "input_normalization":
                        input_normalization,

                    "target_normalization":
                        target_normalization,

                    "occurrence_threshold":
                        0.5,

                    "occurrence_pos_weight":
                        occurrence_pos_weight,

                    "time_stride":
                        TIME_STRIDE,

                    "architecture":
                        "Mask-aware 2-layer spatial GNN with separate multi-task heads",

                    "precipitation_strategy":
                        "Occurrence classification + conditional log-amount regression",

                    "target":
                        "next-hour multi-variable weather prediction",

                    "training_period":
                        "2021-2023",

                    "validation_period":
                        "2024",

                    "test_period":
                        "2025",
                },
                MODEL_PATH
            )

            print(
                "  ✓ Best V5 model saved"
            )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print()
    print(
        "Loading best V5 model..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL 2025 TEST"
    )
    print("=" * 70)

    test_metrics = evaluate_model(
        model,
        test_timestamps,
        test_lookup,
        test_targets,
        city_grid_mapping,
        input_normalization,
        target_normalization,
        edge_index
    )

    print()

    for name in TARGET_NAMES:

        metrics = test_metrics.get(
            name,
            {}
        )

        unit = ""

        if name == "temperature":
            unit = " °C"

        elif name == "humidity":
            unit = " %"

        elif name == "wind_speed":
            unit = " km/h"

        elif name == "precipitation":
            unit = " mm"

        print(
            name.upper()
        )

        print(
            f"  MAE : "
            f"{metrics.get('MAE', 0):.4f}"
            f"{unit}"
        )

        print(
            f"  RMSE: "
            f"{metrics.get('RMSE', 0):.4f}"
        )

        print(
            f"  R²  : "
            f"{metrics.get('R2', 0):.4f}"
        )

        print()

    occurrence_metrics = (
        test_metrics.get(
            "precipitation_occurrence",
            {}
        )
    )

    print(
        "PRECIPITATION OCCURRENCE"
    )

    print(
        f"  Precision: "
        f"{occurrence_metrics.get('Precision', 0):.4f}"
    )

    print(
        f"  Recall   : "
        f"{occurrence_metrics.get('Recall', 0):.4f}"
    )

    print(
        f"  F1       : "
        f"{occurrence_metrics.get('F1', 0):.4f}"
    )

    print(
        f"  Accuracy : "
        f"{occurrence_metrics.get('Accuracy', 0):.4f}"
    )

    if (
        "precipitation_wet_only"
        in test_metrics
    ):

        wet_metrics = (
            test_metrics[
                "precipitation_wet_only"
            ]
        )

        print()
        print(
            "PRECIPITATION AMOUNT "
            "(ACTUAL WET CASES)"
        )

        print(
            f"  MAE : "
            f"{wet_metrics.get('MAE', 0):.4f}"
        )

        print(
            f"  RMSE: "
            f"{wet_metrics.get('RMSE', 0):.4f}"
        )

        print(
            f"  R²  : "
            f"{wet_metrics.get('R2', 0):.4f}"
        )

    print()
    print(
        f"Test samples: "
        f"{test_metrics.get('samples', 0)}"
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    print()
    print("Model Status")
    print("------------")

    print(
        "GNN implemented: True"
    )

    print(
        "Multi-task prediction: True"
    )

    print(
        "Separate weather heads: True"
    )

    print(
        "Precipitation occurrence head: True"
    )

    print(
        "Precipitation amount head: True"
    )

    print(
        "Next-hour targets: True"
    )

    print(
        "Target normalization: True"
    )

    print(
        "Temporal split: True"
    )

    print(
        "2025 holdout evaluation: True"
    )

    print(
        "Nationwide graph topology: True"
    )

    print(
        "Nationwide ground-truth validation: False"
    )

    print(
        "True 5 km downscaling: False"
    )

    print(
        "Live weather data: False"
    )

    print()

    print(
        f"Saved model: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()

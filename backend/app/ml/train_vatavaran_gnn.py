from pathlib import Path
import math
import random

import joblib
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

DATA_PATH = BASE_DIR / "data" / "multi_city_weather_ml_2021_2025.csv"
GRAPH_PATH = BASE_DIR / "data" / "weather_graph_tensors.pt"
MODEL_PATH = BASE_DIR / "models" / "vatavaran_gnn_v2_temperature.pt"

SEED = 42

# Every 4th hour instead of every hour.
TIME_STRIDE = 4

# CPU-friendly training.
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

TARGET = "target_temperature"

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

torch.set_num_threads(max(1, min(6, torch.get_num_threads())))


# ============================================================
# GNN
# ============================================================

class MaskAwareGraphConv(nn.Module):
    """
    Mask-aware graph convolution.

    Only originally observed weather nodes are allowed
    to send weather information to neighboring nodes.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index, observation_mask):
        num_nodes = x.size(0)

        source = edge_index[0]
        target = edge_index[1]

        # Only observed source nodes can send weather information.
        source_observed = observation_mask[source] > 0.5

        valid_source = source[source_observed]
        valid_target = target[source_observed]

        messages = x[valid_source]

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

        # Self information is always retained.
        aggregated = aggregated + x
        degree = degree + 1.0

        aggregated = aggregated / degree.unsqueeze(1)

        return self.linear(aggregated)


class VatavaranGNN(nn.Module):

    def __init__(
        self,
        input_features=8,
        hidden_features=32,
        output_features=1
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

        self.prediction_head = nn.Linear(
            hidden_features,
            output_features
        )

    def forward(self, x, edge_index):

        # Last feature is the observation mask.
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

        return self.prediction_head(x)


# ============================================================
# GRID MAPPING
# ============================================================

def build_city_grid_mapping(graph):

    mapping = {}

    nodes = graph["nodes"]

    for city, coordinates in CITY_COORDINATES.items():

        latitude = coordinates[0]
        longitude = coordinates[1]

        best_node = None
        best_distance = float("inf")

        for node in nodes:

            dlat = node.latitude - latitude
            dlon = node.longitude - longitude

            distance = (
                dlat * dlat +
                dlon * dlon
            )

            if distance < best_distance:

                best_distance = distance
                best_node = node

        mapping[city] = best_node.node_id

    return mapping


def node_id_to_index(node_id):

    parts = node_id.split("_")

    row = int(parts[1])
    column = int(parts[2])

    # Current India grid has 30 longitude columns.
    return row * 30 + column


# ============================================================
# NORMALIZATION
# ============================================================

def calculate_normalization(train_df):

    statistics = {}

    for feature in WEATHER_FEATURES:

        mean = float(
            train_df[feature].mean()
        )

        std = float(
            train_df[feature].std()
        )

        if not np.isfinite(std) or std < 1e-6:
            std = 1.0

        statistics[feature] = {
            "mean": mean,
            "std": std,
        }

    return statistics


def normalize_value(
    value,
    feature,
    statistics,
):

    if value is None or not np.isfinite(value):

        return 0.0

    mean = statistics[feature]["mean"]
    std = statistics[feature]["std"]

    return (
        float(value) - mean
    ) / std


# ============================================================
# GRAPH INPUT
# ============================================================

def build_snapshot(
    timestamp,
    city_lookup,
    city_grid_mapping,
    normalization,
):

    node_count = 900

    # 7 weather features + observation mask.
    x = np.zeros(
        (node_count, 8),
        dtype=np.float32
    )

    observed_nodes = set()

    timestamp_rows = city_lookup.get(
        timestamp,
        {}
    )

    for city in CITY_NAMES:

        row = timestamp_rows.get(city)

        if row is None:
            continue

        node_id = city_grid_mapping.get(city)

        if node_id is None:
            continue

        node_index = node_id_to_index(node_id)

        observed_nodes.add(node_index)

        for feature_index, feature in enumerate(
            WEATHER_FEATURES
        ):

            value = row.get(feature)

            x[
                node_index,
                feature_index
            ] = normalize_value(
                value,
                feature,
                normalization
            )

        # Observation mask.
        x[
            node_index,
            7
        ] = 1.0

    return torch.tensor(
        x,
        dtype=torch.float32
    )


# ============================================================
# CITY DATA LOOKUP
# ============================================================

def build_city_lookup(df):

    lookup = {}

    for row in df.itertuples(index=False):

        timestamp = row.time
        city = row.location

        if timestamp not in lookup:
            lookup[timestamp] = {}

        lookup[timestamp][city] = {
            feature: getattr(row, feature)
            for feature in WEATHER_FEATURES
        }

    return lookup


# ============================================================
# TARGET LOOKUP
# ============================================================

def build_target_lookup(df):

    lookup = {}

    for row in df.itertuples(index=False):

        timestamp = row.time
        city = row.location

        if timestamp not in lookup:
            lookup[timestamp] = {}

        lookup[timestamp][city] = float(
            row.target_temperature
        )

    return lookup


# ============================================================
# OBSERVED NODE TARGETS
# ============================================================

def build_target_tensor(
    timestamp,
    target_lookup,
    city_grid_mapping,
):

    target = torch.zeros(
        900,
        dtype=torch.float32
    )

    target_mask = torch.zeros(
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

        node_id = city_grid_mapping.get(city)

        if node_id is None:
            continue

        node_index = node_id_to_index(node_id)

        target[node_index] = rows[city]

        target_mask[node_index] = True

    return target, target_mask


# ============================================================
# TIMESTAMP SAMPLING
# ============================================================

def sample_timestamps(
    timestamps,
    count,
):

    timestamps = list(timestamps)

    if len(timestamps) <= count:
        return timestamps

    return random.sample(
        timestamps,
        count
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    timestamps,
    city_lookup,
    target_lookup,
    city_grid_mapping,
    normalization,
    edge_index,
):

    model.eval()

    predictions = []
    actuals = []

    with torch.no_grad():

        for timestamp in timestamps:

            x = build_snapshot(
                timestamp,
                city_lookup,
                city_grid_mapping,
                normalization,
            ).to(DEVICE)

            output = model(
                x,
                edge_index
            ).squeeze(1)

            target, mask = build_target_tensor(
                timestamp,
                target_lookup,
                city_grid_mapping,
            )

            if mask.sum() == 0:
                continue

            predictions.extend(
                output[mask].cpu().numpy()
            )

            actuals.extend(
                target[mask].cpu().numpy()
            )

    predictions = np.array(
        predictions
    )

    actuals = np.array(
        actuals
    )

    if len(actuals) == 0:
        return {}

    errors = predictions - actuals

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
        (actuals - actuals.mean()) ** 2
    )

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot > 0
        else 0.0
    )

    return {
        "samples": int(len(actuals)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "VATAVARAN Optimized GNN Temperature Training"
    )
    print(
        "--------------------------------------------"
    )

    print(
        f"Dataset: {DATA_PATH}"
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------------

    train_df = df[
        df["time"].dt.year <= 2024
    ].copy()

    test_df = df[
        df["time"].dt.year == 2025
    ].copy()

    print()
    print("Dataset Split")
    print("-------------")

    print(
        f"Training rows: {len(train_df)}"
    )

    print(
        f"Testing rows: {len(test_df)}"
    )

    print(
        f"Training period: "
        f"{train_df['time'].min()} → "
        f"{train_df['time'].max()}"
    )

    print(
        f"Testing period: "
        f"{test_df['time'].min()} → "
        f"{test_df['time'].max()}"
    )

    # --------------------------------------------------------
    # BUILD GRAPH
    # --------------------------------------------------------

    print()
    print("Building nationwide graph...")

    graph = build_graph_structure(
        grid_size=1.0
    )

    city_grid_mapping = build_city_grid_mapping(
        graph
    )

    print(
        f"Grid nodes: {len(graph['nodes'])}"
    )

    print(
        f"Graph edges: {len(graph['edges'])}"
    )

    print()
    print("City → Grid mapping")

    for city, node_id in city_grid_mapping.items():

        print(
            f"{city:12s} → {node_id}"
        )

    # --------------------------------------------------------
    # EDGE INDEX
    # --------------------------------------------------------

    edge_list = []

    for edge in graph["edges"]:

        source = node_id_to_index(
            edge[0]
        )

        target = node_id_to_index(
            edge[1]
        )

        edge_list.append(
            [source, target]
        )

        edge_list.append(
            [target, source]
        )

    edge_index = torch.tensor(
        edge_list,
        dtype=torch.long,
        device=DEVICE
    ).t().contiguous()

    print()
    print(
        f"Directed edges: "
        f"{edge_index.shape[1]}"
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    print()
    print("Calculating normalization statistics...")

    normalization = calculate_normalization(
        train_df
    )

    # --------------------------------------------------------
    # LOOKUPS
    # --------------------------------------------------------

    print(
        "Building weather lookup..."
    )

    train_lookup = build_city_lookup(
        train_df
    )

    test_lookup = build_city_lookup(
        test_df
    )

    train_targets = build_target_lookup(
        train_df
    )

    test_targets = build_target_lookup(
        test_df
    )

    # --------------------------------------------------------
    # TIMESTAMPS
    # --------------------------------------------------------

    all_train_timestamps = sorted(
        train_lookup.keys()
    )

    all_test_timestamps = sorted(
        test_lookup.keys()
    )

    # Every 4th hour.
    sampled_train_timestamps = (
        all_train_timestamps[
            ::TIME_STRIDE
        ]
    )

    # Validation = final part of 2024.
    validation_timestamps = [
        t for t in sampled_train_timestamps
        if t.year == 2024
    ]

    training_timestamps = [
        t for t in sampled_train_timestamps
        if t.year <= 2023
    ]

    # Keep enough samples but avoid huge CPU workload.
    training_timestamps = sample_timestamps(
        training_timestamps,
        min(
            12000,
            len(training_timestamps)
        )
    )

    validation_timestamps = sample_timestamps(
        validation_timestamps,
        min(
            1500,
            len(validation_timestamps)
        )
    )

    test_timestamps = all_test_timestamps[
        ::TIME_STRIDE
    ]

    print()
    print("Training Configuration")
    print("----------------------")

    print(
        f"Time stride: every {TIME_STRIDE} hours"
    )

    print(
        f"Training timestamps: "
        f"{len(training_timestamps)}"
    )

    print(
        f"Validation timestamps: "
        f"{len(validation_timestamps)}"
    )

    print(
        f"2025 test timestamps: "
        f"{len(test_timestamps)}"
    )

    print(
        f"Samples per epoch: "
        f"{min(SAMPLES_PER_EPOCH, len(training_timestamps))}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = VatavaranGNN(
        input_features=8,
        hidden_features=HIDDEN_FEATURES,
        output_features=1,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_function = nn.MSELoss()

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print()
    print("Starting training...")
    print()

    best_validation_mae = float(
        "inf"
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        epoch_timestamps = sample_timestamps(
            training_timestamps,
            min(
                SAMPLES_PER_EPOCH,
                len(training_timestamps)
            )
        )

        total_loss = 0.0

        for timestamp in epoch_timestamps:

            x = build_snapshot(
                timestamp,
                train_lookup,
                city_grid_mapping,
                normalization,
            ).to(DEVICE)

            target, mask = build_target_tensor(
                timestamp,
                train_targets,
                city_grid_mapping,
            )

            target = target.to(DEVICE)
            mask = mask.to(DEVICE)

            if mask.sum() == 0:
                continue

            optimizer.zero_grad()

            output = model(
                x,
                edge_index
            ).squeeze(1)

            loss = loss_function(
                output[mask],
                target[mask]
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                float(loss.item())
            )

        average_loss = (
            total_loss /
            max(1, len(epoch_timestamps))
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        validation_metrics = evaluate_model(
            model,
            validation_timestamps,
            train_lookup,
            train_targets,
            city_grid_mapping,
            normalization,
            edge_index,
        )

        validation_mae = validation_metrics.get(
            "MAE",
            float("inf")
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {average_loss:.4f} | "
            f"Val MAE: {validation_mae:.4f} | "
            f"Val RMSE: "
            f"{validation_metrics.get('RMSE', 0):.4f} | "
            f"Val R2: "
            f"{validation_metrics.get('R2', 0):.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if validation_mae < best_validation_mae:

            best_validation_mae = validation_mae

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "input_features": 8,

                    "hidden_features":
                        HIDDEN_FEATURES,

                    "output_features": 1,

                    "weather_features":
                        WEATHER_FEATURES,

                    "city_grid_mapping":
                        city_grid_mapping,

                    "normalization":
                        normalization,

                    "time_stride":
                        TIME_STRIDE,

                    "architecture":
                        "Mask-aware 2-layer spatial GNN",

                    "target":
                        "next-hour temperature",

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
                "  ✓ Best model saved"
            )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print()
    print("Loading best model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # FINAL 2025 TEST
    # --------------------------------------------------------

    print()
    print("Final 2025 Test Evaluation")
    print("--------------------------")

    test_metrics = evaluate_model(
        model,
        test_timestamps,
        test_lookup,
        test_targets,
        city_grid_mapping,
        normalization,
        edge_index,
    )

    print(
        f"Test samples: "
        f"{test_metrics.get('samples', 0)}"
    )

    print(
        f"MAE: "
        f"{test_metrics.get('MAE', 0):.4f} °C"
    )

    print(
        f"RMSE: "
        f"{test_metrics.get('RMSE', 0):.4f} °C"
    )

    print(
        f"R²: "
        f"{test_metrics.get('R2', 0):.4f}"
    )

    print()
    print("Model Status")
    print("------------")

    print("GNN implemented: True")
    print("GNN trained: True")
    print("Temporal split: True")
    print("2025 holdout evaluation: True")
    print("Nationwide graph topology: True")
    print("Nationwide ground-truth validation: False")
    print("True 5 km downscaling: False")
    print("Live weather data: False")

    print()
    print(
        f"Saved model: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()

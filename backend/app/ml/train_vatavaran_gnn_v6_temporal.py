from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from app.ml.nationwide_spatial_grid import build_graph_structure
from app.ml.weather_graph_builder import CITY_COORDINATES

# Reuse only stable V5 data/graph utilities.
# V5 itself is NOT modified.
from app.ml.train_vatavaran_gnn_v5_multitask import (
    WEATHER_FEATURES,
    REGRESSION_TARGETS,
    TARGET_NAMES,
    CITY_NAMES,
    DEVICE,
    MaskAwareGraphConv,
    build_city_grid_mapping,
    node_id_to_index,
    calculate_input_normalization,
    calculate_target_normalization,
    normalize_value,
    denormalize_value,
    create_targets,
    build_snapshot,
    build_city_lookup,
    build_target_lookup,
    build_target_tensor,
    regression_metrics,
    binary_metrics,
)


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
    / "vatavaran_gnn_v6_temporal.pt"
)

SEED = 42

# Same temporal sampling as V5.
TIME_STRIDE = 4

# Six sampled snapshots:
# t-5h, t-4h, t-3h, t-2h, t-1h, t
SEQUENCE_LENGTH = 6

EPOCHS = 8
SAMPLES_PER_EPOCH = 3000

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

HIDDEN_FEATURES = 32
GRU_HIDDEN_FEATURES = 32

NODE_COUNT = 900

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(
    max(
        1,
        min(
            6,
            torch.get_num_threads()
        )
    )
)


# ============================================================
# TEMPORAL-SPATIAL MODEL
# ============================================================

class VatavaranGNNV6(nn.Module):

    def __init__(
        self,
        input_features=8,
        hidden_features=32,
        gru_hidden_features=32,
    ):

        super().__init__()

        # ----------------------------------------------------
        # Spatial encoder
        # ----------------------------------------------------

        self.conv1 = MaskAwareGraphConv(
            input_features,
            hidden_features
        )

        self.conv2 = MaskAwareGraphConv(
            hidden_features,
            hidden_features
        )

        # ----------------------------------------------------
        # Temporal encoder
        #
        # One GRU is applied independently to the temporal
        # sequence of every graph node.
        # ----------------------------------------------------

        self.temporal_gru = nn.GRU(
            input_size=hidden_features,
            hidden_size=gru_hidden_features,
            num_layers=1,
            batch_first=True
        )

        # ----------------------------------------------------
        # Prediction heads
        # ----------------------------------------------------

        self.temperature_head = nn.Linear(
            gru_hidden_features,
            1
        )

        self.humidity_head = nn.Linear(
            gru_hidden_features,
            1
        )

        self.wind_head = nn.Linear(
            gru_hidden_features,
            1
        )

        self.precipitation_occurrence_head = nn.Linear(
            gru_hidden_features,
            1
        )

        self.precipitation_amount_head = nn.Linear(
            gru_hidden_features,
            1
        )

    def encode_spatial_snapshot(
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

        return x

    def forward(
        self,
        sequence,
        edge_index
    ):
        """
        sequence shape:

            [sequence_length, node_count, input_features]

        Example:

            [6, 900, 8]

        Spatial GNN is applied to every timestamp.

        Then:

            [6, 900, 32]

        is rearranged into:

            [900, 6, 32]

        so every graph node gets its own temporal sequence.
        """

        temporal_embeddings = []

        for timestep in range(
            sequence.size(0)
        ):

            snapshot = sequence[
                timestep
            ]

            embedding = (
                self.encode_spatial_snapshot(
                    snapshot,
                    edge_index
                )
            )

            temporal_embeddings.append(
                embedding
            )

        spatial_sequence = torch.stack(
            temporal_embeddings,
            dim=0
        )

        # [time, nodes, hidden]
        # ->
        # [nodes, time, hidden]
        node_sequences = (
            spatial_sequence
            .permute(
                1,
                0,
                2
            )
            .contiguous()
        )

        gru_output, _ = (
            self.temporal_gru(
                node_sequences
            )
        )

        # Last temporal state.
        final_state = gru_output[
            :,
            -1,
            :
        ]

        return {
            "temperature":
                self.temperature_head(
                    final_state
                ),

            "humidity":
                self.humidity_head(
                    final_state
                ),

            "wind_speed":
                self.wind_head(
                    final_state
                ),

            "precipitation_occurrence":
                self.precipitation_occurrence_head(
                    final_state
                ),

            "precipitation_amount":
                self.precipitation_amount_head(
                    final_state
                ),
        }


# ============================================================
# TEMPORAL SEQUENCE HELPERS
# ============================================================

def build_sampled_timestamps(
    lookup,
    time_stride
):

    timestamps = sorted(
        lookup.keys()
    )

    return timestamps[
        ::time_stride
    ]


def build_temporal_sequences(
    timestamps,
    sequence_length
):
    """
    Creates valid ending timestamps.

    Example:

        t0
        t1
        t2
        t3
        t4
        t5

    becomes one sequence ending at t5.

    The function works only on the sampled timestamp list,
    therefore V5's TIME_STRIDE behaviour is preserved.
    """

    timestamps = list(
        timestamps
    )

    if len(timestamps) < sequence_length:
        return []

    sequences = []

    for index in range(
        sequence_length - 1,
        len(timestamps)
    ):

        sequence = timestamps[
            index - sequence_length + 1:
            index + 1
        ]

        # Make sure timestamps are actually evenly spaced.
        if len(sequence) > 1:

            deltas = [
                sequence[i + 1]
                - sequence[i]
                for i in range(
                    len(sequence) - 1
                )
            ]

            first_delta = deltas[0]

            if not all(
                delta == first_delta
                for delta in deltas
            ):
                continue

        sequences.append(
            sequence
        )

    return sequences


def build_sequence_tensor(
    timestamp_sequence,
    city_lookup,
    city_grid_mapping,
    input_normalization
):

    snapshots = []

    for timestamp in timestamp_sequence:

        snapshot = build_snapshot(
            timestamp,
            city_lookup,
            city_grid_mapping,
            input_normalization
        )

        snapshots.append(
            snapshot
        )

    return torch.stack(
        snapshots,
        dim=0
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    temporal_sequences,
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

        for timestamp_sequence in temporal_sequences:

            target_timestamp = (
                timestamp_sequence[-1]
            )

            sequence_tensor = (
                build_sequence_tensor(
                    timestamp_sequence,
                    city_lookup,
                    city_grid_mapping,
                    input_normalization
                )
                .to(DEVICE)
            )

            output = model(
                sequence_tensor,
                edge_index
            )

            target_rows = target_lookup.get(
                target_timestamp,
                {}
            )

            timestamp_rows = city_lookup.get(
                target_timestamp,
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
                            output[
                                "temperature"
                            ][
                                node_index
                            ].item()
                        ),

                        float(
                            output[
                                "humidity"
                            ][
                                node_index
                            ].item()
                        ),

                        float(
                            output[
                                "wind_speed"
                            ][
                                node_index
                            ].item()
                        ),

                        float(
                            output[
                                "precipitation_amount"
                            ][
                                node_index
                            ].item()
                        ),
                    ]
                )

                actual_values = [
                    float(
                        raw_target_values[0]
                    ),

                    float(
                        raw_target_values[1]
                    ),

                    float(
                        raw_target_values[2]
                    ),

                    max(
                        0.0,
                        float(
                            np.expm1(
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
                            np.clip(
                                predicted_value,
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

                    wet_predictions.append(
                        predictions[
                            "precipitation"
                        ][-1]
                    )

                    wet_actuals.append(
                        actuals[
                            "precipitation"
                        ][-1]
                    )

    metrics = {}

    for name in TARGET_NAMES:

        metrics[name] = regression_metrics(
            predictions[name],
            actuals[name]
        )

    occurrence_results = binary_metrics(
        occurrence_probabilities,
        occurrence_actuals,
        threshold=occurrence_threshold
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

    precipitation_std = max(
        1e-6,
        target_normalization[
            "target_precipitation_log"
        ]["std"]
    )

    precipitation_mae_normalized = (
        metrics[
            "precipitation"
        ]["MAE"]
        /
        precipitation_std
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
# PRINT METRICS
# ============================================================

def print_metrics(
    metrics,
    title
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    for name in TARGET_NAMES:

        result = metrics.get(
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

        print()
        print(
            name.upper()
        )

        print(
            f"  MAE : "
            f"{result.get('MAE', 0):.4f}"
            f"{unit}"
        )

        print(
            f"  RMSE: "
            f"{result.get('RMSE', 0):.4f}"
        )

        print(
            f"  R²  : "
            f"{result.get('R2', 0):.4f}"
        )

    occurrence = metrics.get(
        "precipitation_occurrence",
        {}
    )

    print()
    print(
        "PRECIPITATION OCCURRENCE"
    )

    print(
        f"  Precision: "
        f"{occurrence.get('Precision', 0):.4f}"
    )

    print(
        f"  Recall   : "
        f"{occurrence.get('Recall', 0):.4f}"
    )

    print(
        f"  F1       : "
        f"{occurrence.get('F1', 0):.4f}"
    )

    print(
        f"  Accuracy : "
        f"{occurrence.get('Accuracy', 0):.4f}"
    )

    if "precipitation_wet_only" in metrics:

        wet = metrics[
            "precipitation_wet_only"
        ]

        print()
        print(
            "PRECIPITATION AMOUNT "
            "(ACTUAL WET CASES)"
        )

        print(
            f"  MAE : "
            f"{wet.get('MAE', 0):.4f}"
        )

        print(
            f"  RMSE: "
            f"{wet.get('RMSE', 0):.4f}"
        )

        print(
            f"  R²  : "
            f"{wet.get('R2', 0):.4f}"
        )

    print()
    print(
        f"Samples: "
        f"{metrics.get('samples', 0)}"
    )

    print(
        f"Selection score: "
        f"{metrics.get('selection_score', float('inf')):.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "VATAVARAN GNN V6 - TEMPORAL-SPATIAL MODEL"
    )
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
        np.sum(
            rain_values == 1
        )
    )

    negative_count = float(
        np.sum(
            rain_values == 0
        )
    )

    occurrence_pos_weight = (
        negative_count
        /
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
    # SAMPLED TIMESTAMPS
    # --------------------------------------------------------

    training_timestamps = (
        build_sampled_timestamps(
            training_lookup,
            TIME_STRIDE
        )
    )

    validation_timestamps = (
        build_sampled_timestamps(
            validation_lookup,
            TIME_STRIDE
        )
    )

    test_timestamps = (
        build_sampled_timestamps(
            test_lookup,
            TIME_STRIDE
        )
    )

    # --------------------------------------------------------
    # TEMPORAL SEQUENCES
    # --------------------------------------------------------

    training_sequences = (
        build_temporal_sequences(
            training_timestamps,
            SEQUENCE_LENGTH
        )
    )

    validation_sequences = (
        build_temporal_sequences(
            validation_timestamps,
            SEQUENCE_LENGTH
        )
    )

    test_sequences = (
        build_temporal_sequences(
            test_timestamps,
            SEQUENCE_LENGTH
        )
    )

    print()
    print(
        f"Training sampled timestamps: "
        f"{len(training_timestamps)}"
    )

    print(
        f"Training sequences: "
        f"{len(training_sequences)}"
    )

    print(
        f"Validation sampled timestamps: "
        f"{len(validation_timestamps)}"
    )

    print(
        f"Validation sequences: "
        f"{len(validation_sequences)}"
    )

    print(
        f"Test sampled timestamps: "
        f"{len(test_timestamps)}"
    )

    print(
        f"Test sequences: "
        f"{len(test_sequences)}"
    )

    if not training_sequences:

        raise RuntimeError(
            "No training temporal sequences were created."
        )

    # --------------------------------------------------------
    # SEQUENCE SHAPE SMOKE TEST
    # --------------------------------------------------------

    print()
    print(
        "Running sequence shape check..."
    )

    example_sequence = (
        build_sequence_tensor(
            training_sequences[0],
            training_lookup,
            city_grid_mapping,
            input_normalization
        )
    )

    print(
        f"Example sequence shape: "
        f"{tuple(example_sequence.shape)}"
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        NODE_COUNT,
        8
    )

    if tuple(
        example_sequence.shape
    ) != expected_shape:

        raise RuntimeError(
            "Unexpected sequence shape. "
            f"Expected {expected_shape}, "
            f"got {tuple(example_sequence.shape)}"
        )

    print(
        "✓ Sequence shape is correct."
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = VatavaranGNNV6(
        input_features=8,
        hidden_features=HIDDEN_FEATURES,
        gru_hidden_features=GRU_HIDDEN_FEATURES,
    ).to(DEVICE)

    # --------------------------------------------------------
    # FORWARD-PASS SMOKE TEST
    # --------------------------------------------------------

    print()
    print(
        "Running V6 forward-pass check..."
    )

    with torch.no_grad():

        smoke_output = model(
            example_sequence.to(DEVICE),
            edge_index
        )

    for name in [
        "temperature",
        "humidity",
        "wind_speed",
        "precipitation_occurrence",
        "precipitation_amount",
    ]:

        shape = tuple(
            smoke_output[name].shape
        )

        print(
            f"  {name:<28} {shape}"
        )

        if shape != (
            NODE_COUNT,
            1
        ):

            raise RuntimeError(
                f"Unexpected output shape for "
                f"{name}: {shape}"
            )

    print(
        "✓ Forward pass is valid."
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

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
        "Starting V6 training..."
    )

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        sampled_sequences = (
            training_sequences
        )

        if len(
            sampled_sequences
        ) > SAMPLES_PER_EPOCH:

            sampled_sequences = random.sample(
                sampled_sequences,
                SAMPLES_PER_EPOCH
            )

        total_loss = 0.0
        valid_batches = 0

        for timestamp_sequence in (
            sampled_sequences
        ):

            sequence_tensor = (
                build_sequence_tensor(
                    timestamp_sequence,
                    training_lookup,
                    city_grid_mapping,
                    input_normalization
                ).to(DEVICE)
            )

            target_timestamp = (
                timestamp_sequence[-1]
            )

            (
                target,
                occurrence_target,
                target_mask,
                wet_mask
            ) = build_target_tensor(
                target_timestamp,
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
                sequence_tensor,
                edge_index
            )

            temperature_loss = (
                regression_loss(
                    output[
                        "temperature"
                    ][
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
                    output[
                        "humidity"
                    ][
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
                    output[
                        "wind_speed"
                    ][
                        target_mask,
                        0
                    ],
                    target[
                        target_mask,
                        2
                    ]
                )
            )

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

            if wet_mask.sum() > 0:

                amount_loss = (
                    regression_loss(
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

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += float(
                loss.item()
            )

            valid_batches += 1

        average_loss = (
            total_loss
            /
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
                validation_sequences,
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
                f"MAE={metrics.get('MAE', 0):.4f}"
                f"{unit} | "
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
            f"Precision="
            f"{occurrence_metrics.get('Precision', 0):.4f} | "
            f"Recall="
            f"{occurrence_metrics.get('Recall', 0):.4f} | "
            f"F1="
            f"{occurrence_metrics.get('F1', 0):.4f}"
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
                f"MAE="
                f"{wet_metrics.get('MAE', 0):.4f} | "
                f"RMSE="
                f"{wet_metrics.get('RMSE', 0):.4f}"
            )

        print(
            f"  Selection score: "
            f"{validation_score:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST V6
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

                    "gru_hidden_features":
                        GRU_HIDDEN_FEATURES,

                    "sequence_length":
                        SEQUENCE_LENGTH,

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
                        "2-layer spatial GNN + temporal GRU with separate multi-task heads",

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

                    "experiment":
                        "V6 temporal-spatial GNN",

                },
                MODEL_PATH
            )

            print(
                "  ✓ Best V6 model saved"
            )

    # --------------------------------------------------------
    # LOAD BEST MODEL
    # --------------------------------------------------------

    print()
    print(
        "Loading best V6 model..."
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

    test_metrics = evaluate_model(
        model,
        test_sequences,
        test_lookup,
        test_targets,
        city_grid_mapping,
        input_normalization,
        target_normalization,
        edge_index
    )

    print_metrics(
        test_metrics,
        "FINAL 2025 V6 TEST"
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    print()
    print(
        "=" * 70
    )

    print(
        "V6 MODEL STATUS"
    )

    print(
        "=" * 70
    )

    print(
        "Spatial GNN: True"
    )

    print(
        "Temporal GRU: True"
    )

    print(
        f"Sequence length: {SEQUENCE_LENGTH}"
    )

    print(
        f"Time stride: {TIME_STRIDE}"
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
        "Temporal split: True"
    )

    print(
        "2025 holdout evaluation: True"
    )

    print(
        "Nationwide graph topology: True"
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
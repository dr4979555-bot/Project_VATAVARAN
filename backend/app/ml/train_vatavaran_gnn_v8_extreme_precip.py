from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
)

from app.ml.nationwide_spatial_grid import build_graph_structure

from app.ml.train_vatavaran_gnn_v5_multitask import (
    WEATHER_FEATURES,
    CITY_NAMES,
    MaskAwareGraphConv,
    build_city_lookup,
    build_city_grid_mapping,
    node_id_to_index,
    calculate_input_normalization,
)

from app.ml.train_vatavaran_gnn_v6_temporal import (
    TIME_STRIDE,
    SEQUENCE_LENGTH,
    build_temporal_sequences,
    build_sequence_tensor,
)


# ============================================================
# VATAVARAN V8 - EXTREME-AWARE MULTI-HORIZON TEMPORAL GNN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "multi_city_weather_ml_2021_2025.csv"
)

MODEL_OUTPUT = (
    BASE_DIR
    / "models"
    / "vatavaran_gnn_v8_extreme_precip.pt"
)

METRICS_OUTPUT = (
    BASE_DIR
    / "data"
    / "v8_extreme_precip_metrics.json"
)

TEST_METRICS_CSV = (
    BASE_DIR
    / "data"
    / "v8_extreme_precip_2025_metrics.csv"
)

GRID_OUTPUT = (
    BASE_DIR
    / "data"
    / "v8_extreme_precip_grid_forecasts_2025.npz"
)


DEVICE = torch.device("cpu")

# Direct multi-horizon targets.
HORIZONS = [1, 6, 12, 24, 48]

INPUT_FEATURES = 8
HIDDEN_FEATURES = 32
GRU_HIDDEN_FEATURES = 32

EPOCHS = 6
SAMPLES_PER_EPOCH = 2500

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

OCCURRENCE_THRESHOLD = 0.5

# Training-only 2021-2023 wet-rain quantiles.
PRECIP_Q90_MM = 2.3
PRECIP_Q95_MM = 3.8
PRECIP_Q99_MM = 7.9

# Extreme-aware precipitation amount loss weights.
PRECIP_DRY_WEIGHT = 0.15
PRECIP_Q90_WEIGHT = 2.0
PRECIP_Q95_WEIGHT = 4.0
PRECIP_Q99_WEIGHT = 8.0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# MODEL
# ============================================================

class VatavaranGNNV7(nn.Module):
    """
    Same V6 spatial GNN + GRU backbone, but with direct
    multi-horizon output heads.

    One input sequence produces:
        +1h
        +6h
        +12h
        +24h
        +48h

    for:
        temperature
        humidity
        wind
        precipitation occurrence
        precipitation amount
    """

    def __init__(
        self,
        input_features=8,
        hidden_features=32,
        gru_hidden_features=32,
        horizons=None,
    ):
        super().__init__()

        if horizons is None:
            horizons = HORIZONS

        self.horizons = list(horizons)

        self.conv1 = MaskAwareGraphConv(
            input_features,
            hidden_features,
        )

        self.conv2 = MaskAwareGraphConv(
            hidden_features,
            hidden_features,
        )

        self.temporal_gru = nn.GRU(
            input_size=hidden_features,
            hidden_size=gru_hidden_features,
            num_layers=1,
            batch_first=True,
        )

        self.horizon_heads = nn.ModuleDict()

        for horizon in self.horizons:
            self.horizon_heads[
                f"h{horizon}"
            ] = nn.ModuleDict(
                {
                    "temperature": nn.Linear(
                        gru_hidden_features,
                        1,
                    ),
                    "humidity": nn.Linear(
                        gru_hidden_features,
                        1,
                    ),
                    "wind_speed": nn.Linear(
                        gru_hidden_features,
                        1,
                    ),
                    "precipitation_occurrence": nn.Linear(
                        gru_hidden_features,
                        1,
                    ),
                    "precipitation_amount": nn.Linear(
                        gru_hidden_features,
                        1,
                    ),
                }
            )

    def encode_spatial_snapshot(
        self,
        x,
        edge_index,
    ):
        observation_mask = x[:, -1]

        x = self.conv1(
            x,
            edge_index,
            observation_mask,
        )

        x = torch.relu(x)

        # Expand the observation mask by one graph hop.
        # Nodes receiving information in Conv1 can participate
        # as message sources in Conv2.
        propagated_mask = observation_mask.clone()

        source = edge_index[0]
        target = edge_index[1]

        source_observed = (
            observation_mask[source] > 0.5
        )

        propagated_targets = target[
            source_observed
        ]

        if propagated_targets.numel() > 0:
            propagated_mask[
                propagated_targets
            ] = 1.0

        x = self.conv2(
            x,
            edge_index,
            propagated_mask,
        )

        x = torch.relu(x)

        return x

    def forward(
        self,
        sequence,
        edge_index,
    ):
        temporal_embeddings = []

        for timestep in range(
            sequence.size(0)
        ):
            embedding = (
                self.encode_spatial_snapshot(
                    sequence[timestep],
                    edge_index,
                )
            )

            temporal_embeddings.append(
                embedding
            )

        spatial_sequence = torch.stack(
            temporal_embeddings,
            dim=0,
        )

        node_sequences = (
            spatial_sequence
            .permute(
                1,
                0,
                2,
            )
            .contiguous()
        )

        gru_output, _ = (
            self.temporal_gru(
                node_sequences
            )
        )

        final_state = gru_output[
            :,
            -1,
            :,
        ]

        outputs = {}

        for horizon in self.horizons:

            heads = self.horizon_heads[
                f"h{horizon}"
            ]

            outputs[f"h{horizon}"] = {
                "temperature": heads[
                    "temperature"
                ](final_state),

                "humidity": heads[
                    "humidity"
                ](final_state),

                "wind_speed": heads[
                    "wind_speed"
                ](final_state),

                "precipitation_occurrence": heads[
                    "precipitation_occurrence"
                ](final_state),

                "precipitation_amount": heads[
                    "precipitation_amount"
                ](final_state),
            }

        return outputs


# ============================================================
# GRAPH
# ============================================================

def build_edge_index(graph):

    sources = []
    targets = []

    for source, target in graph[
        "edges"
    ]:

        source_index = node_id_to_index(
            source
        )

        target_index = node_id_to_index(
            target
        )

        sources.append(
            source_index
        )
        targets.append(
            target_index
        )

        sources.append(
            target_index
        )
        targets.append(
            source_index
        )

    return torch.tensor(
        [
            sources,
            targets,
        ],
        dtype=torch.long,
        device=DEVICE,
    )


# ============================================================
# TARGET NORMALIZATION
# ============================================================

def build_target_normalization(
    train_df,
):
    """
    Training-only normalization.

    Same raw weather variables are used for all horizons.
    The horizon-specific structure is preserved in the checkpoint
    so future versions can use separate distributions if needed.
    """

    variables = {
        "temperature": "temperature_2m",
        "humidity": "relative_humidity_2m",
        "wind_speed": "wind_speed_10m",
    }

    stats = {}

    for horizon in HORIZONS:

        horizon_stats = {}

        for name, column in variables.items():

            values = pd.to_numeric(
                train_df[column],
                errors="coerce",
            ).dropna()

            mean = float(
                values.mean()
                if not values.empty
                else 0.0
            )

            std = float(
                values.std()
                if not values.empty
                else 1.0
            )

            if (
                not np.isfinite(std)
                or std < 1e-6
            ):
                std = 1.0

            horizon_stats[name] = {
                "mean": mean,
                "std": std,
            }

        precip_values = pd.to_numeric(
            train_df[
                "precipitation"
            ],
            errors="coerce",
        ).fillna(0.0)

        log_precip = np.log1p(
            np.maximum(
                precip_values,
                0.0,
            )
        )

        precip_mean = float(
            log_precip.mean()
        )

        precip_std = float(
            log_precip.std()
        )

        if (
            not np.isfinite(
                precip_std
            )
            or precip_std < 1e-6
        ):
            precip_std = 1.0

        horizon_stats[
            "precipitation_log"
        ] = {
            "mean": precip_mean,
            "std": precip_std,
        }

        stats[str(horizon)] = (
            horizon_stats
        )

    return stats


def normalize_value(
    value,
    stats,
):
    return (
        float(value)
        - float(stats["mean"])
    ) / float(stats["std"])


def denormalize_value(
    value,
    stats,
):
    return (
        float(value)
        * float(stats["std"])
        + float(stats["mean"])
    )


# ============================================================
# OCCURRENCE POSITIVE WEIGHTS
# ============================================================

def build_occurrence_weights(
    train_df,
):
    values = pd.to_numeric(
        train_df[
            "precipitation"
        ],
        errors="coerce",
    ).fillna(0.0)

    positive = int(
        (values > 0).sum()
    )

    negative = int(
        (values <= 0).sum()
    )

    if positive <= 0:
        weight = 1.0
    else:
        weight = (
            negative
            / positive
        )

    weight = float(
        np.clip(
            weight,
            1.0,
            20.0,
        )
    )

    return {
        str(horizon): weight
        for horizon in HORIZONS
    }


# ============================================================
# SEQUENCE PREPARATION
# ============================================================

def prepare_sequences(
    df,
    split_end_year,
):
    df = df.sort_values(
        [
            "location",
            "time",
        ]
    ).copy()

    city_lookup = build_city_lookup(
        df
    )

    lookup = (
        df.set_index(
            [
                "time",
                "location",
            ]
        )
        .to_dict(
            "index"
        )
    )

    times = sorted(
        df[
            "time"
        ]
        .drop_duplicates()
        .tolist()
    )

    sampled_times = times[
        ::TIME_STRIDE
    ]

    sequences = build_temporal_sequences(
        sampled_times,
        SEQUENCE_LENGTH,
    )

    split_end = df[
        "time"
    ].max()

    valid_sequences = []

    max_horizon = max(
        HORIZONS
    )

    required_cities = list(
        CITY_NAMES
    )

    for sequence in sequences:

        final_time = pd.Timestamp(
            sequence[-1]
        )

        latest_target = (
            final_time
            + pd.Timedelta(
                hours=max_horizon
            )
        )

        if latest_target > split_end:
            continue

        valid = True

        for horizon in HORIZONS:

            target_time = (
                final_time
                + pd.Timedelta(
                    hours=horizon
                )
            )

            for city in required_cities:

                if (
                    target_time,
                    city,
                ) not in lookup:
                    valid = False
                    break

            if not valid:
                break

        if valid:
            valid_sequences.append(
                sequence
            )

    return (
        valid_sequences,
        city_lookup,
        lookup,
    )


# ============================================================
# TARGET EXTRACTION
# ============================================================

def build_targets_for_sequence(
    sequence,
    lookup,
    target_stats,
):
    final_time = pd.Timestamp(
        sequence[-1]
    )

    targets = {}

    for horizon in HORIZONS:

        target_time = (
            final_time
            + pd.Timedelta(
                hours=horizon
            )
        )

        horizon_targets = {
            "temperature": [],
            "humidity": [],
            "wind_speed": [],
            "precipitation": [],
            "precipitation_occurrence": [],
        }

        for city in CITY_NAMES:

            row = lookup[
                (
                    target_time,
                    city,
                )
            ]

            temperature = float(
                row[
                    "temperature_2m"
                ]
            )

            humidity = float(
                row[
                    "relative_humidity_2m"
                ]
            )

            wind = float(
                row[
                    "wind_speed_10m"
                ]
            )

            precipitation = max(
                0.0,
                float(
                    row[
                        "precipitation"
                    ]
                ),
            )

            precip_log = math.log1p(
                precipitation
            )

            stats = target_stats[
                str(horizon)
            ]

            horizon_targets[
                "temperature"
            ].append(
                normalize_value(
                    temperature,
                    stats[
                        "temperature"
                    ],
                )
            )

            horizon_targets[
                "humidity"
            ].append(
                normalize_value(
                    humidity,
                    stats[
                        "humidity"
                    ],
                )
            )

            horizon_targets[
                "wind_speed"
            ].append(
                normalize_value(
                    wind,
                    stats[
                        "wind_speed"
                    ],
                )
            )

            horizon_targets[
                "precipitation"
            ].append(
                normalize_value(
                    precip_log,
                    stats[
                        "precipitation_log"
                    ],
                )
            )

            horizon_targets[
                "precipitation_occurrence"
            ].append(
                1.0
                if precipitation > 0
                else 0.0
            )

        targets[
            str(horizon)
        ] = {
            key: torch.tensor(
                value,
                dtype=torch.float32,
                device=DEVICE,
            ).unsqueeze(1)
            for key, value in (
                horizon_targets.items()
            )
        }

    return targets


# ============================================================
# LOSS
# ============================================================

def compute_loss(
    outputs,
    targets,
    city_indices,
    occurrence_weights,
    target_stats,
):
    total_loss = torch.tensor(
        0.0,
        device=DEVICE,
    )

    component_losses = []

    for horizon in HORIZONS:

        key = f"h{horizon}"

        prediction = outputs[
            key
        ]

        target = targets[
            str(horizon)
        ]

        city = prediction[
            "temperature"
        ][
            city_indices
        ]

        target_temp = target[
            "temperature"
        ]

        temp_loss = nn.functional.mse_loss(
            city,
            target_temp,
        )

        humidity_loss = nn.functional.mse_loss(
            prediction[
                "humidity"
            ][
                city_indices
            ],
            target[
                "humidity"
            ],
        )

        wind_loss = nn.functional.mse_loss(
            prediction[
                "wind_speed"
            ][
                city_indices
            ],
            target[
                "wind_speed"
            ],
        )

        # ----------------------------------------------------
        # Extreme-aware precipitation amount loss.
        #
        # Targets are normalized log1p precipitation.
        # The raw precipitation is recovered only to assign
        # training weights. Regression remains in normalized
        # log space.
        # ----------------------------------------------------

        precip_prediction = prediction[
            "precipitation_amount"
        ][
            city_indices
        ]

        precip_target = target[
            "precipitation"
        ]

        precip_stats = target_stats[
            str(horizon)
        ][
            "precipitation_log"
        ]

        precip_target_log = (
            precip_target
            * float(precip_stats["std"])
            + float(precip_stats["mean"])
        )

        precip_target_raw = torch.expm1(
            torch.clamp(
                precip_target_log,
                min=-20.0,
            )
        )

        precip_target_raw = torch.clamp(
            precip_target_raw,
            min=0.0,
        )

        precip_weights = torch.ones_like(
            precip_target_raw
        )

        dry_mask = (
            precip_target_raw <= 0.0
        )

        q90_mask = (
            precip_target_raw
            >= PRECIP_Q90_MM
        )

        q95_mask = (
            precip_target_raw
            >= PRECIP_Q95_MM
        )

        q99_mask = (
            precip_target_raw
            >= PRECIP_Q99_MM
        )

        precip_weights = torch.where(
            dry_mask,
            torch.full_like(
                precip_target_raw,
                PRECIP_DRY_WEIGHT,
            ),
            precip_weights,
        )

        precip_weights = torch.where(
            q90_mask,
            torch.full_like(
                precip_target_raw,
                PRECIP_Q90_WEIGHT,
            ),
            precip_weights,
        )

        precip_weights = torch.where(
            q95_mask,
            torch.full_like(
                precip_target_raw,
                PRECIP_Q95_WEIGHT,
            ),
            precip_weights,
        )

        precip_weights = torch.where(
            q99_mask,
            torch.full_like(
                precip_target_raw,
                PRECIP_Q99_WEIGHT,
            ),
            precip_weights,
        )

        precip_squared_error = (
            precip_prediction
            - precip_target
        ) ** 2

        precip_amount_loss = (
            precip_squared_error
            * precip_weights
        ).sum() / precip_weights.sum().clamp(
            min=1.0
        )

        pos_weight = torch.tensor(
            occurrence_weights[
                str(horizon)
            ],
            dtype=torch.float32,
            device=DEVICE,
        )

        occurrence_loss = (
            nn.functional.binary_cross_entropy_with_logits(
                prediction[
                    "precipitation_occurrence"
                ][
                    city_indices
                ],
                target[
                    "precipitation_occurrence"
                ],
                pos_weight=pos_weight,
            )
        )

        horizon_loss = (
            temp_loss
            + humidity_loss
            + wind_loss
            + precip_amount_loss
            + occurrence_loss
        )

        total_loss = (
            total_loss
            + horizon_loss
        )

        component_losses.append(
            horizon_loss.detach()
        )

    total_loss = (
        total_loss
        / len(HORIZONS)
    )

    return (
        total_loss,
        component_losses,
    )


# ============================================================
# METRICS
# ============================================================

def continuous_metrics(
    actual,
    predicted,
):
    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    mask = np.isfinite(
        actual
    ) & np.isfinite(
        predicted
    )

    actual = actual[
        mask
    ]

    predicted = predicted[
        mask
    ]

    if len(actual) == 0:
        return {
            "mae": None,
            "rmse": None,
            "r2": None,
        }

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    try:
        r2 = r2_score(
            actual,
            predicted,
        )
    except ValueError:
        r2 = None

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": (
            float(r2)
            if r2 is not None
            else None
        ),
    }


def evaluate_predictions(
    prediction_records,
):
    """
    prediction_records:
        list of:
            {
              horizon,
              city,
              actual_temp,
              pred_temp,
              ...
            }
    """

    frame = pd.DataFrame(
        prediction_records
    )

    metrics = {}

    selection_values = []

    for horizon in HORIZONS:

        subset = frame[
            frame["horizon"]
            == horizon
        ]

        temp = continuous_metrics(
            subset["actual_temperature"],
            subset["pred_temperature"],
        )

        humidity = continuous_metrics(
            subset["actual_humidity"],
            subset["pred_humidity"],
        )

        wind = continuous_metrics(
            subset["actual_wind"],
            subset["pred_wind"],
        )

        precip = continuous_metrics(
            subset["actual_precipitation"],
            subset["pred_precipitation"],
        )

        actual_occ = (
            subset[
                "actual_occurrence"
            ]
            .astype(int)
            .to_numpy()
        )

        pred_occ_prob = (
            subset[
                "pred_occurrence_probability"
            ]
            .astype(float)
            .to_numpy()
        )

        pred_occ = (
            pred_occ_prob
            >= OCCURRENCE_THRESHOLD
        ).astype(int)

        occurrence = {
            "precision": float(
                precision_score(
                    actual_occ,
                    pred_occ,
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    actual_occ,
                    pred_occ,
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    actual_occ,
                    pred_occ,
                    zero_division=0,
                )
            ),
        }

        wet_mask = (
            actual_occ == 1
        )

        if wet_mask.any():

            wet_actual = (
                subset[
                    "actual_precipitation"
                ]
                .to_numpy()
                [wet_mask]
            )

            wet_predicted = (
                subset[
                    "pred_precipitation"
                ]
                .to_numpy()
                [wet_mask]
            )

            wet_mae = float(
                mean_absolute_error(
                    wet_actual,
                    wet_predicted,
                )
            )

        else:
            wet_mae = None

        metrics[
            str(horizon)
        ] = {
            "temperature": temp,
            "humidity": humidity,
            "wind": wind,
            "precipitation": precip,
            "precipitation_occurrence": occurrence,
            "wet_case_precipitation_mae": wet_mae,
            "samples": int(
                len(subset)
            ),
        }

        # Validation selection proxy:
        # normalized regression MAE. Lower is better.
        scale_values = []

        for col in [
            "normalized_temperature_error",
            "normalized_humidity_error",
            "normalized_wind_error",
            "normalized_precipitation_error",
        ]:

            scale_values.append(
                float(
                    subset[col].mean()
                )
            )

        selection_values.extend(
            scale_values
        )

    metrics["selection_score"] = float(
        np.mean(
            selection_values
        )
    )

    return metrics, frame


# ============================================================
# EVALUATION LOOP
# ============================================================

def run_evaluation(
    model,
    df,
    sequences,
    city_lookup,
    lookup,
    input_normalization,
    target_stats,
    return_grid=False,
):
    model.eval()

    graph = build_graph_structure(
        grid_size=1.0
    )

    edge_index = build_edge_index(
        graph
    )

    city_mapping = build_city_grid_mapping(
        graph
    )

    city_indices = torch.tensor(
        [
            node_id_to_index(
                city_mapping[city]
            )
            for city in CITY_NAMES
        ],
        dtype=torch.long,
        device=DEVICE,
    )

    records = []

    grid_temperature = []
    grid_humidity = []
    grid_wind = []
    grid_precipitation = []
    grid_occurrence = []

    timestamps = []
    target_timestamps = []

    with torch.no_grad():

        for number, sequence in enumerate(
            sequences,
            start=1,
        ):

            if (
                number == 1
                or number % 500 == 0
            ):
                print(
                    f"  Evaluation "
                    f"{number:,}/{len(sequences):,}"
                )

            sequence_tensor = (
                build_sequence_tensor(
                    sequence,
                    city_lookup,
                    city_mapping,
                    input_normalization,
                ).to(DEVICE)
            )

            outputs = model(
                sequence_tensor,
                edge_index,
            )

            targets = build_targets_for_sequence(
                sequence,
                lookup,
                target_stats,
            )

            final_time = pd.Timestamp(
                sequence[-1]
            )

            timestamps.append(
                final_time.isoformat()
            )

            horizon_grid = {
                "temperature": [],
                "humidity": [],
                "wind": [],
                "precipitation": [],
                "occurrence": [],
            }

            for horizon in HORIZONS:

                key = f"h{horizon}"

                prediction = outputs[
                    key
                ]

                stats = target_stats[
                    str(horizon)
                ]

                temp_norm = (
                    prediction[
                        "temperature"
                    ][
                        city_indices
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                humidity_norm = (
                    prediction[
                        "humidity"
                    ][
                        city_indices
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                wind_norm = (
                    prediction[
                        "wind_speed"
                    ][
                        city_indices
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                precip_norm = (
                    prediction[
                        "precipitation_amount"
                    ][
                        city_indices
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                occurrence_logits = (
                    prediction[
                        "precipitation_occurrence"
                    ][
                        city_indices
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                occurrence_probability = (
                    1.0
                    / (
                        1.0
                        + np.exp(
                            -np.clip(
                                occurrence_logits,
                                -50,
                                50,
                            )
                        )
                    )
                )

                pred_temp = (
                    temp_norm
                    * stats[
                        "temperature"
                    ]["std"]
                    + stats[
                        "temperature"
                    ]["mean"]
                )

                pred_humidity = (
                    humidity_norm
                    * stats[
                        "humidity"
                    ]["std"]
                    + stats[
                        "humidity"
                    ]["mean"]
                )

                pred_wind = (
                    wind_norm
                    * stats[
                        "wind_speed"
                    ]["std"]
                    + stats[
                        "wind_speed"
                    ]["mean"]
                )

                pred_precip_log = (
                    precip_norm
                    * stats[
                        "precipitation_log"
                    ]["std"]
                    + stats[
                        "precipitation_log"
                    ]["mean"]
                )

                pred_precip = np.expm1(
                    np.maximum(
                        -20.0,
                        pred_precip_log,
                    )
                )

                pred_humidity = np.clip(
                    pred_humidity,
                    0.0,
                    100.0,
                )

                pred_wind = np.maximum(
                    0.0,
                    pred_wind,
                )

                pred_precip = np.maximum(
                    0.0,
                    pred_precip,
                )

                target = targets[
                    str(horizon)
                ]

                actual_temp = (
                    target[
                        "temperature"
                    ]
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                actual_temp = (
                    actual_temp
                    * stats[
                        "temperature"
                    ]["std"]
                    + stats[
                        "temperature"
                    ]["mean"]
                )

                actual_humidity = (
                    target[
                        "humidity"
                    ]
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                actual_humidity = (
                    actual_humidity
                    * stats[
                        "humidity"
                    ]["std"]
                    + stats[
                        "humidity"
                    ]["mean"]
                )

                actual_wind = (
                    target[
                        "wind_speed"
                    ]
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                actual_wind = (
                    actual_wind
                    * stats[
                        "wind_speed"
                    ]["std"]
                    + stats[
                        "wind_speed"
                    ]["mean"]
                )

                actual_precip_log = (
                    target[
                        "precipitation"
                    ]
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                actual_precip_log = (
                    actual_precip_log
                    * stats[
                        "precipitation_log"
                    ]["std"]
                    + stats[
                        "precipitation_log"
                    ]["mean"]
                )

                actual_precip = np.expm1(
                    actual_precip_log
                )

                actual_precip = np.maximum(
                    0.0,
                    actual_precip,
                )

                actual_occurrence = (
                    target[
                        "precipitation_occurrence"
                    ]
                    .cpu()
                    .numpy()
                    .reshape(-1)
                )

                horizon_target_time = (
                    final_time
                    + pd.Timedelta(
                        hours=horizon
                    )
                )

                if horizon == 1:
                    grid_temperature.append(
                        pred_temp
                        if False
                        else None
                    )

                for city_index, city in enumerate(
                    CITY_NAMES
                ):

                    records.append(
                        {
                            "horizon": horizon,
                            "city": city,
                            "timestamp": final_time,
                            "target_timestamp": horizon_target_time,
                            "actual_temperature": float(
                                actual_temp[
                                    city_index
                                ]
                            ),
                            "pred_temperature": float(
                                pred_temp[
                                    city_index
                                ]
                            ),
                            "actual_humidity": float(
                                actual_humidity[
                                    city_index
                                ]
                            ),
                            "pred_humidity": float(
                                pred_humidity[
                                    city_index
                                ]
                            ),
                            "actual_wind": float(
                                actual_wind[
                                    city_index
                                ]
                            ),
                            "pred_wind": float(
                                pred_wind[
                                    city_index
                                ]
                            ),
                            "actual_precipitation": float(
                                actual_precip[
                                    city_index
                                ]
                            ),
                            "pred_precipitation": float(
                                pred_precip[
                                    city_index
                                ]
                            ),
                            "actual_occurrence": float(
                                actual_occurrence[
                                    city_index
                                ]
                            ),
                            "pred_occurrence_probability": float(
                                occurrence_probability[
                                    city_index
                                ]
                            ),
                            "normalized_temperature_error": float(
                                abs(
                                    temp_norm[
                                        city_index
                                    ]
                                    - target[
                                        "temperature"
                                    ][
                                        city_index,
                                        0,
                                    ]
                                    .item()
                                )
                            ),
                            "normalized_humidity_error": float(
                                abs(
                                    humidity_norm[
                                        city_index
                                    ]
                                    - target[
                                        "humidity"
                                    ][
                                        city_index,
                                        0,
                                    ]
                                    .item()
                                )
                            ),
                            "normalized_wind_error": float(
                                abs(
                                    wind_norm[
                                        city_index
                                    ]
                                    - target[
                                        "wind_speed"
                                    ][
                                        city_index,
                                        0,
                                    ]
                                    .item()
                                )
                            ),
                            "normalized_precipitation_error": float(
                                abs(
                                    precip_norm[
                                        city_index
                                    ]
                                    - target[
                                        "precipitation"
                                    ][
                                        city_index,
                                        0,
                                    ]
                                    .item()
                                )
                            ),
                        }
                    )

                # Full grid output is useful for 2025 final field.
                if return_grid:

                    grid_temperature.append(
                        (
                            horizon,
                            pred_temp,
                        )
                    )

            if return_grid:
                # Rebuild full-grid outputs once for this sequence.
                # Stored later in horizon-specific arrays.
                pass

    # --------------------------------------------------------
    # Full-grid second pass is intentionally avoided in the
    # metric records above. Build it separately only for 2025
    # after metrics, to keep memory predictable.
    # --------------------------------------------------------

    metrics, records_frame = (
        evaluate_predictions(
            records
        )
    )

    return (
        metrics,
        records_frame,
    )


# ============================================================
# FULL 900-NODE GRID FORECAST EXPORT
# ============================================================

def export_full_grid_forecast(
    model,
    df,
    sequences,
    input_normalization,
    graph,
    city_mapping,
):
    print(
        "\nExporting 2025 full 900-node "
        "multi-horizon forecast field..."
    )

    city_lookup = build_city_lookup(
        df
    )

    edge_index = build_edge_index(
        graph
    )

    all_temperature = []
    all_humidity = []
    all_wind = []
    all_precipitation = []
    all_occurrence = []

    timestamps = []
    target_timestamps = []

    with torch.no_grad():

        for number, sequence in enumerate(
            sequences,
            start=1,
        ):

            if (
                number == 1
                or number % 500 == 0
            ):
                print(
                    f"  Grid export "
                    f"{number:,}/{len(sequences):,}"
                )

            sequence_tensor = (
                build_sequence_tensor(
                    sequence,
                    city_lookup,
                    city_mapping,
                    input_normalization,
                ).to(DEVICE)
            )

            outputs = model(
                sequence_tensor,
                edge_index,
            )

            timestamps.append(
                pd.Timestamp(
                    sequence[-1]
                ).isoformat()
            )

            horizon_temperature = []
            horizon_humidity = []
            horizon_wind = []
            horizon_precipitation = []
            horizon_occurrence = []
            horizon_targets = []

            for horizon in HORIZONS:

                key = f"h{horizon}"

                stats = checkpoint_target_stats[
                    str(horizon)
                ]

                prediction = outputs[
                    key
                ]

                temp_norm = (
                    prediction[
                        "temperature"
                    ][
                        :,
                        0,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                humidity_norm = (
                    prediction[
                        "humidity"
                    ][
                        :,
                        0,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                wind_norm = (
                    prediction[
                        "wind_speed"
                    ][
                        :,
                        0,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                precip_norm = (
                    prediction[
                        "precipitation_amount"
                    ][
                        :,
                        0,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                occurrence_logits = (
                    prediction[
                        "precipitation_occurrence"
                    ][
                        :,
                        0,
                    ]
                    .detach()
                    .cpu()
                    .numpy()
                )

                temperature = (
                    temp_norm
                    * stats[
                        "temperature"
                    ]["std"]
                    + stats[
                        "temperature"
                    ]["mean"]
                )

                humidity = (
                    humidity_norm
                    * stats[
                        "humidity"
                    ]["std"]
                    + stats[
                        "humidity"
                    ]["mean"]
                )

                wind = (
                    wind_norm
                    * stats[
                        "wind_speed"
                    ]["std"]
                    + stats[
                        "wind_speed"
                    ]["mean"]
                )

                precip_log = (
                    precip_norm
                    * stats[
                        "precipitation_log"
                    ]["std"]
                    + stats[
                        "precipitation_log"
                    ]["mean"]
                )

                precipitation = np.expm1(
                    np.maximum(
                        -20.0,
                        precip_log,
                    )
                )

                occurrence_probability = (
                    1.0
                    / (
                        1.0
                        + np.exp(
                            -np.clip(
                                occurrence_logits,
                                -50,
                                50,
                            )
                        )
                    )
                )

                humidity = np.clip(
                    humidity,
                    0.0,
                    100.0,
                )

                wind = np.maximum(
                    0.0,
                    wind,
                )

                precipitation = np.maximum(
                    0.0,
                    precipitation,
                )

                horizon_temperature.append(
                    temperature.astype(
                        np.float32
                    )
                )

                horizon_humidity.append(
                    humidity.astype(
                        np.float32
                    )
                )

                horizon_wind.append(
                    wind.astype(
                        np.float32
                    )
                )

                horizon_precipitation.append(
                    precipitation.astype(
                        np.float32
                    )
                )

                horizon_occurrence.append(
                    occurrence_probability.astype(
                        np.float32
                    )
                )

                horizon_targets.append(
                    (
                        pd.Timestamp(
                            sequence[-1]
                        )
                        + pd.Timedelta(
                            hours=horizon
                        )
                    ).isoformat()
                )

            all_temperature.append(
                np.stack(
                    horizon_temperature
                )
            )

            all_humidity.append(
                np.stack(
                    horizon_humidity
                )
            )

            all_wind.append(
                np.stack(
                    horizon_wind
                )
            )

            all_precipitation.append(
                np.stack(
                    horizon_precipitation
                )
            )

            all_occurrence.append(
                np.stack(
                    horizon_occurrence
                )
            )

            target_timestamps.append(
                horizon_targets
            )

    temperature = np.stack(
        all_temperature
    )

    humidity = np.stack(
        all_humidity
    )

    wind = np.stack(
        all_wind
    )

    precipitation = np.stack(
        all_precipitation
    )

    occurrence = np.stack(
        all_occurrence
    )

    print(
        "\nGrid forecast shapes:"
    )

    print(
        "  temperature:",
        temperature.shape,
    )

    print(
        "  humidity:",
        humidity.shape,
    )

    print(
        "  wind:",
        wind.shape,
    )

    print(
        "  precipitation:",
        precipitation.shape,
    )

    print(
        "  occurrence:",
        occurrence.shape,
    )

    np.savez_compressed(
        GRID_OUTPUT,
        horizons=np.array(
            HORIZONS,
            dtype=np.int32,
        ),
        timestamps=np.array(
            timestamps
        ),
        target_timestamps=np.array(
            target_timestamps
        ),
        node_ids=np.array(
            [
                node.node_id
                for node in graph[
                    "nodes"
                ]
            ]
        ),
        latitudes=np.array(
            [
                node.latitude
                for node in graph[
                    "nodes"
                ]
            ],
            dtype=np.float32,
        ),
        longitudes=np.array(
            [
                node.longitude
                for node in graph[
                    "nodes"
                ]
            ],
            dtype=np.float32,
        ),
        temperature=temperature,
        humidity=humidity,
        wind=wind,
        precipitation=precipitation,
        occurrence_probability=occurrence,
    )

    print(
        f"\nSaved grid forecast field:\n"
        f"{GRID_OUTPUT}"
    )


# ============================================================
# MAIN
# ============================================================

checkpoint_target_stats = {}


def main():

    print("=" * 70)
    print(
        "VATAVARAN V7 - MULTI-HORIZON "
        "TEMPORAL GNN"
    )
    print("=" * 70)

    print(
        "\nHorizons:",
        HORIZONS,
    )

    print(
        "\nLoading dataset..."
    )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["time"],
    )

    df = df.sort_values(
        [
            "location",
            "time",
        ]
    ).reset_index(
        drop=True
    )

    train_df = df[
        df["time"].dt.year <= 2023
    ].copy()

    val_df = df[
        df["time"].dt.year == 2024
    ].copy()

    test_df = df[
        df["time"].dt.year == 2025
    ].copy()

    print(
        f"Train rows: {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(val_df):,}"
    )

    print(
        f"Test rows: {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------

    print(
        "\nBuilding nationwide graph..."
    )

    graph = build_graph_structure(
        grid_size=1.0
    )

    print(
        f"Graph nodes: "
        f"{graph['node_count']}"
    )

    print(
        f"Graph base edges: "
        f"{graph['edge_count']}"
    )

    edge_index = build_edge_index(
        graph
    )

    print(
        f"Directed edges: "
        f"{edge_index.shape[1]}"
    )

    city_mapping = build_city_grid_mapping(
        graph
    )

    city_indices = torch.tensor(
        [
            node_id_to_index(
                city_mapping[city]
            )
            for city in CITY_NAMES
        ],
        dtype=torch.long,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Input normalization
    # --------------------------------------------------------

    print(
        "\nBuilding training-only "
        "input normalization..."
    )

    input_normalization = (
        calculate_input_normalization(
            train_df
        )
    )

    # --------------------------------------------------------
    # Target normalization
    # --------------------------------------------------------

    print(
        "Building training-only "
        "multi-horizon target normalization..."
    )

    target_stats = (
        build_target_normalization(
            train_df
        )
    )

    global checkpoint_target_stats
    checkpoint_target_stats = (
        target_stats
    )

    occurrence_weights = (
        build_occurrence_weights(
            train_df
        )
    )

    # --------------------------------------------------------
    # Sequences
    # --------------------------------------------------------

    print(
        "\nPreparing training sequences..."
    )

    train_sequences, train_city_lookup, train_lookup = (
        prepare_sequences(
            train_df,
            2023,
        )
    )

    print(
        f"Training sequences: "
        f"{len(train_sequences):,}"
    )

    print(
        "\nPreparing validation sequences..."
    )

    val_sequences, val_city_lookup, val_lookup = (
        prepare_sequences(
            val_df,
            2024,
        )
    )

    print(
        f"Validation sequences: "
        f"{len(val_sequences):,}"
    )

    print(
        "\nPreparing 2025 test sequences..."
    )

    test_sequences, test_city_lookup, test_lookup = (
        prepare_sequences(
            test_df,
            2025,
        )
    )

    print(
        f"Test sequences: "
        f"{len(test_sequences):,}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nCreating fresh V7 model..."
    )

    model = VatavaranGNNV7(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        gru_hidden_features=GRU_HIDDEN_FEATURES,
        horizons=HORIZONS,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print(
        "\nStarting V7 training..."
    )

    best_validation_score = float(
        "inf"
    )

    best_epoch = 0

    training_history = []

    train_total = len(
        train_sequences
    )

    if train_total == 0:
        raise RuntimeError(
            "No valid training sequences."
        )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        sample_count = min(
            SAMPLES_PER_EPOCH,
            train_total,
        )

        generator = random.Random(
            SEED + epoch
        )

        if sample_count < train_total:

            selected_indices = (
                generator.sample(
                    range(train_total),
                    sample_count,
                )
            )

        else:

            selected_indices = list(
                range(train_total)
            )

        epoch_losses = []

        print(
            "\n" + "-" * 70
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Samples: {sample_count:,}"
        )

        for number, sequence_index in enumerate(
            selected_indices,
            start=1,
        ):

            sequence = train_sequences[
                sequence_index
            ]

            sequence_tensor = (
                build_sequence_tensor(
                    sequence,
                    train_city_lookup,
                    city_mapping,
                    input_normalization,
                ).to(DEVICE)
            )

            targets = (
                build_targets_for_sequence(
                    sequence,
                    train_lookup,
                    target_stats,
                )
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                sequence_tensor,
                edge_index,
            )

            loss, _ = compute_loss(
                outputs,
                targets,
                city_indices,
                occurrence_weights,
                target_stats,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            epoch_losses.append(
                float(
                    loss.detach().cpu()
                )
            )

            if (
                number == 1
                or number % 500 == 0
            ):
                print(
                    f"  {number:,}/{sample_count:,} "
                    f"loss={np.mean(epoch_losses):.5f}"
                )

        mean_train_loss = float(
            np.mean(
                epoch_losses
            )
        )

        # ----------------------------------------------------
        # Validation after every epoch.
        # ----------------------------------------------------

        print(
            "\nValidation..."
        )

        val_metrics, _ = run_evaluation(
            model,
            val_df,
            val_sequences,
            val_city_lookup,
            val_lookup,
            input_normalization,
            target_stats,
            return_grid=False,
        )

        validation_score = float(
            val_metrics[
                "selection_score"
            ]
        )

        print(
            f"\nEpoch {epoch} summary:"
        )

        print(
            f"  Train loss: "
            f"{mean_train_loss:.6f}"
        )

        print(
            f"  Validation selection score: "
            f"{validation_score:.6f}"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "validation_selection_score": (
                validation_score
            ),
        }

        training_history.append(
            epoch_record
        )

        if (
            validation_score
            < best_validation_score
        ):

            best_validation_score = (
                validation_score
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "input_features": INPUT_FEATURES,
                    "hidden_features": HIDDEN_FEATURES,
                    "gru_hidden_features": GRU_HIDDEN_FEATURES,
                    "horizons": HORIZONS,
                    "sequence_length": SEQUENCE_LENGTH,
                    "time_stride": TIME_STRIDE,
                    "weather_features": WEATHER_FEATURES,
                    "city_names": CITY_NAMES,
                    "city_grid_mapping": city_mapping,
                    "input_normalization": input_normalization,
                    "target_normalization": target_stats,
                    "occurrence_pos_weight": occurrence_weights,
                    "occurrence_threshold": OCCURRENCE_THRESHOLD,
                    "architecture": (
                        "2-layer MaskAwareGraphConv + "
                        "temporal GRU + direct multi-horizon heads + "
                        "extreme-aware precipitation loss"
                    ),
                    "target_horizons_hours": HORIZONS,
                    "training_period": "2021-2023",
                    "validation_period": "2024",
                    "test_period": "2025",
                    "best_epoch": best_epoch,
                    "validation_selection_score": (
                        best_validation_score
                    ),
                    "precipitation_loss": {
                        "dry_weight": PRECIP_DRY_WEIGHT,
                        "q90_mm": PRECIP_Q90_MM,
                        "q90_weight": PRECIP_Q90_WEIGHT,
                        "q95_mm": PRECIP_Q95_MM,
                        "q95_weight": PRECIP_Q95_WEIGHT,
                        "q99_mm": PRECIP_Q99_MM,
                        "q99_weight": PRECIP_Q99_WEIGHT,
                    },
                    "experiment": (
                        "V8 extreme-aware precipitation "
                        "PS26078 experiment"
                    ),
                },
                MODEL_OUTPUT,
            )

            print(
                "  ✓ New best checkpoint saved."
            )

    # --------------------------------------------------------
    # Reload best checkpoint.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "Loading best V7 checkpoint..."
    )

    checkpoint = torch.load(
        MODEL_OUTPUT,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    # --------------------------------------------------------
    # Final validation.
    # --------------------------------------------------------

    print(
        "\nFinal 2024 validation..."
    )

    validation_metrics, _ = (
        run_evaluation(
            model,
            val_df,
            val_sequences,
            val_city_lookup,
            val_lookup,
            input_normalization,
            target_stats,
            return_grid=False,
        )
    )

    # --------------------------------------------------------
    # FINAL 2025 HOLDOUT
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL 2025 MULTI-HORIZON HOLDOUT"
    )

    print(
        "=" * 70
    )

    test_metrics, test_frame = (
        run_evaluation(
            model,
            test_df,
            test_sequences,
            test_city_lookup,
            test_lookup,
            input_normalization,
            target_stats,
            return_grid=False,
        )
    )

    # --------------------------------------------------------
    # Save test metrics CSV.
    # --------------------------------------------------------

    metric_rows = []

    for horizon in HORIZONS:

        item = test_metrics[
            str(horizon)
        ]

        metric_rows.append(
            {
                "horizon_hours": horizon,
                "temperature_mae": item[
                    "temperature"
                ]["mae"],
                "temperature_rmse": item[
                    "temperature"
                ]["rmse"],
                "temperature_r2": item[
                    "temperature"
                ]["r2"],
                "humidity_mae": item[
                    "humidity"
                ]["mae"],
                "humidity_rmse": item[
                    "humidity"
                ]["rmse"],
                "humidity_r2": item[
                    "humidity"
                ]["r2"],
                "wind_mae": item[
                    "wind"
                ]["mae"],
                "wind_rmse": item[
                    "wind"
                ]["rmse"],
                "wind_r2": item[
                    "wind"
                ]["r2"],
                "precipitation_mae": item[
                    "precipitation"
                ]["mae"],
                "precipitation_rmse": item[
                    "precipitation"
                ]["rmse"],
                "precipitation_r2": item[
                    "precipitation"
                ]["r2"],
                "rain_precision": item[
                    "precipitation_occurrence"
                ]["precision"],
                "rain_recall": item[
                    "precipitation_occurrence"
                ]["recall"],
                "rain_f1": item[
                    "precipitation_occurrence"
                ]["f1"],
                "wet_case_precipitation_mae": item[
                    "wet_case_precipitation_mae"
                ],
                "samples": item[
                    "samples"
                ],
            }
        )

    test_metrics_df = pd.DataFrame(
        metric_rows
    )

    test_metrics_df.to_csv(
        TEST_METRICS_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # Final grid export.
    # --------------------------------------------------------

    print(
        "\nExporting 2025 full grid..."
    )

    export_full_grid_forecast(
        model,
        test_df,
        test_sequences,
        input_normalization,
        graph,
        city_mapping,
    )

    # --------------------------------------------------------
    # Final metrics JSON.
    # --------------------------------------------------------

    final_payload = {
        "project": "VATAVARAN",
        "problem_statement": "PS26078",
        "model": "VATAVARAN V7 Multi-Horizon Temporal GNN",
        "fresh_experiment": True,
        "v6_checkpoint_modified": False,
        "horizons_hours": HORIZONS,
        "sequence_length": SEQUENCE_LENGTH,
        "time_stride": TIME_STRIDE,
        "grid_nodes": int(
            graph["node_count"]
        ),
        "directed_edges": int(
            edge_index.shape[1]
        ),
        "training_rows": int(
            len(train_df)
        ),
        "validation_rows": int(
            len(val_df)
        ),
        "test_rows": int(
            len(test_df)
        ),
        "training_sequences": int(
            len(train_sequences)
        ),
        "validation_sequences": int(
            len(val_sequences)
        ),
        "test_sequences": int(
            len(test_sequences)
        ),
        "best_epoch": int(
            checkpoint[
                "best_epoch"
            ]
        ),
        "best_validation_selection_score": float(
            checkpoint[
                "validation_selection_score"
            ]
        ),
        "training_history": (
            training_history
        ),
        "validation_metrics_2024": (
            validation_metrics
        ),
        "test_metrics_2025": (
            test_metrics
        ),
        "outputs": {
            "checkpoint": str(
                MODEL_OUTPUT
            ),
            "test_metrics_csv": str(
                TEST_METRICS_CSV
            ),
            "grid_forecast_npz": str(
                GRID_OUTPUT
            ),
        },
    }

    with METRICS_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            final_payload,
            handle,
            indent=2,
        )

    # --------------------------------------------------------
    # Console final report.
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "VATAVARAN V7 FINAL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"Best epoch: "
        f"{checkpoint['best_epoch']}"
    )

    print(
        f"Validation selection score: "
        f"{checkpoint['validation_selection_score']:.6f}"
    )

    print(
        "\n2025 holdout:"
    )

    print(
        test_metrics_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved checkpoint:\n"
        f"{MODEL_OUTPUT}"
    )

    print(
        f"\nSaved test metrics:\n"
        f"{TEST_METRICS_CSV}"
    )

    print(
        f"\nSaved grid forecast:\n"
        f"{GRID_OUTPUT}"
    )

    print(
        f"\nSaved complete metrics JSON:\n"
        f"{METRICS_OUTPUT}"
    )

    print(
        "\nV6 checkpoint remains untouched."
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()

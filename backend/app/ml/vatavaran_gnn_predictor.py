from pathlib import Path

import numpy as np
import pandas as pd
import torch

from app.ml.train_vatavaran_gnn import (
    VatavaranGNN,
    WEATHER_FEATURES,
    CITY_NAMES,
    build_city_grid_mapping,
    build_snapshot,
    node_id_to_index,
)

from app.ml.nationwide_spatial_grid import build_graph_structure


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "multi_city_weather_ml_2021_2025.csv"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "vatavaran_gnn_temperature.pt"
)


DEVICE = torch.device("cpu")


class VatavaranGNNPredictor:

    def __init__(self):

        self.device = DEVICE

        # ----------------------------------------------------
        # Load trained checkpoint
        # ----------------------------------------------------

        self.checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
            weights_only=False,
        )

        # ----------------------------------------------------
        # Build graph
        # ----------------------------------------------------

        self.graph = build_graph_structure(
            grid_size=1.0
        )

        self.city_grid_mapping = (
            build_city_grid_mapping(
                self.graph
            )
        )

        # ----------------------------------------------------
        # Build edge index
        # ----------------------------------------------------

        edge_list = []

        for edge in self.graph["edges"]:

            source = node_id_to_index(
                edge[0]
            )

            target = node_id_to_index(
                edge[1]
            )

            # Both directions.
            edge_list.append(
                [source, target]
            )

            edge_list.append(
                [target, source]
            )

        self.edge_index = torch.tensor(
            edge_list,
            dtype=torch.long,
            device=self.device,
        ).t().contiguous()

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        self.model = VatavaranGNN(
            input_features=8,
            hidden_features=self.checkpoint[
                "hidden_features"
            ],
            output_features=1,
        ).to(self.device)

        self.model.load_state_dict(
            self.checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

        # ----------------------------------------------------
        # Load normalization
        # ----------------------------------------------------

        self.normalization = (
            self.checkpoint[
                "normalization"
            ]
        )

        # ----------------------------------------------------
        # Load latest historical data
        # ----------------------------------------------------

        self.df = pd.read_csv(
            DATA_PATH
        )

        self.df["time"] = pd.to_datetime(
            self.df["time"]
        )

        self.df = self.df.sort_values(
            [
                "location",
                "time",
            ]
        ).reset_index(
            drop=True
        )

    # ========================================================
    # NORMALIZE
    # ========================================================

    def normalize_value(
        self,
        value,
        feature,
    ):

        if value is None:
            return 0.0

        try:

            value = float(value)

        except (TypeError, ValueError):

            return 0.0

        if not np.isfinite(value):
            return 0.0

        statistics = self.normalization[
            feature
        ]

        return (
            value - statistics["mean"]
        ) / statistics["std"]

    # ========================================================
    # BUILD GRAPH FROM CURRENT WEATHER
    # ========================================================

    def build_current_graph(
        self,
        weather_rows,
    ):

        node_count = 900

        x = np.zeros(
            (node_count, 8),
            dtype=np.float32,
        )

        for city in CITY_NAMES:

            row = weather_rows.get(
                city
            )

            if row is None:
                continue

            node_id = (
                self.city_grid_mapping.get(
                    city
                )
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
                ] = self.normalize_value(
                    value,
                    feature,
                )

            # Observation mask.
            x[
                node_index,
                7
            ] = 1.0

        return torch.tensor(
            x,
            dtype=torch.float32,
            device=self.device,
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        location,
    ):

        if location not in CITY_NAMES:

            raise ValueError(
                f"Unsupported location: {location}. "
                f"Supported locations: {', '.join(CITY_NAMES)}"
            )

        # ----------------------------------------------------
        # Latest observation for requested city
        # ----------------------------------------------------

        location_df = self.df[
            self.df["location"] == location
        ]

        if location_df.empty:

            raise ValueError(
                f"No historical data found for {location}"
            )

        latest_row = (
            location_df.iloc[-1]
        )

        timestamp = latest_row["time"]

        # ----------------------------------------------------
        # Build graph using the latest timestamp
        # for every available city.
        # ----------------------------------------------------

        timestamp_df = self.df[
            self.df["time"] == timestamp
        ]

        weather_rows = {}

        for _, row in timestamp_df.iterrows():

            weather_rows[
                row["location"]
            ] = {
                feature: row[feature]
                for feature in WEATHER_FEATURES
            }

        x = self.build_current_graph(
            weather_rows
        )

        # ----------------------------------------------------
        # GNN inference
        # ----------------------------------------------------

        with torch.no_grad():

            output = self.model(
                x,
                self.edge_index,
            )

        node_id = (
            self.city_grid_mapping[
                location
            ]
        )

        node_index = node_id_to_index(
            node_id
        )

        prediction = float(
            output[
                node_index,
                0
            ].item()
        )

        current_temperature = float(
            latest_row[
                "temperature_2m"
            ]
        )

        return {
            "location": location,
            "grid_node": node_id,
            "data_timestamp": (
                timestamp.isoformat()
            ),
            "current_temperature_c": round(
                current_temperature,
                3,
            ),
            "next_hour_temperature_c": round(
                prediction,
                3,
            ),
            "temperature_change_c": round(
                prediction -
                current_temperature,
                3,
            ),
            "model": (
                "VATAVARAN Spatial GNN"
            ),
            "model_type": (
                "Mask-aware 2-layer "
                "spatial Graph Neural Network"
            ),
            "target": (
                "next-hour temperature"
            ),
            "graph_nodes": 900,
            "graph_directed_edges": int(
                self.edge_index.shape[1]
            ),
            "trained": True,
            "live_data": False,
            "ground_truth_validated_nationwide": False,
        }


# ============================================================
# SINGLETON
# ============================================================

_predictor = None


def get_gnn_predictor():

    global _predictor

    if _predictor is None:

        _predictor = (
            VatavaranGNNPredictor()
        )

    return _predictor


def predict_gnn_temperature(
    location,
):

    predictor = get_gnn_predictor()

    return predictor.predict(
        location
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    result = predict_gnn_temperature(
        "Patna"
    )

    print()
    print(
        "VATAVARAN GNN Temperature Prediction"
    )
    print(
        "------------------------------------"
    )

    for key, value in result.items():

        print(
            f"{key}: {value}"
        )

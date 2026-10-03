from pathlib import Path
import math
import pandas as pd

from app.ml.nationwide_spatial_grid import (
    build_spatial_grid,
    build_grid_index,
    build_neighbor_map,
    build_graph_structure,
)

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "multi_city_weather_ml_2021_2025.csv"

INDIA_GRID_SIZE = 1.0

CITY_COORDINATES = {
    "Ahmedabad": (23.0225, 72.5714),
    "Bengaluru": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Delhi": (28.6139, 77.2090),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Mumbai": (19.0760, 72.8777),
    "Patna": (25.5941, 85.1376),
}

WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two geographic points.
    """

    radius = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c


def nearest_grid_node(latitude, longitude, grid_nodes):
    """
    Find the nearest grid node using geographic distance.
    """

    best_node = None
    best_distance = float("inf")

    for node in grid_nodes:

        distance = haversine_km(
            latitude,
            longitude,
            node.latitude,
            node.longitude,
        )

        if distance < best_distance:
            best_distance = distance
            best_node = node

    return best_node, best_distance


def build_city_grid_mapping(grid_nodes):
    """
    Map each available city observation to its nearest India grid node.
    """

    mapping = {}

    for city, (latitude, longitude) in CITY_COORDINATES.items():

        node, distance = nearest_grid_node(
            latitude,
            longitude,
            grid_nodes,
        )

        mapping[city] = {
            "node_id": node.node_id,
            "city_latitude": latitude,
            "city_longitude": longitude,
            "grid_latitude": node.latitude,
            "grid_longitude": node.longitude,
            "distance_km": round(distance, 3),
        }

    return mapping


def load_weather_data():
    """
    Load historical multi-city weather dataset.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Weather dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    df["time"] = pd.to_datetime(df["time"])

    return df


def build_grid_snapshot(
    df,
    timestamp,
    grid_nodes,
    grid_index,
    city_mapping,
):
    """
    Build a complete 900-node graph snapshot.

    Every grid node exists in the graph.

    Nodes with weather observations:
        observed = 1

    Nodes without observations:
        observed = 0

    Missing weather values are represented by None.
    """

    timestamp = pd.Timestamp(timestamp)

    snapshot_df = df[df["time"] == timestamp].copy()

    observed_by_node = {}

    for _, row in snapshot_df.iterrows():

        location = row["location"]

        if location not in city_mapping:
            continue

        node_id = city_mapping[location]["node_id"]

        observed_by_node[node_id] = {
            "location": location,
            "temperature_2m": float(row["temperature_2m"]),
            "relative_humidity_2m": float(
                row["relative_humidity_2m"]
            ),
            "precipitation": float(row["precipitation"]),
            "rain": float(row["rain"]),
            "surface_pressure": float(row["surface_pressure"]),
            "wind_speed_10m": float(row["wind_speed_10m"]),
            "wind_direction_10m": float(
                row["wind_direction_10m"]
            ),
        }

    nodes = []

    for node in grid_nodes:

        weather = observed_by_node.get(node.node_id)

        if weather is None:

            node_record = {
                "node_id": node.node_id,
                "latitude": node.latitude,
                "longitude": node.longitude,
                "observed": 0,
                "location": None,
                "features": {
                    feature: None
                    for feature in WEATHER_FEATURES
                },
            }

        else:

            node_record = {
                "node_id": node.node_id,
                "latitude": node.latitude,
                "longitude": node.longitude,
                "observed": 1,
                "location": weather["location"],
                "features": {
                    feature: weather[feature]
                    for feature in WEATHER_FEATURES
                },
            }

        nodes.append(node_record)

    edges = []

    for node_id, neighbors in build_neighbor_map(
        grid_nodes
    ).items():

        for neighbor_id in neighbors:

            if node_id < neighbor_id:

                edges.append(
                    {
                        "source": node_id,
                        "target": neighbor_id,
                    }
                )

    observed_count = sum(
        node["observed"] for node in nodes
    )

    return {
        "timestamp": timestamp.isoformat(),
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "observed_node_count": observed_count,
        "unobserved_node_count": (
            len(nodes) - observed_count
        ),
        "graph_type": "8-connected geographic grid",
        "gnn_ready": True,
        "nationwide_data": observed_count == len(nodes),
    }


def build_weather_graph_snapshot(timestamp=None):

    df = load_weather_data()

    grid_nodes = build_spatial_grid(
        grid_size=INDIA_GRID_SIZE
    )

    grid_index = build_grid_index(grid_nodes)

    city_mapping = build_city_grid_mapping(
        grid_nodes
    )

    if timestamp is None:

        timestamp = df["time"].min()

    snapshot = build_grid_snapshot(
        df=df,
        timestamp=timestamp,
        grid_nodes=grid_nodes,
        grid_index=grid_index,
        city_mapping=city_mapping,
    )

    return {
        "dataset": str(DATA_PATH),
        "grid_size_degrees": INDIA_GRID_SIZE,
        "grid_node_count": len(grid_nodes),
        "city_mapping": city_mapping,
        "snapshot": snapshot,
    }


if __name__ == "__main__":

    print("VATAVARAN Full Weather Graph Builder")
    print("------------------------------------")

    print(f"Dataset: {DATA_PATH}")

    result = build_weather_graph_snapshot()

    print()
    print("India Grid")
    print("----------")

    print(
        f"Grid nodes: "
        f"{result['grid_node_count']}"
    )

    print()
    print("City → Grid Node Mapping")
    print("------------------------")

    for city, info in result["city_mapping"].items():

        print(
            f"{city:<10} → "
            f"{info['node_id']} | "
            f"city=("
            f"{info['city_latitude']:.4f}, "
            f"{info['city_longitude']:.4f}"
            f") | "
            f"grid=("
            f"{info['grid_latitude']:.1f}, "
            f"{info['grid_longitude']:.1f}"
            f") | "
            f"distance={info['distance_km']:.2f} km"
        )

    snapshot = result["snapshot"]

    print()
    print("Graph Snapshot")
    print("--------------")

    print(
        f"Timestamp: "
        f"{snapshot['timestamp']}"
    )

    print(
        f"Total nodes: "
        f"{snapshot['node_count']}"
    )

    print(
        f"Observed nodes: "
        f"{snapshot['observed_node_count']}"
    )

    print(
        f"Unobserved nodes: "
        f"{snapshot['unobserved_node_count']}"
    )

    print(
        f"Edges: "
        f"{snapshot['edge_count']}"
    )

    print(
        f"Graph type: "
        f"{snapshot['graph_type']}"
    )

    print(
        f"GNN ready: "
        f"{snapshot['gnn_ready']}"
    )

    print(
        f"Nationwide data: "
        f"{snapshot['nationwide_data']}"
    )

    print()
    print("First observed node")
    print("-------------------")

    first_observed = next(
        (
            node
            for node in snapshot["nodes"]
            if node["observed"] == 1
        ),
        None,
    )

    print(first_observed)

    print()
    print("First unobserved node")
    print("---------------------")

    first_unobserved = next(
        (
            node
            for node in snapshot["nodes"]
            if node["observed"] == 0
        ),
        None,
    )

    print(first_unobserved)

    print()
    print("First 5 graph edges")
    print("-------------------")

    for edge in snapshot["edges"][:5]:
        print(edge)

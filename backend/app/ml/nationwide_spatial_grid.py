from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, List, Tuple


# Approximate geographic bounds covering mainland India.
# These are intentionally broad so the grid can later be extended
# with an official India boundary mask.
INDIA_BOUNDS = {
    "min_latitude": 8.0,
    "max_latitude": 37.5,
    "min_longitude": 68.0,
    "max_longitude": 97.5,
}


@dataclass(frozen=True)
class GridNode:
    """
    Represents one geographic grid cell/node.

    This structure is intentionally simple so it can later be
    converted directly into a graph representation for GNNs.
    """

    node_id: str
    row: int
    column: int
    latitude: float
    longitude: float


def validate_grid_size(grid_size: float) -> None:
    """Validate the geographic grid spacing."""

    if not isinstance(grid_size, (int, float)):
        raise TypeError("grid_size must be a number.")

    if grid_size <= 0:
        raise ValueError("grid_size must be greater than zero.")

    if grid_size > 5:
        raise ValueError(
            "grid_size is too large. Use a value <= 5 degrees."
        )


def build_spatial_grid(
    grid_size: float = 1.0,
) -> List[GridNode]:
    """
    Build a regular latitude/longitude grid covering India.

    Parameters
    ----------
    grid_size:
        Grid spacing in degrees.

        Examples:
        1.0 -> coarse research grid
        0.5 -> finer grid
        0.25 -> much finer grid

    Returns
    -------
    List[GridNode]
        List of geographic grid nodes.
    """

    validate_grid_size(grid_size)

    min_lat = INDIA_BOUNDS["min_latitude"]
    max_lat = INDIA_BOUNDS["max_latitude"]
    min_lon = INDIA_BOUNDS["min_longitude"]
    max_lon = INDIA_BOUNDS["max_longitude"]

    nodes: List[GridNode] = []

    row = 0
    latitude = min_lat

    while latitude <= max_lat + 1e-9:
        column = 0
        longitude = min_lon

        while longitude <= max_lon + 1e-9:
            node_id = f"GRID_{row:03d}_{column:03d}"

            nodes.append(
                GridNode(
                    node_id=node_id,
                    row=row,
                    column=column,
                    latitude=round(latitude, 6),
                    longitude=round(longitude, 6),
                )
            )

            column += 1
            longitude = min_lon + column * grid_size

        row += 1
        latitude = min_lat + row * grid_size

    return nodes


def build_grid_index(
    nodes: List[GridNode],
) -> Dict[str, GridNode]:
    """
    Convert grid nodes into a fast node_id -> node lookup.
    """

    return {
        node.node_id: node
        for node in nodes
    }


def build_neighbor_map(
    nodes: List[GridNode],
) -> Dict[str, List[str]]:
    """
    Build 8-connected spatial neighbors for every grid node.

    Neighbor directions:

        NW   N   NE
         W   X    E
        SW   S   SE

    This adjacency structure is the initial graph topology
    that can later be supplied to a GNN.
    """

    if not nodes:
        return {}

    lookup = {
        (node.row, node.column): node.node_id
        for node in nodes
    }

    neighbors: Dict[str, List[str]] = {}

    directions: List[Tuple[int, int]] = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for node in nodes:
        node_neighbors: List[str] = []

        for row_offset, column_offset in directions:
            key = (
                node.row + row_offset,
                node.column + column_offset,
            )

            neighbor_id = lookup.get(key)

            if neighbor_id is not None:
                node_neighbors.append(neighbor_id)

        neighbors[node.node_id] = node_neighbors

    return neighbors


def estimate_grid_dimensions(
    grid_size: float = 1.0,
) -> Dict[str, int]:
    """
    Estimate number of rows and columns in the grid.
    """

    validate_grid_size(grid_size)

    latitude_range = (
        INDIA_BOUNDS["max_latitude"]
        - INDIA_BOUNDS["min_latitude"]
    )

    longitude_range = (
        INDIA_BOUNDS["max_longitude"]
        - INDIA_BOUNDS["min_longitude"]
    )

    rows = int(round(latitude_range / grid_size)) + 1
    columns = int(round(longitude_range / grid_size)) + 1

    return {
        "rows": rows,
        "columns": columns,
        "total_nodes": rows * columns,
    }


def build_graph_structure(
    grid_size: float = 1.0,
) -> Dict:
    """
    Build the complete initial spatial graph representation.

    This is GNN-ready topology, but it is NOT a GNN model yet.
    """

    nodes = build_spatial_grid(grid_size)
    neighbors = build_neighbor_map(nodes)

    edges: List[Tuple[str, str]] = []

    for source_node, target_nodes in neighbors.items():
        for target_node in target_nodes:
            edges.append((source_node, target_node))

    return {
        "grid_size_degrees": grid_size,
        "bounds": INDIA_BOUNDS,
        "nodes": nodes,
        "neighbors": neighbors,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "graph_type": "8-connected geographic grid",
        "gnn_ready": True,
        "gnn_model": None,
    }


if __name__ == "__main__":
    graph = build_graph_structure(grid_size=1.0)

    print("VATAVARAN Nationwide Spatial Grid")
    print("---------------------------------")
    print(f"Grid size: {graph['grid_size_degrees']} degrees")
    print(f"Nodes: {graph['node_count']}")
    print(f"Edges: {graph['edge_count']}")
    print(f"Graph type: {graph['graph_type']}")
    print(f"GNN ready: {graph['gnn_ready']}")

    print("\nFirst 5 nodes:")

    for node in graph["nodes"][:5]:
        print(
            node.node_id,
            "| lat:", node.latitude,
            "| lon:", node.longitude,
        )

    first_node = graph["nodes"][0]

    print("\nNeighbors of", first_node.node_id)
    print(graph["neighbors"][first_node.node_id])
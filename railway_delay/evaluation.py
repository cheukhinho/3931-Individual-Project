"""
evaluation.py
=============
Metrics and visualisation helpers for the delay propagation and
optimisation pipeline.

Metrics
-------
total_delay_minutes
    Sum of all simulated departure delays across the network.

delay_propagation_depth
    Maximum shortest-path length (in hops) from any initially disrupted node
    to the furthest affected node (departure delay > 0).

affected_services_count
    Number of distinct services with at least one node whose simulated
    departure delay exceeds a threshold.

Visualisation
-------------
:func:`plot_delay_histogram`
    Histogram of per-node departure delays.

:func:`plot_delay_over_time`
    Scatter plot of departure delay vs. scheduled departure time.

:func:`plot_network_delay`
    Node-link diagram of the temporal graph with nodes coloured by delay.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def total_delay_minutes(G: nx.DiGraph, *, exclude_inf: bool = True) -> float:
    """Sum of simulated departure delays across all nodes.

    Parameters
    ----------
    G:
        Temporal graph after running :func:`~railway_delay.simulation.propagate_delays`.
    exclude_inf:
        Exclude ``float('inf')`` values (cancelled services) from the sum.

    Returns
    -------
    float
    """
    total = 0.0
    for _, attrs in G.nodes(data=True):
        d = attrs.get("simulated_departure_delay", 0.0)
        if exclude_inf and d == float("inf"):
            continue
        total += d
    return total


def affected_services(
    G: nx.DiGraph, threshold_min: float = 1.0
) -> set[str]:
    """Return the set of service IDs with any node delayed beyond *threshold_min*.

    Parameters
    ----------
    G:
        Temporal graph.
    threshold_min:
        Minimum departure delay (minutes) to count a node as affected.

    Returns
    -------
    set of str
    """
    services: set[str] = set()
    for _, attrs in G.nodes(data=True):
        d = attrs.get("simulated_departure_delay", 0.0)
        if d >= threshold_min:
            svc = attrs.get("service_id", "")
            if svc:
                services.add(svc)
    return services


def affected_services_count(G: nx.DiGraph, threshold_min: float = 1.0) -> int:
    """Count distinct services with departure delay ≥ *threshold_min*.

    Parameters
    ----------
    G:
        Temporal graph.
    threshold_min:
        Minimum delay threshold (minutes).

    Returns
    -------
    int
    """
    return len(affected_services(G, threshold_min))


def delay_propagation_depth(
    G: nx.DiGraph,
    source_nodes: Optional[list[str]] = None,
    threshold_min: float = 1.0,
) -> int:
    """Compute the maximum number of hops delay has propagated from source nodes.

    Parameters
    ----------
    G:
        Temporal graph.
    source_nodes:
        Nodes that were directly disrupted.  If ``None``, all nodes with
        departure delay > 0 are treated as potential sources.
    threshold_min:
        Minimum delay to count a node as "affected".

    Returns
    -------
    int
        Maximum BFS depth from any source to any affected node.
    """
    if source_nodes is None:
        source_nodes = [
            n
            for n, attrs in G.nodes(data=True)
            if attrs.get("simulated_departure_delay", 0.0) >= threshold_min
        ]

    if not source_nodes:
        return 0

    affected = {
        n
        for n, attrs in G.nodes(data=True)
        if attrs.get("simulated_departure_delay", 0.0) >= threshold_min
    }

    max_depth = 0
    for src in source_nodes:
        try:
            lengths = nx.single_source_shortest_path_length(G, src)
        except nx.NetworkXError:
            continue
        for node, depth in lengths.items():
            if node in affected:
                max_depth = max(max_depth, depth)

    return max_depth


def compute_metrics(
    G: nx.DiGraph,
    source_nodes: Optional[list[str]] = None,
    threshold_min: float = 1.0,
) -> dict[str, Any]:
    """Compute all standard metrics in one call.

    Parameters
    ----------
    G:
        Temporal graph after propagation.
    source_nodes:
        Directly disrupted nodes (used for propagation depth).
    threshold_min:
        Minimum delay threshold.

    Returns
    -------
    dict with keys:
        ``"total_delay_minutes"``,
        ``"affected_services_count"``,
        ``"delay_propagation_depth"``,
        ``"affected_services"`` (set of service IDs).
    """
    return {
        "total_delay_minutes": total_delay_minutes(G),
        "affected_services_count": affected_services_count(G, threshold_min),
        "delay_propagation_depth": delay_propagation_depth(G, source_nodes, threshold_min),
        "affected_services": affected_services(G, threshold_min),
    }


def delays_to_dataframe(G: nx.DiGraph) -> pd.DataFrame:
    """Extract node-level delay information into a DataFrame.

    Parameters
    ----------
    G:
        Temporal graph.

    Returns
    -------
    pd.DataFrame
        Columns: ``node_key``, ``service_id``, ``station_crs``,
        ``scheduled_departure``, ``simulated_departure_delay``,
        ``simulated_arrival_delay``.
    """
    rows = []
    for node, attrs in G.nodes(data=True):
        rows.append(
            {
                "node_key": node,
                "service_id": attrs.get("service_id", ""),
                "station_crs": attrs.get("station_crs", ""),
                "scheduled_departure": attrs.get("scheduled_departure"),
                "simulated_departure_delay": attrs.get("simulated_departure_delay", 0.0),
                "simulated_arrival_delay": attrs.get("simulated_arrival_delay", 0.0),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualisation helpers (require matplotlib)
# ---------------------------------------------------------------------------


def plot_delay_histogram(G: nx.DiGraph, *, ax=None, title: str = "Departure Delay Distribution"):
    """Plot a histogram of per-node departure delays.

    Parameters
    ----------
    G:
        Temporal graph after propagation.
    ax:
        Matplotlib ``Axes`` object.  If ``None``, a new figure is created.
    title:
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    delays = [
        attrs.get("simulated_departure_delay", 0.0)
        for _, attrs in G.nodes(data=True)
        if attrs.get("simulated_departure_delay", 0.0) != float("inf")
    ]
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(delays, bins=20, edgecolor="black")
    ax.set_xlabel("Departure delay (minutes)")
    ax.set_ylabel("Number of nodes")
    ax.set_title(title)
    return ax


def plot_delay_over_time(
    G: nx.DiGraph,
    *,
    ax=None,
    title: str = "Departure Delay vs Scheduled Departure Time",
):
    """Scatter plot of departure delay vs. scheduled departure time.

    Parameters
    ----------
    G:
        Temporal graph after propagation.
    ax:
        Matplotlib ``Axes`` object.
    title:
        Plot title.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    df = delays_to_dataframe(G)
    df = df[df["simulated_departure_delay"] != float("inf")]
    df = df.dropna(subset=["scheduled_departure"])

    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(
        pd.to_datetime(df["scheduled_departure"]),
        df["simulated_departure_delay"],
        alpha=0.6,
        s=20,
    )
    ax.set_xlabel("Scheduled departure")
    ax.set_ylabel("Departure delay (minutes)")
    ax.set_title(title)
    return ax


def plot_network_delay(
    G: nx.DiGraph,
    *,
    ax=None,
    title: str = "Network Delay Map",
    layout_seed: int = 42,
):
    """Draw the temporal graph with nodes coloured by departure delay.

    Parameters
    ----------
    G:
        Temporal graph after propagation.
    ax:
        Matplotlib ``Axes`` object.
    title:
        Plot title.
    layout_seed:
        Random seed for the spring layout.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415
    import matplotlib.colors as mcolors  # noqa: PLC0415

    delays = [
        min(attrs.get("simulated_departure_delay", 0.0), 60.0)
        for _, attrs in G.nodes(data=True)
    ]
    norm = mcolors.Normalize(vmin=0, vmax=max(delays) if delays else 1)
    cmap = plt.cm.YlOrRd  # type: ignore[attr-defined]
    node_colors = [cmap(norm(d)) for d in delays]

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 7))

    pos = nx.spring_layout(G, seed=layout_seed)
    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_colors,
        with_labels=False,
        node_size=40,
        arrows=True,
        ax=ax,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # type: ignore[attr-defined]
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Departure delay (min)")
    ax.set_title(title)
    return ax

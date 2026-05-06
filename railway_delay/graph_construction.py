"""
graph_construction.py
=====================
Build a *temporal graph* from processed railway data.

Graph model
-----------
Node
    ``(service_id, station_crs, departure_time)`` – a *departure event*.
    The node key is a string ``"<service_id>|<crs>|<departure_iso>"``.

    Node attributes:
    - ``service_id``   – service UID
    - ``station_crs``  – three-letter CRS code
    - ``departure_time`` – scheduled departure (``datetime``)
    - ``arrival_time``   – scheduled arrival at this stop (``datetime``)
    - ``scheduled_departure``
    - ``actual_departure``
    - ``scheduled_arrival``
    - ``actual_arrival``
    - ``departure_delay_min``
    - ``arrival_delay_min``

Edge (movement edge)
    Connects consecutive departure events within the **same service**.
    Source → Destination along the route.

    Edge attributes:
    - ``edge_type``        – ``"movement"``
    - ``service_id``
    - ``scheduled_travel_min``
    - ``actual_travel_min``

Edge (dependency / connection edge)  [Phase 3]
    Connects an *arrival event* at station X to a *departure event* at the
    same station X for a **different service** whose scheduled departure
    falls within a configurable window after the arrival.  This captures
    cross-platform passenger connections and crew/rolling-stock turn-rounds.

    Two flavours:

    *Generic dependency* (``add_dependency_edges``):
        Time-window match only.  Edge attributes:
        - ``edge_type``     – ``"dependency"``
        - ``turnaround_min``

    *Turnaround dependency* (``add_dependency_edges_turnaround``):
        Hierarchical confidence model based on rolling-stock reuse.
        HIGH confidence when a ``unit_id`` column is present and the
        same unit is seen on both sides of the turn; MEDIUM confidence
        when the ``unit_id`` is absent or unknown and matching falls back
        to a time-proximity heuristic.  Platform equality (if available)
        can promote a medium edge to high.  Edge attributes:
        - ``edge_type``   – ``"dependency"``
        - ``subtype``     – ``"turnaround"``
        - ``time_diff``   – gap in minutes (arrival → departure)
        - ``confidence``  – ``"high"`` or ``"medium"``

Edge (interaction edge)  [Phase 2]
    Connects any two nodes at the same station whose temporal anchors
    (``scheduled_departure``, falling back to ``scheduled_arrival``) are
    within a short configurable window of each other and belong to
    **different services**.  The edge is directed from the earlier event to
    the later one.  This models proximity conflicts – two trains sharing the
    same station area in quick succession may propagate delays to each other.
    No platform data is required.

    Edge attributes:
    - ``edge_type`` – ``"interaction"``
    - ``gap_min``   – temporal gap between the two anchor times (minutes)

This structure directly supports delay propagation: a delay on node A flows
forward along outgoing edges to downstream nodes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Node-key helper
# ---------------------------------------------------------------------------


def node_key(service_id: str, station_crs: str, departure_time: datetime) -> str:
    """Return a deterministic string key for a departure event node.

    Parameters
    ----------
    service_id:
        Service UID.
    station_crs:
        Three-letter CRS station code.
    departure_time:
        Scheduled departure ``datetime``.
    """
    ts = departure_time.isoformat() if departure_time else "?"
    return f"{service_id}|{station_crs}|{ts}"


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def build_movement_graph(stops_df: pd.DataFrame) -> nx.DiGraph:
    """Create a directed temporal graph of intra-service movement edges.

    Each consecutive pair of stops for the same service produces:
    - A *departure* node for the origin stop.
    - A *departure* node for the destination stop.
    - A directed *movement* edge from origin to destination.

    The terminal stop of each service is represented as a node with its
    scheduled **arrival** time stored in ``departure_time`` (there is no
    outbound movement from the last stop).

    Parameters
    ----------
    stops_df:
        Cleaned stops table from :func:`~railway_delay.data_processing.build_stops_table`.

    Returns
    -------
    nx.DiGraph
    """
    G = nx.DiGraph()

    for (service_id, run_date), grp in stops_df.groupby(
        ["service_id", "run_date"], sort=False
    ):
        grp = grp.sort_values("stop_index").reset_index(drop=True)

        for i, row in grp.iterrows():
            # Use departure time as the node's temporal anchor; fall back to
            # arrival time for the terminal stop.
            t_dep = row["scheduled_departure"]
            t_arr = row["scheduled_arrival"]
            anchor = t_dep if not pd.isnull(t_dep) else t_arr
            if pd.isnull(anchor):
                continue

            key = node_key(service_id, row["station_crs"], anchor)
            G.add_node(
                key,
                service_id=service_id,
                run_date=run_date,
                station_crs=row["station_crs"],
                station_name=row.get("station_name", ""),
                stop_index=int(row["stop_index"]),
                scheduled_departure=t_dep,
                actual_departure=row.get("actual_departure"),
                scheduled_arrival=t_arr,
                actual_arrival=row.get("actual_arrival"),
                departure_delay_min=row.get("departure_delay_min"),
                arrival_delay_min=row.get("arrival_delay_min"),
                # Mutable simulation state
                simulated_departure_delay=0.0,
                simulated_arrival_delay=0.0,
            )

        # Add movement edges between consecutive stops
        for i in range(len(grp) - 1):
            src_row = grp.iloc[i]
            dst_row = grp.iloc[i + 1]

            t_dep_src = src_row["scheduled_departure"]
            if pd.isnull(t_dep_src):
                continue
            t_dep_dst = dst_row["scheduled_departure"]
            t_arr_dst = dst_row["scheduled_arrival"]
            anchor_dst = t_dep_dst if not pd.isnull(t_dep_dst) else t_arr_dst
            if pd.isnull(anchor_dst):
                continue

            src_key = node_key(service_id, src_row["station_crs"], t_dep_src)
            dst_key = node_key(service_id, dst_row["station_crs"], anchor_dst)

            if src_key not in G or dst_key not in G:
                continue

            sched_travel = _safe_minutes(t_dep_src, t_arr_dst)
            act_travel = _safe_minutes(
                src_row.get("actual_departure"), dst_row.get("actual_arrival")
            )

            G.add_edge(
                src_key,
                dst_key,
                edge_type="movement",
                service_id=service_id,
                scheduled_travel_min=sched_travel,
                actual_travel_min=act_travel,
            )

    return G


def add_dependency_edges(
    G: nx.DiGraph,
    stops_df: pd.DataFrame,
    min_connection_min: float = 2.0,
    max_connection_min: float = 30.0,
) -> nx.DiGraph:
    """Add cross-service *dependency* edges to an existing movement graph.

    A dependency edge is added from the arrival node of service A at station X
    to the departure node of service B at station X when B's scheduled
    departure is between *min_connection_min* and *max_connection_min* after
    A's scheduled arrival.  This models connecting passengers or
    crew/rolling-stock transfers.

    Parameters
    ----------
    G:
        Directed graph produced by :func:`build_movement_graph`.
    stops_df:
        Cleaned stops table.
    min_connection_min:
        Minimum gap (minutes) for a dependency to be created.
    max_connection_min:
        Maximum gap (minutes) for a dependency to be created.

    Returns
    -------
    nx.DiGraph
        The same graph object with dependency edges added in-place.
    """
    # Index nodes by station for fast lookup
    station_nodes: dict[str, list[str]] = {}
    for nkey, attrs in G.nodes(data=True):
        crs = attrs.get("station_crs", "")
        station_nodes.setdefault(crs, []).append(nkey)

    for crs, node_keys in station_nodes.items():
        # Separate arrivals (terminal nodes / nodes with arrival times) from
        # departures (nodes with departure times from this station)
        arrivals = [
            (nkey, G.nodes[nkey]["scheduled_arrival"])
            for nkey in node_keys
            if not pd.isnull(G.nodes[nkey].get("scheduled_arrival"))
        ]
        departures = [
            (nkey, G.nodes[nkey]["scheduled_departure"])
            for nkey in node_keys
            if not pd.isnull(G.nodes[nkey].get("scheduled_departure"))
        ]

        for arr_key, arr_time in arrivals:
            arr_svc = G.nodes[arr_key]["service_id"]
            for dep_key, dep_time in departures:
                dep_svc = G.nodes[dep_key]["service_id"]
                if dep_svc == arr_svc:
                    continue  # same service – already a movement edge
                if pd.isnull(arr_time) or pd.isnull(dep_time):
                    continue
                gap = _safe_minutes(arr_time, dep_time)
                if gap is None:
                    continue
                if min_connection_min <= gap <= max_connection_min:
                    G.add_edge(
                        arr_key,
                        dep_key,
                        edge_type="dependency",
                        turnaround_min=gap,
                    )

    return G


def add_dependency_edges_turnaround(
    stops_df: pd.DataFrame,
    min_turnaround_min: float = 5.0,
    max_turnaround_high_min: float = 45.0,
    max_turnaround_medium_min: float = 30.0,
) -> list[dict]:
    """Detect rolling-stock turnaround events and return dependency edge descriptors.

    Uses a hierarchical detection strategy:

    1. **HIGH confidence** (fleet heuristic): when *both* the arrival and
       departure rows carry the same non-null ``unit_id``, and the gap between
       the scheduled arrival and scheduled departure falls within
       [*min_turnaround_min*, *max_turnaround_high_min*].

    2. **MEDIUM confidence** (time-based fallback): when ``unit_id`` data is
       absent or null for either event, a turnaround is inferred from time
       proximity alone – gap within [*min_turnaround_min*,
       *max_turnaround_medium_min*].

    3. **Platform refinement** (optional promotion): if both events share the
       same non-null ``platform`` value, a MEDIUM edge is promoted to HIGH.
       Platform equality is never a strict filter – it only raises confidence.

    The function groups events by station to avoid O(n²) global comparisons.

    Parameters
    ----------
    stops_df:
        Stops DataFrame.  Required columns: ``service_id``, ``station_crs``,
        ``scheduled_arrival``, ``scheduled_departure``.  Optional columns:
        ``unit_id``, ``platform``.
    min_turnaround_min:
        Minimum gap (minutes) between arrival and departure.  Defaults to
        ``5.0``.
    max_turnaround_high_min:
        Maximum gap (minutes) for a high-confidence (unit-match) turnaround.
        Defaults to ``45.0``.
    max_turnaround_medium_min:
        Maximum gap (minutes) for a medium-confidence (time-based) turnaround.
        Defaults to ``30.0``.

    Returns
    -------
    list[dict]
        Each entry is an edge descriptor::

            {
                "from":       <node_key str>,   # arrival event node
                "to":         <node_key str>,   # departure event node
                "type":       "dependency",
                "subtype":    "turnaround",
                "time_diff":  <float minutes>,
                "confidence": "high" | "medium",
            }

        Node keys follow the ``node_key()`` convention:
        arrival events are keyed by ``scheduled_arrival``;
        departure events are keyed by ``scheduled_departure``.
    """
    has_unit = "unit_id" in stops_df.columns
    has_platform = "platform" in stops_df.columns

    edges: list[dict] = []

    for crs, grp in stops_df.groupby("station_crs", sort=False):
        arr_events = grp[grp["scheduled_arrival"].notna()]
        dep_events = grp[grp["scheduled_departure"].notna()]

        if arr_events.empty or dep_events.empty:
            continue

        for _, arr_row in arr_events.iterrows():
            arr_svc = arr_row["service_id"]
            arr_time = pd.Timestamp(arr_row["scheduled_arrival"])
            arr_unit = arr_row["unit_id"] if has_unit else None
            arr_platform = arr_row["platform"] if has_platform else None

            arr_key = node_key(arr_svc, str(crs), arr_time.to_pydatetime())

            for _, dep_row in dep_events.iterrows():
                dep_svc = dep_row["service_id"]
                if dep_svc == arr_svc:
                    continue  # same service – already covered by movement edges

                dep_time = pd.Timestamp(dep_row["scheduled_departure"])
                dep_unit = dep_row["unit_id"] if has_unit else None
                dep_platform = dep_row["platform"] if has_platform else None

                gap = (dep_time - arr_time).total_seconds() / 60.0

                # Determine whether both sides have usable unit_id data
                both_have_unit = (
                    has_unit
                    and arr_unit is not None
                    and dep_unit is not None
                    and not pd.isnull(arr_unit)
                    and not pd.isnull(dep_unit)
                )

                if both_have_unit:
                    # Strict fleet matching: units must agree and gap must fit
                    if (
                        arr_unit == dep_unit
                        and min_turnaround_min <= gap <= max_turnaround_high_min
                    ):
                        confidence: str = "high"
                    else:
                        continue  # units disagree or gap out of range
                elif min_turnaround_min <= gap <= max_turnaround_medium_min:
                    # Time-based fallback when unit_id is unavailable
                    confidence = "medium"
                else:
                    continue  # gap outside time-based window

                dep_key = node_key(dep_svc, str(crs), dep_time.to_pydatetime())

                # Platform refinement: promote medium → high when platforms match
                if (
                    confidence == "medium"
                    and has_platform
                    and arr_platform is not None
                    and dep_platform is not None
                    and not pd.isnull(arr_platform)
                    and not pd.isnull(dep_platform)
                    and arr_platform == dep_platform
                ):
                    confidence = "high"

                edges.append(
                    {
                        "from": arr_key,
                        "to": dep_key,
                        "type": "dependency",
                        "subtype": "turnaround",
                        "time_diff": round(gap, 2),
                        "confidence": confidence,
                    }
                )

    return edges


def add_turnaround_edges(
    G: nx.DiGraph,
    stops_df: pd.DataFrame,
    min_turnaround_min: float = 5.0,
    max_turnaround_high_min: float = 45.0,
    max_turnaround_medium_min: float = 30.0,
) -> nx.DiGraph:
    """Add rolling-stock turnaround *dependency* edges to an existing graph.

    Calls :func:`add_dependency_edges_turnaround` to detect turnaround pairs
    and inserts each resulting edge into *G*.  Only edges whose both endpoint
    nodes already exist in the graph are added; the rest are silently skipped.

    Parameters
    ----------
    G:
        Directed graph produced by :func:`build_movement_graph`.
    stops_df:
        Stops DataFrame passed through to
        :func:`add_dependency_edges_turnaround`.
    min_turnaround_min:
        Minimum gap (minutes) for a turnaround.
    max_turnaround_high_min:
        Maximum gap for high-confidence (unit-match) turnarounds.
    max_turnaround_medium_min:
        Maximum gap for medium-confidence (time-based) turnarounds.

    Returns
    -------
    nx.DiGraph
        The same graph with turnaround dependency edges added in-place.
    """
    edge_dicts = add_dependency_edges_turnaround(
        stops_df,
        min_turnaround_min=min_turnaround_min,
        max_turnaround_high_min=max_turnaround_high_min,
        max_turnaround_medium_min=max_turnaround_medium_min,
    )
    for ed in edge_dicts:
        from_key = ed["from"]
        to_key = ed["to"]
        if from_key not in G or to_key not in G:
            continue
        G.add_edge(
            from_key,
            to_key,
            edge_type=ed["type"],
            subtype=ed["subtype"],
            time_diff=ed["time_diff"],
            confidence=ed["confidence"],
        )
    return G


def add_interaction_edges(
    G: nx.DiGraph,
    max_interaction_min: float = 5.0,
) -> nx.DiGraph:
    """Add cross-service *interaction* edges for Phase 2 proximity conflicts.

    For every station, all nodes are sorted by their temporal anchor
    (``scheduled_departure``, falling back to ``scheduled_arrival``).
    Any two nodes from **different services** whose anchors are within
    *max_interaction_min* of each other receive a directed edge from the
    earlier event to the later one.

    This models unintended proximity conflicts: two trains occupying the same
    station area in quick succession may propagate delays to each other.
    Unlike :func:`add_dependency_edges`, no minimum gap is enforced and no
    platform data is needed.

    Parameters
    ----------
    G:
        Directed graph produced by :func:`build_movement_graph`.
    max_interaction_min:
        Maximum gap (minutes) between two event anchors for an interaction
        edge to be created.  Defaults to ``5.0`` minutes.

    Returns
    -------
    nx.DiGraph
        The same graph object with interaction edges added in-place.
    """
    # Group nodes by station CRS
    station_nodes: dict[str, list[str]] = {}
    for nkey, attrs in G.nodes(data=True):
        crs = attrs.get("station_crs", "")
        station_nodes.setdefault(crs, []).append(nkey)

    for crs, keys in station_nodes.items():
        # Build a list of (node_key, anchor_timestamp, service_id)
        events: list[tuple[str, pd.Timestamp, str]] = []
        for nkey in keys:
            attrs = G.nodes[nkey]
            t_dep = attrs.get("scheduled_departure")
            t_arr = attrs.get("scheduled_arrival")
            anchor = t_dep if (t_dep is not None and not pd.isnull(t_dep)) else t_arr
            if anchor is None or pd.isnull(anchor):
                continue
            events.append((nkey, pd.Timestamp(anchor), attrs["service_id"]))

        # Sort chronologically so the inner loop can break early
        events.sort(key=lambda x: x[1])

        for i in range(len(events)):
            nk_a, t_a, svc_a = events[i]
            for j in range(i + 1, len(events)):
                nk_b, t_b, svc_b = events[j]
                gap = (t_b - t_a).total_seconds() / 60.0
                if gap > max_interaction_min:
                    break  # sorted order means no further pair can be within window
                if svc_a == svc_b:
                    continue  # same service – movement edge already covers this
                G.add_edge(nk_a, nk_b, edge_type="interaction", gap_min=gap)

    return G


def build_temporal_graph(
    stops_df: pd.DataFrame,
    add_dependencies: bool = True,
    min_connection_min: float = 2.0,
    max_connection_min: float = 30.0,
    add_interactions: bool = False,
    max_interaction_min: float = 5.0,
    add_turnarounds: bool = False,
    min_turnaround_min: float = 5.0,
    max_turnaround_high_min: float = 45.0,
    max_turnaround_medium_min: float = 30.0,
) -> nx.DiGraph:
    """End-to-end graph construction from a stops DataFrame.

    Parameters
    ----------
    stops_df:
        Cleaned stops table from :mod:`data_processing`.
    add_dependencies:
        Whether to add cross-service dependency (connection) edges.
    min_connection_min:
        Minimum connection window for dependency edges (minutes).
    max_connection_min:
        Maximum connection window for dependency edges (minutes).
    add_interactions:
        Whether to add Phase 2 proximity-conflict interaction edges.
    max_interaction_min:
        Maximum gap (minutes) between two event anchors for an interaction
        edge to be created.
    add_turnarounds:
        Whether to add Phase 3 rolling-stock turnaround dependency edges.
    min_turnaround_min:
        Minimum gap (minutes) for a turnaround dependency edge.
    max_turnaround_high_min:
        Maximum gap (minutes) for a high-confidence turnaround.
    max_turnaround_medium_min:
        Maximum gap (minutes) for a medium-confidence turnaround.

    Returns
    -------
    nx.DiGraph
        Temporal graph ready for simulation.
    """
    G = build_movement_graph(stops_df)
    if add_dependencies:
        G = add_dependency_edges(G, stops_df, min_connection_min, max_connection_min)
    if add_interactions:
        G = add_interaction_edges(G, max_interaction_min)
    if add_turnarounds:
        G = add_turnaround_edges(
            G,
            stops_df,
            min_turnaround_min=min_turnaround_min,
            max_turnaround_high_min=max_turnaround_high_min,
            max_turnaround_medium_min=max_turnaround_medium_min,
        )
    return G


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_minutes(t1, t2) -> Optional[float]:
    """Return ``(t2 − t1)`` in minutes, handling None / NaT safely."""
    try:
        if t1 is None or t2 is None:
            return None
        if pd.isnull(t1) or pd.isnull(t2):
            return None
        delta = pd.Timestamp(t2) - pd.Timestamp(t1)
        return delta.total_seconds() / 60.0
    except (TypeError, ValueError):
        return None

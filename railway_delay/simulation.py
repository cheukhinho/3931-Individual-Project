"""
simulation.py
=============
Rule-based delay propagation on a temporal graph.

Propagation rules
-----------------
1. **Late arrival → late departure** (same service, same stop)
   If a train arrives late at a station, its departure is delayed by at least
   the same amount (minus the minimum dwell time, floored at 0).

2. **Movement edges** – same train continuing along its route.
   The upstream departure delay flows to the next stop's arrival, minus a
   (potentially variable) recovery time.  When ``recovery_time_min > 0``,
   longer segments (``time_diff ≥ 10 min``) use 1.5× the base recovery and
   shorter segments use 0.5× the base recovery::

       factor = 1.5 if time_diff ≥ 10 min else 0.5
       effective_recovery = recovery_time_min * factor
       delay_next = max(0, delay_current − effective_recovery)

3. **Dependency edges** – rolling-stock turnaround or crew transfer.
   The scheduled buffer absorbs part of the delay.  Only the portion that
   exceeds the buffer propagates::

       buffer = scheduled_departure_B − scheduled_arrival_A
       delay_B = max(0, delay_A − buffer)

   **Cascading failure**: if ``delay_A > buffer + CASCADE_THRESHOLD_MIN``,
   the downstream service is penalised with ``CASCADE_LARGE_DELAY_MIN``
   (60 min) to model a severely disrupted turnaround.

4. **Interaction edges** – proximity conflicts at the same station.
   The decision between two orderings uses a **future-cost-aware** comparison
   to avoid the local-only myopia of comparing only immediate node costs.
   Both scenarios are treated symmetrically (same α-transfer mechanism):

   - **A first**: A departs first; B absorbs ``α × delay_A``.
   - **B first**: B is given priority; A incurs a holding penalty of
     ``max(INTERACTION_HOLDING_MIN, α × delay_B)``.

   Total cost per option::

       total_cost = local_cost + threshold_surcharge + β × future_cost

   where:
   - *local_cost* = C(A) + C(B) using the updated delays
   - *threshold_surcharge* = penalty for crossing a 15/30/60-min milestone
   - *future_cost* = α × delay_source × AVG_COST_PER_MINUTE
   - *β* = FUTURE_COST_BETA (0.3)

   The option with the lower *total_cost* is chosen.  If B-first is selected,
   the holding is applied to A's departure delay **before** movement edges are
   processed (two-pass edge loop), so the increased delay propagates correctly
   to A's downstream stops.

The simulation updates two mutable attributes on every node:
  ``simulated_departure_delay``  – minutes late at departure
  ``simulated_arrival_delay``    – minutes late at arrival

The graph is processed in **topological order** (chronological order as a
fallback) so every upstream node is fully resolved before its downstream
neighbours.
"""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import pandas as pd

from railway_delay import config as _cfg
from railway_delay import cost as _cost

logger = logging.getLogger(__name__)

# Module-level defaults drawn from centralised config so callers that do
# not pass explicit parameters always use the canonical values.
DEFAULT_MIN_DWELL_MIN: float = _cfg.MIN_DWELL_MIN
DEFAULT_MAX_INTERACTION_MIN: float = _cfg.MAX_INTERACTION_MIN
DEFAULT_ALPHA: float = _cfg.ALPHA
DEFAULT_RECOVERY_TIME_MIN: float = _cfg.RECOVERY_TIME_MIN

#: Segment travel-time threshold (minutes) above which a movement segment is
#: considered "long" and receives a proportionally larger recovery benefit.
_LONG_SEGMENT_MIN: float = 10.0


def reset_delays(G: nx.DiGraph) -> None:
    """Reset all simulated delay attributes to zero.

    Parameters
    ----------
    G:
        Temporal graph produced by :mod:`graph_construction`.
    """
    for node in G.nodes:
        G.nodes[node]["simulated_departure_delay"] = 0.0
        G.nodes[node]["simulated_arrival_delay"] = 0.0


def inject_delay(
    G: nx.DiGraph,
    node_key: str,
    delay_minutes: float,
    *,
    affect_arrival: bool = True,
    affect_departure: bool = True,
) -> None:
    """Inject an initial delay onto a specific node.

    Parameters
    ----------
    G:
        Temporal graph.
    node_key:
        Node identifier.
    delay_minutes:
        Delay magnitude in minutes.
    affect_arrival:
        Whether to apply the delay to the node's arrival.
    affect_departure:
        Whether to apply the delay to the node's departure.
    """
    if node_key not in G:
        raise KeyError(f"Node {node_key!r} not found in graph.")
    if affect_arrival:
        G.nodes[node_key]["simulated_arrival_delay"] = float(delay_minutes)
    if affect_departure:
        G.nodes[node_key]["simulated_departure_delay"] = float(delay_minutes)


def propagate_delays(
    G: nx.DiGraph,
    min_dwell_min: float = DEFAULT_MIN_DWELL_MIN,
    min_connection_min: float = 2.0,
    alpha: float = DEFAULT_ALPHA,
    max_interaction_min: float = DEFAULT_MAX_INTERACTION_MIN,
    recovery_time_min: float = DEFAULT_RECOVERY_TIME_MIN,
) -> nx.DiGraph:
    """Run the full delay propagation simulation on the graph.

    The graph is traversed in topological order (chronological order as a
    fallback when cycles are present).  At each node the propagation rules
    are applied in sequence.

    Parameters
    ----------
    G:
        Temporal graph.  Node attributes ``simulated_departure_delay`` and
        ``simulated_arrival_delay`` must be present (call :func:`reset_delays`
        before injecting disruptions if needed).
    min_dwell_min:
        Minimum dwell time at a stop (minutes).  A late arrival cannot be
        recovered faster than this.
    min_connection_min:
        Minimum connection gap required across dependency edges when no
        scheduled times are available (minutes).
    alpha:
        Base interaction strength (0–1).  Used for interaction edges when
        *max_interaction_min* is zero or the edge carries no ``gap_min``.
    max_interaction_min:
        Maximum temporal gap (minutes) used in the improved alpha formula::

            alpha = max(0, 1 − gap_min / max_interaction_min)

        Set to 0 to use the constant *alpha* value instead.
    recovery_time_min:
        Recovery time applied on movement edges (minutes)::

            delay_next = max(0, delay_current − recovery_time_min)

        Defaults to 0 (no additional recovery beyond the dwell-time rule).

    Returns
    -------
    nx.DiGraph
        The same graph, with updated delay attributes.
    """
    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        logger.warning(
            "Graph contains cycles – falling back to chronological order for "
            "propagation."
        )
        order = _chronological_order(G)

    for node in order:
        attrs = G.nodes[node]
        arr_delay = attrs.get("simulated_arrival_delay", 0.0)

        # Rule 1: Late arrival → late departure (same service, same stop).
        # dep_delay = max(existing dep_delay, arr_delay − min_dwell_min, 0)
        # i.e. the departure is delayed by at least (arrival_delay − dwell),
        # meaning the dwell time partially absorbs the arrival delay.
        dep_delay = attrs.get("simulated_departure_delay", 0.0)
        dep_delay = max(dep_delay, arr_delay - min_dwell_min)
        dep_delay = max(dep_delay, 0.0)
        attrs["simulated_departure_delay"] = dep_delay

        # Skip propagation from cancelled nodes (delay == inf) – all their
        # edges should already have been removed, but guard here as well.
        if dep_delay == float("inf"):
            continue

        # ---------------------------------------------------------------
        # Pass 1: Interaction edges
        # These are processed first so that a B-first holding decision
        # updates this node's departure delay BEFORE movement/dependency
        # edges propagate it downstream.
        # ---------------------------------------------------------------
        for _, neighbour, edge_data in G.out_edges(node, data=True):
            if edge_data.get("edge_type") != "interaction":
                continue

            # Rule 4: Interaction – proximity conflict at the same station.
            gap_min = edge_data.get("gap_min", 0.0)
            if max_interaction_min > 0:
                effective_alpha = max(
                    0.0, 1.0 - gap_min / max_interaction_min
                )
            else:
                effective_alpha = alpha

            n_attrs = G.nodes[neighbour]
            delay_b_current = n_attrs.get("simulated_departure_delay", 0.0)

            # ── Symmetric holding for B-first ──────────────────────────
            # Both scenarios use the same α-transfer mechanism.
            # A-first: B gets  max(delay_B, α × delay_A)
            # B-first: A gets  delay_A + max(HOLDING_MIN, α × delay_B)
            holding = max(
                _cfg.INTERACTION_HOLDING_MIN,
                effective_alpha * delay_b_current,
            )

            delay_b_after_a_first = max(
                delay_b_current, effective_alpha * dep_delay
            )
            delay_a_after_b_first = dep_delay + holding

            # ── Local cost ─────────────────────────────────────────────
            local_a = (
                _cost.compute_node_cost(attrs, dep_delay)
                + _cost.compute_node_cost(n_attrs, delay_b_after_a_first)
            )
            local_b = (
                _cost.compute_node_cost(attrs, delay_a_after_b_first)
                + _cost.compute_node_cost(n_attrs, delay_b_current)
            )

            # ── Threshold-crossing surcharge ────────────────────────────
            # Penalise options that push a delay across a milestone
            # (15/30/60 min) to make decisions threshold-aware.
            surcharge_a = _cost.threshold_crossing_surcharge(
                delay_b_current, delay_b_after_a_first
            )
            surcharge_b = _cost.threshold_crossing_surcharge(
                dep_delay, delay_a_after_b_first
            )

            # ── Future cost approximation ───────────────────────────────
            # Approximate the downstream monetary cost of the delay that
            # will propagate from the chosen option:
            #   future_cost = β × propagated_delay × avg_cost_per_minute
            future_a = (
                _cfg.FUTURE_COST_BETA
                * (effective_alpha * dep_delay)
                * _cfg.AVG_COST_PER_MINUTE
            )
            future_b = (
                _cfg.FUTURE_COST_BETA
                * holding
                * _cfg.AVG_COST_PER_MINUTE
            )

            cost_a_first = local_a + surcharge_a + future_a
            cost_b_first = local_b + surcharge_b + future_b

            if cost_a_first <= cost_b_first:
                # A goes first: transfer a fraction of A's delay to B
                delay_transfer = effective_alpha * dep_delay
                if delay_transfer > 0:
                    current_arr_delay = n_attrs.get(
                        "simulated_arrival_delay", 0.0
                    )
                    n_attrs["simulated_arrival_delay"] = max(
                        current_arr_delay, delay_transfer
                    )
                logger.debug(
                    "Interaction %s → %s: A-first chosen "
                    "(cost_a=%.2f ≤ cost_b=%.2f); transfer=%.1f min",
                    node,
                    neighbour,
                    cost_a_first,
                    cost_b_first,
                    delay_transfer if delay_transfer > 0 else 0.0,
                )
            else:
                # B goes first: apply holding to A's departure delay so it
                # propagates correctly through A's movement edges (pass 2).
                dep_delay = dep_delay + holding
                attrs["simulated_departure_delay"] = dep_delay
                logger.debug(
                    "Interaction %s → %s: B-first chosen "
                    "(cost_b=%.2f < cost_a=%.2f); holding=%.1f min → "
                    "A dep_delay now %.1f min",
                    node,
                    neighbour,
                    cost_b_first,
                    cost_a_first,
                    holding,
                    dep_delay,
                )

        # Re-read dep_delay in case interaction pass modified it (B-first).
        dep_delay = attrs.get("simulated_departure_delay", 0.0)

        # ---------------------------------------------------------------
        # Pass 2: Movement and dependency edges
        # These use the final dep_delay (including any B-first holding).
        # ---------------------------------------------------------------
        for _, neighbour, edge_data in G.out_edges(node, data=True):
            edge_type = edge_data.get("edge_type", "movement")
            if edge_type == "interaction":
                continue  # already handled in pass 1

            n_attrs = G.nodes[neighbour]

            if edge_type == "movement":
                # Rule 2: Movement – same train continuing.
                # Apply variable recovery when recovery_time_min > 0:
                # longer segments recover more, shorter segments less.
                if recovery_time_min > 0:
                    time_diff = edge_data.get("time_diff", 0.0)
                    factor = 1.5 if time_diff >= _LONG_SEGMENT_MIN else 0.5
                    effective_recovery = recovery_time_min * factor
                else:
                    effective_recovery = 0.0
                recovered_delay = max(0.0, dep_delay - effective_recovery)
                current_arr_delay = n_attrs.get("simulated_arrival_delay", 0.0)
                n_attrs["simulated_arrival_delay"] = max(
                    current_arr_delay, recovered_delay
                )

            elif edge_type == "dependency":
                # Rule 3: Dependency – rolling-stock turnaround / connection.
                # Only the delay that exceeds the scheduled buffer propagates.
                sched_dep = n_attrs.get("scheduled_departure")
                sched_arr = attrs.get("scheduled_arrival")

                if (
                    sched_dep is not None
                    and sched_arr is not None
                    and not pd.isnull(sched_dep)
                    and not pd.isnull(sched_arr)
                ):
                    buffer_min = (
                        pd.Timestamp(sched_dep) - pd.Timestamp(sched_arr)
                    ).total_seconds() / 60.0
                else:
                    buffer_min = edge_data.get(
                        "turnaround_min",
                        edge_data.get("time_diff", min_connection_min),
                    )

                # Cascading failure: when the upstream delay exceeds the
                # buffer by more than the cascade threshold, the downstream
                # service is penalised with CASCADE_LARGE_DELAY_MIN (60 min).
                if arr_delay > buffer_min + _cfg.CASCADE_THRESHOLD_MIN:
                    cur_dep_delay = n_attrs.get("simulated_departure_delay", 0.0)
                    n_attrs["simulated_departure_delay"] = max(
                        cur_dep_delay, _cfg.CASCADE_LARGE_DELAY_MIN
                    )
                    logger.debug(
                        "Cascading failure on dependency %s → %s "
                        "(arr_delay=%.1f > buffer=%.1f + threshold=%.1f)",
                        node,
                        neighbour,
                        arr_delay,
                        buffer_min,
                        _cfg.CASCADE_THRESHOLD_MIN,
                    )
                else:
                    delay_b = max(0.0, arr_delay - buffer_min)
                    if delay_b > 0:
                        cur_dep_delay = n_attrs.get("simulated_departure_delay", 0.0)
                        n_attrs["simulated_departure_delay"] = max(
                            cur_dep_delay, delay_b
                        )

    return G


def propagate_delay(
    G: nx.DiGraph,
    source_node: str,
    initial_delay: float,
    min_dwell_min: float = DEFAULT_MIN_DWELL_MIN,
    min_connection_min: float = 2.0,
    alpha: float = DEFAULT_ALPHA,
    max_interaction_min: float = DEFAULT_MAX_INTERACTION_MIN,
    recovery_time_min: float = DEFAULT_RECOVERY_TIME_MIN,
) -> nx.DiGraph:
    """Inject a single delay and propagate it through the network.

    This is a convenience wrapper around :func:`reset_delays`,
    :func:`inject_delay`, and :func:`propagate_delays`.

    Parameters
    ----------
    G:
        Temporal graph produced by :mod:`graph_construction`.
    source_node:
        Node key where the initial delay is injected.
    initial_delay:
        Delay magnitude in minutes.
    min_dwell_min:
        Minimum dwell time at a stop (minutes).
    min_connection_min:
        Minimum connection gap for dependency edges when no scheduled times
        are available (minutes).
    alpha:
        Base interaction strength (0–1).
    max_interaction_min:
        Maximum gap used for the improved alpha formula (minutes).
    recovery_time_min:
        Recovery time on movement edges (minutes).

    Returns
    -------
    nx.DiGraph
        The same graph with updated delay attributes.

    Example
    -------
    >>> G = propagate_delay(G, "SVC003|MIR|2026-04-30T10:22:00", 10.0)
    """
    reset_delays(G)
    inject_delay(G, source_node, initial_delay)
    return propagate_delays(
        G,
        min_dwell_min=min_dwell_min,
        min_connection_min=min_connection_min,
        alpha=alpha,
        max_interaction_min=max_interaction_min,
        recovery_time_min=recovery_time_min,
    )


def run_simulation(
    G: nx.DiGraph,
    disruptions: list[dict],
    min_dwell_min: float = DEFAULT_MIN_DWELL_MIN,
    min_connection_min: float = 2.0,
    alpha: float = DEFAULT_ALPHA,
    max_interaction_min: float = DEFAULT_MAX_INTERACTION_MIN,
    recovery_time_min: float = DEFAULT_RECOVERY_TIME_MIN,
) -> nx.DiGraph:
    """Reset, inject disruptions, then propagate.

    Parameters
    ----------
    G:
        Temporal graph.
    disruptions:
        List of disruption dicts.  Each dict must have:
        - ``"node_key"``      – target node
        - ``"delay_minutes"`` – delay to inject (minutes)
        Optional keys:
        - ``"affect_arrival"``   (default ``True``)
        - ``"affect_departure"`` (default ``True``)
    min_dwell_min:
        Minimum dwell time (minutes).
    min_connection_min:
        Minimum connection gap for dependency edges (minutes).
    alpha:
        Base interaction strength (0–1) for interaction edges.
    max_interaction_min:
        Maximum gap used for the improved alpha formula (minutes).
    recovery_time_min:
        Recovery time on movement edges (minutes).

    Returns
    -------
    nx.DiGraph
        Updated graph after propagation.

    Example
    -------
    >>> disruptions = [{"node_key": "W12345|EUS|2024-01-15T07:00:00",
    ...                 "delay_minutes": 10}]
    >>> G = run_simulation(G, disruptions)
    """
    reset_delays(G)
    for d in disruptions:
        inject_delay(
            G,
            d["node_key"],
            d["delay_minutes"],
            affect_arrival=d.get("affect_arrival", True),
            affect_departure=d.get("affect_departure", True),
        )
    return propagate_delays(
        G,
        min_dwell_min=min_dwell_min,
        min_connection_min=min_connection_min,
        alpha=alpha,
        max_interaction_min=max_interaction_min,
        recovery_time_min=recovery_time_min,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chronological_order(G: nx.DiGraph) -> list[str]:
    """Return nodes sorted by their temporal anchor (scheduled departure,
    falling back to scheduled arrival).  Nodes without any time anchor are
    appended at the end in arbitrary order.
    """

    def _anchor(node: str) -> pd.Timestamp:
        attrs = G.nodes[node]
        t_dep = attrs.get("scheduled_departure")
        t_arr = attrs.get("scheduled_arrival")
        for t in (t_dep, t_arr):
            if t is not None:
                try:
                    if not pd.isnull(t):
                        return pd.Timestamp(t)
                except (TypeError, ValueError):
                    pass
        return pd.Timestamp.max  # unknown times sort to the end

    return sorted(G.nodes, key=_anchor)

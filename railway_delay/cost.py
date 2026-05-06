"""
cost.py
=======
Cost model for the cost-aware disruption decision framework (Phase 5).

Mathematical model
------------------
For a node n with delay d_n (minutes):

1. Compensation rate (step function):

       R(d) = 0.00  if d < 15
              0.25  if 15 ≤ d < 30
              0.50  if 30 ≤ d < 60
              1.00  if d ≥ 60

2. Node delay cost:

       C_n = p_n × s_n × c_n × R(d_n)

   where
     p_n – passenger weight = peak_factor × station_factor
           peak_factor:    PEAK_WEIGHT (2.0) during peak hours, else OFF_PEAK_WEIGHT (1.0)
           station_factor: from STATION_FACTORS dict (default 1.0 for unknown stations)
     s_n – service importance weight (node attribute, default 1.0)
     c_n – average ticket price (node attribute, default £10.0)

3. Cancellation cost (full cost, no R factor):

       C_cancel_n = p_n × s_n × c_n

4. Total cost:

       C_total = Σ C_n (active nodes)
               + Σ C_cancel_n (cancelled nodes, delay == inf)
               + Σ missed_connection_penalty (nodes where d_n > connection_buffer)

   Note: threshold-crossing penalties are NOT included in C_total to avoid
   double-counting with the step-based R(d) function.  Threshold awareness
   is expressed via :func:`threshold_crossing_surcharge`, which is used only
   during interaction-ordering decisions in :mod:`~railway_delay.simulation`.

Peak hours
----------
Morning peak: 07:00–09:00 (inclusive of start, exclusive of end)
Evening peak: 17:00–19:00 (inclusive of start, exclusive of end)
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import pandas as pd

from railway_delay import config as _cfg

# ---------------------------------------------------------------------------
# Constants – sourced from config.py so all parameters live in one place.
# ---------------------------------------------------------------------------

#: Peak hour windows as ``(start_hour_inclusive, end_hour_exclusive)`` pairs.
PEAK_WINDOWS: tuple[tuple[int, int], ...] = _cfg.PEAK_WINDOWS

#: Passenger weight multiplier applied during peak hours.
PEAK_WEIGHT: float = _cfg.PEAK_WEIGHT

#: Passenger weight multiplier applied during off-peak hours.
OFF_PEAK_WEIGHT: float = _cfg.OFF_PEAK_WEIGHT

#: Default average ticket price (£) used when not stored on the node.
DEFAULT_AVG_TICKET_PRICE: float = _cfg.DEFAULT_AVG_TICKET_PRICE

#: Default service importance weight used when not stored on the node.
DEFAULT_SERVICE_WEIGHT: float = _cfg.DEFAULT_SERVICE_WEIGHT

#: Default threshold (minutes) beyond which a missed connection penalty applies.
DEFAULT_CONNECTION_BUFFER_MIN: float = _cfg.DEFAULT_CONNECTION_BUFFER_MIN

#: Default missed connection penalty (£) per affected node.
DEFAULT_MISSED_CONNECTION_PENALTY: float = _cfg.DEFAULT_MISSED_CONNECTION_PENALTY


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def threshold_crossing_penalty(delay: float) -> float:
    """Return the total threshold-crossing penalty for a given delay.

    .. deprecated::
        This function is retained as a reference utility.  It is **no longer
        called by** :func:`compute_total_cost` to avoid double-counting with
        the step-based :func:`compensation_rate`.  For decision-time threshold
        awareness use :func:`threshold_crossing_surcharge` instead.

    Each time a delay crosses a milestone (15 min, 30 min, 60 min) an extra
    penalty is accumulated.

    Parameters
    ----------
    delay:
        Simulated departure delay in minutes.  Negative values or ``inf``
        both return 0.

    Returns
    -------
    float
        Sum of all applicable threshold penalties (£).
    """
    if delay <= 0 or delay == float("inf"):
        return 0.0
    penalty = 0.0
    if delay >= 15:
        penalty += _cfg.THRESHOLD_15_PENALTY
    if delay >= 30:
        penalty += _cfg.THRESHOLD_30_PENALTY
    if delay >= 60:
        penalty += _cfg.THRESHOLD_60_PENALTY
    return penalty


def threshold_crossing_surcharge(delay_before: float, delay_after: float) -> float:
    """Extra cost when *delay_after* crosses a Delay Repay threshold that
    *delay_before* did not cross.

    Used during **interaction-ordering decisions** in
    :mod:`~railway_delay.simulation` to steer the model away from options
    that push a train's delay over a compensation milestone (15, 30, or 60
    minutes).  This is a decision heuristic, not a network-cost component,
    so it is not included in :func:`compute_total_cost`.

    Parameters
    ----------
    delay_before:
        Delay in minutes before the decision is applied.
    delay_after:
        Delay in minutes after the decision would be applied.

    Returns
    -------
    float
        Sum of penalty values for each newly crossed threshold (£).

    Examples
    --------
    >>> threshold_crossing_surcharge(10, 20)   # crosses 15-min band
    10.0
    >>> threshold_crossing_surcharge(14, 31)   # crosses 15 and 30-min bands
    35.0
    >>> threshold_crossing_surcharge(20, 25)   # no threshold crossed
    0.0
    """
    if delay_after <= delay_before:
        return 0.0
    surcharge = 0.0
    for threshold, penalty in (
        (15.0, _cfg.THRESHOLD_15_PENALTY),
        (30.0, _cfg.THRESHOLD_30_PENALTY),
        (60.0, _cfg.THRESHOLD_60_PENALTY),
    ):
        if delay_before < threshold <= delay_after:
            surcharge += penalty
    return surcharge


def passenger_weight(scheduled_departure, station_crs: str | None = None) -> float:
    """Return the passenger weight multiplier for a scheduled departure time
    and optional station.

    The weight is computed as::

        passenger_weight = peak_factor × station_factor

    where *peak_factor* is :data:`PEAK_WEIGHT` during peak windows and
    :data:`OFF_PEAK_WEIGHT` otherwise, and *station_factor* is looked up from
    :data:`~railway_delay.config.STATION_FACTORS` (defaulting to 1.0 for
    unknown stations).

    Parameters
    ----------
    scheduled_departure:
        Scheduled departure time; any type accepted by :class:`pandas.Timestamp`.
        ``None`` or NaT is treated as off-peak.
    station_crs:
        Three-letter CRS code of the station (e.g. ``"EUS"``).  ``None``
        or unknown codes default to a station factor of 1.0.

    Returns
    -------
    float
        Combined passenger weight (≥ 0).

    Examples
    --------
    >>> passenger_weight(pd.Timestamp("2026-04-30 08:00"))
    2.0
    >>> passenger_weight(pd.Timestamp("2026-04-30 11:00"))
    1.0
    >>> passenger_weight(pd.Timestamp("2026-04-30 08:00"), station_crs="EUS")
    3.0
    """
    peak_factor = OFF_PEAK_WEIGHT
    if scheduled_departure is not None:
        try:
            if not pd.isnull(scheduled_departure):
                hour = pd.Timestamp(scheduled_departure).hour
                for start, end in PEAK_WINDOWS:
                    if start <= hour < end:
                        peak_factor = PEAK_WEIGHT
                        break
        except (TypeError, ValueError):
            pass

    station_factor: float = _cfg.STATION_FACTORS.get(station_crs or "", 1.0)
    return peak_factor * station_factor


def compensation_rate(delay: float) -> float:
    """Return the step-based compensation rate R(d).

    Parameters
    ----------
    delay:
        Delay in minutes.  Negative values are treated as 0.

    Returns
    -------
    float
        0.00 if d < 15,
        0.25 if 15 ≤ d < 30,
        0.50 if 30 ≤ d < 60,
        1.00 if d ≥ 60.

    Examples
    --------
    >>> compensation_rate(0)
    0.0
    >>> compensation_rate(14.9)
    0.0
    >>> compensation_rate(15)
    0.25
    >>> compensation_rate(30)
    0.5
    >>> compensation_rate(60)
    1.0
    """
    if delay >= 60:
        return 1.0
    if delay >= 30:
        return 0.5
    if delay >= 15:
        return 0.25
    return 0.0


def compute_node_cost(node_attrs: dict, delay: float) -> float:
    """Compute the delay cost for a single active node.

    C_n = p_n × s_n × c_n × R(d_n)

    where p_n = peak_factor × station_factor (see :func:`passenger_weight`).

    Parameters
    ----------
    node_attrs:
        Node attribute dict from the temporal graph.  Recognised keys:
        - ``scheduled_departure`` – used to derive *peak_factor*
        - ``station_crs``         – used to look up *station_factor*
        - ``service_weight``      – *s_n* (default :data:`DEFAULT_SERVICE_WEIGHT`)
        - ``avg_ticket_price``    – *c_n* (default :data:`DEFAULT_AVG_TICKET_PRICE`)
    delay:
        Simulated departure delay in minutes.

    Returns
    -------
    float
        Cost in the same monetary unit as ``avg_ticket_price``.
    """
    p_n = passenger_weight(
        node_attrs.get("scheduled_departure"),
        station_crs=node_attrs.get("station_crs"),
    )
    s_n = float(node_attrs.get("service_weight", DEFAULT_SERVICE_WEIGHT))
    c_n = float(node_attrs.get("avg_ticket_price", DEFAULT_AVG_TICKET_PRICE))
    return p_n * s_n * c_n * compensation_rate(delay)


def compute_cancellation_cost(node_attrs: dict) -> float:
    """Compute the full cancellation cost for a single cancelled node.

    C_cancel_n = p_n × s_n × c_n

    Parameters
    ----------
    node_attrs:
        Node attribute dict.  Same keys as :func:`compute_node_cost`.

    Returns
    -------
    float
        Cancellation cost.
    """
    p_n = passenger_weight(
        node_attrs.get("scheduled_departure"),
        station_crs=node_attrs.get("station_crs"),
    )
    s_n = float(node_attrs.get("service_weight", DEFAULT_SERVICE_WEIGHT))
    c_n = float(node_attrs.get("avg_ticket_price", DEFAULT_AVG_TICKET_PRICE))
    return p_n * s_n * c_n


def compute_total_cost(
    graph: nx.DiGraph,
    cancelled_nodes: Optional[set[str]] = None,
    connection_buffer_min: float = DEFAULT_CONNECTION_BUFFER_MIN,
    missed_connection_penalty: float = DEFAULT_MISSED_CONNECTION_PENALTY,
) -> float:
    """Compute the total network cost after a simulation run.

    C_total = Σ C_n (active nodes, delay < inf)
            + Σ C_cancel_n (cancelled nodes, delay == inf or in cancelled_nodes)
            + Σ penalty (nodes where delay > connection_buffer_min)

    Note: threshold-crossing penalties are intentionally excluded here to
    avoid double-counting with the step-based :func:`compensation_rate`.
    Threshold awareness is expressed through
    :func:`threshold_crossing_surcharge`, which influences interaction
    decisions during propagation.

    Parameters
    ----------
    graph:
        Temporal graph with ``simulated_departure_delay`` updated after
        :func:`~railway_delay.simulation.propagate_delays`.
    cancelled_nodes:
        Optional set of node keys to treat as cancelled regardless of their
        simulated delay.  If ``None``, only nodes with
        ``simulated_departure_delay == float('inf')`` are treated as cancelled.
    connection_buffer_min:
        Threshold (minutes) above which a missed connection penalty is added
        per affected node.
    missed_connection_penalty:
        Monetary penalty per node whose delay exceeds *connection_buffer_min*.

    Returns
    -------
    float
        Total cost across all nodes.
    """
    if cancelled_nodes is None:
        cancelled_nodes = set()

    total = 0.0
    for node, attrs in graph.nodes(data=True):
        delay = attrs.get("simulated_departure_delay", 0.0)

        is_cancelled = (delay == float("inf")) or (node in cancelled_nodes)

        if is_cancelled:
            total += compute_cancellation_cost(attrs)
        else:
            total += compute_node_cost(attrs, delay)
            if delay > connection_buffer_min:
                total += missed_connection_penalty

    return total

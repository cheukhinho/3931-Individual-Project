"""
disruption.py
=============
Synthetic disruption scenario generation for evaluation of the delay
propagation and optimisation pipeline.

A *disruption scenario* is a list of node-level delay injections that can be
passed directly to :func:`~railway_delay.simulation.run_simulation`.

Scenario types
--------------
single_point
    One random node receives a delay of the specified magnitude.

multi_point
    Several random nodes receive independent delays.

station_incident
    All services departing from a given station within a time window are
    delayed simultaneously (simulates a platform incident).

cascade_seed
    A single high-delay injection at an origin node; the cascade then
    propagates naturally through the graph via simulation.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_departure_nodes(G: nx.DiGraph) -> list[str]:
    """Return nodes that have a non-null scheduled departure."""
    return [
        n
        for n, attrs in G.nodes(data=True)
        if attrs.get("scheduled_departure") is not None
        and not pd.isnull(attrs["scheduled_departure"])
    ]


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------


def single_point_disruption(
    G: nx.DiGraph,
    delay_minutes: float = 10.0,
    *,
    seed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Inject a delay on one randomly selected departure node.

    Parameters
    ----------
    G:
        Temporal graph.
    delay_minutes:
        Delay magnitude in minutes.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list of dict
        Single-element list suitable for :func:`~railway_delay.simulation.run_simulation`.
    """
    rng = random.Random(seed)
    candidates = _all_departure_nodes(G)
    if not candidates:
        raise ValueError("Graph has no eligible departure nodes.")
    chosen = rng.choice(candidates)
    return [{"node_key": chosen, "delay_minutes": delay_minutes}]


def multi_point_disruption(
    G: nx.DiGraph,
    n_disruptions: int = 3,
    delay_range: tuple[float, float] = (5.0, 20.0),
    *,
    seed: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Inject delays on *n_disruptions* independently chosen nodes.

    Parameters
    ----------
    G:
        Temporal graph.
    n_disruptions:
        Number of disruption points to generate.
    delay_range:
        ``(min_delay, max_delay)`` in minutes; each disruption draws
        uniformly from this range.
    seed:
        Random seed.

    Returns
    -------
    list of dict
    """
    rng = random.Random(seed)
    candidates = _all_departure_nodes(G)
    if not candidates:
        raise ValueError("Graph has no eligible departure nodes.")
    n = min(n_disruptions, len(candidates))
    chosen = rng.sample(candidates, n)
    scenarios = []
    for node in chosen:
        delay = rng.uniform(*delay_range)
        scenarios.append({"node_key": node, "delay_minutes": round(delay, 1)})
    return scenarios


def station_incident(
    G: nx.DiGraph,
    station_crs: str,
    delay_minutes: float = 15.0,
    window_start: Optional[pd.Timestamp] = None,
    window_end: Optional[pd.Timestamp] = None,
) -> list[dict[str, Any]]:
    """Delay all departures at a given station within a time window.

    Parameters
    ----------
    G:
        Temporal graph.
    station_crs:
        Three-letter CRS code of the affected station.
    delay_minutes:
        Delay to apply to every affected departure (minutes).
    window_start:
        Start of the incident window.  ``None`` means no lower bound.
    window_end:
        End of the incident window.  ``None`` means no upper bound.

    Returns
    -------
    list of dict
    """
    scenarios = []
    for node, attrs in G.nodes(data=True):
        if attrs.get("station_crs") != station_crs:
            continue
        t_dep = attrs.get("scheduled_departure")
        if t_dep is None or pd.isnull(t_dep):
            continue
        t = pd.Timestamp(t_dep)
        if window_start is not None and t < window_start:
            continue
        if window_end is not None and t > window_end:
            continue
        scenarios.append({"node_key": node, "delay_minutes": delay_minutes})
    return scenarios


def generate_scenarios(
    G: nx.DiGraph,
    n_scenarios: int = 5,
    disruption_type: str = "single_point",
    delay_minutes: float = 10.0,
    n_disruptions: int = 3,
    delay_range: tuple[float, float] = (5.0, 20.0),
    seed: Optional[int] = None,
) -> list[list[dict[str, Any]]]:
    """Generate multiple independent disruption scenarios.

    Parameters
    ----------
    G:
        Temporal graph.
    n_scenarios:
        Number of scenarios to generate.
    disruption_type:
        One of ``"single_point"``, ``"multi_point"``.
    delay_minutes:
        Fixed delay for ``single_point`` scenarios (minutes).
    n_disruptions:
        Number of disruption points per ``multi_point`` scenario.
    delay_range:
        Delay range for ``multi_point`` scenarios.
    seed:
        Base random seed; each scenario increments it by 1.

    Returns
    -------
    list of list of dict
        Each inner list is one scenario (a list of disruption dicts).
    """
    scenarios = []
    for i in range(n_scenarios):
        scenario_seed = None if seed is None else seed + i
        if disruption_type == "single_point":
            s = single_point_disruption(G, delay_minutes, seed=scenario_seed)
        elif disruption_type == "multi_point":
            s = multi_point_disruption(
                G, n_disruptions, delay_range, seed=scenario_seed
            )
        else:
            raise ValueError(
                f"Unknown disruption_type {disruption_type!r}. "
                "Choose 'single_point' or 'multi_point'."
            )
        scenarios.append(s)
    return scenarios

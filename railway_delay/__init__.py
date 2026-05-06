"""
Railway Delay Propagation System
=================================
A modular pipeline for modelling and simulating railway delay propagation
using temporal graphs.

Modules
-------
data_ingestion      : Fetch and parse data from the RealTimeTrains API.
data_processing     : Transform raw JSON into structured pandas DataFrames.
graph_construction  : Build a temporal graph with NetworkX.
simulation          : Rule-based delay propagation on the graph.
disruption          : Synthetic disruption scenario generation.
cost                : Cost model (compensation rates, node/total cost).
optimisation        : Greedy heuristic and cost-aware decision optimisation.
evaluation          : Metrics and visualisation helpers.
"""

from railway_delay import (
    data_ingestion,
    data_processing,
    graph_construction,
    simulation,
    disruption,
    cost,
    optimisation,
    evaluation,
)

__all__ = [
    "data_ingestion",
    "data_processing",
    "graph_construction",
    "simulation",
    "disruption",
    "cost",
    "optimisation",
    "evaluation",
]

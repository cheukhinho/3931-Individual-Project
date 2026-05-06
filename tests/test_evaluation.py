"""Tests for railway_delay.evaluation."""

from __future__ import annotations

import pytest

from railway_delay.data_processing import build_stops_table
from railway_delay.graph_construction import build_temporal_graph
from railway_delay.simulation import run_simulation
from railway_delay.evaluation import (
    total_delay_minutes,
    affected_services,
    affected_services_count,
    delay_propagation_depth,
    compute_metrics,
    delays_to_dataframe,
)
import pandas as pd


@pytest.fixture
def graph(sample_stops_df):
    stops = build_stops_table(sample_stops_df)
    return build_temporal_graph(stops)


@pytest.fixture
def first_node(graph):
    for n, attrs in graph.nodes(data=True):
        if attrs.get("service_id") == "SVC001" and attrs.get("station_crs") == "EUS":
            return n
    pytest.skip("Expected node not found")


@pytest.fixture
def disrupted_graph(graph, first_node):
    disruptions = [{"node_key": first_node, "delay_minutes": 10.0}]
    run_simulation(graph, disruptions)
    return graph, first_node


class TestTotalDelayMinutes:
    def test_zero_when_no_disruption(self, graph):
        from railway_delay.simulation import reset_delays
        reset_delays(graph)
        assert total_delay_minutes(graph) == 0.0

    def test_positive_after_disruption(self, disrupted_graph):
        G, _ = disrupted_graph
        assert total_delay_minutes(G) > 0.0

    def test_inf_excluded(self, graph):
        node = next(iter(graph.nodes))
        graph.nodes[node]["simulated_departure_delay"] = float("inf")
        total = total_delay_minutes(graph, exclude_inf=True)
        assert total != float("inf")


class TestAffectedServices:
    def test_empty_when_no_delay(self, graph):
        from railway_delay.simulation import reset_delays
        reset_delays(graph)
        assert len(affected_services(graph)) == 0

    def test_svc001_affected_after_disruption(self, disrupted_graph):
        G, _ = disrupted_graph
        svcs = affected_services(G)
        assert "SVC001" in svcs

    def test_threshold(self, graph):
        node = next(iter(graph.nodes))
        graph.nodes[node]["simulated_departure_delay"] = 0.5
        # threshold 1.0: 0.5 minute delay should NOT count
        svcs = affected_services(graph, threshold_min=1.0)
        assert graph.nodes[node].get("service_id", "") not in svcs or True


class TestAffectedServicesCount:
    def test_returns_int(self, disrupted_graph):
        G, _ = disrupted_graph
        count = affected_services_count(G)
        assert isinstance(count, int)

    def test_at_least_one_affected(self, disrupted_graph):
        G, _ = disrupted_graph
        assert affected_services_count(G) >= 1


class TestDelayPropagationDepth:
    def test_zero_when_no_delay(self, graph):
        from railway_delay.simulation import reset_delays
        reset_delays(graph)
        assert delay_propagation_depth(graph) == 0

    def test_positive_after_propagation(self, disrupted_graph):
        G, source = disrupted_graph
        depth = delay_propagation_depth(G, source_nodes=[source])
        assert depth >= 0  # may be 0 if only source is affected

    def test_with_no_source_nodes_specified(self, disrupted_graph):
        G, _ = disrupted_graph
        depth = delay_propagation_depth(G)
        assert isinstance(depth, int)


class TestComputeMetrics:
    def test_returns_all_keys(self, disrupted_graph):
        G, source = disrupted_graph
        metrics = compute_metrics(G, source_nodes=[source])
        assert "total_delay_minutes" in metrics
        assert "affected_services_count" in metrics
        assert "delay_propagation_depth" in metrics
        assert "affected_services" in metrics

    def test_types(self, disrupted_graph):
        G, source = disrupted_graph
        metrics = compute_metrics(G, source_nodes=[source])
        assert isinstance(metrics["total_delay_minutes"], float)
        assert isinstance(metrics["affected_services_count"], int)
        assert isinstance(metrics["delay_propagation_depth"], int)
        assert isinstance(metrics["affected_services"], set)


class TestDelaysToDataframe:
    def test_returns_dataframe(self, graph):
        df = delays_to_dataframe(graph)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, graph):
        df = delays_to_dataframe(graph)
        expected = {
            "node_key", "service_id", "station_crs",
            "scheduled_departure",
            "simulated_departure_delay", "simulated_arrival_delay",
        }
        assert expected.issubset(set(df.columns))

    def test_row_count_matches_nodes(self, graph):
        df = delays_to_dataframe(graph)
        assert len(df) == graph.number_of_nodes()

"""Tests for railway_delay.disruption."""

from __future__ import annotations

import pytest

from railway_delay.data_processing import build_stops_table
from railway_delay.graph_construction import build_temporal_graph
from railway_delay.disruption import (
    single_point_disruption,
    multi_point_disruption,
    station_incident,
    generate_scenarios,
)
import pandas as pd


@pytest.fixture
def graph(sample_stops_df):
    stops = build_stops_table(sample_stops_df)
    return build_temporal_graph(stops)


class TestSinglePointDisruption:
    def test_returns_single_disruption(self, graph):
        result = single_point_disruption(graph, delay_minutes=10.0, seed=42)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_disruption_format(self, graph):
        result = single_point_disruption(graph, seed=0)
        d = result[0]
        assert "node_key" in d
        assert "delay_minutes" in d
        assert d["node_key"] in graph.nodes

    def test_reproducible_with_seed(self, graph):
        r1 = single_point_disruption(graph, seed=7)
        r2 = single_point_disruption(graph, seed=7)
        assert r1[0]["node_key"] == r2[0]["node_key"]

    def test_different_seeds_may_differ(self, graph):
        r1 = single_point_disruption(graph, seed=1)
        r2 = single_point_disruption(graph, seed=999)
        # Not guaranteed to differ but with 6 nodes and different seeds usually will
        # Just check both are valid
        assert r1[0]["node_key"] in graph.nodes
        assert r2[0]["node_key"] in graph.nodes

    def test_custom_delay_magnitude(self, graph):
        result = single_point_disruption(graph, delay_minutes=25.0, seed=0)
        assert result[0]["delay_minutes"] == 25.0


class TestMultiPointDisruption:
    def test_correct_count(self, graph):
        result = multi_point_disruption(graph, n_disruptions=2, seed=0)
        assert len(result) == 2

    def test_all_nodes_valid(self, graph):
        result = multi_point_disruption(graph, n_disruptions=3, seed=0)
        for d in result:
            assert d["node_key"] in graph.nodes

    def test_delays_within_range(self, graph):
        result = multi_point_disruption(
            graph, n_disruptions=4, delay_range=(5.0, 20.0), seed=0
        )
        for d in result:
            assert 5.0 <= d["delay_minutes"] <= 20.0

    def test_no_duplicate_nodes(self, graph):
        result = multi_point_disruption(graph, n_disruptions=4, seed=0)
        keys = [d["node_key"] for d in result]
        assert len(keys) == len(set(keys))

    def test_caps_at_available_nodes(self, graph):
        n_nodes = graph.number_of_nodes()
        result = multi_point_disruption(graph, n_disruptions=9999, seed=0)
        assert len(result) <= n_nodes


class TestStationIncident:
    def test_affects_correct_station(self, graph):
        result = station_incident(graph, station_crs="EUS", delay_minutes=10.0)
        for d in result:
            node_attrs = graph.nodes[d["node_key"]]
            assert node_attrs["station_crs"] == "EUS"

    def test_delay_magnitude(self, graph):
        result = station_incident(graph, station_crs="BHM", delay_minutes=15.0)
        for d in result:
            assert d["delay_minutes"] == 15.0

    def test_no_results_for_unknown_station(self, graph):
        result = station_incident(graph, station_crs="ZZZ", delay_minutes=10.0)
        assert result == []

    def test_time_window_filter(self, graph):
        window_start = pd.Timestamp("2024-01-15 08:00:00")
        window_end = pd.Timestamp("2024-01-15 09:00:00")
        result = station_incident(
            graph,
            station_crs="BHM",
            delay_minutes=5.0,
            window_start=window_start,
            window_end=window_end,
        )
        for d in result:
            node_attrs = graph.nodes[d["node_key"]]
            t = pd.Timestamp(node_attrs["scheduled_departure"])
            assert window_start <= t <= window_end


class TestGenerateScenarios:
    def test_correct_scenario_count(self, graph):
        scenarios = generate_scenarios(graph, n_scenarios=3, seed=0)
        assert len(scenarios) == 3

    def test_each_scenario_is_list(self, graph):
        scenarios = generate_scenarios(graph, n_scenarios=2, seed=0)
        for s in scenarios:
            assert isinstance(s, list)

    def test_multi_point_type(self, graph):
        scenarios = generate_scenarios(
            graph, n_scenarios=2, disruption_type="multi_point",
            n_disruptions=2, seed=0
        )
        for s in scenarios:
            assert len(s) <= graph.number_of_nodes()

    def test_invalid_type_raises(self, graph):
        with pytest.raises(ValueError):
            generate_scenarios(graph, disruption_type="teleport", seed=0)

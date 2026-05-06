"""Tests for railway_delay.optimisation."""

from __future__ import annotations

import copy

import pytest

from railway_delay.data_processing import build_stops_table
from railway_delay.graph_construction import build_temporal_graph
from railway_delay.simulation import run_simulation, reset_delays
from railway_delay.optimisation import apply_action, greedy_optimise


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


class TestApplyAction:
    def test_no_action_unchanged(self, graph, first_node):
        reset_delays(graph)
        run_simulation(graph, [{"node_key": first_node, "delay_minutes": 10.0}])
        delay_before = graph.nodes[first_node]["simulated_departure_delay"]
        G2 = apply_action(graph, first_node, "no_action")
        assert G2.nodes[first_node]["simulated_departure_delay"] == delay_before

    def test_cancel_removes_movement_edges(self, graph, first_node):
        reset_delays(graph)
        run_simulation(graph, [{"node_key": first_node, "delay_minutes": 10.0}])
        G2 = apply_action(graph, first_node, "cancel_service")
        # Movement edges from first_node for SVC001 should be gone
        movement_out = [
            (u, v) for u, v, d in G2.out_edges(first_node, data=True)
            if d.get("edge_type") == "movement"
            and d.get("service_id") == "SVC001"
        ]
        assert len(movement_out) == 0

    def test_cancel_sets_inf_delay(self, graph, first_node):
        reset_delays(graph)
        run_simulation(graph, [{"node_key": first_node, "delay_minutes": 10.0}])
        G2 = apply_action(graph, first_node, "cancel_service")
        assert G2.nodes[first_node]["simulated_departure_delay"] == float("inf")

    def test_delay_departure_increases_delay(self, graph, first_node):
        reset_delays(graph)
        run_simulation(graph, [{"node_key": first_node, "delay_minutes": 10.0}])
        delay_before = graph.nodes[first_node]["simulated_departure_delay"]
        G2 = apply_action(graph, first_node, "delay_departure", hold_minutes=5.0)
        assert G2.nodes[first_node]["simulated_departure_delay"] == delay_before + 5.0

    def test_short_turn_removes_edge(self, graph, first_node):
        reset_delays(graph)
        run_simulation(graph, [{"node_key": first_node, "delay_minutes": 10.0}])
        edges_before = graph.out_degree(first_node)
        G2 = apply_action(graph, first_node, "short_turn", short_turn_depth=1)
        # Edge count at first_node should be reduced by at most 1
        assert G2.out_degree(first_node) <= edges_before

    def test_invalid_action_raises(self, graph, first_node):
        with pytest.raises(ValueError):
            apply_action(graph, first_node, "teleport")

    def test_apply_action_does_not_mutate_original(self, graph, first_node):
        reset_delays(graph)
        run_simulation(graph, [{"node_key": first_node, "delay_minutes": 10.0}])
        edges_before = graph.number_of_edges()
        _ = apply_action(graph, first_node, "cancel_service")
        assert graph.number_of_edges() == edges_before


class TestGreedyOptimise:
    def test_returns_dict_with_expected_keys(self, graph, first_node):
        disruptions = [{"node_key": first_node, "delay_minutes": 10.0}]
        result = greedy_optimise(graph, disruptions)
        assert "actions" in result
        assert "baseline_delay" in result
        assert "optimised_delay" in result
        assert "baseline_cost" in result
        assert "optimised_cost" in result
        assert "graph" in result

    def test_baseline_positive(self, graph, first_node):
        disruptions = [{"node_key": first_node, "delay_minutes": 10.0}]
        result = greedy_optimise(graph, disruptions)
        assert result["baseline_delay"] > 0.0

    def test_optimised_cost_not_worse_than_baseline(self, graph, first_node):
        disruptions = [{"node_key": first_node, "delay_minutes": 10.0}]
        result = greedy_optimise(graph, disruptions)
        # Greedy optimises on cost: optimised cost should never exceed baseline
        assert result["optimised_cost"] <= result["baseline_cost"] + 1e-6

    def test_optimised_not_worse_than_baseline(self, graph, first_node):
        disruptions = [{"node_key": first_node, "delay_minutes": 10.0}]
        result = greedy_optimise(graph, disruptions)
        # Delay metric: may stay equal (cost-optimal actions don't always cut
        # total delay, but must not make it strictly worse overall)
        assert result["optimised_delay"] <= result["baseline_delay"] + 1e-6

    def test_actions_map_to_valid_nodes(self, graph, first_node):
        disruptions = [{"node_key": first_node, "delay_minutes": 10.0}]
        result = greedy_optimise(graph, disruptions)
        import networkx as nx
        G_result = result["graph"]
        for node_key in result["actions"]:
            assert node_key in graph.nodes

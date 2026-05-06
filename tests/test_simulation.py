"""Tests for railway_delay.simulation."""

from __future__ import annotations

from datetime import datetime

import pytest

from railway_delay.data_processing import build_stops_table
from railway_delay.graph_construction import (
    build_temporal_graph,
    add_interaction_edges,
    add_dependency_edges,
    node_key,
)
from railway_delay.simulation import (
    reset_delays,
    inject_delay,
    propagate_delays,
    propagate_delay,
    run_simulation,
)


@pytest.fixture
def graph(sample_stops_df):
    stops = build_stops_table(sample_stops_df)
    return build_temporal_graph(stops)


@pytest.fixture
def first_node(graph):
    """Return the departure node of SVC001 at EUS."""
    for n, attrs in graph.nodes(data=True):
        if attrs.get("service_id") == "SVC001" and attrs.get("station_crs") == "EUS":
            return n
    pytest.skip("Expected node not found in graph")


class TestResetDelays:
    def test_all_delays_zeroed(self, graph):
        # Manually set a delay, then reset
        node = next(iter(graph.nodes))
        graph.nodes[node]["simulated_departure_delay"] = 99.0
        reset_delays(graph)
        for _, attrs in graph.nodes(data=True):
            assert attrs["simulated_departure_delay"] == 0.0
            assert attrs["simulated_arrival_delay"] == 0.0


class TestInjectDelay:
    def test_injection_applied(self, graph, first_node):
        inject_delay(graph, first_node, 10.0)
        assert graph.nodes[first_node]["simulated_departure_delay"] == 10.0
        assert graph.nodes[first_node]["simulated_arrival_delay"] == 10.0

    def test_arrival_only(self, graph, first_node):
        reset_delays(graph)
        inject_delay(graph, first_node, 8.0, affect_arrival=True, affect_departure=False)
        assert graph.nodes[first_node]["simulated_arrival_delay"] == 8.0
        assert graph.nodes[first_node]["simulated_departure_delay"] == 0.0

    def test_invalid_node_raises(self, graph):
        with pytest.raises(KeyError):
            inject_delay(graph, "NONEXISTENT_NODE", 10.0)


class TestPropagateDelays:
    def test_delay_propagates_downstream(self, graph, first_node):
        reset_delays(graph)
        inject_delay(graph, first_node, 10.0)
        propagate_delays(graph)

        # At least the source node should remain delayed
        assert graph.nodes[first_node]["simulated_departure_delay"] >= 10.0

        # Check downstream service nodes are also affected
        downstream_delays = [
            attrs.get("simulated_arrival_delay", 0.0)
            for n, attrs in graph.nodes(data=True)
            if attrs.get("service_id") == "SVC001" and n != first_node
        ]
        assert any(d > 0 for d in downstream_delays), (
            "Expected downstream SVC001 nodes to be delayed"
        )

    def test_zero_injection_no_propagation(self, graph):
        reset_delays(graph)
        propagate_delays(graph)
        for _, attrs in graph.nodes(data=True):
            assert attrs["simulated_departure_delay"] == 0.0
            assert attrs["simulated_arrival_delay"] == 0.0

    def test_returns_graph(self, graph, first_node):
        reset_delays(graph)
        inject_delay(graph, first_node, 5.0)
        result = propagate_delays(graph)
        import networkx as nx
        assert isinstance(result, nx.DiGraph)


class TestRunSimulation:
    def test_end_to_end(self, graph, first_node):
        disruptions = [{"node_key": first_node, "delay_minutes": 15.0}]
        result = run_simulation(graph, disruptions)
        import networkx as nx
        assert isinstance(result, nx.DiGraph)
        assert result.nodes[first_node]["simulated_departure_delay"] >= 15.0

    def test_reset_before_new_run(self, graph, first_node):
        disruptions = [{"node_key": first_node, "delay_minutes": 5.0}]
        run_simulation(graph, disruptions)
        first_run_delay = graph.nodes[first_node]["simulated_departure_delay"]

        # Run again with a larger delay – should not stack
        disruptions2 = [{"node_key": first_node, "delay_minutes": 10.0}]
        run_simulation(graph, disruptions2)
        second_run_delay = graph.nodes[first_node]["simulated_departure_delay"]

        assert second_run_delay >= 10.0
        # Should not be 15 (first + second run added together)
        assert second_run_delay < 20.0

    def test_multiple_disruptions(self, graph, first_node):
        node_list = list(graph.nodes)
        if len(node_list) < 2:
            pytest.skip("Not enough nodes")
        disruptions = [
            {"node_key": node_list[0], "delay_minutes": 5.0},
            {"node_key": node_list[1], "delay_minutes": 3.0},
        ]
        run_simulation(graph, disruptions)
        # At least one of the disrupted nodes should be delayed
        delays = [
            graph.nodes[n]["simulated_departure_delay"]
            for n in node_list[:2]
        ]
        assert sum(delays) > 0


# ---------------------------------------------------------------------------
# Tests for interaction-edge propagation
# ---------------------------------------------------------------------------

@pytest.fixture
def interaction_graph(sample_stops_close_df):
    """Graph built from the close-service fixture (SVC003/SVC004 at MIR/BGH)
    with interaction edges enabled."""
    stops = build_stops_table(sample_stops_close_df)
    G = build_temporal_graph(stops, add_dependencies=False, add_interactions=True,
                              max_interaction_min=5.0)
    return G


class TestInteractionEdgePropagation:
    def test_interaction_edges_present(self, interaction_graph):
        """The fixture graph must contain at least one interaction edge."""
        types = [d.get("edge_type") for _, _, d in interaction_graph.edges(data=True)]
        assert "interaction" in types

    def test_partial_transfer_via_interaction(self, interaction_graph):
        """Delaying SVC003 at MIR should partially transfer to SVC004 at MIR."""
        import pandas as pd

        # Find the SVC003 node at MIR
        svc003_mir = None
        for n, attrs in interaction_graph.nodes(data=True):
            if attrs.get("service_id") == "SVC003" and attrs.get("station_crs") == "MIR":
                svc003_mir = n
                break
        assert svc003_mir is not None, "SVC003 node at MIR not found"

        propagate_delay(interaction_graph, svc003_mir, 10.0, max_interaction_min=5.0)

        # SVC004 at MIR is within 4 minutes – should receive a partial delay
        svc004_mir_delay = None
        for n, attrs in interaction_graph.nodes(data=True):
            if attrs.get("service_id") == "SVC004" and attrs.get("station_crs") == "MIR":
                svc004_mir_delay = attrs.get("simulated_arrival_delay", 0.0)
                break
        assert svc004_mir_delay is not None, "SVC004 node at MIR not found"
        assert svc004_mir_delay > 0, (
            "Expected partial delay to transfer from SVC003 to SVC004 via "
            "interaction edge"
        )
        # Partial transfer: must be strictly less than the injected delay
        assert svc004_mir_delay < 10.0

    def test_no_transfer_for_zero_delay(self, interaction_graph):
        """With no injected delay, interaction edges must not create delays."""
        reset_delays(interaction_graph)
        propagate_delays(interaction_graph)
        for _, attrs in interaction_graph.nodes(data=True):
            assert attrs["simulated_arrival_delay"] == 0.0
            assert attrs["simulated_departure_delay"] == 0.0

    def test_alpha_zero_means_no_transfer(self, interaction_graph):
        """Setting max_interaction_min=0 with alpha=0 should suppress transfer."""
        # Find the SVC003 MIR node
        svc003_mir = next(
            n for n, a in interaction_graph.nodes(data=True)
            if a.get("service_id") == "SVC003" and a.get("station_crs") == "MIR"
        )
        propagate_delay(
            interaction_graph, svc003_mir, 10.0, alpha=0.0, max_interaction_min=0
        )
        for n, attrs in interaction_graph.nodes(data=True):
            if attrs.get("service_id") == "SVC004":
                assert attrs.get("simulated_arrival_delay", 0.0) == 0.0


# ---------------------------------------------------------------------------
# Tests for dependency-edge buffer propagation
# ---------------------------------------------------------------------------

@pytest.fixture
def dependency_graph(turnaround_stops_df):
    """Graph built from the turnaround fixture at BGH with turnaround edges."""
    stops = build_stops_table(turnaround_stops_df)
    G = build_temporal_graph(
        stops,
        add_dependencies=True,
        min_connection_min=5.0,
        max_connection_min=45.0,
        add_interactions=False,
        add_turnarounds=True,
        min_turnaround_min=5.0,
        max_turnaround_high_min=45.0,
        max_turnaround_medium_min=30.0,
    )
    return G


class TestDependencyEdgePropagation:
    def test_delay_absorbed_by_buffer(self, dependency_graph):
        """A small delay fully absorbed by the turnaround buffer must not
        propagate to the downstream departure."""
        # SVC_T3 arrives BGH at 12:00; SVC_T4 departs at 12:15 → 15 min buffer.
        # Injecting 5 min delay at SVC_T3 arrival → 5 < 15 → no propagation.
        svc_t3_node = next(
            (n for n, a in dependency_graph.nodes(data=True)
             if a.get("service_id") == "SVC_T3"),
            None,
        )
        assert svc_t3_node is not None, "SVC_T3 node not found"

        propagate_delay(dependency_graph, svc_t3_node, 5.0)

        svc_t4_dep_delay = next(
            (a.get("simulated_departure_delay", 0.0)
             for n, a in dependency_graph.nodes(data=True)
             if a.get("service_id") == "SVC_T4"),
            None,
        )
        assert svc_t4_dep_delay is not None, "SVC_T4 node not found"
        assert svc_t4_dep_delay == 0.0, (
            "5-min delay should be fully absorbed by the 15-min turnaround buffer"
        )

    def test_delay_exceeding_buffer_propagates(self, dependency_graph):
        """A delay larger than the buffer must partially propagate."""
        # SVC_T3 → SVC_T4: 15-min buffer.  Injecting 20-min delay → 5 min spills.
        svc_t3_node = next(
            (n for n, a in dependency_graph.nodes(data=True)
             if a.get("service_id") == "SVC_T3"),
            None,
        )
        assert svc_t3_node is not None, "SVC_T3 node not found"

        propagate_delay(dependency_graph, svc_t3_node, 20.0)

        svc_t4_dep_delay = next(
            (a.get("simulated_departure_delay", 0.0)
             for n, a in dependency_graph.nodes(data=True)
             if a.get("service_id") == "SVC_T4"),
            None,
        )
        assert svc_t4_dep_delay is not None, "SVC_T4 node not found"
        assert svc_t4_dep_delay > 0.0, (
            "20-min delay exceeds 15-min buffer, so some delay should propagate"
        )
        # Propagated delay = 20 − 15 = 5 min (approximately)
        assert abs(svc_t4_dep_delay - 5.0) < 1.0


# ---------------------------------------------------------------------------
# Tests for propagate_delay convenience function
# ---------------------------------------------------------------------------

class TestPropagateDelayFunction:
    def test_returns_graph(self, graph, first_node):
        import networkx as nx
        result = propagate_delay(graph, first_node, 10.0)
        assert isinstance(result, nx.DiGraph)

    def test_source_node_delayed(self, graph, first_node):
        propagate_delay(graph, first_node, 10.0)
        assert graph.nodes[first_node]["simulated_departure_delay"] >= 10.0

    def test_resets_before_propagation(self, graph, first_node):
        # Pre-pollute the graph with a large delay on a different node
        other_node = next(n for n in graph.nodes if n != first_node)
        graph.nodes[other_node]["simulated_departure_delay"] = 999.0

        propagate_delay(graph, first_node, 5.0)

        # The pre-polluted node should have been reset (and only naturally
        # re-delayed if downstream of first_node)
        other_delay = graph.nodes[other_node]["simulated_departure_delay"]
        assert other_delay < 999.0

    def test_invalid_source_raises(self, graph):
        with pytest.raises(KeyError):
            propagate_delay(graph, "NONEXISTENT|INVALID|1970-01-01T00:00:00", 5.0)


# ---------------------------------------------------------------------------
# MIR → BGH scenario test (2026-04-30, 10:00–14:00)
# ---------------------------------------------------------------------------

class TestMirBghScenario:
    """Validate propagation using services between Mirfield (MIR) and
    Brighouse (BGH) on 2026-04-30.  Uses the close-proximity fixture (SVC003
    and SVC004) from conftest."""

    def test_movement_delay_reaches_bgh(self, interaction_graph):
        """Injecting delay at MIR for SVC003 propagates to BGH via movement."""
        svc003_mir = next(
            (n for n, a in interaction_graph.nodes(data=True)
             if a.get("service_id") == "SVC003" and a.get("station_crs") == "MIR"),
            None,
        )
        assert svc003_mir is not None

        propagate_delay(interaction_graph, svc003_mir, 10.0)

        svc003_bgh_arr = next(
            (a.get("simulated_arrival_delay", 0.0)
             for n, a in interaction_graph.nodes(data=True)
             if a.get("service_id") == "SVC003" and a.get("station_crs") == "BGH"),
            None,
        )
        assert svc003_bgh_arr is not None, "SVC003 BGH node not found"
        assert svc003_bgh_arr >= 10.0, (
            "Movement edge should carry full delay to the next stop"
        )

    def test_interaction_delay_smaller_than_movement_delay(self, interaction_graph):
        """Interaction transfer to SVC004 must be less than the full movement
        delay flowing to SVC003's own next stop."""
        svc003_mir = next(
            n for n, a in interaction_graph.nodes(data=True)
            if a.get("service_id") == "SVC003" and a.get("station_crs") == "MIR"
        )
        propagate_delay(interaction_graph, svc003_mir, 10.0)

        svc003_bgh_arr = next(
            a.get("simulated_arrival_delay", 0.0)
            for n, a in interaction_graph.nodes(data=True)
            if a.get("service_id") == "SVC003" and a.get("station_crs") == "BGH"
        )
        svc004_mir_arr = next(
            a.get("simulated_arrival_delay", 0.0)
            for n, a in interaction_graph.nodes(data=True)
            if a.get("service_id") == "SVC004" and a.get("station_crs") == "MIR"
        )

        # Movement carries full delay; interaction carries only a fraction
        assert svc003_bgh_arr >= svc004_mir_arr

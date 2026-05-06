"""Tests for railway_delay.graph_construction."""

from __future__ import annotations

import pandas as pd
import networkx as nx
import pytest

from railway_delay.data_processing import build_stops_table
from railway_delay.graph_construction import (
    build_movement_graph,
    add_dependency_edges,
    add_interaction_edges,
    add_dependency_edges_turnaround,
    add_turnaround_edges,
    build_temporal_graph,
    node_key,
)
from datetime import datetime


class TestNodeKey:
    def test_format(self):
        key = node_key("SVC001", "EUS", datetime(2024, 1, 15, 7, 0))
        assert key == "SVC001|EUS|2024-01-15T07:00:00"


class TestBuildMovementGraph:
    def test_returns_digraph(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        G = build_movement_graph(stops)
        assert isinstance(G, nx.DiGraph)

    def test_node_count(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        G = build_movement_graph(stops)
        # SVC001: 3 nodes (EUS-dep, MKC-dep, BHM-arr)
        # SVC002: 3 nodes (BHM-dep, CRE-dep, LIV-arr)
        assert G.number_of_nodes() == 6

    def test_movement_edge_count(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        G = build_movement_graph(stops)
        movement_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "movement"
        ]
        # SVC001: 2 edges; SVC002: 2 edges
        assert len(movement_edges) == 4

    def test_node_attributes(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        G = build_movement_graph(stops)
        for _, attrs in G.nodes(data=True):
            assert "service_id" in attrs
            assert "station_crs" in attrs
            assert "simulated_departure_delay" in attrs
            assert attrs["simulated_departure_delay"] == 0.0

    def test_edge_attributes(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        G = build_movement_graph(stops)
        for u, v, d in G.edges(data=True):
            assert "edge_type" in d
            assert d["edge_type"] == "movement"
            assert "service_id" in d

    def test_no_self_loops(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        G = build_movement_graph(stops)
        assert not any(u == v for u, v in G.edges())


class TestAddDependencyEdges:
    def test_dependency_edges_added(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        G = build_movement_graph(stops)
        before = G.number_of_edges()
        G = add_dependency_edges(G, stops, min_connection_min=2.0, max_connection_min=30.0)
        # SVC001 arrives BHM at 08:30, SVC002 departs BHM at 08:45 → gap 15 min
        dep_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "dependency"
        ]
        assert len(dep_edges) >= 1

    def test_dependency_edge_attributes(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        G = build_movement_graph(stops)
        G = add_dependency_edges(G, stops, min_connection_min=2.0, max_connection_min=30.0)
        for u, v, d in G.edges(data=True):
            if d.get("edge_type") == "dependency":
                assert "turnaround_min" in d


class TestBuildTemporalGraph:
    def test_end_to_end(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        G = build_temporal_graph(stops)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_no_dependencies_flag(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        G = build_temporal_graph(stops, add_dependencies=False)
        dep_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "dependency"
        ]
        assert len(dep_edges) == 0

    def test_interactions_flag(self, sample_stops_close_df):
        stops = build_stops_table(sample_stops_close_df)
        G = build_temporal_graph(
            stops,
            add_dependencies=False,
            add_interactions=True,
            max_interaction_min=5.0,
        )
        int_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "interaction"
        ]
        # SVC003 and SVC004 are within 4 min at both MIR and BGH
        assert len(int_edges) == 2

    def test_no_interactions_by_default(self, sample_stops_close_df):
        stops = build_stops_table(sample_stops_close_df)
        G = build_temporal_graph(stops, add_dependencies=False)
        int_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "interaction"
        ]
        assert len(int_edges) == 0


class TestAddInteractionEdges:
    def test_interaction_edges_added(self, sample_stops_close_df):
        stops = build_stops_table(sample_stops_close_df)
        G = build_movement_graph(stops)
        G = add_interaction_edges(G, max_interaction_min=5.0)
        int_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "interaction"
        ]
        # SVC003 and SVC004 within 4 min at MIR and at BGH → 2 edges
        assert len(int_edges) == 2

    def test_no_interaction_beyond_threshold(self, sample_stops_close_df):
        stops = build_stops_table(sample_stops_close_df)
        G = build_movement_graph(stops)
        G = add_interaction_edges(G, max_interaction_min=5.0)
        # SVC005 departs MIR 38 min after SVC004 – should have no interaction
        int_edges = [
            (u, v, d)
            for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "interaction"
        ]
        involved_services = set()
        for u, v, _ in int_edges:
            involved_services.add(G.nodes[u]["service_id"])
            involved_services.add(G.nodes[v]["service_id"])
        assert "SVC005" not in involved_services

    def test_interaction_edge_attributes(self, sample_stops_close_df):
        stops = build_stops_table(sample_stops_close_df)
        G = build_movement_graph(stops)
        G = add_interaction_edges(G, max_interaction_min=5.0)
        for u, v, d in G.edges(data=True):
            if d.get("edge_type") == "interaction":
                assert "gap_min" in d
                assert 0 < d["gap_min"] <= 5.0

    def test_no_same_service_interaction(self, sample_stops_close_df):
        stops = build_stops_table(sample_stops_close_df)
        G = build_movement_graph(stops)
        G = add_interaction_edges(G, max_interaction_min=5.0)
        for u, v, d in G.edges(data=True):
            if d.get("edge_type") == "interaction":
                assert G.nodes[u]["service_id"] != G.nodes[v]["service_id"]

    def test_direction_earlier_to_later(self, sample_stops_close_df):
        stops = build_stops_table(sample_stops_close_df)
        G = build_movement_graph(stops)
        G = add_interaction_edges(G, max_interaction_min=5.0)
        for u, v, d in G.edges(data=True):
            if d.get("edge_type") == "interaction":
                u_attrs = G.nodes[u]
                v_attrs = G.nodes[v]
                t_dep_u = u_attrs.get("scheduled_departure")
                t_u = t_dep_u if (t_dep_u is not None and not pd.isnull(t_dep_u)) else u_attrs.get("scheduled_arrival")
                t_dep_v = v_attrs.get("scheduled_departure")
                t_v = t_dep_v if (t_dep_v is not None and not pd.isnull(t_dep_v)) else v_attrs.get("scheduled_arrival")
                assert pd.Timestamp(t_u) <= pd.Timestamp(t_v)

    def test_no_edges_without_close_events(self, sample_stops_df):
        # Original fixture: SVC001 arrives BHM at 08:30, SVC002 departs at 08:45 → 15 min gap
        stops = build_stops_table(sample_stops_df)
        G = build_movement_graph(stops)
        G = add_interaction_edges(G, max_interaction_min=5.0)
        int_edges = [
            (u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "interaction"
        ]
        assert len(int_edges) == 0


class TestAddDependencyEdgesTurnaround:
    """Tests for add_dependency_edges_turnaround and add_turnaround_edges."""

    def test_returns_list(self, turnaround_stops_df):
        edges = add_dependency_edges_turnaround(turnaround_stops_df)
        assert isinstance(edges, list)

    def test_edge_schema(self, turnaround_stops_df):
        edges = add_dependency_edges_turnaround(turnaround_stops_df)
        for e in edges:
            assert e["type"] == "dependency"
            assert e["subtype"] == "turnaround"
            assert e["confidence"] in ("high", "medium")
            assert isinstance(e["time_diff"], float)
            assert isinstance(e["from"], str)
            assert isinstance(e["to"], str)

    def test_high_confidence_unit_match(self, turnaround_stops_df):
        # SVC_T3 (UNIT_A, arr 12:00) → SVC_T4 (UNIT_A, dep 12:15) → HIGH
        edges = add_dependency_edges_turnaround(turnaround_stops_df)
        high_edges = [e for e in edges if e["confidence"] == "high"]
        from_keys = [e["from"] for e in high_edges]
        to_keys = [e["to"] for e in high_edges]
        expected_from = node_key("SVC_T3", "BGH", datetime(2026, 4, 30, 12, 0))
        expected_to = node_key("SVC_T4", "BGH", datetime(2026, 4, 30, 12, 15))
        assert expected_from in from_keys
        assert expected_to in to_keys

    def test_medium_confidence_time_fallback(self, turnaround_stops_df):
        # SVC_T1 (no unit_id, arr 11:00) → SVC_T2 (no unit_id, dep 11:20) → MEDIUM
        edges = add_dependency_edges_turnaround(turnaround_stops_df)
        medium_edges = [e for e in edges if e["confidence"] == "medium"]
        from_keys = [e["from"] for e in medium_edges]
        to_keys = [e["to"] for e in medium_edges]
        expected_from = node_key("SVC_T1", "BGH", datetime(2026, 4, 30, 11, 0))
        expected_to = node_key("SVC_T2", "BGH", datetime(2026, 4, 30, 11, 20))
        assert expected_from in from_keys
        assert expected_to in to_keys

    def test_platform_promotes_medium_to_high(self, turnaround_stops_df):
        # SVC_T10 (no unit_id, platform '3') → SVC_T11 (no unit_id, platform '3')
        # 20 min gap, same platform → MEDIUM promoted to HIGH
        edges = add_dependency_edges_turnaround(turnaround_stops_df)
        expected_from = node_key("SVC_T10", "BGH", datetime(2026, 4, 30, 10, 15))
        expected_to = node_key("SVC_T11", "BGH", datetime(2026, 4, 30, 10, 35))
        matching = [
            e for e in edges if e["from"] == expected_from and e["to"] == expected_to
        ]
        assert len(matching) == 1
        assert matching[0]["confidence"] == "high"

    def test_no_edge_unit_mismatch(self, turnaround_stops_df):
        # SVC_T6 (UNIT_B) → SVC_T7 (UNIT_C): different units → no edge
        edges = add_dependency_edges_turnaround(turnaround_stops_df)
        from_keys = {e["from"] for e in edges}
        expected_from = node_key("SVC_T6", "BGH", datetime(2026, 4, 30, 13, 0))
        assert expected_from not in from_keys

    def test_no_edge_below_minimum_gap(self, turnaround_stops_df):
        # SVC_T8 (arr 13:30) → SVC_T9 (dep 13:33): gap 3 min < 5 min minimum → no edge
        edges = add_dependency_edges_turnaround(turnaround_stops_df)
        expected_from = node_key("SVC_T8", "BGH", datetime(2026, 4, 30, 13, 30))
        matching = [e for e in edges if e["from"] == expected_from]
        assert len(matching) == 0

    def test_no_edge_above_medium_maximum(self, turnaround_stops_df):
        # SVC_T3 → SVC_T5: 50 min gap, no unit_id → above medium max → no edge
        edges = add_dependency_edges_turnaround(turnaround_stops_df)
        expected_from = node_key("SVC_T3", "BGH", datetime(2026, 4, 30, 12, 0))
        expected_to = node_key("SVC_T5", "BGH", datetime(2026, 4, 30, 12, 50))
        matching = [
            e for e in edges if e["from"] == expected_from and e["to"] == expected_to
        ]
        assert len(matching) == 0

    def test_no_same_service_turnaround(self, turnaround_stops_df):
        edges = add_dependency_edges_turnaround(turnaround_stops_df)
        for e in edges:
            from_svc = e["from"].split("|")[0]
            to_svc = e["to"].split("|")[0]
            assert from_svc != to_svc

    def test_direction_arrival_to_departure(self, turnaround_stops_df):
        # time_diff must be positive (departure after arrival)
        edges = add_dependency_edges_turnaround(turnaround_stops_df)
        for e in edges:
            assert e["time_diff"] > 0

    def test_no_unit_id_column_uses_time_fallback(self):
        # Fixture without unit_id column at all → all matches are medium
        import pandas as pd
        df = pd.DataFrame(
            [
                {
                    "service_id": "A",
                    "station_crs": "XX",
                    "scheduled_arrival": datetime(2026, 4, 30, 10, 0),
                    "scheduled_departure": None,
                },
                {
                    "service_id": "B",
                    "station_crs": "XX",
                    "scheduled_arrival": None,
                    "scheduled_departure": datetime(2026, 4, 30, 10, 20),
                },
            ]
        )
        edges = add_dependency_edges_turnaround(df)
        assert len(edges) == 1
        assert edges[0]["confidence"] == "medium"

    def test_add_turnaround_edges_adds_to_graph(self, turnaround_stops_df):
        # Build a minimal graph whose nodes match the expected turnaround node keys
        G = nx.DiGraph()
        # Add arrival node for SVC_T3 (terminal, keyed by arrival time)
        arr_key = node_key("SVC_T3", "BGH", datetime(2026, 4, 30, 12, 0))
        dep_key = node_key("SVC_T4", "BGH", datetime(2026, 4, 30, 12, 15))
        G.add_node(arr_key, service_id="SVC_T3", station_crs="BGH")
        G.add_node(dep_key, service_id="SVC_T4", station_crs="BGH")
        G = add_turnaround_edges(G, turnaround_stops_df)
        assert G.has_edge(arr_key, dep_key)
        ed = G[arr_key][dep_key]
        assert ed["edge_type"] == "dependency"
        assert ed["subtype"] == "turnaround"
        assert ed["confidence"] == "high"

    def test_add_turnaround_edges_skips_missing_nodes(self, turnaround_stops_df):
        # Empty graph → no edges added (nodes don't exist)
        G = nx.DiGraph()
        G = add_turnaround_edges(G, turnaround_stops_df)
        assert G.number_of_edges() == 0

    def test_build_temporal_graph_with_turnarounds(self, turnaround_stops_df):
        from railway_delay.data_processing import build_stops_table
        # build_stops_table strips unit_id/platform; use raw df directly
        G = build_movement_graph(turnaround_stops_df)
        G = add_turnaround_edges(G, turnaround_stops_df)
        ta_edges = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if d.get("subtype") == "turnaround"
        ]
        assert len(ta_edges) >= 1

    def test_build_temporal_graph_turnarounds_flag(self, turnaround_stops_df):
        G = build_temporal_graph(
            turnaround_stops_df,
            add_dependencies=False,
            add_interactions=False,
            add_turnarounds=True,
        )
        ta_edges = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if d.get("subtype") == "turnaround"
        ]
        assert len(ta_edges) >= 1

    def test_build_temporal_graph_no_turnarounds_by_default(self, turnaround_stops_df):
        G = build_temporal_graph(
            turnaround_stops_df,
            add_dependencies=False,
            add_interactions=False,
        )
        ta_edges = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if d.get("subtype") == "turnaround"
        ]
        assert len(ta_edges) == 0

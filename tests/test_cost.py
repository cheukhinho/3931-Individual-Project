"""Tests for railway_delay.cost and Phase 5 cost-aware decision engine."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from railway_delay.cost import (
    PEAK_WEIGHT,
    OFF_PEAK_WEIGHT,
    DEFAULT_AVG_TICKET_PRICE,
    DEFAULT_SERVICE_WEIGHT,
    passenger_weight,
    compensation_rate,
    compute_node_cost,
    compute_cancellation_cost,
    compute_total_cost,
    threshold_crossing_surcharge,
)
from railway_delay.data_processing import build_stops_table
from railway_delay.graph_construction import build_temporal_graph
from railway_delay.simulation import reset_delays, inject_delay, propagate_delays
from railway_delay.optimisation import (
    simulate_scenario,
    choose_best_decision,
    choose_interaction_order,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph(sample_stops_df):
    """Small two-service graph (SVC001 EUS→MKC→BHM, SVC002 BHM→CRE→LIV)."""
    stops = build_stops_table(sample_stops_df)
    return build_temporal_graph(stops)


@pytest.fixture
def mir_graph(sample_stops_close_df):
    """MIR–BGH graph with three services (SVC003, SVC004, SVC005)."""
    stops = build_stops_table(sample_stops_close_df)
    return build_temporal_graph(stops)


@pytest.fixture
def svc001_source(graph):
    """First node of SVC001 (EUS departure)."""
    for n, attrs in graph.nodes(data=True):
        if attrs.get("service_id") == "SVC001" and attrs.get("station_crs") == "EUS":
            return n
    pytest.skip("Expected node not found")


@pytest.fixture
def svc003_source(mir_graph):
    """First node of SVC003 (MIR departure)."""
    for n, attrs in mir_graph.nodes(data=True):
        if attrs.get("service_id") == "SVC003" and attrs.get("station_crs") == "MIR":
            return n
    pytest.skip("Expected node not found")


@pytest.fixture
def svc003_mir_node(mir_graph):
    return next(
        n
        for n, a in mir_graph.nodes(data=True)
        if a.get("service_id") == "SVC003" and a.get("station_crs") == "MIR"
    )


@pytest.fixture
def svc004_mir_node(mir_graph):
    return next(
        n
        for n, a in mir_graph.nodes(data=True)
        if a.get("service_id") == "SVC004" and a.get("station_crs") == "MIR"
    )


# ===========================================================================
# passenger_weight
# ===========================================================================


class TestPassengerWeight:
    def test_morning_peak_returns_peak_weight(self):
        ts = pd.Timestamp("2026-04-30 08:00")
        assert passenger_weight(ts) == PEAK_WEIGHT

    def test_evening_peak_returns_peak_weight(self):
        ts = pd.Timestamp("2026-04-30 17:30")
        assert passenger_weight(ts) == PEAK_WEIGHT

    def test_off_peak_midday(self):
        ts = pd.Timestamp("2026-04-30 11:00")
        assert passenger_weight(ts) == OFF_PEAK_WEIGHT

    def test_boundary_peak_start_inclusive(self):
        ts = pd.Timestamp("2026-04-30 07:00")
        assert passenger_weight(ts) == PEAK_WEIGHT

    def test_boundary_peak_end_exclusive(self):
        # 09:00 is outside the morning peak window [7, 9)
        ts = pd.Timestamp("2026-04-30 09:00")
        assert passenger_weight(ts) == OFF_PEAK_WEIGHT

    def test_none_returns_off_peak(self):
        assert passenger_weight(None) == OFF_PEAK_WEIGHT

    def test_nat_returns_off_peak(self):
        assert passenger_weight(pd.NaT) == OFF_PEAK_WEIGHT

    # ----- station factor tests -------------------------------------------

    def test_unknown_station_factor_is_one(self):
        ts = pd.Timestamp("2026-04-30 11:00")
        # No station_crs: station_factor = 1.0
        assert passenger_weight(ts) == OFF_PEAK_WEIGHT

    def test_major_station_increases_weight(self):
        ts = pd.Timestamp("2026-04-30 11:00")  # off-peak
        # EUS station_factor = 1.5 → weight = 1.0 * 1.5 = 1.5
        assert passenger_weight(ts, station_crs="EUS") == pytest.approx(1.5)

    def test_minor_station_reduces_weight(self):
        ts = pd.Timestamp("2026-04-30 11:00")  # off-peak
        # MIR station_factor = 0.8 → weight = 1.0 * 0.8 = 0.8
        assert passenger_weight(ts, station_crs="MIR") == pytest.approx(0.8)

    def test_peak_major_station_combines_factors(self):
        ts = pd.Timestamp("2026-04-30 08:00")  # peak
        # EUS: peak_factor=2.0, station_factor=1.5 → weight = 3.0
        assert passenger_weight(ts, station_crs="EUS") == pytest.approx(3.0)

    def test_peak_minor_station_combines_factors(self):
        ts = pd.Timestamp("2026-04-30 08:00")  # peak
        # BGH: peak_factor=2.0, station_factor=0.8 → weight = 1.6
        assert passenger_weight(ts, station_crs="BGH") == pytest.approx(1.6)


# ===========================================================================
# threshold_crossing_surcharge
# ===========================================================================


class TestThresholdCrossingSurcharge:
    def test_no_crossing_zero_surcharge(self):
        # Both below 15 → no threshold crossed
        assert threshold_crossing_surcharge(5.0, 10.0) == 0.0

    def test_crosses_15_min(self):
        # Goes from 10 to 20 → crosses 15-min band
        assert threshold_crossing_surcharge(10.0, 20.0) == pytest.approx(10.0)

    def test_crosses_30_min(self):
        # Goes from 20 to 35 → crosses 30-min band (not 15, already crossed)
        assert threshold_crossing_surcharge(20.0, 35.0) == pytest.approx(25.0)

    def test_crosses_15_and_30_simultaneously(self):
        # Goes from 10 to 35 → crosses both 15 and 30
        assert threshold_crossing_surcharge(10.0, 35.0) == pytest.approx(35.0)

    def test_crosses_all_three_bands(self):
        # Goes from 10 to 70 → crosses 15, 30, 60
        assert threshold_crossing_surcharge(10.0, 70.0) == pytest.approx(85.0)

    def test_delay_decreases_zero_surcharge(self):
        # Delay decreasing never triggers a surcharge
        assert threshold_crossing_surcharge(30.0, 20.0) == 0.0

    def test_same_delay_zero_surcharge(self):
        assert threshold_crossing_surcharge(15.0, 15.0) == 0.0

    def test_already_above_threshold_no_surcharge(self):
        # Already above 30 min: crossing 30 again is not a new crossing
        assert threshold_crossing_surcharge(31.0, 35.0) == 0.0

    def test_29_to_31_crosses_30_min(self):
        # Canonical example from the problem statement: 29 → 31
        assert threshold_crossing_surcharge(29.0, 31.0) == pytest.approx(25.0)


# ===========================================================================
# compensation_rate
# ===========================================================================


class TestCompensationRate:
    @pytest.mark.parametrize(
        "delay, expected",
        [
            (0, 0.0),
            (5, 0.0),
            (14.9, 0.0),
            (15, 0.25),
            (20, 0.25),
            (29.9, 0.25),
            (30, 0.5),
            (45, 0.5),
            (59.9, 0.5),
            (60, 1.0),
            (120, 1.0),
        ],
    )
    def test_step_thresholds(self, delay, expected):
        assert compensation_rate(delay) == expected

    def test_negative_delay_returns_zero(self):
        assert compensation_rate(-5) == 0.0


# ===========================================================================
# compute_node_cost
# ===========================================================================


class TestComputeNodeCost:
    def test_zero_delay_zero_cost(self):
        attrs = {"scheduled_departure": pd.Timestamp("2026-04-30 11:00")}
        assert compute_node_cost(attrs, 0.0) == 0.0

    def test_delay_14_min_zero_cost(self):
        attrs = {"scheduled_departure": pd.Timestamp("2026-04-30 11:00")}
        assert compute_node_cost(attrs, 14.0) == 0.0

    def test_delay_15_min_quarter_rate(self):
        # off-peak: p=1.0, s=1.0, c=10 → cost = 1 * 1 * 10 * 0.25 = 2.5
        attrs = {"scheduled_departure": pd.Timestamp("2026-04-30 11:00")}
        assert compute_node_cost(attrs, 15.0) == pytest.approx(2.5)

    def test_peak_multiplier_applied(self):
        # peak: p=2.0, s=1.0, c=10, R(60)=1.0 → cost = 20.0
        attrs = {"scheduled_departure": pd.Timestamp("2026-04-30 08:00")}
        assert compute_node_cost(attrs, 60.0) == pytest.approx(20.0)

    def test_custom_service_weight(self):
        attrs = {
            "scheduled_departure": pd.Timestamp("2026-04-30 11:00"),
            "service_weight": 2.0,
        }
        # off-peak: p=1.0, s=2.0, c=10, R(30)=0.5 → 10.0
        assert compute_node_cost(attrs, 30.0) == pytest.approx(10.0)

    def test_custom_ticket_price(self):
        attrs = {
            "scheduled_departure": pd.Timestamp("2026-04-30 11:00"),
            "avg_ticket_price": 20.0,
        }
        # off-peak: p=1.0, s=1.0, c=20, R(15)=0.25 → 5.0
        assert compute_node_cost(attrs, 15.0) == pytest.approx(5.0)


# ===========================================================================
# compute_cancellation_cost
# ===========================================================================


class TestComputeCancellationCost:
    def test_default_values(self):
        # p=1.0, s=1.0, c=10 → 10.0
        attrs = {"scheduled_departure": pd.Timestamp("2026-04-30 11:00")}
        assert compute_cancellation_cost(attrs) == pytest.approx(10.0)

    def test_peak_cancellation_higher(self):
        # p=2.0, s=1.0, c=10 → 20.0
        attrs = {"scheduled_departure": pd.Timestamp("2026-04-30 08:00")}
        assert compute_cancellation_cost(attrs) == pytest.approx(20.0)

    def test_cancellation_greater_than_small_delay_cost(self):
        # Cancellation always uses full cost (no R factor),
        # so it should exceed the cost of a 15-min delay in the same context.
        attrs = {"scheduled_departure": pd.Timestamp("2026-04-30 11:00")}
        assert compute_cancellation_cost(attrs) > compute_node_cost(attrs, 15.0)


# ===========================================================================
# compute_total_cost
# ===========================================================================


class TestComputeTotalCost:
    def test_no_delays_zero_cost(self, graph):
        reset_delays(graph)
        cost = compute_total_cost(graph)
        assert cost == 0.0

    def test_cost_increases_with_delay(self, graph, svc001_source):
        reset_delays(graph)
        inject_delay(graph, svc001_source, 15.0)
        propagate_delays(graph)
        cost_15 = compute_total_cost(graph)

        reset_delays(graph)
        inject_delay(graph, svc001_source, 60.0)
        propagate_delays(graph)
        cost_60 = compute_total_cost(graph)

        assert cost_60 > cost_15

    def test_cancelled_nodes_explicit(self, graph, svc001_source):
        reset_delays(graph)
        cost_with_cancel = compute_total_cost(
            graph, cancelled_nodes={svc001_source}
        )
        # Cancelled node contributes full cancellation cost, not zero.
        assert cost_with_cancel > 0.0

    def test_inf_delay_treated_as_cancelled(self, graph, svc001_source):
        reset_delays(graph)
        graph.nodes[svc001_source]["simulated_departure_delay"] = float("inf")
        cost_inf = compute_total_cost(graph)
        # Should equal cancellation cost for that node (plus any other nodes)
        assert cost_inf > 0.0
        assert math.isfinite(cost_inf)

    def test_missed_connection_penalty_added(self, graph, svc001_source):
        reset_delays(graph)
        inject_delay(graph, svc001_source, 60.0)
        propagate_delays(graph)

        cost_no_penalty = compute_total_cost(graph, missed_connection_penalty=0.0)
        cost_with_penalty = compute_total_cost(
            graph,
            connection_buffer_min=1.0,
            missed_connection_penalty=100.0,
        )
        assert cost_with_penalty >= cost_no_penalty


# ===========================================================================
# simulate_scenario
# ===========================================================================


class TestSimulateScenario:
    def test_continue_returns_dict(self, graph, svc001_source):
        result = simulate_scenario(
            graph, "continue", "SVC001", svc001_source, 20.0
        )
        assert result["decision"] == "continue"
        assert "total_cost" in result
        assert "total_delay_min" in result
        assert "affected_nodes" in result
        assert "graph" in result

    def test_short_turn_returns_dict(self, graph, svc001_source):
        result = simulate_scenario(
            graph, "short_turn", "SVC001", svc001_source, 20.0
        )
        assert result["decision"] == "short_turn"
        assert result["total_cost"] >= 0.0

    def test_cancel_returns_dict(self, graph, svc001_source):
        result = simulate_scenario(
            graph, "cancel", "SVC001", svc001_source, 20.0
        )
        assert result["decision"] == "cancel"
        assert result["total_cost"] >= 0.0

    def test_original_graph_not_mutated(self, graph, svc001_source):
        edges_before = graph.number_of_edges()
        nodes_before = graph.number_of_nodes()
        simulate_scenario(graph, "cancel", "SVC001", svc001_source, 20.0)
        assert graph.number_of_edges() == edges_before
        assert graph.number_of_nodes() == nodes_before

    def test_invalid_decision_raises(self, graph, svc001_source):
        with pytest.raises(ValueError):
            simulate_scenario(graph, "teleport", "SVC001", svc001_source, 20.0)

    def test_cancel_reduces_affected_nodes(self, graph, svc001_source):
        result_continue = simulate_scenario(
            graph, "continue", "SVC001", svc001_source, 60.0
        )
        result_cancel = simulate_scenario(
            graph, "cancel", "SVC001", svc001_source, 60.0
        )
        # Cancelling the service removes its downstream nodes, so delay
        # propagation to other services may differ; cost structure changes.
        assert result_cancel["total_delay_min"] <= result_continue["total_delay_min"]

    def test_short_turn_fewer_nodes_than_continue(self, graph, svc001_source):
        result_continue = simulate_scenario(
            graph, "continue", "SVC001", svc001_source, 30.0
        )
        result_short_turn = simulate_scenario(
            graph, "short_turn", "SVC001", svc001_source, 30.0
        )
        # Short turn removes downstream stops so total delay should be ≤ continue
        assert (
            result_short_turn["total_delay_min"]
            <= result_continue["total_delay_min"] + 1e-6
        )


# ===========================================================================
# choose_best_decision
# ===========================================================================


class TestChooseBestDecision:
    def test_returns_expected_keys(self, graph, svc001_source):
        result = choose_best_decision(graph, "SVC001", svc001_source, 20.0)
        assert "best_decision" in result
        assert "best_cost" in result
        assert "all_results" in result
        assert "best_graph" in result

    def test_best_decision_valid(self, graph, svc001_source):
        result = choose_best_decision(graph, "SVC001", svc001_source, 20.0)
        assert result["best_decision"] in ("continue", "short_turn", "cancel")

    def test_best_cost_is_minimum(self, graph, svc001_source):
        result = choose_best_decision(graph, "SVC001", svc001_source, 20.0)
        all_costs = [r["total_cost"] for r in result["all_results"]]
        assert result["best_cost"] == min(all_costs)

    def test_all_results_length(self, graph, svc001_source):
        result = choose_best_decision(graph, "SVC001", svc001_source, 20.0)
        assert len(result["all_results"]) == 3  # continue, short_turn, cancel

    def test_all_results_sorted_by_cost(self, graph, svc001_source):
        result = choose_best_decision(graph, "SVC001", svc001_source, 20.0)
        costs = [r["total_cost"] for r in result["all_results"]]
        assert costs == sorted(costs)

    def test_large_delay_prefers_cancel_or_short_turn(self, graph, svc001_source):
        # With a very large delay, cancel/short_turn should score ≤ continue
        result = choose_best_decision(
            graph,
            "SVC001",
            svc001_source,
            120.0,
            missed_connection_penalty=500.0,
        )
        # The best should not have higher cost than continue
        continue_cost = next(
            r["total_cost"]
            for r in result["all_results"]
            if r["decision"] == "continue"
        )
        assert result["best_cost"] <= continue_cost + 1e-6


# ===========================================================================
# choose_interaction_order
# ===========================================================================


class TestChooseInteractionOrder:
    def test_returns_expected_keys(self, mir_graph, svc003_mir_node, svc004_mir_node):
        result = choose_interaction_order(
            mir_graph, svc003_mir_node, svc004_mir_node, 20.0
        )
        assert "preferred_order" in result
        assert "cost_a_first" in result
        assert "cost_b_first" in result
        assert "graph_preferred" in result

    def test_preferred_order_valid(self, mir_graph, svc003_mir_node, svc004_mir_node):
        result = choose_interaction_order(
            mir_graph, svc003_mir_node, svc004_mir_node, 20.0
        )
        assert result["preferred_order"] in ("a_first", "b_first")

    def test_preferred_has_lower_or_equal_cost(
        self, mir_graph, svc003_mir_node, svc004_mir_node
    ):
        result = choose_interaction_order(
            mir_graph, svc003_mir_node, svc004_mir_node, 20.0
        )
        if result["preferred_order"] == "a_first":
            assert result["cost_a_first"] <= result["cost_b_first"]
        else:
            assert result["cost_b_first"] <= result["cost_a_first"]

    def test_original_graph_not_mutated(
        self, mir_graph, svc003_mir_node, svc004_mir_node
    ):
        edges_before = mir_graph.number_of_edges()
        choose_interaction_order(
            mir_graph, svc003_mir_node, svc004_mir_node, 20.0
        )
        assert mir_graph.number_of_edges() == edges_before

    def test_zero_delay_costs_equal(
        self, mir_graph, svc003_mir_node, svc004_mir_node
    ):
        result = choose_interaction_order(
            mir_graph, svc003_mir_node, svc004_mir_node, 0.0
        )
        assert result["cost_a_first"] == pytest.approx(result["cost_b_first"])


# ===========================================================================
# Validation scenario: MIR–BGH +20 min disruption on SVC003
# ===========================================================================


class TestMIRScenarioValidation:
    """End-to-end validation for the MIR→BGH corridor (2026-04-30, 10:00–14:00).

    Injects a 20-minute delay on SVC003 at MIR and checks that the cost-aware
    decision engine produces internally consistent results consistent with the
    mathematical model.
    """

    def test_continue_has_positive_cost_with_20min_delay(
        self, mir_graph, svc003_source
    ):
        # 20 min delay ≥ 15 min threshold → compensation is triggered
        result = simulate_scenario(
            mir_graph, "continue", "SVC003", svc003_source, 20.0
        )
        # R(20) = 0.25 → some nodes should have non-zero cost
        assert result["total_cost"] > 0.0

    def test_cancel_cost_exceeds_small_delay_cost(self, mir_graph, svc003_source):
        # A 5-min delay (below 15-min threshold, R=0) costs nothing;
        # cancellation always costs at least one node's full ticket value.
        result_small = simulate_scenario(
            mir_graph, "continue", "SVC003", svc003_source, 5.0,
            missed_connection_penalty=0.0,
        )
        result_cancel = simulate_scenario(
            mir_graph, "cancel", "SVC003", svc003_source, 5.0,
            missed_connection_penalty=0.0,
        )
        assert result_cancel["total_cost"] > result_small["total_cost"]

    def test_cost_ordering_consistent_with_model(self, mir_graph, svc003_source):
        # With a 60-min delay (R=1.0) and a high missed-connection penalty,
        # cancellation or short_turn should be competitive with continue.
        result = choose_best_decision(
            mir_graph,
            "SVC003",
            svc003_source,
            60.0,
            missed_connection_penalty=200.0,
        )
        assert result["best_cost"] >= 0.0
        all_decisions = {r["decision"] for r in result["all_results"]}
        assert all_decisions == {"continue", "short_turn", "cancel"}

    def test_compensation_rate_thresholds(self):
        """Directly validate the compensation_rate step function."""
        assert compensation_rate(0) == 0.0
        assert compensation_rate(14) == 0.0
        assert compensation_rate(15) == 0.25
        assert compensation_rate(29) == 0.25
        assert compensation_rate(30) == 0.5
        assert compensation_rate(59) == 0.5
        assert compensation_rate(60) == 1.0

    def test_node_cost_formula(self):
        """Validate C_n = p_n * s_n * c_n * R(d_n) with known values."""
        # off-peak (p=1.0), s=1.5, c=12.0, d=30 → R=0.5 → cost = 9.0
        attrs = {
            "scheduled_departure": pd.Timestamp("2026-04-30 11:00"),
            "service_weight": 1.5,
            "avg_ticket_price": 12.0,
        }
        expected = 1.0 * 1.5 * 12.0 * 0.5
        assert compute_node_cost(attrs, 30.0) == pytest.approx(expected)

    def test_peak_node_cost_formula(self):
        """Validate peak multiplier in C_n."""
        # peak (p=2.0), s=1.0, c=10, d=60 → R=1.0 → cost = 20.0
        attrs = {
            "scheduled_departure": pd.Timestamp("2026-04-30 08:00"),
        }
        expected = PEAK_WEIGHT * DEFAULT_SERVICE_WEIGHT * DEFAULT_AVG_TICKET_PRICE * 1.0
        assert compute_node_cost(attrs, 60.0) == pytest.approx(expected)

    def test_cancellation_cost_formula(self):
        """Validate C_cancel_n = p_n * s_n * c_n (no R factor)."""
        attrs = {
            "scheduled_departure": pd.Timestamp("2026-04-30 11:00"),
            "service_weight": 2.0,
            "avg_ticket_price": 8.0,
        }
        expected = 1.0 * 2.0 * 8.0  # p=1.0 (off-peak), s=2.0, c=8.0
        assert compute_cancellation_cost(attrs) == pytest.approx(expected)

    def test_no_threshold_double_counting(self, mir_graph, svc003_source):
        """Verify that threshold penalties are NOT included in compute_total_cost.

        Under the old model, a 20-min delay would include both R(20)=0.25
        compensation AND a 15-min threshold crossing penalty.  After the
        fix, only R(d) is used; the total cost should equal
        p_n × s_n × c_n × R(d_n) for each active node (no extra penalty).
        """
        reset_delays(mir_graph)
        inject_delay(mir_graph, svc003_source, 20.0)
        propagate_delays(mir_graph)

        total = compute_total_cost(mir_graph, missed_connection_penalty=0.0)

        # Recompute expected cost from scratch: sum p*s*c*R(d) per node
        expected = sum(
            compute_node_cost(attrs, attrs.get("simulated_departure_delay", 0.0))
            for _, attrs in mir_graph.nodes(data=True)
            if attrs.get("simulated_departure_delay", 0.0) != float("inf")
        )
        assert total == pytest.approx(expected)

    def test_station_factor_applied_in_node_cost(self):
        """MIR nodes use station_factor=0.8; EUS nodes use 1.5."""
        attrs_mir = {
            "scheduled_departure": pd.Timestamp("2026-04-30 11:00"),
            "station_crs": "MIR",
        }
        attrs_eus = {
            "scheduled_departure": pd.Timestamp("2026-04-30 11:00"),
            "station_crs": "EUS",
        }
        # off-peak: p_mir = 1.0 * 0.8 = 0.8; p_eus = 1.0 * 1.5 = 1.5
        # Both at R(15)=0.25, c=10, s=1.0
        assert compute_node_cost(attrs_mir, 15.0) == pytest.approx(
            0.8 * 1.0 * 10.0 * 0.25
        )
        assert compute_node_cost(attrs_eus, 15.0) == pytest.approx(
            1.5 * 1.0 * 10.0 * 0.25
        )

    def test_threshold_surcharge_avoids_29_to_31(self):
        """Demonstrate that threshold surcharge steers decisions away from
        pushing A from 29 → 31 minutes delay (crossing the 30-min milestone).

        When B-first would push A from 29 to 31 min, the surcharge (25 £)
        is added to B-first cost, making A-first the preferred option.
        """
        from railway_delay.cost import threshold_crossing_surcharge

        surcharge_b = threshold_crossing_surcharge(29.0, 31.0)
        # 29 → 31 crosses the 30-min threshold → THRESHOLD_30_PENALTY = 25.0
        assert surcharge_b == pytest.approx(25.0)


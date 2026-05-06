"""
optimisation.py
===============
Disruption response optimisation: greedy heuristic (delay-based) and
cost-aware decision engine (Phase 5).

Greedy heuristic (Phase 4)
--------------------------
Given a disrupted temporal graph, choose the best *action* for each delayed
service to minimise the total delay across the network.

Available actions
-----------------
``"no_action"``
    Do nothing; the delay propagates naturally.

``"delay_departure"``
    Hold the train at the current station for *n* extra minutes to
    wait for a connecting service.  Sometimes reduces secondary delays at
    the cost of a primary delay increase.

``"cancel_service"``
    Cancel the delayed service entirely.  Sets its departure delay to
    ``+inf`` (no further propagation from this node).

``"short_turn"``
    Terminate and reverse the service at an intermediate station, reducing
    the number of downstream nodes affected.

Greedy strategy
---------------
For each delayed node (sorted by descending departure delay), evaluate all
available actions and choose the one that yields the lowest *total network
delay* after re-simulation.

The greedy pass is single-shot (no look-ahead) which keeps complexity linear
in the number of delayed nodes.

Cost-aware decision engine (Phase 5)
-------------------------------------
:func:`simulate_scenario`
    Apply one of ``"continue"``, ``"short_turn"``, or ``"cancel"`` to a
    disrupted service, propagate delays, and compute the total monetary cost
    using the :mod:`~railway_delay.cost` module.

:func:`choose_best_decision`
    Evaluate all three scenario decisions and return the one with the lowest
    total cost together with a full comparison table.

:func:`choose_interaction_order`
    At an interaction edge between two nodes (local conflict), compare
    which ordering minimises total cost and return the preferred choice.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import networkx as nx

from railway_delay import simulation as _sim
from railway_delay import evaluation as _eval
from railway_delay import cost as _cost

logger = logging.getLogger(__name__)

ACTIONS = ("no_action", "delay_departure", "cancel_service", "short_turn")

#: Decisions available to the cost-aware engine.
DECISIONS = ("continue", "short_turn", "cancel")


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------


def apply_action(
    G: nx.DiGraph,
    node_key: str,
    action: str,
    *,
    hold_minutes: float = 5.0,
    short_turn_depth: int = 1,
) -> nx.DiGraph:
    """Apply a single action to *node_key* on a **copy** of the graph.

    Parameters
    ----------
    G:
        Temporal graph (already simulated).
    node_key:
        Target node.
    action:
        One of ``"no_action"``, ``"delay_departure"``,
        ``"cancel_service"``, ``"short_turn"``.
    hold_minutes:
        Extra dwell added by ``"delay_departure"`` (minutes).
    short_turn_depth:
        Retained for API compatibility.  The ``"short_turn"`` action now
        removes **all** downstream service nodes beyond the turn point
        regardless of this value (consistent with
        :func:`_apply_decision`).

    Returns
    -------
    nx.DiGraph
        Modified copy of the graph.
    """
    G2 = copy.deepcopy(G)

    if action == "no_action":
        pass

    elif action == "delay_departure":
        # Intentionally increase the departure delay – useful when holding
        # the train helps connecting passengers.
        attrs = G2.nodes[node_key]
        attrs["simulated_departure_delay"] = (
            attrs.get("simulated_departure_delay", 0.0) + hold_minutes
        )

    elif action == "cancel_service":
        # Remove all other nodes of this service (automatically removes all
        # their edges too).  For the source node itself, mark it as cancelled
        # (delay = inf) and remove ALL connected edges so no propagation can
        # leak through interaction or dependency edges.
        service_id = G2.nodes[node_key].get("service_id", "")
        other_nodes = [
            n
            for n, a in G2.nodes(data=True)
            if a.get("service_id") == service_id and n != node_key
        ]
        G2.remove_nodes_from(other_nodes)
        if node_key in G2:
            G2.nodes[node_key]["simulated_departure_delay"] = float("inf")
            # Remove ALL edges (movement, dependency, interaction) so the
            # cancelled node cannot propagate delay to other services.
            all_edges = list(G2.in_edges(node_key)) + list(G2.out_edges(node_key))
            G2.remove_edges_from(all_edges)

    elif action == "short_turn":
        # Remove all downstream nodes of this service beyond the turn point.
        # NetworkX automatically removes all edges connected to removed nodes,
        # so dependency and interaction edges are cleaned up as well.
        service_id = G2.nodes[node_key].get("service_id", "")
        source_stop = G2.nodes[node_key].get("stop_index", 0)
        nodes_to_remove = [
            n
            for n, a in G2.nodes(data=True)
            if a.get("service_id") == service_id
            and a.get("stop_index", 0) > source_stop
        ]
        G2.remove_nodes_from(nodes_to_remove)

    else:
        raise ValueError(f"Unknown action {action!r}.")

    return G2


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def _total_delay(G: nx.DiGraph) -> float:
    """Sum of all finite simulated departure delays across the network."""
    return sum(
        attrs.get("simulated_departure_delay", 0.0)
        for _, attrs in G.nodes(data=True)
        if attrs.get("simulated_departure_delay", 0.0) != float("inf")
    )


# ---------------------------------------------------------------------------
# Greedy optimiser
# ---------------------------------------------------------------------------


def greedy_optimise(
    G: nx.DiGraph,
    disruptions: list[dict[str, Any]],
    candidate_actions: tuple[str, ...] = ACTIONS,
    hold_minutes: float = 5.0,
    short_turn_depth: int = 1,
    min_dwell_min: float = 0.5,
    min_connection_min: float = 2.0,
    threshold_delay_min: float = 1.0,
) -> dict[str, Any]:
    """Greedily assign the best response action to each delayed node.

    The optimisation criterion is **total monetary cost** (computed via
    :func:`~railway_delay.cost.compute_total_cost`) rather than raw delay
    minutes.  For each delayed node the action that yields the lowest cost
    after re-simulation is applied.

    Parameters
    ----------
    G:
        Temporal graph *before* disruptions are injected.
    disruptions:
        List of disruption dicts (same format as for
        :func:`~railway_delay.simulation.run_simulation`).
    candidate_actions:
        Tuple of action names to evaluate.
    hold_minutes:
        Extra hold for ``"delay_departure"`` action (minutes).
    short_turn_depth:
        Retained for API compatibility; ``"short_turn"`` now removes all
        downstream nodes beyond the turn point.
    min_dwell_min:
        Minimum dwell passed to simulation.
    min_connection_min:
        Minimum connection gap passed to simulation.
    threshold_delay_min:
        Only consider nodes with departure delay ≥ this value (minutes).

    Returns
    -------
    dict with keys:
        ``"actions"``         – dict mapping node_key → chosen action
        ``"baseline_delay"``  – total finite delay before optimisation (min)
        ``"optimised_delay"`` – total finite delay after optimisation (min)
        ``"baseline_cost"``   – total monetary cost before optimisation
        ``"optimised_cost"``  – total monetary cost after optimisation
        ``"graph"``           – final optimised graph
    """
    # Run baseline simulation
    G_base = copy.deepcopy(G)
    _sim.run_simulation(G_base, disruptions, min_dwell_min, min_connection_min)
    baseline_delay = _total_delay(G_base)
    baseline_cost = _cost.compute_total_cost(G_base)

    # Identify delayed nodes sorted by descending delay (most affected first)
    delayed_nodes = sorted(
        [
            (n, G_base.nodes[n].get("simulated_departure_delay", 0.0))
            for n in G_base.nodes
            if G_base.nodes[n].get("simulated_departure_delay", 0.0)
            >= threshold_delay_min
        ],
        key=lambda x: -x[1],
    )

    actions_chosen: dict[str, str] = {}
    G_working = copy.deepcopy(G_base)

    for node_key_iter, current_delay in delayed_nodes:
        # Skip nodes that were removed by a previous action (e.g. short_turn
        # removed downstream nodes)
        if node_key_iter not in G_working:
            actions_chosen[node_key_iter] = "no_action"
            continue

        best_action = "no_action"
        best_total_cost = _cost.compute_total_cost(G_working)

        for action in candidate_actions:
            if action == "no_action":
                continue
            G_trial = apply_action(
                G_working,
                node_key_iter,
                action,
                hold_minutes=hold_minutes,
                short_turn_depth=short_turn_depth,
            )
            # Re-propagate on the trial graph
            _sim.propagate_delays(G_trial, min_dwell_min, min_connection_min)
            trial_cost = _cost.compute_total_cost(G_trial)
            if trial_cost < best_total_cost:
                best_total_cost = trial_cost
                best_action = action

        if best_action != "no_action":
            G_working = apply_action(
                G_working,
                node_key_iter,
                best_action,
                hold_minutes=hold_minutes,
                short_turn_depth=short_turn_depth,
            )
            _sim.propagate_delays(G_working, min_dwell_min, min_connection_min)

        actions_chosen[node_key_iter] = best_action

    optimised_delay = _total_delay(G_working)
    optimised_cost = _cost.compute_total_cost(G_working)
    logger.info(
        "Greedy optimisation: baseline_delay=%.1f min (cost=%.2f), "
        "optimised_delay=%.1f min (cost=%.2f)",
        baseline_delay,
        baseline_cost,
        optimised_delay,
        optimised_cost,
    )

    return {
        "actions": actions_chosen,
        "baseline_delay": baseline_delay,
        "optimised_delay": optimised_delay,
        "baseline_cost": baseline_cost,
        "optimised_cost": optimised_cost,
        "graph": G_working,
    }


# ===========================================================================
# Phase 5 – Cost-aware decision engine
# ===========================================================================


def _apply_decision(
    G: nx.DiGraph,
    decision: str,
    disrupted_service: str,
    source_node: str,
    initial_delay: float,
) -> nx.DiGraph:
    """Return a modified copy of *G* reflecting *decision* for *disrupted_service*.

    Parameters
    ----------
    G:
        Base temporal graph (delays should be reset beforehand).
    decision:
        One of ``"continue"``, ``"short_turn"``, ``"cancel"``.
    disrupted_service:
        ``service_id`` of the disrupted service.
    source_node:
        Node key where the initial delay is injected.
    initial_delay:
        Delay magnitude in minutes.

    Returns
    -------
    nx.DiGraph
        Deep copy of *G* with the decision applied and the initial delay
        injected.  Delay has **not** been propagated yet.
    """
    G2 = copy.deepcopy(G)
    _sim.reset_delays(G2)

    if decision == "continue":
        # No structural change – simply inject the delay.
        _sim.inject_delay(G2, source_node, initial_delay)

    elif decision == "short_turn":
        # Remove all downstream movement nodes/edges belonging to the
        # disrupted service that lie *after* source_node.
        service_nodes = sorted(
            [
                n
                for n, attrs in G2.nodes(data=True)
                if attrs.get("service_id") == disrupted_service
            ],
            key=lambda n: G2.nodes[n].get("stop_index", 0),
        )
        source_stop = G2.nodes[source_node].get("stop_index", 0)
        nodes_to_remove = [
            n
            for n in service_nodes
            if G2.nodes[n].get("stop_index", 0) > source_stop
        ]
        G2.remove_nodes_from(nodes_to_remove)
        # Inject the delay on the short-turn point itself.
        if source_node in G2:
            _sim.inject_delay(G2, source_node, initial_delay)

    elif decision == "cancel":
        # Remove all nodes and edges belonging to the disrupted service, then
        # keep the source node as a "ghost" so cost calculations can account
        # for the cancellation.  Remove ALL edges from the source node so that
        # no delay can propagate via interaction or dependency connections.
        service_nodes = [
            n
            for n, attrs in G2.nodes(data=True)
            if attrs.get("service_id") == disrupted_service
        ]
        other_service_nodes = [n for n in service_nodes if n != source_node]
        G2.remove_nodes_from(other_service_nodes)
        if source_node in G2:
            G2.nodes[source_node]["simulated_departure_delay"] = float("inf")
            # Remove ALL edges (movement, dependency, interaction) to prevent
            # any leakage of delay into or out of the cancelled node.
            all_edges = (
                list(G2.in_edges(source_node)) + list(G2.out_edges(source_node))
            )
            G2.remove_edges_from(all_edges)

    else:
        raise ValueError(
            f"Unknown decision {decision!r}. Choose 'continue', 'short_turn', or 'cancel'."
        )

    return G2


def simulate_scenario(
    graph: nx.DiGraph,
    decision: str,
    disrupted_service: str,
    source_node: str,
    initial_delay: float,
    *,
    min_dwell_min: float = 0.5,
    min_connection_min: float = 2.0,
    alpha: float = 0.5,
    max_interaction_min: float = 5.0,
    connection_buffer_min: float = _cost.DEFAULT_CONNECTION_BUFFER_MIN,
    missed_connection_penalty: float = _cost.DEFAULT_MISSED_CONNECTION_PENALTY,
) -> dict[str, Any]:
    """Simulate one disruption management decision and compute its total cost.

    Steps
    -----
    1. Deep-copy *graph* and reset delays.
    2. Modify the graph according to *decision*:
       - ``"continue"``    → no structural change; inject delay at *source_node*.
       - ``"short_turn"``  → remove downstream nodes of *disrupted_service*
                             beyond *source_node*; inject delay there.
       - ``"cancel"``      → remove all nodes of *disrupted_service* (except
                             a ghost source node marked with ``delay = inf``).
    3. Propagate delays through the modified graph.
    4. Compute total cost via :func:`~railway_delay.cost.compute_total_cost`.

    Parameters
    ----------
    graph:
        Base temporal graph (before disruption).
    decision:
        One of ``"continue"``, ``"short_turn"``, ``"cancel"``.
    disrupted_service:
        ``service_id`` of the disrupted service.
    source_node:
        Node key where the initial delay originates.
    initial_delay:
        Delay magnitude in minutes.
    min_dwell_min:
        Minimum dwell time passed to :func:`~railway_delay.simulation.propagate_delays`.
    min_connection_min:
        Minimum connection gap passed to :func:`~railway_delay.simulation.propagate_delays`.
    alpha:
        Base interaction strength (0–1).
    max_interaction_min:
        Maximum interaction gap for alpha calculation.
    connection_buffer_min:
        Threshold (minutes) for missed connection penalty.
    missed_connection_penalty:
        Monetary penalty per node exceeding the connection buffer.

    Returns
    -------
    dict with keys:
        ``"decision"``         – the decision string.
        ``"total_cost"``       – total monetary cost.
        ``"total_delay_min"``  – sum of all finite departure delays (minutes).
        ``"affected_nodes"``   – number of nodes with departure delay > 0.
        ``"graph"``            – the modified/simulated graph.
    """
    G2 = _apply_decision(
        graph, decision, disrupted_service, source_node, initial_delay
    )

    # Propagate delays through the modified graph.  For "cancel", the source
    # node already has delay=inf; propagation still runs so that any residual
    # interaction/dependency effects on other services are captured.
    _sim.propagate_delays(
        G2,
        min_dwell_min=min_dwell_min,
        min_connection_min=min_connection_min,
        alpha=alpha,
        max_interaction_min=max_interaction_min,
    )

    total_cost = _cost.compute_total_cost(
        G2,
        connection_buffer_min=connection_buffer_min,
        missed_connection_penalty=missed_connection_penalty,
    )
    total_delay = sum(
        attrs.get("simulated_departure_delay", 0.0)
        for _, attrs in G2.nodes(data=True)
        if attrs.get("simulated_departure_delay", 0.0) != float("inf")
    )
    affected_nodes = sum(
        1
        for _, attrs in G2.nodes(data=True)
        if attrs.get("simulated_departure_delay", 0.0) > 0
        and attrs.get("simulated_departure_delay", 0.0) != float("inf")
    )

    return {
        "decision": decision,
        "total_cost": total_cost,
        "total_delay_min": total_delay,
        "affected_nodes": affected_nodes,
        "graph": G2,
    }


def choose_best_decision(
    graph: nx.DiGraph,
    disrupted_service: str,
    source_node: str,
    initial_delay: float,
    *,
    decisions: tuple[str, ...] = DECISIONS,
    min_dwell_min: float = 0.5,
    min_connection_min: float = 2.0,
    alpha: float = 0.5,
    max_interaction_min: float = 5.0,
    connection_buffer_min: float = _cost.DEFAULT_CONNECTION_BUFFER_MIN,
    missed_connection_penalty: float = _cost.DEFAULT_MISSED_CONNECTION_PENALTY,
) -> dict[str, Any]:
    """Evaluate all disruption decisions and return the one with minimum cost.

    For each decision in *decisions*, :func:`simulate_scenario` is called and
    the total monetary cost is computed.  The decision with the lowest cost is
    selected.

    Parameters
    ----------
    graph:
        Base temporal graph (before disruption).
    disrupted_service:
        ``service_id`` of the disrupted service.
    source_node:
        Node key where the initial delay originates.
    initial_delay:
        Delay magnitude in minutes.
    decisions:
        Tuple of decision strings to evaluate.  Defaults to
        ``("continue", "short_turn", "cancel")``.
    min_dwell_min:
        Minimum dwell time for propagation.
    min_connection_min:
        Minimum connection gap for propagation.
    alpha:
        Base interaction strength.
    max_interaction_min:
        Maximum interaction gap for alpha calculation.
    connection_buffer_min:
        Missed connection threshold (minutes).
    missed_connection_penalty:
        Monetary penalty per missed connection node.

    Returns
    -------
    dict with keys:
        ``"best_decision"``    – name of the selected decision.
        ``"best_cost"``        – total cost under the selected decision.
        ``"all_results"``      – list of :func:`simulate_scenario` result dicts
                                 (one per decision, sorted by cost ascending).
        ``"best_graph"``       – graph after applying the best decision.
    """
    results = []
    for decision in decisions:
        result = simulate_scenario(
            graph,
            decision,
            disrupted_service,
            source_node,
            initial_delay,
            min_dwell_min=min_dwell_min,
            min_connection_min=min_connection_min,
            alpha=alpha,
            max_interaction_min=max_interaction_min,
            connection_buffer_min=connection_buffer_min,
            missed_connection_penalty=missed_connection_penalty,
        )
        results.append(result)
        logger.debug(
            "Decision %-10s  cost=%.2f  total_delay=%.1f min  affected=%d",
            decision,
            result["total_cost"],
            result["total_delay_min"],
            result["affected_nodes"],
        )

    results_sorted = sorted(results, key=lambda r: r["total_cost"])
    best = results_sorted[0]

    logger.info(
        "Best decision: %s (cost=%.2f)",
        best["decision"],
        best["total_cost"],
    )

    return {
        "best_decision": best["decision"],
        "best_cost": best["total_cost"],
        "all_results": results_sorted,
        "best_graph": best["graph"],
    }


def choose_interaction_order(
    graph: nx.DiGraph,
    node_a: str,
    node_b: str,
    initial_delay: float,
    *,
    min_dwell_min: float = 0.5,
    min_connection_min: float = 2.0,
    alpha: float = 0.5,
    max_interaction_min: float = 5.0,
    connection_buffer_min: float = _cost.DEFAULT_CONNECTION_BUFFER_MIN,
    missed_connection_penalty: float = _cost.DEFAULT_MISSED_CONNECTION_PENALTY,
) -> dict[str, Any]:
    """Local optimisation at an interaction edge: choose which train goes first.

    At a conflict between *node_a* and *node_b* (connected by an interaction
    edge), one train must yield and absorb the delay.  This function evaluates
    two scenarios:

    - **A first**: the initial delay is injected at *node_a* (A is prioritised;
      B may receive a fraction of A's delay via the interaction edge).
    - **B first**: the initial delay is injected at *node_b* (B is prioritised;
      A may receive a fraction of B's delay via the interaction edge).

    The scenario with the lower total cost is returned as the preferred ordering.

    Parameters
    ----------
    graph:
        Base temporal graph.
    node_a:
        First node in the interaction conflict.
    node_b:
        Second node in the interaction conflict.
    initial_delay:
        Delay magnitude in minutes to inject at the "held" node.
    min_dwell_min:
        Minimum dwell time for propagation.
    min_connection_min:
        Minimum connection gap for propagation.
    alpha:
        Base interaction strength.
    max_interaction_min:
        Maximum interaction gap for alpha calculation.
    connection_buffer_min:
        Missed connection threshold (minutes).
    missed_connection_penalty:
        Monetary penalty per missed connection node.

    Returns
    -------
    dict with keys:
        ``"preferred_order"``  – ``"a_first"`` or ``"b_first"``.
        ``"cost_a_first"``     – total cost when A is prioritised.
        ``"cost_b_first"``     – total cost when B is prioritised.
        ``"graph_preferred"``  – graph after the preferred scenario.
    """
    def _run(source: str) -> tuple[float, nx.DiGraph]:
        G2 = copy.deepcopy(graph)
        _sim.reset_delays(G2)
        _sim.inject_delay(G2, source, initial_delay)
        _sim.propagate_delays(
            G2,
            min_dwell_min=min_dwell_min,
            min_connection_min=min_connection_min,
            alpha=alpha,
            max_interaction_min=max_interaction_min,
        )
        cost = _cost.compute_total_cost(
            G2,
            connection_buffer_min=connection_buffer_min,
            missed_connection_penalty=missed_connection_penalty,
        )
        return cost, G2

    cost_a, graph_a = _run(node_a)
    cost_b, graph_b = _run(node_b)

    if cost_a <= cost_b:
        preferred = "a_first"
        graph_preferred = graph_a
    else:
        preferred = "b_first"
        graph_preferred = graph_b

    logger.info(
        "Interaction order: %s preferred (cost_a=%.2f, cost_b=%.2f)",
        preferred,
        cost_a,
        cost_b,
    )

    return {
        "preferred_order": preferred,
        "cost_a_first": cost_a,
        "cost_b_first": cost_b,
        "graph_preferred": graph_preferred,
    }

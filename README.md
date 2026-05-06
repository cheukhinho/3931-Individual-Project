# Railway Delay Propagation System

A modular Python pipeline for modelling and simulating railway delay
propagation using temporal graphs.

## Architecture

```
RealTimeTrains API  (OAuth2 Bearer token — https://api-portal.rtt.io)
       │
       ▼
config.py           ← load RTT_BEARER_TOKEN from .env
       │
       ▼
data_ingestion.py   ← exchange refresh token for access token, fetch & parse JSON → pandas DataFrame
       │
       ▼
data_processing.py  ← build services / stops / route_edges tables
       │
       ▼
graph_construction.py  ← build temporal NetworkX DiGraph
       │
       ▼
simulation.py       ← rule-based delay propagation
       │
       ├── disruption.py   ← inject synthetic disruption scenarios
       │
       └── optimisation.py ← greedy heuristic response selection
                │
                ▼
         evaluation.py  ← metrics & visualisation
```

## Modules

| Module | Description |
|---|---|
| `config` | Load `RTT_BEARER_TOKEN` from the `.env` file; expose `get_rtt_token()` |
| `data_ingestion` | OAuth2 token exchange; fetch train service data from the RTT API; parse JSON into DataFrames |
| `data_processing` | Transform raw stop data into `services`, `stops`, and `route_edges` tables |
| `graph_construction` | Build a temporal directed graph (nodes = departure events, edges = movements + dependencies) |
| `simulation` | Rule-based delay propagation: late arrival → late departure, turnaround constraints, connection dependencies |
| `disruption` | Generate synthetic disruption scenarios (single-point, multi-point, station incident) |
| `optimisation` | Greedy heuristic optimisation: evaluate `no_action`, `delay_departure`, `cancel_service`, `short_turn` |
| `evaluation` | Compute metrics (total delay, propagation depth, affected services) and plot results |

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

The RTT next-generation API authenticates via Bearer token.  Sign up at
[https://api-portal.rtt.io](https://api-portal.rtt.io) to obtain a token.
You may receive either a *long-lived access token* or a *refresh token* —
RTT will make clear which type you hold.

1. Copy the example env file and add your token:

   ```bash
   cp .env.example .env   # or create .env manually
   ```

2. Edit `.env`:

   ```
   RTT_BEARER_TOKEN=your_rtt_bearer_token_here
   ```

The `config` module loads this value automatically via `python-dotenv`.

> **Long-life access token holders:** pass the token directly to
> `fetch_service_data()` — you do not need to call `get_access_token()`.
>
> **Refresh token holders:** call `get_access_token(bearer_token)` first to
> exchange your refresh token for a short-lived access token, then use that
> access token for data calls.

## Quick Start

```python
from railway_delay.config import get_rtt_token
from railway_delay.data_ingestion import get_access_token, fetch_service_data
from railway_delay.data_processing import process_raw_data
from railway_delay.graph_construction import build_temporal_graph
from railway_delay.simulation import run_simulation
from railway_delay.disruption import single_point_disruption
from railway_delay.evaluation import compute_metrics

# 1. Exchange your refresh token for a short-lived access token
# (skip this step if you hold a long-lived access token)
bearer_token = get_rtt_token()          # reads RTT_BEARER_TOKEN from .env
access_token = get_access_token(bearer_token)

# 2. Fetch data for a station
df = fetch_service_data("EUS", token=access_token)

# 3. Process into structured tables
tables = process_raw_data(df)
stops = tables["stops"]

# 4. Build temporal graph
G = build_temporal_graph(stops)

# 5. Inject a disruption and propagate
disruptions = single_point_disruption(G, delay_minutes=10, seed=42)
G = run_simulation(G, disruptions)

# 6. Evaluate
metrics = compute_metrics(G, source_nodes=[disruptions[0]["node_key"]])
print(metrics)
# {
#   'total_delay_minutes': 45.0,
#   'affected_services_count': 2,
#   'delay_propagation_depth': 3,
#   'affected_services': {'SVC001', 'SVC002'}
# }
```

## Graph Model

**Nodes** represent departure events: `(service_id, station_crs, departure_time)`.

Node attributes include scheduled/actual times and mutable simulation state
(`simulated_departure_delay`, `simulated_arrival_delay`).

**Edges** are directed and of two types:
- `movement` – consecutive stops within the same service
- `dependency` – cross-service connections at the same station within a configurable time window

## Delay Propagation Rules

1. **Late arrival → late departure** – a train cannot depart before its arrival delay has been absorbed (minus a small dwell buffer).
2. **Movement propagation** – departure delay at node *A* becomes arrival delay at the next node *B*.
3. **Connection dependency** – if a connecting service departs within the minimum connection window of a delayed arriving service, its departure is pushed back.

## Optimisation

The greedy optimiser evaluates four actions per delayed node:

| Action | Effect |
|---|---|
| `no_action` | Delay propagates naturally |
| `delay_departure` | Hold the train for extra dwell (useful for connections) |
| `cancel_service` | Remove outgoing movement edges; freeze this node's delay |
| `short_turn` | Truncate the service at an intermediate station |

The action yielding the lowest total network delay is applied, and the
process repeats for the next most-delayed node.

## Importing a Simulation Dataset

The test suite and simulation engine both consume a **stops DataFrame** — a
flat table where every row is one train stop.  You can supply this data in
three ways.

### Option 1 — inline Python dict (quickest for one-off experiments)

Copy the structure used by `tests/conftest.py` and pass it directly:

```python
import pandas as pd
from datetime import datetime

stops_data = [
    {
        "service_id": "SVC001",
        "run_date": "2024-01-15",
        "operator": "VT",
        "origin": "London Euston",
        "destination": "Birmingham New Street",
        "stop_index": 0,
        "station_crs": "EUS",
        "station_name": "London Euston",
        "scheduled_arrival": None,
        "scheduled_departure": datetime(2024, 1, 15, 7, 0),
        "actual_arrival": None,
        "actual_departure": datetime(2024, 1, 15, 7, 5),
        "arrival_delay_min": None,
        "departure_delay_min": 5.0,
    },
    # … add more stop rows …
]

stops_df = pd.DataFrame(stops_data)
```

Required columns: `service_id`, `stop_index`, `station_crs`,
`scheduled_departure`, `scheduled_arrival`, `actual_departure`,
`actual_arrival`, `departure_delay_min`, `arrival_delay_min`.

Optional enrichment columns (used by the turnaround detector):
`unit_id`, `platform`.

### Option 2 — CSV file

Prepare a CSV with the same column names and parse it:

```python
import pandas as pd

stops_df = pd.read_csv(
    "my_stops.csv",
    parse_dates=["scheduled_departure", "scheduled_arrival",
                 "actual_departure", "actual_arrival"],
)
```

> Dates **must** be parsed as `datetime` objects (not strings) before being
> passed to `build_stops_table()` or `build_temporal_graph()`.

### Option 3 — live RTT API data

Follow the [Quick Start](#quick-start) section to fetch live data, then pass
`tables["stops"]` directly to `build_temporal_graph()`.

### Building the graph from your dataset

```python
from railway_delay.data_processing import build_stops_table
from railway_delay.graph_construction import build_temporal_graph

# Clean and normalise the raw stops
stops = build_stops_table(stops_df)

# Build the full temporal graph (all edge types enabled)
G = build_temporal_graph(
    stops,
    add_dependencies=True,
    min_connection_min=2.0,
    max_connection_min=30.0,
    add_interactions=True,
    max_interaction_min=5.0,
    add_turnarounds=True,
    min_turnaround_min=5.0,
    max_turnaround_high_min=45.0,
    max_turnaround_medium_min=30.0,
)
print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
```

### Using the built-in test fixtures as a simulation dataset

`tests/conftest.py` ships three ready-made fixtures you can reuse outside of
pytest by importing the underlying dicts:

```python
from tests.conftest import SAMPLE_STOPS_DICT, CLOSE_STOPS_DICT, TURNAROUND_STOPS_DICT
import pandas as pd

# Two services EUS→MKC→BHM and BHM→CRE→LIV (basic propagation)
stops_df = pd.DataFrame(SAMPLE_STOPS_DICT)

# Three services at MIR/BGH within a 5-minute window (interaction testing)
close_df = pd.DataFrame(CLOSE_STOPS_DICT)

# Ten services at BGH with unit_id / platform data (turnaround testing)
turnaround_df = pd.DataFrame(TURNAROUND_STOPS_DICT)
```

---

## Simulation Testing Guide

All test commands run from the repository root.  Run the full suite first to
confirm your environment is clean:

```bash
pytest tests/ -v
```

### Graph Correctness

Verify that `build_temporal_graph()` produces the right node/edge counts,
that every node carries the expected attributes, and that no self-loops exist.

```bash
pytest tests/test_graph_construction.py -v
```

Key test classes:

| Class | What it checks |
|---|---|
| `TestNodeKey` | Node key format `"SVC\|CRS\|ISO8601"` |
| `TestBuildMovementGraph` | Node count, edge count, required attributes, no self-loops |
| `TestAddDependencyEdges` | At least one cross-service dependency edge within the connection window |
| `TestAddInteractionEdges` | Interaction edges only for services within `max_interaction_min`; direction earlier→later |
| `TestAddDependencyEdgesTurnaround` | High/medium confidence turnaround edges; exclusion rules (mismatch, gap too small/large) |
| `TestBuildTemporalGraph` | End-to-end graph, flags `add_dependencies`, `add_interactions`, `add_turnarounds` |

Run a single class:

```bash
pytest tests/test_graph_construction.py::TestBuildMovementGraph -v
```

### Propagation Logic

Verify that injected delays flow correctly through movement and dependency
edges, and that the reset/inject/propagate cycle is idempotent.

```bash
pytest tests/test_simulation.py -v
```

Key test classes:

| Class | What it checks |
|---|---|
| `TestResetDelays` | All node delays zeroed after `reset_delays()` |
| `TestInjectDelay` | `simulated_departure_delay` and `simulated_arrival_delay` set correctly |
| `TestPropagateDelays` | Delay flows downstream; zero injection produces no propagation |
| `TestRunSimulation` | End-to-end run; second run does not stack on first |
| `TestDependencyEdgePropagation` | Buffer absorption: 5-min delay absorbed by 15-min turnaround; 20-min delay spills 5 min |
| `TestMirBghScenario` | Movement carries full delay; interaction carries only a fraction |

Run propagation tests only:

```bash
pytest tests/test_simulation.py::TestPropagateDelays -v
```

### Cost Model

Verify compensation-rate thresholds, per-node cost formula
`C_n = p × s × c × R(d)`, cancellation cost, peak multiplier, and total-cost
aggregation.

```bash
pytest tests/test_cost.py -v
```

Key test classes:

| Class | What it checks |
|---|---|
| `TestPassengerWeight` | Peak/off-peak multiplier; boundary conditions; `None`/`NaT` handling |
| `TestCompensationRate` | Step-function thresholds (0, 15, 30, 60 min) |
| `TestComputeNodeCost` | Zero delay → zero cost; formula with custom `service_weight`/`avg_ticket_price` |
| `TestComputeCancellationCost` | Full cost (no R factor); peak multiplier applied |
| `TestComputeTotalCost` | No delays → zero; cost increases with delay; `inf` delay treated as cancellation; missed-connection penalty |
| `TestMIRScenarioValidation` | End-to-end cost ordering for a 20-min disruption on the MIR→BGH corridor |

Run cost-model tests only:

```bash
pytest tests/test_cost.py::TestCompensationRate tests/test_cost.py::TestComputeNodeCost -v
```

### Decision Logic

Verify that `choose_best_decision()` always selects the minimum-cost action
and that `simulate_scenario()` does not mutate the original graph.

```bash
pytest tests/test_cost.py::TestSimulateScenario tests/test_cost.py::TestChooseBestDecision -v
```

Key checks:

| Test | What it checks |
|---|---|
| `TestSimulateScenario::test_original_graph_not_mutated` | Scenario simulation is non-destructive |
| `TestSimulateScenario::test_cancel_reduces_affected_nodes` | Cancel ≤ continue for affected delay |
| `TestChooseBestDecision::test_best_cost_is_minimum` | Best cost equals `min` of all evaluated costs |
| `TestChooseBestDecision::test_all_results_sorted_by_cost` | Results returned in ascending cost order |
| `TestChooseBestDecision::test_large_delay_prefers_cancel_or_short_turn` | With high missed-connection penalty, cancel/short_turn is competitive |

### Interaction Decisions

Verify that `choose_interaction_order()` selects the cheaper sequencing of
two competing services and that zero-delay inputs produce equal costs.

```bash
pytest tests/test_cost.py::TestChooseInteractionOrder -v
```

Key checks:

| Test | What it checks |
|---|---|
| `test_preferred_has_lower_or_equal_cost` | Preferred order has cost ≤ the alternative |
| `test_original_graph_not_mutated` | Graph unchanged after ordering decision |
| `test_zero_delay_costs_equal` | No preference when there is no delay |

Also run the simulation-side interaction tests:

```bash
pytest tests/test_simulation.py::TestInteractionEdgePropagation -v
```

These verify partial delay transfer via interaction edges (transfer must be
`> 0` and `< injected_delay`), and that setting `alpha=0` suppresses all
transfer.

### Cascading Effects

Verify that a disruption at one point cascades correctly through the entire
network, including cross-service dependencies and turnaround chains.

```bash
pytest tests/test_simulation.py::TestMirBghScenario \
       tests/test_simulation.py::TestDependencyEdgePropagation \
       tests/test_disruption.py -v
```

Key checks:

| Test | What it checks |
|---|---|
| `TestMirBghScenario::test_movement_delay_reaches_bgh` | Full delay carried to next stop via movement edge |
| `TestMirBghScenario::test_interaction_delay_smaller_than_movement_delay` | Interaction transfer ≤ movement transfer |
| `TestDependencyEdgePropagation::test_delay_absorbed_by_buffer` | Delay within turnaround buffer does not propagate |
| `TestDependencyEdgePropagation::test_delay_exceeding_buffer_propagates` | Excess delay beyond buffer propagates correctly |
| `TestGenerateScenarios` | Multi-point and station-incident disruption generators |
| `TestStationIncident::test_time_window_filter` | Only nodes within the specified time window are disrupted |

To run a full end-to-end cascading scenario manually:

```python
from tests.conftest import SAMPLE_STOPS_DICT
import pandas as pd
from railway_delay.data_processing import build_stops_table
from railway_delay.graph_construction import build_temporal_graph
from railway_delay.disruption import single_point_disruption
from railway_delay.simulation import run_simulation
from railway_delay.evaluation import compute_metrics

stops = build_stops_table(pd.DataFrame(SAMPLE_STOPS_DICT))
G = build_temporal_graph(stops, add_dependencies=True, add_interactions=True)
disruptions = single_point_disruption(G, delay_minutes=20, seed=42)
G = run_simulation(G, disruptions)
metrics = compute_metrics(G, source_nodes=[disruptions[0]["node_key"]])
print(metrics)
```

---

## Running Tests

```bash
pytest tests/ -v
```

102 tests covering all modules.

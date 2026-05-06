# Railway Delay Propagation System

A modular Python pipeline for modelling and simulating railway delay
propagation using temporal graphs.

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

## Running Tests

```bash
pytest tests/ -v
```


"""
config.py
=========
Centralised configuration for the railway delay propagation system.

This module serves two purposes:

1. **RTT API credentials** – loads ``RTT_BEARER_TOKEN`` from a ``.env`` file
   (or the environment) for use with :func:`get_rtt_token`.

2. **Model / cost parameters** – module-level constants used by
   :mod:`~railway_delay.simulation`, :mod:`~railway_delay.cost`, and
   :mod:`~railway_delay.optimisation`.  Changing a value here propagates
   automatically to all modules that import from this file.

Usage (credentials)
-------------------
Place a ``.env`` file in the project root with::

    RTT_BEARER_TOKEN=your_rtt_bearer_token_here

Then import :func:`get_rtt_token` wherever you need the credential::

    from railway_delay.config import get_rtt_token
    token = get_rtt_token()

Usage (model parameters)
------------------------
Import any constant directly::

    from railway_delay.config import ALPHA, PEAK_WEIGHT, CASCADE_THRESHOLD_MIN
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

# ---------------------------------------------------------------------------
# Simulation / propagation parameters
# ---------------------------------------------------------------------------

#: Base interaction strength (0–1); fraction of delay transferred via
#: interaction edges when *MAX_INTERACTION_MIN* is zero.
ALPHA: float = 0.5

#: Maximum temporal gap (minutes) for interaction edges; used in the
#: improved alpha formula::
#:
#:     effective_alpha = max(0, 1 − gap_min / MAX_INTERACTION_MIN)
MAX_INTERACTION_MIN: float = 5.0

#: Minimum dwell time at a stop (minutes) – lower bound on station recovery.
MIN_DWELL_MIN: float = 0.5

#: Minimum connection gap for dependency edges when no scheduled times exist.
MIN_CONNECTION_MIN: float = 2.0

#: Recovery time subtracted from delay on movement edges (minutes)::
#:
#:     delay_j = max(0, delay_i − RECOVERY_TIME_MIN)
RECOVERY_TIME_MIN: float = 0.0

#: Holding penalty (minutes) added to the delayed train when the other train
#: is given priority at an interaction conflict (used in the local cost
#: comparison inside :func:`~railway_delay.simulation.propagate_delays`).
INTERACTION_HOLDING_MIN: float = 2.0

# ---------------------------------------------------------------------------
# Cascading failure parameters
# ---------------------------------------------------------------------------

#: Additional threshold above the scheduled buffer (minutes).  When
#: ``delay_upstream > buffer + CASCADE_THRESHOLD_MIN``, a cascading failure
#: is triggered and the downstream service is penalised.
CASCADE_THRESHOLD_MIN: float = 10.0

#: Large delay value (minutes) assigned to a downstream service when a
#: cascading failure is triggered.  Set to 60 minutes to represent severe
#: disruption while remaining realistic and not overwhelming other costs.
CASCADE_LARGE_DELAY_MIN: float = 60.0

# ---------------------------------------------------------------------------
# Passenger weighting / cost model
# ---------------------------------------------------------------------------

#: Peak-hour windows as ``(start_hour_inclusive, end_hour_exclusive)`` pairs.
PEAK_WINDOWS: tuple[tuple[int, int], ...] = ((7, 9), (17, 19))

#: Passenger weight multiplier applied during peak hours.
PEAK_WEIGHT: float = 2.0

#: Passenger weight multiplier applied during off-peak hours.
OFF_PEAK_WEIGHT: float = 1.0

# ---------------------------------------------------------------------------
# Future-cost approximation parameters (interaction decision, Phase 5+)
# ---------------------------------------------------------------------------

#: Discount factor (0–1) applied to the estimated future (downstream) cost
#: when comparing interaction orderings.  A value of 0.3 gives downstream
#: effects a 30 % weight relative to local costs, keeping the approximation
#: computationally light while avoiding local-only myopia.
FUTURE_COST_BETA: float = 0.3

#: Approximate monetary cost per minute of delay propagated downstream (£).
#: Used to convert a predicted propagated delay into a monetary estimate:
#:
#:     estimated_future_cost = FUTURE_COST_BETA × propagated_delay × AVG_COST_PER_MINUTE
#:
#: Calibrated from DEFAULT_AVG_TICKET_PRICE ÷ typical journey minutes (~20 min).
AVG_COST_PER_MINUTE: float = 0.5

# ---------------------------------------------------------------------------
# Station importance weights (passenger weighting extension)
# ---------------------------------------------------------------------------

#: Per-station importance factors used to scale passenger weights.
#: Major interchange / terminus stations have higher factors; small rural
#: or single-platform halts have lower factors.  Stations not in this dict
#: default to 1.0 (neutral weight).
STATION_FACTORS: dict[str, float] = {
    # Major London termini
    "EUS": 1.5,   # London Euston
    "KGX": 1.5,   # London King's Cross
    "VIC": 1.5,   # London Victoria
    "WAT": 1.5,   # London Waterloo
    "PAD": 1.5,   # London Paddington
    # Major regional interchanges
    "MAN": 1.4,   # Manchester Piccadilly
    "BHM": 1.3,   # Birmingham New Street
    "LDS": 1.3,   # Leeds
    "LIV": 1.2,   # Liverpool Lime Street
    "SHF": 1.2,   # Sheffield
    "BRI": 1.2,   # Bristol Temple Meads
    "NCL": 1.2,   # Newcastle
    "GLQ": 1.2,   # Glasgow Central
    # Medium-sized stations
    "MKC": 1.0,   # Milton Keynes Central
    "CRE": 1.0,   # Crewe
    "YRK": 1.1,   # York
    # Small / rural stations
    "MIR": 0.8,   # Mirfield
    "BGH": 0.8,   # Brighouse
}

#: Default average ticket price (£) used when not stored on a node.
DEFAULT_AVG_TICKET_PRICE: float = 10.0

#: Default service importance weight used when not stored on a node.
DEFAULT_SERVICE_WEIGHT: float = 1.0

#: Minimum delay (minutes) before a missed-connection penalty is applied.
DEFAULT_CONNECTION_BUFFER_MIN: float = 10.0

#: Monetary penalty (£) per node whose delay exceeds the connection buffer.
DEFAULT_MISSED_CONNECTION_PENALTY: float = 50.0

# ---------------------------------------------------------------------------
# Delay-threshold awareness – surcharge applied during interaction decisions
# ---------------------------------------------------------------------------
#: These values are used ONLY during the interaction-ordering decision
#: (in :mod:`~railway_delay.simulation`) to penalise options that push a
#: train's delay across a milestone threshold.  They are NOT added to the
#: global network cost (which relies solely on the step-based R(d) function
#: to avoid double-counting).

#: Surcharge (£) applied when an interaction decision crosses the 15-min mark.
THRESHOLD_15_PENALTY: float = 10.0

#: Surcharge (£) applied when an interaction decision crosses the 30-min mark.
THRESHOLD_30_PENALTY: float = 25.0

#: Surcharge (£) applied when an interaction decision crosses the 60-min mark.
THRESHOLD_60_PENALTY: float = 50.0


def get_rtt_token() -> str:
    """Return the Bearer token for the RTT next-generation API.

    Reads ``RTT_BEARER_TOKEN`` from the environment (or the ``.env`` file
    in the project root).  The token may be a long-lived access token (use
    it directly with :func:`~railway_delay.data_ingestion.fetch_service_data`)
    or a short-lived refresh token (exchange it first via
    :func:`~railway_delay.data_ingestion.get_access_token`).

    Raises
    ------
    EnvironmentError
        If ``RTT_BEARER_TOKEN`` is missing or empty.
    """
    token = os.environ.get("RTT_BEARER_TOKEN")
    if not token:
        raise EnvironmentError(
            "Missing environment variable: RTT_BEARER_TOKEN. "
            "Copy .env.example to .env and fill in your RTT API token."
        )
    return token

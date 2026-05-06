"""
Shared test fixtures and sample data used across multiple test modules.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Sample raw stops data
# ---------------------------------------------------------------------------

SAMPLE_STOPS_DICT = [
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
    {
        "service_id": "SVC001",
        "run_date": "2024-01-15",
        "operator": "VT",
        "origin": "London Euston",
        "destination": "Birmingham New Street",
        "stop_index": 1,
        "station_crs": "MKC",
        "station_name": "Milton Keynes Central",
        "scheduled_arrival": datetime(2024, 1, 15, 7, 35),
        "scheduled_departure": datetime(2024, 1, 15, 7, 37),
        "actual_arrival": datetime(2024, 1, 15, 7, 40),
        "actual_departure": datetime(2024, 1, 15, 7, 42),
        "arrival_delay_min": 5.0,
        "departure_delay_min": 5.0,
    },
    {
        "service_id": "SVC001",
        "run_date": "2024-01-15",
        "operator": "VT",
        "origin": "London Euston",
        "destination": "Birmingham New Street",
        "stop_index": 2,
        "station_crs": "BHM",
        "station_name": "Birmingham New Street",
        "scheduled_arrival": datetime(2024, 1, 15, 8, 30),
        "scheduled_departure": None,
        "actual_arrival": datetime(2024, 1, 15, 8, 35),
        "actual_departure": None,
        "arrival_delay_min": 5.0,
        "departure_delay_min": None,
    },
    # Second service
    {
        "service_id": "SVC002",
        "run_date": "2024-01-15",
        "operator": "LM",
        "origin": "Birmingham New Street",
        "destination": "Liverpool Lime Street",
        "stop_index": 0,
        "station_crs": "BHM",
        "station_name": "Birmingham New Street",
        "scheduled_arrival": None,
        "scheduled_departure": datetime(2024, 1, 15, 8, 45),
        "actual_arrival": None,
        "actual_departure": datetime(2024, 1, 15, 8, 45),
        "arrival_delay_min": None,
        "departure_delay_min": 0.0,
    },
    {
        "service_id": "SVC002",
        "run_date": "2024-01-15",
        "operator": "LM",
        "origin": "Birmingham New Street",
        "destination": "Liverpool Lime Street",
        "stop_index": 1,
        "station_crs": "CRE",
        "station_name": "Crewe",
        "scheduled_arrival": datetime(2024, 1, 15, 9, 30),
        "scheduled_departure": datetime(2024, 1, 15, 9, 32),
        "actual_arrival": datetime(2024, 1, 15, 9, 30),
        "actual_departure": datetime(2024, 1, 15, 9, 32),
        "arrival_delay_min": 0.0,
        "departure_delay_min": 0.0,
    },
    {
        "service_id": "SVC002",
        "run_date": "2024-01-15",
        "operator": "LM",
        "origin": "Birmingham New Street",
        "destination": "Liverpool Lime Street",
        "stop_index": 2,
        "station_crs": "LIV",
        "station_name": "Liverpool Lime Street",
        "scheduled_arrival": datetime(2024, 1, 15, 10, 15),
        "scheduled_departure": None,
        "actual_arrival": datetime(2024, 1, 15, 10, 15),
        "actual_departure": None,
        "arrival_delay_min": 0.0,
        "departure_delay_min": None,
    },
]


@pytest.fixture
def sample_stops_df():
    return pd.DataFrame(SAMPLE_STOPS_DICT)


# ---------------------------------------------------------------------------
# Close-proximity fixture: two services within 5 minutes at MIR and BGH
# ---------------------------------------------------------------------------

CLOSE_STOPS_DICT = [
    # SVC003: MIR → BGH
    {
        "service_id": "SVC003",
        "run_date": "2026-04-30",
        "operator": "NT",
        "origin": "Mirfield",
        "destination": "Brighouse",
        "stop_index": 0,
        "station_crs": "MIR",
        "station_name": "Mirfield",
        "scheduled_arrival": None,
        "scheduled_departure": datetime(2026, 4, 30, 10, 22),
        "actual_arrival": None,
        "actual_departure": datetime(2026, 4, 30, 10, 22),
        "arrival_delay_min": None,
        "departure_delay_min": 0.0,
    },
    {
        "service_id": "SVC003",
        "run_date": "2026-04-30",
        "operator": "NT",
        "origin": "Mirfield",
        "destination": "Brighouse",
        "stop_index": 1,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": datetime(2026, 4, 30, 10, 31),
        "scheduled_departure": None,
        "actual_arrival": datetime(2026, 4, 30, 10, 31),
        "actual_departure": None,
        "arrival_delay_min": 0.0,
        "departure_delay_min": None,
    },
    # SVC004: MIR → BGH, departs MIR 4 minutes after SVC003
    {
        "service_id": "SVC004",
        "run_date": "2026-04-30",
        "operator": "NT",
        "origin": "Mirfield",
        "destination": "Brighouse",
        "stop_index": 0,
        "station_crs": "MIR",
        "station_name": "Mirfield",
        "scheduled_arrival": None,
        "scheduled_departure": datetime(2026, 4, 30, 10, 26),
        "actual_arrival": None,
        "actual_departure": datetime(2026, 4, 30, 10, 26),
        "arrival_delay_min": None,
        "departure_delay_min": 0.0,
    },
    {
        "service_id": "SVC004",
        "run_date": "2026-04-30",
        "operator": "NT",
        "origin": "Mirfield",
        "destination": "Brighouse",
        "stop_index": 1,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": datetime(2026, 4, 30, 10, 35),
        "scheduled_departure": None,
        "actual_arrival": datetime(2026, 4, 30, 10, 35),
        "actual_departure": None,
        "arrival_delay_min": 0.0,
        "departure_delay_min": None,
    },
    # SVC005: MIR → BGH, departs MIR 38 minutes after SVC004 – outside window
    {
        "service_id": "SVC005",
        "run_date": "2026-04-30",
        "operator": "NT",
        "origin": "Mirfield",
        "destination": "Brighouse",
        "stop_index": 0,
        "station_crs": "MIR",
        "station_name": "Mirfield",
        "scheduled_arrival": None,
        "scheduled_departure": datetime(2026, 4, 30, 11, 4),
        "actual_arrival": None,
        "actual_departure": datetime(2026, 4, 30, 11, 4),
        "arrival_delay_min": None,
        "departure_delay_min": 0.0,
    },
    {
        "service_id": "SVC005",
        "run_date": "2026-04-30",
        "operator": "NT",
        "origin": "Mirfield",
        "destination": "Brighouse",
        "stop_index": 1,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": datetime(2026, 4, 30, 11, 13),
        "scheduled_departure": None,
        "actual_arrival": datetime(2026, 4, 30, 11, 13),
        "actual_departure": None,
        "arrival_delay_min": 0.0,
        "departure_delay_min": None,
    },
]


@pytest.fixture
def sample_stops_close_df():
    """Two services (SVC003, SVC004) at MIR and BGH within 4 minutes of each
    other, plus a third service (SVC005) outside the 5-minute window."""
    return pd.DataFrame(CLOSE_STOPS_DICT)


# ---------------------------------------------------------------------------
# Turnaround fixture: rolling-stock reuse at BGH, 2026-04-30, 10:00–14:00
# ---------------------------------------------------------------------------

TURNAROUND_STOPS_DICT = [
    # ── SVC_T1: arrives BGH at 11:00 (terminal), no unit_id ───────────────
    {
        "service_id": "SVC_T1",
        "run_date": "2026-04-30",
        "operator": "NT",
        "stop_index": 1,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": datetime(2026, 4, 30, 11, 0),
        "scheduled_departure": None,
        "actual_arrival": datetime(2026, 4, 30, 11, 0),
        "actual_departure": None,
        "arrival_delay_min": 0.0,
        "departure_delay_min": None,
        "unit_id": None,
        "platform": "1",
    },
    # ── SVC_T2: departs BGH at 11:20 (20 min after T1) ───────────────────
    #    No unit_id, different platform → time-based fallback → MEDIUM confidence
    {
        "service_id": "SVC_T2",
        "run_date": "2026-04-30",
        "operator": "NT",
        "stop_index": 0,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": None,
        "scheduled_departure": datetime(2026, 4, 30, 11, 20),
        "actual_arrival": None,
        "actual_departure": datetime(2026, 4, 30, 11, 20),
        "arrival_delay_min": None,
        "departure_delay_min": 0.0,
        "unit_id": None,
        "platform": "2",
    },
    # ── SVC_T3: arrives BGH at 12:00 (terminal), unit_id='UNIT_A' ─────────
    {
        "service_id": "SVC_T3",
        "run_date": "2026-04-30",
        "operator": "NT",
        "stop_index": 1,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": datetime(2026, 4, 30, 12, 0),
        "scheduled_departure": None,
        "actual_arrival": datetime(2026, 4, 30, 12, 0),
        "actual_departure": None,
        "arrival_delay_min": 0.0,
        "departure_delay_min": None,
        "unit_id": "UNIT_A",
        "platform": "2",
    },
    # ── SVC_T4: departs BGH at 12:15 (15 min after T3), unit_id='UNIT_A' ──
    #    Matching unit_id → HIGH confidence
    {
        "service_id": "SVC_T4",
        "run_date": "2026-04-30",
        "operator": "NT",
        "stop_index": 0,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": None,
        "scheduled_departure": datetime(2026, 4, 30, 12, 15),
        "actual_arrival": None,
        "actual_departure": datetime(2026, 4, 30, 12, 15),
        "arrival_delay_min": None,
        "departure_delay_min": 0.0,
        "unit_id": "UNIT_A",
        "platform": "2",
    },
    # ── SVC_T5: departs BGH at 12:50 (50 min after T3) ───────────────────
    #    No unit_id, gap 50 min > 30 min medium max → NO edge
    {
        "service_id": "SVC_T5",
        "run_date": "2026-04-30",
        "operator": "NT",
        "stop_index": 0,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": None,
        "scheduled_departure": datetime(2026, 4, 30, 12, 50),
        "actual_arrival": None,
        "actual_departure": datetime(2026, 4, 30, 12, 50),
        "arrival_delay_min": None,
        "departure_delay_min": 0.0,
        "unit_id": None,
        "platform": "1",
    },
    # ── SVC_T6: arrives BGH at 13:00 (terminal), unit_id='UNIT_B' ─────────
    {
        "service_id": "SVC_T6",
        "run_date": "2026-04-30",
        "operator": "NT",
        "stop_index": 1,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": datetime(2026, 4, 30, 13, 0),
        "scheduled_departure": None,
        "actual_arrival": datetime(2026, 4, 30, 13, 0),
        "actual_departure": None,
        "arrival_delay_min": 0.0,
        "departure_delay_min": None,
        "unit_id": "UNIT_B",
        "platform": "1",
    },
    # ── SVC_T7: departs BGH at 13:15, unit_id='UNIT_C' (different unit) ───
    #    unit_ids disagree → NO edge even within time window
    {
        "service_id": "SVC_T7",
        "run_date": "2026-04-30",
        "operator": "NT",
        "stop_index": 0,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": None,
        "scheduled_departure": datetime(2026, 4, 30, 13, 15),
        "actual_arrival": None,
        "actual_departure": datetime(2026, 4, 30, 13, 15),
        "arrival_delay_min": None,
        "departure_delay_min": 0.0,
        "unit_id": "UNIT_C",
        "platform": "2",
    },
    # ── SVC_T8: arrives BGH at 13:30 (terminal), no unit_id ───────────────
    {
        "service_id": "SVC_T8",
        "run_date": "2026-04-30",
        "operator": "NT",
        "stop_index": 1,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": datetime(2026, 4, 30, 13, 30),
        "scheduled_departure": None,
        "actual_arrival": datetime(2026, 4, 30, 13, 30),
        "actual_departure": None,
        "arrival_delay_min": 0.0,
        "departure_delay_min": None,
        "unit_id": None,
        "platform": "1",
    },
    # ── SVC_T9: departs BGH at 13:33 (3 min after T8) ────────────────────
    #    Gap < 5 min minimum → NO edge
    {
        "service_id": "SVC_T9",
        "run_date": "2026-04-30",
        "operator": "NT",
        "stop_index": 0,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": None,
        "scheduled_departure": datetime(2026, 4, 30, 13, 33),
        "actual_arrival": None,
        "actual_departure": datetime(2026, 4, 30, 13, 33),
        "arrival_delay_min": None,
        "departure_delay_min": 0.0,
        "unit_id": None,
        "platform": "1",
    },
    # ── SVC_T10: arrives BGH at 10:15 (terminal), no unit_id ──────────────
    #    platform promotion test: SVC_T11 departs 20 min later, same platform
    {
        "service_id": "SVC_T10",
        "run_date": "2026-04-30",
        "operator": "NT",
        "stop_index": 1,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": datetime(2026, 4, 30, 10, 15),
        "scheduled_departure": None,
        "actual_arrival": datetime(2026, 4, 30, 10, 15),
        "actual_departure": None,
        "arrival_delay_min": 0.0,
        "departure_delay_min": None,
        "unit_id": None,
        "platform": "3",
    },
    # ── SVC_T11: departs BGH at 10:35 (20 min after T10), same platform ───
    #    No unit_id, same platform → MEDIUM promoted to HIGH via platform
    {
        "service_id": "SVC_T11",
        "run_date": "2026-04-30",
        "operator": "NT",
        "stop_index": 0,
        "station_crs": "BGH",
        "station_name": "Brighouse",
        "scheduled_arrival": None,
        "scheduled_departure": datetime(2026, 4, 30, 10, 35),
        "actual_arrival": None,
        "actual_departure": datetime(2026, 4, 30, 10, 35),
        "arrival_delay_min": None,
        "departure_delay_min": 0.0,
        "unit_id": None,
        "platform": "3",
    },
]


@pytest.fixture
def turnaround_stops_df():
    """Turnaround test fixture at BGH on 2026-04-30 covering:
    - SVC_T1 → SVC_T2: medium confidence (time-based, no unit_id, 20 min)
    - SVC_T3 → SVC_T4: high confidence (unit_id='UNIT_A' match, 15 min)
    - SVC_T3 → SVC_T5: no edge (50 min gap, above medium window)
    - SVC_T6 → SVC_T7: no edge (unit_id mismatch: UNIT_B vs UNIT_C)
    - SVC_T8 → SVC_T9: no edge (3 min gap, below minimum)
    - SVC_T10 → SVC_T11: high confidence (platform promotion, 20 min, same platform)
    """
    return pd.DataFrame(TURNAROUND_STOPS_DICT)

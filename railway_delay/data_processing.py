"""
data_processing.py
==================
Transform raw calling-pattern data (as returned by :mod:`data_ingestion`)
into three normalised tables that the rest of the pipeline consumes.

Tables produced
---------------
services
    One row per service run.  Columns::

        service_id, run_date, operator, origin, destination,
        origin_crs, destination_crs, scheduled_departure_origin,
        scheduled_arrival_destination

stops
    One row per station call within a service.  Columns::

        service_id, run_date, stop_index, station_crs, station_name,
        scheduled_arrival, scheduled_departure,
        actual_arrival, actual_departure,
        arrival_delay_min, departure_delay_min

route_edges
    One row per consecutive pair of stops (i.e. each inter-station movement).
    Columns::

        service_id, run_date,
        from_crs, from_name, to_crs, to_name,
        scheduled_departure, scheduled_arrival,
        actual_departure, actual_arrival,
        scheduled_travel_min, actual_travel_min
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minutes_between(
    t1: Optional[pd.Timestamp], t2: Optional[pd.Timestamp]
) -> Optional[float]:
    """Return ``(t2 − t1)`` in minutes, or ``None`` if either timestamp is NaT."""
    if pd.isnull(t1) or pd.isnull(t2):  # type: ignore[arg-type]
        return None
    return (t2 - t1).total_seconds() / 60.0  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Public transformation functions
# ---------------------------------------------------------------------------


def build_services_table(stops_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the *services* table from a *stops* DataFrame.

    The function aggregates per-stop rows into one row per
    ``(service_id, run_date)`` pair, capturing the origin, destination,
    and journey-level times.

    Parameters
    ----------
    stops_df:
        DataFrame in the format produced by
        :func:`~railway_delay.data_ingestion.parse_service_detail`
        or assembled from multiple such calls.  Must contain columns:
        ``service_id``, ``run_date``, ``operator``, ``stop_index``,
        ``station_crs``, ``station_name``,
        ``scheduled_departure``, ``scheduled_arrival``.

    Returns
    -------
    pd.DataFrame
        One row per ``(service_id, run_date)``.
    """
    required = {
        "service_id",
        "run_date",
        "stop_index",
        "station_crs",
        "station_name",
        "scheduled_departure",
        "scheduled_arrival",
    }
    missing = required - set(stops_df.columns)
    if missing:
        raise ValueError(f"stops_df is missing columns: {missing}")

    records = []
    for (service_id, run_date), grp in stops_df.groupby(
        ["service_id", "run_date"], sort=False
    ):
        grp = grp.sort_values("stop_index")
        first = grp.iloc[0]
        last = grp.iloc[-1]

        operator = grp["operator"].iloc[0] if "operator" in grp.columns else ""
        records.append(
            {
                "service_id": service_id,
                "run_date": run_date,
                "operator": operator,
                "origin": first["station_name"],
                "origin_crs": first["station_crs"],
                "destination": last["station_name"],
                "destination_crs": last["station_crs"],
                "scheduled_departure_origin": first["scheduled_departure"],
                "scheduled_arrival_destination": last["scheduled_arrival"],
            }
        )

    return pd.DataFrame(records)


def build_stops_table(raw_stops: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise a raw stops DataFrame.

    Ensures all expected columns are present, converts time columns to
    ``datetime64[ns]``, and computes delay columns when missing.

    Parameters
    ----------
    raw_stops:
        DataFrame with at minimum ``service_id``, ``run_date``,
        ``stop_index``, ``station_crs``.

    Returns
    -------
    pd.DataFrame
        Cleaned stops table with consistent dtypes.
    """
    df = raw_stops.copy()

    # Add any missing non-time columns with sensible defaults
    if "run_date" not in df.columns:
        df["run_date"] = ""
    if "stop_index" not in df.columns:
        df["stop_index"] = range(len(df))
    if "station_crs" not in df.columns:
        df["station_crs"] = ""

    time_cols = [
        "scheduled_arrival",
        "scheduled_departure",
        "actual_arrival",
        "actual_departure",
    ]
    for col in time_cols:
        if col not in df.columns:
            df[col] = pd.NaT
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for sched, actual, delay_col in [
        ("scheduled_arrival", "actual_arrival", "arrival_delay_min"),
        ("scheduled_departure", "actual_departure", "departure_delay_min"),
    ]:
        if delay_col not in df.columns:
            df[delay_col] = df.apply(
                lambda row, s=sched, a=actual: _minutes_between(row[s], row[a]),
                axis=1,
            )

    if "station_name" not in df.columns:
        df["station_name"] = ""

    return df[
        [
            "service_id",
            "run_date",
            "stop_index",
            "station_crs",
            "station_name",
            "scheduled_arrival",
            "scheduled_departure",
            "actual_arrival",
            "actual_departure",
            "arrival_delay_min",
            "departure_delay_min",
        ]
    ]


def build_route_edges_table(stops_df: pd.DataFrame) -> pd.DataFrame:
    """Build the *route_edges* table from a *stops* DataFrame.

    Each row represents the movement from stop *i* to stop *i+1* within the
    same service.

    Parameters
    ----------
    stops_df:
        Cleaned stops table (output of :func:`build_stops_table`).

    Returns
    -------
    pd.DataFrame
        One row per consecutive pair of stops.
    """
    records = []
    for (service_id, run_date), grp in stops_df.groupby(
        ["service_id", "run_date"], sort=False
    ):
        grp = grp.sort_values("stop_index").reset_index(drop=True)
        for i in range(len(grp) - 1):
            src = grp.iloc[i]
            dst = grp.iloc[i + 1]
            sched_travel = _minutes_between(
                src["scheduled_departure"], dst["scheduled_arrival"]
            )
            actual_travel = _minutes_between(
                src["actual_departure"], dst["actual_arrival"]
            )
            records.append(
                {
                    "service_id": service_id,
                    "run_date": run_date,
                    "from_crs": src["station_crs"],
                    "from_name": src["station_name"],
                    "to_crs": dst["station_crs"],
                    "to_name": dst["station_name"],
                    "scheduled_departure": src["scheduled_departure"],
                    "scheduled_arrival": dst["scheduled_arrival"],
                    "actual_departure": src["actual_departure"],
                    "actual_arrival": dst["actual_arrival"],
                    "scheduled_travel_min": sched_travel,
                    "actual_travel_min": actual_travel,
                }
            )

    return pd.DataFrame(records)


def process_raw_data(raw_stops: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run the full data-processing pipeline in one call.

    Parameters
    ----------
    raw_stops:
        Raw stops DataFrame (e.g. from concatenating multiple
        :func:`~railway_delay.data_ingestion.parse_service_detail` results).

    Returns
    -------
    dict with keys ``"services"``, ``"stops"``, ``"route_edges"``.
    """
    stops = build_stops_table(raw_stops)
    services = build_services_table(stops)
    route_edges = build_route_edges_table(stops)
    return {"services": services, "stops": stops, "route_edges": route_edges}

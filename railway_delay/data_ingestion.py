"""
data_ingestion.py
=================
Fetch train service data from the RealTimeTrains (RTT) API and parse the
response into pandas DataFrames that feed the rest of the pipeline.

RTT API reference: https://data.rtt.io / https://github.com/realtimetrains/api-specification

Authentication
--------------
The RTT next-generation API uses Bearer token authentication.  You may hold
either a *long-lived access token* or a *refresh token*:

* **Long-life access token** — use it directly as the ``token`` argument to
  :func:`fetch_service_data`, :func:`fetch_location_data`, or
  :func:`fetch_service_detail`.

* **Refresh token** — exchange it first by calling :func:`get_access_token`
  with the refresh token.  This performs a ``GET`` request to
  ``https://data.rtt.io/api/get_access_token`` and returns the short-lived
  access token from the ``token`` field of the JSON response.

The typical two-step flow (works for refresh-token holders)::

    from railway_delay.config import get_rtt_token
    from railway_delay.data_ingestion import get_access_token, fetch_service_data

    bearer_token = get_rtt_token()               # reads RTT_BEARER_TOKEN from .env
    access_token = get_access_token(bearer_token) # exchanges refresh → access token
    df = fetch_service_data("EUS", token=access_token)

JSON → schema mapping
---------------------
RTT ``/gb-nr/location`` response (v2 API)::

    {
      "query": {
        "location": { "description": "London Euston", "shortCodes": ["EUS"] },
        "timeFrom": "2024-01-15T00:00:00",
        "timeTo": "2024-01-15T23:59:00"
      },
      "services": [
        {
          "scheduleMetadata": {
            "identity": "W12345",
            "departureDate": "2024-01-15",
            "trainReportingIdentity": "1A23",
            "operator": { "code": "VT", "name": "Avanti West Coast" }
          },
          "temporalData": {
            "arrival":   { "scheduleAdvertised": "2024-01-15T08:29:00",
                           "realtimeActual":     "2024-01-15T08:30:00" },
            "departure": { "scheduleAdvertised": "2024-01-15T08:34:00",
                           "realtimeActual":     "2024-01-15T08:35:00" }
          },
          "origin":      [{"location": {"description": "London Euston"}}],
          "destination": [{"location": {"description": "Birmingham New Street"}}]
        },
        ...
      ]
    }

Fields extracted into the ``services`` table:
  service_id, run_date, train_identity, operator, origin, destination

Fields extracted into the ``stops`` table (one row per station call):
  service_id, run_date, station_crs, station_name,
  scheduled_arrival, scheduled_departure,
  actual_arrival, actual_departure,
  arrival_delay_min, departure_delay_min

RTT ``/gb-nr/allocations_by_service`` response (v2 API)::

    {
      "allocationData": [
        {
          "allocationIndex": 0,
          "leadingClass": "444",
          "passengerVehicles": 10,
          "allocationItems": [
            {
              "stockType": "UNIT",
              "identity": "444045",
              "inReverse": false,
              "identitySuppressed": false,
              "numberOfVehicles": 5,
              "componentVehicles": [
                {
                  "identity": "63895",
                  "isPassengerVehicle": true,
                  "isLocomotive": false,
                  "index": 0
                }
              ]
            }
          ]
        }
      ]
    }

The unit identity used for turnaround matching is ``allocationItems[].identity``
(e.g. ``"444045"``).  The ``componentVehicles[].identity`` values (e.g.
``"63895"``) identify individual vehicles *within* a unit and must **not** be
used for unit comparison.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RTT_BASE_URL = "https://data.rtt.io"
RTT_TOKEN_URL = "https://data.rtt.io/api/get_access_token"

# Full-day query window in minutes (23 h × 60 + 59 = 1439, the API maximum)
_FULL_DAY_WINDOW_MINUTES = 1439


# ---------------------------------------------------------------------------
# OAuth2 token exchange
# ---------------------------------------------------------------------------


def get_access_token(bearer_token: str, *, token_url: str = RTT_TOKEN_URL) -> str:
    """Exchange a refresh token for a short-lived access token.

    Sends a ``GET`` request to *token_url* with the refresh token in the
    ``Authorization: Bearer`` header and returns the ``token`` value from
    the JSON response.

    If you already hold a *long-lived access token* you do not need to call
    this function — pass the token directly to :func:`fetch_service_data`.

    Parameters
    ----------
    bearer_token:
        The refresh token obtained from the RTT API portal
        (https://api-portal.rtt.io).
    token_url:
        Override the token endpoint URL (useful for testing).

    Returns
    -------
    str
        The short-lived access token to use in subsequent API requests.

    Raises
    ------
    requests.HTTPError
        If the token endpoint returns a 4xx or 5xx status code.
    KeyError
        If the response JSON does not contain a ``token`` field.
    """
    headers = {"Authorization": f"Bearer {bearer_token}"}
    response = requests.get(token_url, headers=headers)
    response.raise_for_status()
    return response.json()["token"]


# ---------------------------------------------------------------------------
# Low-level HTTP helpers
# ---------------------------------------------------------------------------


def _get(
    url: str,
    token: str,
    params: Optional[dict[str, Any]] = None,
    timeout: int = 15,
) -> dict[str, Any]:
    """Send a GET request and return the parsed JSON body.

    Parameters
    ----------
    url:
        Full URL to request.
    token:
        OAuth2 access token sent as a Bearer token in the ``Authorization``
        header.
    params:
        Optional query parameters to append to the URL.
    timeout:
        Request timeout in seconds.

    Returns
    -------
    dict
        Parsed JSON response.

    Raises
    ------
    requests.HTTPError
        If the server returns a 4xx or 5xx status code.
    """
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Time parsing helpers
# ---------------------------------------------------------------------------


def _parse_hhmm(hhmm: Optional[str], run_date: str) -> Optional[datetime]:
    """Convert a four-digit ``HHMM`` string plus a date string to a datetime.

    RTT encodes times as plain strings such as ``"0835"`` or ``"2359"``.
    Day-boundary crossings (e.g. ``"0010"`` for a service that departed at
    23:55) are handled by adding one day when the parsed time is more than
    three hours *before* the run date midnight – a simple heuristic that
    covers typical overnight services.

    Parameters
    ----------
    hhmm:
        Four-character time string, e.g. ``"0835"``.  ``None`` or empty
        strings are returned as ``None``.
    run_date:
        ISO date string ``"YYYY-MM-DD"`` used as the base date.

    Returns
    -------
    datetime or None
    """
    if not hhmm:
        return None
    try:
        base = datetime.strptime(run_date, "%Y-%m-%d")
        t = datetime.strptime(hhmm, "%H%M")
        dt = base.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        return dt
    except ValueError:
        logger.warning("Could not parse time string %r on date %s", hhmm, run_date)
        return None


def _parse_iso(iso_str: Optional[str]) -> Optional[datetime]:
    """Convert an ISO 8601 datetime string to a naive :class:`datetime`.

    Handles strings with or without timezone offsets (e.g. ``+01:00`` or
    ``Z``).  Timezone information is stripped so that the returned datetime
    is always naive, consistent with the datetimes produced by
    :func:`_parse_hhmm`.

    Parameters
    ----------
    iso_str:
        ISO 8601 datetime string, e.g. ``"2024-01-15T08:35:00"`` or
        ``"2024-01-15T08:35:00+01:00"``.  ``None`` or empty strings are
        returned as ``None``.

    Returns
    -------
    datetime or None
    """
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.replace(tzinfo=None)
    except (ValueError, TypeError):
        logger.warning("Could not parse ISO datetime string %r", iso_str)
        return None


def _delay_minutes(
    scheduled: Optional[datetime], actual: Optional[datetime]
) -> Optional[float]:
    """Return ``actual − scheduled`` in minutes, or ``None`` if either is absent."""
    if scheduled is None or actual is None:
        return None
    return (actual - scheduled).total_seconds() / 60.0


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def fetch_location_data(
    crs: str,
    token: str,
    run_date: Optional[date] = None,
    *,
    base_url: str = RTT_BASE_URL,
) -> dict[str, Any]:
    """Fetch all services calling at *crs* on *run_date* from the RTT API.

    Parameters
    ----------
    crs:
        Three-letter CRS station code, e.g. ``"EUS"`` (London Euston).
    token:
        OAuth2 access token obtained via :func:`get_access_token`.
    run_date:
        Date to query.  Defaults to today.
    base_url:
        Override the API base URL (useful for testing with mock servers).

    Returns
    -------
    dict
        Raw JSON response as a Python dictionary.
    """
    if run_date is None:
        run_date = date.today()
    time_from = f"{run_date.strftime('%Y-%m-%d')}T00:00:00"
    url = f"{base_url}/gb-nr/location"
    params: dict[str, Any] = {
        "code": crs.upper(),
        "timeFrom": time_from,
        "timeWindow": _FULL_DAY_WINDOW_MINUTES,
    }
    logger.info("Fetching location data for %s on %s", crs, run_date)
    return _get(url, token, params=params)


def fetch_service_detail(
    service_uid: str,
    run_date: str,
    token: str,
    *,
    base_url: str = RTT_BASE_URL,
) -> dict[str, Any]:
    """Fetch the full calling-pattern detail for a single service.

    Parameters
    ----------
    service_uid:
        RTT service UID, e.g. ``"W12345"``.
    run_date:
        ISO date string ``"YYYY-MM-DD"``.
    token:
        OAuth2 access token obtained via :func:`get_access_token`.
    base_url:
        Override the API base URL.

    Returns
    -------
    dict
        Raw JSON response.
    """
    url = f"{base_url}/gb-nr/service"
    params: dict[str, Any] = {"identity": service_uid, "departureDate": run_date}
    logger.info("Fetching service detail for %s on %s", service_uid, run_date)
    return _get(url, token, params=params)


def fetch_allocations_by_service(
    service_uid: str,
    run_date: str,
    token: str,
    *,
    base_url: str = RTT_BASE_URL,
) -> dict[str, Any]:
    """Fetch rolling-stock allocation data for a single service.

    Calls the ``/gb-nr/allocations_by_service`` endpoint which returns the
    physical train unit(s) assigned to a service.

    Parameters
    ----------
    service_uid:
        RTT service UID, e.g. ``"W12345"``.
    run_date:
        ISO date string ``"YYYY-MM-DD"``.
    token:
        OAuth2 access token obtained via :func:`get_access_token`.
    base_url:
        Override the API base URL.

    Returns
    -------
    dict
        Raw JSON response.
    """
    url = f"{base_url}/gb-nr/allocations_by_service"
    params: dict[str, Any] = {"identity": service_uid, "departureDate": run_date}
    logger.info("Fetching allocations for %s on %s", service_uid, run_date)
    return _get(url, token, params=params)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_allocations_response(raw: dict[str, Any]) -> list[str]:
    """Extract unit identities from a ``/gb-nr/allocations_by_service`` response.

    Iterates over every ``allocationData[].allocationItems[]`` entry and
    collects the top-level ``identity`` field (e.g. ``"444045"``).

    The ``componentVehicles[].identity`` values (e.g. ``"63895"``) identify
    individual vehicles *within* a unit and are intentionally **not** extracted.
    Using them for turnaround matching would produce false negatives because
    two services running with the same unit have matching
    ``allocationItems[].identity`` values but different (and non-comparable)
    ``componentVehicles[].identity`` sets.

    Parameters
    ----------
    raw:
        Parsed JSON from :func:`fetch_allocations_by_service`.

    Returns
    -------
    list[str]
        Sorted list of allocation-item identities.  Empty list if none are
        found or if the response is malformed.
    """
    allocation_data: list[dict] = raw.get("allocationData") or []
    identities: list[str] = []
    for allocation in allocation_data:
        items: list[dict] = allocation.get("allocationItems") or []
        for item in items:
            identity = item.get("identity")
            if identity:
                identities.append(str(identity))
    return sorted(identities)


def enrich_stops_with_unit_ids(
    stops_df: pd.DataFrame,
    token: str,
    *,
    base_url: str = RTT_BASE_URL,
) -> pd.DataFrame:
    """Add a ``unit_id`` column to *stops_df* using RTT allocation data.

    For each unique ``(service_id, run_date)`` pair in *stops_df*, calls
    :func:`fetch_allocations_by_service` and populates ``unit_id`` with a
    comma-joined, sorted string of the unit identities returned (e.g.
    ``"444045"`` for a single unit or ``"444045,444046"`` for a coupled pair).
    Rows whose allocation lookup fails or returns no identities receive
    ``None``.

    The resulting ``unit_id`` column is consumed by
    :func:`~railway_delay.graph_construction.add_dependency_edges_turnaround`
    for high-confidence turnaround detection.

    Parameters
    ----------
    stops_df:
        Stops DataFrame with at minimum ``service_id`` and ``run_date``
        columns.
    token:
        OAuth2 access token obtained via :func:`get_access_token`.
    base_url:
        Override the API base URL.

    Returns
    -------
    pd.DataFrame
        A copy of *stops_df* with a ``unit_id`` column added (or
        overwritten if already present).
    """
    df = stops_df.copy()
    unit_map: dict[tuple[str, str], Optional[str]] = {}

    for row in stops_df[["service_id", "run_date"]].drop_duplicates().itertuples(index=False):
        service_id = row.service_id
        run_date = str(row.run_date)
        key = (service_id, run_date)
        try:
            raw = fetch_allocations_by_service(service_id, run_date, token, base_url=base_url)
            identities = parse_allocations_response(raw)
            unit_map[key] = ",".join(identities) if identities else None
        except Exception:
            logger.warning("Failed to fetch allocations for %s on %s", service_id, run_date)
            unit_map[key] = None

    df["unit_id"] = df.apply(
        lambda r: unit_map.get((r["service_id"], str(r["run_date"]))),
        axis=1,
    )
    return df


def parse_location_response(raw: dict[str, Any]) -> pd.DataFrame:
    """Parse a ``/gb-nr/location`` response into a *stops* DataFrame.

    Each row represents one service calling at the queried station.

    Parameters
    ----------
    raw:
        Parsed JSON from :func:`fetch_location_data`.

    Returns
    -------
    pd.DataFrame
        Columns: ``service_id``, ``run_date``, ``train_identity``,
        ``operator``, ``origin``, ``destination``, ``station_crs``,
        ``station_name``, ``scheduled_arrival``, ``scheduled_departure``,
        ``actual_arrival``, ``actual_departure``,
        ``arrival_delay_min``, ``departure_delay_min``.
    """
    services_raw: list[dict] = raw.get("services") or []
    location: dict = (raw.get("query") or {}).get("location") or {}
    short_codes: list = location.get("shortCodes") or []
    if not short_codes and "shortCodes" in location:
        logger.warning("Location has an empty 'shortCodes' list; station_crs will be blank")
    station_crs: str = short_codes[0] if short_codes else ""
    station_name: str = location.get("description", "")

    rows: list[dict[str, Any]] = []
    for svc in services_raw:
        sched_meta: dict = svc.get("scheduleMetadata") or {}
        temporal: dict = svc.get("temporalData") or {}
        arr_data: dict = temporal.get("arrival") or {}
        dep_data: dict = temporal.get("departure") or {}

        run_date: str = sched_meta.get("departureDate", "")
        operator_data: dict = sched_meta.get("operator") or {}

        origin_list = svc.get("origin") or []
        destination_list = svc.get("destination") or []
        origin = (
            (origin_list[0].get("location") or {}).get("description", "")
            if origin_list
            else ""
        )
        destination = (
            (destination_list[0].get("location") or {}).get("description", "")
            if destination_list
            else ""
        )

        sched_arr = _parse_iso(arr_data.get("scheduleAdvertised"))
        sched_dep = _parse_iso(dep_data.get("scheduleAdvertised"))
        act_arr = _parse_iso(arr_data.get("realtimeActual"))
        act_dep = _parse_iso(dep_data.get("realtimeActual"))

        rows.append(
            {
                "service_id": sched_meta.get("identity", ""),
                "run_date": run_date,
                "train_identity": sched_meta.get("trainReportingIdentity", ""),
                "operator": operator_data.get("code", ""),
                "origin": origin,
                "destination": destination,
                "station_crs": station_crs,
                "station_name": station_name,
                "scheduled_arrival": sched_arr,
                "scheduled_departure": sched_dep,
                "actual_arrival": act_arr,
                "actual_departure": act_dep,
                "arrival_delay_min": _delay_minutes(sched_arr, act_arr),
                "departure_delay_min": _delay_minutes(sched_dep, act_dep),
            }
        )

    return pd.DataFrame(rows)


def parse_service_detail(raw: dict[str, Any]) -> pd.DataFrame:
    """Parse a ``/gb-nr/service`` response into a full calling-pattern DataFrame.

    Each row represents one station call in the service's journey.

    Parameters
    ----------
    raw:
        Parsed JSON from :func:`fetch_service_detail`.

    Returns
    -------
    pd.DataFrame
        Columns: ``service_id``, ``run_date``, ``operator``,
        ``origin``, ``destination``, ``stop_index``,
        ``station_crs``, ``station_name``,
        ``scheduled_arrival``, ``scheduled_departure``,
        ``actual_arrival``, ``actual_departure``,
        ``arrival_delay_min``, ``departure_delay_min``.
    """
    service: dict = raw.get("service") or {}
    sched_meta: dict = service.get("scheduleMetadata") or {}
    service_uid: str = sched_meta.get("identity", "")
    run_date: str = sched_meta.get("departureDate", "")
    operator_data: dict = sched_meta.get("operator") or {}
    operator: str = operator_data.get("code", "")

    locations_raw: list[dict] = service.get("locations") or []

    origin_list = service.get("origin") or []
    destination_list = service.get("destination") or []
    origin = (
        (origin_list[0].get("location") or {}).get("description", "")
        if origin_list
        else ""
    )
    destination = (
        (destination_list[0].get("location") or {}).get("description", "")
        if destination_list
        else ""
    )

    rows: list[dict[str, Any]] = []
    for idx, loc in enumerate(locations_raw):
        location_info: dict = loc.get("location") or {}
        short_codes: list = location_info.get("shortCodes") or []
        if not short_codes and "shortCodes" in location_info:
            logger.warning(
                "Stop %d has an empty 'shortCodes' list; station_crs will be blank", idx
            )
        crs = short_codes[0] if short_codes else ""

        temporal: dict = loc.get("temporalData") or {}
        arr_data: dict = temporal.get("arrival") or {}
        dep_data: dict = temporal.get("departure") or {}

        sched_arr = _parse_iso(arr_data.get("scheduleAdvertised"))
        sched_dep = _parse_iso(dep_data.get("scheduleAdvertised"))
        act_arr = _parse_iso(arr_data.get("realtimeActual"))
        act_dep = _parse_iso(dep_data.get("realtimeActual"))

        rows.append(
            {
                "service_id": service_uid,
                "run_date": run_date,
                "operator": operator,
                "origin": origin,
                "destination": destination,
                "stop_index": idx,
                "station_crs": crs,
                "station_name": location_info.get("description", ""),
                "scheduled_arrival": sched_arr,
                "scheduled_departure": sched_dep,
                "actual_arrival": act_arr,
                "actual_departure": act_dep,
                "arrival_delay_min": _delay_minutes(sched_arr, act_arr),
                "departure_delay_min": _delay_minutes(sched_dep, act_dep),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def fetch_service_data(
    crs: str,
    token: str,
    run_date: Optional[date] = None,
    *,
    base_url: str = RTT_BASE_URL,
) -> pd.DataFrame:
    """High-level helper: fetch and parse all services at *crs* on *run_date*.

    Parameters
    ----------
    crs:
        Three-letter CRS station code.
    token:
        OAuth2 access token obtained via :func:`get_access_token`.
    run_date:
        Date to query.  Defaults to today.
    base_url:
        Override the API base URL.

    Returns
    -------
    pd.DataFrame
        One row per service call at the station, with both scheduled and
        actual times plus computed delay columns.

    Example
    -------
    >>> from railway_delay.config import get_rtt_token
    >>> from railway_delay.data_ingestion import get_access_token, fetch_service_data
    >>> token = get_access_token(get_rtt_token())
    >>> df = fetch_service_data("EUS", token=token)
    >>> df.columns.tolist()
    ['service_id', 'run_date', 'train_identity', 'operator', ...]
    """
    raw = fetch_location_data(crs, token, run_date, base_url=base_url)
    return parse_location_response(raw)

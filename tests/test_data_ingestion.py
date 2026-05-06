"""Tests for railway_delay.data_ingestion."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from railway_delay.data_ingestion import (
    _parse_hhmm,
    _parse_iso,
    _delay_minutes,
    parse_location_response,
    parse_service_detail,
    parse_allocations_response,
    fetch_allocations_by_service,
    enrich_stops_with_unit_ids,
    fetch_service_data,
    get_access_token,
)


# ---------------------------------------------------------------------------
# get_access_token
# ---------------------------------------------------------------------------

class TestGetAccessToken:
    def test_returns_access_token(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "token": "short-life-access-token",
            "validUntil": "2026-05-01T00:00:00Z",
            "entitlements": [],
        }
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = get_access_token("my-refresh-token")

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer my-refresh-token"
        assert result == "short-life-access-token"

    def test_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("401 Unauthorized")

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(Exception, match="401"):
                get_access_token("bad-token")

    def test_raises_on_missing_token_field(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"error": "invalid_grant"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(KeyError):
                get_access_token("bad-token")

    def test_custom_token_url(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {"token": "tok"}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            get_access_token("r", token_url="https://example.com/token")

        assert mock_get.call_args[0][0] == "https://example.com/token"


# ---------------------------------------------------------------------------
# _parse_hhmm
# ---------------------------------------------------------------------------

class TestParseHhmm:
    def test_basic(self):
        result = _parse_hhmm("0835", "2024-01-15")
        assert result == datetime(2024, 1, 15, 8, 35)

    def test_midnight_boundary(self):
        result = _parse_hhmm("0000", "2024-01-15")
        assert result == datetime(2024, 1, 15, 0, 0)

    def test_none_returns_none(self):
        assert _parse_hhmm(None, "2024-01-15") is None

    def test_empty_string_returns_none(self):
        assert _parse_hhmm("", "2024-01-15") is None

    def test_invalid_string_returns_none(self):
        result = _parse_hhmm("ABCD", "2024-01-15")
        assert result is None


class TestParseIso:
    def test_basic(self):
        result = _parse_iso("2024-01-15T08:35:00")
        assert result == datetime(2024, 1, 15, 8, 35)

    def test_with_utc_offset(self):
        result = _parse_iso("2024-01-15T08:35:00+01:00")
        assert result == datetime(2024, 1, 15, 8, 35)

    def test_with_z_suffix(self):
        result = _parse_iso("2024-01-15T08:35:00Z")
        assert result == datetime(2024, 1, 15, 8, 35)

    def test_none_returns_none(self):
        assert _parse_iso(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_iso("") is None

    def test_invalid_string_returns_none(self):
        assert _parse_iso("not-a-date") is None


# ---------------------------------------------------------------------------
# _delay_minutes
# ---------------------------------------------------------------------------

class TestDelayMinutes:
    def test_positive_delay(self):
        sched = datetime(2024, 1, 15, 8, 30)
        actual = datetime(2024, 1, 15, 8, 35)
        assert _delay_minutes(sched, actual) == 5.0

    def test_negative_delay(self):
        sched = datetime(2024, 1, 15, 8, 30)
        actual = datetime(2024, 1, 15, 8, 25)
        assert _delay_minutes(sched, actual) == -5.0

    def test_none_returns_none(self):
        assert _delay_minutes(None, datetime(2024, 1, 15, 8, 30)) is None
        assert _delay_minutes(datetime(2024, 1, 15, 8, 30), None) is None


# ---------------------------------------------------------------------------
# parse_location_response
# ---------------------------------------------------------------------------

SAMPLE_LOCATION_RESPONSE = {
    "query": {
        "location": {
            "shortCodes": ["EUS"],
            "description": "London Euston",
        },
        "timeFrom": "2024-01-15T00:00:00",
        "timeTo": "2024-01-15T23:59:00",
    },
    "services": [
        {
            "scheduleMetadata": {
                "identity": "W12345",
                "departureDate": "2024-01-15",
                "trainReportingIdentity": "1A23",
                "operator": {"code": "VT", "name": "Avanti West Coast"},
            },
            "temporalData": {
                "arrival": {
                    "scheduleAdvertised": "2024-01-15T08:30:00",
                    "realtimeActual": "2024-01-15T08:32:00",
                },
                "departure": {
                    "scheduleAdvertised": "2024-01-15T08:35:00",
                    "realtimeActual": "2024-01-15T08:38:00",
                },
            },
            "origin": [{"location": {"description": "London Euston"}}],
            "destination": [{"location": {"description": "Birmingham New Street"}}],
        }
    ],
}


class TestParseLocationResponse:
    def test_returns_dataframe(self):
        df = parse_location_response(SAMPLE_LOCATION_RESPONSE)
        assert isinstance(df, pd.DataFrame)

    def test_column_count(self):
        df = parse_location_response(SAMPLE_LOCATION_RESPONSE)
        expected_cols = {
            "service_id", "run_date", "train_identity", "operator",
            "origin", "destination", "station_crs", "station_name",
            "scheduled_arrival", "scheduled_departure",
            "actual_arrival", "actual_departure",
            "arrival_delay_min", "departure_delay_min",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_service_id_extracted(self):
        df = parse_location_response(SAMPLE_LOCATION_RESPONSE)
        assert df["service_id"].iloc[0] == "W12345"

    def test_delays_computed(self):
        df = parse_location_response(SAMPLE_LOCATION_RESPONSE)
        # arrival: 0832 - 0830 = 2 min; departure: 0838 - 0835 = 3 min
        assert df["arrival_delay_min"].iloc[0] == 2.0
        assert df["departure_delay_min"].iloc[0] == 3.0

    def test_empty_services(self):
        df = parse_location_response({"query": {"location": {"shortCodes": ["EUS"], "description": "x"}}, "services": []})
        assert len(df) == 0

    def test_missing_services_key(self):
        df = parse_location_response({"query": {"location": {"shortCodes": ["EUS"], "description": "x"}}})
        assert len(df) == 0


# ---------------------------------------------------------------------------
# parse_service_detail
# ---------------------------------------------------------------------------

SAMPLE_SERVICE_RESPONSE = {
    "service": {
        "scheduleMetadata": {
            "identity": "W12345",
            "departureDate": "2024-01-15",
            "operator": {"code": "VT", "name": "Avanti West Coast"},
        },
        "origin": [{"location": {"description": "London Euston"}}],
        "destination": [{"location": {"description": "Birmingham New Street"}}],
        "locations": [
            {
                "location": {"shortCodes": ["EUS"], "description": "London Euston"},
                "temporalData": {
                    "departure": {
                        "scheduleAdvertised": "2024-01-15T07:00:00",
                        "realtimeActual": "2024-01-15T07:05:00",
                    }
                },
            },
            {
                "location": {"shortCodes": ["MKC"], "description": "Milton Keynes Central"},
                "temporalData": {
                    "arrival": {
                        "scheduleAdvertised": "2024-01-15T07:35:00",
                        "realtimeActual": "2024-01-15T07:40:00",
                    },
                    "departure": {
                        "scheduleAdvertised": "2024-01-15T07:37:00",
                        "realtimeActual": "2024-01-15T07:42:00",
                    },
                },
            },
            {
                "location": {"shortCodes": ["BHM"], "description": "Birmingham New Street"},
                "temporalData": {
                    "arrival": {
                        "scheduleAdvertised": "2024-01-15T08:30:00",
                        "realtimeActual": "2024-01-15T08:35:00",
                    }
                },
            },
        ],
    }
}


class TestParseServiceDetail:
    def test_returns_dataframe(self):
        df = parse_service_detail(SAMPLE_SERVICE_RESPONSE)
        assert isinstance(df, pd.DataFrame)

    def test_row_count(self):
        df = parse_service_detail(SAMPLE_SERVICE_RESPONSE)
        assert len(df) == 3  # 3 stops

    def test_stop_index(self):
        df = parse_service_detail(SAMPLE_SERVICE_RESPONSE)
        assert list(df["stop_index"]) == [0, 1, 2]

    def test_station_crs(self):
        df = parse_service_detail(SAMPLE_SERVICE_RESPONSE)
        assert list(df["station_crs"]) == ["EUS", "MKC", "BHM"]


# ---------------------------------------------------------------------------
# fetch_service_data (integration-level, mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchServiceData:
    def test_calls_api_and_parses(self):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_LOCATION_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            from datetime import date
            df = fetch_service_data("EUS", token="test-access-token", run_date=date(2024, 1, 15))

        assert mock_get.called
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["headers"]["Authorization"] == "Bearer test-access-token"
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# parse_allocations_response
# ---------------------------------------------------------------------------

# Mirrors the example JSON from the problem statement.
SAMPLE_ALLOCATIONS_RESPONSE = {
    "allocationData": [
        {
            "allocationIndex": 0,
            "leadingClass": "444",
            "passengerVehicles": 10,
            "allocationItems": [
                {
                    "stockType": "UNIT",
                    "identity": "444045",
                    "inReverse": False,
                    "identitySuppressed": False,
                    "numberOfVehicles": 5,
                    "componentVehicles": [
                        {
                            "identity": "63895",
                            "isPassengerVehicle": True,
                            "isLocomotive": False,
                            "index": 0,
                        }
                    ],
                }
            ],
        }
    ]
}

# Two-unit coupled formation.
SAMPLE_ALLOCATIONS_COUPLED = {
    "allocationData": [
        {
            "allocationIndex": 0,
            "leadingClass": "444",
            "passengerVehicles": 20,
            "allocationItems": [
                {
                    "stockType": "UNIT",
                    "identity": "444045",
                    "componentVehicles": [{"identity": "63895"}],
                },
                {
                    "stockType": "UNIT",
                    "identity": "444012",
                    "componentVehicles": [{"identity": "63810"}],
                },
            ],
        }
    ]
}


class TestParseAllocationsResponse:
    def test_returns_list(self):
        result = parse_allocations_response(SAMPLE_ALLOCATIONS_RESPONSE)
        assert isinstance(result, list)

    def test_extracts_allocation_item_identity(self):
        result = parse_allocations_response(SAMPLE_ALLOCATIONS_RESPONSE)
        assert "444045" in result

    def test_does_not_extract_component_vehicle_identity(self):
        # "63895" is a componentVehicles identity and must NOT be returned
        result = parse_allocations_response(SAMPLE_ALLOCATIONS_RESPONSE)
        assert "63895" not in result

    def test_single_unit_result(self):
        result = parse_allocations_response(SAMPLE_ALLOCATIONS_RESPONSE)
        assert result == ["444045"]

    def test_coupled_units_sorted(self):
        result = parse_allocations_response(SAMPLE_ALLOCATIONS_COUPLED)
        assert result == ["444012", "444045"]

    def test_empty_allocation_data(self):
        assert parse_allocations_response({}) == []
        assert parse_allocations_response({"allocationData": []}) == []

    def test_allocation_items_missing(self):
        raw = {"allocationData": [{"allocationIndex": 0}]}
        assert parse_allocations_response(raw) == []

    def test_identity_missing_from_item(self):
        raw = {"allocationData": [{"allocationItems": [{"stockType": "UNIT"}]}]}
        assert parse_allocations_response(raw) == []


# ---------------------------------------------------------------------------
# fetch_allocations_by_service (mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchAllocationsByService:
    def test_calls_correct_endpoint(self):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_ALLOCATIONS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            result = fetch_allocations_by_service(
                "W12345", "2024-01-15", "test-token",
                base_url="https://data.rtt.io",
            )

        call_url = mock_get.call_args[0][0]
        assert call_url == "https://data.rtt.io/gb-nr/allocations_by_service"
        call_params = mock_get.call_args[1]["params"]
        assert call_params["identity"] == "W12345"
        assert call_params["departureDate"] == "2024-01-15"
        assert result == SAMPLE_ALLOCATIONS_RESPONSE

    def test_uses_bearer_auth(self):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_ALLOCATIONS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            fetch_allocations_by_service("W12345", "2024-01-15", "my-token")

        headers = mock_get.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer my-token"

    def test_raises_on_http_error(self):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")

        with patch("requests.get", return_value=mock_response):
            with pytest.raises(Exception, match="404"):
                fetch_allocations_by_service("W99999", "2024-01-15", "tok")


# ---------------------------------------------------------------------------
# enrich_stops_with_unit_ids
# ---------------------------------------------------------------------------

class TestEnrichStopsWithUnitIds:
    def _make_stops(self):
        return pd.DataFrame(
            [
                {"service_id": "W12345", "run_date": "2024-01-15", "station_crs": "EUS"},
                {"service_id": "W12345", "run_date": "2024-01-15", "station_crs": "MKC"},
                {"service_id": "W67890", "run_date": "2024-01-15", "station_crs": "BHM"},
            ]
        )

    def test_adds_unit_id_column(self):
        stops = self._make_stops()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_ALLOCATIONS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = enrich_stops_with_unit_ids(stops, "tok")

        assert "unit_id" in result.columns

    def test_unit_id_populated_for_all_stops_of_service(self):
        stops = self._make_stops()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_ALLOCATIONS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = enrich_stops_with_unit_ids(stops, "tok")

        w12345_rows = result[result["service_id"] == "W12345"]
        assert all(w12345_rows["unit_id"] == "444045")

    def test_coupled_formation_joined_sorted(self):
        stops = self._make_stops()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_ALLOCATIONS_COUPLED
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = enrich_stops_with_unit_ids(stops, "tok")

        assert result["unit_id"].iloc[0] == "444012,444045"

    def test_failed_lookup_yields_none(self):
        stops = self._make_stops()

        with patch(
            "railway_delay.data_ingestion.fetch_allocations_by_service",
            side_effect=Exception("network error"),
        ):
            result = enrich_stops_with_unit_ids(stops, "tok")

        assert result["unit_id"].isna().all()

    def test_empty_allocation_yields_none(self):
        stops = self._make_stops()
        mock_response = MagicMock()
        mock_response.json.return_value = {"allocationData": []}
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            result = enrich_stops_with_unit_ids(stops, "tok")

        assert result["unit_id"].isna().all()

    def test_does_not_mutate_input(self):
        stops = self._make_stops()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_ALLOCATIONS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response):
            enrich_stops_with_unit_ids(stops, "tok")

        assert "unit_id" not in stops.columns

    def test_api_called_once_per_service(self):
        stops = self._make_stops()
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_ALLOCATIONS_RESPONSE
        mock_response.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_response) as mock_get:
            enrich_stops_with_unit_ids(stops, "tok")

        # Two unique (service_id, run_date) pairs → two API calls
        assert mock_get.call_count == 2

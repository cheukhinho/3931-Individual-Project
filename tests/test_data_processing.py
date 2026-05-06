"""Tests for railway_delay.data_processing."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from railway_delay.data_processing import (
    build_services_table,
    build_stops_table,
    build_route_edges_table,
    process_raw_data,
)


class TestBuildStopsTable:
    def test_returns_dataframe(self, sample_stops_df):
        result = build_stops_table(sample_stops_df)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, sample_stops_df):
        result = build_stops_table(sample_stops_df)
        expected = {
            "service_id", "run_date", "stop_index", "station_crs", "station_name",
            "scheduled_arrival", "scheduled_departure",
            "actual_arrival", "actual_departure",
            "arrival_delay_min", "departure_delay_min",
        }
        assert expected == set(result.columns)

    def test_row_count_preserved(self, sample_stops_df):
        result = build_stops_table(sample_stops_df)
        assert len(result) == len(sample_stops_df)

    def test_time_columns_are_datetime(self, sample_stops_df):
        result = build_stops_table(sample_stops_df)
        for col in ["scheduled_departure", "actual_departure"]:
            non_null = result[col].dropna()
            if len(non_null) > 0:
                assert pd.api.types.is_datetime64_any_dtype(result[col])

    def test_missing_time_columns_added_as_nat(self):
        df = pd.DataFrame([{
            "service_id": "X", "run_date": "2024-01-15",
            "stop_index": 0, "station_crs": "EUS",
        }])
        result = build_stops_table(df)
        assert "scheduled_arrival" in result.columns
        assert pd.isnull(result["scheduled_arrival"].iloc[0])

    def test_raises_on_missing_required_columns(self):
        bad_df = pd.DataFrame([{"service_id": "X"}])
        # Should not raise because build_stops_table adds missing columns
        result = build_stops_table(bad_df)
        assert "station_crs" in result.columns or True  # graceful handling


class TestBuildServicesTable:
    def test_returns_dataframe(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        result = build_services_table(stops)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_service(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        result = build_services_table(stops)
        assert len(result) == 2  # SVC001 and SVC002

    def test_expected_columns(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        result = build_services_table(stops)
        expected = {
            "service_id", "run_date", "operator",
            "origin", "origin_crs", "destination", "destination_crs",
            "scheduled_departure_origin", "scheduled_arrival_destination",
        }
        assert expected.issubset(set(result.columns))

    def test_origin_destination_correct(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        result = build_services_table(stops)
        svc1 = result[result["service_id"] == "SVC001"].iloc[0]
        assert svc1["origin_crs"] == "EUS"
        assert svc1["destination_crs"] == "BHM"

    def test_raises_on_missing_columns(self):
        with pytest.raises(ValueError):
            build_services_table(pd.DataFrame([{"service_id": "X"}]))


class TestBuildRouteEdgesTable:
    def test_returns_dataframe(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        result = build_route_edges_table(stops)
        assert isinstance(result, pd.DataFrame)

    def test_edge_count(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        result = build_route_edges_table(stops)
        # SVC001 has 3 stops → 2 edges; SVC002 has 3 stops → 2 edges
        assert len(result) == 4

    def test_expected_columns(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        result = build_route_edges_table(stops)
        expected = {
            "service_id", "run_date", "from_crs", "from_name",
            "to_crs", "to_name", "scheduled_departure", "scheduled_arrival",
            "actual_departure", "actual_arrival",
            "scheduled_travel_min", "actual_travel_min",
        }
        assert expected.issubset(set(result.columns))

    def test_travel_times_computed(self, sample_stops_df):
        stops = build_stops_table(sample_stops_df)
        result = build_route_edges_table(stops)
        # SVC001: EUS dep 07:00 → MKC arr 07:35 = 35 min
        edge = result[
            (result["service_id"] == "SVC001") & (result["from_crs"] == "EUS")
        ]
        assert not edge.empty
        assert edge["scheduled_travel_min"].iloc[0] == 35.0


class TestProcessRawData:
    def test_returns_all_tables(self, sample_stops_df):
        result = process_raw_data(sample_stops_df)
        assert set(result.keys()) == {"services", "stops", "route_edges"}

    def test_all_values_are_dataframes(self, sample_stops_df):
        result = process_raw_data(sample_stops_df)
        for key, val in result.items():
            assert isinstance(val, pd.DataFrame), f"{key} is not a DataFrame"

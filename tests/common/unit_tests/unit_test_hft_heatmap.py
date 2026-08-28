import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
UTILITIES_PATH = REPO_ROOT / "tests/high_frequency_telemetry/utilities.py"
E2E_PATH = REPO_ROOT / "tests/high_frequency_telemetry/test_hft_end_to_end.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def heatmap_modules():
    utilities = _load_module("unit_hft_utilities", UTILITIES_PATH)
    module_name = "tests.high_frequency_telemetry.utilities"
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = utilities
    try:
        e2e = _load_module("unit_hft_e2e", E2E_PATH)
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
    return utilities, e2e


def _valid_row(bounds, values, start=1_000_000_000):
    count = len(values)
    row = {
        "time": "1970-01-01T00:00:02Z",
        "start_time_unix_nano": start,
        "count": count,
        "sum": sum(values),
        "min": min(values),
        "max": max(values),
        "+Inf": count,
        **{
            str(bound): sum(value <= bound for value in values)
            for bound in bounds
        },
    }
    row.update({
        "service.name": "countersyncd",
        "otel.library.name": "countersyncd",
        "otel.library.version": "1.0",
        "object_name": "Ethernet0",
        "sai_type_id": "1",
        "sai_stat_id": "2",
        "heatmap_value_kind": "delta",
        "heatmap_quantity": "delta_bytes",
        "heatmap_schema": "schema",
        "hft_session": "profile|PORT",
    })
    return row


def _validate(e2e, rows, bounds, expected_observations=None):
    return e2e._validate_histogram_rows(
        rows,
        bounds,
        {
            **e2e.OTEL_IDENTITY_TAGS,
            "object_name": "Ethernet0",
            "sai_type_id": "1",
            "sai_stat_id": "2",
            "heatmap_value_kind": "delta",
            "heatmap_quantity": "delta_bytes",
            "heatmap_schema": "schema",
            "hft_session": "profile|PORT",
        },
        heatmap_interval_us=1_000_000,
        max_observations=10,
        expected_observations=expected_observations,
    )


def test_delta_bytes_default_layout_and_schema(heatmap_modules):
    utilities, _ = heatmap_modules
    bounds = utilities.build_delta_bytes_default_bounds(100_000)

    assert len(bounds) == 42
    assert bounds[:2] == [0, 625_000_000]
    assert bounds[-1] == 20_000_000_000
    assert utilities.build_heatmap_schema(
        "delta", "delta_bytes", bounds
    ) == "hft-explicit-v2:delta:delta_bytes:fnv1a64-a09523fcb7c0174d"
    assert utilities.build_heatmap_schema(
        "delta", "delta_bytes", [0, 128, 1024, 8192]
    ) == "hft-explicit-v2:delta:delta_bytes:fnv1a64-a9a91448548c2f4d"


def test_validate_histogram_accepts_consistent_rows(heatmap_modules):
    _, e2e = heatmap_modules
    bounds = [0, 128, 1024, 8192]
    first = _valid_row(bounds, [64, 256, 2048])
    second = _valid_row(bounds, [32, 512, 4096], start=2_000_000_000)
    second["time"] = "1970-01-01T00:00:03Z"

    validated = _validate(e2e, [second, first], bounds)

    assert set(validated) == {1_000_000_000, 2_000_000_000}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("+Inf"),
        lambda row: row.update({"+Inf": 2.5}),
        lambda row: row.update({"128": -1}),
        lambda row: row.update({"128": row["count"]}),
        lambda row: row.update({"start_time_unix_nano": 1_500_000_000}),
        lambda row: row.update({"heatmap_schema": "unexpected"}),
        lambda row: row.update({"unexpected_field": None}),
        lambda row: row.update({"unexpected_tag": "value"}),
    ],
)
def test_validate_histogram_rejects_malformed_rows(heatmap_modules, mutation):
    _, e2e = heatmap_modules
    bounds = [0, 128, 1024, 8192]
    row = _valid_row(bounds, [64, 256, 2048])
    mutation(row)

    with pytest.raises(pytest.fail.Exception):
        _validate(e2e, [row], bounds)


def test_validate_histogram_requires_full_window_count(heatmap_modules):
    _, e2e = heatmap_modules
    bounds = [0, 128, 1024, 8192]
    row = _valid_row(bounds, [64, 256, 2048])

    with pytest.raises(pytest.fail.Exception):
        _validate(e2e, [row], bounds, expected_observations=10)


def test_heatmap_query_groups_by_every_tag(heatmap_modules):
    utilities, _ = heatmap_modules
    ptfhost = MagicMock()
    sink = utilities.InfluxDbSink(ptfhost)
    sink._query = MagicMock(return_value={"results": []})
    series = utilities.HftSeries("heatmap", "Ethernet0", 1, 2, "counter")

    assert sink.heatmap_rows(series, limit=8) == []
    sink._query.assert_called_once_with(
        'SELECT * FROM "heatmap" GROUP BY * ORDER BY time DESC LIMIT 8'
    )


def test_heatmap_query_preserves_unknown_tags_and_fields(heatmap_modules):
    utilities, _ = heatmap_modules
    ptfhost = MagicMock()
    sink = utilities.InfluxDbSink(ptfhost)
    sink._query = MagicMock(return_value={
        "results": [{
            "series": [{
                "name": "heatmap",
                "tags": {"unexpected_tag": "value"},
                "columns": ["time", "unexpected_field"],
                "values": [["1970-01-01T00:00:01Z", None]],
            }],
        }],
    })
    series = utilities.HftSeries("heatmap", "Ethernet0", 1, 2, "counter")

    assert sink.heatmap_rows(series) == [{
        "time": "1970-01-01T00:00:01Z",
        "unexpected_field": None,
        "unexpected_tag": "value",
    }]


def test_heatmap_cleanup_reports_both_failures(heatmap_modules, monkeypatch):
    utilities, _ = heatmap_modules

    def fail_profile(*args, **kwargs):
        raise AssertionError("profile failure")

    def fail_aggregator(*args, **kwargs):
        raise AssertionError("aggregator failure")

    monkeypatch.setattr(utilities, "cleanup_hft_config", fail_profile)
    monkeypatch.setattr(utilities, "cleanup_hft_aggregator", fail_aggregator)

    with pytest.raises(pytest.fail.Exception) as error:
        utilities.cleanup_hft_heatmap_config(MagicMock(), "profile", "agg")

    assert "profile failure" in str(error.value)
    assert "aggregator failure" in str(error.value)

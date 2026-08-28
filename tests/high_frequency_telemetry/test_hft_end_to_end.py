import logging
import math

import pytest

from tests.common.helpers.assertions import pytest_assert
from tests.high_frequency_telemetry.counter_profiles import (
    CounterObjectType,
)
from tests.high_frequency_telemetry.utilities import (
    ContinuousTraffic,
    _parse_rfc3339_timestamp,
    build_delta_bytes_default_bounds,
    build_expected_heatmap_series,
    build_expected_series,
    build_heatmap_schema,
    cleanup_hft_config,
    cleanup_hft_heatmap_config,
    setup_hft_config,
    setup_hft_heatmap_config,
)

logger = logging.getLogger(__name__)

OTEL_IDENTITY_TAGS = {
    "service.name": "countersyncd",
    "otel.library.name": "countersyncd",
    "otel.library.version": "1.0",
}

pytestmark = [
    pytest.mark.topology("any"),
]


@pytest.mark.hft_requirements(
    CounterObjectType.PORT, counter="IF_IN_OCTETS"
)
def test_hft_end_to_end_influxdb(
        duthosts, enum_rand_one_per_hwsku_hostname, hft_influxdb,
        skip_unsupported_hft_test):
    """Smoke-test the daemon-to-OTEL-to-InfluxDB HFT data path."""
    duthost = duthosts[enum_rand_one_per_hwsku_hostname]
    profile_name = "e2e_port_profile"
    ports = skip_unsupported_hft_test["objects"][:2]
    counters = ["IF_IN_OCTETS"]

    try:
        setup_hft_config(
            duthost,
            profile_name,
            "PORT",
            ports,
            counters,
            poll_interval=10_000,
            stream_state="enabled",
        )
        expected = build_expected_series(
            CounterObjectType.PORT, ports, counters
        )
        stats = hft_influxdb.wait_and_validate(
            expected, 10_000, min_points=100, timeout=45
        )
        logger.info("Validated end-to-end HFT series: %s", stats)
    finally:
        cleanup_hft_config(duthost, profile_name, ["PORT"])


def _nonnegative_integer(row, field):
    value = row.get(field)
    pytest_assert(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"Histogram field {field} is not numeric: {row}",
    )
    value = float(value)
    pytest_assert(
        math.isfinite(value) and value >= 0 and value.is_integer(),
        f"Histogram field {field} is not a nonnegative integer: {row}",
    )
    return int(value)


def _validate_histogram_rows(rows, bounds, expected_tags,
                             heatmap_interval_us, max_observations,
                             expected_observations=None):
    required_fields = {
        "time", "start_time_unix_nano", "count", "sum", "min", "max",
        "+Inf",
        *(str(bound) for bound in bounds),
    }
    validated = {}
    for row in rows:
        actual_fields = {
            key for key in row
            if key not in expected_tags
        }
        pytest_assert(
            actual_fields == required_fields,
            "Unexpected histogram field layout: "
            f"actual={sorted(actual_fields)}, "
            f"expected={sorted(required_fields)}; row={row}",
        )
        for key, value in expected_tags.items():
            pytest_assert(
                str(row.get(key)) == str(value),
                f"Unexpected {key}: {row.get(key)!r}, expected {value!r}",
            )

        pytest_assert(
            isinstance(row["start_time_unix_nano"], int)
            and not isinstance(row["start_time_unix_nano"], bool)
            and row["start_time_unix_nano"] >= 0,
            f"Invalid start_time_unix_nano: {row}",
        )
        start_time_unix_nano = row["start_time_unix_nano"]
        count = _nonnegative_integer(row, "count")
        total = _nonnegative_integer(row, "sum")
        minimum = _nonnegative_integer(row, "min")
        maximum = _nonnegative_integer(row, "max")
        cumulative_counts = [
            _nonnegative_integer(row, str(bound)) for bound in bounds
        ] + [_nonnegative_integer(row, "+Inf")]
        if expected_observations is None:
            pytest_assert(
                0 < count <= max_observations,
                f"Invalid observation count: {row}",
            )
        else:
            pytest_assert(
                count == expected_observations,
                f"Expected {expected_observations} observations: {row}",
            )
        pytest_assert(0 <= minimum <= maximum, f"Invalid min/max: {row}")
        pytest_assert(
            minimum * count <= total <= maximum * count,
            f"Histogram sum is inconsistent with count/min/max: {row}",
        )
        pytest_assert(
            cumulative_counts == sorted(cumulative_counts),
            f"Histogram buckets are not cumulative: {row}",
        )
        pytest_assert(
            all(bucket <= count for bucket in cumulative_counts),
            f"Histogram bucket exceeds count: {row}",
        )
        pytest_assert(
            cumulative_counts[-1] == count,
            f"Infinity bucket does not equal count: {row}",
        )
        for bound, bucket_count in zip(bounds, cumulative_counts[:-1]):
            pytest_assert(
                (bucket_count == 0) == (minimum > bound),
                f"Bucket {bound} is inconsistent with minimum: {row}",
            )
            pytest_assert(
                (bucket_count == count) == (maximum <= bound),
                f"Bucket {bound} is inconsistent with maximum: {row}",
            )

        end_time = _parse_rfc3339_timestamp(row["time"])
        window_seconds = heatmap_interval_us / 1_000_000.0
        pytest_assert(
            abs(
                end_time
                - start_time_unix_nano / 1_000_000_000.0
                - window_seconds
            ) <= 0.000001,
            f"Unexpected histogram start/end timestamps: {row}",
        )
        pytest_assert(
            start_time_unix_nano not in validated,
            f"Duplicate histogram window: {row}",
        )
        validated[start_time_unix_nano] = row

    timestamps = sorted(
        rows, key=lambda row: _parse_rfc3339_timestamp(row["time"])
    )
    expected_seconds = heatmap_interval_us / 1_000_000.0
    intervals = [
        _parse_rfc3339_timestamp(current["time"])
        - _parse_rfc3339_timestamp(previous["time"])
        for previous, current in zip(timestamps, timestamps[1:])
    ]
    pytest_assert(
        all(
            abs(interval - expected_seconds) <= 0.05
            for interval in intervals
        ),
        f"Unexpected heatmap intervals: {intervals}",
    )
    return validated


@pytest.mark.hft_requirements(
    CounterObjectType.PORT,
    counters=("IF_IN_OCTETS", "IF_OUT_OCTETS"),
    heatmap=True,
    oper_up_port=True,
)
def test_hft_heatmap_explicit_and_default_histograms(
        duthosts, enum_rand_one_per_hwsku_hostname, tbinfo, ptfadapter,
        hft_influxdb, skip_unsupported_hft_test):
    """Validate explicit and semantic heatmap layouts through InfluxDB."""
    duthost = duthosts[enum_rand_one_per_hwsku_hostname]
    required_counters = ["IF_IN_OCTETS", "IF_OUT_OCTETS"]
    profile_name = "heatmap_e2e_profile"
    aggregator_name = "heatmap_e2e_aggregator"
    port = skip_unsupported_hft_test["objects"][0]
    poll_interval_us = 10_000
    reporting_rate_us = 100_000
    heatmap_interval_us = 1_000_000
    custom_bounds = [0, 128, 1024, 8192]
    default_bounds = build_delta_bytes_default_bounds(reporting_rate_us)
    gauge_series = build_expected_series(
        CounterObjectType.PORT, [port], required_counters
    )
    heatmap_series = build_expected_heatmap_series(
        CounterObjectType.PORT, [port], required_counters
    )
    bounds_by_counter = {
        "IF_IN_OCTETS": custom_bounds,
        "IF_OUT_OCTETS": default_bounds,
    }
    ptf_port_index = duthost.get_extended_minigraph_facts(tbinfo)[
        "minigraph_ptf_indices"
    ][port]
    traffic = ContinuousTraffic(
        ptfadapter, ptf_port_index, duthost.facts["router_mac"]
    )
    rows_by_counter = {}

    try:
        setup_hft_heatmap_config(
            duthost,
            profile_name,
            aggregator_name,
            "PORT",
            [port],
            required_counters,
            heatmap_interval=heatmap_interval_us,
            poll_interval=poll_interval_us,
            reporting_rate=reporting_rate_us,
            explicit_bounds={"IF_IN_OCTETS": custom_bounds},
        )
        traffic.start()
        gauge_stats = hft_influxdb.wait_and_validate(
            gauge_series,
            reporting_rate_us,
            min_points=30,
            timeout=45,
        )
        logger.info("Validated reporting-rate gauge series: %s", gauge_stats)

        for series in heatmap_series:
            bounds = bounds_by_counter[series.counter_name]
            schema = build_heatmap_schema("delta", "delta_bytes", bounds)
            rows = hft_influxdb.wait_for_heatmap_rows(
                series,
                "delta",
                "delta_bytes",
                schema,
                f"{profile_name}|PORT",
                min_points=4,
                timeout=45,
            )
            rows_by_counter[series.counter_name] = _validate_histogram_rows(
                rows,
                bounds,
                {
                    **OTEL_IDENTITY_TAGS,
                    "object_name": port,
                    "sai_type_id": str(series.type_id),
                    "sai_stat_id": str(series.stat_id),
                    "heatmap_value_kind": "delta",
                    "heatmap_quantity": "delta_bytes",
                    "heatmap_schema": schema,
                    "hft_session": f"{profile_name}|PORT",
                },
                heatmap_interval_us,
                max_observations=heatmap_interval_us // reporting_rate_us,
            )
        expected_measurements = {
            series.measurement
            for series in gauge_series + heatmap_series
        }
        observed_measurements = hft_influxdb.measurements()
        pytest_assert(
            observed_measurements == expected_measurements,
            f"Unexpected InfluxDB measurements: actual="
            f"{sorted(observed_measurements)}, "
            f"expected={sorted(expected_measurements)}",
        )
        traffic.stop()
        pytest_assert(traffic.packet_count > 0, "No PTF traffic was transmitted")
        pytest_assert(
            not traffic.errors,
            f"PTF traffic sender errors: {traffic.errors[:10]}",
        )

        common_windows = set.intersection(*(
            set(rows) for rows in rows_by_counter.values()
        ))
        pytest_assert(
            len(common_windows) >= 4,
            f"Counters do not have four aligned heatmap windows: "
            f"{rows_by_counter}",
        )
        complete_windows = sorted(common_windows)[-3:]
        expected_observations = heatmap_interval_us // reporting_rate_us
        gauges_by_counter = {
            series.counter_name: series for series in gauge_series
        }
        for series in heatmap_series:
            counter_name = series.counter_name
            bounds = bounds_by_counter[counter_name]
            schema = build_heatmap_schema("delta", "delta_bytes", bounds)
            selected_rows = [
                rows_by_counter[counter_name][window]
                for window in complete_windows
            ]
            _validate_histogram_rows(
                selected_rows,
                bounds,
                {
                    **OTEL_IDENTITY_TAGS,
                    "object_name": port,
                    "sai_type_id": str(series.type_id),
                    "sai_stat_id": str(series.stat_id),
                    "heatmap_value_kind": "delta",
                    "heatmap_quantity": "delta_bytes",
                    "heatmap_schema": schema,
                    "hft_session": f"{profile_name}|PORT",
                },
                heatmap_interval_us,
                max_observations=expected_observations,
                expected_observations=expected_observations,
            )
            for row in selected_rows:
                start_value, end_value = hft_influxdb.gauge_window_values(
                    gauges_by_counter[counter_name],
                    row["start_time_unix_nano"],
                    row["time"],
                )
                start_counter = _nonnegative_integer(
                    {"value": start_value}, "value"
                )
                end_counter = _nonnegative_integer(
                    {"value": end_value}, "value"
                )
                gauge_delta = end_counter - start_counter
                precision_tolerance = 0
                if max(start_counter, end_counter) >= 2 ** 53:
                    precision_tolerance = math.ceil(
                        (
                            math.ulp(float(start_value))
                            + math.ulp(float(end_value))
                        ) / 2
                    )
                histogram_sum = _nonnegative_integer(row, "sum")
                pytest_assert(
                    gauge_delta >= 0
                    and abs(gauge_delta - histogram_sum)
                    <= precision_tolerance,
                    f"Histogram sum is not the cumulative gauge delta for "
                    f"{counter_name}: start={start_value}, end={end_value}, "
                    f"tolerance={precision_tolerance}, row={row}",
                )

        ingress_rows = [
            rows_by_counter["IF_IN_OCTETS"][window]
            for window in complete_windows
        ]
        pytest_assert(
            all(_nonnegative_integer(row, "sum") > 0 for row in ingress_rows),
            f"Ingress traffic did not produce nonzero byte deltas: {ingress_rows}",
        )
        pytest_assert(
            any(
                _nonnegative_integer(row, "max") > custom_bounds[1]
                for row in ingress_rows
            ),
            f"Ingress traffic did not exercise custom bounds: {ingress_rows}",
        )
    finally:
        traffic.stop()
        cleanup_hft_heatmap_config(
            duthost, profile_name, aggregator_name, ["PORT"]
        )

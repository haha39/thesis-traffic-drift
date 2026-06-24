import math
import tempfile
import unittest
from pathlib import Path

from scripts.audit_milan_internet import (
    CellPolicyStats,
    GroupState,
    Segment,
    aggregate_observation_counts,
    build_cell_policy_stats,
    ks_distance,
    local_ms,
    parse_internet_value,
    policy_denominator,
    policy_mean,
    quantile,
    rank_cells,
    read_day_aggregates,
    segment_for_timestamp,
    timestamps_in_range,
)


class AuditMilanInternetTests(unittest.TestCase):
    def test_quantile_interpolates_deterministically(self):
        self.assertEqual(quantile([4, 1, 3, 2], 0.5), 2.5)
        self.assertEqual(quantile([10], 0.9), 10)

    def test_ks_distance_detects_identical_and_shifted_samples(self):
        self.assertEqual(ks_distance([1, 2, 3], [1, 2, 3]), 0.0)
        self.assertGreater(ks_distance([1, 1, 1], [2, 2, 2]), 0.9)

    def test_parse_internet_value_preserves_raw_states(self):
        self.assertEqual(parse_internet_value(""), ("null", None))
        self.assertEqual(parse_internet_value("0"), ("numeric", 0.0))
        self.assertEqual(parse_internet_value("-1.5"), ("numeric", -1.5))
        self.assertEqual(parse_internet_value("bad"), ("malformed", None))
        self.assertEqual(parse_internet_value("nan"), ("nonfinite", None))
        self.assertEqual(parse_internet_value("inf"), ("nonfinite", None))

    def test_read_day_aggregates_sums_country_code_rows_and_preserves_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sms-call-internet-mi-2013-11-01.txt"
            path.write_text(
                "\n".join(
                    [
                        "1\t1383260400000\t0\t\t\t\t\t",
                        "1\t1383260400000\t39\t\t\t\t\t2.5",
                        "1\t1383260400000\t40\t\t\t\t\t3.0",
                        "2\t1383260400000\t39\t\t\t\t\t0",
                        "3\t1383260400000\t39\t\t\t\t\tbad",
                        "4\t1383260400000\t39\t\t\t\t\tinf",
                        "5\t1383260400000\t39\t\t\t\t\t-2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            aggregates, stats = read_day_aggregates(path)

        group = aggregates[(1383260400000, 1)]
        self.assertEqual(group.state, "NUMERIC_OBSERVED")
        self.assertEqual(group.raw_rows, 3)
        self.assertEqual(group.numeric_rows, 2)
        self.assertEqual(group.null_rows, 1)
        self.assertTrue(group.mixed_numeric_plus_null)
        self.assertEqual(group.finite_sum, 5.5)
        self.assertEqual(aggregates[(1383260400000, 2)].explicit_zero_rows, 1)
        self.assertEqual(aggregates[(1383260400000, 3)].malformed_rows, 1)
        self.assertEqual(aggregates[(1383260400000, 4)].nonfinite_rows, 1)
        self.assertEqual(aggregates[(1383260400000, 5)].negative_rows, 1)
        self.assertEqual(stats["null_internet_rows"], 1)
        self.assertEqual(stats["explicit_zero_internet_rows"], 1)
        self.assertEqual(stats["malformed_internet_rows"], 1)
        self.assertEqual(stats["nonfinite_internet_rows"], 1)
        self.assertEqual(stats["negative_internet_rows"], 1)
        self.assertEqual(stats["duplicate_timestamp_cell_groups"], 1)

    def test_all_null_and_absent_pairs_remain_distinct(self):
        ts = 1000
        all_null = GroupState()
        all_null.add_null()
        numeric_zero = GroupState()
        numeric_zero.add_numeric(0.0)
        groups = {ts: {1: all_null, 2: numeric_zero}}

        counts = aggregate_observation_counts(groups, [ts], [1, 2, 3])

        self.assertEqual(counts["all_null"], 1)
        self.assertEqual(counts["numeric_observed"], 1)
        self.assertEqual(counts["explicit_zero"], 1)
        self.assertEqual(counts["absent"], 1)

    def test_policy_denominators_differ_for_numeric_null_and_absent(self):
        stats = CellPolicyStats(observed_sum=10.0, present_count=2, numeric_count=1, all_null_count=1, absent_count=1)
        self.assertEqual(policy_denominator(3, stats, "P1_OBSERVED_ONLY"), 1)
        self.assertEqual(policy_denominator(3, stats, "P2_NULL_ZERO_ABSENT_MASKED"), 2)
        self.assertEqual(policy_denominator(3, stats, "P3_LEGACY_ZERO_FILL"), 3)
        self.assertEqual(policy_mean(3, stats, "P1_OBSERVED_ONLY"), 10.0)
        self.assertEqual(policy_mean(3, stats, "P2_NULL_ZERO_ABSENT_MASKED"), 5.0)
        self.assertAlmostEqual(policy_mean(3, stats, "P3_LEGACY_ZERO_FILL"), 10.0 / 3.0)

    def test_europe_rome_timestamp_conversion(self):
        self.assertEqual(local_ms("2013-11-01 00:00"), 1383260400000)
        self.assertEqual(local_ms("2013-12-13 00:00"), 1386889200000)

    def test_end_exclusive_split_boundaries(self):
        train = Segment("train", local_ms("2013-11-01 00:00"), local_ms("2013-11-29 00:00"))
        val = Segment("validation", local_ms("2013-11-29 00:00"), local_ms("2013-12-06 00:00"))
        segments = [train, val]
        self.assertEqual(segment_for_timestamp(local_ms("2013-11-28 23:50"), segments), "train")
        self.assertEqual(segment_for_timestamp(local_ms("2013-11-29 00:00"), segments), "validation")
        self.assertEqual(len(timestamps_in_range(train.start_ms, train.end_ms)), 28 * 144)
        self.assertEqual(len(timestamps_in_range(val.start_ms, val.end_ms)), 7 * 144)

    def test_deterministic_topk_tie_handling(self):
        stats = {cell_id: CellPolicyStats(observed_sum=1.0, present_count=1, numeric_count=1) for cell_id in range(1, 4)}
        self.assertEqual(rank_cells(stats, 1, "P1_OBSERVED_ONLY", 3), [1, 2, 3])

    def test_topk_ranking_uses_training_only_no_validation_leakage(self):
        t0 = local_ms("2013-11-01 00:00")
        t1 = local_ms("2013-11-29 00:00")
        g_train = GroupState()
        g_train.add_numeric(1.0)
        g_val = GroupState()
        g_val.add_numeric(1000.0)
        groups = {t0: {1: g_train}, t1: {2: g_val}}

        train_stats = build_cell_policy_stats(groups, [t0], 2)
        ranking = rank_cells(train_stats, 1, "P1_OBSERVED_ONLY", 2)

        self.assertEqual(ranking, [1, 2])
        self.assertEqual(train_stats[2].observed_sum, 0.0)
        self.assertEqual(train_stats[2].absent_count, 1)

    def test_observation_state_counts_remain_preserved(self):
        ts = 1000
        numeric_null = GroupState()
        numeric_null.add_numeric(2.0)
        numeric_null.add_null()
        invalid = GroupState()
        invalid.add_malformed()
        invalid.add_nonfinite()
        negative = GroupState()
        negative.add_numeric(-3.0)
        groups = {ts: {1: numeric_null, 2: invalid, 3: negative}}

        counts = aggregate_observation_counts(groups, [ts], [1, 2, 3, 4])

        self.assertEqual(counts["numeric_observed"], 2)
        self.assertEqual(counts["mixed_numeric_plus_null"], 1)
        self.assertEqual(counts["invalid_only"], 1)
        self.assertEqual(counts["malformed"], 1)
        self.assertEqual(counts["nonfinite"], 1)
        self.assertEqual(counts["negative"], 1)
        self.assertEqual(counts["absent"], 1)
        self.assertFalse(math.isnan(counts["numeric_observed"]))


if __name__ == "__main__":
    unittest.main()

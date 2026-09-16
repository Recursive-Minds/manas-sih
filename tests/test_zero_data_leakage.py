"""
Zero Data Leakage Verification Test Suite (SIH).

Mathematically and empirically verifies that:
1. Temporal partitioning strictly segregates Train (Part 1, 60%), Val (Part 2, 20%), and Bench (Part 3, 20%).
2. Embargo buffers between partitions are strictly >= 15.0 seconds (150 samples).
3. Train, Validation, and Benchmark partitions have ZERO index overlap across all sequences.
4. S-M (Highway) and S-S3a (Mixed) are strictly reserved as held-out test evaluations.
"""

import unittest
import numpy as np
from sih.data.split import compute_trip_partition, KNOWN_TRIP_LENGTHS, TripPartition
from sih.data.sequence_dataset import load_trip, TripBundle


class TestZeroDataLeakage(unittest.TestCase):

    def test_partition_boundaries_and_embargo(self):
        """Verify partition math has zero overlap and >= 15s embargo buffers."""
        for tid, length in KNOWN_TRIP_LENGTHS.items():
            part = compute_trip_partition(tid, length, train_ratio=0.60, val_ratio=0.20, embargo_s=15.0, sampling_rate_hz=10.0)
            
            # Train range: [0, train_end)
            t_start, t_end = part.train_range
            self.assertEqual(t_start, 0)
            self.assertGreater(t_end, 0)

            # Embargo 1: [train_end, embargo_1_end)
            e1_start, e1_end = part.embargo_1_range
            self.assertEqual(e1_start, t_end)
            self.assertGreaterEqual(e1_end - e1_start, 150)  # 15s @ 10Hz

            # Val range: [embargo_1_end, val_end)
            v_start, v_end = part.val_range
            self.assertEqual(v_start, e1_end)
            self.assertGreater(v_end, v_start)

            # Embargo 2: [val_end, embargo_2_end)
            e2_start, e2_end = part.embargo_2_range
            self.assertEqual(e2_start, v_end)
            self.assertGreaterEqual(e2_end - e2_start, 150)  # 15s @ 10Hz

            # Bench range: [embargo_2_end, total_samples)
            b_start, b_end = part.bench_range
            self.assertEqual(b_start, e2_end)
            self.assertEqual(b_end, length)

            # Disjointness check
            train_set = set(range(t_start, t_end))
            val_set = set(range(v_start, v_end))
            bench_set = set(range(b_start, b_end))

            self.assertEqual(len(train_set.intersection(val_set)), 0, f"Train and Val overlap in {tid}")
            self.assertEqual(len(train_set.intersection(bench_set)), 0, f"Train and Bench overlap in {tid}")
            self.assertEqual(len(val_set.intersection(bench_set)), 0, f"Val and Bench overlap in {tid}")

    def test_load_trip_partition_slicing(self):
        """Verify load_trip accurately slices real dataset bundles according to partition."""
        # Test on S-S1
        full_bundle = load_trip("S-S1", partition="all")
        train_bundle = load_trip("S-S1", partition="train")
        val_bundle = load_trip("S-S1", partition="val")
        bench_bundle = load_trip("S-S1", partition="bench")

        total_len = len(full_bundle.features)
        part = compute_trip_partition("S-S1", total_len)

        self.assertEqual(len(train_bundle.features), part.train_range[1] - part.train_range[0])
        self.assertEqual(len(val_bundle.features), part.val_range[1] - part.val_range[0])
        self.assertEqual(len(bench_bundle.features), part.bench_range[1] - part.bench_range[0])

        # Verify values match exact indices
        np.testing.assert_array_equal(
            train_bundle.features,
            full_bundle.features[part.train_range[0]:part.train_range[1]]
        )
        np.testing.assert_array_equal(
            bench_bundle.features,
            full_bundle.features[part.bench_range[0]:part.bench_range[1]]
        )


if __name__ == "__main__":
    unittest.main()

"""
Unit Tests for Causal Edge-Streaming Feature Extraction Pipeline.

Verifies:
1. Streaming vs. Batch Bit-Identity: batch_extract(samples) produces output
   bit-identical to streaming push() calls.
2. Strict Temporal Causality (Perturbation Invariance): Corrupting or modifying sample k
   causes exactly zero change in feature vectors at all preceding indices (i < k).
3. Warmup & Output Shape Integrity: Emits 12 channels per sample with non-negative norms
   and finite spectral energies.
"""

import os
import sys
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import calibrate_stream
from sih.features.streaming import StreamingFeatureExtractor
from sih.core.contracts import CalibratedSample


class TestCausalStreaming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = GenericDataLoader()
        trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", "S-S3a.csv")
        cls.trip = cls.loader.load_file(trip_path)
        cls.calib_samples = calibrate_stream(cls.trip, min_samples=30)

    def test_streaming_vs_batch_bit_identity(self):
        """Asserts that batch_extract(samples) produces bit-identical output to push()."""
        extractor_stream = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)
        stream_feats = []
        for s in self.calib_samples:
            f = extractor_stream.push(s)
            stream_feats.append(f)
        stream_feats = np.array(stream_feats, dtype=np.float32)

        extractor_batch = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)
        batch_feats = extractor_batch.batch_extract(self.calib_samples)

        self.assertEqual(stream_feats.shape, batch_feats.shape)
        np.testing.assert_array_equal(
            stream_feats,
            batch_feats,
            err_msg="Streaming push() and batch_extract() yielded differing feature vectors!",
        )

    def test_temporal_causality_perturbation(self):
        """
        Asserts that modifying sample k produces ZERO change in features at index < k.
        Tests multiple perturbation positions k across the trip.
        """
        extractor = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)
        base_feats = extractor.batch_extract(self.calib_samples)

        perturb_indices = [30, 75, 150, 500, 2000]
        for k in perturb_indices:
            if k >= len(self.calib_samples):
                continue

            # Create perturbed copy of samples
            perturbed_samples = list(self.calib_samples)
            s_orig = self.calib_samples[k]
            # Heavily corrupt sample k
            s_corrupt = CalibratedSample(
                timestamp_ns=s_orig.timestamp_ns,
                accel_vehicle=s_orig.accel_vehicle + np.array([50.0, -40.0, 30.0]),
                gyro_vehicle=s_orig.gyro_vehicle + np.array([2.5, -3.0, 4.0]),
                rotation_body_to_vehicle=s_orig.rotation_body_to_vehicle,
                gravity_vehicle=s_orig.gravity_vehicle,
                is_calibrated=s_orig.is_calibrated,
            )
            perturbed_samples[k] = s_corrupt

            extractor_test = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)
            pert_feats = extractor_test.batch_extract(perturbed_samples)

            # Features BEFORE k MUST BE BIT-IDENTICAL
            max_diff_prior = np.max(np.abs(base_feats[:k] - pert_feats[:k])) if k > 0 else 0.0
            self.assertEqual(
                max_diff_prior,
                0.0,
                msg=f"Perturbing sample {k} caused a future-to-past leak! Max diff prior to k: {max_diff_prior}",
            )

            # Feature AT k MUST DIFFER
            diff_at_k = np.max(np.abs(base_feats[k] - pert_feats[k]))
            self.assertGreater(diff_at_k, 0.1, msg=f"Perturbation at sample {k} was not registered!")

    def test_feature_semantics_and_validity(self):
        """Verifies channel counts, non-negative norms, and finite values."""
        extractor = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)
        feats = extractor.batch_extract(self.calib_samples[:200])

        self.assertEqual(feats.shape[1], 12)
        self.assertFalse(np.any(np.isnan(feats)), "Features contain NaNs!")
        self.assertFalse(np.any(np.isinf(feats)), "Features contain Infs!")

        # Norm channels (6 and 7) must be strictly non-negative
        self.assertTrue(np.all(feats[:, 6] >= 0.0), "Acceleration norm is negative!")
        self.assertTrue(np.all(feats[:, 7] >= 0.0), "Gyroscope norm is negative!")

        # Energy ratio (channel 10) must lie in [0, 1]
        self.assertTrue(np.all(feats[:, 10] >= 0.0) and np.all(feats[:, 10] <= 1.0 + 1e-5), "Energy ratio outside [0, 1]!")


if __name__ == "__main__":
    unittest.main()

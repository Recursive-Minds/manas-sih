"""
Leak-Free Verification Unit Test Suite for App Architecture (SIH PS 26168)
-------------------------------------------------------------------------
Verifies:
1. Architectural Import Boundary:
   server/engine_adapter.py must NEVER import server/evaluator.py or LiveEvaluator.
   Evaluator is the sole module that has access to post-blackout ground truth GNSS.
2. Router Ingestion Firewall:
   While state == 'WARMING_UP', GNSS samples reach both the engine and the evaluator.
   While state == 'BLACKOUT', GNSS samples reach ONLY the evaluator, and ZERO GNSS data
   reaches the dead-reckoning engine.
3. Engine State Protection:
   Even if on_gnss() were directly called during blackout, the engine rejects it and
   keeps its pre-blackout calibration, heading seed, and history completely frozen.
4. End-to-End Bit-Identical Trajectory Replay:
   Replays a real scenario twice:
   - Session A: Post-START GNSS consists of real ground truth GNSS fixes.
   - Session B: Post-START GNSS consists of NaNs and corrupted garbage coordinates.
   The dead-reckoning output trajectory (latitude, longitude, heading, velocities) must be
   100% bit-identical between Session A and Session B across the entire blackout duration.
"""

import os
import sys
import ast
import math
import copy
import asyncio
import unittest
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader
from sih.core.contracts import IMUSample, GNSSSample
from sih.calibration.mount import MountCalibrator
from server.engine_adapter import EngineAdapterStageA
from server.evaluator import LiveEvaluator
from server.router import NavigationRouter


class TestAppNoLeak(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = GenericDataLoader()
        trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", "S-M.csv")
        cls.trip = cls.loader.load_file(trip_path)
        cls.moving_fixes = [
            g for g in cls.trip.gnss_samples
            if g.is_valid and (g.speed_mps or 0.0) >= 2.0 and g.bearing_deg is not None
        ]
        assert len(cls.moving_fixes) >= 2, "Trip S-M should have at least 2 moving GNSS fixes"

    def test_architectural_import_boundary(self):
        """
        Enforces structural isolation: engine_adapter.py must NEVER import evaluator.py.
        Evaluator is the ONLY module permitted to observe post-START ground-truth GNSS.
        """
        adapter_path = os.path.join(ROOT_DIR, "server", "engine_adapter.py")
        with open(adapter_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=adapter_path)

        disallowed_modules = {"evaluator", "server.evaluator", "LiveEvaluator"}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotIn(
                        alias.name,
                        disallowed_modules,
                        f"Leak bug: engine_adapter.py imports disallowed module '{alias.name}'"
                    )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                self.assertNotIn(
                    mod,
                    disallowed_modules,
                    f"Leak bug: engine_adapter.py imports from disallowed module '{mod}'"
                )
                for alias in node.names:
                    self.assertNotIn(
                        alias.name,
                        disallowed_modules,
                        f"Leak bug: engine_adapter.py imports disallowed symbol '{alias.name}' from '{mod}'"
                    )

    def test_router_firewall_drops_gnss_during_blackout(self):
        """
        Verifies that NavigationRouter routes GNSS to the engine ONLY in WARMING_UP state,
        and completely blocks it when state == 'BLACKOUT'.
        """
        router = NavigationRouter()
        self.assertEqual(router.state, "WARMING_UP")

        first_gnss = self.moving_fixes[0]
        batch_warmup = {
            "state": "WARMING_UP",
            "gnss": [{
                "timestamp_ns": first_gnss.timestamp_ns,
                "latitude_deg": first_gnss.latitude_deg,
                "longitude_deg": first_gnss.longitude_deg,
                "altitude_m": first_gnss.altitude_m,
                "speed_mps": first_gnss.speed_mps,
                "bearing_deg": first_gnss.bearing_deg,
                "accuracy_h_m": first_gnss.accuracy_h_m,
                "is_valid": first_gnss.is_valid
            }],
            "imu": []
        }

        asyncio.run(router.process_batch(batch_warmup))
        # Engine should have received and registered the moving fix
        self.assertEqual(router.engine.moving_gnss_fixes_count, 1)
        self.assertEqual(len(router.evaluator.gnss_fixes), 1)

        # Transition to blackout
        router.set_blackout(True)
        self.assertEqual(router.state, "BLACKOUT")

        second_gnss = self.moving_fixes[1]
        batch_blackout = {
            "state": "BLACKOUT",
            "gnss": [{
                "timestamp_ns": second_gnss.timestamp_ns,
                "latitude_deg": second_gnss.latitude_deg,
                "longitude_deg": second_gnss.longitude_deg,
                "altitude_m": second_gnss.altitude_m,
                "speed_mps": second_gnss.speed_mps,
                "bearing_deg": second_gnss.bearing_deg,
                "accuracy_h_m": second_gnss.accuracy_h_m,
                "is_valid": second_gnss.is_valid
            }],
            "imu": []
        }

        asyncio.run(router.process_batch(batch_blackout))
        # Engine's moving GNSS count must STILL be 1 (strictly blocked by router firewall)
        self.assertEqual(
            router.engine.moving_gnss_fixes_count, 1,
            "Leak error: Engine received GNSS sample during blackout!"
        )
        # Evaluator MUST have received it (for ground truth error tracking)
        self.assertEqual(len(router.evaluator.gnss_fixes), 2)

    def test_engine_adapter_defensive_gnss_drop(self):
        """
        Verifies that even if on_gnss() is directly invoked on the engine during blackout,
        it drops the sample defensively and does not modify its internal state.
        """
        engine = EngineAdapterStageA()
        g0 = self.moving_fixes[0]
        engine.on_gnss(g0)
        self.assertEqual(engine.moving_gnss_fixes_count, 1)

        engine.set_blackout(True)
        g1 = self.moving_fixes[1]
        engine.on_gnss(g1)
        self.assertEqual(
            engine.moving_gnss_fixes_count, 1,
            "Defensive leak error: EngineAdapter Stage A updated GNSS state during active blackout!"
        )

    def test_replayed_session_bit_identical_under_nan_and_garbage_injection(self):
        """
        The Master No-Leak Replay Test:
        Replays an entire scenario through NavigationRouter twice:
        1. Clean Session: Real GNSS provided after START.
        2. Corrupted Session: Post-START GNSS replaced by NaNs and garbage coordinates.
        Every DR position coordinate (lat, lon, heading, ENU velocities) must be BIT-IDENTICAL.
        """
        warmup_duration_s = 30.0
        blackout_duration_s = 45.0

        t0_ns = self.trip.imu_samples[0].timestamp_ns
        t_bo_start_ns = t0_ns + int(warmup_duration_s * 1e9)
        t_bo_end_ns = t_bo_start_ns + int(blackout_duration_s * 1e9)

        # Slice relevant trip samples
        imu_segment = [
            im for im in self.trip.imu_samples
            if t0_ns <= im.timestamp_ns <= t_bo_end_ns
        ]
        gnss_segment = [
            g for g in self.trip.gnss_samples
            if t0_ns <= g.timestamp_ns <= t_bo_end_ns
        ]

        # Function to run a full session through NavigationRouter
        def run_router_session(inject_corrupted_post_start_gnss: bool):
            router = NavigationRouter(
                ref_lat=self.trip.gnss_samples[0].latitude_deg,
                ref_lon=self.trip.gnss_samples[0].longitude_deg,
                ref_alt=self.trip.gnss_samples[0].altitude_m,
            )

            dr_outputs = []
            gnss_ptr = 0
            n_gnss = len(gnss_segment)

            for imu in imu_segment:
                ts = imu.timestamp_ns

                # Check state transition
                if ts >= t_bo_start_ns and router.state == "WARMING_UP":
                    router.set_blackout(True)

                # Feed any pending GNSS fixes up to this IMU timestamp
                while gnss_ptr < n_gnss and gnss_segment[gnss_ptr].timestamp_ns <= ts:
                    g = gnss_segment[gnss_ptr]
                    gnss_ptr += 1

                    if router.state == "BLACKOUT" and inject_corrupted_post_start_gnss:
                        # Corrupt post-blackout GNSS with NaNs and impossible coordinates
                        g_payload = {
                            "timestamp_ns": g.timestamp_ns,
                            "latitude_deg": float("nan"),
                            "longitude_deg": 89.99999,
                            "altitude_m": -9999.0,
                            "speed_mps": 999.9,
                            "bearing_deg": float("nan"),
                            "accuracy_h_m": 0.001,
                            "is_valid": True,
                        }
                    else:
                        g_payload = {
                            "timestamp_ns": g.timestamp_ns,
                            "latitude_deg": g.latitude_deg,
                            "longitude_deg": g.longitude_deg,
                            "altitude_m": g.altitude_m,
                            "speed_mps": g.speed_mps,
                            "bearing_deg": g.bearing_deg,
                            "accuracy_h_m": g.accuracy_h_m,
                            "is_valid": g.is_valid,
                        }

                    batch_g = {
                        "state": router.state,
                        "gnss": [g_payload],
                        "imu": []
                    }
                    asyncio.run(router.process_batch(batch_g))

                # Feed IMU sample
                batch_im = {
                    "state": router.state,
                    "gnss": [],
                    "imu": [{
                        "timestamp_ns": imu.timestamp_ns,
                        "accel": imu.accel.tolist(),
                        "gyro": imu.gyro.tolist(),
                    }]
                }
                asyncio.run(router.process_batch(batch_im))

                if router.state == "BLACKOUT":
                    latest_dr = router.engine.latest_fused_position
                    if latest_dr is not None:
                        dr_outputs.append((
                            float(latest_dr.latitude_deg),
                            float(latest_dr.longitude_deg),
                            float(latest_dr.heading_rad),
                            float(latest_dr.velocity_enu_mps[0]),
                            float(latest_dr.velocity_enu_mps[1]),
                        ))

            return dr_outputs

        # Run both sessions
        dr_clean = run_router_session(inject_corrupted_post_start_gnss=False)
        dr_corrupted = run_router_session(inject_corrupted_post_start_gnss=True)

        self.assertGreater(len(dr_clean), 0, "No DR samples recorded during blackout!")
        self.assertEqual(len(dr_clean), len(dr_corrupted), "Sample counts during blackout differ!")

        # Assert strict bit-identity across all samples
        for i, (clean, corrupted) in enumerate(zip(dr_clean, dr_corrupted)):
            self.assertEqual(
                clean[0], corrupted[0],
                f"Latitude divergence at step {i}: {clean[0]} != {corrupted[0]}"
            )
            self.assertEqual(
                clean[1], corrupted[1],
                f"Longitude divergence at step {i}: {clean[1]} != {corrupted[1]}"
            )
            self.assertEqual(
                clean[2], corrupted[2],
                f"Heading divergence at step {i}: {clean[2]} != {corrupted[2]}"
            )
            self.assertEqual(
                clean[3], corrupted[3],
                f"Velocity East divergence at step {i}: {clean[3]} != {corrupted[3]}"
            )
            self.assertEqual(
                clean[4], corrupted[4],
                f"Velocity North divergence at step {i}: {clean[4]} != {corrupted[4]}"
            )


if __name__ == "__main__":
    unittest.main()

"""
scripts/export_phone_parity_bundle.py
--------------------------------------
Exports the S-S3a trip data, road network, and laptop batch reference results
into a compact pickle bundle to be replayed directly on the Android device for Step 4.
"""

import os
import sys
import pickle
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sih.data.loader import GenericDataLoader
from sih.calibration.mount import calibrate_stream
from sih.map.network import load_trip_road_network
from sih.models.inference import load_ai_model, predict_velocities
from sih.engine.dead_reckoning_engine import run_dead_reckoning_scenario
from sih.data.geo import geodetic_to_enu
from sih.models.predictor import create_predictor


def main():
    print("Loading trip S-S3a...")
    trip_path = os.path.join(ROOT_DIR, "data", "raw", "iovnbd_trips", "S-S3a.csv")
    trip = GenericDataLoader().load_file(trip_path)
    calibs = calibrate_stream(trip, min_samples=30)

    print("Loading road network...")
    rnet, _ = load_trip_road_network(trip, map_source="osm", cache_dir="data/maps/cache")

    print("Computing batch velocity references...")
    import torch
    device = torch.device("cpu")
    model, norm_mean, norm_std, model_type = load_ai_model(device)
    v_preds = predict_velocities(
        model,
        calibs,
        norm_mean,
        norm_std,
        device,
        model_type=model_type,
        apply_smoothing=True,
    )

    targets = [
        {"id": 22, "dur": 45.0, "dist": 475.2},
        {"id": 23, "dur": 75.0, "dist": 1128.4},
        {"id": 25, "dur": 45.0, "dist": 614.3},
        {"id": 26, "dur": 75.0, "dist": 892.8},
        {"id": 30, "dur": 60.0, "dist": 244.2},
    ]

    cand_gnss = [
        g for g in trip.gnss_samples
        if g.is_valid and g.speed_mps is not None and g.speed_mps >= 1.0 and g.bearing_deg is not None
    ]

    scenarios = []
    for tgt in targets:
        best_g = None
        best_diff = float("inf")
        for g in cand_gnss:
            bo_start_ns = g.timestamp_ns
            bo_end_ns = bo_start_ns + int(tgt["dur"] * 1e9)
            bo_gnss = [x for x in trip.gnss_samples if bo_start_ns <= x.timestamp_ns <= bo_end_ns and x.is_valid]
            if len(bo_gnss) < 3:
                continue
            pts = [
                geodetic_to_enu(x.latitude_deg, x.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
                for x in bo_gnss
            ]
            pts = np.array(pts)
            dist = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            diff = abs(dist - tgt["dist"])
            if diff < best_diff:
                best_diff = diff
                best_g = g
                if diff < 1.0:
                    break

        batch_res = run_dead_reckoning_scenario(trip, calibs, v_preds, rnet, best_g, tgt["dur"], domain="Mixed")
        bo_start_ns = best_g.timestamp_ns
        bo_end_ns = bo_start_ns + int(tgt["dur"] * 1e9)
        bo_gnss = [x for x in trip.gnss_samples if bo_start_ns <= x.timestamp_ns <= bo_end_ns and x.is_valid]
        eval_t = float(bo_gnss[-1].timestamp_ns)
        gt_end_enu = geodetic_to_enu(
            bo_gnss[-1].latitude_deg, bo_gnss[-1].longitude_deg, 0.0,
            trip.reference_lat_deg, trip.reference_lon_deg, 0.0
        )[:2]

        batch_pts = batch_res["map_pts"]
        batch_ts = batch_res["time_rel_s"] * 1e9 + bo_start_ns
        b_e = float(np.interp(eval_t, batch_ts, batch_pts[:, 0]))
        b_n = float(np.interp(eval_t, batch_ts, batch_pts[:, 1]))
        b_eval_pt = np.array([b_e, b_n])
        batch_err = float(np.linalg.norm(b_eval_pt - gt_end_enu))

        scenarios.append({
            "target": tgt,
            "entry_gnss": best_g,
            "batch_err_m": batch_err,
            "batch_pts": batch_pts,
            "batch_ts": batch_ts,
            "eval_t": eval_t,
            "b_e": b_e,
            "b_n": b_n,
            "gt_end_enu": gt_end_enu,
            "dist_m": float(batch_res["dist_m"]),
        })
        print(f"Scenario #{tgt['id']}: Laptop Batch Err = {batch_err:.2f} m, Dist = {batch_res['dist_m']:.1f} m")

    bundle = {
        "trip": trip,
        "calibs": calibs,
        "rnet": rnet,
        "norm_mean": norm_mean,
        "norm_std": norm_std,
        "scenarios": scenarios,
    }

    out_dir = os.path.join(ROOT_DIR, "data", "exported")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "s_s3a_parity_bundle.pkl")
    print(f"Serializing bundle to {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f, protocol=4)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Bundle successfully created: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()

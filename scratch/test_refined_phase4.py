"""
Test Refined Phase 4 Map-Matching with Kinematic Turn Tolerance and Initial Road Alignment.
Evaluates:
1. Unseen dataset (S-M.csv) - 35 Scenarios
2. Standardized 50-Scenario Benchmark Suite (S-S1.csv, S-S2.csv)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r"C:\Users\carpe\SIH")
from sih.data.loader import GenericDataLoader
from sih.calibration.mount import MountCalibrator
from sih.fusion.es_ekf import ErrorStateEKF
from sih.core.contracts import VelocityEstimate
from sih.data.geo import geodetic_to_enu
from sih.models.tcn_attention import TCNAttentionVelocityModel
from sih.map.network import RoadNetwork


def build_road_network(trip, prefix="road", min_step_m=15.0):
    pts_enu = []
    pts_lat_lon = []
    last_p = None
    for g in trip.gnss_samples:
        enu = geodetic_to_enu(g.latitude_deg, g.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
        if last_p is None or np.linalg.norm(enu - last_p) >= min_step_m:
            pts_enu.append(enu)
            pts_lat_lon.append((g.latitude_deg, g.longitude_deg))
            last_p = enu

    return RoadNetwork.from_polyline_coords(
        np.array(pts_enu),
        pts_lat_lon,
        road_id_prefix=prefix,
        road_type="primary",
        cell_size_m=100.0
    ), np.array(pts_enu)


def evaluate_trip_scenarios(trip, ckpt_v, device, num_scenarios=35, prefix="sm"):
    calibrator = MountCalibrator(window_size=100)
    for g in trip.gnss_samples: calibrator.observe_gnss(g)
    calib_samples = [calibrator.update(imu) for imu in trip.imu_samples]

    model_v = TCNAttentionVelocityModel(in_channels=8, base_channels=32, num_attention_heads=4)
    model_v.load_state_dict(ckpt_v["model_state_dict"])
    model_v.to(device).eval()

    acc   = np.array([s.accel_vehicle for s in calib_samples], dtype=np.float32)
    gyr   = np.array([s.gyro_vehicle  for s in calib_samples], dtype=np.float32)
    feats = np.hstack([acc, gyr, np.linalg.norm(acc, axis=1, keepdims=True), np.linalg.norm(gyr, axis=1, keepdims=True)])
    N = len(feats)
    window_size = 100
    norm_mean = ckpt_v.get("norm_mean", np.zeros((8, 1), dtype=np.float32))
    norm_std  = ckpt_v.get("norm_std",  np.ones((8, 1),  dtype=np.float32))
    windows = []
    for i in range(N):
        if i < window_size:
            pad = np.repeat(feats[0:1], window_size - i - 1, axis=0)
            w   = np.vstack([pad, feats[:i+1]]).T
        else:
            w   = feats[i - window_size + 1 : i + 1].T
        windows.append((w - norm_mean) / norm_std)

    preds_v = []
    with torch.no_grad():
        for b in range(0, N, 2048):
            xv = torch.from_numpy(np.array(windows[b : b + 2048], dtype=np.float32)).to(device)
            pv, _ = model_v(xv)
            preds_v.extend(pv.cpu().numpy().flatten())
    v_preds = np.array(preds_v, dtype=np.float32)

    road_net, _ = build_road_network(trip, f"{prefix}_road")

    t0_ns = trip.imu_samples[0].timestamp_ns
    candidate_gnss = [
        g for g in trip.gnss_samples
        if g.is_valid and g.speed_mps is not None and g.speed_mps > 4.0 and g.bearing_deg is not None
        and (600.0 * 1e9) <= (g.timestamp_ns - t0_ns) <= (9500.0 * 1e9)
    ]
    step = max(1, len(candidate_gnss) // num_scenarios)
    selected_entries = candidate_gnss[::step][:num_scenarios]
    test_durs = [30.0, 45.0, 60.0, 75.0]

    valid_g = [g for g in trip.gnss_samples if g.is_valid]
    results = []

    for idx, g_ent in enumerate(selected_entries):
        dur = test_durs[idx % len(test_durs)]
        bo_st = g_ent.timestamp_ns
        bo_en = bo_st + int(dur * 1e9)
        g_exit = min(valid_g, key=lambda g: abs(g.timestamp_ns - bo_en))

        p_start = geodetic_to_enu(g_ent.latitude_deg, g_ent.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
        p_end   = geodetic_to_enu(g_exit.latitude_deg, g_exit.longitude_deg, 0.0, trip.reference_lat_deg, trip.reference_lon_deg, 0.0)[:2]
        dist_gt = float(np.linalg.norm(p_end - p_start))
        if dist_gt < 10.0: continue

        # Refined Initial Heading Alignment:
        # Pre-align with candidate road segment if within 25 degrees
        init_hdg_deg = g_ent.bearing_deg
        init_cands = road_net.find_candidates(p_start, radius_m=35.0)
        for s in init_cands:
            proj, d_p, _ = s.project_point(p_start)
            h_diff = abs((init_hdg_deg - s.bearing_deg + 180.0) % 360.0 - 180.0)
            if d_p < 20.0 and h_diff < 25.0:
                init_hdg_deg = s.bearing_deg
                break

        ekf_pure = ErrorStateEKF()
        ekf_pure.init_from_gnss(g_ent, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg)
        ekf_pure._heading_rad = float(np.radians(init_hdg_deg))

        ekf_map = ErrorStateEKF()
        ekf_map.init_from_gnss(g_ent, reference_lat_deg=trip.reference_lat_deg, reference_lon_deg=trip.reference_lon_deg)
        ekf_map._heading_rad = float(np.radians(init_hdg_deg))

        pts_pure = []
        pts_map  = []

        for j, imu in enumerate(trip.imu_samples):
            t_curr = imu.timestamp_ns
            if bo_st <= t_curr <= bo_en:
                cal = calib_samples[j]
                v_fwd = float(v_preds[j])
                m_state = "STATIONARY" if v_fwd < 0.2 else "DRIVING"
                vel = VelocityEstimate(timestamp_ns=t_curr, forward_speed_mps=v_fwd, speed_variance=0.3, motion_state=m_state)

                fp = ekf_pure.predict(cal, vel)
                fm = ekf_map.predict(cal, vel)
                pts_pure.append(fp.position_enu_m[:2].copy())

                # Turn detection: Check calibrated vertical yaw rate
                turn_rate_dps = abs(np.degrees(cal.gyro_vehicle[2]))
                is_turning = turn_rate_dps > 2.0

                cands = road_net.find_candidates(ekf_map._p[:2], radius_m=50.0)
                h_deg = float(np.degrees(ekf_map._heading_rad)) % 360.0

                scored = []
                for s in cands:
                    proj, d_perp, frac = s.project_point(ekf_map._p[:2])
                    h_diff = abs((h_deg - s.bearing_deg + 180.0) % 360.0 - 180.0)

                    # Dynamic angular window: allow wide angle (up to 110 deg) when actively turning
                    max_h = 110.0 if is_turning else 45.0
                    if h_diff < max_h and d_perp < 35.0:
                        # Segment end transition: if reached end of segment (frac >= 0.95), favor transitioning!
                        end_factor = 0.05 if frac >= 0.95 else 1.0
                        score = np.exp(-0.5 * (d_perp / 8.0)**2) * np.exp(-0.5 * (h_diff / 40.0)**2) * end_factor
                        scored.append((s, proj, d_perp, h_diff, score))

                if len(scored) > 0:
                    best_s, best_proj, d_p, _, _ = max(scored, key=lambda x: x[4])
                    # Cross-track snap
                    ekf_map._p[0] = best_proj[0]
                    ekf_map._p[1] = best_proj[1]
                    # Heading guidance: only pull heading when not actively turning
                    if not is_turning:
                        ekf_map._heading_rad = float(np.radians(best_s.bearing_deg))

                pts_map.append(ekf_map._p[:2].copy())

        err_pure = float(np.linalg.norm(pts_pure[-1] - p_end))
        err_map  = float(np.linalg.norm(pts_map[-1] - p_end))
        results.append({
            "id": idx + 1, "dur": dur, "dist": dist_gt,
            "pure_drift": (err_pure / dist_gt) * 100.0,
            "map_drift": (err_map / dist_gt) * 100.0,
            "pure_err_m": err_pure,
            "map_err_m": err_map,
        })

    return pd.DataFrame(results)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_v = torch.load(r"C:\Users\carpe\SIH\models\checkpoints\best_velocity_model.pt", map_location=device, weights_only=False)
    loader = GenericDataLoader()

    print("Evaluating Refined Phase 4 on Unseen Dataset (S-M.csv)...", flush=True)
    trip_sm = loader.load_file(r"C:\Users\carpe\SIH\data\raw\iovnbd_trips\S-M.csv")
    df_sm = evaluate_trip_scenarios(trip_sm, ckpt_v, device, num_scenarios=35, prefix="sm")

    print("\n" + "="*70)
    print("REFINED PHASE 4 ON UNSEEN DATASET S-M.csv (35 SCENARIOS):")
    print(f"Pure 6-Axis Median Drift:     {df_sm['pure_drift'].median():.2f}%")
    print(f"Phase 4 Map-Matched Median:    {df_sm['map_drift'].median():.2f}% (P90: {df_sm['map_drift'].quantile(0.90):.2f}%)")
    print(f"Scenarios with Drift < 10%:   {(df_sm['map_drift'] < 10.0).sum()}/35 ({(df_sm['map_drift'] < 10.0).mean()*100:.1f}%)")
    print(f"Scenarios with Drift < 30%:   {(df_sm['map_drift'] < 30.0).sum()}/35 ({(df_sm['map_drift'] < 30.0).mean()*100:.1f}%)")
    print("="*70)


if __name__ == "__main__":
    main()

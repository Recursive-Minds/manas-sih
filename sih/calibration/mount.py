"""
Mount Auto-Calibration Engine for Smartphone-to-Vehicle Frame Alignment.

Decouples smartphone physical mounting orientation (portrait, landscape, tilted,
windshield mount, dashboard cradle) from the vehicle body frame (X=Forward, Y=Left, Z=Up).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import numpy as np
from scipy.spatial.transform import Rotation as R

from sih.core.contracts import IMUSample, GNSSSample, CalibratedSample
from sih.core.interfaces import ICalibration
from sih.core.pipeline import register_calibration


@dataclass(frozen=True)
class MountAlignment:
    """Represents the calibrated spatial relationship between phone and vehicle."""
    is_calibrated: bool
    R_phone_to_vehicle: R
    forward_axis_phone: np.ndarray
    lateral_axis_phone: np.ndarray
    vertical_axis_phone: np.ndarray
    yaw_axis_index: int       # 0, 1, 2
    yaw_axis_sign: float      # +1.0 or -1.0
    mount_yaw_offset_rad: float
    pitch_deg: float
    roll_deg: float


class MountCalibrator(ICalibration):
    """
    Online Auto-Calibration Stage implementing ICalibration.
    
    Observes pre-blackout smartphone accelerometer, gyroscope, and GNSS ground track
    to automatically determine the phone's 3D orientation in the vehicle cradle.
    """

    def __init__(self, min_samples: int = 30, min_speed_mps: float = 2.0, **params) -> None:
        self.min_samples = min_samples
        self.min_speed_mps = min_speed_mps
        self.reset()

    def reset(self) -> None:
        self._accel_buf: List[np.ndarray] = []
        self._gyro_buf: List[np.ndarray] = []
        self._imu_ts: List[int] = []
        self._gnss_buf: List[GNSSSample] = []
        self._alignment: Optional[MountAlignment] = None
        self._turn_events: List[Tuple[float, float, float, float]] = []
        self._prev_turn_gnss: Optional[GNSSSample] = None
        self._yaw_locked: bool = False

    @property
    def is_calibrated(self) -> bool:
        return self._alignment is not None and self._alignment.is_calibrated

    def is_aligned(self) -> bool:
        return self.is_calibrated

    def notify_mount_change(self) -> None:
        self.reset()

    @property
    def alignment(self) -> Optional[MountAlignment]:
        return self._alignment

    def observe_gnss(self, gnss: GNSSSample) -> None:
        """Accumulate GNSS samples for turn-rate and course-over-ground calibration."""
        if gnss.is_valid and gnss.bearing_deg is not None and (gnss.speed_mps or 0.0) >= self.min_speed_mps:
            self._gnss_buf.append(gnss)

            # Detect genuine turn event between consecutive moving GNSS fixes
            if self._prev_turn_gnss is not None and not self._yaw_locked:
                prev_g = self._prev_turn_gnss
                dt_g = (gnss.timestamp_ns - prev_g.timestamp_ns) * 1e-9
                if 0.2 <= dt_g <= 15.0:
                    d_b_deg = (gnss.bearing_deg - prev_g.bearing_deg + 180.0) % 360.0 - 180.0
                    if abs(d_b_deg) >= 2.5:
                        t1, t2 = prev_g.timestamp_ns, gnss.timestamp_ns
                        all_ts = np.array(self._imu_ts)
                        mask = (all_ts >= t1) & (all_ts <= t2)
                        if np.sum(mask) > 1:
                            sub_gy = np.array(self._gyro_buf)[mask]
                            dt_imu = np.diff(all_ts[mask]) * 1e-9
                            d_th = np.sum(0.5 * (sub_gy[:-1] + sub_gy[1:]) * dt_imu[:, None], axis=0)
                            self._turn_events.append((float(np.radians(d_b_deg)), float(d_th[0]), float(d_th[1]), float(d_th[2])))
            self._prev_turn_gnss = gnss

            if len(self._accel_buf) >= self.min_samples and len(self._gnss_buf) >= 3 and not self._yaw_locked:
                self._compute_calibration()

    def _compute_calibration(self) -> None:
        accels = np.array(self._accel_buf[:min(len(self._accel_buf), 300)])
        g_body = np.mean(accels, axis=0)
        g_norm = np.linalg.norm(g_body)
        g_body_norm = g_body / max(g_norm, 1e-4)

        # 1. Leveling: Align measured gravity with Vehicle Up [0, 0, 1]
        up_veh = np.array([0.0, 0.0, 1.0])
        cross = np.cross(g_body_norm, up_veh)
        dot = float(np.dot(g_body_norm, up_veh))
        cross_norm = float(np.linalg.norm(cross))

        if cross_norm < 1e-6:
            R_level = R.identity() if dot > 0 else R.from_rotvec(np.array([np.pi, 0.0, 0.0]))
        else:
            axis = cross / cross_norm
            angle = float(np.arctan2(cross_norm, dot))
            R_level = R.from_rotvec(axis * angle)

        # 2. Correlate gyro channels with genuine GNSS turn events
        yaw_idx = self._alignment.yaw_axis_index if self._alignment else 2
        yaw_sign = self._alignment.yaw_axis_sign if self._alignment else 1.0
        is_locked = self._yaw_locked

        if len(self._turn_events) >= 8:
            evs = np.array(self._turn_events)
            energies = np.array([np.mean(np.abs(evs[:, a + 1])) for a in range(3)])
            corrs = np.array([np.corrcoef(evs[:, 0], evs[:, a + 1])[0, 1] for a in range(3)])
            valid_c = np.where(np.isnan(corrs), 0.0, corrs)
            scores = np.abs(valid_c) * (energies + 1e-6)
            best_a = int(np.argmax(scores))

            # Directional slope: in EKF, heading_rad = heading_rad - w_z * dt
            # If d_bearing > 0 (right turn) and d_theta < 0, w_z must be positive to increase heading -> yaw_sign = +1.0
            # If d_bearing > 0 and d_theta > 0, yaw_sign = -1.0
            slope = np.polyfit(evs[:, best_a + 1], evs[:, 0], 1)[0]
            computed_sign = -1.0 if slope > 0 else 1.0

            if abs(valid_c[best_a]) >= 0.12:
                yaw_idx = best_a
                yaw_sign = computed_sign
                sorted_scores = np.sort(scores)
                separation = sorted_scores[-1] / max(sorted_scores[-2], 1e-6)
                if len(self._turn_events) >= 15 and abs(valid_c[best_a]) >= 0.35 and separation >= 1.5:
                    is_locked = True
        elif len(self._accel_buf) >= 30 and len(self._gyro_buf) >= 30 and not self._alignment:
            # Initial heuristic based on gyro dynamic variance across axes
            gyros_arr = np.array(self._gyro_buf[:min(len(self._gyro_buf), 200)])
            stds = np.std(gyros_arr, axis=0)
            yaw_idx = int(np.argmax(stds))
            yaw_sign = 1.0

        euler = R_level.as_euler("xyz", degrees=True)
        self._alignment = MountAlignment(
            is_calibrated=True,
            R_phone_to_vehicle=R_level,
            forward_axis_phone=R_level.as_matrix()[0, :],
            lateral_axis_phone=R_level.as_matrix()[1, :],
            vertical_axis_phone=R_level.as_matrix()[2, :],
            yaw_axis_index=yaw_idx,
            yaw_axis_sign=yaw_sign,
            mount_yaw_offset_rad=0.0,
            pitch_deg=float(euler[0]),
            roll_deg=float(euler[1]),
        )
        self._yaw_locked = is_locked

    def update(self, imu: IMUSample) -> CalibratedSample:
        """Transform IMUSample to vehicle CalibratedSample."""
        self._accel_buf.append(imu.accel.copy())
        self._gyro_buf.append(imu.gyro.copy())
        self._imu_ts.append(imu.timestamp_ns)

        if not self.is_calibrated and len(self._accel_buf) >= self.min_samples:
            self._compute_calibration()

        if self._alignment is None or not self._alignment.is_calibrated:
            return CalibratedSample(
                timestamp_ns=imu.timestamp_ns,
                accel_vehicle=imu.accel.copy(),
                gyro_vehicle=imu.gyro.copy(),
                rotation_body_to_vehicle=np.eye(3, dtype=np.float64),
                gravity_vehicle=np.array([0.0, 0.0, 9.80665], dtype=np.float64),
                is_calibrated=False,
            )

        R_mat = self._alignment.R_phone_to_vehicle.as_matrix()
        acc_v = R_mat @ imu.accel

        # Transform 3D gyro vector into leveled vehicle frame:
        # Z-axis is assigned the signed vehicle yaw rate; X and Y represent roll and pitch
        gyro_v = np.zeros(3, dtype=np.float64)
        yaw_idx = self._alignment.yaw_axis_index
        gyro_v[2] = self._alignment.yaw_axis_sign * imu.gyro[yaw_idx]
        rem_axes = [a for a in [0, 1, 2] if a != yaw_idx]
        gyro_v[0] = imu.gyro[rem_axes[0]]
        gyro_v[1] = imu.gyro[rem_axes[1]]

        return CalibratedSample(
            timestamp_ns=imu.timestamp_ns,
            accel_vehicle=acc_v,
            gyro_vehicle=gyro_v,
            rotation_body_to_vehicle=R_mat,
            gravity_vehicle=np.array([0.0, 0.0, 9.80665], dtype=np.float64),
            is_calibrated=True,
        )

    def calibrate(self, raw: IMUSample) -> CalibratedSample:
        return self.update(raw)


# Register in factory registry
register_calibration(
    "mount_calibrator",
    lambda **params: MountCalibrator(**params)
)
register_calibration(
    "auto",
    lambda **params: MountCalibrator(**params)
)

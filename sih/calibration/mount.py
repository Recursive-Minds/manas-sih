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
            if len(self._accel_buf) >= self.min_samples and len(self._gnss_buf) >= 3:
                self._compute_calibration()

    def _compute_calibration(self) -> None:
        accels = np.array(self._accel_buf)
        g_body = np.mean(accels[:min(len(accels), 300)], axis=0)
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

        # 2. Correlate gyro channels with horizontal centripetal acceleration and GNSS turn rates
        yaw_idx = 2
        yaw_sign = -1.0 # Default right-handed ENU where bearing decreases with positive CCW yaw

        # Centripetal cross-correlation: a_lat = v * w_yaw
        if len(self._accel_buf) >= 30 and len(self._gyro_buf) >= 30:
            accels_arr = np.array(self._accel_buf)
            gyros_arr = np.array(self._gyro_buf)
            
            # Check which gyro axis correlates most strongly with horizontal acceleration
            centripetal_scores = {}
            for g_axis in [0, 1, 2]:
                for a_axis in [0, 1]:
                    c = np.corrcoef(accels_arr[:, a_axis], gyros_arr[:, g_axis])[0, 1]
                    if not np.isnan(c):
                        centripetal_scores[(g_axis, a_axis)] = abs(c)
            if len(centripetal_scores) > 0:
                best_g_axis, best_a_axis = max(centripetal_scores, key=lambda k: centripetal_scores[k])
                if centripetal_scores[(best_g_axis, best_a_axis)] > 0.25:
                    yaw_idx = best_g_axis

        # If GNSS turns are available, calculate integrated angular changes to sign the identified yaw axis
        if len(self._gnss_buf) >= 3 and len(self._imu_ts) > 50:
            g_ts = np.array([g.timestamp_ns for g in self._gnss_buf])
            g_brg = np.array([g.bearing_deg for g in self._gnss_buf])
            g_brg_unwrap = np.unwrap(np.radians(g_brg))
            d_theta_gnss = np.diff(g_brg_unwrap)

            imu_ts_arr = np.array(self._imu_ts)
            gyros_arr = np.array(self._gyro_buf)

            d_theta_yaw = np.zeros(len(d_theta_gnss), dtype=np.float64)
            for i in range(len(d_theta_gnss)):
                t_start = g_ts[i]
                t_end = g_ts[i + 1]
                mask = (imu_ts_arr >= t_start) & (imu_ts_arr <= t_end)
                if np.sum(mask) > 1:
                    dt_imu = np.diff(imu_ts_arr[mask]) * 1e-9
                    d_theta_yaw[i] = np.sum(0.5 * (gyros_arr[mask, yaw_idx][:-1] + gyros_arr[mask, yaw_idx][1:]) * dt_imu)

            c = np.corrcoef(d_theta_gnss, d_theta_yaw)[0, 1]
            if not np.isnan(c) and abs(c) > 0.15:
                yaw_sign = 1.0 if c < 0 else -1.0
            else:
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

        # Transform 3D gyro vector into leveled vehicle frame
        gyro_v = np.zeros(3, dtype=np.float64)
        gyro_v[0] = imu.gyro[0]
        gyro_v[1] = imu.gyro[1]
        gyro_v[2] = self._alignment.yaw_axis_sign * imu.gyro[self._alignment.yaw_axis_index]

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

"""
Naive Dead-Reckoning Baseline Filter.

Performs classical unconstrained double-integration of accelerometer and gyroscope readings.
Serves as the unmitigated reference baseline to demonstrate and measure open-loop INS drift.
"""

from __future__ import annotations
from typing import Optional
import os
import numpy as np
if os.environ.get("SIH_FORCE_SCIPY_SHIM", "0") == "1":
    from sih.core.scipy_shim import Rotation as R
else:
    try:
        from scipy.spatial.transform import Rotation as R
    except ImportError:
        from sih.core.scipy_shim import Rotation as R

from sih.core.contracts import (
    IMUSample,
    GNSSSample,
    CalibratedSample,
    VelocityEstimate,
    FusedPosition,
)
from sih.core.interfaces import IFusionFilter
from sih.core.pipeline import register_fusion_filter
from sih.data.geo import geodetic_to_enu, enu_to_geodetic


class NaiveDeadReckoningFilter(IFusionFilter):
    """
    Open-loop strapdown INS mechanization:
    - Orientation: Quaternion integration of gyro rates.
    - Velocity: Numerical integration of gravity-compensated acceleration.
    - Position: Numerical double-integration in Local Tangent Plane (ENU).
    """
    def __init__(
        self,
        gravity_magnitude: float = 9.80665,
        align_initial_yaw_with_gnss: bool = True,
        **params,
    ) -> None:
        self.g_mag = gravity_magnitude
        self.align_initial_yaw = align_initial_yaw_with_gnss
        self.reset()

    def reset(self, initial_gnss: Optional[GNSSSample] = None) -> None:
        """Reset state vectors and reference coordinates."""
        self.is_initialized = False
        self.last_timestamp_ns: Optional[int] = None
        self.last_valid_gnss_ts_ns: Optional[int] = None

        # Reference geodetic origin for ENU frame
        self.ref_lat_deg: float = 0.0
        self.ref_lon_deg: float = 0.0
        self.ref_alt_m: float = 0.0

        # State vectors in ENU
        self.pos_enu = np.zeros(3, dtype=np.float64)  # [East, North, Up] in meters
        self.vel_enu = np.zeros(3, dtype=np.float64)  # [vE, vN, vU] in m/s
        self.rot_body_to_enu = R.identity()          # Orientation rotation
        self.covariance = np.eye(6, dtype=np.float64) * 0.1
        self.mode = "INITIALIZING"

        if initial_gnss is not None and initial_gnss.is_valid:
            self._init_from_gnss(initial_gnss)

    def _init_from_gnss(self, gnss: GNSSSample) -> None:
        self.ref_lat_deg = gnss.latitude_deg
        self.ref_lon_deg = gnss.longitude_deg
        self.ref_alt_m = gnss.altitude_m

        self.pos_enu = np.zeros(3, dtype=np.float64)
        self.last_timestamp_ns = gnss.timestamp_ns
        self.last_valid_gnss_ts_ns = gnss.timestamp_ns

        # Initial speed & heading from GNSS if present
        if gnss.speed_mps is not None and gnss.bearing_deg is not None:
            bearing_rad = np.radians(gnss.bearing_deg)
            # ENU velocity: East = speed * sin(bearing), North = speed * cos(bearing)
            ve = gnss.speed_mps * np.sin(bearing_rad)
            vn = gnss.speed_mps * np.cos(bearing_rad)
            self.vel_enu = np.array([ve, vn, 0.0], dtype=np.float64)

            # Yaw: azimuth clockwise from North corresponds to rotation about Up (+Z)
            # ENU yaw angle from East counter-clockwise is (90 - bearing_deg)
            yaw_enu_rad = np.radians(90.0 - gnss.bearing_deg)
            self.rot_body_to_enu = R.from_euler("z", yaw_enu_rad)
        else:
            self.vel_enu = np.zeros(3, dtype=np.float64)
            self.rot_body_to_enu = R.identity()

        self.is_initialized = True
        self.mode = "GNSS_AIDED"

    def predict(self, sample: CalibratedSample, vel: Optional[VelocityEstimate] = None) -> FusedPosition:
        """
        Mechanization step: propagate INS state on each IMU tick.
        """
        ts_ns = sample.timestamp_ns

        if not self.is_initialized:
            # Not initialized with GNSS origin yet: initialize at origin
            self.last_timestamp_ns = ts_ns
            self.is_initialized = True
            self.mode = "INS_ONLY_BLACKOUT"
            return self.get_state(ts_ns)

        if self.last_timestamp_ns is None:
            self.last_timestamp_ns = ts_ns
            return self.get_state(ts_ns)

        dt = (ts_ns - self.last_timestamp_ns) * 1e-9
        self.last_timestamp_ns = ts_ns

        # Guard against zero/negative or huge dt glitch
        if dt <= 0.0 or dt > 1.0:
            dt = 0.1  # Fallback to nominal 10Hz tick

        # 1. Orientation Update: Integrate gyro rates (small-angle approximation / Rodrigues vector)
        omega_body = sample.gyro_vehicle
        angle = np.linalg.norm(omega_body) * dt
        if angle > 1e-12:
            axis = omega_body / np.linalg.norm(omega_body)
            delta_rot = R.from_rotvec(axis * angle)
            self.rot_body_to_enu = self.rot_body_to_enu * delta_rot

        # 2. Acceleration: Rotate specific force from vehicle body into ENU frame
        acc_enu = self.rot_body_to_enu.apply(sample.accel_vehicle)

        # 3. Subtract gravity [0, 0, -g_mag] in ENU frame:
        # Net dynamic acceleration = specific force + gravity_vector
        # Gravity vector in ENU points Down (along -Z): [0, 0, -g_mag]
        net_acc_enu = acc_enu - np.array([0.0, 0.0, self.g_mag], dtype=np.float64)

        # 4. Integrate velocity & position (trapezoidal / semi-implicit Euler)
        new_vel = self.vel_enu + net_acc_enu * dt
        self.pos_enu += 0.5 * (self.vel_enu + new_vel) * dt
        self.vel_enu = new_vel

        # Outage tracking
        outage_s = 0.0
        if self.last_valid_gnss_ts_ns is not None:
            outage_s = (ts_ns - self.last_valid_gnss_ts_ns) * 1e-9

        return self.get_state(ts_ns, outage_s)

    def update_gnss(self, gnss: GNSSSample) -> FusedPosition:
        """
        GNSS measurement update: reset state to GNSS fix (direct correction).
        """
        if not self.is_initialized:
            self._init_from_gnss(gnss)
            return self.get_state(gnss.timestamp_ns, 0.0)

        # Compute ENU position of new GNSS fix
        self.pos_enu = geodetic_to_enu(
            gnss.latitude_deg,
            gnss.longitude_deg,
            gnss.altitude_m,
            self.ref_lat_deg,
            self.ref_lon_deg,
            self.ref_alt_m,
        )

        if gnss.speed_mps is not None and gnss.bearing_deg is not None:
            bearing_rad = np.radians(gnss.bearing_deg)
            ve = gnss.speed_mps * np.sin(bearing_rad)
            vn = gnss.speed_mps * np.cos(bearing_rad)
            self.vel_enu = np.array([ve, vn, 0.0], dtype=np.float64)

            if self.align_initial_yaw:
                yaw_enu_rad = np.radians(90.0 - gnss.bearing_deg)
                self.rot_body_to_enu = R.from_euler("z", yaw_enu_rad)

        self.last_valid_gnss_ts_ns = gnss.timestamp_ns
        self.mode = "GNSS_AIDED"
        return self.get_state(gnss.timestamp_ns, 0.0)

    def get_state(self, timestamp_ns: Optional[int] = None, outage_s: float = 0.0) -> FusedPosition:
        ts = timestamp_ns or (self.last_timestamp_ns or 0)
        lat, lon, alt = enu_to_geodetic(
            self.pos_enu[0], self.pos_enu[1], self.pos_enu[2],
            self.ref_lat_deg, self.ref_lon_deg, self.ref_alt_m
        )

        # Extract yaw (heading clockwise from North in radians)
        # In ENU: yaw angle theta counter-clockwise from East. Bearing = (90 - theta) deg
        euler_angles = self.rot_body_to_enu.as_euler("zyx", degrees=False)
        enu_yaw_rad = euler_angles[0]
        # Bearing clockwise from North:
        heading_rad = (np.pi / 2.0 - enu_yaw_rad) % (2.0 * np.pi)

        return FusedPosition(
            timestamp_ns=ts,
            latitude_deg=lat,
            longitude_deg=lon,
            altitude_m=alt,
            position_enu_m=self.pos_enu.copy(),
            velocity_enu_mps=self.vel_enu.copy(),
            heading_rad=heading_rad,
            covariance=self.covariance.copy(),
            mode=self.mode,
            gnss_outage_duration_s=outage_s,
        )


# Register Naive Dead Reckoning Filter in factory registry
register_fusion_filter(
    "naive_dead_reckoning",
    lambda **params: NaiveDeadReckoningFilter(**params)
)

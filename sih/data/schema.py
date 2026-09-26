"""
Schema definitions and column mapping for flexible IMU and GNSS log ingestion.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import re
try:
    import pandas as pd
except ImportError:
    pd = None


@dataclass
class ColumnMapping:
    """
    Mapping rules for resolving arbitrary CSV column headers to standardized sensor fields.
    Each field contains candidate header names or regex patterns.
    """
    # Time
    timestamp: List[str] = field(default_factory=lambda: [
        "timestamp", "time", "time_ms", "time_s", "timestamp_ns", "date", "sample_period",
        r"time.*start.*", r"t"
    ])
    time_unit: str = "auto"  # 'auto', 's', 'ms', 'us', 'ns', 'iso'

    # Accelerometer
    accel_x: List[str] = field(default_factory=lambda: [
        r".*accel.*[_\s]x\b.*", r".*accel.*x\b.*", r".*ax[_\s].*", r".*ax\b.*", "x-acceleration", "acc_x", "acceleration_x"
    ])
    accel_y: List[str] = field(default_factory=lambda: [
        r".*accel.*[_\s]y\b.*", r".*accel.*y\b.*", r".*ay[_\s].*", r".*ay\b.*", "y-acceleration", "acc_y", "acceleration_y"
    ])
    accel_z: List[str] = field(default_factory=lambda: [
        r".*accel.*[_\s]z\b.*", r".*accel.*z\b.*", r".*az[_\s].*", r".*az\b.*", "z-acceleration", "acc_z", "acceleration_z"
    ])
    accel_unit: str = "m/s2"  # 'm/s2', 'g'

    # Gyroscope
    gyro_x: List[str] = field(default_factory=lambda: [
        r".*gyro.*[_\s]x\b.*", r".*gyro.*x\b.*", r".*wx\b", r".*gyro.*roll.*", r".*roll.*rate.*", "gyro_x", "angular_velocity_x"
    ])
    gyro_y: List[str] = field(default_factory=lambda: [
        r".*gyro.*[_\s]y\b.*", r".*gyro.*y\b.*", r".*wy\b", r".*gyro.*pitch.*", r".*pitch.*rate.*", "gyro_y", "angular_velocity_y"
    ])
    gyro_z: List[str] = field(default_factory=lambda: [
        r".*gyro.*[_\s]z\b.*", r".*gyro.*z\b.*", r".*wz\b", r".*gyro.*yaw.*", r".*yaw.*rate.*", "gyro_z", "angular_velocity_z"
    ])
    gyro_unit: str = "rad/s"  # 'rad/s', 'deg/s', 'rpm'

    # Magnetometer (optional)
    mag_x: List[str] = field(default_factory=lambda: [r".*mag.*x.*", "mag_x", "magnetic_field_x"])
    mag_y: List[str] = field(default_factory=lambda: [r".*mag.*y.*", "mag_y", "magnetic_field_y"])
    mag_z: List[str] = field(default_factory=lambda: [r".*mag.*z.*", "mag_z", "magnetic_field_z"])

    # GNSS / GPS
    gps_lat: List[str] = field(default_factory=lambda: [
        r".*lat.*", "latitude", "gps_latitude", "gps latitude", "lat"
    ])
    gps_lon: List[str] = field(default_factory=lambda: [
        r".*lon.*", "longitude", "gps_longitude", "gps longitude", "lng", "lon"
    ])
    gps_alt: List[str] = field(default_factory=lambda: [
        r".*alt.*", r".*height.*", "altitude", "gps_height", "gps height", "gps_altitude"
    ])
    gps_alt_unit: str = "m"  # 'm', 'km', 'ft'

    gps_speed: List[str] = field(default_factory=lambda: [
        r".*speed.*", r".*velocity.*", "gps_speed", "gps_velocity", "gps velocity"
    ])
    gps_speed_unit: str = "auto"  # 'auto', 'm/s', 'km/h', 'mph'

    gps_bearing: List[str] = field(default_factory=lambda: [
        r".*bearing.*", r".*heading.*", r".*course.*", "gps_heading", "gps heading", "orientation"
    ])
    gps_bearing_unit: str = "deg"  # 'deg', 'rad'

    gps_accuracy: List[str] = field(default_factory=lambda: [
        r".*acc.*", r".*accuracy.*", "gps_accuracy", "horizontal_accuracy", "h_accuracy"
    ])


def find_column(df_columns: List[str], candidates: List[str]) -> Optional[str]:
    """Find matching column name from dataframe columns using exact or regex matching."""
    # 1. Exact case-insensitive match
    for cand in candidates:
        cand_lower = cand.lower().strip()
        for col in df_columns:
            if col.lower().strip() == cand_lower:
                return col

    # 2. Regex match
    for cand in candidates:
        pattern = re.compile(cand, re.IGNORECASE)
        for col in df_columns:
            if pattern.fullmatch(col.strip()) or pattern.match(col.strip()):
                return col

    return None

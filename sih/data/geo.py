"""
Geodetic coordinate transformations, WGS-84 to Local Tangent Plane (ENU) conversions,
and geodesic distance calculations.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Union


# WGS-84 Ellipsoid Constants
WGS84_A = 6378137.0          # Semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563 # Flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F) # Semi-minor axis (meters)
WGS84_E2 = 2.0 * WGS84_F - WGS84_F ** 2 # First eccentricity squared


def geodetic_to_ecef(lat_deg: Union[float, np.ndarray],
                     lon_deg: Union[float, np.ndarray],
                     alt_m: Union[float, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geodetic coordinates (WGS-84 lat, lon, alt) to Earth-Centered Earth-Fixed (ECEF) x, y, z."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Radius of curvature in the prime vertical
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * (sin_lat ** 2))

    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat

    return x, y, z


def ecef_to_enu(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> np.ndarray:
    """Convert ECEF coordinates to East-North-Up (ENU) coordinates relative to a reference geodetic point."""
    ref_x, ref_y, ref_z = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)

    dx = x - ref_x
    dy = y - ref_y
    dz = z - ref_z

    ref_lat_rad = np.radians(ref_lat_deg)
    ref_lon_rad = np.radians(ref_lon_deg)

    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    if np.isscalar(e):
        return np.array([e, n, u], dtype=np.float64)
    return np.stack([e, n, u], axis=-1)


def geodetic_to_enu(lat_deg: Union[float, np.ndarray],
                    lon_deg: Union[float, np.ndarray],
                    alt_m: Union[float, np.ndarray],
                    ref_lat_deg: float,
                    ref_lon_deg: float,
                    ref_alt_m: float) -> np.ndarray:
    """Direct conversion from Geodetic (Lat, Lon, Alt) to Local Tangent Plane ENU coordinates."""
    x, y, z = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(x, y, z, ref_lat_deg, ref_lon_deg, ref_alt_m)


def enu_to_geodetic(e: float, n: float, u: float,
                    ref_lat_deg: float, ref_lon_deg: float, ref_alt_m: float) -> Tuple[float, float, float]:
    """Convert Local Tangent Plane ENU coordinates back to Geodetic (Lat, Lon, Alt)."""
    ref_lat_rad = np.radians(ref_lat_deg)
    ref_lon_rad = np.radians(ref_lon_deg)

    sin_lat = np.sin(ref_lat_rad)
    cos_lat = np.cos(ref_lat_rad)
    sin_lon = np.sin(ref_lon_rad)
    cos_lon = np.cos(ref_lon_rad)

    # Inverse rotation matrix (transpose)
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    ref_x, ref_y, ref_z = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    x = ref_x + dx
    y = ref_y + dy
    z = ref_z + dz

    # ECEF to Geodetic (Bowring's method)
    p = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    e2_prime = (WGS84_A ** 2 - WGS84_B ** 2) / (WGS84_B ** 2)

    lat = np.arctan2(z + e2_prime * WGS84_B * (np.sin(theta) ** 3),
                     p - WGS84_E2 * WGS84_A * (np.cos(theta) ** 3))
    lon = np.arctan2(y, x)

    n_cur = WGS84_A / np.sqrt(1.0 - WGS84_E2 * (np.sin(lat) ** 2))
    alt = p / np.cos(lat) - n_cur

    return float(np.degrees(lat)), float(np.degrees(lon)), float(alt)


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two geodetic coordinates in meters."""
    r = 6371000.0  # Earth's mean radius in meters
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r * c)


def compute_cumulative_distance(latitudes: np.ndarray, longitudes: np.ndarray) -> float:
    """Calculate the total path length traversed in meters."""
    if len(latitudes) < 2:
        return 0.0
    lat1 = latitudes[:-1]
    lon1 = longitudes[:-1]
    lat2 = latitudes[1:]
    lon2 = longitudes[1:]

    r = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * (np.sin(dlam / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(np.sum(r * c))

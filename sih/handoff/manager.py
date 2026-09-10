"""
sih.handoff.manager
-------------------
Production-grade 6-state Seamless GNSS <-> INS Handoff State Machine.
Implements IGNSSHandoffPolicy for robust blackout entry and zero-jump reacquisition.
"""

from __future__ import annotations
from dataclasses import dataclass, fields
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np

from sih.core.contracts import (
    GNSSSample,
    CalibratedSample,
    VelocityEstimate,
    FusedPosition,
)
from sih.core.interfaces import IGNSSHandoffPolicy
from sih.handoff.integrity import (
    compute_position_nis,
    check_kinematic_feasibility,
    evaluate_signal_quality,
)
from sih.handoff.reconciliation import HermiteReconciler
from sih.data.geo import geodetic_to_enu, enu_to_geodetic


class HandoffState(str, Enum):
    """Operational states of the GNSS-INS handoff finite state machine."""
    INITIALIZING = "INITIALIZING"
    GNSS_HEALTHY = "GNSS_HEALTHY"
    GNSS_DEGRADED = "GNSS_DEGRADED"
    INS_DEAD_RECKONING = "INS_DEAD_RECKONING"
    REACQUISITION_VERIFY = "REACQUISITION_VERIFY"
    REACQUISITION_BLENDING = "REACQUISITION_BLENDING"


@dataclass
class HandoffConfig:
    """Configuration parameters for the seamless handoff manager."""
    healthy_accuracy_max_m: float = 15.0       # Accuracy threshold for full healthy operation
    degraded_accuracy_max_m: float = 35.0      # Accuracy threshold before declaring total outage
    blackout_timeout_s: float = 1.5            # Seconds without acceptable fix before entering full DR
    reacquisition_fixes_required: int = 3      # Consecutive consistent fixes needed to verify exit
    max_reacq_nis: float = 15.0                # Innovation Chi-Square gate during reacquisition
    blend_duration_s: float = 1.2              # Duration of cubic Hermite zero-jump blend in seconds
    v_max_kinematic_mps: float = 45.0          # Max physical speed limit for kinematic validation (m/s)
    enable_zero_jump_blend: bool = True        # Enable UI smoothstep reconciliation


class SeamlessGNSSHandoffManager(IGNSSHandoffPolicy):
    """
    Seamless GNSS <-> INS Handoff Manager.

    Governs the transition into, through, and out of satellite blackouts:
    - Quarantines degraded fixes at tunnel entry, freezing scale/bias adaptation before multipath enters.
    - Tracks dead-reckoning trajectory and expands uncertainty during blackout.
    - Buffers and verifies N consecutive fixes at tunnel exit to reject portal multipath.
    - Smoothly reconciles display coordinates via C^2 Hermite spline, eliminating 15m-50m UI jumps.
    """

    def __init__(
        self,
        config: Optional[HandoffConfig] = None,
        **kwargs,
    ) -> None:
        if config is None:
            # Map common alias names
            mapped = dict(kwargs)
            if "nis_gate" in mapped:
                mapped["max_reacq_nis"] = mapped.pop("nis_gate")
            if "reacq_required_consecutive" in mapped:
                mapped["reacquisition_fixes_required"] = mapped.pop("reacq_required_consecutive")
            if "max_velocity_mps" in mapped:
                mapped["v_max_kinematic_mps"] = mapped.pop("max_velocity_mps")

            # Filter to valid HandoffConfig fields
            valid_fields = {f.name for f in fields(HandoffConfig)}
            filtered = {k: v for k, v in mapped.items() if k in valid_fields}
            self.config = HandoffConfig(**filtered)
        else:
            self.config = config

        self.reconciler = HermiteReconciler(blend_duration_s=self.config.blend_duration_s)
        self.reset()

    def reset(self, initial_gnss: Optional[GNSSSample] = None) -> None:
        """Resets the state machine and reconciler to default initial state."""
        self._state: HandoffState = HandoffState.INITIALIZING
        self._last_valid_fix_ts_ns: Optional[int] = None
        self._last_verified_p_enu: Optional[np.ndarray] = None
        self._p_dr_exit_snapshot: Optional[np.ndarray] = None
        self._verify_buffer: List[Tuple[int, np.ndarray, GNSSSample]] = []
        self._latest_p_enu: np.ndarray = np.zeros(3, dtype=np.float64)
        if initial_gnss is not None and initial_gnss.is_valid:
            self._ref_lat_deg = initial_gnss.latitude_deg
            self._ref_lon_deg = initial_gnss.longitude_deg
            self._ref_alt_m = initial_gnss.altitude_m
        else:
            self._ref_lat_deg = None
            self._ref_lon_deg = None
            self._ref_alt_m = 0.0
        self.reconciler.reset()

    def set_reference_origin(self, lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> None:
        """Sets geodetic origin for ENU conversions."""
        self._ref_lat_deg = lat_deg
        self._ref_lon_deg = lon_deg
        self._ref_alt_m = alt_m

    def get_mode(self) -> str:
        """Returns current operational state as a string contract."""
        return self._state.value

    @property
    def state(self) -> HandoffState:
        """Returns current operational state enum."""
        return self._state

    @property
    def freeze_parameters(self) -> bool:
        """Indicates whether parameter adaptation is frozen."""
        return self.should_freeze_adaptation()

    @property
    def quarantine_updates(self) -> bool:
        """Indicates whether incoming GNSS fixes are quarantined from EKF update."""
        return self._state in (
            HandoffState.GNSS_DEGRADED,
            HandoffState.INS_DEAD_RECKONING,
            HandoffState.REACQUISITION_VERIFY,
        )

    def should_freeze_adaptation(self) -> bool:
        """
        Indicates whether the EKF should freeze online parameter adaptation
        (speed scale s_v and gyro bias b_g learning).
        """
        return self._state in (
            HandoffState.GNSS_DEGRADED,
            HandoffState.INS_DEAD_RECKONING,
            HandoffState.REACQUISITION_VERIFY,
        )

    def get_covariance_inflation_factor(self) -> float:
        """
        Returns measurement noise covariance multiplier to adapt Kalman gain
        during degradation or initial reacquisition blending.
        """
        if self._state == HandoffState.GNSS_DEGRADED:
            return 4.0
        elif self._state == HandoffState.REACQUISITION_BLENDING:
            return 2.0
        return 1.0

    def notify_imu_step(
        self,
        timestamp_ns: int,
        p_current_enu: np.ndarray,
        dt_s: float = 0.1,
    ) -> None:
        """
        Notifies the handoff manager of an IMU prediction timestep.
        Enables time tracking and blackout timeout detection when zero GNSS fixes arrive.

        Parameters
        ----------
        timestamp_ns : int
            Current IMU sample timestamp in nanoseconds.
        p_current_enu : np.ndarray
            Current dead-reckoning position from EKF (ENU, meters).
        dt_s : float
            Timestep interval in seconds.
        """
        self._latest_p_enu = p_current_enu.copy().astype(np.float64)

        # Check for blackout timeout when satellite fixes stop arriving entirely
        if self._last_valid_fix_ts_ns is not None:
            elapsed_s = (timestamp_ns - self._last_valid_fix_ts_ns) * 1e-9

            if self._state in (HandoffState.GNSS_HEALTHY, HandoffState.GNSS_DEGRADED):
                if elapsed_s > self.config.blackout_timeout_s:
                    # Transition into full Dead Reckoning
                    self._state = HandoffState.INS_DEAD_RECKONING
                    self._p_dr_exit_snapshot = self._latest_p_enu.copy()

        # Update active blending transition
        if self._state == HandoffState.REACQUISITION_BLENDING:
            if not self.reconciler.is_blending:
                self._state = HandoffState.GNSS_HEALTHY

    def evaluate_gnss(
        self,
        gnss: GNSSSample,
        current_state: Optional[FusedPosition] = None,
    ) -> bool:
        """
        Evaluates incoming GNSS fix quality against state machine criteria.

        Parameters
        ----------
        gnss : GNSSSample
            Incoming GNSS fix.
        current_state : Optional[FusedPosition]
            Current estimated filter state.

        Returns
        -------
        bool
            True if fix is statistically accepted for Kalman measurement update.
        """
        if not evaluate_signal_quality(gnss, max_accuracy_h_m=self.config.degraded_accuracy_max_m):
            self._handle_unacceptable_fix(gnss.timestamp_ns)
            return False

        # Lazy initialize geodetic origin from first valid fix
        if self._ref_lat_deg is None:
            self.set_reference_origin(gnss.latitude_deg, gnss.longitude_deg, gnss.altitude_m)

        gnss_enu = geodetic_to_enu(
            gnss.latitude_deg, gnss.longitude_deg, gnss.altitude_m,
            self._ref_lat_deg, self._ref_lon_deg, self._ref_alt_m
        )

        acc = float(gnss.accuracy_h_m) if gnss.accuracy_h_m is not None else 10.0

        # --- State 0: INITIALIZING ---
        if self._state == HandoffState.INITIALIZING:
            if acc <= self.config.healthy_accuracy_max_m:
                self._state = HandoffState.GNSS_HEALTHY
            else:
                self._state = HandoffState.GNSS_DEGRADED
            self._last_valid_fix_ts_ns = gnss.timestamp_ns
            self._last_verified_p_enu = gnss_enu.copy()
            return True

        # --- State 1: GNSS_HEALTHY ---
        if self._state == HandoffState.GNSS_HEALTHY:
            if acc > self.config.healthy_accuracy_max_m:
                # Degraded accuracy: freeze parameter learning immediately
                self._state = HandoffState.GNSS_DEGRADED
                self._last_valid_fix_ts_ns = gnss.timestamp_ns
                self._last_verified_p_enu = gnss_enu.copy()
                return True

            # Kinematic check against previous verified fix
            if self._last_verified_p_enu is not None and self._last_valid_fix_ts_ns is not None:
                dt_fix = (gnss.timestamp_ns - self._last_valid_fix_ts_ns) * 1e-9
                if not check_kinematic_feasibility(gnss_enu, self._last_verified_p_enu, dt_fix, self.config.v_max_kinematic_mps):
                    # Spurious multipath spike: reject and transition to degraded
                    self._state = HandoffState.GNSS_DEGRADED
                    return False

            self._last_valid_fix_ts_ns = gnss.timestamp_ns
            self._last_verified_p_enu = gnss_enu.copy()
            return True

        # --- State 2: GNSS_DEGRADED ---
        if self._state == HandoffState.GNSS_DEGRADED:
            if acc <= self.config.healthy_accuracy_max_m:
                self._state = HandoffState.GNSS_HEALTHY
                self._last_valid_fix_ts_ns = gnss.timestamp_ns
                self._last_verified_p_enu = gnss_enu.copy()
                return True

            # Check if degradation has timed out into a complete blackout
            if self._last_valid_fix_ts_ns is not None:
                elapsed_s = (gnss.timestamp_ns - self._last_valid_fix_ts_ns) * 1e-9
                if elapsed_s > self.config.blackout_timeout_s:
                    self._state = HandoffState.INS_DEAD_RECKONING
                    self._p_dr_exit_snapshot = self._latest_p_enu.copy()
                    return False

            self._last_valid_fix_ts_ns = gnss.timestamp_ns
            self._last_verified_p_enu = gnss_enu.copy()
            return True

        # --- State 3: INS_DEAD_RECKONING ---
        if self._state == HandoffState.INS_DEAD_RECKONING:
            # First potential satellite reacquisition fix received!
            # Do NOT accept immediately; transition to verification buffer
            self._state = HandoffState.REACQUISITION_VERIFY
            self._verify_buffer = [(gnss.timestamp_ns, gnss_enu.copy(), gnss)]
            self._p_dr_exit_snapshot = self._latest_p_enu.copy()
            # Reject this first single fix from immediate EKF update to avoid portal multipath jump
            return False

        # --- State 4: REACQUISITION_VERIFY ---
        if self._state == HandoffState.REACQUISITION_VERIFY:
            # Check kinematic consistency against the previous buffered candidate
            prev_ts, prev_enu, _ = self._verify_buffer[-1]
            dt_step = (gnss.timestamp_ns - prev_ts) * 1e-9

            if not check_kinematic_feasibility(gnss_enu, prev_enu, dt_step, self.config.v_max_kinematic_mps):
                # Inconsistent jump: clear verification buffer and revert to DR
                self._verify_buffer = []
                self._state = HandoffState.INS_DEAD_RECKONING
                return False

            self._verify_buffer.append((gnss.timestamp_ns, gnss_enu.copy(), gnss))

            if len(self._verify_buffer) >= self.config.reacquisition_fixes_required:
                # Reacquisition verified! Initiate zero-jump trajectory blend
                self._last_valid_fix_ts_ns = gnss.timestamp_ns
                self._last_verified_p_enu = gnss_enu.copy()

                if self.config.enable_zero_jump_blend:
                    self.reconciler.initiate_blend(
                        timestamp_ns=gnss.timestamp_ns,
                        p_dead_reckoning=self._latest_p_enu.copy(),
                        p_fused=gnss_enu,
                    )
                    self._state = HandoffState.REACQUISITION_BLENDING
                else:
                    self._state = HandoffState.GNSS_HEALTHY

                self._verify_buffer = []
                return True
            else:
                # Still accumulating verification fixes
                return False

        # --- State 5: REACQUISITION_BLENDING ---
        if self._state == HandoffState.REACQUISITION_BLENDING:
            self._last_valid_fix_ts_ns = gnss.timestamp_ns
            self._last_verified_p_enu = gnss_enu.copy()

            if not self.reconciler.is_blending:
                self._state = HandoffState.GNSS_HEALTHY

            return True

        return False

    def _handle_unacceptable_fix(self, timestamp_ns: int) -> None:
        """Handles receipt of an invalid, NaN, or out-of-bounds accuracy fix."""
        if self._state == HandoffState.REACQUISITION_VERIFY:
            # Reset verification buffer on dropped fix
            self._verify_buffer = []
            self._state = HandoffState.INS_DEAD_RECKONING

        if self._last_valid_fix_ts_ns is not None:
            elapsed_s = (timestamp_ns - self._last_valid_fix_ts_ns) * 1e-9
            if elapsed_s > self.config.blackout_timeout_s and self._state != HandoffState.INS_DEAD_RECKONING:
                self._state = HandoffState.INS_DEAD_RECKONING
                self._p_dr_exit_snapshot = self._latest_p_enu.copy()

    def get_display_position(self, fused: FusedPosition) -> FusedPosition:
        """
        Reconciles the mathematical fused position with the zero-jump display spline.

        Parameters
        ----------
        fused : FusedPosition
            Mathematical state estimate from the EKF.

        Returns
        -------
        FusedPosition
            Reconciled position contract with zero visual jump.
        """
        if not self.reconciler.is_blending or self._ref_lat_deg is None:
            return fused

        p_fused = np.array(fused.position_enu_m, dtype=np.float64)
        p_display, alpha = self.reconciler.get_blended_position(fused.timestamp_ns, p_fused)

        lat, lon, alt = enu_to_geodetic(
            p_display[0], p_display[1], p_display[2],
            self._ref_lat_deg, self._ref_lon_deg, self._ref_alt_m
        )

        return FusedPosition(
            timestamp_ns=fused.timestamp_ns,
            latitude_deg=float(lat),
            longitude_deg=float(lon),
            altitude_m=float(alt),
            position_enu_m=np.asarray(p_display, dtype=np.float64),
            velocity_enu_mps=fused.velocity_enu_mps,
            heading_rad=fused.heading_rad,
            covariance=fused.covariance,
            mode="REACQUISITION_BLENDING" if self.reconciler.is_blending else fused.mode,
            gnss_outage_duration_s=fused.gnss_outage_duration_s,
        )

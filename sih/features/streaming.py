"""
Causal Edge-Streaming Feature Extraction Pipeline for Smartphone Dead Reckoning.

Guarantees:
1. Strict Causality: Zero future lookahead. All digital filtering (sosfilt with carried state)
   and spectral energy windowing use only past and current sensor samples.
2. Identical Streaming & Batch Semantics: batch_extract(samples) is a direct loop over push(),
   guaranteeing bit-identity across edge streaming and offline training.
3. 12 Physical Channels matching Bayesian Dual-Expert MoE velocity estimators:
   - Ch 0-2: Causal low-pass filtered linear acceleration [ax, ay, az] (m/s^2)
   - Ch 3-5: Causal low-pass filtered angular rate [gx, gy, gz] (rad/s)
   - Ch 6: Acceleration Euclidean magnitude |a| (m/s^2)
   - Ch 7: Gyroscope Euclidean magnitude |w| (rad/s)
   - Ch 8: Band A macroscopic chassis dynamics energy (0.1 - 1.5 Hz)
   - Ch 9: Band B tyre/road acoustic interaction energy (1.5 - 4.5 Hz)
   - Ch 10: Spectral energy ratio E_ratio = E_bandB / (E_bandA + E_bandB + eps)
   - Ch 11: Kinetic speed proxy v_proxy = clip(E_bandB / (E_bandA + eps), 0, 10)
"""

from __future__ import annotations
from typing import Optional, List, Tuple, Any
import os
import hashlib
from collections import deque
import numpy as np
if os.environ.get("SIH_FORCE_SCIPY_SHIM", "0") == "1":
    from sih.core import scipy_shim as signal
else:
    try:
        from scipy import signal
    except ImportError:
        from sih.core import scipy_shim as signal

from sih.core.contracts import CalibratedSample


class StreamingFeatureExtractor:
    """
    Stateful, streaming causal feature extractor for smartphone IMU navigation.
    Maintains internal filter delays and rolling window buffers to emit 12-channel
    features causally on every IMU tick.
    """

    def __init__(
        self,
        sampling_rate: float = 10.0,
        cutoff_hz: float = 3.5,
        max_jerk_mps3: float = 15.0,
        window_len: int = 60,
        spectral_stride: int = 5,
        band_a: Tuple[float, float] = (0.1, 1.5),
        band_b: Tuple[float, float] = (1.5, 4.5),
        allow_unwarmed: bool = True,
        eps: float = 1e-6,
    ) -> None:
        self.fs = float(sampling_rate)
        self.cutoff_hz = float(cutoff_hz)
        self.max_jerk_mps3 = float(max_jerk_mps3)
        self.dt = 1.0 / self.fs
        self.max_delta_a = self.max_jerk_mps3 * self.dt
        self.window_len = int(window_len)
        self.spectral_stride = int(spectral_stride)
        self.band_a = band_a
        self.band_b = band_b
        self.allow_unwarmed = bool(allow_unwarmed)
        self.eps = float(eps)

        # 2nd-order Butterworth low-pass filter in Second-Order Sections (SOS) form
        nyquist = 0.5 * self.fs
        norm_cutoff = min(self.cutoff_hz / nyquist, 0.95)
        self.sos = signal.butter(2, norm_cutoff, btype="low", output="sos")
        self.zi_base = signal.sosfilt_zi(self.sos)  # (n_sections, 2)

        self.reset()

    def reset(self) -> None:
        """Reset internal filter state and circular buffers."""
        self._zi_accel: Optional[np.ndarray] = None  # (n_sections, 2, 3)
        self._zi_gyro: Optional[np.ndarray] = None   # (n_sections, 2, 3)
        self._prev_filtered_accel: Optional[np.ndarray] = None  # (3,)
        self._accel_buf: deque = deque(maxlen=self.window_len)
        self._sample_count: int = 0
        self._last_spectral: Optional[np.ndarray] = None  # (4,)

    @property
    def is_warm(self) -> bool:
        """True when the rolling spectral buffer is fully populated."""
        return len(self._accel_buf) >= self.window_len

    def push(self, sample: CalibratedSample) -> Optional[np.ndarray]:
        """
        Ingest a single CalibratedSample and return the 12-channel feature vector.
        Zero future lookahead: strictly depends only on current and past samples.

        Returns
        -------
        feat_12 : np.ndarray of shape (12,) and dtype float32, or None if not warm
                  and allow_unwarmed=False.
        """
        raw_accel = np.asarray(sample.accel_vehicle, dtype=np.float64)
        raw_gyro = np.asarray(sample.gyro_vehicle, dtype=np.float64)

        # 1. Causal IIR Low-Pass Filtering via carried SOS state
        if self._zi_accel is None:
            # Initialize filter steady-state from the first sample
            self._zi_accel = self.zi_base[:, :, None] * raw_accel[None, None, :]
            self._zi_gyro = self.zi_base[:, :, None] * raw_gyro[None, None, :]

        y_acc, self._zi_accel = signal.sosfilt(self.sos, raw_accel[None, :], zi=self._zi_accel, axis=0)
        y_gyr, self._zi_gyro = signal.sosfilt(self.sos, raw_gyro[None, :], zi=self._zi_gyro, axis=0)
        f_accel = y_acc[0]
        f_gyro = y_gyr[0]

        # 2. Causal Physical Jerk Clamping
        if self._prev_filtered_accel is not None:
            delta = f_accel - self._prev_filtered_accel
            delta_clamped = np.clip(delta, -self.max_delta_a, self.max_delta_a)
            f_accel = self._prev_filtered_accel + delta_clamped
        self._prev_filtered_accel = f_accel.copy()

        # 3. Euclidean Norms
        norm_a = float(np.linalg.norm(f_accel))
        norm_w = float(np.linalg.norm(f_gyro))

        # 4. Update Rolling Acceleration Buffer for Spectral Estimation
        self._accel_buf.append(f_accel.copy())

        # 5. Causal Spectral Feature Extraction with Zero-Order Hold (ZOH)
        # Evaluates every spectral_stride steps and holds value; never interpolates forward.
        if len(self._accel_buf) < 8:
            spectral_feats = np.zeros(4, dtype=np.float32)
            self._last_spectral = spectral_feats
        else:
            if self._sample_count % self.spectral_stride == 0 or self._last_spectral is None:
                spectral_feats = self._compute_spectral_window(self._accel_buf)
                self._last_spectral = spectral_feats
            else:
                spectral_feats = self._last_spectral

        self._sample_count += 1

        if not self.allow_unwarmed and not self.is_warm:
            return None

        # Assemble 12-channel vector
        feat_12 = np.empty(12, dtype=np.float32)
        feat_12[0:3] = f_accel
        feat_12[3:6] = f_gyro
        feat_12[6] = norm_a
        feat_12[7] = norm_w
        feat_12[8:12] = spectral_feats
        return feat_12

    def _compute_spectral_window(self, buf: deque) -> np.ndarray:
        """Compute 4-channel spectral energy vector over current trailing buffer."""
        accels = np.array(buf, dtype=np.float64)  # (L, 3)
        a_mag = np.linalg.norm(accels, axis=1)    # (L,)
        a_detrend = a_mag - np.mean(a_mag)

        L = len(a_detrend)
        if L < 8 or np.all(np.abs(a_detrend) < 1e-6):
            return np.zeros(4, dtype=np.float32)

        nperseg = min(L, 32)
        freqs, psd = signal.welch(a_detrend, fs=self.fs, nperseg=nperseg, noverlap=nperseg // 2)

        mask_a = (freqs >= self.band_a[0]) & (freqs <= self.band_a[1])
        mask_b = (freqs >= self.band_b[0]) & (freqs <= self.band_b[1])

        trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
        if trapz_func is None:
            from scipy import integrate
            trapz_func = integrate.trapezoid

        e_a = float(trapz_func(psd[mask_a], freqs[mask_a])) if np.any(mask_a) else 0.0
        e_b = float(trapz_func(psd[mask_b], freqs[mask_b])) if np.any(mask_b) else 0.0

        e_a = max(0.0, e_a)
        e_b = max(0.0, e_b)

        e_ratio = float(e_b / (e_a + e_b + self.eps))
        v_proxy = float(np.clip(e_b / (e_a + self.eps), 0.0, 10.0))

        return np.array([e_a, e_b, e_ratio, v_proxy], dtype=np.float32)

    def batch_extract(
        self,
        samples: List[CalibratedSample],
        reset_first: bool = True,
    ) -> np.ndarray:
        """
        Extract features across a sequence by iterating over push().
        Guaranteed to be bit-identical to streaming ingestion.

        Returns
        -------
        features : np.ndarray of shape (N, 12) and dtype float32.
        """
        if reset_first:
            self.reset()

        N = len(samples)
        out = np.empty((N, 12), dtype=np.float32)
        for i, s in enumerate(samples):
            f = self.push(s)
            if f is None:
                out[i] = 0.0
            else:
                out[i] = f
        return out


def get_causal_feature_cache_path(
    trip_id: str,
    in_channels: int = 12,
    cache_dir: str = "data/cache",
) -> str:
    """Returns unique causal feature cache filepath with pipeline hash."""
    pipeline_signature = f"v1_sosfilt_cutoff3.5_jerk15_win60_stride5_bands0.1_1.5_4.5_ch{in_channels}"
    import hashlib
    h = hashlib.sha256(pipeline_signature.encode()).hexdigest()[:8]
    return os.path.join(cache_dir, f"{trip_id}_features_{in_channels}ch_causal_{h}.npz")


def load_or_compute_causal_features(
    trip: Any,
    in_channels: int = 12,
    cache_dir: str = "data/cache",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads or computes causal 12-channel features using calibrate_stream and StreamingFeatureExtractor.
    Returns (feats, f_accel, f_gyro).
    """
    import os
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = get_causal_feature_cache_path(trip.trip_id, in_channels, cache_dir)
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        return cached["feats"], cached["f_accel"], cached["f_gyro"]

    from sih.calibration.mount import calibrate_stream
    calib_samples = calibrate_stream(trip, min_samples=30)
    extractor = StreamingFeatureExtractor(sampling_rate=10.0, window_len=60, spectral_stride=5)
    feats = extractor.batch_extract(calib_samples)
    f_accel = feats[:, :3]
    f_gyro = feats[:, 3:6]

    np.savez_compressed(cache_path, feats=feats, f_accel=f_accel, f_gyro=f_gyro)
    return feats, f_accel, f_gyro


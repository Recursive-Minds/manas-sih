"""Drift-aware temporal alignment between smartphone and reference (CAN/VBOX) streams.

The IO-VNBD "synchronised" trips are distributed as row-aligned CSV pairs, but the
smartphone and vehicle loggers use independent clocks. Cross-correlating a physical
quantity that BOTH devices observe (rotation rate about the gravity/down axis) shows
that the true offset can be several seconds, and can drift linearly within a trip:

    S-S1   +0.2 s  (constant)
    S-S2   +8.7 s  (constant)
    S-S3a  -6.7 s  (constant)
    S-M    +0.9 s -> +3.2 s  (linear drift)

Training on row-aligned data therefore asks the network to predict a speed label that
belongs to a different moment in time. The Bayes-optimal response to a randomly shifted
label is the LOCAL MEAN of the label, which is precisely the "flat / average speed"
pathology observed in later phases. This module removes that failure at the source.

Alignment signal
----------------
We correlate the gravity-referenced yaw rate omega_v = omega . g_hat (see
``engine.orientation``) against the reference yaw rate. omega_v is invariant to how the
phone is mounted (both vectors rotate together under a device rotation), so the estimator
works for an arbitrarily oriented phone -- unlike correlating a hard-coded gyro axis.

The offset is modelled as an affine function of time, lag(t) = a + b t, which captures
both a fixed logger offset and linear clock drift. Fitting is done on per-segment
cross-correlation peaks with a correlation-quality weight, so low-motion segments (where
the peak is meaningless) do not corrupt the fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "LagEstimate",
    "SyncResult",
    "normalized_xcorr_lag",
    "estimate_segment_lags",
    "fit_lag_model",
    "estimate_time_sync",
    "apply_lag_model",
]


@dataclass
class LagEstimate:
    """Cross-correlation peak for one segment."""

    center_index: float
    lag_samples: int
    correlation: float


@dataclass
class SyncResult:
    """Affine alignment model mapping phone index -> reference index.

    ``reference_index = phone_index + intercept + slope * phone_index``
    """

    intercept: float
    slope: float
    quality: float
    segments: Sequence[LagEstimate] = field(default_factory=tuple)
    drifting: bool = False

    def lag_at(self, phone_index: np.ndarray | float) -> np.ndarray:
        idx = np.asarray(phone_index, dtype=np.float64)
        return self.intercept + self.slope * idx

    def reference_index(self, phone_index: np.ndarray | float) -> np.ndarray:
        idx = np.asarray(phone_index, dtype=np.float64)
        return idx + self.lag_at(idx)

    def as_dict(self) -> dict:
        return {
            "intercept_samples": float(self.intercept),
            "slope_samples_per_sample": float(self.slope),
            "quality": float(self.quality),
            "drifting": bool(self.drifting),
            "n_segments": len(self.segments),
        }


def _zero_mean_unit(x: np.ndarray) -> Optional[np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    s = float(np.std(x))
    if not np.isfinite(s) or s < 1e-9:
        return None
    return x / s


def normalized_xcorr_lag(
    a: np.ndarray,
    b: np.ndarray,
    max_lag: int = 150,
    min_overlap: int = 400,
    min_correlation: float = 0.40,
    min_psr: float = 2.8,
) -> Tuple[int, float]:
    """Return the integer lag maximising the normalised correlation of ``a`` and ``b``.

    A positive lag ``L`` means ``b[i + L]`` corresponds to ``a[i]`` -- i.e. the reference
    stream ``b`` runs ``L`` samples AHEAD of the phone stream ``a``.

    Returns ``(0, -1.0)`` when neither signal carries enough excitation to localise a peak.
    """
    an = _zero_mean_unit(a)
    bn = _zero_mean_unit(b)
    if an is None or bn is None:
        return 0, -1.0

    n = min(len(an), len(bn))
    max_lag = int(min(max_lag, max(0, n - min_overlap)))
    if max_lag <= 0:
        return 0, -1.0

    lags = []
    corrs = []
    r_zero = 0.0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x, y = an[-lag:], bn[: len(bn) + lag]
        elif lag > 0:
            x, y = an[: len(an) - lag], bn[lag:]
        else:
            x, y = an, bn
        length = min(len(x), len(y))
        if length < min_overlap:
            continue
        x = x[:length]
        y = y[:length]
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        if denom < 1e-9:
            continue
        r = float(np.dot(x, y) / denom)
        lags.append(lag)
        corrs.append(r)
        if lag == 0:
            r_zero = r

    if not corrs:
        return 0, -1.0

    corrs_arr = np.array(corrs, dtype=np.float64)
    lags_arr = np.array(lags, dtype=np.int32)
    best_idx = int(np.argmax(corrs_arr))
    best_lag = int(lags_arr[best_idx])
    best_r = float(corrs_arr[best_idx])

    # 1. Absolute correlation confidence floor
    if best_r < min_correlation:
        return 0, best_r

    # 2. Zero-lag prior: if correlation at lag 0 is close to peak, retain zero lag
    if r_zero > 0.0 and (best_r - r_zero) < 0.12 and abs(best_lag) <= 6:
        return 0, r_zero

    # 3. Peak-to-Sidelobe Ratio (PSR) test:
    # Exclude the immediate neighborhood (+/- 5 samples) around best peak
    sidelobe_mask = np.abs(lags_arr - best_lag) > 5
    if np.sum(sidelobe_mask) >= 15:
        sidelobes = corrs_arr[sidelobe_mask]
        s_mean = float(np.mean(sidelobes))
        s_std = float(np.std(sidelobes))
        psr = (best_r - s_mean) / max(s_std, 1e-6)
        if psr < min_psr:
            # Correlation surface lacks a distinctive peak (e.g. random noise or flat harmonic)
            return (0, r_zero) if r_zero >= min_correlation else (0, -1.0)

    return best_lag, best_r


def estimate_segment_lags(
    phone_signal: np.ndarray,
    reference_signal: np.ndarray,
    n_segments: int = 8,
    max_lag: int = 150,
    min_overlap: int = 400,
    min_correlation: float = 0.40,
) -> list[LagEstimate]:
    """Cross-correlate ``n_segments`` equal slices to expose drift in the offset."""
    n = int(min(len(phone_signal), len(reference_signal)))
    if n < min_overlap:
        return []

    n_segments = max(1, int(n_segments))
    seg_len = n // n_segments
    if seg_len < min_overlap + max_lag:
        n_segments = max(1, n // (min_overlap + max_lag))
        seg_len = n // n_segments

    out: list[LagEstimate] = []
    for k in range(n_segments):
        start = k * seg_len
        end = n if k == n_segments - 1 else (k + 1) * seg_len
        lag, r = normalized_xcorr_lag(
            phone_signal[start:end],
            reference_signal[start:end],
            max_lag=max_lag,
            min_overlap=min_overlap,
            min_correlation=min_correlation,
        )
        if r > 0.0:
            out.append(LagEstimate(center_index=0.5 * (start + end), lag_samples=lag, correlation=r))
    return out


def fit_lag_model(
    segments: Sequence[LagEstimate],
    n_samples: int,
    min_correlation: float = 0.40,
    drift_tolerance_samples: float = 5.0,
    max_lag_std: float = 8.0,
) -> SyncResult:
    """Fit a correlation-weighted affine lag model to per-segment peaks.

    Segments below ``min_correlation`` are discarded (typically stationary stretches where
    the yaw-rate signal carries no information). A drift term is only accepted when it
    explains more than ``drift_tolerance_samples`` of movement across the trip, so that a
    genuinely constant offset is not turned into a spurious ramp by peak noise.
    """
    usable = [s for s in segments if s.correlation >= min_correlation]
    if not usable:
        return SyncResult(intercept=0.0, slope=0.0, quality=0.0, segments=tuple(segments), drifting=False)

    x = np.array([s.center_index for s in usable], dtype=np.float64)
    y = np.array([float(s.lag_samples) for s in usable], dtype=np.float64)
    w = np.array([s.correlation for s in usable], dtype=np.float64)
    quality = float(np.average(w))

    if len(usable) < 3 or np.ptp(x) < 1e-6:
        if len(usable) >= 2 and float(np.std(y)) > max_lag_std:
            return SyncResult(intercept=0.0, slope=0.0, quality=0.0, segments=tuple(segments), drifting=False)
        avg_lag = float(np.average(y, weights=w))
        if abs(avg_lag) < 2.0:
            avg_lag = 0.0
        return SyncResult(
            intercept=avg_lag,
            slope=0.0,
            quality=quality,
            segments=tuple(segments),
            drifting=False,
        )

    # Weighted least squares on [1, x]
    design = np.column_stack([np.ones_like(x), x])
    wsqrt = np.sqrt(w)
    coeffs, *_ = np.linalg.lstsq(design * wsqrt[:, None], y * wsqrt, rcond=None)
    intercept, slope = float(coeffs[0]), float(coeffs[1])

    total_drift = abs(slope) * max(1, n_samples)
    y_fit = intercept + slope * x
    residual_std = float(np.std(y - y_fit))

    if total_drift < drift_tolerance_samples:
        if float(np.std(y)) > max_lag_std:
            return SyncResult(intercept=0.0, slope=0.0, quality=0.0, segments=tuple(segments), drifting=False)
        avg_lag = float(np.average(y, weights=w))
        if abs(avg_lag) < 2.0:
            avg_lag = 0.0
        return SyncResult(
            intercept=avg_lag,
            slope=0.0,
            quality=quality,
            segments=tuple(segments),
            drifting=False,
        )

    # Drift model consensus check: residuals around the linear drift line must be tight
    if residual_std > max_lag_std:
        return SyncResult(intercept=0.0, slope=0.0, quality=0.0, segments=tuple(segments), drifting=False)

    return SyncResult(
        intercept=intercept,
        slope=slope,
        quality=quality,
        segments=tuple(segments),
        drifting=True,
    )


def estimate_time_sync(
    phone_yaw_rate: np.ndarray,
    reference_yaw_rate: np.ndarray,
    n_segments: int = 8,
    max_lag: int = 150,
    min_overlap: int = 400,
    min_correlation: float = 0.40,
) -> SyncResult:
    """Estimate the affine phone->reference alignment from two yaw-rate streams.

    ``phone_yaw_rate`` should be the gravity-referenced yaw rate (mount invariant).
    Evaluates positive polarity first, falling back to inverted polarity only when
    positive correlation fails.
    """
    phone_yaw_rate = np.asarray(phone_yaw_rate, dtype=np.float64)
    reference_yaw_rate = np.asarray(reference_yaw_rate, dtype=np.float64)
    n = int(min(len(phone_yaw_rate), len(reference_yaw_rate)))
    phone_yaw_rate = phone_yaw_rate[:n]
    reference_yaw_rate = reference_yaw_rate[:n]

    # 1. Positive polarity evaluation (standard physical convention)
    pos_segments = estimate_segment_lags(
        phone_yaw_rate,
        reference_yaw_rate,
        n_segments=n_segments,
        max_lag=max_lag,
        min_overlap=min_overlap,
        min_correlation=min_correlation,
    )
    res_pos = fit_lag_model(pos_segments, n_samples=n, min_correlation=min_correlation)

    # If positive polarity yields coherent segments with decent quality, accept it directly
    if res_pos.quality >= min_correlation and any(s.correlation >= min_correlation for s in res_pos.segments):
        return res_pos

    # 2. Inverted polarity evaluation (only if positive polarity failed)
    neg_segments = estimate_segment_lags(
        phone_yaw_rate,
        -reference_yaw_rate,
        n_segments=n_segments,
        max_lag=max_lag,
        min_overlap=min_overlap,
        min_correlation=min_correlation,
    )
    res_neg = fit_lag_model(neg_segments, n_samples=n, min_correlation=min_correlation)

    if res_neg.quality > res_pos.quality and res_neg.quality >= min_correlation:
        return res_neg

    return res_pos if res_pos.quality > 0.0 else SyncResult(0.0, 0.0, 0.0, (), False)


def apply_lag_model(
    reference_series: np.ndarray,
    sync: SyncResult,
    n_phone: int,
    fill: str = "edge",
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a reference-timeline series onto the phone timeline.

    Args:
        reference_series: (M,) or (M, C) values on the reference clock.
        sync: fitted alignment model.
        n_phone: number of phone samples to produce.
        fill: ``"edge"`` clamps out-of-range requests to the first/last reference sample;
            any other value marks them invalid.

    Returns:
        ``(values, valid)`` where ``values`` has shape ``(n_phone, ...)`` and ``valid`` is a
        boolean mask marking samples whose source index fell inside the reference record.
    """
    ref = np.asarray(reference_series, dtype=np.float64)
    squeeze = ref.ndim == 1
    if squeeze:
        ref = ref[:, None]
    m = len(ref)

    phone_idx = np.arange(n_phone, dtype=np.float64)
    src = sync.reference_index(phone_idx)
    valid = (src >= 0.0) & (src <= m - 1)

    src_clipped = np.clip(src, 0.0, m - 1)
    grid = np.arange(m, dtype=np.float64)
    out = np.empty((n_phone, ref.shape[1]), dtype=np.float64)
    for c in range(ref.shape[1]):
        out[:, c] = np.interp(src_clipped, grid, ref[:, c])

    if fill != "edge":
        out[~valid] = np.nan

    if squeeze:
        out = out[:, 0]
    return out, valid

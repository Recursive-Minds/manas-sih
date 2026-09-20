package com.recursiveminds.idr.data

import com.google.gson.annotations.SerializedName

data class ImuPoint(
    @SerializedName("timestamp_ns") val timestampNs: Long,
    @SerializedName("accel") val accel: DoubleArray,
    @SerializedName("gyro") val gyro: DoubleArray
)

data class GnssPoint(
    @SerializedName("timestamp_ns") val timestampNs: Long,
    @SerializedName("latitude_deg") val latitudeDeg: Double,
    @SerializedName("longitude_deg") val longitudeDeg: Double,
    @SerializedName("altitude_m") val altitudeM: Double = 0.0,
    @SerializedName("speed_mps") val speedMps: Float? = null,
    @SerializedName("bearing_deg") val bearingDeg: Float? = null,
    @SerializedName("accuracy_h_m") val accuracyHM: Float = 5.0f,
    @SerializedName("is_valid") val isValid: Boolean = true
)

data class SensorBatch(
    @SerializedName("type") val type: String = "sensor_batch",
    @SerializedName("state") val state: String,
    @SerializedName("timestamp_ns") val timestampNs: Long,
    @SerializedName("imu") val imu: List<ImuPoint>,
    @SerializedName("gnss") val gnss: List<GnssPoint>
)

data class ControlMessage(
    @SerializedName("type") val type: String = "control",
    @SerializedName("command") val command: String
)

data class PositionPoint(
    @SerializedName("lat") val lat: Double,
    @SerializedName("lon") val lon: Double,
    @SerializedName("speed_mps") val speedMps: Double? = null,
    @SerializedName("heading_deg") val headingDeg: Double? = null,
    @SerializedName("bearing_deg") val bearingDeg: Double? = null
)

data class SessionSummary(
    @SerializedName("final_error_m") val finalErrorM: Double = 0.0,
    @SerializedName("drift_pct") val driftPct: Double = 0.0,
    @SerializedName("drift_label") val driftLabel: String = "Drift: 0.00% (target <10%)",
    @SerializedName("target_met") val targetMet: Boolean = true,
    @SerializedName("max_error_m") val maxErrorM: Double = 0.0,
    @SerializedName("duration_s") val durationS: Double = 0.0,
    @SerializedName("dr_dist_m") val drDistM: Double = 0.0,
    @SerializedName("gnss_dist_m") val gnssDistM: Double = 0.0,
    @SerializedName("along_track_m") val alongTrackM: Double = 0.0,
    @SerializedName("cross_track_m") val crossTrackM: Double = 0.0,
    @SerializedName("speed_regime") val speedRegime: String = "City (20-50 km/h)",
    @SerializedName("tier") val tier: String = "City (20-50 km/h)"
)

data class LiveMetrics(
    @SerializedName("elapsed_s") val elapsedS: Double = 0.0,
    @SerializedName("dr_dist_m") val drDistM: Double = 0.0,
    @SerializedName("gnss_dist_m") val gnssDistM: Double = 0.0,
    @SerializedName("horizontal_error_m") val horizontalErrorM: Double = 0.0,
    @SerializedName("along_track_m") val alongTrackM: Double = 0.0,
    @SerializedName("cross_track_m") val crossTrackM: Double = 0.0,
    @SerializedName("drift_pct") val driftPct: Double = 0.0,
    @SerializedName("speed_regime") val speedRegime: String = "City (20-50 km/h)",
    @SerializedName("tier") val tier: String = "City (20-50 km/h)",
    @SerializedName("dr_speed_mps") val drSpeedMps: Double = 0.0,
    @SerializedName("gnss_speed_mps") val gnssSpeedMps: Double = 0.0,
    @SerializedName("dr_heading_deg") val drHeadingDeg: Double = 0.0,
    @SerializedName("gnss_bearing_deg") val gnssBearingDeg: Double = 0.0,
    @SerializedName("max_horizontal_error_m") val maxHorizontalErrorM: Double = 0.0,
    @SerializedName("gnss_accuracy_h_m") val gnssAccuracyHM: Double = 5.0,
    @SerializedName("is_blackout") val isBlackout: Boolean = false,
    @SerializedName("session_summary") val sessionSummary: SessionSummary? = null
)

data class WarmupStatus(
    @SerializedName("is_ready") val isReady: Boolean = false,
    @SerializedName("gravity_converged") val gravityConverged: Boolean = false,
    @SerializedName("mount_locked") val mountLocked: Boolean = false,
    @SerializedName("mount_status") val mountStatus: String = "Mount: Initializing",
    @SerializedName("turn_events") val turnEvents: Int = 0,
    @SerializedName("turn_events_target") val turnEventsTarget: Int = 8,
    @SerializedName("turns_display") val turnsDisplay: String = "turns: 0/8",
    @SerializedName("buffer_warm") val bufferWarm: Boolean = false,
    @SerializedName("alpha_learned") val alphaLearned: Boolean = false
)

data class HudUpdate(
    @SerializedName("type") val type: String,
    @SerializedName("state") val state: String,
    @SerializedName("mount_status") val mountStatus: String,
    @SerializedName("warmup") val warmup: WarmupStatus?,
    @SerializedName("dr_pos") val drPos: PositionPoint?,
    @SerializedName("gnss_pos") val gnssPos: PositionPoint?,
    @SerializedName("metrics") val metrics: LiveMetrics?
)

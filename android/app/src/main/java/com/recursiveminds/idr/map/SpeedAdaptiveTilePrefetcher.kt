package com.recursiveminds.idr.map

import android.content.Context
import android.util.Log
import org.osmdroid.tileprovider.cachemanager.CacheManager
import org.osmdroid.util.BoundingBox
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

/**
 * SpeedAdaptiveTilePrefetcher
 *
 * Dynamically prefetches and caches OpenStreetMap vector/raster tiles into local SQLite storage
 * ahead of the vehicle based on current velocity and heading while network/internet is active.
 *
 * This ensures that when the vehicle enters GNSS / cellular blackouts (tunnels, underground parking,
 * flyovers, or rural transit dead zones), all high-resolution tiles (zooms 15..17) are already
 * saved locally on device and display seamlessly without any latency or blank tiles.
 */
class SpeedAdaptiveTilePrefetcher(
    private val context: Context,
    private val mapView: MapView
) {
    companion object {
        private const val TAG = "TilePrefetcher"
        private const val METERS_PER_DEG_LAT = 111320.0
        private const val MIN_TRIGGER_DIST_M = 180.0
        private const val MIN_TRIGGER_TIME_MS = 20_000L // 20s
        private const val MAX_PENDING_JOBS = 5
        private const val MAX_TILES_PER_BATCH = 350
    }

    private var cacheManager: CacheManager? = null
    private var lastPrefetchPoint: GeoPoint? = null
    private var lastPrefetchTimeMs: Long = 0L
    private var isTaskRunning = false
    private var hasPrefetchedInitialRegion = false

    init {
        try {
            cacheManager = CacheManager(mapView)
            Log.i(TAG, "CacheManager initialized successfully. Capacity: ${cacheManager?.cacheCapacity()} bytes")
        } catch (e: Exception) {
            Log.w(TAG, "Could not initialize CacheManager: ${e.message}")
        }
    }

    /**
     * Prefetch an immediate initial bubble around starting location (e.g. at warmup or initial GPS fix).
     */
    fun prefetchInitialRegion(lat: Double, lon: Double, radiusM: Double = 800.0) {
        if (hasPrefetchedInitialRegion || cacheManager == null || lat == 0.0 || lon == 0.0) return
        hasPrefetchedInitialRegion = true
        try {
            val dLat = radiusM / METERS_PER_DEG_LAT
            val dLon = radiusM / (METERS_PER_DEG_LAT * cos(Math.toRadians(lat)))
            val bbox = BoundingBox(lat + dLat, lon + dLon, lat - dLat, lon - dLon)
            triggerDownload(bbox, "Initial Startup Area (${radiusM.toInt()}m)")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to prefetch initial region: ${e.message}")
        }
    }

    /**
     * Update vehicle motion and dynamically trigger lookahead tile caching based on speed.
     *
     * @param lat Current latitude in degrees
     * @param lon Current longitude in degrees
     * @param speedMps Current vehicle speed in meters/second
     * @param headingDeg Current vehicle heading/bearing in degrees (0 = North, 90 = East)
     */
    fun onMotionUpdate(lat: Double, lon: Double, speedMps: Double, headingDeg: Double) {
        val cm = cacheManager ?: return
        if (lat == 0.0 || lon == 0.0) return

        val now = System.currentTimeMillis()
        val currentPoint = GeoPoint(lat, lon)

        // Rate limiting check
        val lastPt = lastPrefetchPoint
        if (lastPt != null) {
            val dist = currentPoint.distanceToAsDouble(lastPt)
            val dt = now - lastPrefetchTimeMs
            if (dist < MIN_TRIGGER_DIST_M && dt < MIN_TRIGGER_TIME_MS) {
                return
            }
        }

        if (cm.pendingJobs >= MAX_PENDING_JOBS || isTaskRunning) {
            return
        }

        // Speed-dependent lookahead calculation
        // Stop / crawl (< 2.5 m/s): 600m radius
        // City driving (2.5 - 15 m/s): lookahead = speed * 60s (300m - 900m)
        // High-speed / highway (> 15 m/s): lookahead = speed * 90s (up to 3,500m)
        val effSpeed = max(speedMps, 0.0)
        val lookaheadM = when {
            effSpeed < 2.5 -> 600.0
            effSpeed <= 15.0 -> (effSpeed * 65.0).coerceIn(500.0, 1500.0)
            else -> (effSpeed * 90.0).coerceIn(1500.0, 3500.0)
        }

        // Cross-track safety buffer (for turns, side streets, intersections)
        val crossTrackBufferM = (effSpeed * 20.0).coerceIn(350.0, 800.0)

        val rad = Math.toRadians(headingDeg)
        val dNorthM = lookaheadM * cos(rad)
        val dEastM = lookaheadM * sin(rad)

        val targetLat = lat + (dNorthM / METERS_PER_DEG_LAT)
        val cosLat = max(cos(Math.toRadians(lat)), 0.01)
        val targetLon = lon + (dEastM / (METERS_PER_DEG_LAT * cosLat))

        val bufLat = crossTrackBufferM / METERS_PER_DEG_LAT
        val bufLon = crossTrackBufferM / (METERS_PER_DEG_LAT * cosLat)

        val north = max(lat, targetLat) + bufLat
        val south = min(lat, targetLat) - bufLat
        val east = max(lon, targetLon) + bufLon
        val west = min(lon, targetLon) - bufLon

        val bbox = BoundingBox(north, east, south, west)
        val reason = "Speed-Adaptive (v=%.1f m/s, lookahead=%.0fm)".format(effSpeed, lookaheadM)

        lastPrefetchPoint = currentPoint
        lastPrefetchTimeMs = now

        triggerDownload(bbox, reason)
    }

    private fun triggerDownload(bbox: BoundingBox, reason: String) {
        val cm = cacheManager ?: return
        try {
            // Check how many tiles would be downloaded across zoom levels 15 to 17
            val minZoom = 15
            val maxZoom = 17
            val tileCount = cm.possibleTilesInArea(bbox, minZoom, maxZoom)

            if (tileCount > MAX_TILES_PER_BATCH) {
                Log.d(TAG, "Tile count $tileCount exceeds batch limit $MAX_TILES_PER_BATCH for $reason. Adjusting to zoom 16..17.")
                val trimmedCount = cm.possibleTilesInArea(bbox, 16, 17)
                if (trimmedCount > MAX_TILES_PER_BATCH) return
                executeDownload(bbox, 16, 17, trimmedCount, reason)
            } else if (tileCount > 0) {
                executeDownload(bbox, minZoom, maxZoom, tileCount, reason)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Error calculating tiles for $reason: ${e.message}")
        }
    }

    private fun executeDownload(bbox: BoundingBox, minZoom: Int, maxZoom: Int, count: Int, reason: String) {
        val cm = cacheManager ?: return
        isTaskRunning = true
        Log.i(TAG, "Starting download of $count tiles ($minZoom..$maxZoom) for $reason")

        try {
            cm.downloadAreaAsyncNoUI(
                context,
                bbox,
                minZoom,
                maxZoom,
                object : CacheManager.CacheManagerCallback {
                    override fun onTaskComplete() {
                        isTaskRunning = false
                        Log.i(TAG, "Tile prefetch complete for $reason ($count tiles cached).")
                    }

                    override fun updateProgress(progress: Int, current: Int, total: Int, errors: Int) {
                        // Background progress update
                    }

                    override fun downloadStarted() {
                        Log.d(TAG, "Tile prefetch task initiated for $reason.")
                    }

                    override fun setPossibleTilesInArea(total: Int) {
                        // Called with exact tile count
                    }

                    override fun onTaskFailed(errors: Int) {
                        isTaskRunning = false
                        Log.w(TAG, "Tile prefetch completed with $errors failed tiles for $reason.")
                    }
                }
            )
        } catch (e: Exception) {
            isTaskRunning = false
            Log.w(TAG, "Exception during downloadAreaAsyncNoUI: ${e.message}")
        }
    }
}

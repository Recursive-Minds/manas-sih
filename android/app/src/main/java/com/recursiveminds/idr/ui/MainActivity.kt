package com.recursiveminds.idr.ui

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.graphics.Paint
import android.location.Location
import android.location.LocationManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.preference.PreferenceManager
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.recursiveminds.idr.R
import com.recursiveminds.idr.data.HudUpdate
import com.recursiveminds.idr.data.LiveMetrics
import com.recursiveminds.idr.data.SessionSummary
import com.recursiveminds.idr.data.WarmupStatus
import com.recursiveminds.idr.map.SpeedAdaptiveTilePrefetcher
import com.recursiveminds.idr.service.SensorStreamService
import java.util.Locale
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.tileprovider.tilesource.TileSourcePolicy
import org.osmdroid.tileprovider.tilesource.XYTileSource
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline

data class BenchmarkScenarioItem(
    val id: Int,
    val trip: String,
    val domain: String,
    val durationS: Int,
    val distanceM: Int,
    val driftPct: Double,
    val isPass: Boolean,
    val isRandom: Boolean = false
) {
    override fun toString(): String {
        if (isRandom) {
            return "🎲 Random Held-Out Scenario (All Trips)"
        }
        val status = if (isPass) "PASS" else "FAIL"
        return String.format(
            Locale.US,
            "Scenario #%02d: %s (%s, %ds, %dm) - %.1f%% Drift [%s]",
            id, domain, trip, durationS, distanceM, driftPct, status
        )
    }
}

class MainActivity : AppCompatActivity() {

    private var streamService: SensorStreamService? = null
    private var isBound = false

    // Operational Mode Separation
    enum class AppMode {
        LIVE_DRIVE,
        BENCHMARK_EVALUATION
    }
    private var currentMode = AppMode.LIVE_DRIVE

    // Header & Connection
    private lateinit var tvConnStatus: TextView
    private lateinit var etServerIp: EditText
    private lateinit var etServerPort: EditText
    private lateinit var btnConnect: Button
    private lateinit var btnBenchmarkSuite: Button

    // Dedicated Benchmark Suite Panel
    private lateinit var cardBenchmark: LinearLayout
    private lateinit var tvBenchmarkBadge: TextView
    private lateinit var spinnerScenarios: Spinner
    private lateinit var spinnerSpeed: Spinner
    private lateinit var btnRunBenchmark: Button
    private lateinit var btnCloseBenchmark: Button

    // Warm-Up Readiness Panel
    private lateinit var tvReadyHeadline: TextView
    private lateinit var tvStateBadge: TextView
    private lateinit var tvCondGravity: TextView
    private lateinit var tvCondMount: TextView
    private lateinit var tvCondBuffer: TextView
    private lateinit var tvCondAlpha: TextView
    private lateinit var tvMountStatusDetail: TextView

    // Controls: START, STOP, RESET
    private lateinit var btnStart: Button
    private lateinit var btnStop: Button
    private lateinit var btnReset: Button

    // HUD Metrics Row 1
    private lateinit var tvDriftPct: TextView
    private lateinit var tvHorizError: TextView
    private lateinit var tvMaxError: TextView
    private lateinit var tvGpsAccuracy: TextView

    // HUD Metrics Row 2
    private lateinit var tvDistances: TextView
    private lateinit var tvAlongCross: TextView
    private lateinit var tvSpeeds: TextView
    private lateinit var tvHeadings: TextView

    // Map
    private lateinit var mapView: MapView
    private lateinit var gnssPolyline: Polyline
    private lateinit var drPolyline: Polyline
    private var vehicleMarker: Marker? = null
    private var hasCenteredMap = false
    private var tilePrefetcher: SpeedAdaptiveTilePrefetcher? = null

    // Session Summary Overlay
    private lateinit var cardSummaryModal: LinearLayout
    private lateinit var tvSummaryDrift: TextView
    private lateinit var tvSummaryTier: TextView
    private lateinit var tvSummaryDetails: TextView
    private lateinit var btnDismissSummary: Button

    // Bottom Bar
    private lateinit var tvCsvStatus: TextView
    private lateinit var tvSampleRate: TextView

    // CSV Recording Card
    private lateinit var cardCsvRecording: LinearLayout
    private lateinit var btnCsvMode: Button
    private lateinit var btnCsvRecord: Button
    private lateinit var btnCsvStop: Button
    private lateinit var btnCsvShare: Button
    private lateinit var tvCsvRecBadge: TextView
    private lateinit var tvCsvFilePath: TextView

    // State Tracking
    private var latestMetrics: LiveMetrics? = null
    private var isEngineReady = false
    private var isInBlackout = false
    private var isSummaryDismissed = false

    private val benchmarkScenarioList = mutableListOf(
        BenchmarkScenarioItem(-1, "All", "Diverse", 0, 0, 0.0, true, isRandom = true),
        // S-S3a
        BenchmarkScenarioItem(30, "S-S3a", "Mixed", 60, 244, 5.4, true),
        BenchmarkScenarioItem(26, "S-S3a", "Mixed", 75, 892, 0.5, true),
        BenchmarkScenarioItem(25, "S-S3a", "Mixed", 45, 614, 3.6, true),
        BenchmarkScenarioItem(21, "S-S3a", "Mixed", 30, 325, 8.0, true),
        BenchmarkScenarioItem(22, "S-S3a", "Mixed", 45, 475, 19.6, false),
        BenchmarkScenarioItem(23, "S-S3a", "Mixed", 75, 1128, 20.5, false),
        // S-M
        BenchmarkScenarioItem(8, "S-M", "Highway", 60, 314, 4.6, true),
        BenchmarkScenarioItem(2, "S-M", "Highway", 45, 600, 17.9, false),
        BenchmarkScenarioItem(3, "S-M", "Highway", 75, 1174, 10.6, false),
        BenchmarkScenarioItem(1, "S-M", "Highway", 30, 301, 33.5, false),
        // S-S2
        BenchmarkScenarioItem(11, "S-S2", "Arterial", 60, 435, 0.8, true),
        BenchmarkScenarioItem(10, "S-S2", "Arterial", 30, 245, 7.1, true),
        BenchmarkScenarioItem(12, "S-S2", "Arterial", 45, 261, 6.2, true),
        BenchmarkScenarioItem(13, "S-S2", "Arterial", 45, 331, 11.4, false),
        BenchmarkScenarioItem(9, "S-S2", "Arterial", 75, 872, 90.4, false),
        // S-S1
        BenchmarkScenarioItem(18, "S-S1", "Urban", 45, 98, 12.3, false),
        BenchmarkScenarioItem(15, "S-S1", "Urban", 45, 399, 12.8, false),
        BenchmarkScenarioItem(16, "S-S1", "Urban", 30, 200, 25.6, false),
        BenchmarkScenarioItem(20, "S-S1", "Urban", 60, 135, 32.5, false),
        // S-S4
        BenchmarkScenarioItem(32, "S-S4", "Arterial", 75, 610, 4.8, true),
        BenchmarkScenarioItem(36, "S-S4", "Arterial", 45, 739, 7.5, true),
        BenchmarkScenarioItem(38, "S-S4", "Arterial", 60, 931, 7.7, true),
        BenchmarkScenarioItem(31, "S-S4", "Arterial", 45, 490, 9.4, true),
        BenchmarkScenarioItem(33, "S-S4", "Arterial", 60, 443, 11.3, false),
        BenchmarkScenarioItem(34, "S-S4", "Arterial", 45, 328, 28.2, false)
    )
    private lateinit var scenarioAdapter: ArrayAdapter<BenchmarkScenarioItem>

    private val connection = object : ServiceConnection {
        override fun onServiceConnected(className: ComponentName, service: IBinder) {
            val binder = service as SensorStreamService.LocalBinder
            val s = binder.getService()
            streamService = s
            isBound = true
            setupServiceCallbacks(s)
        }

        override fun onServiceDisconnected(arg0: ComponentName) {
            streamService = null
            isBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize OSMDroid configuration with persistent 500MB offline tile cache
        Configuration.getInstance().load(this, PreferenceManager.getDefaultSharedPreferences(this))
        Configuration.getInstance().userAgentValue = "IDR-DeadReckoning/1.0 (Android; support@recursiveminds.com)"
        Configuration.getInstance().cacheMapTileCount = 500.toShort()
        Configuration.getInstance().tileFileSystemCacheMaxBytes = 1024L * 1024L * 500L
        Configuration.getInstance().tileFileSystemCacheTrimBytes = 1024L * 1024L * 400L

        setContentView(R.layout.activity_main)

        initViews()
        initMap()
        checkAndRequestPermissions()
        startAndBindService()
    }

    private fun initViews() {
        // Header & Connection
        tvConnStatus = findViewById(R.id.tvConnStatus)
        etServerIp = findViewById(R.id.etServerIp)
        etServerPort = findViewById(R.id.etServerPort)
        btnConnect = findViewById(R.id.btnConnect)
        btnBenchmarkSuite = findViewById(R.id.btnBenchmarkSuite)

        // Dedicated Benchmark Suite Panel
        cardBenchmark = findViewById(R.id.cardBenchmark)
        tvBenchmarkBadge = findViewById(R.id.tvBenchmarkBadge)
        spinnerScenarios = findViewById(R.id.spinnerScenarios)
        spinnerSpeed = findViewById(R.id.spinnerSpeed)
        btnRunBenchmark = findViewById(R.id.btnRunBenchmark)
        btnCloseBenchmark = findViewById(R.id.btnCloseBenchmark)

        // CSV Recording Card
        cardCsvRecording = findViewById(R.id.cardCsvRecording)
        btnCsvMode = findViewById(R.id.btnCsvMode)
        btnCsvRecord = findViewById(R.id.btnCsvRecord)
        btnCsvStop = findViewById(R.id.btnCsvStop)
        btnCsvShare = findViewById(R.id.btnCsvShare)
        tvCsvRecBadge = findViewById(R.id.tvCsvRecBadge)
        tvCsvFilePath = findViewById(R.id.tvCsvFilePath)

        initBenchmarkSpinners()

        btnCsvMode.setOnClickListener {
            cardCsvRecording.visibility = if (cardCsvRecording.visibility == View.VISIBLE) View.GONE else View.VISIBLE
            // Hide benchmark panel when CSV panel opens
            if (cardCsvRecording.visibility == View.VISIBLE) cardBenchmark.visibility = View.GONE
        }

        btnCsvRecord.setOnClickListener {
            streamService?.startCsvRecording()
        }

        btnCsvStop.setOnClickListener {
            streamService?.stopCsvRecording()
        }

        btnCsvShare.setOnClickListener {
            shareCsvFile()
        }

        btnBenchmarkSuite.setOnClickListener {
            if (cardBenchmark.visibility == View.VISIBLE) {
                cardBenchmark.visibility = View.GONE
                // User closed benchmark: deload benchmark mode
                streamService?.stopBenchmark()
                currentMode = AppMode.LIVE_DRIVE
                tvBenchmarkBadge.text = "LIVE SENSORS"
                tvBenchmarkBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
            } else {
                cardBenchmark.visibility = View.VISIBLE
                cardCsvRecording.visibility = View.GONE
                // User opened benchmark: prepare scenario and load ticks!
                val scItem = benchmarkScenarioList.getOrNull(spinnerScenarios.selectedItemPosition)
                    ?: benchmarkScenarioList.firstOrNull { it.id == 30 }
                    ?: benchmarkScenarioList[0]
                val scId = if (scItem.id > 0) scItem.id else 30
                currentMode = AppMode.BENCHMARK_EVALUATION
                tvBenchmarkBadge.text = "PRELOADING #${scId}..."
                tvBenchmarkBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_amber))
                streamService?.prepareBenchmark(scId)
            }
        }

        btnCloseBenchmark.setOnClickListener {
            cardBenchmark.visibility = View.GONE
            streamService?.stopBenchmark()
            currentMode = AppMode.LIVE_DRIVE
            tvBenchmarkBadge.text = "LIVE SENSORS"
            tvBenchmarkBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
        }

        btnRunBenchmark.setOnClickListener {
            val s = streamService
            if (s == null || !s.isConnected.get()) {
                Toast.makeText(this, "Connect to IDR server first!", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val scenarioItem = benchmarkScenarioList.getOrNull(spinnerScenarios.selectedItemPosition)
                ?: benchmarkScenarioList.firstOrNull { it.id == 30 }
                ?: benchmarkScenarioList[0]
            val scenarioId = scenarioItem.id

            val speed = when (spinnerSpeed.selectedItemPosition) {
                0 -> 2.0 // Demo 2.0x
                1 -> 1.0 // Real-time 1.0x
                2 -> 5.0 // Rapid 5.0x
                else -> 2.0
            }

            currentMode = AppMode.BENCHMARK_EVALUATION
            isSummaryDismissed = false
            cardSummaryModal.visibility = View.GONE

            // Clear previous tracks for clean benchmark run
            gnssPolyline.actualPoints.clear()
            drPolyline.actualPoints.clear()
            hasCenteredMap = false
            mapView.invalidate()

            tvBenchmarkBadge.text = "LAUNCHING #$scenarioId"
            tvBenchmarkBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_amber))
            Toast.makeText(this, "Running Scenario #$scenarioId @ ${speed}x...", Toast.LENGTH_SHORT).show()

            s.startBenchmark(scenarioId, speed)
        }

        // Warm-up
        tvReadyHeadline = findViewById(R.id.tvReadyHeadline)
        tvStateBadge = findViewById(R.id.tvStateBadge)
        tvCondGravity = findViewById(R.id.tvCondGravity)
        tvCondMount = findViewById(R.id.tvCondMount)
        tvCondBuffer = findViewById(R.id.tvCondBuffer)
        tvCondAlpha = findViewById(R.id.tvCondAlpha)
        tvMountStatusDetail = findViewById(R.id.tvMountStatusDetail)

        // Controls
        btnStart = findViewById(R.id.btnStart)
        btnStop = findViewById(R.id.btnStop)
        btnReset = findViewById(R.id.btnReset)

        // HUD Row 1
        tvDriftPct = findViewById(R.id.tvDriftPct)
        tvHorizError = findViewById(R.id.tvHorizError)
        tvMaxError = findViewById(R.id.tvMaxError)
        tvGpsAccuracy = findViewById(R.id.tvGpsAccuracy)

        // HUD Row 2
        tvDistances = findViewById(R.id.tvDistances)
        tvAlongCross = findViewById(R.id.tvAlongCross)
        tvSpeeds = findViewById(R.id.tvSpeeds)
        tvHeadings = findViewById(R.id.tvHeadings)

        // Map
        mapView = findViewById(R.id.mapView)

        // Summary Card
        cardSummaryModal = findViewById(R.id.cardSummaryModal)
        tvSummaryDrift = findViewById(R.id.tvSummaryDrift)
        tvSummaryTier = findViewById(R.id.tvSummaryTier)
        tvSummaryDetails = findViewById(R.id.tvSummaryDetails)
        btnDismissSummary = findViewById(R.id.btnDismissSummary)

        // Bottom Bar
        tvCsvStatus = findViewById(R.id.tvCsvStatus)
        tvSampleRate = findViewById(R.id.tvSampleRate)

        // Listeners
        btnConnect.setOnClickListener {
            val s = streamService
            if (s == null) {
                Log.w("MainActivity", "btnConnect clicked but streamService is null!")
                Toast.makeText(this, "Waiting for service to bind...", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            if (s.isConnected.get()) {
                Log.i("MainActivity", "Disconnecting from server...")
                s.disconnectServer()
            } else {
                val ip = etServerIp.text.toString().trim()
                val port = etServerPort.text.toString().trim().toIntOrNull() ?: 8765
                Log.i("MainActivity", "Connecting to server $ip:$port...")
                s.connectServer(ip, port)
            }
        }

        btnStart.setOnClickListener {
            val s = streamService ?: return@setOnClickListener
            if (!isEngineReady) {
                Toast.makeText(this, "Starting Dead Reckoning blackout (Warmup still completing...)", Toast.LENGTH_SHORT).show()
            }
            s.startBlackout()
            isInBlackout = true
            isSummaryDismissed = false
            updateControlButtons()
            cardSummaryModal.visibility = View.GONE
            tvStateBadge.text = "BLACKOUT"
            tvStateBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_rose))
        }

        btnStop.setOnClickListener {
            val s = streamService ?: return@setOnClickListener
            s.stopBlackout()
            isInBlackout = false
            updateControlButtons()
            tvStateBadge.text = "STOPPED"
            tvStateBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_amber))

            // Show summary card immediately from latest metrics or session summary
            showSummaryCard()
        }

        btnReset.setOnClickListener {
            val s = streamService ?: return@setOnClickListener
            s.resetSession()
            isInBlackout = false
            isSummaryDismissed = false
            isEngineReady = false
            updateControlButtons()
            tvStateBadge.text = "WARMING UP"
            tvStateBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
            cardSummaryModal.visibility = View.GONE

            // Clear map tracks
            gnssPolyline.actualPoints.clear()
            drPolyline.actualPoints.clear()
            hasCenteredMap = false
            mapView.invalidate()

            // Reset labels
            tvDriftPct.text = "0.00%"
            tvHorizError.text = "0.0 m"
            tvMaxError.text = "0.0 m"
            tvDistances.text = "0 / 0 m"
            tvAlongCross.text = "0.0 / 0.0 m"
            tvSpeeds.text = "0.0 / 0.0"
            tvHeadings.text = "0° / 0°"

            currentMode = AppMode.LIVE_DRIVE
            s.setSensorStreamingMuted(false)
            tvBenchmarkBadge.text = "ZERO LEAKAGE"
            tvBenchmarkBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
            Toast.makeText(this, "Session reset", Toast.LENGTH_SHORT).show()
        }

        btnDismissSummary.setOnClickListener {
            isSummaryDismissed = true
            cardSummaryModal.visibility = View.GONE
        }

        updateControlButtons()
    }

    private fun initBenchmarkSpinners() {
        scenarioAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, benchmarkScenarioList)
        scenarioAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerScenarios.adapter = scenarioAdapter

        // Default selection to Scenario #30 (pos 1)
        if (benchmarkScenarioList.size > 1) {
            spinnerScenarios.setSelection(1)
        }

        val speeds = arrayOf("2.0x (Demo Replay)", "1.0x (Real-Time)", "5.0x (Rapid)")
        val speedAdapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, speeds)
        speedAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        spinnerSpeed.adapter = speedAdapter

        spinnerScenarios.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                if (cardBenchmark.visibility == View.VISIBLE) {
                    val scItem = benchmarkScenarioList.getOrNull(position)
                    if (scItem != null) {
                        val scId = if (scItem.id > 0) scItem.id else 30
                        tvBenchmarkBadge.text = "PRELOADING #${scId}..."
                        tvBenchmarkBadge.setTextColor(ContextCompat.getColor(this@MainActivity, R.color.accent_amber))
                        streamService?.prepareBenchmark(scId)
                    }
                }
            }
            override fun onNothingSelected(parent: AdapterView<*>?) {}
        }
    }

    private fun initMap() {
        val tilePolicy = TileSourcePolicy(8, 0)
        val osmTileSource = XYTileSource(
            "Mapnik_IDR",
            0,
            19,
            256,
            ".png",
            arrayOf("https://tile.openstreetmap.org/"),
            "© OpenStreetMap contributors",
            tilePolicy
        )
        mapView.setTileSource(osmTileSource)
        mapView.setMultiTouchControls(true)
        mapView.controller.setZoom(17.0)
        mapView.setUseDataConnection(true)
        mapView.isHorizontalMapRepetitionEnabled = false
        mapView.isVerticalMapRepetitionEnabled = false

        // GNSS Ground Truth track (emerald green)
        gnssPolyline = Polyline(mapView)
        gnssPolyline.outlinePaint.color = Color.parseColor("#10B981")
        gnssPolyline.outlinePaint.strokeWidth = 8f
        gnssPolyline.outlinePaint.strokeCap = Paint.Cap.ROUND
        mapView.overlays.add(gnssPolyline)

        // Dead Reckoning track (electric cyan)
        drPolyline = Polyline(mapView)
        drPolyline.outlinePaint.color = Color.parseColor("#00F2FE")
        drPolyline.outlinePaint.strokeWidth = 10f
        drPolyline.outlinePaint.strokeCap = Paint.Cap.ROUND
        mapView.overlays.add(drPolyline)

        // Vehicle position marker
        vehicleMarker = Marker(mapView)
        vehicleMarker?.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        mapView.overlays.add(vehicleMarker)

        // Initialize speed-adaptive tile prefetcher for offline resilience
        tilePrefetcher = SpeedAdaptiveTilePrefetcher(this, mapView)

        // Center immediately on real last known location if available
        try {
            val lm = getSystemService(Context.LOCATION_SERVICE) as? LocationManager
            val lastGps: Location? = lm?.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            val lastNet: Location? = lm?.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            val best: Location? = when {
                lastGps != null && lastNet != null -> if (lastGps.time >= lastNet.time) lastGps else lastNet
                lastGps != null -> lastGps
                else -> lastNet
            }
            if (best != null) {
                val pt = GeoPoint(best.latitude, best.longitude)
                mapView.controller.setCenter(pt)
                vehicleMarker?.position = pt
                hasCenteredMap = true
            }
        } catch (_: SecurityException) {}
    }

    private fun setupServiceCallbacks(s: SensorStreamService) {
        s.onConnectionStateChanged = { connected ->
            runOnUiThread {
                if (connected) {
                    tvConnStatus.text = "Connected"
                    tvConnStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
                    btnConnect.text = "Disconnect"
                    btnConnect.backgroundTintList = ColorStateList.valueOf(ContextCompat.getColor(this, R.color.card_border))
                    val ip = etServerIp.text.toString().trim().ifEmpty { "127.0.0.1" }
                    val port = etServerPort.text.toString().trim().toIntOrNull() ?: 8765
                    fetchCanonicalScenariosAsync(ip, port)
                } else {
                    tvConnStatus.text = "Disconnected"
                    tvConnStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_amber))
                    btnConnect.text = "Connect"
                    btnConnect.backgroundTintList = ColorStateList.valueOf(ContextCompat.getColor(this, R.color.accent_blue))
                    isEngineReady = false
                    updateControlButtons()
                }
            }
        }

        s.onSensorRateUpdate = { imuRate, gnssRate ->
            runOnUiThread {
                tvSampleRate.text = "IMU: $imuRate Hz | GPS: $gnssRate Hz"
            }
        }

        s.onCsvStateChanged = { recording, filePath ->
            runOnUiThread {
                if (recording) {
                    tvCsvRecBadge.text = "● REC ACTIVE"
                    tvCsvRecBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_rose))
                    btnCsvRecord.isEnabled = false
                    btnCsvRecord.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#475569"))
                    btnCsvStop.isEnabled = true
                    btnCsvStop.backgroundTintList = ColorStateList.valueOf(ContextCompat.getColor(this, R.color.accent_rose))
                    btnCsvShare.isEnabled = false
                    btnCsvShare.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#475569"))
                    tvCsvStatus.text = "REC: Active"
                    tvCsvStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_rose))
                } else {
                    tvCsvRecBadge.text = "● IDLE"
                    tvCsvRecBadge.setTextColor(ContextCompat.getColor(this, R.color.text_muted))
                    btnCsvRecord.isEnabled = true
                    btnCsvRecord.backgroundTintList = ColorStateList.valueOf(ContextCompat.getColor(this, R.color.accent_rose))
                    btnCsvStop.isEnabled = false
                    btnCsvStop.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#475569"))
                    btnCsvShare.isEnabled = filePath != null
                    btnCsvShare.backgroundTintList = ColorStateList.valueOf(if (filePath != null) ContextCompat.getColor(this, R.color.accent_blue) else Color.parseColor("#475569"))
                    tvCsvStatus.text = "REC: Idle"
                    tvCsvStatus.setTextColor(ContextCompat.getColor(this, R.color.text_muted))
                }
                if (filePath != null) {
                    val fileName = java.io.File(filePath).name
                    tvCsvFilePath.text = "File: $fileName"
                }
            }
        }

        s.onLocalGnssUpdate = { gnss ->
            runOnUiThread {
                if (currentMode == AppMode.BENCHMARK_EVALUATION) {
                    // Mute physical desk phone GPS from moving map during benchmark replay
                    return@runOnUiThread
                }
                if (gnss.latitudeDeg != 0.0 && gnss.longitudeDeg != 0.0) {
                    val pt = GeoPoint(gnss.latitudeDeg, gnss.longitudeDeg)
                    // Always update accuracy label regardless of speed
                    tvGpsAccuracy.text = String.format("±%.1f m", gnss.accuracyHM)
                    tilePrefetcher?.prefetchInitialRegion(gnss.latitudeDeg, gnss.longitudeDeg)

                    if (currentMode == AppMode.LIVE_DRIVE) {
                        // Only update marker + track when actually moving.
                        // GPS multipath jitter (2-5m) at stationary causes pointer "drift"
                        // when the phone is sitting on a desk — this prevents that.
                        val speedMps = gnss.speedMps?.toDouble() ?: 0.0
                        val isMoving = speedMps > 0.5

                        if (isMoving) {
                            tilePrefetcher?.onMotionUpdate(
                                gnss.latitudeDeg,
                                gnss.longitudeDeg,
                                speedMps,
                                gnss.bearingDeg?.toDouble() ?: 0.0
                            )
                            vehicleMarker?.position = pt
                            if (gnss.bearingDeg != null) {
                                vehicleMarker?.rotation = MarkerHeading.toMarkerRotation(gnss.bearingDeg)  // [ROUND1] T1
                            }
                            val lastPt = gnssPolyline.actualPoints.lastOrNull()
                            val distToLast = if (lastPt != null) pt.distanceToAsDouble(lastPt) else 1000.0
                            if (distToLast > 500.0) {
                                gnssPolyline.actualPoints.clear()
                                gnssPolyline.addPoint(pt)
                                mapView.controller.animateTo(pt)
                            } else if (distToLast >= 2.0) {
                                gnssPolyline.addPoint(pt)
                            }
                        }

                        // Always center map on first valid fix (even stationary), then follow motion
                        if (!hasCenteredMap) {
                            mapView.controller.setCenter(pt)
                            mapView.controller.setZoom(18.0)
                            // Place marker at initial position even if stationary
                            vehicleMarker?.position = pt
                            hasCenteredMap = true
                        }
                        mapView.invalidate()
                    }
                }
            }
        }

        s.onHudUpdateListener = { hud ->
            runOnUiThread {
                updateHudUi(hud)
            }
        }

        // Auto-connect to server on bind
        val ip = etServerIp.text.toString().trim().ifEmpty { "127.0.0.1" }
        val port = etServerPort.text.toString().trim().toIntOrNull() ?: 8765
        s.connectServer(ip, port)
    }

    private fun updateControlButtons() {
        if (isInBlackout) {
            btnStart.isEnabled = false
            btnStart.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#475569"))
            btnStop.isEnabled = true
            btnStop.backgroundTintList = ColorStateList.valueOf(ContextCompat.getColor(this, R.color.accent_rose))
        } else {
            btnStop.isEnabled = false
            btnStop.backgroundTintList = ColorStateList.valueOf(Color.parseColor("#475569"))
            btnStart.isEnabled = true
            val startColor = if (isEngineReady) {
                ContextCompat.getColor(this, R.color.accent_emerald)
            } else {
                ContextCompat.getColor(this, R.color.accent_amber)
            }
            btnStart.backgroundTintList = ColorStateList.valueOf(startColor)
        }
    }

    private fun updateHudUi(hud: HudUpdate) {
        // Benchmark mode badge tracking
        if (hud.benchmarkActive) {
            currentMode = AppMode.BENCHMARK_EVALUATION
            val scId = hud.benchmarkScenario ?: 30
            if (isInBlackout || (latestMetrics != null && latestMetrics?.isBlackout == true)) {
                tvBenchmarkBadge.text = "REPLAYING #$scId"
                tvBenchmarkBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_rose))
                btnRunBenchmark.text = "REPLAYING..."
                btnRunBenchmark.isEnabled = false
            } else {
                tvBenchmarkBadge.text = "BENCHMARK #$scId READY"
                tvBenchmarkBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
                btnRunBenchmark.text = "RUN BENCHMARK"
                btnRunBenchmark.isEnabled = true
            }

            // Synchronize scenario spinner if triggered externally
            val targetPos = benchmarkScenarioList.indexOfFirst { it.id == scId }
            if (targetPos >= 0 && spinnerScenarios.selectedItemPosition != targetPos) {
                spinnerScenarios.setSelection(targetPos)
            }
        } else {
            btnRunBenchmark.text = "RUN BENCHMARK"
            btnRunBenchmark.isEnabled = true
            if (currentMode == AppMode.BENCHMARK_EVALUATION && cardBenchmark.visibility != View.VISIBLE) {
                // Benchmark closed or stopped — transition cleanly back to LIVE_DRIVE
                currentMode = AppMode.LIVE_DRIVE
                tvBenchmarkBadge.text = "LIVE SENSORS"
                tvBenchmarkBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
                gnssPolyline.actualPoints.clear()
                drPolyline.actualPoints.clear()
                mapView.invalidate()
                hasCenteredMap = false
            }
        }

        // State update from server
        if (hud.state.equals("BLACKOUT", ignoreCase = true)) {
            if (!isInBlackout) {
                isInBlackout = true
                isSummaryDismissed = false
                tvStateBadge.text = "BLACKOUT"
                tvStateBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_rose))
                updateControlButtons()
            }
        } else if (hud.state.equals("WARMING_UP", ignoreCase = true)) {
            if (isInBlackout) {
                isInBlackout = false
                tvStateBadge.text = "WARMING UP"
                tvStateBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
                updateControlButtons()
            }
        }

        // Warmup status
        hud.warmup?.let { w ->
            isEngineReady = w.isReady
            updateWarmupPanel(w)
            updateControlButtons()
        }

        // Mount Status
        tvMountStatusDetail.text = hud.mountStatus
        if (hud.mountStatus.contains("reused", ignoreCase = true) || hud.mountStatus.contains("locked", ignoreCase = true)) {
            tvMountStatusDetail.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
        } else if (hud.mountStatus.contains("changed", ignoreCase = true)) {
            tvMountStatusDetail.setTextColor(ContextCompat.getColor(this, R.color.accent_rose))
        } else {
            tvMountStatusDetail.setTextColor(ContextCompat.getColor(this, R.color.accent_cyan))
        }

        // Metrics
        val m = hud.metrics
        if (m != null) {
            latestMetrics = m
            tvDriftPct.text = String.format("%.2f%%", m.driftPct)
            if (m.driftPct < 10.0) {
                tvDriftPct.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
            } else if (m.driftPct < 20.0) {
                tvDriftPct.setTextColor(ContextCompat.getColor(this, R.color.accent_amber))
            } else {
                tvDriftPct.setTextColor(ContextCompat.getColor(this, R.color.accent_rose))
            }

            tvHorizError.text = String.format("%.1f m", m.horizontalErrorM)
            tvMaxError.text = String.format("%.1f m", m.maxHorizontalErrorM)
            tvGpsAccuracy.text = String.format("±%.1f m", m.gnssAccuracyHM)

            tvDistances.text = String.format("%.0f / %.0f m", m.drDistM, m.gnssDistM)
            tvAlongCross.text = String.format("%.1f / %.1f m", m.alongTrackM, m.crossTrackM)

            val gnssSpeed = m.gnssSpeedMps
            tvSpeeds.text = String.format("%.1f / %.1f", m.drSpeedMps, gnssSpeed)

            val gnssHdg = m.gnssBearingDeg
            tvHeadings.text = String.format("%.0f° / %.0f°", m.drHeadingDeg, gnssHdg)

            // If session summary received while stopped, show summary card unless dismissed
            if (!isInBlackout && m.sessionSummary != null && !isSummaryDismissed) {
                renderSummary(m.sessionSummary)
            }
        }

        // Map updates — ONLY apply server-side positions during BENCHMARK_EVALUATION.
        // In LIVE_DRIVE mode the map is driven exclusively by the phone's own GPS
        // via onLocalGnssUpdate. Letting server benchmark coordinates through in
        // LIVE_DRIVE mode is the root cause of the map-mixing bug.
        if (currentMode == AppMode.BENCHMARK_EVALUATION) {
            hud.gnssPos?.let { g ->
                if (g.lat != 0.0 && g.lon != 0.0) {
                    val pt = GeoPoint(g.lat, g.lon)
                    tilePrefetcher?.prefetchInitialRegion(g.lat, g.lon)
                    tilePrefetcher?.onMotionUpdate(
                        g.lat,
                        g.lon,
                        hud.metrics?.gnssSpeedMps ?: 0.0,
                        g.bearingDeg ?: 0.0
                    )
                    val lastPt = gnssPolyline.actualPoints.lastOrNull()
                    val distToLast = if (lastPt != null) pt.distanceToAsDouble(lastPt) else 1000.0
                    if (distToLast > 500.0) {
                        // Scenario jump or trip switch: reset track
                        gnssPolyline.actualPoints.clear()
                        gnssPolyline.addPoint(pt)
                        mapView.controller.animateTo(pt)
                    } else if (distToLast >= 1.0) {
                        gnssPolyline.addPoint(pt)
                    }
                    if (!hasCenteredMap) {
                        mapView.controller.animateTo(pt)
                        hasCenteredMap = true
                    }
                    if (hud.drPos == null) {
                        vehicleMarker?.position = pt
                        vehicleMarker?.rotation = MarkerHeading.toMarkerRotation(g.bearingDeg)  // [ROUND1] T1
                    }
                }
            }

            hud.drPos?.let { d ->
                if (d.lat != 0.0 && d.lon != 0.0) {
                    val pt = GeoPoint(d.lat, d.lon)
                    tilePrefetcher?.onMotionUpdate(
                        d.lat,
                        d.lon,
                        hud.metrics?.drSpeedMps ?: 0.0,
                        d.headingDeg ?: 0.0
                    )
                    val lastPt = drPolyline.actualPoints.lastOrNull()
                    val distToLast = if (lastPt != null) pt.distanceToAsDouble(lastPt) else 1000.0
                    if (distToLast > 500.0) {
                        drPolyline.actualPoints.clear()
                        drPolyline.addPoint(pt)
                    } else if (distToLast >= 0.5) {
                        drPolyline.addPoint(pt)
                    }
                    vehicleMarker?.position = pt
                    vehicleMarker?.rotation = MarkerHeading.toMarkerRotation(d.headingDeg)  // [ROUND1] T1
                }
            }
        }
        mapView.invalidate()
    }

    private fun updateWarmupPanel(w: WarmupStatus) {
        val emerald = ContextCompat.getColor(this, R.color.accent_emerald)
        val muted = ContextCompat.getColor(this, R.color.text_muted)
        val isBenchmark = (currentMode == AppMode.BENCHMARK_EVALUATION)

        // 1. Gravity Leveling — phone held still for 3s so accelerometer baseline settles
        if (w.gravityConverged) {
            tvCondGravity.text = "✓ Gravity"
            tvCondGravity.setTextColor(emerald)
        } else {
            tvCondGravity.text = "✗ Gravity (hold still)"
            tvCondGravity.setTextColor(muted)
        }

        // 2. Mount Calibration — "Reused" in benchmark means the UK-trip calibration is loaded,
        //    not your phone's live cradle angle. In LIVE_DRIVE this shows real turn progress.
        if (w.mountLocked) {
            val label = when {
                isBenchmark && w.mountStatus.contains("reused", ignoreCase = true) -> "✓ Bench-Calib"
                w.mountStatus.contains("reused", ignoreCase = true) -> "✓ Mount Reused"
                else -> "✓ ${w.turnsDisplay}"
            }
            tvCondMount.text = label
            tvCondMount.setTextColor(emerald)
        } else {
            tvCondMount.text = "✗ Mount (${w.turnsDisplay})"
            tvCondMount.setTextColor(muted)
        }

        // 3. Macro Feature Buffer — 6s of IMU history for the AI velocity model
        if (w.bufferWarm) {
            tvCondBuffer.text = "✓ Buffer 6s"
            tvCondBuffer.setTextColor(emerald)
        } else {
            tvCondBuffer.text = "✗ Buffer (need 6s)"
            tvCondBuffer.setTextColor(muted)
        }

        // 4. Alpha Adaptive Scaling — speed-scale factor learned from ≥3 moving GNSS fixes
        if (w.alphaLearned) {
            tvCondAlpha.text = "✓ Alpha"
            tvCondAlpha.setTextColor(emerald)
        } else {
            tvCondAlpha.text = if (isBenchmark) "✗ Alpha (need motion)" else "✗ Alpha (drive slowly)"
            tvCondAlpha.setTextColor(muted)
        }

        // Headline
        if (w.isReady) {
            tvReadyHeadline.text = if (isBenchmark) "BENCHMARK: ENGINE READY" else "WARM-UP STATUS: READY TO START"
            tvReadyHeadline.setTextColor(emerald)
        } else {
            tvReadyHeadline.text = if (isBenchmark) "BENCHMARK: WARMING ENGINE..." else "WARM-UP STATUS: WAITING FOR READY"
            tvReadyHeadline.setTextColor(ContextCompat.getColor(this, R.color.accent_amber))
        }
    }

    private fun showSummaryCard() {
        val m = latestMetrics
        val s = m?.sessionSummary ?: SessionSummary(
            finalErrorM = m?.horizontalErrorM ?: 0.0,
            driftPct = m?.driftPct ?: 0.0,
            maxErrorM = m?.maxHorizontalErrorM ?: (m?.horizontalErrorM ?: 0.0),
            durationS = m?.elapsedS ?: 0.0,
            drDistM = m?.drDistM ?: 0.0,
            gnssDistM = m?.gnssDistM ?: 0.0,
            alongTrackM = m?.alongTrackM ?: 0.0,
            crossTrackM = m?.crossTrackM ?: 0.0,
            speedRegime = m?.speedRegime ?: "City (20-50 km/h)",
            tier = m?.speedRegime ?: "City (20-50 km/h)"
        )
        renderSummary(s)
    }

    private fun renderSummary(s: SessionSummary) {
        cardBenchmark.visibility = View.GONE
        cardSummaryModal.bringToFront()
        cardSummaryModal.elevation = 40f
        cardSummaryModal.visibility = View.VISIBLE
        tvSummaryDrift.text = String.format("Drift: %.2f%% (target <10%%)", s.driftPct)
        tvSummaryTier.text = if (s.speedRegime.isNotEmpty()) s.speedRegime else s.tier

        val driftColor = if (s.driftPct < 10.0) {
            ContextCompat.getColor(this, R.color.accent_emerald)
        } else if (s.driftPct < 20.0) {
            ContextCompat.getColor(this, R.color.accent_amber)
        } else {
            ContextCompat.getColor(this, R.color.accent_rose)
        }
        tvSummaryDrift.setTextColor(driftColor)
        tvSummaryTier.setTextColor(ContextCompat.getColor(this, R.color.accent_cyan))

        tvSummaryDetails.text = String.format(
            "Final Error: %.1f m | Max Error: %.1f m\nDuration: %.1f s | Distance: %.1f m\nAlong-Track: %.1f m | Cross-Track: %.1f m",
            s.finalErrorM,
            s.maxErrorM,
            s.durationS,
            s.drDistM,
            s.alongTrackM,
            s.crossTrackM
        )
    }

    private fun shareCsvFile() {
        val s = streamService ?: return
        s.flushCsv()
        val file = s.getCurrentCsvFile()
        if (file == null || !file.exists() || file.length() == 0L) {
            Toast.makeText(this, "CSV file is empty or not found yet", Toast.LENGTH_SHORT).show()
            return
        }

        try {
            val uri = FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                file
            )
            val shareIntent = Intent(Intent.ACTION_SEND).apply {
                type = "text/csv"
                putExtra(Intent.EXTRA_STREAM, uri)
                putExtra(Intent.EXTRA_SUBJECT, "IDR Sensor Telemetry CSV")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            }
            startActivity(Intent.createChooser(shareIntent, "Share IDR Telemetry CSV"))
        } catch (e: Exception) {
            Toast.makeText(this, "Sharing error: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun fetchCanonicalScenariosAsync(serverIp: String, port: Int) {
        Thread {
            try {
                val url = java.net.URL("http://$serverIp:$port/api/scenarios")
                val conn = url.openConnection() as java.net.HttpURLConnection
                conn.connectTimeout = 3000
                conn.readTimeout = 3000
                if (conn.responseCode == 200) {
                    val jsonStr = conn.inputStream.bufferedReader().use { it.readText() }
                    val jsonObj = org.json.JSONObject(jsonStr)
                    val arr = jsonObj.getJSONArray("scenarios")
                    val items = mutableListOf<BenchmarkScenarioItem>()
                    for (i in 0 until arr.length()) {
                        val obj = arr.getJSONObject(i)
                        val id = obj.getInt("id")
                        val isRandom = id <= 0
                        val trip = obj.optString("trip", if (isRandom) "All" else "")
                        val domain = obj.optString("env", "")
                        val dur = obj.optDouble("duration_s", 0.0).toInt()
                        val dist = obj.optDouble("distance_m", 0.0).toInt()
                        val drift = obj.optDouble("benchmark_drift_pct", 0.0)
                        val isPass = drift < 10.0
                        items.add(
                            BenchmarkScenarioItem(
                                id = id,
                                trip = trip,
                                domain = domain,
                                durationS = dur,
                                distanceM = dist,
                                driftPct = drift,
                                isPass = isPass,
                                isRandom = isRandom
                            )
                        )
                    }
                    if (items.isNotEmpty()) {
                        runOnUiThread {
                            val curId = benchmarkScenarioList.getOrNull(spinnerScenarios.selectedItemPosition)?.id ?: 30
                            benchmarkScenarioList.clear()
                            benchmarkScenarioList.addAll(items)
                            scenarioAdapter.notifyDataSetChanged()
                            val targetPos = benchmarkScenarioList.indexOfFirst { it.id == curId }
                            if (targetPos >= 0) {
                                spinnerScenarios.setSelection(targetPos)
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.d("MainActivity", "Remote scenarios fetch note: ${e.message}")
            }
        }.start()
    }

    private fun checkAndRequestPermissions() {
        val permissions = mutableListOf(
            Manifest.permission.ACCESS_FINE_LOCATION,
            Manifest.permission.ACCESS_COARSE_LOCATION
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }

        val missing = permissions.filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }

        if (missing.isNotEmpty()) {
            ActivityCompat.requestPermissions(this, missing.toTypedArray(), 1001)
        }
    }

    private fun startAndBindService() {
        val intent = Intent(this, SensorStreamService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }
        bindService(intent, connection, Context.BIND_AUTO_CREATE)
    }

    override fun onResume() {
        super.onResume()
        mapView.onResume()
    }

    override fun onPause() {
        super.onPause()
        mapView.onPause()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isBound) {
            unbindService(connection)
            isBound = false
        }
    }
}

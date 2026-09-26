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
import androidx.appcompat.app.AlertDialog
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
    val isRandom: Boolean = false
) {
    override fun toString(): String {
        if (isRandom) {
            return "🎲 Random Held-Out Scenario (All Trips)"
        }
        return String.format(
            Locale.US,
            "Scenario #%02d: %s (%s, %ds, %dm)",
            id, domain, trip, durationS, distanceM
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
    private lateinit var cardConnection: LinearLayout
    private lateinit var tvConnStatus: TextView
    private lateinit var etServerIp: EditText
    private lateinit var etServerPort: EditText
    private lateinit var btnConnect: Button
    private lateinit var btnBenchmarkSuite: Button

    // Map & Feature Controls
    private lateinit var cardMapControls: LinearLayout
    private lateinit var btnPrefetchArea: Button
    private lateinit var btnToggleMapMatching: Button
    private lateinit var tvHandoffState: TextView

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
    private var drMarker: Marker? = null
    private var reconciledMarker: Marker? = null
    private var hasCenteredMap = false
    private var tilePrefetcher: SpeedAdaptiveTilePrefetcher? = null

    private var isMapMatchingEnabled = true
    private var latestWarmup: WarmupStatus? = null

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
    private var currentPreparedScenarioId: Int? = null

    private val benchmarkScenarioList = mutableListOf<BenchmarkScenarioItem>()

    private fun loadCanonicalScenariosFromAssets(): List<BenchmarkScenarioItem> {
        val list = mutableListOf<BenchmarkScenarioItem>()
        list.add(BenchmarkScenarioItem(-1, "All", "Diverse", 0, 0, isRandom = true))
        try {
            val jsonString = assets.open("scenarios_canonical.json").bufferedReader().use { it.readText() }
            val array = org.json.JSONArray(jsonString)
            for (i in 0 until array.length()) {
                val obj = array.getJSONObject(i)
                list.add(
                    BenchmarkScenarioItem(
                        id = obj.getInt("scenario_id"),
                        trip = obj.getString("trip"),
                        domain = obj.getString("domain"),
                        durationS = obj.getInt("duration_s"),
                        distanceM = obj.getDouble("gt_dist_m").toInt(),
                        isRandom = false
                    )
                )
            }
        } catch (e: Exception) {
            android.util.Log.e("MainActivity", "Failed to load scenarios from assets: ${e.message}")
        }
        return list
    }
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

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIntent(intent)
    }

    private fun handleIntent(intent: Intent?) {
        val action = intent?.getStringExtra("action")
        if (action != null) {
            runOnUiThread {
                when (action.lowercase()) {
                    "stop" -> {
                        streamService?.stopBenchmark()
                        streamService?.stopBlackout()
                    }
                    "reset" -> {
                        streamService?.resetSession()
                    }
                    "live" -> {
                        cardBenchmark.visibility = View.GONE
                        streamService?.stopBenchmark()
                        currentMode = AppMode.LIVE_DRIVE
                        tvBenchmarkBadge.text = "LIVE SENSORS"
                        tvBenchmarkBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
                    }
                }
            }
        }

        val scId = intent?.getIntExtra("scenario", -1) ?: -1
        if (scId > 0) {
            val autoStart = intent?.getBooleanExtra("autostart", false) ?: false
            val speed = intent?.getDoubleExtra("speed", 2.0) ?: 2.0
            runOnUiThread {
                cardBenchmark.visibility = View.VISIBLE
                val targetPos = benchmarkScenarioList.indexOfFirst { it.id == scId }
                if (targetPos >= 0 && spinnerScenarios.selectedItemPosition != targetPos) {
                    currentPreparedScenarioId = scId
                    spinnerScenarios.setSelection(targetPos)
                }
                streamService?.prepareBenchmark(scId)
                if (autoStart) {
                    UiTraceLogger.reset()
                    gnssPolyline.actualPoints.clear()
                    drPolyline.actualPoints.clear()
                    hasCenteredMap = false
                    streamService?.startBenchmark(scId, speed)
                }
            }
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

        UiTraceLogger.init(this)
        initViews()
        initMap()
        checkAndRequestPermissions()
        startAndBindService()
        handleIntent(intent)
    }

    private fun initViews() {
        // Header & Connection
        cardConnection = findViewById(R.id.cardConnection)
        tvConnStatus = findViewById(R.id.tvConnStatus)
        etServerIp = findViewById(R.id.etServerIp)
        etServerPort = findViewById(R.id.etServerPort)
        btnConnect = findViewById(R.id.btnConnect)
        btnBenchmarkSuite = findViewById(R.id.btnBenchmarkSuite)

        // Map & Feature Controls
        cardMapControls = findViewById(R.id.cardMapControls)
        btnPrefetchArea = findViewById(R.id.btnPrefetchArea)
        btnToggleMapMatching = findViewById(R.id.btnToggleMapMatching)
        tvHandoffState = findViewById(R.id.tvHandoffState)

        // Autonomous On-Device Mode UI adjustments
        if (packageName.endsWith(".ondevice")) {
            cardConnection.visibility = View.GONE
            tvConnStatus.text = "Autonomous On-Device Engine"
            tvConnStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
        }

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
            UiTraceLogger.reset()
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

        // Map Controls Listeners
        btnPrefetchArea.setOnClickListener {
            val s = streamService
            if (s == null) {
                Toast.makeText(this, "Service not bound", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            val center = mapView.mapCenter
            Toast.makeText(this, "Prefetching road network (3km radius)...", Toast.LENGTH_SHORT).show()
            s.prefetchArea(center.latitude, center.longitude, 3000.0) { success, msg ->
                Toast.makeText(this, msg, if (success) Toast.LENGTH_SHORT else Toast.LENGTH_LONG).show()
            }
        }

        btnToggleMapMatching.setOnClickListener {
            val s = streamService ?: return@setOnClickListener
            isMapMatchingEnabled = !isMapMatchingEnabled
            s.setMapMatching(isMapMatchingEnabled)
            if (isMapMatchingEnabled) {
                btnToggleMapMatching.text = "MAP MATCH: ON"
                btnToggleMapMatching.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
                Toast.makeText(this, "Map Matching: ENABLED", Toast.LENGTH_SHORT).show()
            } else {
                btnToggleMapMatching.text = "MAP MATCH: OFF"
                btnToggleMapMatching.setTextColor(ContextCompat.getColor(this, R.color.accent_rose))
                Toast.makeText(this, "Map Matching: DISABLED", Toast.LENGTH_SHORT).show()
            }
        }

        btnStart.setOnClickListener {
            val s = streamService ?: return@setOnClickListener
            val w = latestWarmup
            val mountState = w?.mountState ?: "UNLEVELLED"
            val isLocked = mountState == "YAW_LOCKED" || mountState == "REUSED" || w?.mountLocked == true

            if (!isLocked) {
                val turns = w?.turnsDisplay ?: "turns: 0/15"
                AlertDialog.Builder(this)
                    .setTitle("Warning: Mount Not Locked")
                    .setMessage("Mount calibration has not achieved yaw lock yet ($turns, state: $mountState).\n\nDead reckoning heading accuracy may be reduced without full yaw lock. Do you want to proceed anyway?")
                    .setPositiveButton("Start Anyway") { _, _ ->
                        Log.i("PHONE", "User chose START before lock (state=$mountState, turns=$turns)")
                        executeStartBlackout(s)
                    }
                    .setNegativeButton("Wait for Lock") { _, _ ->
                        Log.i("PHONE", "User chose to wait for yaw lock (state=$mountState)")
                    }
                    .show()
            } else {
                executeStartBlackout(s)
            }
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

            // Clear map tracks and markers
            gnssPolyline.actualPoints.clear()
            drPolyline.actualPoints.clear()
            drMarker?.isEnabled = false
            reconciledMarker?.isEnabled = false
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

    private fun executeStartBlackout(s: SensorStreamService) {
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

    private fun initBenchmarkSpinners() {
        if (benchmarkScenarioList.isEmpty()) {
            benchmarkScenarioList.addAll(loadCanonicalScenariosFromAssets())
        }
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
                        if (currentPreparedScenarioId == scId) {
                            return
                        }
                        currentPreparedScenarioId = scId
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

        // Dead Reckoning track (amber / gold for clear visual contrast with GNSS track)
        drPolyline = Polyline(mapView)
        drPolyline.outlinePaint.color = Color.parseColor("#F59E0B")
        drPolyline.outlinePaint.strokeWidth = 10f
        drPolyline.outlinePaint.strokeCap = Paint.Cap.ROUND
        mapView.overlays.add(drPolyline)

        // Vehicle GNSS position marker (emerald pointer)
        vehicleMarker = Marker(mapView)
        vehicleMarker?.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        try {
            vehicleMarker?.icon = ContextCompat.getDrawable(this, R.drawable.ic_gnss_marker)
        } catch (_: Exception) {}
        mapView.overlays.add(vehicleMarker)

        // Dead Reckoning vehicle marker (amber pointer)
        drMarker = Marker(mapView)
        drMarker?.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        try {
            drMarker?.icon = ContextCompat.getDrawable(this, R.drawable.ic_dr_marker)
        } catch (_: Exception) {}
        drMarker?.isEnabled = false
        mapView.overlays.add(drMarker)

        // Reconciled / Blended handoff marker (cyan diamond)
        reconciledMarker = Marker(mapView)
        reconciledMarker?.setAnchor(Marker.ANCHOR_CENTER, Marker.ANCHOR_CENTER)
        try {
            reconciledMarker?.icon = ContextCompat.getDrawable(this, R.drawable.ic_reconciled_marker)
        } catch (_: Exception) {}
        reconciledMarker?.isEnabled = false
        mapView.overlays.add(reconciledMarker)

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
                    handleIntent(intent)
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
                        val accuracyM = gnss.accuracyHM ?: 100.0f
                        // Real vehicle motion requires speed > 1.2 m/s (~4.3 km/h) and acceptable accuracy (<= 35m).
                        // Stationary phone on a desk experiences indoor multipath hops where Android calculates
                        // false speeds of 0.5 - 1.0 m/s — this strict gate locks the marker in place.
                        val isMoving = speedMps > 1.2 && accuracyM <= 35.0f

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
                            UiTraceLogger.logMarker("gnss", pt.latitude, pt.longitude, vehicleMarker?.rotation?.toDouble() ?: 0.0, true)
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
                currentPreparedScenarioId = scId
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

            UiTraceLogger.logHud("speed", tvSpeeds.text.toString(), m.drSpeedMps)
            UiTraceLogger.logHud("heading", tvHeadings.text.toString(), m.drHeadingDeg)
            UiTraceLogger.logHud("drift", tvDriftPct.text.toString(), m.driftPct)
            UiTraceLogger.logHud("turns", tvCondMount.text.toString(), null)
            UiTraceLogger.logHud("calib_s", tvCondBuffer.text.toString(), null)

            // If session summary received while stopped, show summary card unless dismissed
            if (!isInBlackout && m.sessionSummary != null && !isSummaryDismissed) {
                renderSummary(m.sessionSummary)
            }
        }

        // Map updates:
        if (currentMode == AppMode.LIVE_DRIVE) {
            if (isInBlackout) {
                hud.drPos?.let { d ->
                    if (d.lat != 0.0 && d.lon != 0.0) {
                        val pt = GeoPoint(d.lat, d.lon)
                        val lastPt = drPolyline.actualPoints.lastOrNull()
                        val distToLast = if (lastPt != null) pt.distanceToAsDouble(lastPt) else 1000.0
                        if (distToLast > 500.0) {
                            drPolyline.actualPoints.clear()
                            drPolyline.addPoint(pt)
                        } else if (distToLast >= 0.5) {
                            drPolyline.addPoint(pt)
                        }
                        drMarker?.position = pt
                        drMarker?.rotation = MarkerHeading.toMarkerRotation(d.headingDeg)
                        drMarker?.isEnabled = true
                        UiTraceLogger.logMarker("dr", pt.latitude, pt.longitude, drMarker?.rotation?.toDouble() ?: 0.0, true)

                        tilePrefetcher?.onMotionUpdate(
                            d.lat,
                            d.lon,
                            hud.metrics?.drSpeedMps ?: 0.0,
                            d.headingDeg ?: 0.0
                        )
                        mapView.controller.animateTo(pt)
                    }
                }

                hud.reconciledPos?.let { r ->
                    val hState = hud.warmup?.handoffState ?: ""
                    if (r.lat != 0.0 && r.lon != 0.0 && (hState == "REACQUISITION_VERIFY" || hState == "REACQUISITION_BLENDING")) {
                        val rPt = GeoPoint(r.lat, r.lon)
                        reconciledMarker?.position = rPt
                        reconciledMarker?.isEnabled = true
                        UiTraceLogger.logMarker("reconciled", rPt.latitude, rPt.longitude, 0.0, true)
                    } else {
                        reconciledMarker?.isEnabled = false
                        UiTraceLogger.logMarker("reconciled", 0.0, 0.0, 0.0, false)
                    }
                } ?: run {
                    reconciledMarker?.isEnabled = false
                    UiTraceLogger.logMarker("reconciled", 0.0, 0.0, 0.0, false)
                }
            } else {
                drMarker?.isEnabled = false
                UiTraceLogger.logMarker("dr", drMarker?.position?.latitude ?: 0.0, drMarker?.position?.longitude ?: 0.0, drMarker?.rotation?.toDouble() ?: 0.0, false)
                reconciledMarker?.isEnabled = false
                UiTraceLogger.logMarker("reconciled", 0.0, 0.0, 0.0, false)
            }
        } else if (currentMode == AppMode.BENCHMARK_EVALUATION) {
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
                    vehicleMarker?.position = pt
                    vehicleMarker?.rotation = MarkerHeading.toMarkerRotation(g.bearingDeg)
                    vehicleMarker?.isEnabled = true
                    UiTraceLogger.logMarker("gnss", pt.latitude, pt.longitude, vehicleMarker?.rotation?.toDouble() ?: 0.0, true)
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
                    drMarker?.position = pt
                    drMarker?.rotation = MarkerHeading.toMarkerRotation(d.headingDeg)
                    drMarker?.isEnabled = true
                    UiTraceLogger.logMarker("dr", pt.latitude, pt.longitude, drMarker?.rotation?.toDouble() ?: 0.0, true)
                }
            }

            hud.reconciledPos?.let { r ->
                val hState = hud.warmup?.handoffState ?: ""
                if (r.lat != 0.0 && r.lon != 0.0 && (hState == "REACQUISITION_VERIFY" || hState == "REACQUISITION_BLENDING")) {
                    val rPt = GeoPoint(r.lat, r.lon)
                    reconciledMarker?.position = rPt
                    reconciledMarker?.isEnabled = true
                    UiTraceLogger.logMarker("reconciled", rPt.latitude, rPt.longitude, 0.0, true)
                } else {
                    reconciledMarker?.isEnabled = false
                    UiTraceLogger.logMarker("reconciled", 0.0, 0.0, 0.0, false)
                }
            } ?: run {
                reconciledMarker?.isEnabled = false
                UiTraceLogger.logMarker("reconciled", 0.0, 0.0, 0.0, false)
            }
        }
        mapView.invalidate()
    }

    private fun updateWarmupPanel(w: WarmupStatus) {
        latestWarmup = w
        val emerald = ContextCompat.getColor(this, R.color.accent_emerald)
        val muted = ContextCompat.getColor(this, R.color.text_muted)
        val amber = ContextCompat.getColor(this, R.color.accent_amber)
        val rose = ContextCompat.getColor(this, R.color.accent_rose)
        val cyan = ContextCompat.getColor(this, R.color.accent_cyan)
        val isBenchmark = (currentMode == AppMode.BENCHMARK_EVALUATION)

        // 1. Gravity Leveling — phone held still for 3s so accelerometer baseline settles
        if (w.gravityConverged) {
            tvCondGravity.text = "✓ Gravity"
            tvCondGravity.setTextColor(emerald)
        } else {
            tvCondGravity.text = "✗ Gravity (hold still)"
            tvCondGravity.setTextColor(muted)
        }

        // 2. Mount Calibration: 15-turn target
        val isMountLocked = w.mountLocked || w.mountState == "YAW_LOCKED" || w.mountState == "REUSED"
        if (isMountLocked) {
            val label = when {
                isBenchmark && w.mountStatus.contains("reused", ignoreCase = true) -> "✓ Bench-Calib"
                w.mountState == "REUSED" || w.mountStatus.contains("reused", ignoreCase = true) -> "✓ Mount Reused"
                else -> "✓ ${w.turnsDisplay}"
            }
            tvCondMount.text = label
            tvCondMount.setTextColor(emerald)
        } else {
            tvCondMount.text = "✗ ${w.turnsDisplay}"
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

        // 4. Speed Calibration: Speed calibration n/180 s
        val calibDone = w.alphaLearned || w.speedCalibS >= 180
        if (calibDone) {
            tvCondAlpha.text = "✓ Speed calib ${w.speedCalibS}/180s"
            tvCondAlpha.setTextColor(emerald)
        } else {
            tvCondAlpha.text = "✗ Speed calib ${w.speedCalibS}/180s"
            tvCondAlpha.setTextColor(muted)
        }

        // Detail status line: Mount state + Handoff state
        tvMountStatusDetail.text = "Mount: ${w.mountState} (${w.turnsDisplay}) | Handoff: ${w.handoffState}"
        if (isMountLocked) {
            tvMountStatusDetail.setTextColor(emerald)
        } else {
            tvMountStatusDetail.setTextColor(cyan)
        }

        // Handoff state indicator
        tvHandoffState.text = "FSM: ${w.handoffState}"
        val handoffColor = when (w.handoffState) {
            "GNSS_HEALTHY" -> emerald
            "INS_DEAD_RECKONING" -> rose
            "REACQUISITION_VERIFY" -> cyan
            "REACQUISITION_BLENDING" -> ContextCompat.getColor(this, R.color.accent_blue)
            "GNSS_DEGRADED" -> amber
            else -> muted
        }
        tvHandoffState.setTextColor(handoffColor)

        // Synchronize map matching toggle button
        isMapMatchingEnabled = w.mapMatchingEnabled
        btnToggleMapMatching.text = if (w.mapMatchingEnabled) "MAP MATCH: ON" else "MAP MATCH: OFF"
        btnToggleMapMatching.setTextColor(if (w.mapMatchingEnabled) emerald else rose)

        // Headline
        if (w.isReady) {
            tvReadyHeadline.text = if (isBenchmark) "BENCHMARK: ENGINE READY" else "WARM-UP STATUS: READY TO START"
            tvReadyHeadline.setTextColor(emerald)
        } else {
            tvReadyHeadline.text = if (isBenchmark) "BENCHMARK: WARMING ENGINE..." else "WARM-UP STATUS: WAITING FOR READY"
            tvReadyHeadline.setTextColor(amber)
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
                        items.add(
                            BenchmarkScenarioItem(
                                id = id,
                                trip = trip,
                                domain = domain,
                                durationS = dur,
                                distanceM = dist,
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

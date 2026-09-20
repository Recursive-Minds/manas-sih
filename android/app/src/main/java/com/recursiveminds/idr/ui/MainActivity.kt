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
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.preference.PreferenceManager
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
import com.recursiveminds.idr.service.SensorStreamService
import org.osmdroid.config.Configuration
import org.osmdroid.tileprovider.tilesource.TileSourceFactory
import org.osmdroid.util.GeoPoint
import org.osmdroid.views.MapView
import org.osmdroid.views.overlay.Marker
import org.osmdroid.views.overlay.Polyline

class MainActivity : AppCompatActivity() {

    private var streamService: SensorStreamService? = null
    private var isBound = false

    // Header & Connection
    private lateinit var tvConnStatus: TextView
    private lateinit var etServerIp: EditText
    private lateinit var etServerPort: EditText
    private lateinit var btnConnect: Button

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

    // Session Summary Overlay
    private lateinit var cardSummaryModal: LinearLayout
    private lateinit var tvSummaryDrift: TextView
    private lateinit var tvSummaryTier: TextView
    private lateinit var tvSummaryDetails: TextView
    private lateinit var btnDismissSummary: Button

    // Bottom Bar
    private lateinit var btnShareCsv: Button
    private lateinit var tvSampleRate: TextView

    // State Tracking
    private var latestMetrics: LiveMetrics? = null
    private var isEngineReady = false
    private var isInBlackout = false

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

        // Initialize OSMDroid configuration
        Configuration.getInstance().load(this, PreferenceManager.getDefaultSharedPreferences(this))
        Configuration.getInstance().userAgentValue = packageName

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
        btnShareCsv = findViewById(R.id.btnShareCsv)
        tvSampleRate = findViewById(R.id.tvSampleRate)

        // Listeners
        btnConnect.setOnClickListener {
            val s = streamService ?: return@setOnClickListener
            if (s.isConnected.get()) {
                s.disconnectServer()
            } else {
                val ip = etServerIp.text.toString().trim()
                val port = etServerPort.text.toString().trim().toIntOrNull() ?: 8765
                s.connectServer(ip, port)
            }
        }

        btnStart.setOnClickListener {
            val s = streamService ?: return@setOnClickListener
            s.startBlackout()
            isInBlackout = true
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
            updateControlButtons()
            tvStateBadge.text = "WARMING UP"
            tvStateBadge.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
            cardSummaryModal.visibility = View.GONE

            // Clear map tracks
            gnssPolyline.actualPoints.clear()
            drPolyline.actualPoints.clear()
            mapView.invalidate()

            // Reset labels
            tvDriftPct.text = "0.00%"
            tvHorizError.text = "0.0 m"
            tvMaxError.text = "0.0 m"
            tvDistances.text = "0 / 0 m"
            tvAlongCross.text = "0.0 / 0.0 m"
            tvSpeeds.text = "0.0 / 0.0"
            tvHeadings.text = "0° / 0°"
            Toast.makeText(this, "Session reset", Toast.LENGTH_SHORT).show()
        }

        btnDismissSummary.setOnClickListener {
            cardSummaryModal.visibility = View.GONE
        }

        btnShareCsv.setOnClickListener {
            shareCsvFile()
        }
    }

    private fun initMap() {
        mapView.setTileSource(TileSourceFactory.MAPNIK)
        mapView.setMultiTouchControls(true)
        mapView.controller.setZoom(17.0)

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
    }

    private fun setupServiceCallbacks(s: SensorStreamService) {
        s.onConnectionStateChanged = { connected ->
            runOnUiThread {
                if (connected) {
                    tvConnStatus.text = "Connected"
                    tvConnStatus.setTextColor(ContextCompat.getColor(this, R.color.accent_emerald))
                    btnConnect.text = "Disconnect"
                    btnConnect.backgroundTintList = ColorStateList.valueOf(ContextCompat.getColor(this, R.color.card_border))
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

        s.onLocalGnssUpdate = { gnss ->
            runOnUiThread {
                if (gnss.latitudeDeg != 0.0 && gnss.longitudeDeg != 0.0) {
                    val pt = GeoPoint(gnss.latitudeDeg, gnss.longitudeDeg)
                    tvGpsAccuracy.text = String.format("±%.1f m", gnss.accuracyHM)
                    if (!s.isConnected.get()) {
                        gnssPolyline.addPoint(pt)
                        if (!hasCenteredMap) {
                            mapView.controller.animateTo(pt)
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
            btnStart.isEnabled = isEngineReady
            val startColor = if (isEngineReady) {
                ContextCompat.getColor(this, R.color.accent_emerald)
            } else {
                Color.parseColor("#475569")
            }
            btnStart.backgroundTintList = ColorStateList.valueOf(startColor)
        }
    }

    private fun updateHudUi(hud: HudUpdate) {
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

            // If session summary received while stopped, show summary card
            if (!isInBlackout && m.sessionSummary != null) {
                renderSummary(m.sessionSummary)
            }
        }

        // Map updates
        hud.gnssPos?.let { g ->
            if (g.lat != 0.0 && g.lon != 0.0) {
                val pt = GeoPoint(g.lat, g.lon)
                gnssPolyline.addPoint(pt)
                if (!hasCenteredMap) {
                    mapView.controller.animateTo(pt)
                    hasCenteredMap = true
                }
            }
        }

        hud.drPos?.let { d ->
            if (d.lat != 0.0 && d.lon != 0.0) {
                val pt = GeoPoint(d.lat, d.lon)
                drPolyline.addPoint(pt)
                vehicleMarker?.position = pt
                vehicleMarker?.rotation = d.headingDeg?.toFloat() ?: 0f
            }
        }
        mapView.invalidate()
    }

    private fun updateWarmupPanel(w: WarmupStatus) {
        val emerald = ContextCompat.getColor(this, R.color.accent_emerald)
        val muted = ContextCompat.getColor(this, R.color.text_muted)

        // 1. Gravity Leveling
        if (w.gravityConverged) {
            tvCondGravity.text = "✓ Gravity"
            tvCondGravity.setTextColor(emerald)
        } else {
            tvCondGravity.text = "✗ Gravity"
            tvCondGravity.setTextColor(muted)
        }

        // 2. Mount Calibration
        if (w.mountLocked) {
            val label = if (w.mountStatus.contains("reused", ignoreCase = true)) "✓ Reused" else "✓ ${w.turnsDisplay}"
            tvCondMount.text = label
            tvCondMount.setTextColor(emerald)
        } else {
            tvCondMount.text = "✗ ${w.turnsDisplay}"
            tvCondMount.setTextColor(muted)
        }

        // 3. Macro Feature Buffer
        if (w.bufferWarm) {
            tvCondBuffer.text = "✓ Buffer 6s"
            tvCondBuffer.setTextColor(emerald)
        } else {
            tvCondBuffer.text = "✗ Buffer"
            tvCondBuffer.setTextColor(muted)
        }

        // 4. Alpha Adaptive Scaling
        if (w.alphaLearned) {
            tvCondAlpha.text = "✓ Alpha"
            tvCondAlpha.setTextColor(emerald)
        } else {
            tvCondAlpha.text = "✗ Alpha"
            tvCondAlpha.setTextColor(muted)
        }

        // Headline
        if (w.isReady) {
            tvReadyHeadline.text = "WARM-UP STATUS: READY TO START"
            tvReadyHeadline.setTextColor(emerald)
        } else {
            tvReadyHeadline.text = "WARM-UP STATUS: WAITING FOR READY"
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

package com.recursiveminds.idr.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Binder
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.google.gson.Gson
import com.recursiveminds.idr.data.*
import kotlinx.coroutines.*
import okhttp3.*
import java.io.File
import java.io.FileWriter
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class SensorStreamService : Service(), SensorEventListener, LocationListener {

    private val binder = LocalBinder()
    private val gson = Gson()
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private lateinit var sensorManager: SensorManager
    private lateinit var locationManager: LocationManager

    private var accelSensor: Sensor? = null
    private var gyroSensor: Sensor? = null

    // Latest raw readings
    @Volatile private var latestAx = 0.0
    @Volatile private var latestAy = 0.0
    @Volatile private var latestAz = 9.81
    @Volatile private var latestGx = 0.0
    @Volatile private var latestGy = 0.0
    @Volatile private var latestGz = 0.0

    private val imuQueue = ConcurrentLinkedQueue<ImuPoint>()
    private val gnssQueue = ConcurrentLinkedQueue<GnssPoint>()

    // Networking
    private var okHttpClient: OkHttpClient? = null
    private var webSocket: WebSocket? = null
    val isConnected = AtomicBoolean(false)
    val isBlackout = AtomicBoolean(false)

    // CSV logging: explicit Start/Stop controlled by user
    private var csvFile: File? = null
    private var csvWriter: FileWriter? = null
    val isCsvRecording = AtomicBoolean(false)

    // Callbacks to Activity
    var onHudUpdateListener: ((HudUpdate) -> Unit)? = null
    var onConnectionStateChanged: ((Boolean) -> Unit)? = null
    var onSensorRateUpdate: ((Int, Int) -> Unit)? = null
    var onLocalGnssUpdate: ((GnssPoint) -> Unit)? = null
    var onCsvStateChanged: ((recording: Boolean, filePath: String?) -> Unit)? = null

    // Diagnostics
    private var imuCount = 0
    private var gnssCount = 0
    private var lastRateCalcTime = System.currentTimeMillis()

    inner class LocalBinder : Binder() {
        fun getService(): SensorStreamService = this@SensorStreamService
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        startForegroundNotification()

        sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        locationManager = getSystemService(Context.LOCATION_SERVICE) as LocationManager

        accelSensor = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

        // Do NOT auto-start CSV — user controls it explicitly via REC button
        registerSensors()
        startBatchDispatcher()
    }

    private fun startForegroundNotification() {
        val channelId = "idr_stream_channel"
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                channelId,
                "IDR Sensor Streaming",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }

        val notification: Notification = NotificationCompat.Builder(this, channelId)
            .setContentTitle("SIH IDR Service")
            .setContentText("Streaming IMU & GNSS sensor telemetry")
            .setSmallIcon(android.R.drawable.ic_dialog_map)
            .setOngoing(true)
            .build()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(101, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_LOCATION)
        } else {
            startForeground(101, notification)
        }
    }

    private fun registerSensors() {
        accelSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
        gyroSensor?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }

        try {
            if (locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.GPS_PROVIDER,
                    1000L,
                    0.0f,
                    this
                )
            }
            if (locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER)) {
                locationManager.requestLocationUpdates(
                    LocationManager.NETWORK_PROVIDER,
                    1000L,
                    0.0f,
                    this
                )
            }

            // Immediately emit best last known location on startup
            val lastGps = locationManager.getLastKnownLocation(LocationManager.GPS_PROVIDER)
            val lastNet = locationManager.getLastKnownLocation(LocationManager.NETWORK_PROVIDER)
            val bestLast = when {
                lastGps != null && lastNet != null -> if (lastGps.time >= lastNet.time) lastGps else lastNet
                lastGps != null -> lastGps
                else -> lastNet
            }
            bestLast?.let { onLocationChanged(it) }
        } catch (e: SecurityException) {
            Log.e("IDRService", "Location permission missing: ${e.message}")
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        val ts = event.timestamp
        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            latestAx = event.values[0].toDouble()
            latestAy = event.values[1].toDouble()
            latestAz = event.values[2].toDouble()
        } else if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            latestGx = event.values[0].toDouble()
            latestGy = event.values[1].toDouble()
            latestGz = event.values[2].toDouble()
        }

        val imu = ImuPoint(
            timestampNs = ts,
            accel = doubleArrayOf(latestAx, latestAy, latestAz),
            gyro = doubleArrayOf(latestGx, latestGy, latestGz)
        )
        imuQueue.offer(imu)
        imuCount++

        logImuToCsv(imu)
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    override fun onLocationChanged(location: Location) {
        val ts = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.JELLY_BEAN_MR1) {
            location.elapsedRealtimeNanos
        } else {
            location.time * 1_000_000L
        }

        val gnss = GnssPoint(
            timestampNs = ts,
            latitudeDeg = location.latitude,
            longitudeDeg = location.longitude,
            altitudeM = location.altitude,
            speedMps = if (location.hasSpeed()) location.speed else null,
            bearingDeg = if (location.hasBearing()) location.bearing else null,
            accuracyHM = if (location.hasAccuracy()) location.accuracy else 5.0f,
            isValid = location.accuracy <= 50.0f
        )
        gnssQueue.offer(gnss)
        gnssCount++

        logGnssToCsv(gnss)

        scope.launch(Dispatchers.Main) {
            onLocalGnssUpdate?.invoke(gnss)
        }
    }

    override fun onStatusChanged(provider: String?, status: Int, extras: Bundle?) {}
    override fun onProviderEnabled(provider: String) {}
    override fun onProviderDisabled(provider: String) {}

    fun connectServer(serverIp: String, port: Int) {
        disconnectServer()

        val wsUrl = "ws://$serverIp:$port/ws/stream"
        Log.i("IDRService", "Attempting WebSocket connection to: $wsUrl")
        val request = Request.Builder().url(wsUrl).build()

        okHttpClient = OkHttpClient.Builder()
            .proxy(java.net.Proxy.NO_PROXY)
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .build()

        webSocket = okHttpClient?.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i("IDRService", "WebSocket connection opened successfully!")
                isConnected.set(true)
                scope.launch(Dispatchers.Main) {
                    onConnectionStateChanged?.invoke(true)
                }
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val update = gson.fromJson(text, HudUpdate::class.java)
                    scope.launch(Dispatchers.Main) {
                        onHudUpdateListener?.invoke(update)
                    }
                } catch (e: Exception) {
                    Log.e("IDRService", "Failed to parse HUD update: ${e.message}")
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i("IDRService", "WebSocket closed: code=$code reason=$reason")
                isConnected.set(false)
                scope.launch(Dispatchers.Main) {
                    onConnectionStateChanged?.invoke(false)
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("IDRService", "WebSocket onFailure: ${t.message}", t)
                isConnected.set(false)
                scope.launch(Dispatchers.Main) {
                    onConnectionStateChanged?.invoke(false)
                }
            }
        })
    }

    fun disconnectServer() {
        webSocket?.close(1000, "User disconnected")
        webSocket = null
        isConnected.set(false)
        onConnectionStateChanged?.invoke(false)
    }

    fun startBlackout() {
        isBlackout.set(true)
        sendControl("start_blackout")
    }

    fun stopBlackout() {
        isBlackout.set(false)
        sendControl("stop_blackout")
    }

    fun resetSession() {
        isBlackout.set(false)
        sendControl("reset")
    }

    val isMuted = AtomicBoolean(false)

    fun setSensorStreamingMuted(muted: Boolean) {
        isMuted.set(muted)
        Log.i("IDRService", "Sensor streaming muted=$muted")
    }

    fun startBenchmark(scenarioId: Int, speed: Double) {
        setSensorStreamingMuted(true)
        val msg = ControlMessage(
            command = "start_benchmark",
            scenarioId = scenarioId,
            speed = speed
        )
        webSocket?.send(gson.toJson(msg))
    }

    fun stopBenchmark() {
        val msg = ControlMessage(command = "stop_benchmark")
        webSocket?.send(gson.toJson(msg))
        setSensorStreamingMuted(false)
    }

    private fun sendControl(command: String) {
        val msg = ControlMessage(command = command)
        webSocket?.send(gson.toJson(msg))
    }

    private fun startBatchDispatcher() {
        scope.launch {
            while (isActive) {
                delay(100L) // 100 ms batches = 10 Hz telemetry packets

                // Drain queues
                val imuList = mutableListOf<ImuPoint>()
                while (!imuQueue.isEmpty()) {
                    imuQueue.poll()?.let { imuList.add(it) }
                }

                val gnssList = mutableListOf<GnssPoint>()
                while (!gnssQueue.isEmpty()) {
                    gnssQueue.poll()?.let { gnssList.add(it) }
                }

                if ((imuList.isNotEmpty() || gnssList.isNotEmpty()) && !isMuted.get()) {
                    val stateStr = if (isBlackout.get()) "BLACKOUT" else "WARMING_UP"
                    val batch = SensorBatch(
                        source = "device",
                        state = stateStr,
                        timestampNs = System.nanoTime(),
                        imu = imuList,
                        gnss = gnssList
                    )
                    val json = gson.toJson(batch)
                    webSocket?.send(json)
                }

                // Update rate telemetry once per second
                val now = System.currentTimeMillis()
                if (now - lastRateCalcTime >= 1000L) {
                    val dt = (now - lastRateCalcTime) / 1000.0
                    val imuRate = (imuCount / dt).toInt()
                    val gnssRate = (gnssCount / dt).toInt()
                    imuCount = 0
                    gnssCount = 0
                    lastRateCalcTime = now
                    scope.launch(Dispatchers.Main) {
                        onSensorRateUpdate?.invoke(imuRate, gnssRate)
                    }
                }
            }
        }
    }

    fun startCsvRecording() {
        if (isCsvRecording.get()) return
        try {
            val dir = getExternalFilesDir(null) ?: filesDir
            val ts = System.currentTimeMillis()
            val file = File(dir, "idr_telemetry_$ts.csv")
            csvFile = file
            csvWriter = FileWriter(file)
            csvWriter?.append("type,timestamp_ns,val1,val2,val3,val4,val5,val6\n")
            csvWriter?.flush()
            isCsvRecording.set(true)
            Log.i("IDRService", "CSV recording started: ${file.absolutePath}")
            scope.launch(Dispatchers.Main) {
                onCsvStateChanged?.invoke(true, file.absolutePath)
            }
        } catch (e: Exception) {
            Log.e("IDRService", "Error starting CSV recording: ${e.message}")
        }
    }

    fun stopCsvRecording() {
        if (!isCsvRecording.get()) return
        try {
            csvWriter?.flush()
            csvWriter?.close()
            csvWriter = null
            isCsvRecording.set(false)
            Log.i("IDRService", "CSV recording stopped. File: ${csvFile?.absolutePath}")
            scope.launch(Dispatchers.Main) {
                onCsvStateChanged?.invoke(false, csvFile?.absolutePath)
            }
        } catch (e: Exception) {
            Log.e("IDRService", "Error stopping CSV recording: ${e.message}")
        }
    }

    fun getCurrentCsvFile(): File? = csvFile

    fun flushCsv() {
        try {
            csvWriter?.flush()
        } catch (e: Exception) {}
    }

    private fun logImuToCsv(imu: ImuPoint) {
        if (!isCsvRecording.get()) return
        try {
            csvWriter?.append("IMU,${imu.timestampNs},${imu.accel[0]},${imu.accel[1]},${imu.accel[2]},${imu.gyro[0]},${imu.gyro[1]},${imu.gyro[2]}\n")
        } catch (e: Exception) {}
    }

    private fun logGnssToCsv(gnss: GnssPoint) {
        if (!isCsvRecording.get()) return
        try {
            csvWriter?.append("GNSS,${gnss.timestampNs},${gnss.latitudeDeg},${gnss.longitudeDeg},${gnss.altitudeM},${gnss.speedMps ?: 0.0},${gnss.bearingDeg ?: 0.0},${gnss.accuracyHM}\n")
            csvWriter?.flush()
        } catch (e: Exception) {}
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorManager.unregisterListener(this)
        locationManager.removeUpdates(this)
        disconnectServer()
        try {
            csvWriter?.flush()
            csvWriter?.close()
        } catch (e: Exception) {}
        scope.cancel()
    }
}

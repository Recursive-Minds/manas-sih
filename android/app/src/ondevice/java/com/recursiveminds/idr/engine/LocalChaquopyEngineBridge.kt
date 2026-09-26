package com.recursiveminds.idr.engine

import android.content.Context
import android.util.Log
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.google.gson.Gson
import com.recursiveminds.idr.data.ControlMessage
import com.recursiveminds.idr.data.HudUpdate
import com.recursiveminds.idr.data.SensorBatch
import com.recursiveminds.idr.inference.TFLitePredictorBridge
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class LocalChaquopyEngineBridge(private val context: Context) : IEngineBridge {

    private val gson = Gson()
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    // Dedicated single-thread worker for local dead reckoning engine execution
    private val executor = Executors.newSingleThreadExecutor { Thread(it, "IDROnDeviceEngine") }
    private val batchQueue = ConcurrentLinkedQueue<SensorBatch>()
    private val isDraining = AtomicBoolean(false)

    private val _isConnected = AtomicBoolean(false)
    private val _isBlackout = AtomicBoolean(false)

    override val isConnected: Boolean get() = _isConnected.get()
    override val isBlackout: Boolean get() = _isBlackout.get()

    override var onHudUpdateListener: ((HudUpdate) -> Unit)? = null
    override var onConnectionStateChanged: ((Boolean) -> Unit)? = null

    private var py: Python? = null
    private var sessionCore: PyObject? = null
    private var pyJson: PyObject? = null
    private var bridge: TFLitePredictorBridge? = null

    init {
        // Features and model run from app start
        initializeLocalEngine()
    }

    private fun initializeLocalEngine() {
        executor.execute {
            try {
                Log.i("LocalEngineBridge", "Initializing local edge Python & TFLite runtime...")
                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(context.applicationContext))
                }
                py = Python.getInstance()
                pyJson = py?.getModule("json")

                // 1. TFLite Predictor Bridge (XNNPACK, 4 threads, warm-up)
                bridge = TFLitePredictorBridge.create(context.applicationContext, "moe_velocity_model.tflite")
                val predictorModule = py?.getModule("sih.models.predictor")
                val pyPredictor = predictorModule?.callAttr("JavaBridgeVelocityPredictor", bridge, true)

                // 2. Instantiate SessionCore (runs features & model from start)
                val sessionModule = py?.getModule("server.session_core")
                sessionCore = sessionModule?.callAttr(
                    "SessionCore",
                    0.0, 0.0, 0.0, // ref coords (updated when first GNSS fix arrives)
                    null, // saved_alignment
                    null, // road_network
                    "Mixed", // domain
                    "stage_b", // engine_type
                    null, null, null, null,
                    true, // use_speed_smoother
                    pyPredictor,
                    true // enable_handoff (display only)
                )

                _isConnected.set(true)
                Log.i("LocalEngineBridge", "Local SessionCore initialized successfully")

                scope.launch(Dispatchers.Main) {
                    onConnectionStateChanged?.invoke(true)
                }
            } catch (e: Throwable) {
                Log.e("LocalEngineBridge", "Failed to initialize local engine: ${e.message}", e)
                _isConnected.set(false)
                scope.launch(Dispatchers.Main) {
                    onConnectionStateChanged?.invoke(false)
                }
            }
        }
    }

    override fun connect(serverIp: String, port: Int) {
        // In on-device mode, connection is local
        if (sessionCore == null) {
            initializeLocalEngine()
        } else {
            _isConnected.set(true)
            scope.launch(Dispatchers.Main) {
                onConnectionStateChanged?.invoke(true)
            }
        }
    }

    override fun disconnect() {
        _isConnected.set(false)
        scope.launch(Dispatchers.Main) {
            onConnectionStateChanged?.invoke(false)
        }
    }

    override fun startBlackout() {
        _isBlackout.set(true)
        executor.execute {
            sessionCore?.callAttr("set_blackout", true)
        }
    }

    override fun stopBlackout() {
        _isBlackout.set(false)
        executor.execute {
            sessionCore?.callAttr("set_blackout", false)
        }
    }

    override fun resetSession() {
        _isBlackout.set(false)
        executor.execute {
            sessionCore?.callAttr("reset", false)
        }
    }

    override fun pushBatch(batch: SensorBatch) {
        // Enqueue batch - NEVER drop IMU samples
        batchQueue.offer(batch)
        val backlog = batchQueue.size
        if (backlog > 1) {
            Log.w("PHONE", "Batch backlog queue size: $backlog (batch took > 100ms, queueing without dropping)")
        }

        drainQueue()
    }

    private fun drainQueue() {
        if (isDraining.compareAndSet(false, true)) {
            executor.execute {
                try {
                    while (true) {
                        val batch = batchQueue.poll() ?: break
                        val remaining = batchQueue.size
                        if (remaining > 2) {
                            Log.w("PHONE", "Processing backlog batch. Remaining in queue: $remaining")
                        }

                        val sc = sessionCore
                        val pj = pyJson
                        if (sc != null && pj != null) {
                            val batchJson = gson.toJson(batch)
                            val batchDict = pj.callAttr("loads", batchJson)
                            val hudDict = sc.callAttr("push_batch", batchDict, batch.source)
                            val hudJson = pj.callAttr("dumps", hudDict).toString()

                            try {
                                val hud = gson.fromJson(hudJson, HudUpdate::class.java)
                                scope.launch(Dispatchers.Main) {
                                    onHudUpdateListener?.invoke(hud)
                                }
                            } catch (e: Exception) {
                                Log.e("LocalEngineBridge", "Failed to deserialize HUD update: ${e.message}")
                            }
                        }
                    }
                } catch (e: Throwable) {
                    Log.e("LocalEngineBridge", "Error in drainQueue: ${e.message}", e)
                } finally {
                    isDraining.set(false)
                    // Check if more batches arrived while finishing
                    if (!batchQueue.isEmpty()) {
                        drainQueue()
                    }
                }
            }
        }
    }

    override fun setMapMatching(enabled: Boolean) {
        executor.execute {
            try {
                sessionCore?.callAttr("set_map_matching", enabled)
                Log.i("PHONE", "Map matching set to: $enabled")
            } catch (e: Throwable) {
                Log.e("PHONE", "Failed to set map matching: ${e.message}")
            }
        }
    }

    override fun prefetchArea(lat: Double, lon: Double, radiusM: Double, callback: (Boolean, String) -> Unit) {
        executor.execute {
            try {
                Log.i("PHONE", "Prefetching road network area at lat=$lat, lon=$lon, radius=$radiusM")
                val res = sessionCore?.callAttr("prefetch_road_network", lat, lon, radiusM)
                val success = res?.callAttr("get", "success")?.toBoolean() ?: false
                val msg = res?.callAttr("get", "message")?.toString() ?: "Unknown result"
                scope.launch(Dispatchers.Main) {
                    callback(success, msg)
                }
            } catch (e: Throwable) {
                Log.e("PHONE", "Prefetch failed: ${e.message}", e)
                scope.launch(Dispatchers.Main) {
                    callback(false, "Prefetch error: ${e.message}")
                }
            }
        }
    }

    override fun prepareBenchmark(scenarioId: Int) {
        executor.execute {
            try {
                // In on-device mode, benchmark data can be preloaded from assets
                Log.i("PHONE", "Preparing benchmark scenario $scenarioId on-device")
            } catch (e: Throwable) {
                Log.e("PHONE", "Failed to prepare benchmark: ${e.message}")
            }
        }
    }

    override fun startBenchmark(scenarioId: Int, speed: Double) {
        Log.i("PHONE", "Starting benchmark scenario $scenarioId on-device")
    }

    override fun stopBenchmark() {
        Log.i("PHONE", "Stopping benchmark on-device")
    }
}

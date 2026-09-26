package com.recursiveminds.idr.engine

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.recursiveminds.idr.data.ControlMessage
import com.recursiveminds.idr.data.HudUpdate
import com.recursiveminds.idr.data.SensorBatch
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import okhttp3.*
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

class RemoteWebSocketEngineBridge(private val context: Context) : IEngineBridge {

    private val gson = Gson()
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var okHttpClient: OkHttpClient? = null
    private var webSocket: WebSocket? = null

    private val _isConnected = AtomicBoolean(false)
    private val _isBlackout = AtomicBoolean(false)

    override val isConnected: Boolean get() = _isConnected.get()
    override val isBlackout: Boolean get() = _isBlackout.get()

    override var onHudUpdateListener: ((HudUpdate) -> Unit)? = null
    override var onConnectionStateChanged: ((Boolean) -> Unit)? = null

    override fun connect(serverIp: String, port: Int) {
        disconnect()

        val wsUrl = "ws://$serverIp:$port/ws/stream"
        Log.i("RemoteBridge", "Connecting WebSocket to: $wsUrl")
        val request = Request.Builder().url(wsUrl).build()

        okHttpClient = OkHttpClient.Builder()
            .proxy(java.net.Proxy.NO_PROXY)
            .readTimeout(0, TimeUnit.MILLISECONDS)
            .build()

        webSocket = okHttpClient?.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.i("RemoteBridge", "WebSocket connected successfully")
                _isConnected.set(true)
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
                    Log.e("RemoteBridge", "Failed to parse HUD update: ${e.message}")
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.i("RemoteBridge", "WebSocket closed: $code $reason")
                _isConnected.set(false)
                scope.launch(Dispatchers.Main) {
                    onConnectionStateChanged?.invoke(false)
                }
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("RemoteBridge", "WebSocket error: ${t.message}", t)
                _isConnected.set(false)
                scope.launch(Dispatchers.Main) {
                    onConnectionStateChanged?.invoke(false)
                }
            }
        })
    }

    override fun disconnect() {
        webSocket?.close(1000, "User disconnected")
        webSocket = null
        _isConnected.set(false)
        scope.launch(Dispatchers.Main) {
            onConnectionStateChanged?.invoke(false)
        }
    }

    override fun startBlackout() {
        _isBlackout.set(true)
        sendControl("start_blackout")
    }

    override fun stopBlackout() {
        _isBlackout.set(false)
        sendControl("stop_blackout")
    }

    override fun resetSession() {
        _isBlackout.set(false)
        sendControl("reset")
    }

    override fun pushBatch(batch: SensorBatch) {
        if (_isConnected.get()) {
            val json = gson.toJson(batch)
            webSocket?.send(json)
        }
    }

    override fun setMapMatching(enabled: Boolean) {
        val msg = ControlMessage(command = if (enabled) "enable_map_matching" else "disable_map_matching")
        webSocket?.send(gson.toJson(msg))
    }

    override fun prefetchArea(lat: Double, lon: Double, radiusM: Double, callback: (Boolean, String) -> Unit) {
        // In server mode, server manages its own road cache
        callback(true, "Server managed road network")
    }

    override fun prepareBenchmark(scenarioId: Int) {
        val msg = ControlMessage(command = "prepare_benchmark", scenarioId = scenarioId)
        webSocket?.send(gson.toJson(msg))
    }

    override fun startBenchmark(scenarioId: Int, speed: Double) {
        val msg = ControlMessage(command = "start_benchmark", scenarioId = scenarioId, speed = speed)
        webSocket?.send(gson.toJson(msg))
    }

    override fun stopBenchmark() {
        val msg = ControlMessage(command = "stop_benchmark")
        webSocket?.send(gson.toJson(msg))
    }

    private fun sendControl(command: String) {
        val msg = ControlMessage(command = command)
        webSocket?.send(gson.toJson(msg))
    }
}

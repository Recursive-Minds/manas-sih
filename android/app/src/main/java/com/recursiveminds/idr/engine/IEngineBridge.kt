package com.recursiveminds.idr.engine

import com.recursiveminds.idr.data.HudUpdate
import com.recursiveminds.idr.data.SensorBatch

/**
 * Common abstraction for dead reckoning engine communications.
 * Implemented as RemoteWebSocketEngineBridge in 'server' flavor
 * and LocalChaquopyEngineBridge in 'ondevice' flavor.
 */
interface IEngineBridge {
    val isConnected: Boolean
    val isBlackout: Boolean
    val isBenchmarkRunning: Boolean get() = false

    var onHudUpdateListener: ((HudUpdate) -> Unit)?
    var onConnectionStateChanged: ((Boolean) -> Unit)?

    fun connect(serverIp: String = "", port: Int = 8765)
    fun disconnect()
    fun startBlackout()
    fun stopBlackout()
    fun resetSession()
    fun pushBatch(batch: SensorBatch)
    fun setMapMatching(enabled: Boolean)
    fun prefetchArea(lat: Double, lon: Double, radiusM: Double, callback: (Boolean, String) -> Unit)
    fun prepareBenchmark(scenarioId: Int)
    fun startBenchmark(scenarioId: Int, speed: Double)
    fun stopBenchmark()
}

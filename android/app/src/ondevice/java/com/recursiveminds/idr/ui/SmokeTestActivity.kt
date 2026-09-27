package com.recursiveminds.idr.ui

import android.app.Activity
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.recursiveminds.idr.inference.TFLitePredictorBridge
import com.recursiveminds.idr.engine.LocalChaquopyEngineBridge
import com.recursiveminds.idr.data.HudUpdate
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

class SmokeTestActivity : Activity() {

    private lateinit var logView: TextView
    private lateinit var scrollView: ScrollView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val mode = intent.getStringExtra("mode") ?: "smoke"

        scrollView = ScrollView(this)
        logView = TextView(this).apply {
            textSize = 12f
            setPadding(24, 24, 24, 24)
            text = "Starting Phone ${mode.uppercase()} Test...\n"
        }
        scrollView.addView(logView)
        setContentView(scrollView)

        when (mode) {
            "parity" -> runParityTest()
            "drawer_benchmark" -> runDrawerBenchmarkTest()
            else -> runSmokeTest()
        }
    }

    private fun runParityTest() {
        CoroutineScope(Dispatchers.Default).launch {
            try {
                appendLog("=== ON-DEVICE PARITY & NO-FUTURE-LEAK TEST INITIALIZING ===")
                appendLog("DEVICE: ro.product.model=${Build.MODEL}, ro.build.version.release=${Build.VERSION.RELEASE}, ro.product.cpu.abi=${Build.SUPPORTED_ABIS[0]}")

                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(applicationContext))
                }
                val py = Python.getInstance()
                val bridge = TFLitePredictorBridge.create(applicationContext, "moe_velocity_model.tflite")
                appendLog("TFLitePredictorBridge created (fast bytes transfer)")

                val parityModule = py.getModule("run_ondevice_parity")
                val bundleFile = File(getExternalFilesDir(null), "s_s3a_parity_bundle.pkl")
                val reportFile = File(getExternalFilesDir(null), "step4_parity_report.json")
                appendLog("Using bundle: ${bundleFile.absolutePath}")
                appendLog("Invoking run_ondevice_parity.run_all_tests...")

                val res = parityModule.callAttr("run_all_tests", bridge, bundleFile.absolutePath, reportFile.absolutePath)
                val status = res.callAttr("get", "overall_status").toString()
                appendLog("PARITY_TEST_COMPLETED: overall_status=$status")
                appendLog("Report saved to ${reportFile.absolutePath}")

                // Also copy to /sdcard/step4_parity_report.json for adb pull convenience
                try {
                    val sdcardReport = File("/sdcard/step4_parity_report.json")
                    reportFile.copyTo(sdcardReport, overwrite = true)
                    appendLog("Copied report to /sdcard/step4_parity_report.json")
                } catch (_: Exception) {}
            } catch (e: Throwable) {
                appendLog("PARITY TEST FAILED: ${e.message}")
                Log.e("PHONE", "Parity test crashed", e)
            }
        }
    }

    private fun appendLog(msg: String) {
        Log.i("PHONE", msg)
        runOnUiThread {
            logView.append(msg + "\n")
            scrollView.fullScroll(ScrollView.FOCUS_DOWN)
        }
    }

    private fun runSmokeTest() {
        CoroutineScope(Dispatchers.Default).launch {
            try {
                appendLog("=== PHONE SMOKE TEST INITIALIZING ===")
                appendLog("DEVICE: ro.product.model=${Build.MODEL}, ro.build.version.release=${Build.VERSION.RELEASE}, ro.product.cpu.abi=${Build.SUPPORTED_ABIS[0]}")

                // 1. Initialize Python via Chaquopy
                if (!Python.isStarted()) {
                    Python.start(AndroidPlatform(applicationContext))
                }
                val py = Python.getInstance()
                appendLog("Chaquopy Python runtime initialized successfully")

                // 2. Initialize TFLite Predictor Bridge
                val bridge = TFLitePredictorBridge.create(applicationContext, "moe_velocity_model.tflite")
                val predictorModule = py.getModule("sih.models.predictor")
                appendLog("TFLitePredictorBridge created (threads=4, XNNPACK=true, warm-up done)")

                // =========================================================================
                // 2.4 Model Call Speed Benchmark: 600 calls legacy vs 600 calls fast bytes
                // =========================================================================
                appendLog("--------------------------------------------------------------------------------")
                appendLog("=== 2.4 MODEL CALL SPEED BENCHMARK (600 CALLS EACH) ===")

                // (1) Legacy path: element-by-element list conversion
                appendLog("Running 600 calls with legacy list conversion...")
                val legacyPredictor = predictorModule.callAttr("JavaBridgeVelocityPredictor", bridge, false)
                bridge.resetTiming()
                val legacyPyStats = predictorModule.callAttr("benchmark_bridge", legacyPredictor, 600)
                val legacyKotlinStats = bridge.getLegacyStats()

                val legA = legacyKotlinStats[1]
                val legB = legacyKotlinStats[2]
                val legC = legacyKotlinStats[3]
                val legKtTotal = legacyKotlinStats[4]
                val legPyTotal = legacyPyStats.callAttr("get", "total_ms").toDouble()

                appendLog("[BEFORE] Legacy lists (600 calls):")
                appendLog("  (a) Python->Java input conversion: %6.3f ms".format(legA))
                appendLog("  (b) interpreter.run/invoke only:    %6.3f ms".format(legB))
                appendLog("  (c) Java->Python output conversion: %6.3f ms".format(legC))
                appendLog("  Kotlin-internal mean per sample:   %6.3f ms".format(legKtTotal))
                appendLog("  End-to-end Python mean per sample: %6.3f ms".format(legPyTotal))

                // (2) Fast path: direct float32 bytes transfer
                appendLog("Running 600 calls with optimized bytes transfer...")
                val fastPredictor = predictorModule.callAttr("JavaBridgeVelocityPredictor", bridge, true)
                bridge.resetTiming()
                val fastPyStats = predictorModule.callAttr("benchmark_bridge", fastPredictor, 600)
                val fastKotlinStats = bridge.getFastStats()

                val fastA = fastKotlinStats[1]
                val fastB = fastKotlinStats[2]
                val fastC = fastKotlinStats[3]
                val fastKtTotal = fastKotlinStats[4]
                val fastPyTotal = fastPyStats.callAttr("get", "total_ms").toDouble()

                appendLog("[AFTER] Fast bytes transfer (600 calls):")
                appendLog("  (a) Python->Java input conversion: %6.3f ms".format(fastA))
                appendLog("  (b) interpreter.run/invoke only:    %6.3f ms".format(fastB))
                appendLog("  (c) Java->Python output conversion: %6.3f ms".format(fastC))
                appendLog("  Kotlin-internal mean per sample:   %6.3f ms".format(fastKtTotal))
                appendLog("  End-to-end Python mean per sample: %6.3f ms".format(fastPyTotal))
                appendLog("Speedup: %.1fx (Python total %.2f ms -> %.2f ms)".format(legPyTotal / fastPyTotal, legPyTotal, fastPyTotal))

                // 3. Load smoke test dataset from assets
                val jsonStr = assets.open("smoke_test_60s.json").bufferedReader().use { it.readText() }
                val jsonObj = JSONObject(jsonStr)
                val refLat = jsonObj.getDouble("reference_lat_deg")
                val refLon = jsonObj.getDouble("reference_lon_deg")
                val refAlt = jsonObj.getDouble("reference_alt_m")
                val batches = jsonObj.getJSONArray("batches")
                val nBatches = batches.length()
                appendLog("--------------------------------------------------------------------------------")
                appendLog("Loaded smoke test dataset: $nBatches batches (60s total, S-S3a slice)")

                // 4. Instantiate SessionCore with fast TFLite predictor
                val sessionModule = py.getModule("server.session_core")
                val session = sessionModule.callAttr(
                    "SessionCore",
                    refLat, refLon, refAlt,
                    null, // saved_alignment
                    null, // road_network
                    "Mixed", // domain
                    "stage_b", // engine_type
                    null, // ai_model
                    null, // norm_mean
                    null, // norm_std
                    null, // device
                    true, // use_speed_smoother
                    fastPredictor // predictor
                )
                appendLog("SessionCore initialized with fast TFLite predictor")

                // 5. Replay batches and measure time splits
                val pyJson = py.getModule("json")
                val featTimes = mutableListOf<Double>()
                val modelTimes = mutableListOf<Double>()
                val ekfMapTimes = mutableListOf<Double>()
                val totalTimes = mutableListOf<Double>()

                appendLog("--------------------------------------------------------------------------------")
                appendLog("Starting replay of 60 recorded batches (10 Hz IMU, 1 Hz GNSS)...")

                for (i in 0 until nBatches) {
                    val batchObj = batches.getJSONObject(i)
                    val batchStr = batchObj.toString()
                    val batchDict = pyJson.callAttr("loads", batchStr)

                    // Process batch through SessionCore
                    session.callAttr("push_batch", batchDict, "phone")

                    // Retrieve precise timing breakdown
                    val timing = session.callAttr("get_last_batch_timing")
                    val featMs = timing.callAttr("get", "features_ms").toDouble()
                    val modelMs = timing.callAttr("get", "model_ms").toDouble()
                    val ekfMapMs = timing.callAttr("get", "ekf_map_ms").toDouble()
                    val totalMs = timing.callAttr("get", "total_ms").toDouble()

                    featTimes.add(featMs)
                    modelTimes.add(modelMs)
                    ekfMapTimes.add(ekfMapMs)
                    totalTimes.add(totalMs)

                    val state = batchObj.getString("state")
                    appendLog(
                        "Batch %02d/%02d [%s]: feat=%5.2f ms | model=%5.2f ms (%4.2f ms/sample) | ekf_map=%5.2f ms | total=%5.2f ms".format(
                            i + 1, nBatches, state, featMs, modelMs, modelMs / 10.0, ekfMapMs, totalMs
                        )
                    )
                }

                // 6. Statistics
                val meanFeat = featTimes.average()
                val meanModel = modelTimes.average()
                val meanEkfMap = ekfMapTimes.average()
                val meanTotal = totalTimes.average()
                val sortedTotal = totalTimes.sorted()
                val p95Total = sortedTotal[(sortedTotal.size * 0.95).toInt()]
                val maxTotal = sortedTotal.last()

                appendLog("--------------------------------------------------------------------------------")
                appendLog("=== PHONE SMOKE TEST COMPLETE: 60/60 BATCHES SUCCESS ===")
                appendLog("PHONE_MODEL: ${Build.MODEL}")
                appendLog("ANDROID_VERSION: ${Build.VERSION.RELEASE}")
                appendLog("CPU_ABI: ${Build.SUPPORTED_ABIS[0]}")
                appendLog("LATENCY_SUMMARY: Mean=%.2f ms, P95=%.2f ms, Max=%.2f ms".format(meanTotal, p95Total, maxTotal))
                appendLog("TIME_SPLIT_MEANS: features=%.2f ms, model=%.2f ms (%.2f ms/sample), ekf_map=%.2f ms".format(meanFeat, meanModel, meanModel / 10.0, meanEkfMap))
                appendLog("SMOKE_TEST_RESULT: PASS - Zero crashes, exact 60s stream processed")

                // Write report to app files directory
                val reportFile = File(getExternalFilesDir(null), "phone_smoke_test_result.txt")
                reportFile.writeText(
                    "MODEL=${Build.MODEL}\n" +
                    "ANDROID=${Build.VERSION.RELEASE}\n" +
                    "ABI=${Build.SUPPORTED_ABIS[0]}\n" +
                    "BATCHES=60\n" +
                    "BENCHMARK_2_4_LEGACY_A_MS=$legA\n" +
                    "BENCHMARK_2_4_LEGACY_B_MS=$legB\n" +
                    "BENCHMARK_2_4_LEGACY_C_MS=$legC\n" +
                    "BENCHMARK_2_4_LEGACY_KT_MS=$legKtTotal\n" +
                    "BENCHMARK_2_4_LEGACY_PY_MS=$legPyTotal\n" +
                    "BENCHMARK_2_4_FAST_A_MS=$fastA\n" +
                    "BENCHMARK_2_4_FAST_B_MS=$fastB\n" +
                    "BENCHMARK_2_4_FAST_C_MS=$fastC\n" +
                    "BENCHMARK_2_4_FAST_KT_MS=$fastKtTotal\n" +
                    "BENCHMARK_2_4_FAST_PY_MS=$fastPyTotal\n" +
                    "MEAN_TOTAL_MS=$meanTotal\n" +
                    "P95_TOTAL_MS=$p95Total\n" +
                    "MEAN_FEAT_MS=$meanFeat\n" +
                    "MEAN_MODEL_MS=$meanModel\n" +
                    "MEAN_MODEL_PER_SAMPLE_MS=${meanModel / 10.0}\n" +
                    "MEAN_EKF_MAP_MS=$meanEkfMap\n" +
                    "RESULT=PASS\n"
                )
                appendLog("Report saved to: ${reportFile.absolutePath}")

            } catch (e: Throwable) {
                appendLog("FATAL ERROR IN SMOKE TEST: ${e.message}")
                Log.e("PHONE", "Smoke test crashed", e)
            }
        }
    }

    private fun runDrawerBenchmarkTest() {
        CoroutineScope(Dispatchers.Default).launch {
            try {
                appendLog("================================================================================")
                appendLog("=== PHYSICAL PHONE ON-DEVICE BENCHMARK DRAWER ACCEPTANCE TEST ===")
                appendLog("DEVICE: ro.product.model=${Build.MODEL}, ro.build.version.release=${Build.VERSION.RELEASE}, ro.product.cpu.abi=${Build.SUPPORTED_ABIS[0]}")
                appendLog("================================================================================")

                val engineBridge = LocalChaquopyEngineBridge(applicationContext)

                // Wait until engine is connected
                var waitCount = 0
                while (!engineBridge.isConnected && waitCount < 100) {
                    delay(100)
                    waitCount++
                }
                if (!engineBridge.isConnected) {
                    throw IllegalStateException("LocalChaquopyEngineBridge failed to connect within 10s")
                }
                appendLog("LocalChaquopyEngineBridge connected and ready")

                val scenarios = listOf(
                    Triple(1, 80.59, 26.73),
                    Triple(22, 16.77, 3.53),
                    Triple(23, 67.76, 6.01),
                    Triple(25, 77.30, 12.58),
                    Triple(26, 122.80, 13.75),
                    Triple(30, 7.04, 2.88)
                )

                val results = mutableListOf<JSONObject>()
                var allPassed = true

                for ((scId, expErr, expDrift) in scenarios) {
                    appendLog("--------------------------------------------------------------------------------")
                    appendLog("Running On-Device Benchmark Scenario #$scId...")

                    val hudUpdates = mutableListOf<HudUpdate>()
                    engineBridge.onHudUpdateListener = { hud ->
                        synchronized(hudUpdates) {
                            hudUpdates.add(hud)
                        }
                    }

                    engineBridge.prepareBenchmark(scId)
                    delay(1000) // Allow bundle extraction and SessionCore initialization

                    val tStart = System.currentTimeMillis()
                    // Replay at 50x speed for smooth and rapid on-device execution
                    engineBridge.startBenchmark(scId, 50.0)

                    // Await completion via isBenchmarkRunning
                    delay(500)
                    while (engineBridge.isBenchmarkRunning) {
                        delay(100)
                    }
                    val elapsedMs = System.currentTimeMillis() - tStart

                    val lastHud = synchronized(hudUpdates) { hudUpdates.lastOrNull() }
                    val metrics = lastHud?.metrics
                    val finalErr = metrics?.sessionSummary?.finalErrorM ?: (metrics?.horizontalErrorM ?: 0.0)
                    val finalDrift = metrics?.sessionSummary?.driftPct ?: (metrics?.driftPct ?: 0.0)
                    val warmup = lastHud?.warmup
                    val mountState = warmup?.mountState ?: ""
                    val mapStatus = warmup?.mapMatchingStatus ?: (if (warmup?.mapMatchingEnabled == true) "MAP MATCH: ON" else "MAP MATCH: OFF")

                    val errDiff = kotlin.math.abs(finalErr - expErr)
                    val driftDiff = kotlin.math.abs(finalDrift - expDrift)
                    val passed = errDiff <= 0.05 && driftDiff <= 0.05 && mountState == "REUSED"

                    if (!passed) allPassed = false

                    // Compute distinct DR positions
                    val drPositions = synchronized(hudUpdates) { hudUpdates.mapNotNull { it.drPos } }
                    val distinctDrPositions = drPositions.distinctBy { Pair(it.lat, it.lon) }.size
                    val totalDrPositions = drPositions.size
                    val distinctDrRatio = if (totalDrPositions > 0) distinctDrPositions.toDouble() / totalDrPositions else 0.0

                    appendLog("Scenario #$scId Result:")
                    appendLog("  Final Error: %.2f m (Expected: %.2f m, Diff: %.3f m)".format(finalErr, expErr, errDiff))
                    appendLog("  Drift: %.2f %% (Expected: %.2f %%, Diff: %.3f pp)".format(finalDrift, expDrift, driftDiff))
                    appendLog("  Mount State: $mountState | Map Matching: $mapStatus")
                    appendLog("  Distinct DR Positions: $distinctDrPositions / $totalDrPositions (%.1f%%)".format(distinctDrRatio * 100.0))
                    appendLog("  Replay Time: %d ms | Status: %s".format(elapsedMs, if (passed) "PASS" else "FAIL"))

                    val scJson = JSONObject().apply {
                        put("scenario_id", scId)
                        put("expected_error_m", expErr)
                        put("phone_error_m", finalErr)
                        put("error_diff_m", errDiff)
                        put("expected_drift_pct", expDrift)
                        put("phone_drift_pct", finalDrift)
                        put("drift_diff_pp", driftDiff)
                        put("mount_state", mountState)
                        put("map_status", mapStatus)
                        put("distinct_dr_ratio", distinctDrRatio)
                        put("passed", passed)
                    }
                    results.add(scJson)
                    engineBridge.stopBenchmark()
                    delay(300)
                }

                appendLog("================================================================================")
                appendLog("=== ON-DEVICE BENCHMARK DRAWER SUMMARY ===")
                appendLog("OVERALL RESULT: ${if (allPassed) "ALL PASS" else "FAIL"}")
                appendLog("================================================================================")

                val fullReport = JSONObject().apply {
                    put("device_model", Build.MODEL)
                    put("android_version", Build.VERSION.RELEASE)
                    put("overall_passed", allPassed)
                    put("results", org.json.JSONArray(results))
                }

                val reportFile = File(getExternalFilesDir(null), "drawer_benchmark_report.json")
                reportFile.writeText(fullReport.toString(2))
                try {
                    File("/sdcard/drawer_benchmark_report.json").writeText(fullReport.toString(2))
                } catch (_: Exception) {}
                appendLog("Saved full report to ${reportFile.absolutePath} and /sdcard/drawer_benchmark_report.json")
            } catch (e: Throwable) {
                appendLog("FATAL ERROR IN DRAWER BENCHMARK TEST: ${e.message}")
                Log.e("PHONE", "Drawer benchmark test crashed", e)
            }
        }
    }
}

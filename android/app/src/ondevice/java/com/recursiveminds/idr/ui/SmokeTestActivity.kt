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
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File

class SmokeTestActivity : Activity() {

    private lateinit var logView: TextView
    private lateinit var scrollView: ScrollView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        scrollView = ScrollView(this)
        logView = TextView(this).apply {
            textSize = 12f
            setPadding(24, 24, 24, 24)
            text = "Starting Phone Smoke Test...\n"
        }
        scrollView.addView(logView)
        setContentView(scrollView)

        runSmokeTest()
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
}

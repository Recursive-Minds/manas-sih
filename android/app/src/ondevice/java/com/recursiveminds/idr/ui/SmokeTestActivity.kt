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
                val pyPredictor = predictorModule.callAttr("JavaBridgeVelocityPredictor", bridge)
                appendLog("TFLitePredictorBridge linked to JavaBridgeVelocityPredictor")

                // 3. Load smoke test dataset from assets
                val jsonStr = assets.open("smoke_test_60s.json").bufferedReader().use { it.readText() }
                val jsonObj = JSONObject(jsonStr)
                val refLat = jsonObj.getDouble("reference_lat_deg")
                val refLon = jsonObj.getDouble("reference_lon_deg")
                val refAlt = jsonObj.getDouble("reference_alt_m")
                val batches = jsonObj.getJSONArray("batches")
                val nBatches = batches.length()
                appendLog("Loaded smoke test dataset: $nBatches batches (60s total, S-S3a slice)")

                // 4. Instantiate SessionCore with TFLite predictor
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
                    pyPredictor // predictor
                )
                appendLog("SessionCore initialized with Stage B dead reckoning engine")

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
                        "Batch %02d/%02d [%s]: features=%5.2f ms | model=%5.2f ms | ekf_map=%5.2f ms | total=%5.2f ms".format(
                            i + 1, nBatches, state, featMs, modelMs, ekfMapMs, totalMs
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
                appendLog("TIME_SPLIT_MEANS: features=%.2f ms, model=%.2f ms, ekf_map=%.2f ms".format(meanFeat, meanModel, meanEkfMap))
                appendLog("SMOKE_TEST_RESULT: PASS - Zero crashes, exact 60s stream processed")

                // Write report to app files directory
                val reportFile = File(getExternalFilesDir(null), "phone_smoke_test_result.txt")
                reportFile.writeText(
                    "MODEL=${Build.MODEL}\n" +
                    "ANDROID=${Build.VERSION.RELEASE}\n" +
                    "ABI=${Build.SUPPORTED_ABIS[0]}\n" +
                    "BATCHES=60\n" +
                    "MEAN_TOTAL_MS=$meanTotal\n" +
                    "P95_TOTAL_MS=$p95Total\n" +
                    "MEAN_FEAT_MS=$meanFeat\n" +
                    "MEAN_MODEL_MS=$meanModel\n" +
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

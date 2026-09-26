package com.recursiveminds.idr.ui

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.util.Locale

/**
 * UiTraceLogger
 * Records UI marker updates and HUD field updates into ui_trace.csv in app storage.
 * Per Ground Rule / Part 0:
 * ui_trace.csv: t, marker (dr/gnss/reconciled), lat, lon, rotation_deg, visible
 * Also logs each time HUD text fields are set: speed, heading, turns, calib_s, drift.
 */
object UiTraceLogger {
    private const val TAG = "UiTraceLogger"
    private var traceFile: File? = null
    private var writer: PrintWriter? = null

    fun init(context: Context) {
        try {
            val dir = context.filesDir
            traceFile = File(dir, "ui_trace.csv")
            reset()
            Log.i(TAG, "Initialized ui_trace.csv at ${traceFile?.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize UiTraceLogger: ${e.message}")
        }
    }

    @Synchronized
    fun reset() {
        try {
            writer?.close()
            val f = traceFile ?: return
            writer = PrintWriter(FileWriter(f, false))
            writer?.println("t,category,name,lat,lon,rotation_deg,visible,text_val,num_val")
            writer?.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to reset UiTraceLogger: ${e.message}")
        }
    }

    private fun getTimestampSec(): Double {
        return (System.currentTimeMillis() / 1000.0)
    }

    @Synchronized
    fun logMarker(markerName: String, lat: Double, lon: Double, rotationDeg: Double, visible: Boolean) {
        try {
            val t = getTimestampSec()
            val line = String.format(
                Locale.US,
                "%.3f,marker,%s,%.8f,%.8f,%.2f,%b,,",
                t, markerName, lat, lon, rotationDeg, visible
            )
            writer?.println(line)
            writer?.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Error logging marker: ${e.message}")
        }
    }

    @Synchronized
    fun logHud(fieldName: String, textVal: String, numVal: Double?) {
        try {
            val t = getTimestampSec()
            val numStr = if (numVal != null) String.format(Locale.US, "%.3f", numVal) else ""
            val escapedText = textVal.replace(",", ";").replace("\n", " ").trim()
            val line = String.format(
                Locale.US,
                "%.3f,hud,%s,,,,,\"%s\",%s",
                t, fieldName, escapedText, numStr
            )
            writer?.println(line)
            writer?.flush()
        } catch (e: Exception) {
            Log.e(TAG, "Error logging HUD: ${e.message}")
        }
    }
}

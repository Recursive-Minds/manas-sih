package com.recursiveminds.idr.inference

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TFLitePredictorBridge(private val interpreter: Interpreter) {

    // Preallocated direct ByteBuffers (native byte order) created ONCE
    private val inputShortBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 12 * 20 * 4).order(ByteOrder.nativeOrder())
    private val inputLongBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 12 * 60 * 4).order(ByteOrder.nativeOrder())
    private val outputVBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 1 * 4).order(ByteOrder.nativeOrder())
    private val outputVarBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 1 * 4).order(ByteOrder.nativeOrder())

    // Timing metrics for legacy path (600 calls)
    var legacyCalls = 0
    var legacyInputNs = 0L
    var legacyInvokeNs = 0L
    var legacyOutputNs = 0L

    // Timing metrics for fast bytes path (600 calls)
    var fastCalls = 0
    var fastInputNs = 0L
    var fastInvokeNs = 0L
    var fastOutputNs = 0L

    fun resetTiming() {
        legacyCalls = 0
        legacyInputNs = 0L
        legacyInvokeNs = 0L
        legacyOutputNs = 0L

        fastCalls = 0
        fastInputNs = 0L
        fastInvokeNs = 0L
        fastOutputNs = 0L
    }

    // --- Fast path: pass raw float32 bytes directly from Python NumPy arrays ---
    @Synchronized
    fun predictWindowBytes(shortBytes: ByteArray, longBytes: ByteArray): FloatArray {
        // (a) Input conversion: direct byte copies into preallocated native ByteBuffers
        val t0 = System.nanoTime()
        inputShortBuffer.rewind()
        inputShortBuffer.put(shortBytes)
        inputShortBuffer.rewind()

        inputLongBuffer.rewind()
        inputLongBuffer.put(longBytes)
        inputLongBuffer.rewind()
        val t1 = System.nanoTime()

        // (b) Interpreter invoke only
        outputVBuffer.rewind()
        outputVarBuffer.rewind()
        val inputs = arrayOf<Any>(inputShortBuffer, inputLongBuffer)
        val outputs = mapOf<Int, Any>(0 to outputVBuffer, 1 to outputVarBuffer)

        val t2 = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
        val t3 = System.nanoTime()

        // (c) Output conversion: reading floats from output ByteBuffers
        val t4 = System.nanoTime()
        outputVBuffer.rewind()
        outputVarBuffer.rewind()
        val v = outputVBuffer.float
        val varVal = outputVarBuffer.float
        val res = floatArrayOf(v, varVal)
        val t5 = System.nanoTime()

        fastInputNs += (t1 - t0)
        fastInvokeNs += (t3 - t2)
        fastOutputNs += (t5 - t4)
        fastCalls++

        return res
    }

    // Direct float array overload (e.g. from Java/Kotlin or legacy Python jarray)
    @Synchronized
    fun predictWindow(shortFloats: FloatArray, longFloats: FloatArray): FloatArray {
        val t0 = System.nanoTime()
        inputShortBuffer.rewind()
        for (f in shortFloats) inputShortBuffer.putFloat(f)
        inputShortBuffer.rewind()

        inputLongBuffer.rewind()
        for (f in longFloats) inputLongBuffer.putFloat(f)
        inputLongBuffer.rewind()
        val t1 = System.nanoTime()

        outputVBuffer.rewind()
        outputVarBuffer.rewind()

        val inputs = arrayOf<Any>(inputShortBuffer, inputLongBuffer)
        val outputs = mapOf<Int, Any>(0 to outputVBuffer, 1 to outputVarBuffer)

        val t2 = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
        val t3 = System.nanoTime()

        outputVBuffer.rewind()
        outputVarBuffer.rewind()
        val v = outputVBuffer.float
        val varVal = outputVarBuffer.float
        val res = floatArrayOf(v, varVal)
        val t4 = System.nanoTime()

        legacyInputNs += (t1 - t0)
        legacyInvokeNs += (t3 - t2)
        legacyOutputNs += (t4 - t3)
        legacyCalls++

        return res
    }

    // --- Legacy path: Python/Java lists element by element (for 2.4 benchmarking) ---
    @Synchronized
    fun predictWindow(shortList: List<*>, longList: List<*>): FloatArray {
        // (a) Python->Java conversion: traversing Python list proxies via JNI reflection
        val t0 = System.nanoTime()
        val sArr = FloatArray(shortList.size) { (shortList[it] as Number).toFloat() }
        val lArr = FloatArray(longList.size) { (longList[it] as Number).toFloat() }
        inputShortBuffer.rewind()
        for (f in sArr) inputShortBuffer.putFloat(f)
        inputShortBuffer.rewind()
        inputLongBuffer.rewind()
        for (f in lArr) inputLongBuffer.putFloat(f)
        inputLongBuffer.rewind()
        val t1 = System.nanoTime()

        // (b) Interpreter invoke only
        outputVBuffer.rewind()
        outputVarBuffer.rewind()
        val inputs = arrayOf<Any>(inputShortBuffer, inputLongBuffer)
        val outputs = mapOf<Int, Any>(0 to outputVBuffer, 1 to outputVarBuffer)

        val t2 = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
        val t3 = System.nanoTime()

        // (c) Output conversion
        val t4 = System.nanoTime()
        outputVBuffer.rewind()
        outputVarBuffer.rewind()
        val v = outputVBuffer.float
        val varVal = outputVarBuffer.float
        val res = floatArrayOf(v, varVal)
        val t5 = System.nanoTime()

        legacyInputNs += (t1 - t0)
        legacyInvokeNs += (t3 - t2)
        legacyOutputNs += (t5 - t4)
        legacyCalls++

        return res
    }

    fun getLegacyStats(): FloatArray {
        val n = if (legacyCalls > 0) legacyCalls.toFloat() else 1.0f
        val aMs = (legacyInputNs / 1_000_000.0f) / n
        val bMs = (legacyInvokeNs / 1_000_000.0f) / n
        val cMs = (legacyOutputNs / 1_000_000.0f) / n
        return floatArrayOf(legacyCalls.toFloat(), aMs, bMs, cMs, aMs + bMs + cMs)
    }

    fun getFastStats(): FloatArray {
        val n = if (fastCalls > 0) fastCalls.toFloat() else 1.0f
        val aMs = (fastInputNs / 1_000_000.0f) / n
        val bMs = (fastInvokeNs / 1_000_000.0f) / n
        val cMs = (fastOutputNs / 1_000_000.0f) / n
        return floatArrayOf(fastCalls.toFloat(), aMs, bMs, cMs, aMs + bMs + cMs)
    }

    fun close() {
        interpreter.close()
    }

    companion object {
        private const val TAG = "TFLitePredictorBridge"

        fun create(context: Context, assetName: String = "moe_velocity_model.tflite"): TFLitePredictorBridge {
            Log.i(TAG, "Loading TFLite model from assets: $assetName")
            val afd = context.assets.openFd(assetName)
            val inputStream = FileInputStream(afd.fileDescriptor)
            val fileChannel = inputStream.channel
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            
            // XNNPACK on, 4 threads
            val options = Interpreter.Options().apply {
                setNumThreads(4)
                setUseXNNPACK(true)
            }
            val interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "TFLite Interpreter initialized successfully (threads=4, XNNPACK=true)")
            val bridge = TFLitePredictorBridge(interpreter)

            // Warm-up call (1 sample)
            val dummyShort = FloatArray(12 * 20) { 0.0f }
            val dummyLong = FloatArray(12 * 60) { 0.0f }
            bridge.predictWindow(dummyShort, dummyLong)
            bridge.resetTiming()
            Log.i(TAG, "TFLite warm-up inference completed successfully")

            return bridge
        }
    }
}

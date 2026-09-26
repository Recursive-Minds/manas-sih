package com.recursiveminds.idr.inference

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class TFLitePredictorBridge(private val interpreter: Interpreter) {

    private val inputShortBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 12 * 20 * 4).order(ByteOrder.nativeOrder())
    private val inputLongBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 12 * 60 * 4).order(ByteOrder.nativeOrder())
    private val outputVBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 1 * 4).order(ByteOrder.nativeOrder())
    private val outputVarBuffer: ByteBuffer = ByteBuffer.allocateDirect(1 * 1 * 4).order(ByteOrder.nativeOrder())

    @Synchronized
    fun predictWindow(shortFloats: FloatArray, longFloats: FloatArray): FloatArray {
        inputShortBuffer.rewind()
        for (f in shortFloats) inputShortBuffer.putFloat(f)

        inputLongBuffer.rewind()
        for (f in longFloats) inputLongBuffer.putFloat(f)

        outputVBuffer.rewind()
        outputVarBuffer.rewind()

        val inputs = arrayOf<Any>(inputShortBuffer, inputLongBuffer)
        val outputs = mapOf<Int, Any>(0 to outputVBuffer, 1 to outputVarBuffer)

        interpreter.runForMultipleInputsOutputs(inputs, outputs)

        outputVBuffer.rewind()
        outputVarBuffer.rewind()
        val v = outputVBuffer.float
        val varVal = outputVarBuffer.float
        return floatArrayOf(v, varVal)
    }

    // Overload accepting Python/Java lists
    @Synchronized
    fun predictWindow(shortList: List<*>, longList: List<*>): FloatArray {
        val sArr = FloatArray(shortList.size) { (shortList[it] as Number).toFloat() }
        val lArr = FloatArray(longList.size) { (longList[it] as Number).toFloat() }
        return predictWindow(sArr, lArr)
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
            val options = Interpreter.Options().apply {
                setNumThreads(2)
            }
            val interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "TFLite Interpreter initialized successfully")
            return TFLitePredictorBridge(interpreter)
        }
    }
}

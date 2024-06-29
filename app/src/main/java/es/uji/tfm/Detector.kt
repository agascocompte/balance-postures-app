package es.uji.tfm

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import kotlin.math.exp
import kotlin.math.min

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String,
    private val detectorListener: DetectorListener,
) {

    private var interpreter: Interpreter? = null
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    fun printInputShapes() {
        interpreter ?: return

        val numInputs = interpreter!!.inputTensorCount
        for (i in 0 until numInputs) {
            val inputShape = interpreter!!.getInputTensor(i).shape()
            val inputDataType = interpreter!!.getInputTensor(i).dataType()
            Log.d("AAAAAAAAAAAAAAAAAAAAAAAA", "Input Tensor $i: Shape = ${inputShape.contentToString()}, DataType = $inputDataType")
        }
    }

    fun printOutputShapes() {
        interpreter ?: return

        val numOutputs = interpreter!!.outputTensorCount
        for (i in 0 until numOutputs) {
            val outputShape = interpreter!!.getOutputTensor(i).shape()
            val outputDataType = interpreter!!.getOutputTensor(i).dataType()
            Log.d("AAAAAAAAAAAAAAAAAAAAAAAA", "Output Tensor $i: Shape = ${outputShape.contentToString()}, DataType = $outputDataType")
        }
    }

    fun setup(isGpu: Boolean = true) {

        if (interpreter != null) {
            close()
        }

        val options = if (isGpu) {
            val compatList = CompatibilityList()

            Interpreter.Options().apply{
                if(compatList.isDelegateSupportedOnThisDevice){
                    val delegateOptions = compatList.bestOptionsForThisDevice
                    this.addDelegate(GpuDelegate(delegateOptions))
                } else {
                    this.setNumThreads(4)
                }
            }
        } else {
            Interpreter.Options().apply{
                this.setNumThreads(4)
            }
        }


        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return
        val outputShape0 = interpreter?.getOutputTensor(0)?.shape() ?: return
        val outputShape1 = interpreter?.getOutputTensor(1)?.shape() ?: return

        tensorWidth = inputShape[1]
        tensorHeight = inputShape[2]

        // If in case input shape is in format of [1, 3, ..., ...]
        if (inputShape[1] == 3) {
            tensorWidth = inputShape[2]
            tensorHeight = inputShape[3]
        }

        numChannel = outputShape0[1]
        numElements = outputShape0[2]

        try {
            val inputStream: InputStream = context.assets.open(labelPath)
            val reader = BufferedReader(InputStreamReader(inputStream))

            var line: String? = reader.readLine()
            while (line != null && line != "") {
                labels.add(line)
                line = reader.readLine()
            }

            reader.close()
            inputStream.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
        printInputShapes()
        printOutputShapes()
    }

    fun close() {
        interpreter?.close()
        interpreter = null
    }

    fun detect(frame: Bitmap) {
        interpreter ?: return
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return

        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val coordinatesBuffer = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)
        val maskProtoBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 160, 160, 32), OUTPUT_IMAGE_TYPE)

        val outputs = mapOf<Int, Any>(
            0 to coordinatesBuffer.buffer.rewind(),
            1 to maskProtoBuffer.buffer.rewind()
        )
        interpreter?.runForMultipleInputsOutputs(arrayOf(imageBuffer), outputs)

        val coordinates = coordinatesBuffer.floatArray
        val masks = maskProtoBuffer.floatArray
        val filterOutput0 = mutableListOf<Output0>()

        for (c in 0 until numElements) {
            val classIndex = coordinates[c + numElements * 5]
            if (classIndex.toInt() == PERSON_CLASS_INDEX) {
                val cnf = coordinates[c + numElements * 4]
                if (cnf > CONFIDENCE_THRESHOLD) {
                    val cls = coordinates[c + numElements * 5].toInt()
                    val cx = coordinates[c]
                    val cy = coordinates[c + numElements]
                    val w = coordinates[c + numElements * 2]
                    val h = coordinates[c + numElements * 3]

                    val x1 = cx - (w/2F)
                    val y1 = cy - (h/2F)
                    val x2 = cx + (w/2F)
                    val y2 = cy + (h/2F)
                    if (x1 < 0F || x1 > 1F) continue
                    if (y1 < 0F || y1 > 1F) continue
                    if (x2 < 0F || x2 > 1F) continue
                    if (y2 < 0F || y2 > 1F) continue

                    val maskWeight = mutableListOf<Float>()
                    for (index in 0 until 32) {
                        maskWeight.add(coordinates[c + numElements * (index + 5)])
                    }
                    filterOutput0.add(Output0(cx = cx, cy = cy, w = w, h = h, cnf = cnf, cls = cls, maskWeight = maskWeight))
                }
            }
        }

        if (filterOutput0.isEmpty()) return
        val best = applyNMS(filterOutput0).sortedByDescending { it.cnf }[0]

        val output1 = reshapeOutput1(masks)

        val mask = Mat()
        Core.compare(output1[0], Scalar(0.5), mask, Core.CMP_GT)
        val bestBox = BoundingBox(
            x1 = best.cx - best.w / 2,
            y1 = best.cy - best.h / 2,
            x2 = best.cx + best.w / 2,
            y2 = best.cy + best.h / 2,
            cx = best.cx,
            cy = best.cy,
            cls = best.cls,
            clsName = "Person",
            cnf = best.cnf,
            h = best.h,
            w = best.w
        )

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        detectorListener.onDetect(bestBox, inferenceTime, matToBitmap(mask))
    }

    private fun matToBitmap(mat: Mat): Bitmap {
        // Asumiendo que mat es en escala de grises y queremos convertir los máximos a transparente y los mínimos a verde
        val colorMat = Mat(mat.size(), CvType.CV_8UC4) // Usar 4 canales para RGBA
        for (i in 0 until mat.rows()) {
            for (j in 0 until mat.cols()) {
                val value = mat.get(i, j)[0]
                if (value == 255.0) {  // Máximo en escala de grises
                    colorMat.put(i, j, 0.0, 0.0, 0.0, 0.0)  // Transparente
                } else {
                    colorMat.put(i, j, 0.0, 255.0, 0.0, 255.0)  // Verde y opaco
                }
            }
        }
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(colorMat, bitmap)
        colorMat.release()
        return bitmap
    }

    private fun reshapeOutput1(masks: FloatArray): List<Mat> {
        val all = mutableListOf<Mat>()
        val size = 160
        val channelCount = 32

        for (maskIndex in 0 until channelCount) {
            val mat = Mat(size, size, CvType.CV_32F)
            val buffer = FloatArray(size * size)

            for (i in 0 until size) {
                for (j in 0 until size) {
                    //val index = channelCount * size * j + channelCount * i + maskIndex
                    //buffer[j * size + i] = sigmoid(masks[index])
                    buffer[j * size + i] =  masks[channelCount * size * j + channelCount * i + maskIndex]
                }
            }

            mat.put(0, 0, buffer)
            all.add(mat)
        }
        return all
    }

    private fun applyNMS(bestOutput0: List<Output0>): List<Output0> {
        val sortedBoxes = bestOutput0.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<Output0>()

        while(sortedBoxes.isNotEmpty()) {
            val first = sortedBoxes.first()
            selectedBoxes.add(first)
            sortedBoxes.remove(first)

            val iterator = sortedBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                val iou = calculateIoU(first, nextBox)
                if (iou >= IOU_THRESHOLD) {
                    iterator.remove()
                }
            }
        }

        return selectedBoxes
    }
    private fun calculateIoU(b1: Output0, b2: Output0): Float {
        val x1 = maxOf(b1.cx - (b1.w/2F), b2.cx - (b2.w/2F))
        val y1 = maxOf(b1.cy - (b1.h/2F), b2.cy - (b2.h/2F))
        val x2 = minOf(b1.cx + (b1.w/2F), b2.cx + (b2.w/2F))
        val y2 = minOf(b1.cy + (b1.h/2F), b2.cy + (b2.h/2F))

        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = b1.w * b1.h
        val box2Area = b2.w * b2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }

    private fun Mat.multiplyDouble(double: Double) : Mat {
        val result = Mat()
        Core.multiply(this, Scalar(double), result)
        return result
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(bestBox: BoundingBox, inferenceTime: Long, mask: Bitmap)
    }

    companion object {
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val IOU_THRESHOLD = 0.5F
        private const val PERSON_CLASS_INDEX = 0
    }
}

private data class Output0(
    val cx: Float,
    val cy: Float,
    val w: Float,
    val h: Float,
    val cnf: Float,
    val cls: Int,
    val maskWeight: List<Float>
)
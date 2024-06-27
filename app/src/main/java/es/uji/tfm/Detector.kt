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
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        val filterOutput0 = mutableListOf<Output0>()

        for (c in 0 until numElements) {
            val classIndex = coordinates[c + numElements * 5]
            if (classIndex.toInt() == 0) {
                val cnf = coordinates[c + numElements * 4]
                if (cnf > CONFIDENCE_THRESHOLD) {
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
                    filterOutput0.add(Output0(cx = cx, cy = cy, w = w, h = h, cnf = cnf, maskWeight = maskWeight))
                }
            }
        }

        if (filterOutput0.isEmpty()) return
        val best = applyNMS(filterOutput0).sortedByDescending { it.cnf }[0]

        val output1 = reshapeOutput1(masks)

        val multiply = mutableListOf<Mat>()
        for (index in 0 until 32) {
            multiply.add(output1[index].multiplyDouble(best.maskWeight[index].toDouble()))
        }

        val final = multiply[0].clone()
        for (i in 1 until multiply.size) {
            Core.add(final, multiply[i], final)
        }

        val mask = Mat()
        Core.compare(final, Scalar(0.0), mask, Core.CMP_GT)



        /*Log.d("OUTPUT0 SHAPE", output0.shape.contentToString())
        Log.d("OUTPUT1 SHAPE", output1.shape.contentToString())
        val bestBoxes = bestBox(output0.floatArray)
        val segmentationMasks = output1.floatArray
        val maskBytes = threshold(segmentationMasks)
        val maskBitmap = createMaskBitmap(maskBytes, 160, 160)
        // Imprimir los primeros 10 elementos de segmentationMasks


        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }*/
        val bestBox = BoundingBox(
            x1 = best.cx - best.w / 2,
            y1 = best.cy - best.h / 2,
            x2 = best.cx + best.w / 2,
            y2 = best.cy + best.h / 2,
            cx = best.cx,
            cy = best.cy,
            cls = 0,
            clsName = "Person",
            cnf = best.cnf,
            h = best.h,
            w = best.w

        )


        detectorListener.onDetect(bestBox, inferenceTime, matToBitmap(mask))
    }

    fun matToBitmap(mat: Mat): Bitmap {
        val bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
        if (mat.channels() == 1) {  // Grayscale to ARGB
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_GRAY2RGBA)
        }
        Utils.matToBitmap(mat, bitmap)
        return bitmap
    }

    private fun printFloatArray(array: FloatArray) {
        for (i in 0 until (array.size)) {
            println("Elemento $i: ${array[i]}")
        }
    }

    private fun reshapeOutput1(masks: FloatArray) : List<Mat> {
        val all = mutableListOf<Mat>()
        for (mask in 0 until 32) {
            val mat = Mat(160, 160, CvType.CV_32F)
            for (x in 0 until 160) {
                for (y in 0 until 160) {
                    mat.put(y, x, masks[ 32 * 160 *y + 32 *x + mask].toDouble())
                }
            }
            all.add(mat)
        }
        return all
    }
    /*private fun bestBox(array: FloatArray) : List<BoundingBox>? {

        val boundingBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            var maxConf = CONFIDENCE_THRESHOLD
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + numElements * j
            while (j < numChannel){
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += numElements
            }

            if (maxConf > CONFIDENCE_THRESHOLD && maxIdx == PERSON_CLASS_INDEX) {
                val clsName = labels[maxIdx]
                val cx = array[c] // 0
                val cy = array[c + numElements] // 1
                val w = array[c + numElements * 2]
                val h = array[c + numElements * 3]
                val x1 = cx - (w/2F)
                val y1 = cy - (h/2F)
                val x2 = cx + (w/2F)
                val y2 = cy + (h/2F)
                if (x1 < 0F || x1 > 1F) continue
                if (y1 < 0F || y1 > 1F) continue
                if (x2 < 0F || x2 > 1F) continue
                if (y2 < 0F || y2 > 1F) continue

                boundingBoxes.add(
                    BoundingBox(
                        x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                        cx = cx, cy = cy, w = w, h = h,
                        cnf = maxConf, cls = maxIdx, clsName = clsName
                    )
                )
            }
        }

        if (boundingBoxes.isEmpty()) return null

        return applyNMS(boundingBoxes)
    }

    private fun applyNMS(boxes: List<BoundingBox>) : MutableList<BoundingBox> {
        val sortedBoxes = boxes.sortedByDescending { it.cnf }.toMutableList()
        val selectedBoxes = mutableListOf<BoundingBox>()

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

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = maxOf(box1.x1, box2.x1)
        val y1 = maxOf(box1.y1, box2.y1)
        val x2 = minOf(box1.x2, box2.x2)
        val y2 = minOf(box1.y2, box2.y2)
        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = box1.w * box1.h
        val box2Area = box2.w * box2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
    }*/

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

    private fun Mat.toBitmap(): Bitmap {
        val outputBitmap = Bitmap.createBitmap(this.width(), this.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(this, outputBitmap)
        return outputBitmap
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
    val maskWeight: List<Float>
)
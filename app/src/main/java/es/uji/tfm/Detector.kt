package es.uji.tfm

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
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
    private var maskWidth = 0
    private var maskHeight = 0
    private var maskChannels = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

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

        numChannel = outputShape0[1]
        numElements = outputShape0[2]

        maskWidth = outputShape1[1]
        maskHeight = outputShape1[2]
        maskChannels = outputShape1[3]

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
        val maskProtoBuffer = TensorBuffer.createFixedSize(intArrayOf(1, maskWidth, maskHeight, maskChannels), OUTPUT_IMAGE_TYPE)

        val outputs = mapOf<Int, Any>(
            0 to coordinatesBuffer.buffer.rewind(),
            1 to maskProtoBuffer.buffer.rewind()
        )
        interpreter?.runForMultipleInputsOutputs(arrayOf(imageBuffer), outputs)

        val coordinates : FloatArray = coordinatesBuffer.floatArray
        val masks : FloatArray = maskProtoBuffer.floatArray

        val bestBoxes : List<BoundingBox> = findBestBoxes(coordinates)
        if (bestBoxes.isEmpty()) return
        val bestBox : BoundingBox = applyNMS(bestBoxes).sortedByDescending { it.cnf }[0]
        if (bestBox.cls >= 4) return // Remove background

        val reshapedMaskOutput : List<Mat> = reshapeOutput1(masks)
        val mask = Mat()
        Core.compare(reshapedMaskOutput[0], Scalar(0.5), mask, Core.CMP_GT)
        val maskBitmap : Bitmap = mask.toTransparentGreenBitmap()

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime
        detectorListener.onDetect(bestBox, inferenceTime, maskBitmap)
    }

    private fun Mat.toTransparentGreenBitmap(): Bitmap {
        val colorMat = Mat(this.size(), CvType.CV_8UC4)

        val greenMat = Mat(this.size(), CvType.CV_8UC1, Scalar(0.0))
        val alphaMat = Mat(this.size(), CvType.CV_8UC1, Scalar(255.0))

        Core.compare(this, Scalar(255.0), alphaMat, Core.CMP_EQ)
        Core.bitwise_not(alphaMat, alphaMat)
        greenMat.setTo(Scalar(255.0), alphaMat)

        val rgbaChannels = mutableListOf<Mat>()
        rgbaChannels.add(Mat.zeros(this.size(), CvType.CV_8UC1))
        rgbaChannels.add(greenMat)
        rgbaChannels.add(Mat.zeros(this.size(), CvType.CV_8UC1))
        rgbaChannels.add(alphaMat)

        Core.merge(rgbaChannels, colorMat)

        val bitmap = Bitmap.createBitmap(colorMat.cols(), colorMat.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(colorMat, bitmap)

        greenMat.release()
        alphaMat.release()
        for (channel in rgbaChannels) {
            channel.release()
        }
        colorMat.release()

        return bitmap
    }

    private fun reshapeOutput1(masks: FloatArray): List<Mat> {
        val all = mutableListOf<Mat>()

        for (maskIndex in 0 until maskChannels) {
            val mat = Mat(maskWidth, maskHeight, CvType.CV_32F)
            val buffer = FloatArray(maskWidth * maskHeight)

            for (i in 0 until maskWidth) {
                for (j in 0 until maskHeight) {
                    buffer[j * maskWidth + i] =  masks[maskChannels * maskHeight * j + maskChannels * i + maskIndex]
                }
            }

            mat.put(0, 0, buffer)
            all.add(mat)
        }
        return all
    }

    private fun findBestBoxes(coordinates: FloatArray) : List<BoundingBox> {
        val bestBoxes = mutableListOf<BoundingBox>()

        for (c in 0 until numElements) {
            val cnf = coordinates[c + numElements * 4]

            val confidences: MutableList<Float> = ArrayList()
            val start: Int = 4 + labels.size
            for (i in 4 until start) {
                confidences.add(coordinates[c + numElements * i])
            }
            var maxConfidence = -Float.MAX_VALUE
            var classId = -1
            for (i in confidences.indices) {
                if (confidences[i] > maxConfidence) {
                    maxConfidence = confidences[i]
                    classId = i
                }
            }

            if (maxConfidence > CONFIDENCE_THRESHOLD) {
                val cx = coordinates[c]
                val cy = coordinates[c + numElements]
                val w = coordinates[c + numElements * 2]
                val h = coordinates[c + numElements * 3]

                val x1 = cx - (w / 2F)
                val y1 = cy - (h / 2F)
                val x2 = cx + (w / 2F)
                val y2 = cy + (h / 2F)
                if (x1 < 0F || x1 > tensorWidth) continue
                if (y1 < 0F || y1 > tensorHeight) continue
                if (x2 < 0F || x2 > tensorWidth) continue
                if (y2 < 0F || y2 > tensorHeight) continue

                val maskWeight = mutableListOf<Float>()
                for (index in 0 until 32) {
                    maskWeight.add(coordinates[c + numElements * (index + 5)])
                }
                bestBoxes.add(BoundingBox(x1 = cx - w / 2, y1 = cy - h / 2, x2 = cx + w / 2, y2 = cy + h / 2, cx = cx, cy = cy, w = w, h = h, cnf = cnf, cls = classId, clsName = labels[classId], maskWeight = maskWeight))
            }
        }
        return bestBoxes
    }

    private fun applyNMS(bestOutput0: List<BoundingBox>): List<BoundingBox> {
        val sortedBoxes = bestOutput0.sortedByDescending { it.cnf }.toMutableList()
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
    private fun calculateIoU(b1: BoundingBox, b2: BoundingBox): Float {
        val x1 = maxOf(b1.cx - (b1.w/2F), b2.cx - (b2.w/2F))
        val y1 = maxOf(b1.cy - (b1.h/2F), b2.cy - (b2.h/2F))
        val x2 = minOf(b1.cx + (b1.w/2F), b2.cx + (b2.w/2F))
        val y2 = minOf(b1.cy + (b1.h/2F), b2.cy + (b2.h/2F))

        val intersectionArea = maxOf(0F, x2 - x1) * maxOf(0F, y2 - y1)
        val box1Area = b1.w * b1.h
        val box2Area = b2.w * b2.h
        return intersectionArea / (box1Area + box2Area - intersectionArea)
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
    }
}
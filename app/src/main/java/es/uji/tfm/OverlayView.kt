package es.uji.tfm

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat

class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results : BoundingBox? = null
    private var maskBitmap: Bitmap? = null
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()

    private var bounds = Rect()

    init {
        initPaints()
    }

    fun clear() {
        results = null
        maskBitmap = null
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 50f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 8F
        boxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        results?.let { boundingBox ->
            val left = (boundingBox.x1 * width)
            val top = (boundingBox.y1 * height)
            val right = (boundingBox.x2 * width)
            val bottom = (boundingBox.y2 * height)
            val boundingRect = RectF(left, top, right, bottom)

            if (maskBitmap != null) {
                val maskRect = Rect(
                    (left / width * maskBitmap!!.width).toInt(),
                    (top / height * maskBitmap!!.height).toInt(),
                    (right / width * maskBitmap!!.width).toInt(),
                    (bottom / height * maskBitmap!!.height).toInt()
                )
                val croppedMask = Bitmap.createBitmap(maskBitmap!!, maskRect.left, maskRect.top, maskRect.width(), maskRect.height())
                canvas.drawBitmap(croppedMask, null, boundingRect, null)
            }

            boxPaint.style = Paint.Style.STROKE
            canvas.drawRect(boundingRect, boxPaint)
            val drawableText = boundingBox.clsName ?: "Detected"

            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top - textHeight - BOUNDING_RECT_TEXT_PADDING,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING * 2,
                top,
                textBackgroundPaint
            )
            canvas.drawText(drawableText, left + BOUNDING_RECT_TEXT_PADDING, top - BOUNDING_RECT_TEXT_PADDING, textPaint)
        }
    }

    fun setResults(bestBox: BoundingBox, mask: Bitmap?) {
        results = bestBox
        maskBitmap = mask
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}

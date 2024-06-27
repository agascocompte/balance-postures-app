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

        results?.let { bbox ->
            val left = (bbox.x1 * width).toFloat()
            val top = (bbox.y1 * height).toFloat()
            val right = (bbox.x2 * width).toFloat()
            val bottom = (bbox.y2 * height).toFloat()
            val boundingRect = RectF(left, top, right, bottom)

            // Asumiendo que maskBitmap es la máscara completa
            if (maskBitmap != null) {
                // Calcula el rectángulo de recorte en las coordenadas de la máscara
                val maskRect = Rect(
                    (left / width * maskBitmap!!.width).toInt(),
                    (top / height * maskBitmap!!.height).toInt(),
                    (right / width * maskBitmap!!.width).toInt(),
                    (bottom / height * maskBitmap!!.height).toInt()
                )
                // Recorta la parte relevante de la máscara
                val croppedMask = Bitmap.createBitmap(maskBitmap!!, maskRect.left, maskRect.top, maskRect.width(), maskRect.height())
                // Dibuja la máscara recortada en su posición correspondiente sobre la imagen
                canvas.drawBitmap(croppedMask, null, boundingRect, null)
            }

            // Dibuja la caja delimitadora y la etiqueta si están disponibles
            boxPaint.style = Paint.Style.STROKE
            canvas.drawRect(boundingRect, boxPaint)
            val drawableText = bbox.clsName ?: "Detected"

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

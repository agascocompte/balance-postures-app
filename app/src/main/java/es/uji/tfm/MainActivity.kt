package es.uji.tfm

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.Menu
import android.view.MenuItem
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import es.uji.tfm.Constants.LABELS_PATH
import es.uji.tfm.Constants.MODEL_11L_PATH
import es.uji.tfm.Constants.MODEL_11M_PATH
import es.uji.tfm.Constants.MODEL_11N_PATH
import es.uji.tfm.Constants.MODEL_11S_PATH
import es.uji.tfm.Constants.MODEL_11X_PATH
import es.uji.tfm.Constants.MODEL_8L_PATH
import es.uji.tfm.Constants.MODEL_8M_PATH
import es.uji.tfm.Constants.MODEL_8N_PATH
import es.uji.tfm.Constants.MODEL_8S_PATH
import es.uji.tfm.Constants.MODEL_8X_PATH
import es.uji.tfm.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity(), Detector.DetectorListener {
    private lateinit var binding: ActivityMainBinding
    private var isFrontCamera = false
    private var showBBoxMask = false

    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var detector: Detector? = null

    private lateinit var cameraExecutor: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        setSupportActionBar(binding.toolbar)

        cameraExecutor = Executors.newSingleThreadExecutor()

        cameraExecutor.execute {
            detector = Detector(baseContext, MODEL_8X_PATH, LABELS_PATH, this)
            detector?.setup()
        }

        if (!OpenCVLoader.initLocal()) {
            Log.e("OpenCV", "Unable to load OpenCV!")
        } else {
            Log.d("OpenCV", "OpenCV loaded successfully!")
        }

        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS)
        }

        bindListeners()
    }

    private fun bindListeners() {
        binding.apply {
            showMask.setOnCheckedChangeListener { buttonView, isChecked ->
                showBBoxMask = isChecked
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider  = cameraProviderFuture.get()
            bindCameraUseCases()
        }, ContextCompat.getMainExecutor(this))
    }

    private fun bindCameraUseCases() {
        val cameraProvider = cameraProvider ?: throw IllegalStateException("Camera initialization failed.")

        val rotation = binding.viewFinder.display.rotation

        val cameraSelector = CameraSelector
            .Builder()
            .requireLensFacing(
                if (isFrontCamera) CameraSelector.LENS_FACING_FRONT
                else CameraSelector.LENS_FACING_BACK)
            .build()

        preview =  Preview.Builder()
                //.setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setResolutionSelector(ResolutionSelector.Builder().apply {
                setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY) }
                .build())
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer = ImageAnalysis.Builder()
            .setResolutionSelector(ResolutionSelector.Builder().apply {
                    setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY) }
                .build())
            //.setTargetAspectRatio(AspectRatio.RATIO_4_3)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .setTargetRotation(binding.viewFinder.display.rotation)
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .build()

        imageAnalyzer?.setAnalyzer(cameraExecutor) { imageProxy ->
            val bitmapBuffer =
                Bitmap.createBitmap(
                    imageProxy.width,
                    imageProxy.height,
                    Bitmap.Config.ARGB_8888
                )
            imageProxy.use { bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer) }
            imageProxy.close()

            val matrix = Matrix().apply {
                postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())

                if (isFrontCamera) {
                    postScale(
                        -1f,
                        1f,
                        imageProxy.width.toFloat(),
                        imageProxy.height.toFloat()
                    )
                }
            }

            val rotatedBitmap = Bitmap.createBitmap(
                bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height,
                matrix, true
            )

            detector?.detect(rotatedBitmap)
        }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(binding.viewFinder.surfaceProvider)
        } catch(exc: Exception) {
            Log.e(TAG, "Use case binding failed", exc)
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) {
        if (it[Manifest.permission.CAMERA] == true) { startCamera() }
    }

    override fun onDestroy() {
        super.onDestroy()
        detector?.close()
        cameraExecutor.shutdown()
    }

    override fun onResume() {
        super.onResume()
        if (allPermissionsGranted()){
            startCamera()
        } else {
            requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
        }
    }

    override fun onCreateOptionsMenu(menu: Menu): Boolean {
        menuInflater.inflate(R.menu.menu_main, menu)
        val menuItem = menu.findItem(R.id.action_model_selector)
        val spinner = menuItem.actionView as Spinner

        ArrayAdapter.createFromResource(
            this,
            R.array.model_options,
            R.layout.spinner_item
        ).also { adapter ->
            adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
            spinner.adapter = adapter
        }

        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View, position: Int, id: Long) {
                val modelName = parent.getItemAtPosition(position).toString()
                updateModel(modelName)
            }

            override fun onNothingSelected(parent: AdapterView<*>) {}
        }
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.action_switch_camera -> {
                isFrontCamera = !isFrontCamera
                startCamera()
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

    fun updateModel(modelName: String) {
        runOnUiThread {
            findViewById<ProgressBar>(R.id.progressBar).visibility = View.VISIBLE
            findViewById<TextView>(R.id.loadingText).visibility = View.VISIBLE
        }
        cameraExecutor.execute {
            detector?.close()
            detector = null
            detector = Detector(baseContext, getModelPath(modelName), LABELS_PATH, this)
            detector?.setup()
            runOnUiThread {
                findViewById<ProgressBar>(R.id.progressBar).visibility = View.GONE
                findViewById<TextView>(R.id.loadingText).visibility = View.GONE
            }
        }
    }

    private fun getModelPath(modelName: String): String {
        return when (modelName) {
                "YOLOv8n-seg" -> MODEL_8N_PATH
                "YOLOv11n-seg" -> MODEL_11N_PATH
                "YOLOv8s-seg" -> MODEL_8S_PATH
                "YOLOv11s-seg" -> MODEL_11S_PATH
                "YOLOv8m-seg" -> MODEL_8M_PATH
                "YOLOv11m-seg" -> MODEL_11M_PATH
                "YOLOv8l-seg" -> MODEL_8L_PATH
                "YOLOv11l-seg" -> MODEL_11L_PATH
                "YOLOv8x-seg" -> MODEL_8X_PATH
                "YOLOv11x-seg" -> MODEL_11X_PATH
            else -> MODEL_11N_PATH
        }
    }

    companion object {
        private const val TAG = "Camera"
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = mutableListOf (
            Manifest.permission.CAMERA
        ).toTypedArray()
    }

    override fun onEmptyDetect() {
        runOnUiThread {
            binding.overlay.clear()
            binding.label.text = getString(R.string.no_detection)
        }
    }

    @SuppressLint("SetTextI18n")
    override fun onDetect(bestBox: BoundingBox, inferenceTime: Long, mask: Bitmap) {
        runOnUiThread {
            binding.inferenceTime.text = "${inferenceTime}ms"
            if (showBBoxMask) {
                binding.overlay.setResults(bestBox, mask)
            } else {
                binding.overlay.clear()
            }
            binding.label.text = bestBox.clsName
        }
    }
}

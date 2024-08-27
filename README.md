# tfm_android (Balance Test Application)

This is an Android application developed in Kotlin designed to test and evaluate balance posture classification models based on YOLO8. The application enables real-time detection and classification of specific balance postures, such as feet-together, semi-tandem, tandem and no-balance, using the camera on a mobile device. This tool is intended for research and development in the field of frailty assessment and can be used to validate the performance of different deep learning models.

## Features

- **Real-Time Classification:** The application uses pre-trained YOLO8 models to classify balance postures in real-time.
- **Model Selection:** Users can choose between different trained models to compare their performance.
- **Segmentation Mode:** For models that include segmentation (e.g., YOLOv8), the application highlights the segmented region of the lower body.
- **Performance Monitoring:** The application displays inference time in milliseconds, allowing users to estimate the frames per second (FPS) performance of each model.
- **Camera Switching:** Users can switch between the front and rear cameras of the device.

## Installation

### Prerequisites

- Android Studio (latest version recommended)
- A compatible Android device for running the application (API level 24 or higher)
- Pre-trained YOLO models in the appropriate format (e.g., .tflite)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/agascocompte/tfm_android.git
   ```
   
2. **Open the project in Android Studio:**

- Open Android Studio.
- Select "Open an existing project" and navigate to the cloned directory.
- Wait for the Gradle sync to complete.

3. **Add YOLO models:**

- Place your pre-trained YOLO models (e.g., model.tflite) in the assets directory of the project.
- Ensure that the model paths are correctly referenced in the application's code.

4. **Run the application:**

- Connect your Android device via USB or start an emulator.
- Click on the "Run" button in Android Studio to install and launch the application.

## Usage

1. **Launch the application:** Open the app on your Android device.
2. **Select a model:** Choose from the available YOLO models to start the balance posture detection.
3. **Position the subject:** Ensure the person performing the balance test is within the camera's view.
4. **View results:** The application will display the detected balance posture in real-time, along with the inference time.
5. **Switch modes:** If using a YOLO model with segmentation, enable segmentation mode to view the highlighted regions.

## Contact

For any questions or inquiries, please contact [agasco@uji.es](agasco@uji.es).

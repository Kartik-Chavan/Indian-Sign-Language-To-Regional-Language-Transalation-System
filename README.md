
# Indian-Sign-Language-To-Regional-Language-Transalation-System

This project is a Sign Language Detection and Translation System that uses computer vision and machine learning to detect sign language gestures and translate them into multiple regional languages. The system is built using YOLO (You Only Look Once) for object detection and Google's Gemini API for generating coherent sentences from detected gestures. The application provides a user-friendly interface built with PyQt5, allowing users to interact with the system in real-time.

The system supports multiple languages, including English, Hindi, Marathi, Punjabi, and Tamil, and can translate detected gestures into meaningful sentences in the selected language. The project also includes tools for dataset creation and model training, making it a comprehensive solution for sign language recognition and translation.


## Features

### Real-Time Sign Language Detection:

Detects sign language gestures using a pre-trained YOLO model.

Displays the detected gestures in real-time with bounding boxes and labels.

### Multi-Language Translation:

Translates detected gestures into multiple regional languages (English, Hindi, Marathi, Punjabi, Tamil).

Uses Google's Gemini API to generate coherent sentences from the detected gestures.

### User-Friendly Interface:

Built with PyQt5, providing an intuitive and interactive GUI.

Allows users to select the target language, view detected gestures, and generate translations.

### Dataset Creation Tool:

A Python script (DatasetCreation.py) is provided to create custom datasets for training.

Captures images of gestures and organizes them into folders for each class.

### Model Training:

Includes Jupyter notebooks (Model_Training_V12.ipynb and Sign_Lang_YOLOV8.ipynb) for training YOLO models on custom datasets.

Supports both YOLOv8 and YOLOv12 for model training.

### Gesture Collage:

An images folder contains collages of all gestures, making it easy for users to understand and replicate the gestures.

## Project Structure

```Sign-Language-Detection/
├── app.py                   # Main application for sign language detection and translation
├── file.py                   # Secondary GUI application for sign language detection and translation
├── DatasetCreation.py       # Script for creating custom datasets
├── Model_Training_V12.ipynb # Jupyter notebook for training YOLOv12 models
├── Sign_Lang_YOLOV8.ipynb   # Jupyter notebook for training YOLOv8 models
├── images/                  # Folder containing gesture collages for reference
├── src/                     # Source code and resources
│   ├── scripts/             # Font files for different languages
│   └── final_training_Results/ # Pre-trained model weights
└── README.md                # Project documentation
```


## Requirements

To run this project, you need the following dependencies:

* Python 3.8 or higher

* OpenCV (cv2)

* PyQt5

* Ultralytics (for YOLO)

* Google Generative AI (google.generativeai)

* NumPy

* Pillow (PIL)

* Jupyter Notebook (for training models) or Also Compatible with Google Collab

You can install the required packages using the following command:

```python 
   pip install -r requirments.txt
   pip install google-generativeai 
```


## How To Use

 * **Clone the repository:**

    ```gh repo clone Kartik-Chavan/Indian-Sign-Language-To-Regional-Language-Transalation-System```
* Run the main application:

    ```python app.py```
The application will open a window with a live video feed from your webcam. Detected gestures will be displayed in real-time, and you can select the target language from the dropdown menu.

Click the **Translate Sentence** button to generate a coherent sentence from the detected gestures.

* **Creating a Custom Dataset:**
Run the dataset creation script:

     ```python DatasetCreation.py```

     Follow the on-screen instructions to enter class names (gestures) and capture images for each gesture.

The script will organize the captured images into folders for each class, which can be used for training.

* **Training the Model**
Open the Jupyter notebooks (Model_Training_V12.ipynb or Sign_Lang_YOLOV8.ipynb) to train the YOLO model on your custom dataset.

Follow the instructions in the notebook to load the dataset, configure the model, and start training.

Once trained, update the MODEL_PATH in app.py to use your custom model.
   
## Support Gestures

Supported Gestures
The system currently supports the following gestures:

**Numbers:** 1, 2, 3, 4, 5, 6, 8, 9

**Words:** Band Aid, Bye, Cough, Eat, Home, I, Like, Love, Request, Stop, What, When

Refer to the images folder for a visual collage of all supported gestures.

## Customization

* **Add New Gestures:** Use the DatasetCreation.py script to capture images for new gestures and train the model using the provided Jupyter notebooks.

* **Add New Languages:** Update the TRANSLATION_MAP and FONT_PATHS in app.py to support additional languages.

## License

This project is licensed under the [MIT](https://choosealicense.com/licenses/mit/) License.


## Acknowledgements

* **YOLO:** For providing a fast and accurate object detection model.

* **Google Gemini API:** For generating coherent sentences from detected gestures.

* **PyQt5:** For building the user-friendly interface.

## Support

For support, email kartikchavan6254@gmail.com or [Kartik Chavan](www.linkedin.com/in/kartik-chavan-b6b5b4282)


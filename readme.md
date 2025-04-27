# Garbage Detection

This project uses a custom-trained YOLOv8 model to detect various types of garbage items in a video feed.

## Features

- Detects multiple classes of garbage items using a YOLOv8 model.
- Processes video files.
- Draws bounding boxes and labels with confidence scores on detected items.
- Counts the number of detected garbage items in each frame.
- Configurable confidence threshold and minimum items for collection detection.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- cvzone
- Ultralytics (`ultralytics`)

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    ```

    - On Windows:
      ```bash
      .\.venv\Scripts\activate
      ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Ensure you have a video file for processing (e.g., in the `test/` directory) and the trained YOLOv8 model file (`model.pt`) in the root directory.
2.  Update the `video_path` and `model_path` variables in [`main.py`](main.py) if necessary.
3.  Modify the `classNames` list and `GARBAGE_CLASS_NAMES` set in [`main.py`](main.py) to match your specific model's output and desired garbage categories.
4.  Adjust `MIN_ITEMS_FOR_COLLECTION` and `CONFIDENCE_THRESHOLD` in [`main.py`](main.py) as needed.
5.  Run the main script:
    ```bash
    python main.py
    ```
6.  Press 'q' to quit the video display window.

## Model

The detection model (`model.pt`) is a YOLOv8 model trained on a custom dataset for garbage detection having around 8000 labeled images. The specific classes the model can detect are listed in the `classNames` variable within [`main.py`](main.py).

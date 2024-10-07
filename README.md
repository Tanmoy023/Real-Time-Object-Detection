# Real-Time Object Detection with YOLOv8, OpenCV, and cvzone

This project implements real-time object detection using the YOLOv8 model, OpenCV, and cvzone. It captures video from a webcam, detects objects, and displays bounding boxes with labels and confidence scores. You can also switch the input source to a video file.

## Features

- Real-time object detection from a webcam or video file.
- Uses YOLOv8 model from Ultralytics for object detection.
- Displays bounding boxes and labels with confidence scores.
- Supports a wide range of object categories using COCO dataset class labels.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- Ultralytics YOLO (`ultralytics`)
- cvzone (`cvzone`)
- A webcam (optional if using video input)

## Installation

- python -m venv myenv # To create a virtual environment
- myenv\Scripts\activate # To activate this virtual environment
- pip install opencv-python # install opencv library
- pip install ultralytics # install ultralytics
- pip install cvzone # Install cvzone

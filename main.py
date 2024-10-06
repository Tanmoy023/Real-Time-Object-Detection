# python -m venv myenv      # To create a virtual environment
# myenv\Scripts\activate      # To activate this virtual environment

from ultralytics import YOLO # < pip install ultralytics >
import cv2 # < pip install opencv-python >
import cvzone # < pip install cvzone > cvzone is use to display all the ditections using a good looking bounding boxex
import math

cap = cv2.VideoCapture(0) # create a webcam object. using internal webcam give '0' as argument and for external webcam give '1'
cap.set(3, 640) # set width
cap.set(4, 480) # set height

# cap = cv2.VideoCapture("video_data/children.mp4") # create a specific video object

# create model
model = YOLO('yolov8n.pt')

coco_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"] # those are the list of items which YOLO module can detect.

# open webcam
while True:
    success, img = cap.read() # take images using webcam
    result = model(img, stream=True) # using stream=True it will use generators

    # check for individual bounding boxes
    for r in result:
        boxes = r.boxes
        for box in boxes:

            # for use opencv
            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(f"x1:{x1}, y1:{y1} ,x2:{x2} ,y2:{y2}")
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255), 3) # cv2.rectangle(image,(x1,y1),(x2,y2),color,thickness)

            # for use cvzone
            x, y, x2, y2 = box.xyxy[0]
            w, h = x2-x, y2-y
            x, y, w, h = int(x), int(y), int(w), int(h)
            print(f"x:{x}, y:{y} ,w:{w} ,h:{h}")
            cvzone.cornerRect(img, (x, y, w, h))

            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(f"conf : {conf}")

            # class Name
            cls = int(box.cls[0])
            print(f"class name : {cls}")

            # preing object name and his confidence value
            cvzone.putTextRect(img, f'{coco_names[cls]}, {conf}',(max(0,x), max(35,y)), scale=1, thickness=1)

    cv2.imshow("image", img)
    cv2.waitKey(1) # wait for 1 mili-second
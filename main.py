import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import time  


# Load the video capture
cap = cv2.VideoCapture("cars.mp4", cv2.CAP_FFMPEG )

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Define the class names that the model can detect
className = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "dining table", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Load the mask image to define the region of interest
mask =cv2.imread("mask.png")


# tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)


# Define the coordinates of the counting line
limits = [400,297,673,297]
totalCount = []

# Initialize time variables for FPS calculation
prev_frame_time = 0
new_frame_time = 0


# Main loop
while True:
    success, frame =cap.read()

    if not success or frame is None:
        print("[Warning] Frame not read properly or end of video.")
        break
    
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # (width, height)

#  apply bitwise_and
    imgRegion = cv2.bitwise_and(frame, mask_resized)
    
    
    print("Frame Loaded:", success, "Frame shape:", None if frame is None else frame.shape)
    if not success:
      break

    # Load the graphics image
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(frame, imgGraphics, (0, 0))
    results = model(imgRegion, stream=True)


# Initialize an empty array to store the detections
    detections=np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:


        # bounding box
            x1, y1,x2, y2 = box.xyxy[0]
            x1, y1,x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h= x2 - x1 , y2 - y1
          
          # Get the confidence score of the detection
            conf = math.ceil((box.conf[0]*100))/100

            # Get the class ID of the detected object
            cls = int(box.cls[0])

            # Get the class name
            currentClass =className[cls]
            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or currentClass == "motorbike" and  conf > 0.5:
             currentArray = np.array([x1,y1,x2,y2,conf])
             detections = np.vstack((detections,currentArray))


  # Update the tracker with the new detections
    resultsTracker = tracker.update(detections)

    # Draw the counting line on the frame
    cv2.line(frame,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)


   # Iterate through the tracked objects
    for result in resultsTracker:
       x1,y1,x2,y2,id = result
       x1, y1,x2, y2 = int(x1), int(y1), int(x2), int(y2)

       
       print(result)
       w, h= x2 - x1 , y2 - y1
       cvzone.cornerRect(frame, (x1, y1,w,h),l=9, rt = 2, colorR=(255,0,255 ))
       cvzone.putTextRect(frame, f'{int(id)}',(max(0,x1), max(35,y1)), scale=2, thickness=3,offset=10)
       cx, cy = x1 + w // 2, y1 + h // 2
       cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
 
       if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(id) == 0:
                totalCount.append(id)

                # Change the color of the counting line to green
                cv2.line(frame, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)


   # Calculate FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps_text = "FPS = " + str(fps)  # Create the FPS text string

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape
    # Position the FPS text in the top-right corner
    text_x = frame_width - 200  
    text_y = 50  

    cv2.putText(frame, fps_text, (text_x, text_y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 100), 3)  # Display FPS

    # Display the total count of vehicles
    cv2.putText(frame,str(len(totalCount)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8) 

    # Show the processed frame
    cv2.imshow("Vehicle Detection & Counting", frame)
    
    # cv2.imshow("imgRegion", imgRegion)
    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break

            
cap.release()
cv2.destroyAllWindows()

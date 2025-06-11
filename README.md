# Vehicle_Detection_And_Counting

🚗 AI-Based Vehicle Detection and Counting System
This project is developed as part of an AI Internship, utilizing YOLOv8, OpenCV, and the SORT tracking algorithm to detect and count vehicles from video footage. The system is capable of detecting multiple vehicle classes and tracking them as they cross a predefined virtual line.

Features:
✅ Multi-object tracking with SORT
✅ Vehicle counting using a virtual line
✅ FPS (Frames per Second) display for performance monitoring
✅ Graphical overlay support


📁 Project Structure:
.
├── main.py                # Main application code
├── mask.png               # Mask image for the region of interest
├── graphics.png           # Graphics overlay image
├── yolov8n.pt             # Pretrained YOLOv8 model weights
├── sort.py                # SORT tracking algorithm
└── cars.mp4               # Input video file


🧠 Technologies Used:

YOLOv8 (Ultralytics)
OpenCV
cvzone
NumPy
SORT Tracking Algorithm

🖼️ Object Classes Detected

The YOLOv8 model can detect 80 COCO classes. In this project, we focus on counting:
car
truck
bus
motorbike
Only these vehicles are tracked and counted.


🎯 How It Works

Input video is read frame by frame.
Masking is applied to focus only on relevant regions of the frame.
YOLOv8 performs object detection on the masked region.
Detections are filtered by class and confidence threshold.
SORT tracks each object across frames.
A counting line is defined in the frame.
When a vehicle crosses the line, it's counted.
Final output displays bounding boxes, vehicle count, and FPS.



🛠️ Customization

Change the input video: Replace "cars.mp4" with any video file.
Adjust counting line: Modify the limits list in the script.
Update the mask: Change mask.png to reflect a different ROI.
Add/remove classes: Modify the conditional block for currentClass.


📷 Output Visualization

Tracked objects are labeled with unique IDs.
Vehicles crossing the line are counted once.
Bounding boxes are displayed using cvzone.
The FPS counter shows real-time performance.
A counter shows the total number of vehicles detected.

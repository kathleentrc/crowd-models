# pip install --upgrade ultralytics
from ultralytics import YOLO

# uncomment the code below to try yolo 11
# model = YOLO("models/yolo11n.pt")  
model = YOLO("models/yolov8n.pt")  

# run inference on the input image
results = model("raw_images/mrt.jpeg", show=True)
results[0].show()
"""
    File Name: image_processing.py
    Description: This file detects and blurs people using a pre-trained YOLOv8 model 
                 and gaussian blur from the OpenCV Library

    Documentation:
    YOLO:   https://docs.ultralytics.com/datasets/detect/coco/#citations-and-acknowledgments
    OpenCV: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
"""

import cv2 as cv
import os
from ultralytics import YOLO

# load YOLO model
model = YOLO("models/yolov8n.pt") 

# input and output path
image_path = "raw_images/cavite_extension.jpg"
output_path = "processed_images/cavite_extension_blurred.jpg"

# read
os.makedirs("processed_images", exist_ok=True)
img = cv.imread(image_path)
assert img is not None, f"could not read image: {image_path}"

# run inference
results = model(img)
estimated_count = 0

# gaussian blur pipeline
for box in results[0].boxes:
    cls_id = int(box.cls[0])                    # predicted class id
    conf = float(box.conf[0])                   # confidence score of detection
    
    if cls_id == 0:                             # 0 = person in the coco dataset
        estimated_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # get bounding box coodinates
        x1, y1 = max(x1, 0), max(y1, 0)         # clip coordinates to image dimensions
        x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])   

        """
        max(x1, 0)	            prevents negative left edge
        max(y1, 0)	            prevents negative top edge
        min(x2, img.shape[1])	prevents slicing outside the right boundary
        min(y2, img.shape[0])	prevents slicing below the image
        """

        person_region = img[y1:y2, x1:x2]                           # extract region of the detected person
        
        if person_region.size > 0:                                  # check first if region is not empty
            blurred = cv.GaussianBlur(person_region, (51, 51), 0)   # kernel should be positive and odd (see opencv docs)
        img[y1:y2, x1:x2] = blurred                                 # replace original region with blurred ver.

# display estimated count
print(f"People count: {estimated_count}")

# save output
cv.imwrite(output_path, img)
print(f"Saved blurred image to {output_path}")


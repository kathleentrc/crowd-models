import cv2 as cv
import os
from ultralytics import YOLO

def run_yolo_detection(image_path, model_path="models/yolov8n.pt"):
    """
    Load YOLO model, read the image, and run inference.

    YOLO:   https://docs.ultralytics.com/datasets/detect/coco/#citations-and-acknowledgments
    """
    # load YOLO model
    model = YOLO(model_path)

    # read
    img = cv.imread(image_path)
    assert img is not None, f"could not read image: {image_path}"

    # run inference
    results = model(img)
    return img, results

def blur_people_and_save(img, results, output_path):
    """
    This function takes the image and detection results, applies Gaussian blur to detected people,
    and saves the processed image to the output path.

    OpenCV: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    """
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
                img[y1:y2, x1:x2] = blurred                             # replace original region with blurred ver.

    # save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv.imwrite(output_path, img)
    return estimated_count

def print_estimated_count(estimated_count, output_path):
    """Display estimated count and where the blurred image was saved."""
    print(f"People count: {estimated_count}")
    print(f"Saved blurred image to {output_path}")
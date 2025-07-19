# pip install torch torchvision torchaudio

import os
import sys

# === ABSOLUTE FIX ===
# Add the full path to 'src' directly to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH = os.path.join(CURRENT_DIR, "src")
sys.path.append(SRC_PATH)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import cv2 as cv

from ultralytics import YOLO

# ✅ Now this will finally work:
from custom_model import CustomModel
from config import Config

sys.path.append(os.path.abspath('.'))

# Path Setup
image_path = "raw_images/cavite_extension.jpg"
output_path = "processed_images/cavite_extension_blurred.jpg"
os.makedirs("processed_images", exist_ok=True)

# Load YOLO model
yolo_model = YOLO("models/yolov8n.pt")

# Load image and run detection
img = cv.imread(image_path)
assert img is not None, f"could not read image: {image_path}"

results = yolo_model(img)
estimated_count = 0

# Gaussian blur for people in the image
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    if cls_id == 0:  # 0 = person
        estimated_count += 1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
        region = img[y1:y2, x1:x2]
        if region.size > 0:
            blurred = cv.GaussianBlur(region, (51, 51), 0)
            img[y1:y2, x1:x2] = blurred

cv.imwrite(output_path, img)
print(f"People count: {estimated_count}")
print(f"Saved blurred image to {output_path}")

# --- Embedding Extraction Setup ---

# Dummy label input (simulate user input)
crowding_label = "moderately congested"  # options: spacious, lightly congested, moderately congested, congested

# Text to index mapping
label_to_index = {
    "spacious": 0,
    "lightly congested": 1,
    "moderately congested": 2,
    "congested": 3,
}
text_tensor = torch.tensor([label_to_index[crowding_label]])

# Image preprocessing (must match your CustomModel's expected input format)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # or the size expected by your vision encoder
    transforms.ToTensor(),
])

image_tensor = transform(img).unsqueeze(0)  # Add batch dim

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)
text_tensor = text_tensor.to(device)

# Load custom model
model = CustomModel().to(device)
model.eval()

with torch.no_grad():
    image_embedding, text_embedding = model.encode(image_tensor, text_tensor)

print("Image embedding shape:", image_embedding.shape)
print("Text embedding shape:", text_embedding.shape)

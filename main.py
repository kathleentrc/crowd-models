from src.image_processing import run_yolo_detection, blur_people_and_save, print_estimated_count
import os

# Image and report input
image_paths = [
    "raw_images/cavite_extension.jpg",
    "raw_images/few_people.jpg",
    "raw_images/few_people2.jpg",
    "raw_images/empty_train.jpeg"
]

user_reports = [
    "lightly occupied",
    "spacious",
    "spacious",
    "spacious"
]

total_estimated_count = 0

# Process each image: detect and blur people
for image_path in image_paths:
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"processed_images/{filename}_blurred.jpg"
    img, results = run_yolo_detection(image_path)
    count = blur_people_and_save(img, results, output_path)
    total_estimated_count += count
    print_estimated_count(count, output_path)

print(f"\n[5-MINUTE WINDOW] Number of People in the Platform: {total_estimated_count}\n")

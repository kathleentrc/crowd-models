from src.image_processing import run_yolo_detection, blur_people_and_save, print_estimated_count

# input and output path
image_path = "raw_images/cavite_extension.jpg"
output_path = "processed_images/cavite_extension_blurred.jpg"

# pipeline
img, results = run_yolo_detection(image_path)
count = blur_people_and_save(img, results, output_path)
print_estimated_count(count, output_path)
"""
# main.py
    This is the starting point for the crowd density estimation application.
    This file is used for testing the methodolog pipeline
"""

from src.image_processing import run_yolo_detection, blur_people_and_save, print_estimated_count
from src.multimodal import early_fusion
from src.bayesian_neural_network import create_baseline_model, create_bnn_model, compute_predictions_classification, run_experiment, get_train_and_test_splits
from tensorflow import keras
import os
import numpy as np

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

img_count = []

total_estimated_count = 0


""" *******************************************************
        MODALITY-SPECIFIC PROCESSING (IMAGE REPORT)
******************************************************* """ 
# Process each image: detect and blur people
for image_path in image_paths:
    filename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"processed_images/{filename}_blurred.jpg"
    img, results = run_yolo_detection(image_path)
    count = blur_people_and_save(img, results, output_path)
    img_count.append(count)
    total_estimated_count += count
    print_estimated_count(count, output_path)

print(f"\n[5-MINUTE WINDOW] Number of People in the Platform: {total_estimated_count}\n")


""" ****************************************
              MULTIMODAL FUSION
**************************************** """ 
# Early fusion
fused_features = early_fusion(img_count, user_reports)
print(f"Fused features:\n {fused_features}")


""" ****************************************
           BAYESIAN NN CLASSIFIER
**************************************** """ 
# When performing classification, the average model prediction will give the relative 
# probability of each class, which can be considered a measure of uncertainty

# Load the data
csv_path = "csv_files/fused.csv" 
train_dataset, test_dataset, dataset_size, train_size = get_train_and_test_splits(
    csv_path, train_size_ratio=0.85, batch_size=16
)

print(f"Dataset size: {dataset_size}")
print(f"Train size: {train_size}")
print(f"Test size: {dataset_size - train_size}")

# Prepare test examples for prediction demonstration
sample = 10
examples_raw = list(test_dataset.unbatch().shuffle(100).batch(sample))[0]
examples, targets = examples_raw

# Examples are already in dictionary format from the fixed data loading

print("\n" + "="*50)
print("1. TRAINING BASELINE MODEL")
print("="*50)

num_epochs = 100
classification_loss = keras.losses.CategoricalCrossentropy()  # FIXED: Use correct loss

baseline_model = create_baseline_model()
run_experiment(baseline_model, classification_loss, train_dataset, test_dataset, num_epochs)

# Test baseline predictions
predicted = baseline_model(examples).numpy()
predicted_classes = np.argmax(predicted, axis=1)
actual_classes = np.argmax(targets.numpy(), axis=1)

print("\nBaseline Model Predictions:")
class_names = ['spacious', 'lightly_occupied', 'moderately_congested', 'congested']

# FIXED: Add debugging and bounds checking
print(f"Predicted shape: {predicted.shape}")
print(f"Predicted classes: {predicted_classes}")
print(f"Max predicted class: {np.max(predicted_classes)}")
print(f"Min predicted class: {np.min(predicted_classes)}")

for idx in range(sample):
# FIXED: Add bounds checking
    pred_class = predicted_classes[idx]
    actual_class = actual_classes[idx]
    
    # Ensure indices are within bounds
    if pred_class >= len(class_names):
        print(f"WARNING: Predicted class {pred_class} is out of bounds! Using class 3 instead.")
        pred_class = 3  # Default to last class
    
    if actual_class >= len(class_names):
        print(f"WARNING: Actual class {actual_class} is out of bounds! Using class 3 instead.")
        actual_class = 3  # Default to last class
    
    # FIXED: Ensure predicted array indexing is safe
    if pred_class < predicted.shape[1]:
        confidence = predicted[idx][pred_class]
    else:
        confidence = 0.0
        print(f"WARNING: Cannot access confidence for class {pred_class}")
    
    print(f"Predicted: {class_names[pred_class]} (confidence: {confidence:.3f}) - Actual: {class_names[actual_class]}")

print("\n" + "="*50)
print("2. TRAINING BAYESIAN NEURAL NETWORK")
print("="*50)

num_epochs = 500
bnn_model = create_bnn_model(train_size)
run_experiment(bnn_model, classification_loss, train_dataset, test_dataset)

# Test BNN predictions with uncertainty
print("\nBayesian Neural Network Predictions (with epistemic uncertainty):")
prediction_mean, prediction_std, predicted_classes_bnn = compute_predictions_classification(
    bnn_model, examples
)

for idx in range(sample):
    pred_class = predicted_classes_bnn[idx]
    actual_class = actual_classes[idx]
    
    # Ensure indices are within bounds
    if pred_class >= len(class_names):
        print(f"WARNING: BNN predicted class {pred_class} is out of bounds! Using class 3 instead.")
        pred_class = 3
    
    if actual_class >= len(class_names):
        print(f"WARNING: BNN actual class {actual_class} is out of bounds! Using class 3 instead.")
        actual_class = 3
    
    # FIXED: Safe indexing
    if pred_class < prediction_mean.shape[1]:
        confidence = prediction_mean[idx][pred_class]
        uncertainty = prediction_std[idx][pred_class]
    else:
        confidence = 0.0
        uncertainty = 0.0
        print(f"WARNING: Cannot access BNN confidence for class {pred_class}")
    
    print(
        f"Predicted: {class_names[pred_class]} "
        f"(confidence: {confidence:.3f} ± {uncertainty:.3f}) - "
        f"Actual: {class_names[actual_class]}"
    )

print("\n" + "="*50)
print("3. TRAINING PROBABILISTIC BAYESIAN NEURAL NETWORK")
print("="*50)

# Note: For simplicity, we'll use the regular BNN for classification
# Creating a proper probabilistic classifier requires more complex setup
print("Using regular BNN with uncertainty quantification for classification...")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("1. Baseline Model: Single point predictions with confidence scores")
print("2. BNN Model: Predictions with epistemic uncertainty (model uncertainty)")
print("3. For classification, uncertainty is expressed as confidence intervals")
print("   around the predicted class probabilities")
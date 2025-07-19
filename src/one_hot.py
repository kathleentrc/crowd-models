"""
# one_hot.py
    No need to run this file as it only contains functions for one-hot encoding 
    categorical labels of dummy data found in csv_files/bayesian_output.csv. 
    This code is not part of the main crowd forecasting pipeline. Only used for TESTING.
"""

import pandas as pd 

# Load data into DataFrame
df = pd.read_csv("../csv_files/bayesian_output.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Get unique labels from both predicted and actual columns
all_labels = set(df['predicted_label'].unique()) | set(df['actual_label'].unique())
print("Unique labels found:", sorted(all_labels))

# Create one-hot encoding for predicted labels (returns 1 or 0)
predicted_onehot = pd.get_dummies(df['predicted_label'], prefix='predicted')

# Create one-hot encoding for actual labels (returns 1 or 0)
actual_onehot = pd.get_dummies(df['actual_label'], prefix='actual')

# Combine original data with one-hot encoded columns
df_encoded = pd.concat([df, predicted_onehot, actual_onehot], axis=1)

# Display the results
print("\nOriginal DataFrame shape:", df.shape)
print("Encoded DataFrame shape:", df_encoded.shape)

print("\nFirst 10 rows of the encoded dataset:")
print(df_encoded.head(10))

print("\nOne-hot encoded columns:")
onehot_cols = [col for col in df_encoded.columns if col.startswith(('predicted_', 'actual_'))]
print("Predicted label columns:", [col for col in onehot_cols if col.startswith('predicted_')])
print("Actual label columns:", [col for col in onehot_cols if col.startswith('actual_')])

# Save to CSV
df_encoded.to_csv('../csv_files/encoded_bayesian_data.csv', index=False)

print(f"\nDataset summary:")
print(f"Total rows: {len(df_encoded)}")
print(f"Date range: {df_encoded['timestamp'].min()} to {df_encoded['timestamp'].max()}")
print(f"Unique labels: {len(all_labels)}")

# Show sample of the one-hot encoded data (showing 1s and 0s)
print("\nSample of one-hot encoded columns (binary 1 or 0 values):")
sample_cols = ['timestamp', 'confidence'] + onehot_cols
print(df_encoded[sample_cols].head(10))

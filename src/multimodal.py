"""
# multimodal.py
    This file implements early fusion of image features and user reports for multimodal analysis.
    It uses one-hot encoding for user reports and combines them with image features.
"""

from sklearn.preprocessing import OneHotEncoder
import numpy as np

def early_fusion(image_features, user_reports):
    categories = [["empty", "spacious", "lightly occupied", "crowded"]]
    encoder = OneHotEncoder(categories=categories, sparse_output=False, handle_unknown='ignore')
    
    text_features = encoder.fit_transform(np.array(user_reports).reshape(-1, 1))
    image_features = np.array(image_features).reshape(-1, 1)
    
    fused_features = np.hstack([image_features, text_features])
    return fused_features
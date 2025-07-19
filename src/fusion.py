from sklearn.preprocessing import OneHotEncoder
import numpy as np

def early_fusion(image_features, user_reports):
    encoder = OneHotEncoder(sparse_output=False)
    text_features = encoder.fit_transform(np.array(user_reports).reshape(-1, 1))
    fused_features = np.hstack([image_features, text_features])
    return fused_features
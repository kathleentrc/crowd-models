"""
# bayesian_neural_network.py
    This file implements Bayesian Neural Networks for crowd density label prediction.
    Adapted from: https://keras.io/examples/keras_recipes/bayesian_neural_networks/
"""

# pip install tensorflow
# pip install tensorflow-probability

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

# Load and prepare crowd data
def get_train_and_test_splits(csv_path, train_size_ratio=0.85, batch_size=32):
    """Load crowd data from CSV and create train/test splits"""
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Convert time to datetime and extract features
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['minute'] = df['time'].dt.minute
    df['day_of_week'] = df['time'].dt.dayofweek
    
    # Prepare features - we'll use crowd_count and time-based features
    feature_columns = ['crowd_count', 'hour', 'minute', 'day_of_week']
    
    # Create the dataset - FIXED: Convert to dictionary format for model inputs
    X = df[feature_columns].values.astype(np.float32)
    
    # Convert to dictionary format expected by the model
    X_dict = {}
    for i, feature_name in enumerate(feature_columns):
        X_dict[feature_name] = X[:, i:i+1]

    label_columns = ['spacious', 'lightly_occupied', 'moderately_congested', 'congested']
    y = df[label_columns].values.astype(np.float32)  # One-hot labels
    
    # Create TensorFlow dataset with dictionary inputs
    dataset = tf.data.Dataset.from_tensor_slices((X_dict, y))
    dataset = dataset.cache().prefetch(buffer_size=len(X))
    
    # Calculate split sizes
    dataset_size = len(X)
    train_size = int(dataset_size * train_size_ratio)
    
    # Split and batch
    train_dataset = (
        dataset.take(train_size)
        .shuffle(buffer_size=train_size)
        .batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).batch(batch_size)
    
    return train_dataset, test_dataset, dataset_size, train_size

# Configuration
FEATURE_NAMES = [
    "crowd_count",
    "hour", 
    "minute",
    "day_of_week"
]

hidden_units = [8, 8]
learning_rate = 0.001

def run_experiment(model, loss, train_dataset, test_dataset, num_epochs):
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.CategoricalAccuracy()],  # Fixed: Use accuracy for classification
    )

    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished.")
    
    # FIXED: Use accuracy instead of RMSE for classification
    _, accuracy = model.evaluate(train_dataset, verbose=0)
    print(f"Train Accuracy: {round(accuracy, 3)}")
    
    print("Evaluating model performance...")
    _, accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"Test Accuracy: {round(accuracy, 3)}")

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

# 1. Baseline Deterministic Model
def create_baseline_model():
    inputs = create_model_inputs()
    input_values = [value for _, value in sorted(inputs.items())]
    features = keras.layers.concatenate(input_values)
    features = layers.BatchNormalization()(features)
    
    # Create hidden layers with deterministic weights
    for units in hidden_units:
        features = layers.Dense(units, activation="sigmoid")(features)
    
    # FIXED: Ensure exactly 4 output units for 4 classes
    outputs = layers.Dense(units=4, activation="softmax", name="classification_output")(features)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    # FIXED: Print model summary to verify architecture
    print("Model architecture:")
    model.summary()
    
    return model

# 2. Bayesian Neural Network with Weight Uncertainty
# Define the prior weight distribution as Normal of mean=0 and stddev=1
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential([
        tfp.layers.DistributionLambda(
            lambda t: tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(n), scale_diag=tf.ones(n)
            )
        )
    ])
    return prior_model

# Define variational posterior weight distribution as multivariate Gaussian
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential([
        tfp.layers.VariableLayer(
            tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
        ),
        tfp.layers.MultivariateNormalTriL(n),
    ])
    return posterior_model

def create_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)
    
    # Create hidden layers with weight uncertainty using DenseVariational
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)
    
    # Classification output with softmax
    outputs = layers.Dense(units=4, activation="softmax")(features)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# 3. Probabilistic Bayesian Neural Network for Classification
def create_probabilistic_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)
    
    # Create hidden layers with weight uncertainty
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)
    
    # FIXED: For classification, we use Categorical distribution
    # Output logits for 4 classes
    logits = layers.Dense(units=4)(features)
    outputs = tfp.layers.IndependentBernoulli(4, convert_to_tensor_fn=tfp.distributions.Categorical.probs_parameter_supported)(logits)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# FIXED: Use proper loss function for probabilistic classification
def negative_loglikelihood_classification(targets, estimated_distribution):
    return -estimated_distribution.log_prob(tf.argmax(targets, axis=1))

def compute_predictions_classification(model, examples, iterations=100):
    """Compute predictions with uncertainty for BNN classification models"""
    predicted = []
    for _ in range(iterations):
        pred = model(examples).numpy()
        predicted.append(pred)
    predicted = np.array(predicted)  # Shape: (iterations, batch_size, num_classes)
    
    # FIXED: Add shape validation
    print(f"Prediction array shape: {predicted.shape}")
    
    # Compute statistics across iterations
    prediction_mean = np.mean(predicted, axis=0)  # Average probabilities
    prediction_std = np.std(predicted, axis=0)    # Standard deviation of probabilities
    
    # FIXED: Ensure argmax doesn't exceed bounds
    predicted_classes = np.argmax(prediction_mean, axis=1)  # Most likely class
    
    # FIXED: Clip predictions to valid range
    predicted_classes = np.clip(predicted_classes, 0, 3)
    
    return prediction_mean, prediction_std, predicted_classes
    
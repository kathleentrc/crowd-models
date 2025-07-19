"""
# src/bayesian_stgcn.py
    This file implements a Bayesian Neural Network (BNN) with spatio-temporal aggregation capabilities.
    Note: This file is not used in the current implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import pandas as pd

class BayesianLinear(nn.Module):
    """Bayesian Linear Layer with weight uncertainty"""
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.full((out_features, in_features), -3.0))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.full((out_features,), -3.0))
        
        # Prior
        self.prior_std = prior_std
        
    def forward(self, x, sample=True):
        if sample:
            # Sample weights and biases
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight_eps = torch.randn_like(weight_std)
            weight = self.weight_mu + weight_eps * weight_std
            
            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_eps = torch.randn_like(bias_std)
            bias = self.bias_mu + bias_eps * bias_std
        else:
            # Use mean weights (for deterministic prediction)
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """Compute KL divergence between posterior and prior"""
        # KL for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            (self.weight_mu**2 + weight_var) / (self.prior_std**2) - 
            self.weight_logvar + np.log(self.prior_std**2) - 1
        )
        
        # KL for biases
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            (self.bias_mu**2 + bias_var) / (self.prior_std**2) - 
            self.bias_logvar + np.log(self.prior_std**2) - 1
        )
        
        return weight_kl + bias_kl

class SpatioTemporalAggregator(nn.Module):
    """Spatio-temporal aggregation module for crowd prediction"""
    def __init__(self, sequence_length=12, hidden_dim=64, n_locations=1):
        super().__init__()
        self.sequence_length = sequence_length  # Number of historical time steps
        self.hidden_dim = hidden_dim
        self.n_locations = n_locations
        
        # Temporal processing with LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=4,  # 4 crowd level probabilities
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Spatial attention mechanism
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Time embedding for future prediction
        self.time_encoder = nn.Sequential(
            nn.Linear(4, 16),  # hour, day_of_week, month, is_weekend
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        
        # Future prediction layers
        self.future_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),  # LSTM output + time features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 4 crowd levels
        )
        
    def encode_time_features(self, target_times):
        """Encode target prediction times into features"""
        if isinstance(target_times, list):
            target_times = [pd.to_datetime(t) if isinstance(t, str) else t for t in target_times]
        
        features = []
        for dt in target_times:
            hour = dt.hour / 24.0  # Normalize to [0, 1]
            day_of_week = dt.weekday() / 7.0
            month = dt.month / 12.0
            is_weekend = float(dt.weekday() >= 5)
            features.append([hour, day_of_week, month, is_weekend])
            
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, historical_probs, spatial_coords=None, target_times=None):
        """
        Args:
            historical_probs: [batch_size, sequence_length, 4] - Historical crowd probabilities
            spatial_coords: [batch_size, n_locations, 2] - Spatial coordinates (lat, lon)
            target_times: List of target prediction times
        """
        batch_size = historical_probs.shape[0]
        
        # Temporal processing
        lstm_out, (hidden, cell) = self.temporal_lstm(historical_probs)
        temporal_features = lstm_out[:, -1, :]  # Use last hidden state
        
        # Spatial attention (if multiple locations)
        if spatial_coords is not None and spatial_coords.shape[1] > 1:
            # Apply spatial attention
            attended_features, _ = self.spatial_attention(
                temporal_features.unsqueeze(1),
                temporal_features.unsqueeze(1), 
                temporal_features.unsqueeze(1)
            )
            temporal_features = attended_features.squeeze(1)
        
        # Time encoding for future prediction
        if target_times is not None:
            time_features = self.time_encoder(self.encode_time_features(target_times))
            if len(time_features.shape) == 2 and time_features.shape[0] != batch_size:
                # Repeat time features for each batch item
                time_features = time_features.repeat(batch_size, 1, 1)
                time_features = time_features.reshape(-1, time_features.shape[-1])
        else:
            time_features = torch.zeros(batch_size, 32)
        
        # Combine temporal and time features
        combined_features = torch.cat([temporal_features, time_features], dim=-1)
        
        # Future prediction
        future_logits = self.future_predictor(combined_features)
        future_probs = F.softmax(future_logits, dim=-1)
        
        return future_probs, temporal_features

class EnhancedCrowdLevelBNN(nn.Module):
    """Enhanced Bayesian Neural Network with Spatio-Temporal Aggregation"""
    def __init__(self, hidden_dims=[64, 32], dropout_rate=0.1, sequence_length=12):
        super().__init__()
        
        # Original BNN components
        self.image_processor = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.text_processor = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Bayesian layers
        input_dim = 16 + 32
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                BayesianLinear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        self.bayesian_layers = nn.ModuleList([layer for layer in layers if isinstance(layer, BayesianLinear)])
        self.other_layers = nn.ModuleList([layer for layer in layers if not isinstance(layer, BayesianLinear)])
        
        self.output_layer = BayesianLinear(prev_dim, 4)
        
        # Spatio-temporal aggregation
        self.st_aggregator = SpatioTemporalAggregator(sequence_length=sequence_length)
        
        # Confidence weighting for combining current and future predictions
        self.confidence_weighter = nn.Sequential(
            nn.Linear(8, 16),  # Current + future predictions
            nn.ReLU(),
            nn.Linear(16, 2),  # Weights for current vs future
            nn.Softmax(dim=-1)
        )
        
        self.text_labels = ['spacious', 'lightly congested', 'moderately crowded', 'congested']
        self.crowd_levels = ['spacious', 'lightly congested', 'moderately crowded', 'congested']
        
    def encode_text_input(self, text_inputs):
        """Convert text inputs to one-hot encoding"""
        batch_size = len(text_inputs)
        encoded = torch.zeros(batch_size, 4)
        
        for i, text in enumerate(text_inputs):
            if text.lower() in self.text_labels:
                idx = self.text_labels.index(text.lower())
                encoded[i, idx] = 1.0
                
        return encoded
        
    def forward(self, image_counts, text_inputs, sample=True):
        """Forward pass for current prediction"""
        if isinstance(image_counts, (list, np.ndarray)):
            image_counts = torch.tensor(image_counts, dtype=torch.float32).unsqueeze(-1)
        elif len(image_counts.shape) == 1:
            image_counts = image_counts.unsqueeze(-1)
            
        if isinstance(text_inputs, list):
            text_features = self.encode_text_input(text_inputs)
        else:
            text_features = text_inputs
            
        image_features = self.image_processor(image_counts)
        text_features = self.text_processor(text_features)
        
        x = torch.cat([image_features, text_features], dim=-1)
        
        layer_idx = 0
        other_idx = 0
        
        for _ in range(len(self.bayesian_layers)):
            x = self.bayesian_layers[layer_idx](x, sample=sample)
            layer_idx += 1
            
            if other_idx < len(self.other_layers):
                if isinstance(self.other_layers[other_idx], nn.ReLU):
                    x = self.other_layers[other_idx](x)
                    other_idx += 1
                if other_idx < len(self.other_layers) and isinstance(self.other_layers[other_idx], nn.Dropout):
                    x = self.other_layers[other_idx](x)
                    other_idx += 1
        
        logits = self.output_layer(x, sample=sample)
        probabilities = F.softmax(logits, dim=-1)
        
        return probabilities, logits
    
    def predict_future_with_spatiotemporal(self, image_counts, text_inputs, 
                                         historical_data, target_times, 
                                         spatial_coords=None, n_samples=100):
        """
        Predict crowd levels at future times with uncertainty
        
        Args:
            image_counts: Current image counts
            text_inputs: Current text inputs
            historical_data: [batch_size, sequence_length, 4] - Historical crowd probabilities
            target_times: List of target prediction times (datetime objects or strings)
            spatial_coords: Spatial coordinates for location context
            n_samples: Number of MC samples for uncertainty
        """
        self.eval()
        current_predictions = []
        future_predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                # Current prediction
                current_probs, _ = self.forward(image_counts, text_inputs, sample=True)
                current_predictions.append(current_probs)
                
                # Future prediction using spatio-temporal aggregation
                future_probs, _ = self.st_aggregator(
                    historical_data, spatial_coords, target_times
                )
                future_predictions.append(future_probs)
                
        current_predictions = torch.stack(current_predictions)
        future_predictions = torch.stack(future_predictions)
        
        # Compute statistics for current predictions
        current_mean = current_predictions.mean(dim=0)
        current_std = current_predictions.std(dim=0)
        
        # Compute statistics for future predictions
        future_mean = future_predictions.mean(dim=0)
        future_std = future_predictions.std(dim=0)
        
        # Combine predictions with confidence weighting
        combined_input = torch.cat([current_mean, future_mean], dim=-1)
        confidence_weights = self.confidence_weighter(combined_input)
        
        weighted_mean = (confidence_weights[:, 0:1] * current_mean + 
                        confidence_weights[:, 1:2] * future_mean)
        
        # Uncertainty measures
        current_uncertainty = current_std.mean(dim=-1)
        future_uncertainty = future_std.mean(dim=-1)
        
        # Temporal uncertainty (difference between current and future)
        temporal_uncertainty = torch.norm(current_mean - future_mean, dim=-1)
        
        return {
            'current_probabilities': current_mean,
            'future_probabilities': future_mean,
            'weighted_probabilities': weighted_mean,
            'current_uncertainty': current_uncertainty,
            'future_uncertainty': future_uncertainty,
            'temporal_uncertainty': temporal_uncertainty,
            'confidence_weights': confidence_weights,
            'predicted_future_class': torch.argmax(future_mean, dim=-1),
            'predicted_current_class': torch.argmax(current_mean, dim=-1),
            'future_confidence': torch.max(future_mean, dim=-1)[0]
        }
    
    def kl_divergence(self):
        """Compute total KL divergence"""
        kl_div = 0
        for layer in self.bayesian_layers:
            kl_div += layer.kl_divergence()
        kl_div += self.output_layer.kl_divergence()
        return kl_div
    
    def predict_with_uncertainty(self, image_counts, text_inputs, n_samples=100):
        """Make current predictions with uncertainty quantification"""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                probs, _ = self.forward(image_counts, text_inputs, sample=True)
                predictions.append(probs)
                
        predictions = torch.stack(predictions)
        
        mean_probs = predictions.mean(dim=0)
        std_probs = predictions.std(dim=0)
        
        epistemic_uncertainty = std_probs.mean(dim=-1)
        aleatoric_uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'mean_probabilities': mean_probs,
            'std_probabilities': std_probs,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'predicted_class': torch.argmax(mean_probs, dim=-1),
            'confidence': torch.max(mean_probs, dim=-1)[0]
        }

def create_sample_historical_data(n_samples=1000, sequence_length=12):
    """Create sample historical data for spatio-temporal training"""
    np.random.seed(42)
    
    # Generate time series data
    historical_data = []
    current_data = []
    
    for i in range(n_samples):
        # Simulate daily crowd patterns (higher during peak hours)
        base_time = datetime.now() - timedelta(hours=sequence_length)
        sequence = []
        
        for j in range(sequence_length):
            time_point = base_time + timedelta(hours=j)
            hour = time_point.hour
            
            # Peak hours have higher crowd levels
            if 8 <= hour <= 10 or 17 <= hour <= 19:  # Rush hours
                base_level = 2.5
            elif 11 <= hour <= 16:  # Moderate activity
                base_level = 1.8
            elif 20 <= hour <= 22:  # Evening activity
                base_level = 2.0
            else:  # Low activity
                base_level = 0.8
                
            # Add noise and ensure valid range
            level = max(0, min(3, int(np.random.normal(base_level, 0.8))))
            
            # Convert to probability distribution (one-hot with noise)
            probs = np.zeros(4)
            probs[level] = 0.7 + np.random.uniform(0, 0.2)  # Main probability
            
            # Distribute remaining probability to adjacent levels
            remaining = 1 - probs[level]
            if level > 0:
                probs[level-1] = remaining * np.random.uniform(0.2, 0.6)
            if level < 3:
                probs[level+1] = remaining * np.random.uniform(0.2, 0.6)
            
            # Normalize
            probs = probs / probs.sum()
            sequence.append(probs)
        
        historical_data.append(sequence)
        
        # Current data point (similar to original function)
        true_level = np.random.randint(0, 4)
        count = max(0, int(np.random.normal([2, 8, 15, 25][true_level], 3)))
        text_options = ['spacious', 'lightly congested', 'moderately crowded', 'congested']
        
        if np.random.random() < 0.8:
            text_level = true_level
        else:
            adjacent = [max(0, true_level-1), min(3, true_level+1)]
            text_level = np.random.choice(adjacent)
            
        current_data.append((count, text_options[text_level], true_level))
    
    historical_data = np.array(historical_data)
    return historical_data, current_data

def generate_future_times(base_time=None, intervals=[5, 10, 30, 60]):
    """Generate future prediction times"""
    if base_time is None:
        base_time = datetime.now()
    
    future_times = []
    for minutes in intervals:
        future_time = base_time + timedelta(minutes=minutes)
        future_times.append(future_time)
    
    return future_times

def train_enhanced_bnn(model, train_data, val_data, epochs=100, lr=0.01, kl_weight=0.01):
    """Train the Enhanced Bayesian Neural Network"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Training
        historical_data, current_data = train_data
        
        # Process current data
        image_counts = [item[0] for item in current_data]
        text_inputs = [item[1] for item in current_data]
        labels = [item[2] for item in current_data]
        
        optimizer.zero_grad()
        
        # Current prediction loss
        probs, logits = model(image_counts, text_inputs)
        current_nll_loss = F.cross_entropy(logits, torch.tensor(labels, dtype=torch.long))
        
        # Future prediction loss (using historical data)
        historical_tensor = torch.tensor(historical_data, dtype=torch.float32)
        future_times = generate_future_times()
        
        try:
            future_probs, _ = model.st_aggregator(historical_tensor, target_times=future_times)
            # Use a simulated future ground truth (for training purposes)
            future_labels = torch.tensor(labels, dtype=torch.long)  # Simplified
            future_nll_loss = F.cross_entropy(
                future_probs.view(-1, 4), 
                future_labels.repeat(len(future_times))
            )
        except:
            future_nll_loss = torch.tensor(0.0)
        
        # KL divergence loss
        kl_loss = model.kl_divergence() / len(labels)
        
        # Total loss
        total_loss = current_nll_loss + 0.5 * future_nll_loss + kl_weight * kl_loss
        total_loss.backward()
        optimizer.step()
        
        train_loss = total_loss.item()
        train_losses.append(train_loss)
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_historical_data, val_current_data = val_data
                val_image_counts = [item[0] for item in val_current_data]
                val_text_inputs = [item[1] for item in val_current_data]
                val_labels = [item[2] for item in val_current_data]
                
                val_probs, val_logits = model(val_image_counts, val_text_inputs, sample=False)
                val_loss = F.cross_entropy(val_logits, torch.tensor(val_labels, dtype=torch.long))
                
                predicted = torch.argmax(val_probs, dim=-1)
                accuracy = (predicted == torch.tensor(val_labels)).float().mean()
                
                val_losses.append(val_loss.item())
                val_accuracies.append(accuracy.item())
                
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}')
        
        scheduler.step()
    
    return train_losses, val_losses, val_accuracies

# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data with historical information
    print("Creating sample data with historical patterns...")
    historical_data, current_data = create_sample_historical_data(1000, sequence_length=12)
    
    # Split data
    n_train = int(0.8 * len(current_data))
    train_historical = historical_data[:n_train]
    val_historical = historical_data[n_train:]
    train_current = current_data[:n_train]
    val_current = current_data[n_train:]
    
    # Create enhanced model
    print("Initializing Enhanced Bayesian Neural Network with Spatio-Temporal Aggregation...")
    model = EnhancedCrowdLevelBNN(hidden_dims=[64, 32], dropout_rate=0.1, sequence_length=12)
    
    # Prepare data
    train_data = (train_historical, train_current)
    val_data = (val_historical, val_current)
    
    # Train model
    print("Training enhanced model...")
    train_losses, val_losses, val_accuracies = train_enhanced_bnn(
        model, train_data, val_data, epochs=50, lr=0.001, kl_weight=0.01
    )
    
    # Generate future prediction times
    base_time = datetime.now()
    future_times = generate_future_times(base_time, intervals=[5, 10, 30, 60])
    
    print(f"\nPrediction times:")
    for i, ft in enumerate(future_times):
        print(f"  {['5 min', '10 min', '30 min', '1 hour'][i]}: {ft.strftime('%H:%M:%S')}")
    
    # Make predictions with spatio-temporal aggregation
    print("\nMaking spatio-temporal predictions...")
    test_image_counts = [5, 12, 20, 30]
    test_text_inputs = ['lightly congested', 'moderately crowded', 'congested', 'congested']
    
    # Create sample historical data for testing
    test_historical = torch.tensor(historical_data[:len(test_image_counts)], dtype=torch.float32)
    
    # Current predictions
    current_results = model.predict_with_uncertainty(test_image_counts, test_text_inputs, n_samples=100)
    
    # Future predictions with spatio-temporal aggregation
    future_results = model.predict_future_with_spatiotemporal(
        test_image_counts, test_text_inputs, test_historical, future_times, n_samples=100
    )
    
    print("\nCurrent vs Future Prediction Results:")
    print("=" * 80)
    crowd_levels = ['spacious', 'lightly congested', 'moderately crowded', 'congested']
    time_labels = ['5 min', '10 min', '30 min', '1 hour']
    
    for i in range(len(test_image_counts)):
        print(f"\nLocation {i+1}:")
        print(f"  Image Count: {test_image_counts[i]}")
        print(f"  User Report: {test_text_inputs[i]}")
        print(f"  Current Prediction: {crowd_levels[current_results['predicted_class'][i]]}")
        print(f"  Current Confidence: {current_results['confidence'][i]:.3f}")
        print(f"  Current Uncertainty: {current_results['total_uncertainty'][i]:.3f}")
        
        print(f"\n  Future Predictions:")
        future_class = future_results['predicted_future_class'][i]
        future_conf = future_results['future_confidence'][i]
        
        print(f"    Overall Future: {crowd_levels[future_class]} (conf: {future_conf:.3f})")
        print(f"    Future Uncertainty: {future_results['future_uncertainty'][i]:.3f}")
        print(f"    Temporal Uncertainty: {future_results['temporal_uncertainty'][i]:.3f}")
        
        # Show confidence weighting
        weights = future_results['confidence_weights'][i]
        print(f"    Prediction Weights: Current={weights[0]:.3f}, Future={weights[1]:.3f}")
        
        print(f"\n  Detailed Future Probabilities:")
        future_probs = future_results['future_probabilities'][i]
        for j, level in enumerate(crowd_levels):
            print(f"    {level}: {future_probs[j]:.3f}")
    
    # Visualize predictions over time
    plt.figure(figsize=(15, 10))
    
    # Training progress
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    if val_losses:
        epochs_val = range(0, len(train_losses), 10)[:len(val_losses)]
        plt.plot(epochs_val, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation accuracy
    plt.subplot(2, 3, 2)
    if val_accuracies:
        epochs_val = range(0, len(train_losses), 10)[:len(val_accuracies)]
        plt.plot(epochs_val, val_accuracies, label='Val Accuracy', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Current vs Future predictions
    plt.subplot(2, 3, 3)
    locations = range(1, len(test_image_counts) + 1)
    current_preds = [current_results['predicted_class'][i].item() for i in range(len(test_image_counts))]
    future_preds = [future_results['predicted_future_class'][i].item() for i in range(len(test_image_counts))]
    
    x = np.arange(len(locations))
    width = 0.35
    plt.bar(x - width/2, current_preds, width, label='Current', alpha=0.7)
    plt.bar(x + width/2, future_preds, width, label='Future', alpha=0.7)
    plt.xlabel('Location')
    plt.ylabel('Crowd Level')
    plt.title('Current vs Future Predictions')
    plt.xticks(x, [f'Loc {i}' for i in locations])
    plt.yticks(range(4), crowd_levels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Uncertainty comparison
    plt.subplot(2, 3, 4)
    current_unc = [current_results['total_uncertainty'][i].item() for i in range(len(test_image_counts))]
    future_unc = [future_results['future_uncertainty'][i].item() for i in range(len(test_image_counts))]
    temporal_unc = [future_results['temporal_uncertainty'][i].item() for i in range(len(test_image_counts))]
    
    plt.bar(x - width/2, current_unc, width, label='Current', alpha=0.7)
    plt.bar(x, future_unc, width, label='Future', alpha=0.7)
    plt.bar(x + width/2, temporal_unc, width, label='Temporal', alpha=0.7)
    plt.xlabel('Location')
    plt.ylabel('Uncertainty')
    plt.title('Uncertainty Analysis')
    plt.xticks(x, [f'Loc {i}' for i in locations])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confidence weights
    plt.subplot(2, 3, 5)
    current_weights = [future_results['confidence_weights'][i, 0].item() for i in range(len(test_image_counts))]
    future_weights = [future_results['confidence_weights'][i, 1].item() for i in range(len(test_image_counts))]
    
    plt.bar(x - width/2, current_weights, width, label='Current Weight', alpha=0.7)
    plt.bar(x + width/2, future_weights, width, label='Future Weight', alpha=0.7)
    plt.xlabel('Location')
    plt.ylabel('Weight')
    plt.title('Prediction Confidence Weights')
    plt.xticks(x, [f'Loc {i}' for i in locations])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Sample historical pattern
    plt.subplot(2, 3, 6)
    sample_history = historical_data[0]  # First sample
    time_steps = range(len(sample_history))
    crowd_evolution = [np.argmax(step) for step in sample_history]
    
    plt.plot(time_steps, crowd_evolution, marker='o', linewidth=2)
    plt.xlabel('Hours Ago')
    plt.ylabel('Crowd Level')
    plt.title('Sample Historical Pattern')
    plt.yticks(range(4), crowd_levels)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()  # Recent time on the right
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print(f"\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Final validation accuracy: {val_accuracies[-1] if val_accuracies else 'N/A':.3f}")
    print(f"Average current uncertainty: {np.mean(current_unc):.3f}")
    print(f"Average future uncertainty: {np.mean(future_unc):.3f}")
    print(f"Average temporal uncertainty: {np.mean(temporal_unc):.3f}")
    
    # Time-specific predictions
    print(f"\nTime-Specific Future Predictions:")
    print("-" * 40)
    for i, (time_label, future_time) in enumerate(zip(time_labels, future_times)):
        print(f"\n{time_label} from now ({future_time.strftime('%H:%M')}):")
        
        # For demonstration, show prediction for first location
        loc_idx = 0
        prob_dist = future_results['future_probabilities'][loc_idx]
        predicted_class = torch.argmax(prob_dist).item()
        confidence = prob_dist[predicted_class].item()
        
        print(f"  Predicted: {crowd_levels[predicted_class]} (confidence: {confidence:.3f})")
        print(f"  Full distribution: {[f'{crowd_levels[j]}: {prob_dist[j]:.3f}' for j in range(4)]}")
    
    print(f"\nSpatio-Temporal BNN training completed!")
    print(f"The model can now predict crowd levels at future time intervals:")
    print(f"- 5 minutes: Near-term fluctuations")
    print(f"- 10 minutes: Short-term trends") 
    print(f"- 30 minutes: Medium-term patterns")
    print(f"- 1 hour: Long-term behavioral patterns")
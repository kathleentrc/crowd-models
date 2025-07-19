import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

class CrowdLevelBNN(nn.Module):
    """Bayesian Neural Network for Crowd Level Prediction"""
    def __init__(self, hidden_dims=[64, 32], dropout_rate=0.1):
        super().__init__()
        
        # Input processing layers
        self.image_processor = nn.Sequential(
            nn.Linear(1, 16),  # Process image count
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Text embedding layer (more parameters for higher importance)
        self.text_processor = nn.Sequential(
            nn.Linear(4, 32),  # Process text features (one-hot encoded)
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 32),  # Additional layer for text importance
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Bayesian layers
        input_dim = 16 + 32  # image features + text features
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
        
        # Output layer (4 crowd levels)
        self.output_layer = BayesianLinear(prev_dim, 4)
        
        # Text label mapping
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
        """Forward pass through the network"""
        # Process inputs
        if isinstance(image_counts, (list, np.ndarray)):
            image_counts = torch.tensor(image_counts, dtype=torch.float32).unsqueeze(-1)
        elif len(image_counts.shape) == 1:
            image_counts = image_counts.unsqueeze(-1)
            
        if isinstance(text_inputs, list):
            text_features = self.encode_text_input(text_inputs)
        else:
            text_features = text_inputs
            
        # Process through input layers
        image_features = self.image_processor(image_counts)
        text_features = self.text_processor(text_features)
        
        # Combine features
        x = torch.cat([image_features, text_features], dim=-1)
        
        # Pass through Bayesian layers
        layer_idx = 0
        other_idx = 0
        
        for _ in range(len(self.bayesian_layers)):
            x = self.bayesian_layers[layer_idx](x, sample=sample)
            layer_idx += 1
            
            # Apply ReLU and Dropout
            if other_idx < len(self.other_layers):
                if isinstance(self.other_layers[other_idx], nn.ReLU):
                    x = self.other_layers[other_idx](x)
                    other_idx += 1
                if other_idx < len(self.other_layers) and isinstance(self.other_layers[other_idx], nn.Dropout):
                    x = self.other_layers[other_idx](x)
                    other_idx += 1
        
        # Output layer
        logits = self.output_layer(x, sample=sample)
        probabilities = F.softmax(logits, dim=-1)
        
        return probabilities, logits
    
    def kl_divergence(self):
        """Compute total KL divergence"""
        kl_div = 0
        for layer in self.bayesian_layers:
            kl_div += layer.kl_divergence()
        kl_div += self.output_layer.kl_divergence()
        return kl_div
    
    def predict_with_uncertainty(self, image_counts, text_inputs, n_samples=100):
        """Make predictions with uncertainty quantification"""
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                probs, _ = self.forward(image_counts, text_inputs, sample=True)
                predictions.append(probs)
                
        predictions = torch.stack(predictions)  # [n_samples, batch_size, n_classes]
        
        # Compute statistics
        mean_probs = predictions.mean(dim=0)
        std_probs = predictions.std(dim=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = std_probs.mean(dim=-1)
        
        # Aleatoric uncertainty (data uncertainty) - entropy of mean prediction
        aleatoric_uncertainty = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        # Total uncertainty
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

def create_sample_data(n_samples=1000):
    """Create sample training data"""
    np.random.seed(42)
    
    # Generate image counts with some correlation to crowd levels
    image_counts = []
    text_inputs = []
    crowd_labels = []
    
    text_options = ['spacious', 'lightly congested', 'moderately crowded', 'congested']
    
    for i in range(n_samples):
        # True crowd level
        true_level = np.random.randint(0, 4)
        
        # Image count with noise (YOLO might miss or double-count)
        base_counts = [2, 8, 15, 25]  # Expected counts for each level
        count = max(0, int(np.random.normal(base_counts[true_level], 3)))
        image_counts.append(count)
        
        # User text input (might not always match perfectly)
        # Give 80% chance of correct label, 20% chance of adjacent label
        if np.random.random() < 0.8:
            text_level = true_level
        else:
            # Adjacent level with some probability
            adjacent = [max(0, true_level-1), min(3, true_level+1)]
            text_level = np.random.choice(adjacent)
            
        text_inputs.append(text_options[text_level])
        crowd_labels.append(true_level)
    
    return np.array(image_counts), text_inputs, np.array(crowd_labels)

def train_bnn(model, train_data, val_data, epochs=100, lr=0.01, kl_weight=0.01):
    """Train the Bayesian Neural Network"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Training
        image_counts, text_inputs, labels = train_data
        optimizer.zero_grad()
        
        probs, logits = model(image_counts, text_inputs)
        
        # Negative log-likelihood loss
        nll_loss = F.cross_entropy(logits, torch.tensor(labels, dtype=torch.long))
        
        # KL divergence loss
        kl_loss = model.kl_divergence() / len(labels)  # Scale by batch size
        
        # Total loss
        total_loss = nll_loss + kl_weight * kl_loss
        total_loss.backward()
        optimizer.step()
        
        train_loss = total_loss.item()
        train_losses.append(train_loss)
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_image_counts, val_text_inputs, val_labels = val_data
                val_probs, val_logits = model(val_image_counts, val_text_inputs, sample=False)
                val_loss = F.cross_entropy(val_logits, torch.tensor(val_labels, dtype=torch.long))
                
                # Accuracy
                predicted = torch.argmax(val_probs, dim=-1)
                accuracy = (predicted == torch.tensor(val_labels)).float().mean()
                
                val_losses.append(val_loss.item())
                val_accuracies.append(accuracy.item())
                
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.4f}')
        
        scheduler.step()
    
    return train_losses, val_losses, val_accuracies

# Example usage and demonstration
if __name__ == "__main__":
    # Create sample data
    print("Creating sample data...")
    image_counts, text_inputs, labels = create_sample_data(1000)
    
    # Split data
    train_img, val_img, train_text, val_text, train_labels, val_labels = train_test_split(
        image_counts, text_inputs, labels, test_size=0.2, random_state=42
    )
    
    # Create model
    print("Initializing Bayesian Neural Network...")
    model = CrowdLevelBNN(hidden_dims=[64, 32], dropout_rate=0.1)
    
    # Prepare data
    train_data = (train_img, train_text, train_labels)
    val_data = (val_img, val_text, val_labels)
    
    # Train model
    print("Training model...")
    train_losses, val_losses, val_accuracies = train_bnn(
        model, train_data, val_data, epochs=50, lr=0.001, kl_weight=0.01
    )
    
    # Make predictions with uncertainty
    print("\nMaking predictions with uncertainty quantification...")
    test_image_counts = [5, 12, 20, 30]
    test_text_inputs = ['lightly congested', 'moderately crowded', 'congested', 'congested']
    
    results = model.predict_with_uncertainty(test_image_counts, test_text_inputs, n_samples=100)
    
    print("\nPrediction Results:")
    print("-" * 60)
    crowd_levels = ['spacious', 'lightly congested', 'moderately crowded', 'congested']
    
    for i in range(len(test_image_counts)):
        print(f"\nInput {i+1}:")
        print(f"  Image Count: {test_image_counts[i]}")
        print(f"  User Report: {test_text_inputs[i]}")
        print(f"  Predicted Class: {crowd_levels[results['predicted_class'][i]]}")
        print(f"  Confidence: {results['confidence'][i]:.3f}")
        print(f"  Total Uncertainty: {results['total_uncertainty'][i]:.3f}")
        print(f"  Epistemic Uncertainty: {results['epistemic_uncertainty'][i]:.3f}")
        print(f"  Aleatoric Uncertainty: {results['aleatoric_uncertainty'][i]:.3f}")
        print("  Class Probabilities:")
        for j, level in enumerate(crowd_levels):
            mean_prob = results['mean_probabilities'][i, j]
            std_prob = results['std_probabilities'][i, j]
            print(f"    {level}: {mean_prob:.3f} ± {std_prob:.3f}")
    
    # Visualize training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', alpha=0.7)
    if val_losses:
        epochs_val = range(0, len(train_losses), 10)[:len(val_losses)]
        plt.plot(epochs_val, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if val_accuracies:
        epochs_val = range(0, len(train_losses), 10)[:len(val_accuracies)]
        plt.plot(epochs_val, val_accuracies, label='Val Accuracy', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal validation accuracy: {val_accuracies[-1]:.3f}")
    print("Training completed!")
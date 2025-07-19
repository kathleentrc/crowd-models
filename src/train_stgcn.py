"""
# train_stgcn.py
    This file contains functions for training the Spatio-Temporal Graph Convolutional Network (ST-GCN) model.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

# (Helper functions and DataProcessor are unchanged)
def get_crowd_level_label(person_count):
    if person_count <= 20: return "Spacious"
    elif person_count <= 100: return "Lightly Occupied"
    elif person_count <= 250: return "Moderately Crowded"
    else: return "Packed/Congested"

class DataProcessor:
    def __init__(self, csv_path, column_type='_count'):
        self.df = pd.read_csv(csv_path)
        self.time_series_data = self.df.filter(like=column_type).values
        self.min_val = self.time_series_data.min()
        self.max_val = self.time_series_data.max()
    def get_scaled_data(self):
        scaled_data = (self.time_series_data - self.min_val) / (self.max_val - self.min_val)
        return torch.FloatTensor(scaled_data)
    def unscale_prediction(self, scaled_prediction):
        unscaled = scaled_prediction * (self.max_val - self.min_val) + self.min_val
        return unscaled

def create_sliding_windows(data, sequence_length, prediction_horizon):
    X, y = [], []
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        input_seq = data[i : i + sequence_length]
        target_val = data[i + sequence_length + prediction_horizon - 1]
        X.append(input_seq)
        y.append(target_val)
    return torch.stack(X), torch.stack(y)

# --- MODEL DEFINITION (SIGNIFICANTLY UPGRADED) ---
class STGCN(nn.Module):
    """A deeper STGCN with two GCN layers and Dropout for better learning."""
    def __init__(self, num_nodes, in_channels, gcn_hidden_features, gru_out_features, num_gru_layers=1, dropout_rate=0.3):
        super(STGCN, self).__init__()
        
        # A deeper spatial module
        self.gcn1 = GCNConv(in_channels, gcn_hidden_features)
        self.gcn2 = GCNConv(gcn_hidden_features, gru_out_features)
        
        # Temporal module
        self.gru = nn.GRU(
            input_size=num_nodes * gru_out_features,
            hidden_size=num_nodes * gru_out_features,
            num_layers=num_gru_layers,
            batch_first=True
        )
        
        # Output layer
        self.linear = nn.Linear(
            in_features=num_nodes * gru_out_features,
            out_features=num_nodes
        )
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, edge_index):
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # Process through GCN layers
        x_reshaped = x.view(batch_size * seq_len, num_nodes, -1)
        
        gcn_out1 = torch.relu(self.gcn1(x_reshaped, edge_index))
        gcn_out1 = self.dropout(gcn_out1) # Apply dropout
        
        gcn_out2 = torch.relu(self.gcn2(gcn_out1, edge_index))
        
        # Reshape for GRU
        gru_input = gcn_out2.view(batch_size, seq_len, -1)
        gru_output, _ = self.gru(gru_input)
        
        # Get the output from the last time step and apply dropout
        last_time_step_output = self.dropout(gru_output[:, -1, :])
        
        # Final prediction
        output = self.linear(last_time_step_output)
        return output

# --- TRAINING SCRIPT ---
if __name__ == '__main__':
    STATION_NAMES = ["Taft Avenue Station", "Ayala Station", "Cubao Station"]
    NUM_NODES = len(STATION_NAMES)
    SEQUENCE_LENGTH = 4
    PREDICTION_HORIZON = 1
    
    # Give the more powerful model more time to train
    EPOCHS = 700
    LEARNING_RATE = 0.001

    data_processor = DataProcessor('crowd_network_data.csv', column_type='_count')
    scaled_time_series = data_processor.get_scaled_data()
    
    adjacency_matrix = torch.FloatTensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
    edge_index = adjacency_matrix.nonzero().t().contiguous()

    X, y = create_sliding_windows(scaled_time_series, SEQUENCE_LENGTH, PREDICTION_HORIZON)
    
    # --- Initialize the new, more powerful model ---
    model = STGCN(
        num_nodes=NUM_NODES, 
        in_channels=1, 
        gcn_hidden_features=16, # Neurons in the first GCN layer
        gru_out_features=32     # Neurons in the second GCN layer and for the GRU
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("--- Starting Training for DEEPER AI Model ---")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X.unsqueeze(-1), edge_index)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}')
    
    print("--- Training Complete ---")
    
    MODEL_SAVE_PATH = "stgcn_model.pth"
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    model.eval()
    with torch.no_grad():
        last_scaled_sequence = scaled_time_series[-SEQUENCE_LENGTH:].unsqueeze(0)
        scaled_prediction = model(last_scaled_sequence.unsqueeze(-1), edge_index)
        final_prediction = data_processor.unscale_prediction(scaled_prediction)
        
        print("\n--- Example Forecast ---")
        print("Given the last 4 hours of data, the predicted CROWD LEVEL for the next hour is:")
        for i, name in enumerate(STATION_NAMES):
            count = int(final_prediction[0, i])
            label = get_crowd_level_label(count)
            print(f"- {name}: ~{count} people ({label})")
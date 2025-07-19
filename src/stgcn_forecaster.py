"""
# stgcn_forecaster.py
    This file contains functions for building and training the Spatio-Temporal Graph Convolutional Network (ST-GCN) model.
"""

import torch
import pandas as pd
from datetime import datetime, timedelta

from train_stgcn import STGCN, DataProcessor, get_crowd_level_label

def generate_stgcn_forecast(master_data_file, model_path, station_names, sequence_length):
    """
    Generates a forecast using the single master data file and the deep model.
    """
    print("\n" + "#"*70)
    print("#" + " "*12 + "CROWDCAST (ST-GCN)" + " "*13 + "#")
    print("#"*70)

    try:
        num_nodes = len(station_names)
        
        # Initialize the model with all the required arguments to match the trained version
        model = STGCN(
            num_nodes=num_nodes, 
            in_channels=1, 
            gcn_hidden_features=16,
            gru_out_features=32
        )
        
        model.load_state_dict(torch.load(model_path))
        model.eval()

        crowd_data_processor = DataProcessor(master_data_file, column_type='_count')
        scaled_crowd_series = crowd_data_processor.get_scaled_data()

        if len(scaled_crowd_series) < sequence_length:
            print(f"Not enough historical data to predict.")
            return

        df_master = pd.read_csv(master_data_file)
        df_master['timestamp'] = pd.to_datetime(df_master['timestamp'])
        df_master['day_of_week'] = df_master['timestamp'].dt.day_name()

        input_sequence = scaled_crowd_series[-sequence_length:]
        predicted_counts = []
        for _ in range(8):
            model_input = input_sequence[-sequence_length:].unsqueeze(0).unsqueeze(-1)
            adjacency_matrix = torch.FloatTensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
            edge_index = adjacency_matrix.nonzero().t().contiguous()
            with torch.no_grad():
                scaled_pred = model(model_input, edge_index)
            unscaled_pred = crowd_data_processor.unscale_prediction(scaled_pred)
            predicted_counts.append(unscaled_pred.squeeze(0))
            input_sequence = torch.cat([input_sequence[1:], scaled_pred.view(1, num_nodes)], dim=0)

        print("Forecast for the upcoming hours:")
        print("-----------------------------------------------------------------------------------------")
        header = f"{'Time':<10}"
        for name in station_names:
            header += f"| {name:<25}"
        print(header)
        print("-----------------------------------------------------------------------------------------")

        start_time = pd.to_datetime(df_master['timestamp'].iloc[-1])
        for i, pred_counts in enumerate(predicted_counts):
            forecast_time = start_time + timedelta(hours=i + 1)
            time_str = forecast_time.strftime('%I:%M %p')
            day_name = forecast_time.day_name()
            line = f"{time_str:<10}"
            
            for j, station_name in enumerate(station_names):
                count = int(pred_counts[j])
                label = get_crowd_level_label(count).lower()
                
                confidence_col_name = f"{station_name}_confidence"
                day_specific_data = df_master[df_master['day_of_week'] == day_name]
                historical_confidence = day_specific_data[confidence_col_name].mean()
                
                if pd.isna(historical_confidence):
                    historical_confidence = df_master[confidence_col_name].mean()
                
                confidence_str = f"{int(historical_confidence)}%" if not pd.isna(historical_confidence) else "??%"
                line += f"| {confidence_str} {label}"
            print(line)

        print("#"*70)

    except FileNotFoundError:
        print(f"A required file was not found.")
    except Exception as e:
        print(f"An error occurred during AI forecast generation: {e}")

# Define a preprocessing script that will run in the pipeline
# This script will take the data in the feature store, split it, and transform
# it into the format expected by the model

import json
import argparse
import os
import requests
import tempfile

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Select features to use in the model
def get_store_features(row):
    return [
      row["sales"], 
      row["oil"], 
      row["onpromotion"],
      row["is_holiday"], 
      row["hash_0"], 
      row["hash_1"], 
      row["hash_2"], 
      row["hash_3"], 
      row["hash_4"], 
      row["hash_5"], 
      row["hash_6"], 
      row["hash_7"], 
      row["hash_8"], 
      row["hash_9"], 
      row["month_cos"],
      row["month_sin"],
      # row["day_cos"],
      # row["day_sin"],
      row["dow_cos"],
      row["dow_sin"]
]

# Split the data sets into input windows and associated targets
def generate_windows(data, input_seq_length, target_seq_length, stride):
    windows = []
    targets = []
    num_days = data.shape[1]
    
    for i in range(0, num_days, stride):
        if (i+input_seq_length+target_seq_length) <= num_days:
            input_window_end = i + input_seq_length
            target_window_end = input_window_end + target_seq_length
            
            input_window = data[:, i:input_window_end, :]
            target_window = data[:, input_window_end:target_window_end, 0]
            
            windows.append(input_window)
            targets.append(target_window)
            
    return np.array(windows), np.array(targets)



if __name__ == "__main__":
    
    # Base directory inside the pipeline
    base_dir = "/opt/ml/processing"
    
    # Load the data
    df = pd.read_csv(f"{base_dir}/input/input_data.csv", index_col=0)
    
    # Apply feature selection function
    df["features"] = df.apply(get_store_features, axis=1)
    num_continuous_features = 3
    
    # Drop uneeded columns
    drop_columns = [col for col in df.columns if col not in ["date", "store_nbr", "features"]]
    df.drop(columns=drop_columns, inplace=True)
    
    # Pivot the data to be in the format (store number, date, features)
    df_pivoted = df.pivot(index="store_nbr", columns="date", values="features")
    
    # Convert the data to an array
    stacked_df = np.array(df_pivoted.values.tolist())
    
    # Split the data into test/train/val sets with a 80/10/10 split
    n = stacked_df.shape[1]
    train_data = stacked_df[:, :int(n*0.8), :]
    test_data = stacked_df[:, int(n*0.8):int(n*0.9), :]
    val_data = stacked_df[:, int(n*0.9):-7, :]
    
    # Withold the last 7 days of the data for forecasting
    forecast_data = stacked_df[:, -7:, :]
    
    # Get the mean and standard deviation for normalization
    scaler = StandardScaler()

    # Flatten the first 2 dimensions into (stores*instances, features)
    train_data_2d = train_data.reshape(-1, train_data.shape[2])
    test_data_2d = test_data.reshape(-1, test_data.shape[2])
    val_data_2d = val_data.reshape(-1, val_data.shape[2])
    forecast_data_2d = forecast_data.reshape(-1, forecast_data.shape[2])

    # Scale just the continuous features
    train_data_2d[:, :num_continuous_features] = scaler.fit_transform(train_data_2d[:, :num_continuous_features])
    test_data_2d[:, :num_continuous_features] = scaler.transform(test_data_2d[:, :num_continuous_features])
    val_data_2d[:, :num_continuous_features] = scaler.transform(val_data_2d[:, :num_continuous_features])
    forecast_data_2d[:, :num_continuous_features] = scaler.transform(forecast_data_2d[:, :num_continuous_features])

    # Add Gaussian noise to the continuous features
    train_data_2d[:, :num_continuous_features] = train_data_2d[:, :num_continuous_features] + np.random.normal(0, 0.2, train_data_2d[:, :num_continuous_features].shape)
    test_data_2d[:, :num_continuous_features] = test_data_2d[:, :num_continuous_features] + np.random.normal(0, 0.2, test_data_2d[:, :num_continuous_features].shape)
    val_data_2d[:, :num_continuous_features] = val_data_2d[:, :num_continuous_features] + np.random.normal(0, 0.2, val_data_2d[:, :num_continuous_features].shape)
    forecast_data_2d[:, :num_continuous_features] = forecast_data_2d[:, :num_continuous_features] + np.random.normal(0, 0.2, forecast_data_2d[:, :num_continuous_features].shape)

    # Reshape the data back to its original dimensions
    train_data = train_data_2d.reshape(train_data.shape)
    test_data = test_data_2d.reshape(test_data.shape)
    val_data = val_data_2d.reshape(val_data.shape)
    forecast_data = forecast_data_2d.reshape(forecast_data.shape)
    
    # Generate windows for train/test/val sets
    input_seq_length = 7
    target_seq_length = 1
    stride = 1

    # Create the input and target windows for the data splits
    train_inputs, train_targets = generate_windows(train_data, input_seq_length, target_seq_length, stride)
    print(f"Train inputs shape: {train_inputs.shape}")
    print(f"Train targets shape: {train_targets.shape}")

    test_inputs, test_targets = generate_windows(test_data, input_seq_length, target_seq_length, stride)
    print(f"Test inputs shape: {test_inputs.shape}")
    print(f"Test targets shape: {test_targets.shape}")

    val_inputs, val_targets = generate_windows(val_data, input_seq_length, target_seq_length, stride)
    print(f"Validation inputs shape: {val_inputs.shape}")
    print(f"Validation inputs shape: {val_targets.shape}")
    
    # Save data splits
    np.save(f"{base_dir}/train/train_inputs.npy", train_inputs)
    np.save(f"{base_dir}/train/train_targets.npy", train_targets)

    np.save(f"{base_dir}/test/test_inputs.npy", test_inputs)
    np.save(f"{base_dir}/test/test_targets.npy", test_targets)

    np.save(f"{base_dir}/validation/val_inputs.npy", val_inputs)
    np.save(f"{base_dir}/validation/val_targets.npy", val_targets)

    # Save the evaluation data for the batch transform evaluation job
    with open(f"{base_dir}/transform-input/validation_data.ndjson", "w") as f:
        for i, window in enumerate(val_inputs):
            instance = {"input_1": window.tolist()}
            json_line = json.dumps(instance)
            if i < len(val_inputs) - 1:
                f.write(json_line + "\n")
            else:
                f.write(json_line)
    
    
    # Save the forecasting data
    with open(f"{base_dir}/forecast-input/forecast_data.ndjson", "w") as f:
        instance = {"input_1": forecast_data.tolist()}
        json_line = json.dumps(instance)
        f.write(json_line)


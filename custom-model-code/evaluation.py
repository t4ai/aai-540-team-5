
# Define an evaluation script that will run in the pipeline
# This script evaluates the predictions made on the validation set and
# logs an evaluation report with RMSE and MAE scores

import os
import json
import pathlib
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


if __name__ == "__main__":
    
    # Set base directory inside the pipeline
    base_dir = "/opt/ml/processing"
    print(os.getcwd())
    
    # Load validation set true target values
    val_targets = np.load(os.path.join(f"{base_dir}/validation", "val_targets.npy"))
    print(val_targets.shape)
    
    # Load predictions from the batch transform job
    with open(f"{base_dir}/transform-results/validation_data.ndjson.out", "r") as f:
        predictions = []
        for line in f:
            obj = json.loads(line.strip())
            predictions.extend(obj["predictions"])
    
    # Convert the predictions back into a numpy array
    predictions_array = np.array(predictions)
    print(predictions_array.shape)
    
    # Flatten the targets and predictions for computing metrics
    targets_flat = val_targets.flatten()
    predictions_flat = predictions_array.flatten()

    # Compute the RMSE, MAE, and standard deviation of the residuals
    rmse = mean_squared_error(targets_flat, predictions_flat, squared=False)
    mae = mean_absolute_error(targets_flat, predictions_flat)
    std = np.std(targets_flat - predictions_flat)
    print(f"RMSE: {rmse} MAE: {mae}")

    # Write the evaluation metrics out to an evaluation report
    report_dict = {
        "regression_metrics": {
            "rmse": {"value": rmse, "standard_deviation": std},
            "mae": {"value": mae, "standard_deviation": std}
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))

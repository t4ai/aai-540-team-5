import os
import json
import pathlib
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error


if __name__ == "__main__":
    
    base_dir = "/opt/ml/processing"
    print(os.getcwd())
    
    val_targets = np.load(os.path.join(f"{base_dir}/validation", "val_targets.npy"))
    print(val_targets.shape)
    
    with open(f"{base_dir}/transform-results/validation_data.ndjson.out", "r") as f:
        predictions = []
        for line in f:
            obj = json.loads(line.strip())
            predictions.extend(obj["predictions"])

    predictions_array = np.array(predictions)
    print(predictions_array.shape)
    
    targets_flat = val_targets.flatten()
    predictions_flat = predictions_array.flatten()

    rmse = mean_squared_error(targets_flat, predictions_flat, squared=False)
    mae = mean_absolute_error(targets_flat, predictions_flat)
    std = np.std(targets_flat - predictions_flat)
    print(f"RMSE: {rmse} MAE: {mae}")

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

import pandas as pd

from utils.config import Config
from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler
from utils.validation import validate_model


results = {}
cfg = Config()
params = cfg.get_params("validate_params.yaml")
rh = ResultsHandler()

for model_name in cfg.models.keys():
    mts = preprocess_data("val.csv", model_name=model_name)
    model_name = "val_" + model_name
    print(
        f"Validating {model_name} with {params['n_validations']} iterations on train set"
    )
    rmse, ps, acc = validate_model(model_name, mts, params["n_validations"])
    results[model_name] = {"RMSE": rmse, "PredictiveScore": ps, "Accuracy": acc}

rh.write_json_results(results, "validation_metrics")
results = pd.DataFrame(results).transpose().reset_index(names="Model")
top_models = (
    pd.concat(
        [
            results.sort_values("RMSE").iloc[:3],
            results.sort_values("PredictiveScore", ascending=False).iloc[:3],
            results.sort_values("Accuracy", ascending=False).iloc[:3],
        ]
    )
    .sort_values("RMSE")
    .drop_duplicates()
)
rh.write_csv_results(top_models, "validation_results")

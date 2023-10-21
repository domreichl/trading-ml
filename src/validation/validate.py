import pandas as pd

from utils.config import Config
from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler
from utils.validation import validate_model


results = {}
cfg = Config()
params = cfg.get_params("validate_params.yaml")
rh = ResultsHandler()
mts = preprocess_data("val.csv")

for model_name in cfg.models:
    model_name = "val_" + model_name
    print(
        f"Validating {model_name} with {params['n_validations']} iterations on train set"
    )
    mae, rmse, f1 = validate_model(model_name, mts, params["n_validations"])
    results[model_name] = {"MAE": mae, "RMSE": rmse, "F1": f1}

sorted_results = pd.DataFrame(results).transpose().reset_index().sort_values("RMSE")
top_models = list(sorted_results[sorted_results["F1"] > 0.5]["index"])
ranking = pd.DataFrame({"Rank": range(1, len(top_models) + 1), "Model": top_models})
rh.write_json_results(results, "validation_metrics")
rh.write_csv_results(ranking, "validation_ranked")

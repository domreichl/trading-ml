import pandas as pd

from utils.config import Config
from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler
from utils.validation import validate_model


cfg = Config()
rh = ResultsHandler()
mts = preprocess_data("val.csv")
models, maes, rmses = [], [], []

for model_name in cfg.models:
    model_name = "val_" + model_name
    print(
        f"Validating {model_name} with {cfg.n_validations} iterations on train set..."
    )
    mae, rmse = validate_model(model_name, mts, cfg.n_validations)
    models.append(model_name)
    maes.append(mae)
    rmses.append(rmse)
    print(f"Results for {model_name}: MAE={round(mae, 4)}, RMSE={round(rmse, 4)}")

results = pd.DataFrame({"Model": models, "MAE": maes, "RMSE": rmses})
results.sort_values("RMSE", inplace=True)
top_five_models = list(results["Model"][:5])
ranking = pd.DataFrame(
    {"Rank": range(1, len(top_five_models) + 1), "Model": top_five_models}
)
rh.write_csv_results(results, "validation")
rh.write_csv_results(ranking, "validation_ranked")

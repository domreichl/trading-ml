from utils.data_preprocessing import preprocess_data
from utils.file_handling import CkptHandler, ResultsHandler
from utils.training import train_model


mts = preprocess_data("main.csv")
ranked_models = ResultsHandler().load_csv_results("validation_ranked")["Model"].unique()
CkptHandler().reset_dir("main")

for model_name in ranked_models:
    train_model("main_" + model_name, mts)

from utils.config import Config
from utils.data_preprocessing import preprocess_data
from utils.file_handling import CkptHandler, ResultsHandler
from utils.training import train_model


model_dict = Config().models
ranked_models = ResultsHandler().load_csv_results("validation_ranked")["Model"].unique()
CkptHandler().reset_dir("main")

for model_name in ranked_models:
    mts = preprocess_data("main.csv", model_name=model_name)
    train_model(model_name.replace("val_", "main_"), mts)

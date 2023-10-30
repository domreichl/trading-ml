from utils.config import Config
from utils.data_preprocessing import preprocess_data
from utils.file_handling import CkptHandler, ResultsHandler
from utils.training import train_model


model_dict = Config().models
top_val_models = (
    ResultsHandler().load_csv_results("validation_results")["Model"].unique()
)
CkptHandler().reset_dir("main")

for model_name in top_val_models:
    mts = preprocess_data("main.csv", model_name=model_name.replace("val_", ""))
    train_model(model_name.replace("val_", "main_"), mts)

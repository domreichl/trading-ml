from utils.config import Config
from utils.data_preprocessing import preprocess_data
from utils.file_handling import CkptHandler
from utils.training import train_model


mts = preprocess_data("val.csv")
CkptHandler().reset_dir("val")

for model_name in Config().models:
    train_model("val_" + model_name, mts)

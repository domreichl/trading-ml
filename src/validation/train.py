from utils.config import Config
from utils.data_preprocessing import preprocess_data
from utils.file_handling import CkptHandler
from utils.training import train_model


CkptHandler().reset_dir("val")

for model_name in Config().models.keys():
    mts = preprocess_data("val.csv", model_name=model_name)
    train_model("val_" + model_name, mts)

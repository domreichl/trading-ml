import os

from config.model_config import model_config
from models.base import fit_arima, fit_prophet
from models.lstms import LSTMRegression
from utils.data_classes import MultipleTimeSeries
from utils.data_preprocessing import preprocess_data


def train_model(model_name: str, mts: MultipleTimeSeries) -> None:
    if "arima" in model_name:
        fit_arima(mts, model_name)
    elif "lstm" in model_name:
        model = LSTMRegression(mts)
        model.train().save(os.path.join(model_config["ckpt_dir"], model_name))
    elif "prophet" in model_name:
        fit_prophet(mts, model_name)


if __name__ == "__main__":
    mts = preprocess_data()
    for i, model_name in enumerate(model_config["names"]):
        if i in model_config["trainable"]:
            train_model(model_name, mts)

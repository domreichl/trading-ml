from models.base import fit_arima, fit_prophet
from models.lstms import LSTMRegression
from utils.config import Config
from utils.data_classes import MultipleTimeSeries
from utils.data_preprocessing import preprocess_data
from utils.file_handling import reset_dir, get_ckpt_dir


def train_model(model_name: str, mts: MultipleTimeSeries) -> None:
    if "arima" in model_name:
        fit_arima(mts, model_name)
    elif "lstm" in model_name:
        model = LSTMRegression(mts)
        model.train().save(get_ckpt_dir(model_name))
    elif "prophet" in model_name:
        fit_prophet(mts, model_name)


if __name__ == "__main__":
    mts = preprocess_data()
    reset_dir("eval")
    cfg = Config()
    for model_name, model_cfg in cfg.models.items():
        if model_cfg["store_ckpt"]:
            train_model("eval_" + model_name, mts)

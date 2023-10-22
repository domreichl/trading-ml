from models.base import fit_arima, fit_prophet
from models.lstms import LSTMRegression
from utils.config import Config
from utils.data_classes import MultipleTimeSeries

from utils.file_handling import CkptHandler


def train_model(model_name: str, mts: MultipleTimeSeries) -> None:
    cfg = Config()
    if "arima" in model_name:
        fit_arima(mts, model_name)
    elif "lstm" in model_name:
        model = LSTMRegression(mts)
        model.train(cfg.batch_size, cfg.n_epochs).save(
            CkptHandler().get_ckpt_dir(model_name)
        )
    elif "prophet" in model_name:
        fit_prophet(mts, model_name)
    else:
        print(f"No checkpoint needed for '{model_name}' model.")
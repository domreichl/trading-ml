from models.local import fit_arima, fit_prophet
from models.neural_networks import RegressionNet
from utils.data_classes import MultipleTimeSeries

from utils.file_handling import CkptHandler


def train_model(
    model_name: str, mts: MultipleTimeSeries, batch_size: int = 32, n_epochs: int = 10
) -> None:
    if "arima" in model_name:
        fit_arima(mts, model_name)
    elif "_net" in model_name:
        model = RegressionNet(model_name, mts)
        model.train(batch_size, n_epochs)
    elif "prophet" in model_name:
        fit_prophet(mts, model_name)
    else:
        print(f"No checkpoint needed for '{model_name}' model.")

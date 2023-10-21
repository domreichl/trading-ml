from pathlib import Path

from utils.data_classes import MultipleTimeSeries
from utils.data_preprocessing import preprocess_data
from utils.data_processing import stack_array_from_dict
from utils.file_handling import DataHandler, CkptHandler
from models.base import (
    fit_arima,
    predict_arima,
    fit_predict_exponential_smoothing,
    predict_moving_average,
    predict_moving_average_recursive,
    fit_prophet,
    predict_prophet,
    validate_arima,
    validate_exponential_smoothing,
    validate_moving_average,
    validate_prophet,
)
from models.boosting import fit_predict_boosting_model, validate_boosting_model
from models.lstms import LSTMRegression


class UnitTestDataTrimmer:
    def __init__(self, path: Path, days_to_keep: int):
        self.mts = preprocess_data(path)
        self.mts.x_train = self.mts.x_train[-days_to_keep:]
        self.mts.y_train = self.mts.y_train[-days_to_keep:]
        self.mts.dates = self.mts.dates[
            : days_to_keep + self.mts.x_train.shape[1] + len(self.mts.y_test)
        ]

    def get_mts(self) -> MultipleTimeSeries:
        return self.mts


mts = UnitTestDataTrimmer(
    Path(__file__).parent.joinpath("test_data.csv"), days_to_keep=25
).get_mts()


def test_base_fit_predict_arima():
    model_name = "test_arima"
    if not CkptHandler().get_ckpt_dir(model_name).is_dir():
        fit_arima(mts, model_name)
    y_preds = predict_arima(mts, model_name)
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)
    assert y_preds["AT0000937503"][6] > 0


def test_base_validate_arima():
    mse, rmse, f1 = validate_arima(mts, n_validations=2)
    assert mse > 0
    assert rmse > 0
    assert f1 > 0


def test_base_fit_predict_exponential_smoothing():
    y_preds = fit_predict_exponential_smoothing(mts)
    assert y_preds["ExponentialTrend"][8] > 0
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)


def test_base_validate_exponential_smoothing():
    mse, rmse, f1 = validate_exponential_smoothing(mts, n_validations=2)
    assert mse > 0
    assert rmse > 0
    assert f1 > 0


def test_base_predict_moving_average():
    y_preds = predict_moving_average(mts)
    assert y_preds["ExponentialTrend"][8] > 0
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)


def test_base_predict_moving_average_recursive():
    y_preds = predict_moving_average_recursive(mts)
    assert y_preds["ExponentialTrend"][8] > 0
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)


def test_base_validate_moving_average():
    mse, rmse, f1 = validate_moving_average(mts, recursive=False, n_validations=2)
    mse_rec, rmse_rec, f1_rec = validate_moving_average(
        mts, recursive=True, n_validations=2
    )
    assert mse > 0
    assert rmse > 0
    assert f1 > 0
    assert mse_rec > 0
    assert rmse_rec > 0
    assert f1_rec > 0


def test_base_fit_predict_prophet():
    model_name = "test_prophet"
    if not CkptHandler().get_ckpt_dir(model_name).is_dir():
        fit_prophet(mts, model_name)
    y_preds = predict_prophet(mts, model_name)
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)
    assert y_preds["AT0000937503"][6] > 0


def test_base_validate_prophet():
    mse, rmse, f1 = validate_prophet(mts, n_validations=2)
    assert mse > 0
    assert rmse > 0
    assert f1 > 0


def test_boosting_fit_predict_boosting_model():
    for model_name in ["LGBMRegressor", "XGBRegressor"]:
        y_preds = fit_predict_boosting_model(model_name, mts)
        assert y_preds["AT0000743059"][6] > 0
        assert stack_array_from_dict(y_preds, 1).shape == (10, 4)


def test_boosting_validate_boosting_model():
    for model_name in ["LGBMRegressor", "XGBRegressor"]:
        mse, rmse, f1 = validate_boosting_model(model_name, mts, n_validations=2)
        assert mse > 0
        assert rmse > 0
        assert f1 > 0


def test_lstms_lstmregression_predict():
    model = LSTMRegression(mts)
    y_preds = model.predict()
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)


def test_lstms_lstmregression_validate():
    model = LSTMRegression(mts)
    mse, rmse, f1 = model.validate(n_validations=2)
    assert mse > 0
    assert rmse > 0
    assert f1 == 0

import os

from utils.data_classes import MultipleTimeSeries
from utils.data_preprocessing import preprocess_data
from utils.data_processing import stack_array_from_dict
from utils.file_handling import load_csv_data
from models.base import (
    fit_predict_arima,
    fit_predict_exponential_smoothing,
    predict_moving_average,
    predict_moving_average_recursive,
    fit_predict_prophet,
    validate_arima,
    validate_exponential_smoothing,
    validate_moving_average,
    validate_prophet,
)
from models.boosting import fit_predict_boosting_model, validate_boosting_model
from models.lstms import LSTMRegression


class UnitTestDataTrimmer:
    def __init__(self, path: str, days_to_keep: int):
        self.test_data = load_csv_data(path)
        self.days_to_keep = days_to_keep

    def get_mts(self) -> MultipleTimeSeries:
        mts = preprocess_data(self.test_data)
        mts.x_train = mts.x_train[-self.days_to_keep :]
        mts.y_train = mts.y_train[-self.days_to_keep :]
        mts.dates = mts.dates[
            : self.days_to_keep + mts.x_train.shape[1] + len(mts.y_test)
        ]
        return mts


mts = UnitTestDataTrimmer(
    os.path.join(os.path.dirname(__file__), "test_data.csv"), days_to_keep=25
).get_mts()


def test_base_fit_predict_arima():
    y_preds = fit_predict_arima(mts, "test_arima")
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)
    assert y_preds["AT0000937503"][6] > 0


def test_base_validate_arima():
    mse, rmse = validate_arima(mts, n_validations=2)
    assert mse > 0
    assert rmse > 0


def test_base_fit_predict_exponential_smoothing():
    y_preds = fit_predict_exponential_smoothing(mts)
    assert y_preds["ExponentialTrend"][8] > 0
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)


def test_base_validate_exponential_smoothing():
    mse, rmse = validate_exponential_smoothing(mts, n_validations=2)
    assert mse > 0
    assert rmse > 0


def test_base_predict_moving_average():
    y_preds = predict_moving_average(mts)
    assert y_preds["ExponentialTrend"][8] > 0
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)


def test_base_predict_moving_average_recursive():
    y_preds = predict_moving_average_recursive(mts)
    assert y_preds["ExponentialTrend"][8] > 0
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)


def test_base_validate_moving_average():
    mse, rmse = validate_moving_average(mts, recursive=False, n_validations=2)
    mse_rec, rmse_rec = validate_moving_average(mts, recursive=True, n_validations=2)
    assert mse > 0
    assert rmse > 0
    assert mse_rec > 0
    assert rmse_rec > 0


def test_base_fit_predict_prophet():
    y_preds = fit_predict_prophet(mts, "test_prophet")
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)
    assert y_preds["AT0000937503"][6] > 0


def test_base_validate_prophet():
    mse, rmse = validate_prophet(mts, n_validations=2)
    assert mse > 0
    assert rmse > 0


def test_boosting_fit_predict_boosting_model():
    for model_name in ["LGBMRegressor", "XGBRegressor"]:
        y_preds = fit_predict_boosting_model(model_name, mts)
        assert y_preds["AT0000743059"][6] > 0
        assert stack_array_from_dict(y_preds, 1).shape == (10, 4)


def test_boosting_validate_boosting_model():
    for model_name in ["LGBMRegressor", "XGBRegressor"]:
        mse, rmse = validate_boosting_model(model_name, mts, n_validations=2)
        assert mse > 0
        assert rmse > 0


def test_lstms_lstmregression_predict():
    model = LSTMRegression(mts)
    y_preds = model.predict()
    assert stack_array_from_dict(y_preds, 1).shape == (10, 4)


def test_lstms_lstmregression_validate():
    model = LSTMRegression(mts)
    mse, rmse = model.validate(n_validations=2)
    assert mse > 0
    assert rmse > 0

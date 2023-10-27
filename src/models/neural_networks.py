import random
import numpy as np
import tensorflow as tf
from typing import Optional
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM

from utils.data_classes import MultipleTimeSeries
from utils.data_processing import get_signs_from_returns
from utils.evaluation import evaluate_return_predictions, evaluate_sign_predictions
from utils.file_handling import CkptHandler


class LocalLinearNet:
    def __init__(self, mts: MultipleTimeSeries, ts_idx: int):
        self.mts = mts
        self.ts_idx = ts_idx
        self.test_days = len(mts.y_test)
        self.look_back_window_size = mts.x_train.shape[1]
        inputs = Input(shape=(self.look_back_window_size))
        x = Dense(260, activation="linear")(inputs)
        out = Dense(self.test_days)(x)
        self.model = Model(inputs=inputs, outputs=out)

    def train(self, batch_size: int, epochs: int) -> Model:
        self.model.compile(loss="mean_squared_error", optimizer="adam")
        self.model.fit(
            self.mts.x_train[:, :, self.ts_idx],
            self.mts.y_train[:, :, self.ts_idx],
            batch_size=batch_size,
            epochs=epochs,
        )
        return self.model

    def load(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self) -> dict:
        predictions = self.model.predict(
            np.expand_dims(self.mts.x_test[:, self.ts_idx], 0)
        )
        return predictions

    def validate(self, n_validations: int) -> tuple[float, float]:
        mae_lst, rmse_lst, f1_lst = [], [], []
        for _ in range(n_validations):
            trial_idx = random.randint(
                0, len(self.mts.x_train[:, :, self.ts_idx]) - self.test_days
            )
            x = np.expand_dims(self.mts.x_train[trial_idx, :, self.ts_idx], 0)
            y_true = self.mts.y_train[trial_idx, :, self.ts_idx]
            y_pred = np.squeeze(self.model.predict(x))
            assert y_true.shape == y_pred.shape == (self.test_days,)
            gt = self.mts.get_returns_from_features(y_true)
            pr = self.mts.get_returns_from_features(y_pred)
            metrics = evaluate_return_predictions(gt, pr)
            metrics_sign = evaluate_sign_predictions(
                get_signs_from_returns(gt), get_signs_from_returns(pr)
            )
            mae_lst.append(metrics["MAE"])
            rmse_lst.append(metrics["RMSE"])
            f1_lst.append(metrics_sign["F1"])
        return float(np.mean(mae_lst)), float(np.mean(rmse_lst)), float(np.mean(f1_lst))


class GlobalLinearNet:
    def __init__(self, mts: MultipleTimeSeries):
        self.mts = mts
        self.names = mts.names
        self.test_days = len(mts.y_test)
        self.look_back_window_size = mts.x_train.shape[1]
        inputs = Input(shape=(self.look_back_window_size, len(self.names)))
        # TODO: continue here

    def train(self, batch_size: int, epochs: int) -> Model:
        self.model.compile(loss="mean_squared_error", optimizer="adam")
        self.model.fit(
            self.mts.x_train, self.mts.y_train, batch_size=batch_size, epochs=epochs
        )
        return self.model

    def load(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self) -> dict:
        predictions = np.squeeze(self.model.predict(np.expand_dims(self.mts.x_test, 0)))
        return {ts_name: predictions[:, i] for i, ts_name in enumerate(self.names)}

    def validate(self, n_validations: int) -> tuple[float, float]:
        test_days = len(self.mts.y_test)
        mae_lst, rmse_lst, f1_lst = [], [], []
        for _ in range(n_validations):
            trial_idx = random.randint(0, len(self.mts.x_train) - test_days)
            x = np.expand_dims(self.mts.x_train[trial_idx], 0)
            y_true = self.mts.y_train[trial_idx]
            y_pred = np.squeeze(self.model.predict(x))
            assert y_true.shape == y_pred.shape == (test_days, len(self.names))
            gt = self.mts.get_returns_from_features(y_true)
            pr = self.mts.get_returns_from_features(y_pred)
            metrics = evaluate_return_predictions(gt, pr)
            metrics_sign = evaluate_sign_predictions(
                get_signs_from_returns(gt), get_signs_from_returns(pr)
            )
            mae_lst.append(metrics["MAE"])
            rmse_lst.append(metrics["RMSE"])
            f1_lst.append(metrics_sign["F1"])
        return float(np.mean(mae_lst)), float(np.mean(rmse_lst)), float(np.mean(f1_lst))


class LSTMRegression:
    def __init__(self, mts: MultipleTimeSeries, dropout_rate: float = 0.3):
        self.mts = mts
        self.names = mts.names
        self.test_days = len(mts.y_test)
        self.look_back_window_size = mts.x_train.shape[1]
        inputs = Input(shape=(self.look_back_window_size, len(self.names)))
        x = LSTM(260, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        x = LSTM(65)(x)
        x = Dropout(dropout_rate)(x)
        outputs = []
        for _ in range(len(self.names)):
            x_i = Dense(40)(x)
            x_i = Dropout(dropout_rate)(x_i)
            x_i = Dense(self.test_days)(x_i)
            outputs.append(x_i)
        self.model = Model(inputs=inputs, outputs=tf.stack(outputs, 2))

    def train(self, batch_size: int, epochs: int) -> Model:
        self.model.compile(loss="mean_squared_error", optimizer="adam")
        self.model.fit(
            self.mts.x_train,
            self.mts.y_train,
            batch_size=batch_size,
            epochs=epochs,
        )
        return self.model

    def load(self, model_path: str):
        self.model = load_model(model_path)

    def predict(self) -> dict:
        predictions = np.squeeze(self.model.predict(np.expand_dims(self.mts.x_test, 0)))
        return {ts_name: predictions[:, i] for i, ts_name in enumerate(self.names)}

    def validate(self, n_validations: int) -> tuple[float, float]:
        test_days = len(self.mts.y_test)
        mae_lst, rmse_lst, f1_lst = [], [], []
        for _ in range(n_validations):
            trial_idx = random.randint(0, len(self.mts.x_train) - test_days)
            x = np.expand_dims(self.mts.x_train[trial_idx], 0)
            y_true = self.mts.y_train[trial_idx]
            y_pred = np.squeeze(self.model.predict(x))
            assert y_true.shape == y_pred.shape == (test_days, len(self.names))
            gt = self.mts.get_returns_from_features(y_true)
            pr = self.mts.get_returns_from_features(y_pred)
            metrics = evaluate_return_predictions(gt, pr)
            metrics_sign = evaluate_sign_predictions(
                get_signs_from_returns(gt), get_signs_from_returns(pr)
            )
            mae_lst.append(metrics["MAE"])
            rmse_lst.append(metrics["RMSE"])
            f1_lst.append(metrics_sign["F1"])
        return float(np.mean(mae_lst)), float(np.mean(rmse_lst)), float(np.mean(f1_lst))


def load_nn_model(
    model_name: str, mts: MultipleTimeSeries, ts_idx: Optional[int] = None
) -> Model:
    if "local" in model_name:
        model = LocalLinearNet(mts, ts_idx)
    elif "global" in model_name:
        model = GlobalLinearNet(mts)
    elif "lstm" in model_name:
        model = LSTMRegression(mts)
    model.load(CkptHandler().get_ckpt_dir(model_name))
    return model

import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM

from utils.config import Config
from utils.data_classes import MultipleTimeSeries
from utils.evaluation import evaluate_return_predictions
from utils.file_handling import get_ckpt_dir


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

    def train(self, batch_size: int = 30, epochs: int = 10) -> Model:
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

    def validate(
        self, n_validations: int = Config().n_validations
    ) -> tuple[float, float]:
        test_days = len(self.mts.y_test)
        mae_lst, rmse_lst = [], []
        for _ in range(n_validations):
            trial_idx = random.randint(0, len(self.mts.x_train) - test_days)
            x = np.expand_dims(self.mts.x_train[trial_idx], 0)
            y_true = self.mts.y_train[trial_idx]
            y_pred = np.squeeze(self.model.predict(x))
            assert y_true.shape == y_pred.shape == (test_days, len(self.names))
            metrics = evaluate_return_predictions(
                self.mts.get_returns_from_features(y_true),
                self.mts.get_returns_from_features(y_pred),
            )
            mae_lst.append(metrics["MAE"])
            rmse_lst.append(metrics["RMSE"])
        return float(np.mean(mae_lst)), float(np.mean(rmse_lst))


def load_lstm_model(model_name: str, mts: MultipleTimeSeries) -> Model:
    model = LSTMRegression(mts)
    model.load(get_ckpt_dir(model_name))
    return model

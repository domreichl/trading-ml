import random
import numpy as np
import tensorflow as tf
from typing import Optional
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, GlobalAveragePooling1D

from utils.data_classes import MultipleTimeSeries
from utils.data_processing import get_signs_from_returns
from utils.evaluation import evaluate_return_predictions, evaluate_sign_predictions
from utils.file_handling import CkptHandler


class RegressionNet:
    def __init__(self, model_name: str, mts: MultipleTimeSeries):
        self.mts = mts
        self.names = mts.names
        self.test_days = len(mts.y_test)
        self.look_back_window_size = mts.x_train.shape[1]
        self.model_name = model_name
        self.training = False
        if "simple" in model_name:
            self.model = self.set_feed_forward_net(
                self.look_back_window_size, self.test_days, len(self.names)
            )
        elif "recurrent" in model_name:
            self.model = self.set_recurrent_net(
                self.look_back_window_size, self.test_days, len(self.names)
            )

    def set_feed_forward_net(
        self,
        look_back_window_size: int,
        test_days: int,
        n_heads: int,
        dropout_rate: float = 0.3,
    ) -> Model:
        inputs = Input(shape=(look_back_window_size, n_heads))
        x = Dense(260, activation="relu")(inputs)
        x = Dropout(dropout_rate)(x, training=self.training)
        x = Dense(65, activation="relu")(x)
        x = Dropout(dropout_rate)(x, training=self.training)
        x = GlobalAveragePooling1D()(x)
        outputs = self.compute_heads(x, test_days, n_heads)
        return Model(inputs=inputs, outputs=outputs)

    def set_recurrent_net(
        self,
        look_back_window_size: int,
        test_days: int,
        n_heads: int,
        dropout_rate: float = 0.3,
    ) -> Model:
        inputs = Input(shape=(look_back_window_size, n_heads))
        x = LSTM(260, return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x, training=self.training)
        x = LSTM(65)(x)
        x = Dropout(dropout_rate)(x, training=self.training)
        outputs = self.compute_heads(x, test_days, n_heads)
        return Model(inputs=inputs, outputs=outputs)

    def compute_heads(self, x: tf.Tensor, test_days: int, n_heads: int) -> list:
        outputs = []
        for _ in range(n_heads):
            head = Dense(22, activation="relu")(x)
            head = Dense(test_days)(head)
            outputs.append(head)
        return tf.stack(outputs, 2)

    def train(self, batch_size: int, epochs: int) -> None:
        self.training = True
        self.model.compile(loss="mean_squared_error", optimizer="adam")
        self.model.fit(
            self.mts.x_train,
            self.mts.y_train,
            batch_size=batch_size,
            epochs=epochs,
        )
        self.model.save(CkptHandler().get_ckpt_dir(self.model_name))

    def load(self) -> None:
        self.model = load_model(CkptHandler().get_ckpt_dir(self.model_name))

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

import random
import numpy as np
import tensorflow as tf
from typing import Optional
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, GlobalAveragePooling1D

from utils.data_classes import MultipleTimeSeries
from utils.evaluation import get_validation_metrics
from utils.file_handling import CkptHandler


class RegressionNet:
    def __init__(self, model_name: str, mts: MultipleTimeSeries):
        self.mts = mts
        self.names = mts.names
        self.test_days = len(mts.y_test)
        self.look_back_window_size = mts.x_train.shape[1]
        self.n_heads = len(mts.names)
        self.model_name = model_name
        if "simple" in self.model_name:
            self.model = self.set_feed_forward_net()
        elif "recurrent" in self.model_name:
            self.model = self.set_recurrent_net()

    def set_feed_forward_net(self) -> Model:
        inputs = Input(shape=(self.look_back_window_size, self.n_heads))
        x = Dense(260, activation="relu")(inputs)
        x = Dense(130, activation="relu")(x)
        x = GlobalAveragePooling1D()(x)
        outputs = self.compute_heads(x, self.test_days, self.n_heads)
        return Model(inputs=inputs, outputs=outputs)

    def set_recurrent_net(self) -> Model:
        inputs = Input(shape=(self.look_back_window_size, self.n_heads))
        x = LSTM(260, return_sequences=True)(inputs)
        x = LSTM(130)(x)
        outputs = self.compute_heads(x, self.test_days, self.n_heads)
        return Model(inputs=inputs, outputs=outputs)

    def compute_heads(self, x: tf.Tensor, test_days: int, n_heads: int) -> list:
        outputs = []
        for _ in range(n_heads):
            head = Dense(130, activation="relu")(x)
            head = Dense(test_days)(head)
            outputs.append(head)
        return tf.stack(outputs, 2)

    def train(self, batch_size: int = 100, epochs: int = 10) -> None:
        self.model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        )
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
        rmse_lst, ps_lst = [], []
        for _ in range(n_validations):
            trial_idx = random.randint(0, len(self.mts.x_train) - test_days)
            x = np.expand_dims(self.mts.x_train[trial_idx], 0)
            y_true = self.mts.y_train[trial_idx]
            y_pred = np.squeeze(self.model.predict(x))
            rmse, ps = get_validation_metrics(
                self.mts.get_returns_from_features(y_true),
                self.mts.get_returns_from_features(y_pred),
            )
            rmse_lst.append(rmse)
            ps_lst.append(ps)
        return float(np.mean(rmse_lst)), float(np.mean(ps_lst))

import csv

from models.neural_nets import RegressionNet
from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler


MODEL_NAME = "recurrent_regression_net"
N_VALIDATIONS = 100
CSV_PATH = ResultsHandler().results_dir.joinpath("tuning", MODEL_NAME + ".csv")

with open(CSV_PATH, "w") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow(
        [
            "Model",
            "BatchSize",
            "Epochs",
            "LearningRate",
            "LookBackWindowSize",
            "RMSE",
            "PredictiveScore",
        ]
    )

for epochs in [10]:
    for batch_size in [100]:
        for lbws in [5, 10, 20, 25, 65, 260, 520]:
            mts = preprocess_data("exp.csv", look_back_window_size=lbws)
            model = RegressionNet("exp_" + MODEL_NAME, mts)
            model.train(batch_size, epochs)
            rmse, ps = model.validate(N_VALIDATIONS)
            with open(CSV_PATH, "a") as file:
                writer = csv.writer(file, delimiter=";")
                writer.writerow(
                    [
                        MODEL_NAME,
                        batch_size,
                        epochs,
                        1e-3,
                        lbws,
                        rmse,
                        ps,
                    ]
                )

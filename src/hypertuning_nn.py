import csv

from models.neural_networks import RegressionNet
from utils.data_preprocessing import preprocess_data
from utils.file_handling import ResultsHandler


MODEL_NAME = "simple_regression_net"
N_VALIDATIONS = 100

# - Linear vs. Relu
# - LBWS
# - dropout_rate

# - N_Layers

# - batch_size
# - learning_rate
# - epochs

batch_size = 32
epochs = 10

for lbws in [5]:  # [1040, 780, 520, 260, 65, 22, 10, 5]:
    mts = preprocess_data("exp.csv", look_back_window_size=lbws)
    model = RegressionNet("exp_" + MODEL_NAME, mts)
    model.train(batch_size, epochs)

    mae, rmse, f1 = model.validate(N_VALIDATIONS)
    with open(
        ResultsHandler().results_dir.joinpath("tuning", MODEL_NAME + ".csv"), "a"
    ) as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(
            [MODEL_NAME, batch_size, epochs, lbws, mae, rmse, f1, "relu", 0.3]
        )

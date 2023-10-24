from pathlib import Path

from utils.data_preprocessing import preprocess_data


def test_preprocess_data():
    mts = preprocess_data(Path(__file__).parent.joinpath("test_data.csv"))
    assert round(mts.x_train[-1][-1][-1], 4) == 0.4173
    assert round(mts.y_train[0][0][0], 4) == 0.8725
    assert round(mts.x_test[-1][-1], 4) == 0.4042
    assert round(mts.y_test[1][1], 4) == 0.4069
    assert mts.x_train.shape == (4929, 52 * 5, 4)
    assert mts.y_train.shape == (4929, 10, 4)
    assert mts.x_test.shape == (52 * 5, 4)
    assert mts.y_test.shape == (10, 4)
    assert len(mts.names) == len(mts.close_prices.keys()) == 4
    assert len(mts.log_returns) == 4

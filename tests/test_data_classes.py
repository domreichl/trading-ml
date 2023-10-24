from pathlib import Path

from utils.data_preprocessing import preprocess_data


def test_mts_merge_features():
    mts = preprocess_data(Path(__file__).parent.joinpath("test_data.csv"))
    assert mts.x_train.shape == (4929, 260, 4)
    assert mts.y_train.shape == (4929, 10, 4)
    mts.merge_features()
    assert mts.x_train.shape == (4930, 260, 4)
    assert mts.y_train.shape == (0,)
    assert mts.x_test.shape == (0,)
    assert mts.y_test.shape == (10, 4)


def test_mts_merge_features_dl():
    mts = preprocess_data(Path(__file__).parent.joinpath("test_data.csv"))
    mts.merge_features(for_deep_learning=True)
    assert mts.x_train.shape == (4930, 260, 4)
    assert mts.y_train.shape == (4930, 10, 4)


def test_mts_get_forecast_dates():
    mts = preprocess_data(Path(__file__).parent.joinpath("test_data.csv"))
    forecast_dates = mts.get_forecast_dates()
    assert mts.dates[-1] == "2023-09-15"
    assert forecast_dates[0] == "2023-09-18"
    assert forecast_dates[-1] == "2023-09-29"
